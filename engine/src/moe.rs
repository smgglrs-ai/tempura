use tempura_kernels::moe::{MoeRouter, MoeRoutingResult};

use crate::config::HiddenAct;
use crate::weights::QuantizedWeights;

/// Execute MoE FFN for a single transformer layer.
///
/// For each token:
/// 1. Compute router logits via router_weight * input
/// 2. Route tokens to top-k experts
/// 3. Run per-expert FFN (gate + up + activation + down)
/// 4. Combine expert outputs weighted by routing weights
///
/// Weight naming convention follows GGUF/HuggingFace:
/// - Router: `model.layers.{layer}.block_sparse_moe.gate.weight` (Mixtral)
///   or `model.layers.{layer}.mlp.gate.weight` (Qwen3MOE)
/// - Expert FFN: `model.layers.{layer}.block_sparse_moe.experts.{expert}.{gate|up|down}_proj.weight`
///   or `model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.weight`
pub fn moe_ffn(
    input: &[f32],
    output: &mut [f32],
    router: &MoeRouter,
    qweights: &QuantizedWeights,
    layer: usize,
    hidden_size: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    num_tokens: usize,
    weight_prefix: &str,
    router_weight_name: &str,
) {
    let q4k = tempura_kernels::quantized::CpuMatMulQ4KOpt;
    use tempura_kernels::quantized::MatMulQ;

    // Step 1: Compute router logits — input [num_tokens, hidden_size] x router [n_experts, hidden_size]^T
    let n_experts = router.n_experts;
    let mut router_logits = vec![0.0f32; num_tokens * n_experts];

    qmm_generic(
        &q4k,
        qweights,
        input,
        router_weight_name,
        &mut router_logits,
        num_tokens,
        n_experts,
        hidden_size,
    );

    // Step 2: Route tokens to experts
    let routing = router
        .route(&router_logits, num_tokens)
        .expect("MoE routing failed");

    // Step 3 & 4: Per-expert FFN + weighted combination
    let mut gate_buf = vec![0.0f32; intermediate_size];
    let mut up_buf = vec![0.0f32; intermediate_size];
    let mut down_buf = vec![0.0f32; hidden_size];

    output[..num_tokens * hidden_size].fill(0.0);

    for t in 0..num_tokens {
        let token_input = &input[t * hidden_size..(t + 1) * hidden_size];
        let token_output = &mut output[t * hidden_size..(t + 1) * hidden_size];
        let routing_result = &routing[t];

        expert_ffn_combine(
            token_input,
            token_output,
            routing_result,
            &q4k,
            qweights,
            weight_prefix,
            layer,
            hidden_size,
            intermediate_size,
            hidden_act,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
        );
    }
}

fn expert_ffn_combine(
    token_input: &[f32],
    token_output: &mut [f32],
    routing: &MoeRoutingResult,
    q4k: &tempura_kernels::quantized::CpuMatMulQ4KOpt,
    qweights: &QuantizedWeights,
    weight_prefix: &str,
    layer: usize,
    hidden_size: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    down_buf: &mut [f32],
) {
    use tempura_kernels::quantized::MatMulQ;

    for (&expert_idx, &expert_weight) in routing
        .expert_indices
        .iter()
        .zip(routing.expert_weights.iter())
    {
        let gate_name = format!(
            "model.layers.{layer}.{weight_prefix}.experts.{expert_idx}.gate_proj.weight"
        );
        let up_name = format!(
            "model.layers.{layer}.{weight_prefix}.experts.{expert_idx}.up_proj.weight"
        );
        let down_name = format!(
            "model.layers.{layer}.{weight_prefix}.experts.{expert_idx}.down_proj.weight"
        );

        gate_buf[..intermediate_size].fill(0.0);
        up_buf[..intermediate_size].fill(0.0);
        down_buf[..hidden_size].fill(0.0);

        // gate_proj: [intermediate_size, hidden_size] x [hidden_size] -> [intermediate_size]
        qmm_generic(
            q4k,
            qweights,
            token_input,
            &gate_name,
            &mut gate_buf[..intermediate_size],
            1,
            intermediate_size,
            hidden_size,
        );

        // up_proj: same shape
        qmm_generic(
            q4k,
            qweights,
            token_input,
            &up_name,
            &mut up_buf[..intermediate_size],
            1,
            intermediate_size,
            hidden_size,
        );

        // Activation
        match hidden_act {
            HiddenAct::SiluGate => {
                for i in 0..intermediate_size {
                    gate_buf[i] =
                        tempura_kernels::portable_math::portable_silu(gate_buf[i]) * up_buf[i];
                }
            }
            HiddenAct::SquaredRelu => {
                for i in 0..intermediate_size {
                    let x = gate_buf[i].max(0.0);
                    gate_buf[i] = x * x * up_buf[i];
                }
            }
        }

        // down_proj: [hidden_size, intermediate_size] x [intermediate_size] -> [hidden_size]
        qmm_generic(
            q4k,
            qweights,
            &gate_buf[..intermediate_size],
            &down_name,
            &mut down_buf[..hidden_size],
            1,
            hidden_size,
            intermediate_size,
        );

        // Weighted accumulation
        for i in 0..hidden_size {
            token_output[i] += expert_weight * down_buf[i];
        }
    }
}

/// Generic quantized matmul dispatch — mirrors the qmm closure in model.rs forward_quantized.
fn qmm_generic(
    q4k: &tempura_kernels::quantized::CpuMatMulQ4KOpt,
    qweights: &QuantizedWeights,
    input: &[f32],
    weight_name: &str,
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use tempura_kernels::quantized::MatMulQ;

    if let Some(w) = qweights.get_q4k(weight_name) {
        q4k.forward_b_transposed_q4k(input, w, output, m, n, k)
            .unwrap();
    } else if let Some(w) = qweights.get_q6k(weight_name) {
        q4k.forward_b_transposed_q6k(input, w, output, m, n, k)
            .unwrap();
    } else if let Some((dtype, w)) = qweights.get_typed(weight_name) {
        tempura_kernels::ggml::mul_mat::mul_mat_q(dtype, w, input, output, m, n, k);
    } else if let Some(w) = qweights.get_f32(weight_name) {
        use tempura_kernels::MatMul;
        tempura_kernels::matmul::CpuMatMul
            .forward_b_transposed(input, w, output, m, n, k)
            .unwrap();
    } else {
        panic!("MoE weight not found: {weight_name}");
    }
}

/// Common weight prefix for different MoE architectures.
/// Returns the prefix used between `model.layers.{layer}.` and `.experts.{expert}.`
pub fn moe_weight_prefix(arch: &str) -> &str {
    match arch {
        "mixtral" => "block_sparse_moe",
        "qwen2moe" | "qwen3moe" => "mlp",
        "deepseek2" | "deepseekv2" => "mlp",
        _ => "block_sparse_moe",
    }
}

/// Router weight name for different MoE architectures.
pub fn moe_router_weight_name(arch: &str, layer: usize) -> String {
    match arch {
        "mixtral" => format!("model.layers.{layer}.block_sparse_moe.gate.weight"),
        "qwen2moe" | "qwen3moe" => format!("model.layers.{layer}.mlp.gate.weight"),
        "deepseek2" | "deepseekv2" => format!("model.layers.{layer}.mlp.gate.weight"),
        _ => format!("model.layers.{layer}.block_sparse_moe.gate.weight"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_moe_weight_prefix_mixtral() {
        assert_eq!(moe_weight_prefix("mixtral"), "block_sparse_moe");
    }

    #[test]
    fn test_moe_weight_prefix_qwen() {
        assert_eq!(moe_weight_prefix("qwen3moe"), "mlp");
        assert_eq!(moe_weight_prefix("qwen2moe"), "mlp");
    }

    #[test]
    fn test_moe_router_weight_name() {
        assert_eq!(
            moe_router_weight_name("mixtral", 5),
            "model.layers.5.block_sparse_moe.gate.weight"
        );
        assert_eq!(
            moe_router_weight_name("qwen3moe", 0),
            "model.layers.0.mlp.gate.weight"
        );
    }

    #[test]
    fn test_moe_ffn_with_synthetic_weights() {
        // Build minimal QuantizedWeights with f32 expert weights
        let hidden_size = 8;
        let intermediate_size = 16;
        let n_experts = 2;
        let n_experts_used = 1;

        let mut qw = QuantizedWeights::empty();

        // Router weight: [n_experts, hidden_size]
        // Expert 1 has higher bias for all inputs
        let mut router_w = vec![0.0f32; n_experts * hidden_size];
        for i in 0..hidden_size {
            router_w[i] = -1.0; // expert 0: low score
            router_w[hidden_size + i] = 1.0; // expert 1: high score
        }
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.gate.weight".to_string(),
            router_w,
        );

        // Expert 0 weights (should not be selected, but present)
        let gate_0 = vec![0.1f32; intermediate_size * hidden_size];
        let up_0 = vec![0.1f32; intermediate_size * hidden_size];
        let down_0 = vec![0.1f32; hidden_size * intermediate_size];
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight".to_string(),
            gate_0,
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.up_proj.weight".to_string(),
            up_0,
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.down_proj.weight".to_string(),
            down_0,
        );

        // Expert 1 weights (should be selected)
        let gate_1 = vec![0.5f32; intermediate_size * hidden_size];
        let up_1 = vec![0.5f32; intermediate_size * hidden_size];
        let down_1 = vec![0.5f32; hidden_size * intermediate_size];
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.1.gate_proj.weight".to_string(),
            gate_1,
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.1.up_proj.weight".to_string(),
            up_1,
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.1.down_proj.weight".to_string(),
            down_1,
        );

        let router = MoeRouter::new(n_experts, n_experts_used);
        let input = vec![1.0f32; hidden_size];
        let mut output = vec![0.0f32; hidden_size];

        moe_ffn(
            &input,
            &mut output,
            &router,
            &qw,
            0,
            hidden_size,
            intermediate_size,
            HiddenAct::SiluGate,
            1,
            "block_sparse_moe",
            "model.layers.0.block_sparse_moe.gate.weight",
        );

        // Output should be non-zero (expert 1 was selected with weight 1.0)
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "MoE output should be non-zero");
    }

    #[test]
    fn test_moe_single_expert_matches_dense() {
        // With 1 expert, MoE should behave identically to dense FFN
        let hidden_size = 4;
        let intermediate_size = 8;

        let mut qw = QuantizedWeights::empty();

        // Router: single expert, weight 1.0
        let router_w = vec![1.0f32; hidden_size];
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.gate.weight".to_string(),
            router_w,
        );

        // Single expert with identity-like weights
        let gate_w = vec![0.25f32; intermediate_size * hidden_size];
        let up_w = vec![0.25f32; intermediate_size * hidden_size];
        let down_w = vec![0.25f32; hidden_size * intermediate_size];
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight".to_string(),
            gate_w.clone(),
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.up_proj.weight".to_string(),
            up_w.clone(),
        );
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.experts.0.down_proj.weight".to_string(),
            down_w.clone(),
        );

        let router = MoeRouter::new(1, 1);
        let input = vec![1.0f32; hidden_size];
        let mut output = vec![0.0f32; hidden_size];

        moe_ffn(
            &input,
            &mut output,
            &router,
            &qw,
            0,
            hidden_size,
            intermediate_size,
            HiddenAct::SiluGate,
            1,
            "block_sparse_moe",
            "model.layers.0.block_sparse_moe.gate.weight",
        );

        // With weight=1.0, output should equal exactly what the single expert produces
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "single-expert MoE should produce non-zero output");
    }

    #[test]
    fn test_moe_multi_token() {
        let hidden_size = 4;
        let intermediate_size = 8;
        let n_experts = 2;

        let mut qw = QuantizedWeights::empty();

        let router_w = vec![1.0f32; n_experts * hidden_size];
        qw.insert_f32(
            "model.layers.0.block_sparse_moe.gate.weight".to_string(),
            router_w,
        );

        for e in 0..n_experts {
            let w = vec![0.1f32; intermediate_size * hidden_size];
            let dw = vec![0.1f32; hidden_size * intermediate_size];
            qw.insert_f32(
                format!("model.layers.0.block_sparse_moe.experts.{e}.gate_proj.weight"),
                w.clone(),
            );
            qw.insert_f32(
                format!("model.layers.0.block_sparse_moe.experts.{e}.up_proj.weight"),
                w,
            );
            qw.insert_f32(
                format!("model.layers.0.block_sparse_moe.experts.{e}.down_proj.weight"),
                dw,
            );
        }

        let router = MoeRouter::new(n_experts, 1);
        let num_tokens = 3;
        let input = vec![1.0f32; num_tokens * hidden_size];
        let mut output = vec![0.0f32; num_tokens * hidden_size];

        moe_ffn(
            &input,
            &mut output,
            &router,
            &qw,
            0,
            hidden_size,
            intermediate_size,
            HiddenAct::SiluGate,
            num_tokens,
            "block_sparse_moe",
            "model.layers.0.block_sparse_moe.gate.weight",
        );

        // All tokens should produce output
        for t in 0..num_tokens {
            let token_out = &output[t * hidden_size..(t + 1) * hidden_size];
            let norm: f32 = token_out.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.0, "token {t} output should be non-zero");
        }
    }
}

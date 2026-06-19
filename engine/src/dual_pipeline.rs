use serde::Serialize;

use crate::config::ModelConfig;
use crate::session::InferenceSession;
use crate::trace::TraceLevel;
use crate::TransformerModel;
use tempura_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

#[derive(Debug, Clone, Serialize)]
pub struct LayerDivergence {
    pub layer: usize,
    pub max_abs_diff: f32,
    pub l2_distance: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    pub layers: Vec<LayerDivergence>,
    pub logits_max_abs_diff: f32,
    pub logits_l2_distance: f32,
    pub token_match: bool,
    pub token_a: u32,
    pub token_b: u32,
}

pub fn compare_pipelines<A1, N1, R1, M1, S1, A2, N2, R2, M2, S2>(
    model_a: &TransformerModel<A1, N1, R1, M1, S1>,
    model_b: &TransformerModel<A2, N2, R2, M2, S2>,
    config: &ModelConfig,
    token_ids: &[u32],
    params: &SamplingParams,
    seed: u64,
) -> ComparisonResult
where
    A1: Attention,
    N1: RMSNorm,
    R1: RoPE,
    M1: MatMul,
    S1: Sampler,
    A2: Attention,
    N2: RMSNorm,
    R2: RoPE,
    M2: MatMul,
    S2: Sampler,
{
    let mut session_a = InferenceSession::new(config, TraceLevel::Summary, seed);
    let mut session_b = InferenceSession::new(config, TraceLevel::Summary, seed);

    let logits_a = model_a.forward(&mut session_a, token_ids);
    let logits_b = model_b.forward(&mut session_b, token_ids);

    let mut layers = Vec::new();
    let num_layers = session_a
        .trace
        .layers
        .len()
        .min(session_b.trace.layers.len());
    for i in 0..num_layers {
        let snap_a = &session_a.trace.layers[i];
        let snap_b = &session_b.trace.layers[i];

        let (max_abs, l2) = match (&snap_a.pre_norm, &snap_b.pre_norm) {
            (Some(a), Some(b)) => {
                let diff_mean = (a.mean - b.mean).abs();
                let diff_var = (a.variance - b.variance).abs();
                let diff_max = (a.max_abs - b.max_abs).abs();
                let max_abs = diff_mean.max(diff_var).max(diff_max);
                let l2 = (diff_mean * diff_mean + diff_var * diff_var + diff_max * diff_max).sqrt();
                (max_abs, l2)
            }
            _ => (0.0, 0.0),
        };

        layers.push(LayerDivergence {
            layer: i,
            max_abs_diff: max_abs,
            l2_distance: l2,
        });
    }

    let logits_max_abs_diff = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let logits_l2_distance = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    let mut probs_a = vec![0.0f32; logits_a.len()];
    let mut probs_b = vec![0.0f32; logits_b.len()];
    model_a
        .pipeline
        .sampler
        .apply(&logits_a, &mut probs_a, params)
        .unwrap();
    model_b
        .pipeline
        .sampler
        .apply(&logits_b, &mut probs_b, params)
        .unwrap();

    let token_a = model_a.pipeline.sampler.sample_token(&probs_a, None);
    let token_b = model_b.pipeline.sampler.sample_token(&probs_b, None);

    ComparisonResult {
        layers,
        logits_max_abs_diff,
        logits_l2_distance,
        token_match: token_a == token_b,
        token_a,
        token_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::KernelPipeline;

    fn make_model_with_weights(
        config: &ModelConfig,
        seed: u64,
    ) -> TransformerModel<
        tempura_kernels::attention::CpuAttention,
        tempura_kernels::rmsnorm::CpuRMSNorm,
        tempura_kernels::rope::CpuRoPE,
        tempura_kernels::matmul::CpuMatMul,
        tempura_kernels::sampling::CpuSampler,
    > {
        use crate::loader::WeightLoader;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::collections::HashMap;

        let mut rng = StdRng::seed_from_u64(seed);
        let h = config.hidden_size;
        let v = config.vocab_size;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;

        let mut tensors = HashMap::new();
        let mut rand_vec =
            |size: usize| -> Vec<f32> { (0..size).map(|_| rng.random_range(-0.1..0.1)).collect() };

        tensors.insert("model.embed_tokens.weight".into(), rand_vec(v * h));
        tensors.insert("model.norm.weight".into(), vec![1.0; h]);
        tensors.insert("lm_head.weight".into(), rand_vec(v * h));

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");
            tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![1.0; h]);
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![1.0; h],
            );
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                rand_vec(q_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                rand_vec(kv_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                rand_vec(kv_dim * h),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                rand_vec(h * q_dim),
            );
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                rand_vec(inter * h),
            );
            tensors.insert(format!("{prefix}.mlp.up_proj.weight"), rand_vec(inter * h));
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                rand_vec(h * inter),
            );
        }

        struct Loader(HashMap<String, Vec<f32>>);
        impl WeightLoader for Loader {
            fn load_tensor(&self, name: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
                self.0
                    .get(name)
                    .cloned()
                    .ok_or_else(|| crate::loader::LoadError::MissingTensor(name.into()))
            }
            fn tensor_names(&self) -> Vec<String> {
                self.0.keys().cloned().collect()
            }
        }

        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        model.load_weights(&Loader(tensors)).unwrap();
        model
    }

    #[test]
    fn test_identical_pipelines_zero_divergence() {
        let config = ModelConfig::tiny_test();
        let model_a = make_model_with_weights(&config, 42);
        let model_b = make_model_with_weights(&config, 42);

        let params = SamplingParams::default();
        let result = compare_pipelines(&model_a, &model_b, &config, &[1, 2, 3], &params, 42);

        assert_eq!(result.logits_max_abs_diff, 0.0);
        assert_eq!(result.logits_l2_distance, 0.0);
        assert!(result.token_match);
        assert_eq!(result.token_a, result.token_b);
        for layer in &result.layers {
            assert_eq!(layer.max_abs_diff, 0.0);
            assert_eq!(layer.l2_distance, 0.0);
        }
    }
}

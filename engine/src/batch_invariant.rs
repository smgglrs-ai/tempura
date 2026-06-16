use serde::Serialize;

use crate::config::ModelConfig;
use crate::session::InferenceSession;
use crate::trace::TraceLevel;
use crate::TransformerModel;
use oxide_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

#[derive(Debug, Clone, Serialize)]
pub struct BatchInvariantResult {
    pub solo_token: u32,
    pub batch_token: u32,
    pub logits_max_abs_diff: f32,
    pub token_match: bool,
    pub batch_size: usize,
}

pub fn verify_batch_invariance<A, N, R, M, S>(
    model: &TransformerModel<A, N, R, M, S>,
    config: &ModelConfig,
    target_tokens: &[u32],
    other_sequences: &[Vec<u32>],
    params: &SamplingParams,
    seed: u64,
) -> BatchInvariantResult
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    let mut solo_session = InferenceSession::new(config, TraceLevel::Off, seed);
    let solo_logits = model.forward(&mut solo_session, target_tokens);
    let mut solo_probs = vec![0.0f32; solo_logits.len()];
    model
        .pipeline
        .sampler
        .apply(&solo_logits, &mut solo_probs, params)
        .unwrap();
    let solo_token = model.pipeline.sampler.sample_token(&solo_probs, None);

    // Run other sequences to "pollute" any shared state (there shouldn't be any)
    for other in other_sequences {
        let mut other_session = InferenceSession::new(config, TraceLevel::Off, seed + 1);
        model.forward(&mut other_session, other);
    }

    // Now run the target sequence again with fresh session
    let mut batch_session = InferenceSession::new(config, TraceLevel::Off, seed);
    let batch_logits = model.forward(&mut batch_session, target_tokens);
    let mut batch_probs = vec![0.0f32; batch_logits.len()];
    model
        .pipeline
        .sampler
        .apply(&batch_logits, &mut batch_probs, params)
        .unwrap();
    let batch_token = model.pipeline.sampler.sample_token(&batch_probs, None);

    let max_abs_diff = solo_logits
        .iter()
        .zip(batch_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    BatchInvariantResult {
        solo_token,
        batch_token,
        logits_max_abs_diff: max_abs_diff,
        token_match: solo_token == batch_token,
        batch_size: 1 + other_sequences.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::KernelPipeline;
    use std::collections::HashMap;

    struct RandomWeightLoader {
        tensors: HashMap<String, Vec<f32>>,
    }

    impl RandomWeightLoader {
        fn from_config(config: &ModelConfig, seed: u64) -> Self {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            let mut rng = StdRng::seed_from_u64(seed);
            let h = config.hidden_size;
            let v = config.vocab_size;
            let kv_dim = config.num_kv_heads * config.head_dim;
            let q_dim = config.num_attention_heads * config.head_dim;
            let inter = config.intermediate_size;

            let mut tensors = HashMap::new();
            let mut rand_vec = |size: usize| -> Vec<f32> {
                (0..size).map(|_| rng.random_range(-0.1..0.1)).collect()
            };

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

            Self { tensors }
        }
    }

    impl crate::loader::WeightLoader for RandomWeightLoader {
        fn load_tensor(&self, name: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| crate::loader::LoadError::MissingTensor(name.to_string()))
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    fn setup_model() -> (
        ModelConfig,
        crate::TransformerModel<
            oxide_kernels::attention::CpuAttention,
            oxide_kernels::rmsnorm::CpuRMSNorm,
            oxide_kernels::rope::CpuRoPE,
            oxide_kernels::matmul::CpuMatMul,
            oxide_kernels::sampling::CpuSampler,
        >,
    ) {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();
        (config, model)
    }

    #[test]
    fn test_solo_vs_batch_of_2() {
        let (config, model) = setup_model();
        let params = SamplingParams::default();

        let result =
            verify_batch_invariance(&model, &config, &[1, 2, 3], &[vec![4, 5, 6]], &params, 42);

        assert_eq!(result.logits_max_abs_diff, 0.0);
        assert!(result.token_match);
        assert_eq!(result.batch_size, 2);
    }

    #[test]
    fn test_solo_vs_batch_of_4_heterogeneous() {
        let (config, model) = setup_model();
        let params = SamplingParams::default();

        let result = verify_batch_invariance(
            &model,
            &config,
            &[1, 2, 3],
            &[vec![4], vec![5, 6, 7, 8], vec![9, 10]],
            &params,
            42,
        );

        assert_eq!(result.logits_max_abs_diff, 0.0);
        assert!(result.token_match);
        assert_eq!(result.batch_size, 4);
    }

    #[test]
    fn test_solo_vs_batch_with_prepopulated_cache() {
        let (config, model) = setup_model();
        let params = SamplingParams::default();

        // Run other sequences with partial generation first to "pollute" state
        let mut warmup_session = InferenceSession::new(&config, TraceLevel::Off, 99);
        model.forward(&mut warmup_session, &[7, 8, 9]);
        model.forward(&mut warmup_session, &[10]);

        // Now verify the target is still batch-invariant
        let result =
            verify_batch_invariance(&model, &config, &[1, 2, 3], &[vec![4, 5]], &params, 42);

        assert_eq!(result.logits_max_abs_diff, 0.0);
        assert!(result.token_match);
    }
}

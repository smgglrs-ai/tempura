use crate::config::ModelConfig;
use crate::model::TransformerModel;
use crate::session::{InferenceSession, SessionTransfer};
use crate::trace::TraceLevel;
use oxide_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

pub trait PrefillWorker: Send + Sync {
    fn prefill(&self, token_ids: &[u32], seed: u64) -> SessionTransfer;
}

pub trait DecodeWorker: Send + Sync {
    fn decode(
        &self,
        transfer: SessionTransfer,
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32>;
}

pub struct LocalPrefillWorker<A, N, R, M, S> {
    model: std::sync::Arc<TransformerModel<A, N, R, M, S>>,
    config: ModelConfig,
}

impl<A, N, R, M, S> LocalPrefillWorker<A, N, R, M, S> {
    pub fn new(
        model: std::sync::Arc<TransformerModel<A, N, R, M, S>>,
        config: ModelConfig,
    ) -> Self {
        Self { model, config }
    }
}

impl<A, N, R, M, S> PrefillWorker for LocalPrefillWorker<A, N, R, M, S>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    fn prefill(&self, token_ids: &[u32], seed: u64) -> SessionTransfer {
        let mut session = InferenceSession::new(&self.config, TraceLevel::Off, seed);
        let logits = self.model.forward(&mut session, token_ids);
        session.to_transfer_with_logits(logits)
    }
}

pub struct LocalDecodeWorker<A, N, R, M, S> {
    model: std::sync::Arc<TransformerModel<A, N, R, M, S>>,
    config: ModelConfig,
}

impl<A, N, R, M, S> LocalDecodeWorker<A, N, R, M, S> {
    pub fn new(
        model: std::sync::Arc<TransformerModel<A, N, R, M, S>>,
        config: ModelConfig,
    ) -> Self {
        Self { model, config }
    }
}

impl<A, N, R, M, S> DecodeWorker for LocalDecodeWorker<A, N, R, M, S>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    fn decode(
        &self,
        transfer: SessionTransfer,
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32> {
        let last_logits = transfer.last_logits.clone();
        let mut session =
            InferenceSession::from_transfer(transfer, &self.config, TraceLevel::Off);

        let mut probs = vec![0.0f32; last_logits.len()];
        self.model
            .pipeline
            .sampler
            .apply(&last_logits, &mut probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut session.rng)
        };
        let first_token = self.model.pipeline.sampler.sample_token(&probs, rng_ref);
        let mut generated = vec![first_token];

        for _ in 1..max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let mut step_logits =
                self.model.forward(&mut session, &[*generated.last().unwrap()]);
            params.apply_penalties(&mut step_logits, &generated);
            let mut step_probs = vec![0.0f32; step_logits.len()];
            self.model
                .pipeline
                .sampler
                .apply(&step_logits, &mut step_probs, params)
                .unwrap();

            let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                None
            } else {
                Some(&mut session.rng)
            };
            let token = self
                .model
                .pipeline
                .sampler
                .sample_token(&step_probs, rng_ref);
            generated.push(token);
        }

        generated
    }
}

pub struct DisaggregatedPipeline {
    prefill: Box<dyn PrefillWorker>,
    decode: Box<dyn DecodeWorker>,
}

impl DisaggregatedPipeline {
    pub fn new(prefill: Box<dyn PrefillWorker>, decode: Box<dyn DecodeWorker>) -> Self {
        Self { prefill, decode }
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
        seed: u64,
    ) -> Vec<u32> {
        let transfer = self.prefill.prefill(token_ids, seed);
        self.decode
            .decode(transfer, max_new_tokens, params, eos_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::KernelPipeline;
    use std::collections::HashMap;
    use std::sync::Arc;

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

    fn setup() -> (ModelConfig, Arc<TransformerModel<
        oxide_kernels::attention::CpuAttention,
        oxide_kernels::rmsnorm::CpuRMSNorm,
        oxide_kernels::rope::CpuRoPE,
        oxide_kernels::matmul::CpuMatMul,
        oxide_kernels::sampling::CpuSampler,
    >>) {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();
        (config, Arc::new(model))
    }

    #[test]
    fn test_session_transfer_round_trip() {
        let (config, model) = setup();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        let transfer = session.to_transfer_with_logits(logits.clone());
        assert_eq!(transfer.position, 3);
        assert_eq!(transfer.seed, 42);
        assert_eq!(transfer.cache.seq_len, 3);
        assert_eq!(transfer.last_logits.len(), config.vocab_size);

        let restored = InferenceSession::from_transfer(transfer, &config, TraceLevel::Off);
        assert_eq!(restored.position, 3);
        assert_eq!(restored.seed, 42);
        assert_eq!(restored.cache.seq_len, 3);

        for layer in 0..config.num_layers {
            assert_eq!(
                session.cache.k_slice(layer),
                restored.cache.k_slice(layer),
                "K data mismatch at layer {layer}"
            );
            assert_eq!(
                session.cache.v_slice(layer),
                restored.cache.v_slice(layer),
                "V data mismatch at layer {layer}"
            );
        }
    }

    #[test]
    fn test_disaggregated_produces_tokens() {
        let (config, model) = setup();

        let prefill = LocalPrefillWorker::new(model.clone(), config.clone());
        let decode = LocalDecodeWorker::new(model.clone(), config.clone());

        let pipeline = DisaggregatedPipeline::new(Box::new(prefill), Box::new(decode));

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let tokens = pipeline.generate(&[1, 2, 3], 5, &params, None, 42);
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_disaggregated_deterministic() {
        let (config, model) = setup();

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let prefill1 = LocalPrefillWorker::new(model.clone(), config.clone());
        let decode1 = LocalDecodeWorker::new(model.clone(), config.clone());
        let pipe1 = DisaggregatedPipeline::new(Box::new(prefill1), Box::new(decode1));
        let tokens1 = pipe1.generate(&[1, 2, 3], 5, &params, None, 42);

        let prefill2 = LocalPrefillWorker::new(model.clone(), config.clone());
        let decode2 = LocalDecodeWorker::new(model.clone(), config.clone());
        let pipe2 = DisaggregatedPipeline::new(Box::new(prefill2), Box::new(decode2));
        let tokens2 = pipe2.generate(&[1, 2, 3], 5, &params, None, 42);

        assert_eq!(tokens1, tokens2, "same seed must produce identical output");
    }

    #[test]
    fn test_disaggregated_matches_single_worker() {
        let config = ModelConfig::tiny_test();
        let loader = RandomWeightLoader::from_config(&config, 42);

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        // Single-worker generate
        let pipeline_single = KernelPipeline::cpu_verified();
        let mut model_single = TransformerModel::new(config.clone(), pipeline_single);
        model_single.load_weights(&loader).unwrap();
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let single_tokens = model_single.generate(&mut session, &[1, 2, 3], 5, &params, None);

        // Disaggregated: prefill on model A, decode on model B
        let pipeline_a = KernelPipeline::cpu_verified();
        let mut model_a = TransformerModel::new(config.clone(), pipeline_a);
        model_a.load_weights(&loader).unwrap();

        let pipeline_b = KernelPipeline::cpu_verified();
        let mut model_b = TransformerModel::new(config.clone(), pipeline_b);
        model_b.load_weights(&loader).unwrap();

        let prefill = LocalPrefillWorker::new(Arc::new(model_a), config.clone());
        let decode = LocalDecodeWorker::new(Arc::new(model_b), config.clone());
        let pipe = DisaggregatedPipeline::new(Box::new(prefill), Box::new(decode));
        let disagg_tokens = pipe.generate(&[1, 2, 3], 5, &params, None, 42);

        assert_eq!(
            single_tokens, disagg_tokens,
            "disaggregated must produce identical output to single-worker"
        );
    }
}

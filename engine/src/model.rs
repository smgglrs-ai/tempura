use crate::config::ModelConfig;
use crate::pipeline::KernelPipeline;
use crate::session::InferenceSession;
use crate::trace::{LayerSnapshot, TraceLevel};
use crate::weights::ModelWeights;
use oxide_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

pub struct TransformerModel<A, N, R, M, S> {
    pub config: ModelConfig,
    pub pipeline: KernelPipeline<A, N, R, M, S>,
    pub weights: Option<ModelWeights>,
}

impl<A, N, R, M, S> TransformerModel<A, N, R, M, S>
where
    A: Attention,
    N: RMSNorm,
    R: RoPE,
    M: MatMul,
    S: Sampler,
{
    pub fn new(config: ModelConfig, pipeline: KernelPipeline<A, N, R, M, S>) -> Self {
        Self {
            config,
            pipeline,
            weights: None,
        }
    }

    pub fn load_weights(
        &mut self,
        loader: &dyn crate::loader::WeightLoader,
    ) -> Result<(), crate::weights::WeightsError> {
        self.weights = Some(ModelWeights::load(loader, &self.config)?);
        Ok(())
    }

    pub fn forward(&self, session: &mut InferenceSession, token_ids: &[u32]) -> Vec<f32> {
        let weights = match &self.weights {
            Some(w) => w,
            None => return vec![0.0; self.config.vocab_size],
        };

        let cfg = &self.config;
        let seq_len = token_ids.len();
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;

        let embed = weights.get("model.embed_tokens.weight").unwrap();
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            let src = &embed[tok as usize * h..(tok as usize + 1) * h];
            hidden[t * h..(t + 1) * h].copy_from_slice(src);
        }

        let positions: Vec<u32> = (session.position..session.position + seq_len)
            .map(|p| p as u32)
            .collect();

        for layer in 0..cfg.num_layers {
            let prefix = format!("model.layers.{layer}");

            let mut snapshot = LayerSnapshot {
                layer_index: layer,
                pre_norm: None,
                post_norm: None,
                attention_scores: None,
                duration: None,
            };

            // --- attention block ---
            let mut normed = hidden.clone();
            let input_ln_w = weights
                .get(&format!("{prefix}.input_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm(
                    &mut normed,
                    input_ln_w,
                    h,
                    cfg.layer_norm_eps,
                    session.trace_level,
                    &mut Some(&mut snapshot),
                )
                .unwrap();

            let q_w = weights
                .get(&format!("{prefix}.self_attn.q_proj.weight"))
                .unwrap();
            let k_w = weights
                .get(&format!("{prefix}.self_attn.k_proj.weight"))
                .unwrap();
            let v_w = weights
                .get(&format!("{prefix}.self_attn.v_proj.weight"))
                .unwrap();
            let o_w = weights
                .get(&format!("{prefix}.self_attn.o_proj.weight"))
                .unwrap();

            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;
            let mut q = vec![0.0f32; seq_len * q_dim];
            let mut k = vec![0.0f32; seq_len * kv_dim];
            let mut v = vec![0.0f32; seq_len * kv_dim];

            self.pipeline
                .matmul(&normed, q_w, &mut q, seq_len, q_dim, h)
                .unwrap();
            self.pipeline
                .matmul(&normed, k_w, &mut k, seq_len, kv_dim, h)
                .unwrap();
            self.pipeline
                .matmul(&normed, v_w, &mut v, seq_len, kv_dim, h)
                .unwrap();

            self.pipeline
                .rope(
                    &mut q,
                    1,
                    seq_len,
                    n_heads,
                    head_dim,
                    &positions,
                    cfg.rope_theta,
                )
                .unwrap();
            self.pipeline
                .rope(
                    &mut k,
                    1,
                    seq_len,
                    n_kv_heads,
                    head_dim,
                    &positions,
                    cfg.rope_theta,
                )
                .unwrap();

            session.cache.append_kv(layer, &k, &v, seq_len);
            let kv_len = session.cache.seq_len + seq_len;
            let kv_stride = n_kv_heads * head_dim;
            let kv_slice_len = kv_len * kv_stride;
            let k_cache = &session.cache.k[layer][..kv_slice_len];
            let v_cache = &session.cache.v[layer][..kv_slice_len];

            let mut attn_out = vec![0.0f32; seq_len * q_dim];
            self.pipeline
                .attention(
                    &q,
                    k_cache,
                    v_cache,
                    &mut attn_out,
                    1,
                    seq_len,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                )
                .unwrap();

            let mut o_proj = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul(&attn_out, o_w, &mut o_proj, seq_len, h, q_dim)
                .unwrap();

            for i in 0..hidden.len() {
                hidden[i] += o_proj[i];
            }

            // --- FFN block ---
            let mut normed2 = hidden.clone();
            let post_ln_w = weights
                .get(&format!("{prefix}.post_attention_layernorm.weight"))
                .unwrap();
            self.pipeline
                .rmsnorm(
                    &mut normed2,
                    post_ln_w,
                    h,
                    cfg.layer_norm_eps,
                    TraceLevel::Off,
                    &mut None,
                )
                .unwrap();

            let gate_w = weights
                .get(&format!("{prefix}.mlp.gate_proj.weight"))
                .unwrap();
            let up_w = weights
                .get(&format!("{prefix}.mlp.up_proj.weight"))
                .unwrap();
            let down_w = weights
                .get(&format!("{prefix}.mlp.down_proj.weight"))
                .unwrap();

            let mut gate = vec![0.0f32; seq_len * inter];
            let mut up = vec![0.0f32; seq_len * inter];
            self.pipeline
                .matmul(&normed2, gate_w, &mut gate, seq_len, inter, h)
                .unwrap();
            self.pipeline
                .matmul(&normed2, up_w, &mut up, seq_len, inter, h)
                .unwrap();

            // SiLU(gate) * up
            for i in 0..gate.len() {
                let silu = gate[i] / (1.0 + (-gate[i]).exp());
                gate[i] = silu * up[i];
            }

            let mut down_out = vec![0.0f32; seq_len * h];
            self.pipeline
                .matmul(&gate, down_w, &mut down_out, seq_len, h, inter)
                .unwrap();

            for i in 0..hidden.len() {
                hidden[i] += down_out[i];
            }

            if session.trace_level >= TraceLevel::Summary {
                session.trace.layers.push(snapshot);
            }
        }

        session.cache.advance(seq_len);
        session.position += seq_len;

        // final norm
        let final_norm_w = weights.get("model.norm.weight").unwrap();
        self.pipeline
            .rmsnorm(
                &mut hidden,
                final_norm_w,
                h,
                cfg.layer_norm_eps,
                TraceLevel::Off,
                &mut None,
            )
            .unwrap();

        // lm_head: project last token to vocab
        let lm_head_w = weights.get("lm_head.weight").unwrap();
        let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.pipeline
            .matmul(last_hidden, lm_head_w, &mut logits, 1, cfg.vocab_size, h)
            .unwrap();

        logits
    }

    pub fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
    ) -> oxide_kernels::Result<Vec<f32>> {
        let mut probs = vec![0.0; logits.len()];
        self.pipeline.sampler.apply(logits, &mut probs, params)?;
        Ok(probs)
    }
}

pub use oxide_kernels::sampling::SamplingParams as SampleParams;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::KernelPipeline;
    use crate::weights::ModelWeights;
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

    #[test]
    fn test_forward_shape_correctness() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(session.position, 3);
    }

    #[test]
    fn test_forward_nonzero_logits() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let logits = model.forward(&mut session, &[1, 2, 3]);

        let all_zero = logits.iter().all(|&v| v == 0.0);
        assert!(!all_zero, "logits should be non-zero with random weights");
    }

    #[test]
    fn test_forward_with_trace() {
        let config = ModelConfig::tiny_test();
        let pipeline = KernelPipeline::cpu_verified();
        let mut model = TransformerModel::new(config.clone(), pipeline);
        let loader = RandomWeightLoader::from_config(&config, 42);
        model.load_weights(&loader).unwrap();

        let mut session = InferenceSession::new(&config, TraceLevel::Summary, 42);
        let _logits = model.forward(&mut session, &[1, 2]);

        assert_eq!(session.trace.layers.len(), config.num_layers);
        for snap in &session.trace.layers {
            assert!(snap.pre_norm.is_some());
            assert!(snap.post_norm.is_some());
        }
    }
}

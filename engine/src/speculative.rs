use crate::session::InferenceSession;
use crate::trace::{SpeculativeSnapshot, TraceLevel};
use crate::TransformerModel;
use oxide_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

pub struct SpeculativeDecoder<'a, A1, N1, R1, M1, S1, A2, N2, R2, M2, S2>
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
    pub target: &'a TransformerModel<A1, N1, R1, M1, S1>,
    pub draft: &'a TransformerModel<A2, N2, R2, M2, S2>,
    pub k: usize,
}

impl<'a, A1, N1, R1, M1, S1, A2, N2, R2, M2, S2>
    SpeculativeDecoder<'a, A1, N1, R1, M1, S1, A2, N2, R2, M2, S2>
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
    pub fn new(
        target: &'a TransformerModel<A1, N1, R1, M1, S1>,
        draft: &'a TransformerModel<A2, N2, R2, M2, S2>,
        k: usize,
    ) -> Self {
        Self { target, draft, k }
    }

    pub fn generate(
        &self,
        target_session: &mut InferenceSession,
        draft_session: &mut InferenceSession,
        token_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        eos_token: Option<u32>,
    ) -> Vec<u32> {
        let target_logits = self.target.forward(target_session, token_ids);
        let _ = self.draft.forward(draft_session, token_ids);
        target_session.trace.prompt_tokens = token_ids.len();

        let mut target_probs = vec![0.0f32; target_logits.len()];
        self.target
            .pipeline
            .sampler
            .apply(&target_logits, &mut target_probs, params)
            .unwrap();

        let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
            None
        } else {
            Some(&mut target_session.rng)
        };
        let first_token = self
            .target
            .pipeline
            .sampler
            .sample_token(&target_probs, rng_ref);
        let mut generated = vec![first_token];

        let mut step = 0;

        while generated.len() < max_new_tokens {
            if eos_token == Some(*generated.last().unwrap()) {
                break;
            }

            let saved_target_pos = target_session.position;
            let saved_target_seq_len = target_session.cache.seq_len;
            let saved_draft_pos = draft_session.position;
            let saved_draft_seq_len = draft_session.cache.seq_len;

            let mut draft_tokens = Vec::with_capacity(self.k);
            let mut draft_probs_vec = Vec::new();

            let mut last_token = *generated.last().unwrap();
            for _ in 0..self.k {
                if generated.len() + draft_tokens.len() >= max_new_tokens {
                    break;
                }

                let logits = self.draft.forward(draft_session, &[last_token]);
                let mut probs = vec![0.0f32; logits.len()];
                self.draft
                    .pipeline
                    .sampler
                    .apply(&logits, &mut probs, params)
                    .unwrap();

                let token = self.draft.pipeline.sampler.sample_token(&probs, None);
                draft_tokens.push(token);
                draft_probs_vec.push(probs[token as usize]);
                last_token = token;
            }

            if draft_tokens.is_empty() {
                let logits = self
                    .target
                    .forward(target_session, &[*generated.last().unwrap()]);
                let mut probs = vec![0.0f32; logits.len()];
                self.target
                    .pipeline
                    .sampler
                    .apply(&logits, &mut probs, params)
                    .unwrap();
                let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                    None
                } else {
                    Some(&mut target_session.rng)
                };
                let token = self.target.pipeline.sampler.sample_token(&probs, rng_ref);
                generated.push(token);
                continue;
            }

            target_session.cache.rollback(saved_target_seq_len);
            target_session.position = saved_target_pos;

            let mut accepted = Vec::new();
            let mut rejected = Vec::new();
            let mut verify_token = *generated.last().unwrap();

            for (i, &draft_token) in draft_tokens.iter().enumerate() {
                let target_logits = self.target.forward(target_session, &[verify_token]);
                let mut target_probs = vec![0.0f32; target_logits.len()];
                self.target
                    .pipeline
                    .sampler
                    .apply(&target_logits, &mut target_probs, params)
                    .unwrap();

                let target_prob = target_probs[draft_token as usize];
                let draft_prob = draft_probs_vec[i];

                if params.temperature == 0.0 {
                    let target_argmax = target_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);
                    if draft_token == target_argmax {
                        accepted.push(draft_token);
                        verify_token = draft_token;
                    } else {
                        rejected.push(draft_token);
                        accepted.push(target_argmax);
                        break;
                    }
                } else if target_prob >= draft_prob {
                    accepted.push(draft_token);
                    verify_token = draft_token;
                } else {
                    rejected.push(draft_token);
                    let rng_ref: Option<&mut dyn rand::RngCore> = Some(&mut target_session.rng);
                    let corrected = self
                        .target
                        .pipeline
                        .sampler
                        .sample_token(&target_probs, rng_ref);
                    accepted.push(corrected);
                    break;
                }
            }

            if rejected.is_empty() {
                let last_logits = self
                    .target
                    .forward(target_session, &[*accepted.last().unwrap()]);
                let mut last_probs = vec![0.0f32; last_logits.len()];
                self.target
                    .pipeline
                    .sampler
                    .apply(&last_logits, &mut last_probs, params)
                    .unwrap();
                let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                    None
                } else {
                    Some(&mut target_session.rng)
                };
                let bonus = self
                    .target
                    .pipeline
                    .sampler
                    .sample_token(&last_probs, rng_ref);
                accepted.push(bonus);
            }

            let num_accepted_from_draft = accepted.len().saturating_sub(1);
            let num_draft = draft_tokens.len();

            draft_session.cache.rollback(saved_draft_seq_len);
            draft_session.position = saved_draft_pos;
            for &token in &accepted {
                let _ = self.draft.forward(
                    draft_session,
                    &[if generated.is_empty() {
                        token
                    } else {
                        *generated.last().unwrap_or(&token)
                    }],
                );
            }

            generated.extend_from_slice(&accepted);

            if target_session.trace_level >= TraceLevel::Summary {
                let acceptance_rate = if num_draft > 0 {
                    num_accepted_from_draft as f32 / num_draft as f32
                } else {
                    0.0
                };
                target_session
                    .trace
                    .speculative_steps
                    .push(SpeculativeSnapshot {
                        step,
                        draft_proposed: draft_tokens,
                        accepted: accepted[..accepted.len().min(num_accepted_from_draft)].to_vec(),
                        rejected,
                        acceptance_rate,
                    });
            }

            step += 1;
        }

        if generated.len() > max_new_tokens {
            generated.truncate(max_new_tokens);
        }

        target_session.trace.generated_tokens = generated.len();
        generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::pipeline::KernelPipeline;

    fn make_model_with_weights(
        config: &ModelConfig,
        seed: u64,
    ) -> TransformerModel<
        oxide_kernels::attention::CpuAttention,
        oxide_kernels::rmsnorm::CpuRMSNorm,
        oxide_kernels::rope::CpuRoPE,
        oxide_kernels::matmul::CpuMatMul,
        oxide_kernels::sampling::CpuSampler,
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
    fn test_speculative_produces_tokens() {
        let config = ModelConfig::tiny_test();
        let target = make_model_with_weights(&config, 42);
        let draft = make_model_with_weights(&config, 42);
        let decoder = SpeculativeDecoder::new(&target, &draft, 3);

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let mut target_session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let mut draft_session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens = decoder.generate(
            &mut target_session,
            &mut draft_session,
            &[1, 2],
            5,
            &params,
            None,
        );

        assert_eq!(tokens.len(), 5, "should produce exactly max_new_tokens");
    }

    #[test]
    fn test_speculative_records_trace() {
        let config = ModelConfig::tiny_test();
        let target = make_model_with_weights(&config, 42);
        let draft = make_model_with_weights(&config, 42);
        let decoder = SpeculativeDecoder::new(&target, &draft, 2);

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let mut target_session = InferenceSession::new(&config, TraceLevel::Summary, 42);
        let mut draft_session = InferenceSession::new(&config, TraceLevel::Off, 42);
        let tokens = decoder.generate(
            &mut target_session,
            &mut draft_session,
            &[1],
            4,
            &params,
            None,
        );

        assert!(!tokens.is_empty());
        assert_eq!(target_session.trace.prompt_tokens, 1);
        assert_eq!(target_session.trace.generated_tokens, tokens.len());

        for snap in &target_session.trace.speculative_steps {
            assert!(!snap.draft_proposed.is_empty());
            assert!(snap.acceptance_rate >= 0.0 && snap.acceptance_rate <= 1.0);
        }
    }

    #[test]
    fn test_kv_cache_rollback() {
        let config = ModelConfig::tiny_test();
        let mut cache = crate::kv_cache::KvCache::new(&config);

        let stride = config.num_kv_heads * config.head_dim;
        for i in 0..3 {
            let k: Vec<f32> = (0..stride).map(|j| (i * stride + j) as f32).collect();
            let v = k.clone();
            cache.append_kv(0, &k, &v, 1);
            cache.advance(1);
        }
        assert_eq!(cache.seq_len, 3);

        cache.rollback(1);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k_slice(0).len(), stride);
        assert_eq!(cache.remaining_capacity(), config.max_seq_len - 1);
    }

    #[test]
    fn test_kv_cache_rollback_to_zero() {
        let config = ModelConfig::tiny_test();
        let mut cache = crate::kv_cache::KvCache::new(&config);

        let stride = config.num_kv_heads * config.head_dim;
        let k: Vec<f32> = (0..stride).map(|j| j as f32).collect();
        let v = k.clone();
        cache.append_kv(0, &k, &v, 1);
        cache.advance(1);

        cache.rollback(0);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.remaining_capacity(), config.max_seq_len);
    }
}

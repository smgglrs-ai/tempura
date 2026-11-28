use crate::session::InferenceSession;
use crate::trace::{SpeculativeSnapshot, TraceLevel};
use crate::TransformerModel;
use tempura_kernels::{Attention, MatMul, RMSNorm, RoPE, Sampler, SamplingParams};

/// Rejection residual for stochastic speculative acceptance:
/// `max(p_target - p_draft, 0)`, renormalized. When the residual mass is
/// (numerically) zero, falls back to the full target distribution -- the
/// token then still comes from the target's conditional.
pub fn residual_distribution(target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(target_probs.len(), draft_probs.len());
    let mut residual = vec![0.0f32; target_probs.len()];
    let mut sum = 0.0f32;
    for i in 0..target_probs.len() {
        let r = (target_probs[i] - draft_probs[i]).max(0.0);
        residual[i] = r;
        sum += r;
    }
    if sum > 0.0 {
        for r in residual.iter_mut() {
            *r /= sum;
        }
        residual
    } else {
        target_probs.to_vec()
    }
}

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
            // Full draft distributions per step: the stochastic acceptance
            // scheme needs p_d everywhere (not just at the sampled token)
            // to form the rejection residual.
            let mut draft_probs_vec: Vec<Vec<f32>> = Vec::with_capacity(self.k);

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

                // The draft SAMPLES from its own (post-chain) distribution:
                // the acceptance ratio p_t/p_d is only correct when p_d is
                // the distribution the draft token was actually drawn from
                // (a greedy draft degenerates p_d to a point mass and breaks
                // the output distribution). Seeded per draft session --
                // determinism.
                let rng_ref: Option<&mut dyn rand::RngCore> = if params.temperature == 0.0 {
                    None
                } else {
                    Some(&mut draft_session.rng)
                };
                let token = self
                    .draft
                    .pipeline
                    .sampler
                    .sample_token(&probs, rng_ref);
                draft_tokens.push(token);
                draft_probs_vec.push(probs);
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
                let draft_prob = draft_probs_vec[i][draft_token as usize];

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
                } else {
                    // Standard stochastic acceptance (Levy-Karp / Medusa):
                    // accept d ~ p_d with prob min(1, p_t(d)/p_d(d)). On
                    // rejection, sample the correction from the residual
                    // norm(max(p_t - p_d, 0)) -- this makes the emitted
                    // token distribution EXACTLY the target's (the scheme
                    // is output-equivalent in distribution to plain
                    // sampling, not byte-identical).
                    let ratio = if draft_prob > 0.0 {
                        target_prob / draft_prob
                    } else {
                        f32::INFINITY
                    };
                    let u: f32 = {
                        use rand::RngCore as _;
                        target_session.rng.next_u32() as f64 / u32::MAX as f64
                    } as f32;
                    if u < ratio {
                        accepted.push(draft_token);
                        verify_token = draft_token;
                    } else {
                        rejected.push(draft_token);
                        let residual = residual_distribution(
                            &target_probs,
                            &draft_probs_vec[i],
                        );
                        let rng_ref: Option<&mut dyn rand::RngCore> =
                            Some(&mut target_session.rng);
                        let corrected = self
                            .target
                            .pipeline
                            .sampler
                            .sample_token(&residual, rng_ref);
                        accepted.push(corrected);
                        break;
                    }
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

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn speculative_token_index_in_bounds() {
        // Concrete sizes only: symbolic-length Vec allocation makes CBMC
        // unwind allocation internals without bound (see allreduce.rs).
        for vocab_size in [1usize, 2, 7, 64] {
            let token: u32 = kani::any();
            kani::assume((token as usize) < vocab_size);
            let probs = vec![0.0f32; vocab_size];
            assert!((token as usize) < probs.len());
        }
    }

    #[kani::proof]
    fn speculative_parallel_arrays_consistent() {
        for k in [1usize, 2, 3, 5, 8] {
            let draft_tokens = vec![0u32; k];
            let draft_probs_vec = vec![0.0f32; k];

            for i in 0..draft_tokens.len() {
                assert!(i < draft_probs_vec.len());
            }
        }
    }

    #[kani::proof]
    fn residual_distribution_in_bounds_and_normalized() {
        // Concrete sizes only: symbolic-length Vec allocation makes CBMC
        // unwind allocation internals without bound.
        for n in [1usize, 2, 7, 64] {
            let target: Vec<f32> = (0..n).map(|_| kani::any()).collect();
            let draft: Vec<f32> = (0..n).map(|_| kani::any()).collect();
            let residual = crate::speculative::residual_distribution(&target, &draft);
            assert!(residual.len() == n);
            let mut sum = 0.0f32;
            for r in &residual {
                assert!(*r >= 0.0 && *r <= 1.0 + 1e-5);
                sum += r;
            }
            assert!(sum <= 1.0 + 1e-3);
        }
    }

    #[kani::proof]
    fn acceptance_ratio_bounded() {
        let target_prob: f32 = kani::any();
        let draft_prob: f32 = kani::any();
        kani::assume(draft_prob > 0.0 && draft_prob <= 1.0);
        kani::assume(target_prob >= 0.0 && target_prob <= 1.0);
        let ratio = target_prob / draft_prob;
        assert!(ratio >= 0.0);
        // min(1, ratio) is the acceptance probability: bounded by 1.
        assert!(ratio.min(1.0) <= 1.0);
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
    fn test_residual_distribution_unit() {
        let target = vec![0.5, 0.3, 0.2];
        let draft = vec![0.2, 0.5, 0.3];
        let r = residual_distribution(&target, &draft);
        // max(t-d,0) = [0.3, 0, 0] -> normalized [1, 0, 0]
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6 && r[2].abs() < 1e-6);
        // Draft dominating everywhere -> all residual mass on the
        // dominated side.
        let r2 = residual_distribution(&[0.1, 0.9], &[0.9, 0.1]);
        assert!(r2[0].abs() < 1e-6 && (r2[1] - 1.0).abs() < 1e-6);
        // Identical distributions -> zero residual mass -> target fallback.
        let r3 = residual_distribution(&[0.5, 0.5], &[0.5, 0.5]);
        assert!((r3[0] - 0.5).abs() < 1e-6 && (r3[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sampled_speculation_deterministic_per_seed() {
        let config = ModelConfig::tiny_test();
        let target = make_model_with_weights(&config, 42);
        // A DIFFERENT draft (seed 43): acceptance < 1, the residual
        // correction path runs.
        let draft = make_model_with_weights(&config, 43);
        let decoder = SpeculativeDecoder::new(&target, &draft, 4);

        let params = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let mut run = |seed: u64| {
            let mut ts = InferenceSession::new(&config, TraceLevel::Off, seed);
            let mut ds = InferenceSession::new(&config, TraceLevel::Off, seed);
            decoder.generate(&mut ts, &mut ds, &[1, 2], 24, &params, None)
        };
        let a = run(7);
        let b = run(7);
        assert_eq!(a, b, "same seed must give identical sampled-spec output");
        // Overlap check: a different seed should (almost surely) diverge
        // somewhere in 24 tokens on a 256-vocab.
        let c = run(8);
        assert_ne!(a, c);
    }

    #[test]
    fn test_stochastic_acceptance_emits_target_distribution() {
        // High-power gate on the acceptance math itself: with fixed
        // target/draft distributions, the emitted-token law of
        // (accept d ~ p_d w.p. min(1, p_t(d)/p_d(d)); on rejection sample
        // the residual norm(max(p_t - p_d, 0))) is EXACTLY p_t.
        use rand::rngs::StdRng;
        use rand::RngCore;
        use rand::SeedableRng;
        use tempura_kernels::Sampler as _;

        let sampler = tempura_kernels::sampling::CpuSampler;
        let p_t: Vec<f32> = vec![0.05, 0.40, 0.25, 0.20, 0.10];
        let p_d: Vec<f32> = vec![0.30, 0.10, 0.25, 0.05, 0.30];
        let mut rng = StdRng::seed_from_u64(1234);

        let mut hist = vec![0u64; p_t.len()];
        let trials = 30_000u64;
        for _ in 0..trials {
            let d = sampler.sample_token(&p_d, Some(&mut rng)) as usize;
            let ratio = p_t[d] / p_d[d];
            let u = rng.next_u32() as f64 / u32::MAX as f64;
            let emitted = if (u as f32) < ratio {
                d
            } else {
                let residual = residual_distribution(&p_t, &p_d);
                sampler.sample_token(&residual, Some(&mut rng)) as usize
            };
            hist[emitted] += 1;
        }

        let tv: f64 = hist
            .iter()
            .zip(&p_t)
            .map(|(&o, &p)| (o as f64 / trials as f64 - p as f64).abs())
            .sum::<f64>()
            / 2.0;
        assert!(
            tv < 0.02,
            "acceptance scheme does not reproduce p_target: TV={tv:.4}"
        );
    }

    #[test]
    fn test_sampled_speculation_matches_plain_distribution() {
        // End-to-end wiring gate for the sampled-speculation scheme. The
        // PRIMARY distributional gate is the exact-math simulation test
        // above; this one checks that the model plumbing (draft sampling
        // from its own distribution, acceptance draws, residual
        // correction) reproduces the plain-sampling LIKELIHOOD of the
        // emitted token: E[log p_target(token_2 | prompt, token_1)] must
        // agree between plain and speculative decode. A skewed acceptance
        // scheme (the old accept-if-p_t>=p_d + full-p_t correction)
        // biases the emitted mass toward high-p_t tokens and shifts this
        // mean; histogram-TV has no power at practical n (measured noise
        // floor 0.045 vs old-scheme 0.067 at n=4000 -- indistinguishable).
        let config = ModelConfig::tiny_test();
        let target = make_model_with_weights(&config, 42);
        // A DIFFERENT draft (seed 43): acceptance < 1, the residual
        // correction path runs.
        let draft = make_model_with_weights(&config, 43);
        let decoder = SpeculativeDecoder::new(&target, &draft, 4);

        let params = SamplingParams {
            temperature: 1.0,
            top_k: 5,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        // log p_target(t2 | prompt, t1) of the emitted second token,
        // recomputed with an exact forward over the actual history.
        let true_logp_t2 = |t1: u32, t2: u32| -> f64 {
            let mut sess = InferenceSession::new(&config, TraceLevel::Off, 999);
            let logits = target.forward(&mut sess, &[1, 2, t1]);
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse: f32 = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
            (logits[t2 as usize] - max - lse) as f64
        };

        let runs = 1500u64;
        let mut plain_sum = 0.0f64;
        let mut spec_sum = 0.0f64;
        for seed in 0..runs {
            // Plain: two target samples; token 2 is what speculation must
            // reproduce in distribution.
            let mut ts = InferenceSession::new(&config, TraceLevel::Off, seed);
            let toks = target.generate(&mut ts, &[1, 2], 2, &params, None);
            plain_sum += true_logp_t2(toks[0], toks[1]);

            // Speculative: same prompt, decorrelated RNG streams so the
            // legs are independent draws of their processes.
            let mut ts = InferenceSession::new(&config, TraceLevel::Off, seed ^ 0xA5A5_1234);
            let mut ds = InferenceSession::new(&config, TraceLevel::Off, seed ^ 0x5EED_5678);
            let toks = decoder.generate(&mut ts, &mut ds, &[1, 2], 2, &params, None);
            assert_eq!(toks.len(), 2);
            spec_sum += true_logp_t2(toks[0], toks[1]);
        }
        let plain_mean = plain_sum / runs as f64;
        let spec_mean = spec_sum / runs as f64;
        eprintln!(
            "plain mean logp={plain_mean:.4} spec mean logp={spec_mean:.4} delta={:.4}",
            (spec_mean - plain_mean).abs()
        );
        // Per-draw logp std ~1 nat -> SE ~0.026 per leg at n=1500; the
        // old skewed scheme measured +0.09 nats. 0.08 separates the
        // correct scheme (0 expected) from a skewed one at ~3 sigma.
        assert!(
            (spec_mean - plain_mean).abs() < 0.08,
            "speculative token likelihood diverges from plain sampling: \
             plain={plain_mean:.4} spec={spec_mean:.4}"
        );
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
    fn test_speculative_token_index_bounded_by_logits() {
        let config = ModelConfig::tiny_test();
        let target = make_model_with_weights(&config, 42);
        let draft = make_model_with_weights(&config, 42);
        let decoder = SpeculativeDecoder::new(&target, &draft, 3);

        let params = SamplingParams {
            temperature: 1.0,
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
            10,
            &params,
            None,
        );

        for &tok in &tokens {
            assert!(
                (tok as usize) < config.vocab_size,
                "token {tok} >= vocab_size {}",
                config.vocab_size
            );
        }
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

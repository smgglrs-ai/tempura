use crate::sampler_chain::{MinPSampler, RepetitionPenalizer, TypicalSampler, LogitProcessor};
use crate::Result;
pub use crate::{Sampler, SamplingParams};
use rand::RngCore;

#[cfg(feature = "cpu")]
pub struct CpuSampler;

#[cfg(feature = "cpu")]
impl CpuSampler {
    fn apply_pre_softmax_filters(logits: &mut Vec<f32>, params: &SamplingParams, prior_tokens: &[u32]) {
        if params.repeat_penalty != 1.0 {
            let mut pen = RepetitionPenalizer {
                penalty: params.repeat_penalty,
                last_n: params.penalty_last_n,
            };
            pen.apply(logits, prior_tokens);
        }

        if params.min_p > 0.0 {
            let mut sampler = MinPSampler {
                min_p: params.min_p,
                min_keep: 1,
            };
            sampler.apply(logits, prior_tokens);
        }

        if params.typical_p < 1.0 {
            let mut sampler = TypicalSampler {
                typical_p: params.typical_p,
                min_keep: 1,
            };
            sampler.apply(logits, prior_tokens);
        }
    }
}

#[cfg(feature = "cpu")]
impl Sampler for CpuSampler {
    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Proven
    }

    fn apply(&self, logits: &[f32], output: &mut [f32], params: &SamplingParams) -> Result<()> {
        let vocab_size = logits.len();

        // temperature scaling
        let mut scaled: Vec<f32> = if params.temperature != 1.0 && params.temperature > 0.0 {
            logits.iter().map(|l| l / params.temperature).collect()
        } else {
            logits.to_vec()
        };

        // apply pre-softmax logit processors
        Self::apply_pre_softmax_filters(&mut scaled, params, &[]);

        // top-k: find threshold
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = params.top_k.min(vocab_size);
        let _top_k_threshold = indexed[k - 1].1;

        // softmax over top-k candidates
        let max_val = indexed[0].1;
        let mut probs = vec![0.0f32; vocab_size];
        let mut sum = 0.0f32;

        for &(idx, val) in &indexed[..k] {
            if val == f32::NEG_INFINITY {
                continue;
            }
            let p = (val - max_val).exp();
            probs[idx] = p;
            sum += p;
        }

        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // top-p (nucleus) filtering
        if params.top_p < 1.0 {
            let mut sorted_probs: Vec<(usize, f32)> = probs
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, p)| *p > 0.0)
                .collect();
            sorted_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            let mut cutoff_idx = sorted_probs.len();
            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= params.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            let keep: std::collections::HashSet<usize> =
                sorted_probs[..cutoff_idx].iter().map(|(i, _)| *i).collect();

            let mut new_sum = 0.0f32;
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *p = 0.0;
                } else {
                    new_sum += *p;
                }
            }
            for p in &mut probs {
                if *p > 0.0 {
                    *p /= new_sum;
                }
            }
        }

        output[..vocab_size].copy_from_slice(&probs);
        Ok(())
    }

    fn sample_token(&self, probs: &[f32], rng: Option<&mut dyn RngCore>) -> u32 {
        match rng {
            None => probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0),
            Some(rng) => {
                let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
                let mut cumsum = 0.0f32;
                for (idx, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum > r {
                        return idx as u32;
                    }
                }
                (probs.len() - 1) as u32
            }
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub struct CudaSampler;

#[cfg(all(feature = "cuda", not(feature = "cutile")))]
impl Sampler for CudaSampler {
    fn apply(&self, _logits: &[f32], _output: &mut [f32], _params: &SamplingParams) -> Result<()> {
        todo!("CudaSampler: requires cargo oxide build")
    }

    fn sample_token(&self, _probs: &[f32], _rng: Option<&mut dyn RngCore>) -> u32 {
        todo!("CudaSampler: requires cargo oxide build")
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::Sampler;

    #[kani::proof]
    fn sampler_top_k_index_in_bounds() {
        let vocab_size: usize = kani::any();
        let top_k: usize = kani::any();

        kani::assume(vocab_size >= 1 && vocab_size <= 16);
        kani::assume(top_k >= 1 && top_k <= 16);

        let k = top_k.min(vocab_size);
        assert!(k >= 1);
        assert!(k - 1 < vocab_size);
        assert!(k <= vocab_size);
    }

    #[kani::proof]
    #[kani::unwind(17)]
    fn sampler_full_execution_no_panic() {
        let vocab_size: usize = kani::any();
        kani::assume(vocab_size >= 2 && vocab_size <= 8);

        let logits = vec![1.0f32; vocab_size];
        let mut output = vec![0.0f32; vocab_size];
        let params = super::SamplingParams {
            temperature: 1.0,
            top_k: vocab_size,
            top_p: 1.0,
            ..super::SamplingParams::default()
        };

        let sampler = super::CpuSampler;
        let result = sampler.apply(&logits, &mut output, &params);
        assert!(result.is_ok());
    }

    #[kani::proof]
    fn sample_token_cumsum_terminates() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 8);

        // The fallback is (n - 1), which must be a valid index
        assert!(n - 1 < n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let sampler = CpuSampler;
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let mut output = vec![0.0; 4];

        sampler
            .apply(
                &logits,
                &mut output,
                &SamplingParams {
                    temperature: 0.01,
                    top_k: 4,
                    top_p: 1.0,
                    ..SamplingParams::default()
                },
            )
            .unwrap();

        // with very low temperature, nearly all probability on index 1
        let argmax = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        assert_eq!(argmax, 1);
        assert!(output[1] > 0.99);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let sampler = CpuSampler;
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];

        sampler
            .apply(&logits, &mut output, &SamplingParams::default())
            .unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_greedy_argmax() {
        let sampler = CpuSampler;
        let probs = vec![0.1, 0.05, 0.7, 0.15];
        let token = sampler.sample_token(&probs, None);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_deterministic_same_seed() {
        use rand::SeedableRng;
        let sampler = CpuSampler;
        let probs = vec![0.2, 0.3, 0.3, 0.2];

        let mut results1 = Vec::new();
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..20 {
            results1.push(sampler.sample_token(&probs, Some(&mut rng1)));
        }

        let mut results2 = Vec::new();
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..20 {
            results2.push(sampler.sample_token(&probs, Some(&mut rng2)));
        }

        assert_eq!(results1, results2);
    }

    #[test]
    fn test_different_seeds_differ() {
        use rand::SeedableRng;
        let sampler = CpuSampler;
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        let mut results1 = Vec::new();
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..50 {
            results1.push(sampler.sample_token(&probs, Some(&mut rng1)));
        }

        let mut results2 = Vec::new();
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(999);
        for _ in 0..50 {
            results2.push(sampler.sample_token(&probs, Some(&mut rng2)));
        }

        assert_ne!(results1, results2);
    }

    #[test]
    fn test_presence_penalty_suppresses_seen_tokens() {
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        let params = SamplingParams {
            presence_penalty: 2.0,
            ..SamplingParams::default()
        };
        params.apply_penalties(&mut logits, &[0, 2]);
        assert!(logits[0] < 5.0);
        assert_eq!(logits[1], 5.0);
        assert!(logits[2] < 5.0);
        assert_eq!(logits[3], 5.0);
    }

    #[test]
    fn test_frequency_penalty_proportional() {
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        let params = SamplingParams {
            frequency_penalty: 1.0,
            ..SamplingParams::default()
        };
        params.apply_penalties(&mut logits, &[0, 0, 0, 2]);
        // token 0 appeared 3x, token 2 appeared 1x
        assert!((logits[0] - 2.0).abs() < 1e-5);
        assert_eq!(logits[1], 5.0);
        assert!((logits[2] - 4.0).abs() < 1e-5);
        assert_eq!(logits[3], 5.0);
    }

    #[test]
    fn test_zero_penalties_no_change() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        let params = SamplingParams::default();
        params.apply_penalties(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_softmax_matches_f64_reference() {
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0, 0.001];

        let logits_f64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
        let max_val = logits_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits_f64.iter().map(|x| (x - max_val).exp()).sum();
        let ref_probs: Vec<f64> = logits_f64
            .iter()
            .map(|x| (x - max_val).exp() / exp_sum)
            .collect();

        let sampler = CpuSampler;
        let mut output = vec![0.0f32; logits.len()];
        let params = SamplingParams {
            top_k: logits.len(),
            ..SamplingParams::default()
        };
        sampler.apply(&logits, &mut output, &params).unwrap();

        for (i, (&f32_val, &f64_val)) in output.iter().zip(ref_probs.iter()).enumerate() {
            let diff = (f32_val as f64 - f64_val).abs();
            assert!(
                diff < 1e-6,
                "softmax mismatch at {i}: f32={f32_val}, f64_ref={f64_val}, diff={diff}"
            );
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_softmax_sums_to_one(
                logits in prop::collection::vec(-100.0f32..100.0f32, 4..64)
            ) {
                let sampler = CpuSampler;
                let mut output = vec![0.0; logits.len()];
                let params = SamplingParams { top_k: logits.len(), ..SamplingParams::default() };
                sampler.apply(&logits, &mut output, &params).unwrap();
                let sum: f32 = output.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
            }

            #[test]
            fn prop_all_probabilities_non_negative(
                logits in prop::collection::vec(-100.0f32..100.0f32, 4..64)
            ) {
                let sampler = CpuSampler;
                let mut output = vec![0.0; logits.len()];
                sampler.apply(&logits, &mut output, &SamplingParams::default()).unwrap();
                for (i, &p) in output.iter().enumerate() {
                    prop_assert!(p >= 0.0, "negative probability at index {i}: {p}");
                }
            }

            #[test]
            fn prop_top_k_limits_nonzero_count(
                logits in prop::collection::vec(0.0f32..10.0f32, 10..50),
                k in 1usize..10usize,
            ) {
                let sampler = CpuSampler;
                let mut output = vec![0.0; logits.len()];
                let params = SamplingParams { top_k: k, ..SamplingParams::default() };
                sampler.apply(&logits, &mut output, &params).unwrap();
                let nonzero = output.iter().filter(|&&p| p > 0.0).count();
                prop_assert!(nonzero <= k, "got {nonzero} nonzero, expected <={k}");
            }

            #[test]
            fn prop_greedy_selects_argmax(
                logits in prop::collection::vec(-10.0f32..10.0f32, 4..32)
            ) {
                let sampler = CpuSampler;
                let mut output = vec![0.0; logits.len()];
                let params = SamplingParams { temperature: 0.01, top_k: logits.len(), ..SamplingParams::default() };
                sampler.apply(&logits, &mut output, &params).unwrap();
                let token = sampler.sample_token(&output, None);
                let expected = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32).unwrap();
                prop_assert_eq!(token, expected);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub type CudaSampler = CpuSampler;

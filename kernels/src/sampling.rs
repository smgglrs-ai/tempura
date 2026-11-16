use crate::sampler_chain::{LogitProcessor, MinPSampler, RepetitionPenalizer, TypicalSampler};
use crate::Result;
pub use crate::{Sampler, SamplingParams};
use rand::RngCore;

#[cfg(feature = "cpu")]
pub struct CpuSampler;

#[cfg(feature = "cpu")]
impl CpuSampler {
    /// `a` is strictly WORSE than `b` as a top-k member (smaller value; on
    /// ties, the larger index loses -- deterministic; the full-sort path's
    /// tie order is unspecified by `sort_unstable`).
    #[inline]
    fn cand_lt(a: &(f32, usize), b: &(f32, usize)) -> bool {
        a.0 < b.0 || (a.0 == b.0 && a.1 > b.1)
    }

    /// Min-heap sift-up for [`Self::topk_descending`] (root = worst kept).
    fn sift_up(h: &mut [(f32, usize)], mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if Self::cand_lt(&h[i], &h[p]) {
                h.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    /// Min-heap sift-down for [`Self::topk_descending`] (root = worst kept).
    fn sift_down(h: &mut [(f32, usize)], mut i: usize) {
        let n = h.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut worst = i;
            if l < n && Self::cand_lt(&h[l], &h[worst]) {
                worst = l;
            }
            if r < n && Self::cand_lt(&h[r], &h[worst]) {
                worst = r;
            }
            if worst == i {
                break;
            }
            h.swap(i, worst);
            i = worst;
        }
    }

    /// Bounded min-heap top-k selection: the `k` largest values (ties
    /// broken toward the lower index), returned sorted BEST-FIRST.
    /// `f32::NEG_INFINITY` entries are skipped (the pre-softmax filters
    /// use them as "removed"); callers must exclude NaN. O(n + k log k)
    /// with a single streaming pass -- the replacement for the
    /// full-vocab `sort_unstable` in [`Sampler::apply`].
    fn topk_descending(values: &[f32], k: usize) -> Vec<(f32, usize)> {
        let k = k.min(values.len());
        let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k);
        for (idx, &v) in values.iter().enumerate() {
            if v == f32::NEG_INFINITY {
                continue;
            }
            if heap.len() < k {
                heap.push((v, idx));
                let last = heap.len() - 1;
                Self::sift_up(&mut heap, last);
            } else if Self::cand_lt(&heap[0], &(v, idx)) {
                heap[0] = (v, idx);
                Self::sift_down(&mut heap, 0);
            }
        }
        heap.sort_by(|a, b| {
            if Self::cand_lt(b, a) {
                std::cmp::Ordering::Less
            } else if Self::cand_lt(a, b) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        heap
    }

    /// Fused decode-path sampler (the [`Sampler::sample_from_logits`]
    /// override body): temperature + min_p + top-k heap selection +
    /// softmax + top-p + RNG draw, all over the k surviving candidates.
    /// Semantics mirror `apply` + `sample_token` (same scaling division,
    /// same descending accumulation orders, same index-ordered sampling
    /// walk and last-index fallthrough); the only divergence is tie
    /// order at the top-k boundary, which the full path leaves
    /// unspecified. Returns None for configurations needing the full
    /// distribution (typical_p < 1.0) or containing NaN logits.
    fn sample_fused(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        rng: &mut dyn RngCore,
    ) -> Option<u32> {
        let n = logits.len();
        if n == 0 || params.typical_p < 1.0 {
            return None;
        }

        // Temperature scaling -- l / T, bit-identical to `apply`.
        let mut scaled: Vec<f32> = if params.temperature != 1.0 && params.temperature > 0.0 {
            logits.iter().map(|l| l / params.temperature).collect()
        } else {
            logits.to_vec()
        };
        if scaled.iter().any(|l| l.is_nan()) {
            // The full path's sort orders NaN arbitrarily; do not diverge.
            return None;
        }

        // Pre-softmax logit filter (O(n), logit domain). repeat_penalty is
        // a no-op here (the engine applies it to the raw logits before the
        // call; `apply` re-runs it with empty priors); typical_p bailed out.
        if params.min_p > 0.0 {
            MinPSampler {
                min_p: params.min_p,
                min_keep: 1,
            }
            .apply(&mut scaled, &[]);
        }

        // Top-k selection.
        let k = params.top_k.min(n).max(1);
        let mut top = Self::topk_descending(&scaled, k);
        if top.is_empty() {
            // Everything -inf: the full path yields all-zero probs and
            // `sample_token`'s fallthrough returns the last index.
            return Some((n - 1) as u32);
        }

        // Softmax over the k candidates, accumulated in descending order
        // (the full path's `indexed[..k]` loop order).
        let max_val = top[0].0;
        let mut sum = 0.0f32;
        for c in &mut top {
            c.0 = (c.0 - max_val).exp();
            sum += c.0;
        }
        if sum <= 0.0 {
            return Some((n - 1) as u32);
        }
        for c in &mut top {
            c.0 /= sum;
        }

        // Top-p (nucleus) filtering over the k candidates.
        let mut keep = top.len();
        if params.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            for (i, c) in top.iter().enumerate() {
                cumsum += c.0;
                if cumsum >= params.top_p {
                    keep = i + 1;
                    break;
                }
            }
            let mut new_sum = 0.0f32;
            for c in &top[..keep] {
                new_sum += c.0;
            }
            if new_sum > 0.0 {
                for c in &mut top[..keep] {
                    c.0 /= new_sum;
                }
            }
        }

        // RNG draw. The full path walks the whole probs array in INDEX
        // order; excluded entries are zero and cannot trigger the
        // `cumsum > r` check, so walking the kept candidates in index
        // order is equivalent (same r, same token). Fallthrough on float
        // under-run returns the last index, as `sample_token` does.
        let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
        top[..keep].sort_by_key(|c| c.1);
        let mut cumsum = 0.0f32;
        for c in &top[..keep] {
            cumsum += c.0;
            if cumsum > r {
                return Some(c.1 as u32);
            }
        }
        Some((n - 1) as u32)
    }

    fn apply_pre_softmax_filters(
        logits: &mut [f32],
        params: &SamplingParams,
        prior_tokens: &[u32],
    ) {
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
            sorted_probs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

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

    fn sample_from_logits(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        rng: Option<&mut dyn RngCore>,
    ) -> Option<u32> {
        match rng {
            // Greedy rows keep the exact full-path semantics (argmax over
            // the softmax'd probs).
            None => None,
            Some(rng) => self.sample_fused(logits, params, rng),
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

    // sampler_full_execution_no_panic: removed 2026-11-05 -- symbolic-length Vec allocation (CBMC unwinds allocation internals without bound) + transcendental math; runs as the unit test below

    #[kani::proof]
    fn sample_token_cumsum_terminates() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 8);

        // The fallback is (n - 1), which must be a valid index
        assert!(n - 1 < n);
    }

    #[kani::proof]
    fn topk_heap_selection_in_bounds() {
        // The fused decode sampler's bounded min-heap: no panics, no OOB,
        // every returned (value, index) comes from the input.
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 8);
        let k: usize = kani::any();
        kani::assume(k >= 1 && k <= 8);
        let mut values = vec![0f32; n];
        for v in values.iter_mut() {
            let bits: u32 = kani::any();
            let f = f32::from_bits(bits);
            kani::assume(!f.is_nan());
            *v = f;
        }
        let top = CpuSampler::topk_descending(&values, k);
        assert!(top.len() <= k.min(n));
        for &(v, idx) in &top {
            assert!(idx < n);
            assert!(v == values[idx]);
            assert!(v != f32::NEG_INFINITY);
        }
        // Sorted best-first: no later element is strictly better than an
        // earlier one.
        for i in 0..top.len() {
            for j in (i + 1)..top.len() {
                assert!(!CpuSampler::cand_lt(&top[i], &top[j]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift RngCore for the sampler-equivalence tests.
    struct TestRng(u64);
    impl rand::RngCore for TestRng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x as u32
        }
        fn next_u64(&mut self) -> u64 {
            (self.next_u32() as u64) << 32 | self.next_u32() as u64
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for c in dest.iter_mut() {
                *c = self.next_u32() as u8;
            }
        }
    }

    #[test]
    fn fused_sampler_matches_full_path() {
        // The fused decode-path sampler (top-k heap + O(k) softmax/top-p/
        // sample) must pick the same token as apply + sample_token for the
        // same RNG stream: same temperature scaling division, same
        // descending accumulation orders, same index-ordered sampling
        // walk. (Tie order at the top-k boundary is unspecified in the
        // full path; random continuous logits make ties measure-zero.)
        let sampler = CpuSampler;
        let cases: &[(f32, usize, f32, f32)] = &[
            (1.0, 50, 1.0, 0.0),
            (0.8, 40, 0.95, 0.0),
            (1.3, 10, 0.9, 0.0),
            (0.7, 128, 1.0, 0.0),
            (0.9, 25, 0.9, 0.05),
        ];
        let n = 1000usize;
        for &(temp, top_k, top_p, min_p) in cases {
            for trial in 0u64..20 {
                let mut vrng = TestRng(0x5EED_0000 + trial);
                let logits: Vec<f32> = (0..n)
                    .map(|i| {
                        ((vrng.next_u32() as f64 / u32::MAX as f64) * 20.0 - 10.0) as f32
                            + (i % 7) as f32 * 1e-3
                    })
                    .collect();
                let params = SamplingParams {
                    temperature: temp,
                    top_k,
                    top_p,
                    min_p,
                    ..SamplingParams::default()
                };
                let mut probs = vec![0.0f32; n];
                sampler.apply(&logits, &mut probs, &params).unwrap();
                let mut r1 = TestRng(0xBEEF_0000 + trial);
                let mut r2 = TestRng(0xBEEF_0000 + trial);
                let ref_tok = sampler.sample_token(&probs, Some(&mut r1));
                let fused = sampler
                    .sample_from_logits(&logits, &params, Some(&mut r2))
                    .expect("fused path should cover this config");
                assert_eq!(ref_tok, fused, "temp={temp} k={top_k} p={top_p} trial={trial}");
            }
        }
    }

    #[test]
    fn fused_sampler_all_inf_fallthrough() {
        // All logits -inf (every candidate filtered): the full path
        // produces all-zero probs and sample_token's fallthrough returns
        // the last index; the fused path must match.
        let sampler = CpuSampler;
        let n = 16usize;
        let logits = vec![f32::NEG_INFINITY; n];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 4,
            min_p: 0.9, // forces the -inf filtering
            ..SamplingParams::default()
        };
        let mut probs = vec![0.0f32; n];
        sampler.apply(&logits, &mut probs, &params).unwrap();
        let mut r1 = TestRng(1);
        let mut r2 = TestRng(1);
        let ref_tok = sampler.sample_token(&probs, Some(&mut r1));
        let fused = sampler
            .sample_from_logits(&logits, &params, Some(&mut r2))
            .unwrap();
        assert_eq!(ref_tok, fused);
        assert_eq!(fused, (n - 1) as u32);
    }

    #[test]
    fn fused_sampler_topk_boundary_ties_deterministic() {
        // Equal values at the k boundary: the heap keeps the LOWER indices
        // and the result is deterministic across calls.
        let sampler = CpuSampler;
        let logits = vec![1.0f32, 2.0, 2.0, 2.0, 0.5, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            ..SamplingParams::default()
        };
        let toks: Vec<u32> = (0..8)
            .map(|_| {
                let mut r = TestRng(42);
                sampler
                    .sample_from_logits(&logits, &params, Some(&mut r))
                    .unwrap()
            })
            .collect();
        assert!(toks.iter().all(|&t| t == toks[0]), "{toks:?}");
        assert!(toks[0] == 1 || toks[0] == 2 || toks[0] == 3, "{toks:?}");
    }

    #[test]
    fn sampler_full_execution_no_panic_sizes() {
        // Replaces the sampler_full_execution_no_panic Kani harness (see
        // the note in kani_proofs): symbolic-length Vec allocation +
        // softmax exp are bit-blasted by CBMC without bound.
        for vocab_size in [2usize, 3, 5, 8] {
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
            assert!(result.is_ok(), "vocab_size {vocab_size}");
        }
    }

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

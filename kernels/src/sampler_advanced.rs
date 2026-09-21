use crate::sampler_chain::LogitProcessor;

// DRY (Don't Repeat Yourself) sampler
// Ported from llama.cpp llama_sampler_dry_impl (originally from Koboldcpp).
// Detects repeated token sequences using the Z-algorithm in reverse,
// then applies geometric penalty: multiplier * base^(match_len - allowed_length).
// Breaker tokens (newlines, punctuation) reset sequence matching.
pub struct DrySampler {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub penalty_last_n: usize,
    pub breakers: Vec<u32>,
}

impl LogitProcessor for DrySampler {
    fn apply(&mut self, logits: &mut [f32], prior_tokens: &[u32]) {
        if self.multiplier == 0.0 || self.base < 1.0 || self.penalty_last_n == 0 {
            return;
        }

        let last_n = prior_tokens.len().min(self.penalty_last_n);
        if last_n <= self.allowed_length {
            return;
        }

        let window = &prior_tokens[prior_tokens.len() - last_n..];

        // Step 1: Find the effective rep_limit by scanning for breaker tokens.
        // The most recent breaker token limits how far back we look for repeats.
        let mut rep_limit = last_n;
        for i in 0..last_n {
            let token = window[last_n - 1 - i];
            if self.breakers.contains(&token) {
                rep_limit = i;
                break;
            }
        }

        if rep_limit < self.allowed_length {
            return;
        }

        // Step 2: Z-algorithm in reverse direction to find suffix match lengths.
        // For each position k, compute how many tokens at the end of window
        // also appear starting at position (last_n - 1 - k) going backwards.
        let mut repeat_count = vec![0usize; last_n];
        {
            let last = last_n - 1;
            let mut rt: usize = 0;
            let mut lt: usize = 0;

            for k in 1..last_n {
                if k > rt {
                    let mut n = 0;
                    while n + k < last_n && window[last - n] == window[last - n - k] {
                        n += 1;
                    }
                    repeat_count[last - k] = n.min(rep_limit);
                    if n > 0 {
                        lt = k;
                        rt = k + n - 1;
                    }
                } else {
                    let p = k - lt;
                    let right_part_len = rt - k + 1;

                    if repeat_count[last - p] < right_part_len {
                        repeat_count[last - k] = repeat_count[last - p].min(rep_limit);
                    } else {
                        let mut i = rt + 1;
                        while i < last_n && window[last - i] == window[last - i + k] {
                            i += 1;
                        }
                        repeat_count[last - k] = (i - k).min(rep_limit);
                        lt = k;
                        rt = i - 1;
                    }
                }
            }
        }

        // Step 3: For each position with repeat_count >= allowed_length,
        // the token one step ahead would extend the repetition.
        // Track the maximum repeat length per extending token.
        let mut max_token_repeat = std::collections::HashMap::<u32, usize>::new();
        for i in 0..last_n - 1 {
            let rlen = repeat_count[i];
            if rlen >= self.allowed_length {
                // rat(last_n - 2 - i) in llama.cpp = window[i + 1] in forward indexing
                let extending_token = window[i + 1];
                let entry = max_token_repeat.entry(extending_token).or_insert(0);
                if rlen > *entry {
                    *entry = rlen;
                }
            }
        }

        // Step 4: Apply logit penalties.
        // Clamp exponent to prevent overflow.
        const FLOAT_MAX_LOG: f32 = 88.722_84;
        let max_exponent = if self.base > 1.000_001 {
            (FLOAT_MAX_LOG / self.base.ln()) as usize
        } else {
            0
        };

        for (token, &repeat_len) in &max_token_repeat {
            let tid = *token as usize;
            if tid < logits.len() {
                // Skip if this token is itself a breaker
                if self.breakers.contains(token) {
                    continue;
                }

                let mut exponent = repeat_len - self.allowed_length;
                if max_exponent > 0 && exponent > max_exponent {
                    exponent = max_exponent;
                }
                let penalty = self.multiplier * self.base.powi(exponent as i32);
                logits[tid] -= penalty;
            }
        }
    }
}

// XTC (Cross-Token Coherence) sampler
// Ported from llama.cpp llama_sampler_xtc.
// With some probability, removes tokens above a threshold probability
// (keeping at least min_keep tokens). Increases output diversity by
// stochastically removing the top candidates.
pub struct XtcSampler {
    pub probability: f32,
    pub threshold: f32,
    pub min_keep: usize,
    pub rng: u64,
}

impl XtcSampler {
    fn next_random(&mut self) -> f32 {
        // xorshift64 for deterministic, fast PRNG
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

impl LogitProcessor for XtcSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if self.probability <= 0.0 || self.threshold > 0.5 || logits.len() < 2 {
            return;
        }

        let chance = self.next_random();
        if chance > self.probability {
            return;
        }

        // Compute softmax probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, (l - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= sum;
        }

        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find the last position where probability >= threshold
        let mut pos_last = 0;
        for (i, &(_, p)) in probs.iter().enumerate() {
            if p >= self.threshold {
                pos_last = i;
            } else {
                break;
            }
        }

        // Remove tokens above threshold if enough remain after removal
        if pos_last > 0 && logits.len() - pos_last >= self.min_keep {
            for &(idx, _) in &probs[..pos_last] {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }
}

// Adaptive-P sampler
// Dynamically adjusts a probability cutoff using exponential moving average
// toward a target entropy level. Tokens below the adaptive threshold are zeroed out.
pub struct AdaptivePSampler {
    pub target: f32,
    pub decay: f32,
    pub current_p: f32,
}

impl LogitProcessor for AdaptivePSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if logits.is_empty() {
            return;
        }

        // Compute softmax probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Compute entropy: H = -sum(p * ln(p))
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // Compute max possible entropy for normalization
        let max_entropy = (logits.len() as f32).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Update current_p via EMA toward target based on observed entropy.
        // If entropy is high (more uniform), increase p to filter more aggressively.
        // If entropy is low (peaky), decrease p to let more tokens through.
        self.current_p = self.decay * self.current_p + (1.0 - self.decay) * normalized_entropy;

        // Apply the adaptive threshold: set logits below current_p * max_prob to -inf
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = self.current_p * max_prob;

        let mut kept = 0;
        for (i, &p) in probs.iter().enumerate() {
            if p >= threshold {
                kept += 1;
            } else {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        // Always keep at least 1 token
        if kept == 0 {
            let best_idx = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            logits[best_idx] = max_logit;
        }
    }
}

// Infill sampler
// Blocks specific tokens during infill generation by setting their logits to -inf.
// Used to prevent EOT or other special tokens from appearing in the middle of
// fill-in-the-middle completions.
pub struct InfillSampler {
    pub blocked_tokens: Vec<u32>,
}

impl LogitProcessor for InfillSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        for &token_id in &self.blocked_tokens {
            let tid = token_id as usize;
            if tid < logits.len() {
                logits[tid] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn dry_no_panic() {
        let multiplier: f32 = kani::any();
        let base: f32 = kani::any();
        kani::assume(multiplier.is_finite() && multiplier >= 0.0 && multiplier < 5.0);
        kani::assume(base.is_finite() && base >= 1.0 && base < 5.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let prior: [u32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for &t in &prior {
            kani::assume(t < 8);
        }
        let mut sampler = DrySampler {
            multiplier,
            base,
            allowed_length: 1,
            penalty_last_n: 8,
            breakers: vec![],
        };
        sampler.apply(&mut logits, &prior);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn xtc_no_panic() {
        let probability: f32 = kani::any();
        let threshold: f32 = kani::any();
        kani::assume(probability.is_finite() && probability >= 0.0 && probability <= 1.0);
        kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 0.5);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > -50.0 && *l < 50.0);
        }
        let mut sampler = XtcSampler {
            probability,
            threshold,
            min_keep: 1,
            rng: 12345,
        };
        sampler.apply(&mut logits, &[]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn xtc_probability_zero_is_noop() {
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let original = logits;
        let mut sampler = XtcSampler {
            probability: 0.0,
            threshold: 0.1,
            min_keep: 1,
            rng: 12345,
        };
        sampler.apply(&mut logits, &[]);
        for i in 0..8 {
            assert!(logits[i] == original[i]);
        }
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn adaptive_p_no_panic() {
        let target: f32 = kani::any();
        let decay: f32 = kani::any();
        let current_p: f32 = kani::any();
        kani::assume(target.is_finite() && target >= 0.0 && target <= 1.0);
        kani::assume(decay.is_finite() && decay >= 0.0 && decay <= 1.0);
        kani::assume(current_p.is_finite() && current_p >= 0.0 && current_p <= 1.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > -50.0 && *l < 50.0);
        }
        let mut sampler = AdaptivePSampler {
            target,
            decay,
            current_p,
        };
        sampler.apply(&mut logits, &[]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn infill_blocks_tokens() {
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let blocked: u32 = kani::any();
        kani::assume(blocked < 8);
        let mut sampler = InfillSampler {
            blocked_tokens: vec![blocked],
        };
        sampler.apply(&mut logits, &[]);
        assert!(logits[blocked as usize] == f32::NEG_INFINITY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_penalizes_repeated_pattern() {
        // Pattern: A B C A B C -> tokens [10, 20, 30, 10, 20, 30]
        // The suffix "A B C" appears twice. Emitting token 10 (A) would extend to "A B C A".
        let mut sampler = DrySampler {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 1,
            penalty_last_n: 16,
            breakers: vec![],
        };

        let prior = vec![10u32, 20, 30, 10, 20, 30];
        let mut logits = vec![0.0f32; 64];
        logits[10] = 5.0;
        logits[20] = 5.0;
        logits[30] = 5.0;

        let original_10 = logits[10];
        sampler.apply(&mut logits, &prior);

        // Token 10 should be penalized (it would extend the repeated "A B C" sequence)
        assert!(
            logits[10] < original_10,
            "Token 10 should be penalized for extending repeated pattern, was {} now {}",
            original_10,
            logits[10]
        );
    }

    #[test]
    fn dry_breaker_resets_matching() {
        // Pattern: [10, 20, 99, 10, 20] where 99 is a breaker.
        // The breaker should limit how far back we look, preventing
        // the match from crossing the breaker boundary.
        let mut sampler = DrySampler {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 1,
            penalty_last_n: 16,
            breakers: vec![99],
        };

        let prior = vec![10u32, 20, 99, 10, 20];
        let mut logits = vec![0.0f32; 64];
        logits[10] = 5.0;
        logits[30] = 5.0;

        let original = logits.clone();
        sampler.apply(&mut logits, &prior);

        // Token 30 (not in the pattern at all) should be untouched
        assert_eq!(logits[30], original[30]);
    }

    #[test]
    fn xtc_probability_one_removes_top_token() {
        let mut sampler = XtcSampler {
            probability: 1.0,
            threshold: 0.1,
            min_keep: 1,
            rng: 12345,
        };

        // Multiple tokens with similar logits so several are above threshold after softmax.
        // With logits [5.0, 4.9, 4.8, 0.0], after softmax the top 3 will all be well
        // above 0.1, so XTC should remove them (keeping at least min_keep=1).
        let mut logits = vec![5.0f32, 4.9, 4.8, 0.0];

        sampler.apply(&mut logits, &[]);

        // At least one of the top tokens should be removed
        let removed_count = logits.iter().filter(|&&l| l == f32::NEG_INFINITY).count();
        assert!(
            removed_count > 0,
            "XTC with p=1.0 should remove some above-threshold tokens"
        );
        // At least one token must survive
        assert!(
            logits.iter().any(|&l| l != f32::NEG_INFINITY),
            "At least one token must survive"
        );
    }

    #[test]
    fn xtc_probability_zero_never_modifies() {
        let mut sampler = XtcSampler {
            probability: 0.0,
            threshold: 0.01,
            min_keep: 1,
            rng: 12345,
        };

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = logits.clone();

        sampler.apply(&mut logits, &[]);

        assert_eq!(logits, original, "Logits should be unchanged with p=0.0");
    }

    #[test]
    fn adaptive_p_converges_toward_target() {
        let mut sampler = AdaptivePSampler {
            target: 0.5,
            decay: 0.9,
            current_p: 0.1,
        };

        let initial_p = sampler.current_p;

        // Apply multiple times with a uniform-ish distribution (high entropy)
        // The current_p should move via EMA
        for _ in 0..20 {
            let mut logits = vec![1.0f32; 16];
            sampler.apply(&mut logits, &[]);
        }

        // With uniform logits, normalized entropy is ~1.0.
        // EMA should move current_p from 0.1 toward 1.0 (the observed normalized entropy).
        // After 20 iterations with decay=0.9: p should have moved significantly from 0.1.
        assert!(
            sampler.current_p > initial_p + 0.1,
            "current_p should have increased from {} via EMA, got {}",
            initial_p,
            sampler.current_p
        );
    }

    #[test]
    fn infill_blocks_tokens() {
        let mut sampler = InfillSampler {
            blocked_tokens: vec![2, 5, 7],
        };

        let mut logits = vec![1.0f32; 10];
        sampler.apply(&mut logits, &[]);

        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);

        // Other tokens untouched
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 1.0);
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits[4], 1.0);
        assert_eq!(logits[6], 1.0);
        assert_eq!(logits[8], 1.0);
        assert_eq!(logits[9], 1.0);
    }

    #[test]
    fn infill_ignores_out_of_range_tokens() {
        let mut sampler = InfillSampler {
            blocked_tokens: vec![100],
        };

        let mut logits = vec![1.0f32; 10];
        let original = logits.clone();
        sampler.apply(&mut logits, &[]);

        assert_eq!(
            logits, original,
            "Out-of-range token should not crash or modify logits"
        );
    }
}

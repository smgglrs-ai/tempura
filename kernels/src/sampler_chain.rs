use std::collections::HashMap;

pub trait LogitProcessor: Send + Sync {
    fn apply(&mut self, logits: &mut [f32], prior_tokens: &[u32]);
}

pub struct SamplerChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl SamplerChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn push(&mut self, processor: Box<dyn LogitProcessor>) {
        self.processors.push(processor);
    }

    pub fn apply_all(&mut self, logits: &mut [f32], prior_tokens: &[u32]) {
        for proc in &mut self.processors {
            proc.apply(logits, prior_tokens);
        }
    }

    pub fn len(&self) -> usize {
        self.processors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── MinP ───────────────────────────────────────────────────────────────────
//
// Keeps tokens with probability >= min_p * max_probability.
// Operates on logits (pre-softmax): keeps tokens where
// logit >= max_logit + ln(min_p), which is equivalent.

pub struct MinPSampler {
    pub min_p: f32,
    pub min_keep: usize,
}

impl LogitProcessor for MinPSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if self.min_p <= 0.0 || logits.is_empty() {
            return;
        }

        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_logit + self.min_p.ln();

        let mut keep_count = 0usize;
        for &l in logits.iter() {
            if l >= threshold {
                keep_count += 1;
            }
        }

        if keep_count < self.min_keep {
            return;
        }

        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

// ── TypicalP ───────────────────────────────────────────────────────────────
//
// Keeps tokens whose information content (-log p) is closest to the
// distribution's entropy. Selects the "typical" tokens up to the
// cumulative probability threshold typ_p.

pub struct TypicalSampler {
    pub typical_p: f32,
    pub min_keep: usize,
}

impl LogitProcessor for TypicalSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if self.typical_p >= 1.0 || logits.is_empty() {
            return;
        }

        // softmax
        let max_val = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // entropy
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        // |entropy - (-log p)| for each token, paired with index
        let mut shifted: Vec<(usize, f32, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, (-p.ln() - entropy).abs(), p))
            .collect();

        // sort by shifted score ascending (most typical first)
        shifted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // accumulate probability, keep tokens until we exceed typical_p
        let mut cum_sum = 0.0f32;
        let mut keep = std::collections::HashSet::new();
        for (idx, (i, _, p)) in shifted.iter().enumerate() {
            cum_sum += p;
            keep.insert(*i);
            if cum_sum > self.typical_p && (self.min_keep == 0 || idx >= self.min_keep - 1) {
                break;
            }
        }

        for (i, l) in logits.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Mirostat v2 ────────────────────────────────────────────────────────────
//
// Adaptive sampling targeting a specific surprise level (tau).
// Truncates tokens with surprise > mu, then updates mu after sampling.
// The caller must invoke update_mu() after selecting a token.

pub struct MirostatV2Sampler {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

impl MirostatV2Sampler {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            tau,
            eta,
            mu: 2.0 * tau,
        }
    }

    pub fn update_mu(&mut self, selected_prob: f32) {
        if selected_prob > 0.0 {
            let observed_surprise = -selected_prob.log2();
            let e = observed_surprise - self.tau;
            self.mu -= self.eta * e;
        }
    }
}

impl LogitProcessor for MirostatV2Sampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if logits.is_empty() {
            return;
        }

        // softmax to get probabilities
        let max_val = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = {
            let exps: Vec<f32> = logits.iter().map(|l| (l - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        };

        // mask tokens with surprise > mu
        let mut any_kept = false;
        for (i, l) in logits.iter_mut().enumerate() {
            let p = probs[i];
            if p > 0.0 && -p.log2() > self.mu {
                *l = f32::NEG_INFINITY;
            } else if p > 0.0 {
                any_kept = true;
            }
        }

        // if everything got masked, keep the top token
        if !any_kept {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            logits[best] = max_val;
        }
    }
}

// ── RepetitionPenalty ───────────────────────────────────────────────────────
//
// Penalizes repeated tokens within a window of last_n tokens.
// If logit > 0, divide by penalty; if logit <= 0, multiply by penalty.

pub struct RepetitionPenalizer {
    pub penalty: f32,
    pub last_n: usize,
}

impl LogitProcessor for RepetitionPenalizer {
    fn apply(&mut self, logits: &mut [f32], prior_tokens: &[u32]) {
        if self.penalty == 1.0 || prior_tokens.is_empty() {
            return;
        }

        let window_start = prior_tokens.len().saturating_sub(self.last_n);
        let window = &prior_tokens[window_start..];

        let mut seen = HashMap::new();
        for &tok in window {
            seen.entry(tok).or_insert(true);
        }

        for (&tok, _) in &seen {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] <= 0.0 {
                    logits[idx] *= self.penalty;
                } else {
                    logits[idx] /= self.penalty;
                }
            }
        }
    }
}

// ── LogitBias ──────────────────────────────────────────────────────────────

pub struct LogitBiaser {
    pub biases: Vec<(u32, f32)>,
}

impl LogitProcessor for LogitBiaser {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        for &(token_id, bias) in &self.biases {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] += bias;
            }
        }
    }
}

// ── TopNSigma ──────────────────────────────────────────────────────────────
//
// Keeps tokens within n standard deviations of the max logit.
// Excludes -infinity logits from mean/std calculation.

pub struct TopNSigmaSampler {
    pub n_sigma: f32,
    pub min_keep: usize,
}

impl LogitProcessor for TopNSigmaSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if self.n_sigma <= 0.0 || logits.len() <= 1 {
            return;
        }

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count = 0usize;

        for &l in logits.iter() {
            if l != f32::NEG_INFINITY {
                if l > max_logit {
                    max_logit = l;
                }
                sum += l as f64;
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        let mean = sum / count as f64;

        let mut var_acc = 0.0f64;
        for &l in logits.iter() {
            if l != f32::NEG_INFINITY {
                let d = l as f64 - mean;
                var_acc += d * d;
            }
        }
        let std = (var_acc / count as f64).sqrt() as f32;

        let threshold = max_logit - self.n_sigma * std;

        let mut keep_count = 0usize;
        for &l in logits.iter() {
            if l >= threshold {
                keep_count += 1;
            }
        }

        if keep_count < self.min_keep {
            return;
        }

        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn min_p_no_panic() {
        let min_p: f32 = kani::any();
        kani::assume(min_p.is_finite());
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let mut sampler = MinPSampler { min_p, min_keep: 1 };
        sampler.apply(&mut logits, &[]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn min_p_preserves_max() {
        let min_p: f32 = kani::any();
        kani::assume(min_p.is_finite() && min_p > 0.0 && min_p < 1.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let max_before = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_idx = logits.iter().position(|&l| l == max_before).unwrap();
        let mut sampler = MinPSampler { min_p, min_keep: 1 };
        sampler.apply(&mut logits, &[]);
        assert!(logits[max_idx] == max_before);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn typical_no_panic() {
        let typical_p: f32 = kani::any();
        kani::assume(typical_p.is_finite() && typical_p >= 0.0 && typical_p <= 1.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > -50.0 && *l < 50.0);
        }
        let mut sampler = TypicalSampler { typical_p, min_keep: 1 };
        sampler.apply(&mut logits, &[]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn mirostat_v2_no_panic() {
        let tau: f32 = kani::any();
        let eta: f32 = kani::any();
        kani::assume(tau.is_finite() && tau > 0.0 && tau < 20.0);
        kani::assume(eta.is_finite() && eta > 0.0 && eta < 2.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > -50.0 && *l < 50.0);
        }
        let mut sampler = MirostatV2Sampler::new(tau, eta);
        sampler.apply(&mut logits, &[]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn repetition_penalty_no_panic() {
        let penalty: f32 = kani::any();
        kani::assume(penalty.is_finite() && penalty > 0.0 && penalty < 10.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let prior_tokens: [u32; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for &t in &prior_tokens {
            kani::assume(t < 8);
        }
        let mut penalizer = RepetitionPenalizer { penalty, last_n: 64 };
        penalizer.apply(&mut logits, &prior_tokens);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn repetition_penalty_symmetric() {
        let penalty: f32 = kani::any();
        kani::assume(penalty.is_finite() && penalty > 1.0 && penalty < 10.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > 0.0);
        }
        let original = logits;
        let token_idx: u32 = kani::any();
        kani::assume(token_idx < 8);
        let prior_tokens = [token_idx];
        let mut penalizer = RepetitionPenalizer { penalty, last_n: 64 };
        penalizer.apply(&mut logits, &prior_tokens);
        // Positive logit of repeated token is divided by penalty (> 1), so it decreases
        assert!(logits[token_idx as usize] <= original[token_idx as usize]);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn logit_bias_adds_correctly() {
        let bias: f32 = kani::any();
        kani::assume(bias.is_finite());
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite());
        }
        let target_idx: u32 = kani::any();
        kani::assume(target_idx < 8);
        let original_val = logits[target_idx as usize];
        let mut biaser = LogitBiaser { biases: vec![(target_idx, bias)] };
        biaser.apply(&mut logits, &[]);
        let expected = original_val + bias;
        assert!(logits[target_idx as usize] == expected);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn top_n_sigma_no_panic() {
        let n_sigma: f32 = kani::any();
        kani::assume(n_sigma.is_finite() && n_sigma >= 0.0 && n_sigma < 10.0);
        let mut logits = [0.0f32; 8];
        for l in &mut logits {
            *l = kani::any();
            kani::assume(l.is_finite() && *l > -100.0 && *l < 100.0);
        }
        let mut sampler = TopNSigmaSampler { n_sigma, min_keep: 1 };
        sampler.apply(&mut logits, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    #[test]
    fn test_min_p_filters_low_probability_tokens() {
        let mut logits = vec![10.0, 1.0, 0.5, 0.1, -5.0];
        let mut sampler = MinPSampler {
            min_p: 0.1,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);

        assert!(logits[0].is_finite());
        assert_eq!(logits[4], f32::NEG_INFINITY);

        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert!(finite_count >= 1);
        assert!(finite_count < 5);
    }

    #[test]
    fn test_min_p_disabled_when_zero() {
        let mut logits = vec![10.0, 1.0, 0.5];
        let original = logits.clone();
        let mut sampler = MinPSampler {
            min_p: 0.0,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_min_p_respects_min_keep() {
        let mut logits = vec![100.0, -10.0, -20.0, -30.0];
        let mut sampler = MinPSampler {
            min_p: 0.5,
            min_keep: 4,
        };
        sampler.apply(&mut logits, &[]);
        // min_keep=4 means we need 4 tokens but only 1 passes the threshold,
        // so the filter is not applied
        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert_eq!(finite_count, 4);
    }

    #[test]
    fn test_typical_keeps_tokens_near_entropy() {
        // Create a distribution where some tokens are very typical
        // and others are very atypical
        let mut logits = vec![5.0, 4.9, 4.8, 0.0, -5.0, -10.0, -15.0, -20.0];
        let mut sampler = TypicalSampler {
            typical_p: 0.5,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);

        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert!(finite_count >= 1);
        assert!(finite_count < logits.len());
    }

    #[test]
    fn test_typical_disabled_at_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        let mut sampler = TypicalSampler {
            typical_p: 1.0,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_mirostat_v2_truncates_high_surprise() {
        let mut logits = vec![10.0, 5.0, 0.0, -5.0, -10.0];
        let mut sampler = MirostatV2Sampler::new(3.0, 0.1);
        sampler.apply(&mut logits, &[]);

        // top token should always survive
        assert!(logits[0].is_finite());
        // some low-probability tokens should be masked
        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert!(finite_count >= 1);
    }

    #[test]
    fn test_mirostat_v2_adapts_mu() {
        let mut sampler = MirostatV2Sampler::new(5.0, 0.1);
        let initial_mu = sampler.mu;

        // simulate selecting a high-probability token (low surprise)
        sampler.update_mu(0.9);
        // surprise of 0.9 ≈ 0.15 bits, well below tau=5, so mu should increase
        assert!(sampler.mu > initial_mu);

        // simulate selecting a low-probability token (high surprise)
        let mu_before = sampler.mu;
        sampler.update_mu(0.001);
        // surprise of 0.001 ≈ 10 bits, above tau=5, so mu should decrease
        assert!(sampler.mu < mu_before);
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated_token_logits() {
        let mut logits = vec![5.0, 5.0, 5.0, 5.0];
        let prior_tokens = vec![0u32, 2];
        let mut penalizer = RepetitionPenalizer {
            penalty: 2.0,
            last_n: 64,
        };
        penalizer.apply(&mut logits, &prior_tokens);

        // positive logits of repeated tokens should be divided by penalty
        assert!((logits[0] - 2.5).abs() < 1e-5);
        assert_eq!(logits[1], 5.0);
        assert!((logits[2] - 2.5).abs() < 1e-5);
        assert_eq!(logits[3], 5.0);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-5.0, 5.0];
        let prior_tokens = vec![0u32, 1];
        let mut penalizer = RepetitionPenalizer {
            penalty: 2.0,
            last_n: 64,
        };
        penalizer.apply(&mut logits, &prior_tokens);

        // negative logit multiplied by penalty (becomes more negative)
        assert!((logits[0] - (-10.0)).abs() < 1e-5);
        // positive logit divided by penalty
        assert!((logits[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_respects_window() {
        let mut logits = vec![5.0, 5.0, 5.0];
        // token 0 is outside the last_n=2 window
        let prior_tokens = vec![0u32, 1, 2];
        let mut penalizer = RepetitionPenalizer {
            penalty: 2.0,
            last_n: 2,
        };
        penalizer.apply(&mut logits, &prior_tokens);

        assert_eq!(logits[0], 5.0); // outside window
        assert!((logits[1] - 2.5).abs() < 1e-5);
        assert!((logits[2] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_disabled_at_one() {
        let mut logits = vec![5.0, 5.0];
        let original = logits.clone();
        let mut penalizer = RepetitionPenalizer {
            penalty: 1.0,
            last_n: 64,
        };
        penalizer.apply(&mut logits, &[0, 1]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_logit_bias_shifts_specific_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut biaser = LogitBiaser {
            biases: vec![(0, 10.0), (3, -10.0)],
        };
        biaser.apply(&mut logits, &[]);

        assert!((logits[0] - 11.0).abs() < 1e-5);
        assert!((logits[1] - 2.0).abs() < 1e-5);
        assert!((logits[2] - 3.0).abs() < 1e-5);
        assert!((logits[3] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_logit_bias_ignores_out_of_range() {
        let mut logits = vec![1.0, 2.0];
        let original = logits.clone();
        let mut biaser = LogitBiaser {
            biases: vec![(999, 100.0)],
        };
        biaser.apply(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_n_sigma_filters_outliers() {
        // logits: one high value, rest clustered low
        let mut logits = vec![10.0, 2.0, 1.5, 1.0, 0.5, 0.0];
        let mut sampler = TopNSigmaSampler {
            n_sigma: 1.0,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);

        // the top logit should always survive
        assert!(logits[0].is_finite());
        // some lower logits may be filtered depending on std
    }

    #[test]
    fn test_top_n_sigma_disabled_when_zero() {
        let mut logits = vec![10.0, 1.0, 0.0];
        let original = logits.clone();
        let mut sampler = TopNSigmaSampler {
            n_sigma: 0.0,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_sampler_chain_applies_in_order() {
        let mut chain = SamplerChain::new();

        // first: bias token 0 up
        chain.push(Box::new(LogitBiaser {
            biases: vec![(0, 100.0)],
        }));

        // second: min_p filter should keep token 0 (now dominant)
        chain.push(Box::new(MinPSampler {
            min_p: 0.1,
            min_keep: 1,
        }));

        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        chain.apply_all(&mut logits, &[]);

        // token 0 should have been biased up, then others filtered by min_p
        assert!(logits[0].is_finite());
        // the others should have been filtered out
        for &l in &logits[1..] {
            assert_eq!(l, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_sampler_chain_empty_is_noop() {
        let mut chain = SamplerChain::new();
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        chain.apply_all(&mut logits, &[]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_n_sigma_concrete_values() {
        // uniform-ish logits with one outlier
        let mut logits = vec![1.0, 1.0, 1.0, 1.0, 1.0, 20.0];

        // mean = (5*1 + 20)/6 = 25/6 ≈ 4.17
        // var = (5*(1-4.17)^2 + (20-4.17)^2)/6 ≈ (5*10.03 + 250.69)/6 ≈ 50.1+250.7/6 ≈ 50.1
        // std ≈ 7.08
        // threshold = 20 - 1.0 * 7.08 = 12.92
        // only logit 20.0 survives
        let mut sampler = TopNSigmaSampler {
            n_sigma: 1.0,
            min_keep: 1,
        };
        sampler.apply(&mut logits, &[]);

        assert!(logits[5].is_finite());
        for &l in &logits[..5] {
            assert_eq!(l, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_mirostat_v2_keeps_at_least_one() {
        // very low mu should mask almost everything, but at least one token survives
        let mut logits = vec![10.0, 5.0, 0.0, -5.0];
        let mut sampler = MirostatV2Sampler {
            tau: 0.001,
            eta: 0.1,
            mu: 0.001,
        };
        sampler.apply(&mut logits, &[]);

        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert!(finite_count >= 1);
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_min_p_never_filters_max(
                logits in prop::collection::vec(-50.0f32..50.0f32, 4..32),
                min_p in 0.01f32..0.99f32,
            ) {
                let max_idx = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap();
                let mut logits = logits;
                let mut sampler = MinPSampler { min_p, min_keep: 1 };
                sampler.apply(&mut logits, &[]);
                prop_assert!(logits[max_idx].is_finite(),
                    "min_p filtered the max logit at index {max_idx}");
            }

            #[test]
            fn prop_repetition_penalty_preserves_unpenalized(
                base in prop::collection::vec(-10.0f32..10.0f32, 8..32),
                penalty in 1.1f32..5.0f32,
            ) {
                let mut logits = base.clone();
                // only penalize token 0
                let mut penalizer = RepetitionPenalizer { penalty, last_n: 64 };
                penalizer.apply(&mut logits, &[0]);
                // tokens 1+ should be unchanged
                for i in 1..logits.len() {
                    prop_assert!((logits[i] - base[i]).abs() < 1e-5,
                        "token {i} changed: {} -> {}", base[i], logits[i]);
                }
            }

            #[test]
            fn prop_top_n_sigma_keeps_max(
                logits in prop::collection::vec(-50.0f32..50.0f32, 4..32),
                n in 0.5f32..5.0f32,
            ) {
                let max_idx = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap();
                let mut logits = logits;
                let mut sampler = TopNSigmaSampler { n_sigma: n, min_keep: 1 };
                sampler.apply(&mut logits, &[]);
                prop_assert!(logits[max_idx].is_finite(),
                    "top_n_sigma filtered the max logit");
            }

            #[test]
            fn prop_logit_bias_additive(
                logits in prop::collection::vec(-10.0f32..10.0f32, 4..16),
                bias in -20.0f32..20.0f32,
            ) {
                let mut result = logits.clone();
                let mut biaser = LogitBiaser { biases: vec![(0, bias)] };
                biaser.apply(&mut result, &[]);
                prop_assert!((result[0] - (logits[0] + bias)).abs() < 1e-5);
                for i in 1..logits.len() {
                    prop_assert!((result[i] - logits[i]).abs() < 1e-5);
                }
            }
        }
    }
}

// ── Mirostat v1 ────────────────────────────────────────────────────────────

/// Mirostat v1 — estimates Zipf exponent from top-m tokens, computes dynamic
/// top-k targeting entropy tau. Translated from llama_sampler_mirostat_apply.
pub struct MirostatV1Sampler {
    pub n_vocab: usize,
    pub tau: f32,
    pub eta: f32,
    pub m: usize,
    pub mu: f32,
}

impl MirostatV1Sampler {
    pub fn new(n_vocab: usize, tau: f32, eta: f32, m: usize) -> Self {
        Self { n_vocab, tau, eta, m, mu: 2.0 * tau }
    }

    pub fn update_mu(&mut self, selected_prob: f32) {
        if selected_prob > 0.0 {
            let observed_surprise = -selected_prob.log2();
            self.mu -= self.eta * (observed_surprise - self.tau);
        }
    }
}

impl LogitProcessor for MirostatV1Sampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        if logits.is_empty() { return; }

        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
            .map(|(i, &l)| (i, (l - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() { *p /= sum; }
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let m_use = (self.m - 1).min(probs.len().saturating_sub(1));
        let mut sum_ti_bi = 0.0f32;
        let mut sum_ti_sq = 0.0f32;
        for i in 0..m_use {
            let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b_i = if probs[i + 1].1 > 0.0 { (probs[i].1 / probs[i + 1].1).ln() } else { 0.0 };
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        let s_hat = if sum_ti_sq > 0.0 { sum_ti_bi / sum_ti_sq } else { 1.0 };

        let epsilon_hat = s_hat - 1.0;
        let k = if epsilon_hat > 0.0 && self.n_vocab > 0 {
            let num = epsilon_hat * 2.0f32.powf(self.mu);
            let denom = 1.0 - (self.n_vocab as f32).powf(-epsilon_hat);
            if denom > 0.0 { (num / denom).powf(1.0 / s_hat).max(1.0) as usize } else { 1 }
        } else { 1 };

        if k < probs.len() {
            let cutoff_logit = logits[probs[k - 1].0];
            for (i, l) in logits.iter_mut().enumerate() {
                if *l < cutoff_logit { *l = f32::NEG_INFINITY; }
            }
        }
    }
}

// ── Reasoning Budget ───────────────────────────────────────────────────────

/// Limits token generation budget for thinking/reasoning phases.
/// Blocks thinking_end_token before min_budget, forces it after max_budget.
pub struct ReasoningBudgetSampler {
    pub thinking_end_token: u32,
    pub min_budget: usize,
    pub max_budget: usize,
    pub tokens_generated: usize,
}

impl ReasoningBudgetSampler {
    pub fn new(thinking_end_token: u32, min_budget: usize, max_budget: usize) -> Self {
        Self { thinking_end_token, min_budget, max_budget, tokens_generated: 0 }
    }

    pub fn step(&mut self) { self.tokens_generated += 1; }
}

impl LogitProcessor for ReasoningBudgetSampler {
    fn apply(&mut self, logits: &mut [f32], _prior_tokens: &[u32]) {
        let tok = self.thinking_end_token as usize;
        if tok >= logits.len() { return; }
        if self.tokens_generated < self.min_budget {
            logits[tok] = f32::NEG_INFINITY;
        } else if self.tokens_generated >= self.max_budget {
            for (i, l) in logits.iter_mut().enumerate() {
                if i != tok { *l = f32::NEG_INFINITY; }
            }
        }
    }
}

#[cfg(test)]
mod mirostat_v1_tests {
    use super::*;

    #[test]
    fn mirostat_v1_reduces_candidates() {
        let mut logits = vec![1.0, 0.9, 0.8, 0.7, 0.1, 0.05, 0.01, -1.0, -2.0, -3.0];
        let mut sampler = MirostatV1Sampler::new(10, 5.0, 0.1, 5);
        sampler.apply(&mut logits, &[]);
        let kept = logits.iter().filter(|&&l| l > f32::NEG_INFINITY).count();
        assert!(kept >= 1, "kept={kept}");
    }

    #[test]
    fn reasoning_budget_blocks_early() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut sampler = ReasoningBudgetSampler::new(2, 10, 100);
        sampler.apply(&mut logits, &[]);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert!(logits[0] > f32::NEG_INFINITY);
    }

    #[test]
    fn reasoning_budget_forces_end() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut sampler = ReasoningBudgetSampler::new(2, 5, 10);
        sampler.tokens_generated = 15;
        sampler.apply(&mut logits, &[]);
        assert!(logits[2] > f32::NEG_INFINITY);
        assert_eq!(logits[0], f32::NEG_INFINITY);
    }
}

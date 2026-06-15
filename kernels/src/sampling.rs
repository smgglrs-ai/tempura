use crate::Result;

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
        }
    }
}

pub trait Sampler: Send + Sync {
    /// Apply temperature, top-k, and top-p filtering to logits, then return
    /// the probability distribution (softmax over filtered logits).
    ///
    /// - `logits`: [vocab_size] — raw logits for a single token position
    /// - `output`: [vocab_size] — filtered probabilities (zeros for pruned tokens)
    fn apply(&self, logits: &[f32], output: &mut [f32], params: &SamplingParams) -> Result<()>;
}

#[cfg(feature = "cpu")]
pub struct CpuSampler;

#[cfg(feature = "cpu")]
impl Sampler for CpuSampler {
    fn apply(&self, logits: &[f32], output: &mut [f32], params: &SamplingParams) -> Result<()> {
        let vocab_size = logits.len();

        // temperature scaling
        let scaled: Vec<f32> = if params.temperature != 1.0 && params.temperature > 0.0 {
            logits.iter().map(|l| l / params.temperature).collect()
        } else {
            logits.to_vec()
        };

        // top-k: find threshold
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = params.top_k.min(vocab_size);
        let _top_k_threshold = indexed[k - 1].1;

        // softmax over top-k candidates
        let max_val = indexed[0].1;
        let mut probs = vec![0.0f32; vocab_size];
        let mut sum = 0.0f32;

        for &(idx, val) in &indexed[..k] {
            let p = (val - max_val).exp();
            probs[idx] = p;
            sum += p;
        }

        for p in &mut probs {
            *p /= sum;
        }

        // top-p (nucleus) filtering
        if params.top_p < 1.0 {
            let mut sorted_probs: Vec<(usize, f32)> = probs
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, p)| *p > 0.0)
                .collect();
            sorted_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
                },
            )
            .unwrap();

        // with very low temperature, nearly all probability on index 1
        let argmax = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
}

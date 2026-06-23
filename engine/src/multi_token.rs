use tempura_kernels::{Sampler, SamplingParams};

pub struct MedusaHead {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl MedusaHead {
    pub fn predict(&self, hidden_state: &[f32]) -> Vec<f32> {
        let mut logits = self.bias.clone();
        for j in 0..self.vocab_size {
            let mut sum = 0.0f32;
            for i in 0..self.hidden_size {
                sum += hidden_state[i] * self.weights[j * self.hidden_size + i];
            }
            logits[j] += sum;
        }
        logits
    }
}

pub struct MultiTokenPredictor {
    heads: Vec<MedusaHead>,
    pub num_draft_tokens: usize,
}

impl MultiTokenPredictor {
    pub fn new(heads: Vec<MedusaHead>) -> Self {
        let num_draft_tokens = heads.len();
        Self {
            heads,
            num_draft_tokens,
        }
    }

    pub fn draft(
        &self,
        hidden_state: &[f32],
        sampler: &dyn Sampler,
        params: &SamplingParams,
    ) -> Vec<u32> {
        let mut drafts = Vec::with_capacity(self.num_draft_tokens);
        for head in &self.heads {
            let logits = head.predict(hidden_state);
            let mut probs = vec![0.0f32; logits.len()];
            sampler.apply(&logits, &mut probs, params).unwrap();
            let token = sampler.sample_token(&probs, None);
            drafts.push(token);
        }
        drafts
    }

    pub fn verify_and_accept(
        draft_tokens: &[u32],
        _draft_logits: &[Vec<f32>],
        target_logits: &[Vec<f32>],
        sampler: &dyn Sampler,
        params: &SamplingParams,
    ) -> (Vec<u32>, usize) {
        let mut accepted = Vec::new();

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() {
                break;
            }

            let mut target_probs = vec![0.0f32; target_logits[i].len()];
            sampler
                .apply(&target_logits[i], &mut target_probs, params)
                .unwrap();
            let target_token = sampler.sample_token(&target_probs, None);

            if target_token == draft_token {
                accepted.push(draft_token);
            } else {
                accepted.push(target_token);
                break;
            }
        }

        let num_accepted = if accepted.last().map(|&t| {
            draft_tokens
                .get(accepted.len() - 1)
                .map(|&d| d == t)
                .unwrap_or(false)
        }) == Some(true)
        {
            accepted.len()
        } else {
            accepted.len().saturating_sub(1)
        };

        (accepted, num_accepted)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MultiTokenTrace {
    pub step: usize,
    pub draft_tokens: Vec<u32>,
    pub accepted_count: usize,
    pub total_proposed: usize,
    pub acceptance_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempura_kernels::sampling::CpuSampler;

    fn make_head(hidden: usize, vocab: usize) -> MedusaHead {
        MedusaHead {
            weights: vec![0.01f32; vocab * hidden],
            bias: vec![0.0f32; vocab],
            hidden_size: hidden,
            vocab_size: vocab,
        }
    }

    #[test]
    fn test_medusa_head_predict() {
        let head = make_head(4, 8);
        let hidden = vec![1.0f32; 4];
        let logits = head.predict(&hidden);
        assert_eq!(logits.len(), 8);
        assert!(logits.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_multi_token_draft() {
        let heads = vec![make_head(4, 8), make_head(4, 8), make_head(4, 8)];
        let predictor = MultiTokenPredictor::new(heads);
        assert_eq!(predictor.num_draft_tokens, 3);

        let hidden = vec![1.0f32; 4];
        let sampler = CpuSampler;
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let drafts = predictor.draft(&hidden, &sampler, &params);
        assert_eq!(drafts.len(), 3);
    }

    #[test]
    fn test_verify_all_accepted() {
        let sampler = CpuSampler;
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let mut logits = vec![0.0f32; 8];
        logits[3] = 10.0;

        let draft_tokens = vec![3, 3];
        let draft_logits = vec![logits.clone(), logits.clone()];
        let target_logits = vec![logits.clone(), logits.clone()];

        let (accepted, count) = MultiTokenPredictor::verify_and_accept(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            &sampler,
            &params,
        );

        assert_eq!(accepted, vec![3, 3]);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_verify_first_rejected() {
        let sampler = CpuSampler;
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let mut draft_logits_0 = vec![0.0f32; 8];
        draft_logits_0[3] = 10.0;
        let mut target_logits_0 = vec![0.0f32; 8];
        target_logits_0[5] = 10.0;

        let draft_tokens = vec![3, 3];
        let draft_logits = vec![draft_logits_0.clone(), draft_logits_0];
        let target_logits = vec![target_logits_0, draft_logits[1].clone()];

        let (accepted, _count) = MultiTokenPredictor::verify_and_accept(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            &sampler,
            &params,
        );

        assert_eq!(accepted[0], 5);
        assert_eq!(accepted.len(), 1);
    }
}

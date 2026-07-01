use crate::Result;

/// Gating function applied to router logits before expert selection.
///
/// Different MoE architectures use different gating:
/// - Mixtral, Qwen3MOE: softmax over all experts, then top-k
/// - DeepSeekV2/V3: softmax with optional bias, sigmoid for some variants
/// - Llama4: sigmoid gating (weight-before-FFN)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingFunc {
    Softmax,
    Sigmoid,
    /// Softmax applied only to the selected top-k weights (not all experts).
    /// Used by some architectures that pass raw logits through selection first.
    SoftmaxOnSelected,
}

/// Configuration for MoE routing.
#[derive(Debug, Clone)]
pub struct MoeRouter {
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub gating: GatingFunc,
    /// Whether to renormalize selected expert weights to sum to 1.
    /// Most architectures do this; some (sigmoid-gated) skip it.
    pub normalize_weights: bool,
}

/// Per-token routing decision: which experts to use and their weights.
#[derive(Debug, Clone)]
pub struct MoeRoutingResult {
    pub expert_indices: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

impl MoeRouter {
    pub fn new(n_experts: usize, n_experts_used: usize) -> Self {
        Self {
            n_experts,
            n_experts_used,
            gating: GatingFunc::Softmax,
            normalize_weights: true,
        }
    }

    /// Route tokens to experts.
    ///
    /// `router_logits`: flat array of shape [num_tokens, n_experts] — raw logits
    /// from the router linear projection.
    ///
    /// Returns one `MoeRoutingResult` per token.
    pub fn route(
        &self,
        router_logits: &[f32],
        num_tokens: usize,
    ) -> Result<Vec<MoeRoutingResult>> {
        if router_logits.len() != num_tokens * self.n_experts {
            return Err(crate::KernelError::DimensionMismatch {
                expected: num_tokens * self.n_experts,
                got: router_logits.len(),
            });
        }
        if self.n_experts_used > self.n_experts {
            return Err(crate::KernelError::DimensionMismatch {
                expected: self.n_experts,
                got: self.n_experts_used,
            });
        }
        if self.n_experts == 0 || self.n_experts_used == 0 {
            return Err(crate::KernelError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let mut results = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let logits = &router_logits[t * self.n_experts..(t + 1) * self.n_experts];
            let result = self.route_single(logits);
            results.push(result);
        }

        Ok(results)
    }

    fn route_single(&self, logits: &[f32]) -> MoeRoutingResult {
        let probs = match self.gating {
            GatingFunc::Softmax => softmax(logits),
            GatingFunc::Sigmoid => sigmoid_vec(logits),
            GatingFunc::SoftmaxOnSelected => {
                logits.to_vec()
            }
        };

        // Select top-k experts by probability/score
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        // Sort descending by score. Stable sort preserves index order for ties.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(self.n_experts_used);

        let mut expert_weights: Vec<f32> = indexed.iter().map(|(_, w)| *w).collect();

        // For SoftmaxOnSelected, apply softmax to the selected logits now
        if self.gating == GatingFunc::SoftmaxOnSelected {
            expert_weights = softmax(&expert_weights);
        }

        if self.normalize_weights {
            let sum: f32 = expert_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut expert_weights {
                    *w /= sum;
                }
            }
        }

        // Sort by expert index for deterministic dispatch order
        let mut pairs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, _)| *i)
            .zip(expert_weights.iter().copied())
            .collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        let expert_indices: Vec<usize> = pairs.iter().map(|(i, _)| *i).collect();
        let expert_weights: Vec<f32> = pairs.iter().map(|(_, w)| *w).collect();

        MoeRoutingResult {
            expert_indices,
            expert_weights,
        }
    }
}

/// Softmax over a slice of f32.
fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        let uniform = 1.0 / x.len() as f32;
        return vec![uniform; x.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Element-wise sigmoid.
fn sigmoid_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let v = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1]);
        assert!(v[1] > v[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let v = softmax(&[0.0, 0.0, 0.0]);
        for &p in &v {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_range() {
        let v = sigmoid_vec(&[-10.0, 0.0, 10.0]);
        assert!(v[0] < 0.01);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!(v[2] > 0.99);
    }

    #[test]
    fn test_router_top_k_selection() {
        let router = MoeRouter::new(8, 2);
        // Expert 3 and expert 7 should have highest logits
        let logits = [0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 4.0];
        let results = router.route(&logits, 1).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.expert_indices.len(), 2);
        assert!(r.expert_indices.contains(&3));
        assert!(r.expert_indices.contains(&7));
    }

    #[test]
    fn test_router_weights_normalized() {
        let router = MoeRouter::new(4, 2);
        let logits = [1.0, 5.0, 0.5, 3.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        let sum: f32 = r.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights should sum to 1, got {sum}");
    }

    #[test]
    fn test_router_indices_sorted() {
        let router = MoeRouter::new(8, 3);
        let logits = [0.0, 0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 4.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        assert!(r.expert_indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_router_multi_token() {
        let router = MoeRouter::new(4, 2);
        let logits = [
            1.0, 5.0, 0.5, 3.0, // token 0: experts 1, 3
            4.0, 0.1, 0.2, 6.0, // token 1: experts 0, 3
        ];
        let results = router.route(&logits, 2).unwrap();
        assert_eq!(results.len(), 2);

        assert!(results[0].expert_indices.contains(&1));
        assert!(results[0].expert_indices.contains(&3));

        assert!(results[1].expert_indices.contains(&0));
        assert!(results[1].expert_indices.contains(&3));
    }

    #[test]
    fn test_router_single_expert_equals_dense() {
        let router = MoeRouter::new(1, 1);
        let logits = [1.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        assert_eq!(r.expert_indices, vec![0]);
        assert!((r.expert_weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_router_all_experts_used() {
        let router = MoeRouter::new(4, 4);
        let logits = [1.0, 2.0, 3.0, 4.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        assert_eq!(r.expert_indices.len(), 4);
        assert_eq!(r.expert_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_router_dimension_mismatch() {
        let router = MoeRouter::new(8, 2);
        let logits = [1.0, 2.0];
        let result = router.route(&logits, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_router_experts_used_exceeds_total() {
        let router = MoeRouter {
            n_experts: 4,
            n_experts_used: 8,
            gating: GatingFunc::Softmax,
            normalize_weights: true,
        };
        let logits = [1.0, 2.0, 3.0, 4.0];
        let result = router.route(&logits, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_router_sigmoid_gating() {
        let mut router = MoeRouter::new(4, 2);
        router.gating = GatingFunc::Sigmoid;
        let logits = [0.0, 10.0, -10.0, 5.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        assert_eq!(r.expert_indices.len(), 2);
        assert!(r.expert_indices.contains(&1));
        assert!(r.expert_indices.contains(&3));
    }

    #[test]
    fn test_router_softmax_on_selected() {
        let mut router = MoeRouter::new(4, 2);
        router.gating = GatingFunc::SoftmaxOnSelected;
        let logits = [0.0, 10.0, -10.0, 5.0];
        let results = router.route(&logits, 1).unwrap();
        let r = &results[0];
        assert_eq!(r.expert_indices.len(), 2);
        assert!(r.expert_indices.contains(&1));
        assert!(r.expert_indices.contains(&3));
        let sum: f32 = r.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_router_zero_experts() {
        let router = MoeRouter {
            n_experts: 0,
            n_experts_used: 0,
            gating: GatingFunc::Softmax,
            normalize_weights: true,
        };
        let result = router.route(&[], 0);
        assert!(result.is_err());
    }

    #[cfg(kani)]
    mod kani_proofs {
        use super::*;

        #[kani::proof]
        #[kani::unwind(10)]
        fn router_indices_in_bounds() {
            let n_experts: usize = kani::any();
            let n_used: usize = kani::any();
            kani::assume(n_experts >= 1 && n_experts <= 8);
            kani::assume(n_used >= 1 && n_used <= n_experts);

            let router = MoeRouter::new(n_experts, n_used);

            let mut logits = [0.0f32; 8];
            for i in 0..n_experts {
                logits[i] = kani::any();
                kani::assume(logits[i].is_finite());
            }

            let results = router.route(&logits[..n_experts], 1).unwrap();
            let r = &results[0];

            assert_eq!(r.expert_indices.len(), n_used);
            assert_eq!(r.expert_weights.len(), n_used);

            for &idx in &r.expert_indices {
                assert!(idx < n_experts);
            }
        }

        #[kani::proof]
        #[kani::unwind(10)]
        fn router_weights_non_negative() {
            let n_experts: usize = kani::any();
            let n_used: usize = kani::any();
            kani::assume(n_experts >= 1 && n_experts <= 4);
            kani::assume(n_used >= 1 && n_used <= n_experts);

            let router = MoeRouter::new(n_experts, n_used);

            let mut logits = [0.0f32; 4];
            for i in 0..n_experts {
                logits[i] = kani::any();
                kani::assume(logits[i].is_finite());
                kani::assume(logits[i] >= -100.0 && logits[i] <= 100.0);
            }

            let results = router.route(&logits[..n_experts], 1).unwrap();
            for w in &results[0].expert_weights {
                assert!(*w >= 0.0);
            }
        }

        #[kani::proof]
        #[kani::unwind(10)]
        fn router_weights_sum_to_one() {
            let n_experts: usize = kani::any();
            let n_used: usize = kani::any();
            kani::assume(n_experts >= 1 && n_experts <= 4);
            kani::assume(n_used >= 1 && n_used <= n_experts);

            let router = MoeRouter::new(n_experts, n_used);

            let mut logits = [0.0f32; 4];
            for i in 0..n_experts {
                logits[i] = kani::any();
                kani::assume(logits[i].is_finite());
                kani::assume(logits[i] >= -10.0 && logits[i] <= 10.0);
            }

            let results = router.route(&logits[..n_experts], 1).unwrap();
            let sum: f32 = results[0].expert_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "weights sum = {sum}, expected ~1.0");
        }

        #[kani::proof]
        #[kani::unwind(6)]
        fn softmax_no_panic() {
            let mut arr = [0.0f32; 4];
            for i in 0..4 {
                arr[i] = kani::any();
                kani::assume(arr[i].is_finite());
                kani::assume(arr[i] >= -100.0 && arr[i] <= 100.0);
            }
            let result = softmax(&arr);
            assert_eq!(result.len(), 4);
        }
    }
}

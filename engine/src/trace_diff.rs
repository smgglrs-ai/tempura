use crate::trace::InferenceTrace;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TraceDiff {
    pub divergence: DivergenceLevel,
    pub first_divergence: Option<DivergencePoint>,
    pub layer_diffs: Vec<LayerDiff>,
    pub token_diffs: Vec<TokenDiff>,
    pub metadata_match: MetadataMatch,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceLevel {
    None,
    Numerical { max_abs_diff: f32 },
    Semantic { first_mismatch_step: usize },
}

#[derive(Debug, Clone, Serialize)]
pub struct DivergencePoint {
    pub kind: &'static str,
    pub layer_index: Option<usize>,
    pub token_step: Option<usize>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDiff {
    pub layer_index: usize,
    pub pre_norm_diff: Option<NormDiff>,
    pub post_norm_diff: Option<NormDiff>,
    pub attention_max_abs_diff: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NormDiff {
    pub mean_diff: f32,
    pub variance_diff: f32,
    pub max_abs_diff: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenDiff {
    pub step: usize,
    pub token_a: u32,
    pub token_b: u32,
    pub tokens_match: bool,
    pub logit_max_abs_diff: Option<f32>,
    pub prob_max_abs_diff: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetadataMatch {
    pub model_id_match: bool,
    pub prompt_tokens_match: bool,
    pub seed_match: bool,
    pub level_match: bool,
}

pub fn diff_traces(a: &InferenceTrace, b: &InferenceTrace, epsilon: f32) -> TraceDiff {
    let metadata_match = MetadataMatch {
        model_id_match: a.model_id == b.model_id,
        prompt_tokens_match: a.prompt_tokens == b.prompt_tokens,
        seed_match: a.seed == b.seed,
        level_match: a.level == b.level,
    };

    let layer_count = a.layers.len().min(b.layers.len());
    let mut layer_diffs = Vec::with_capacity(layer_count);
    let mut max_layer_diff: f32 = 0.0;
    let mut first_divergence: Option<DivergencePoint> = None;

    for i in 0..layer_count {
        let la = &a.layers[i];
        let lb = &b.layers[i];

        let pre_norm_diff = match (&la.pre_norm, &lb.pre_norm) {
            (Some(na), Some(nb)) => {
                let nd = NormDiff {
                    mean_diff: (na.mean - nb.mean).abs(),
                    variance_diff: (na.variance - nb.variance).abs(),
                    max_abs_diff: (na.max_abs - nb.max_abs).abs(),
                };
                let m = nd.mean_diff.max(nd.variance_diff).max(nd.max_abs_diff);
                max_layer_diff = max_layer_diff.max(m);
                if m > epsilon && first_divergence.is_none() {
                    first_divergence = Some(DivergencePoint {
                        kind: "numerical",
                        layer_index: Some(i),
                        token_step: None,
                        detail: format!("pre_norm divergence {m:.6} at layer {i}"),
                    });
                }
                Some(nd)
            }
            _ => None,
        };

        let post_norm_diff = match (&la.post_norm, &lb.post_norm) {
            (Some(na), Some(nb)) => {
                let nd = NormDiff {
                    mean_diff: (na.mean - nb.mean).abs(),
                    variance_diff: (na.variance - nb.variance).abs(),
                    max_abs_diff: (na.max_abs - nb.max_abs).abs(),
                };
                let m = nd.mean_diff.max(nd.variance_diff).max(nd.max_abs_diff);
                max_layer_diff = max_layer_diff.max(m);
                Some(nd)
            }
            _ => None,
        };

        let attention_max_abs_diff = match (&la.attention_scores, &lb.attention_scores) {
            (Some(sa), Some(sb)) => {
                let len = sa.len().min(sb.len());
                let d = (0..len)
                    .map(|j| (sa[j] - sb[j]).abs())
                    .fold(0.0f32, f32::max);
                max_layer_diff = max_layer_diff.max(d);
                Some(d)
            }
            _ => None,
        };

        layer_diffs.push(LayerDiff {
            layer_index: i,
            pre_norm_diff,
            post_norm_diff,
            attention_max_abs_diff,
        });
    }

    let step_count = a.sampling_steps.len().min(b.sampling_steps.len());
    let mut token_diffs = Vec::with_capacity(step_count);
    let mut any_token_mismatch = false;
    let mut first_mismatch_step = None;

    for i in 0..step_count {
        let sa = &a.sampling_steps[i];
        let sb = &b.sampling_steps[i];
        let tokens_match = sa.selected_token == sb.selected_token;
        if !tokens_match && !any_token_mismatch {
            any_token_mismatch = true;
            first_mismatch_step = Some(i);
            if first_divergence.is_none() {
                first_divergence = Some(DivergencePoint {
                    kind: "semantic",
                    layer_index: None,
                    token_step: Some(i),
                    detail: format!(
                        "token mismatch at step {i}: {} vs {}",
                        sa.selected_token, sb.selected_token
                    ),
                });
            }
        }

        let logit_max_abs_diff = if !sa.top_logits.is_empty() && !sb.top_logits.is_empty() {
            let len = sa.top_logits.len().min(sb.top_logits.len());
            Some(
                (0..len)
                    .map(|j| (sa.top_logits[j].1 - sb.top_logits[j].1).abs())
                    .fold(0.0f32, f32::max),
            )
        } else {
            None
        };

        let prob_max_abs_diff = if !sa.top_probs.is_empty() && !sb.top_probs.is_empty() {
            let len = sa.top_probs.len().min(sb.top_probs.len());
            Some(
                (0..len)
                    .map(|j| (sa.top_probs[j].1 - sb.top_probs[j].1).abs())
                    .fold(0.0f32, f32::max),
            )
        } else {
            None
        };

        token_diffs.push(TokenDiff {
            step: i,
            token_a: sa.selected_token,
            token_b: sb.selected_token,
            tokens_match,
            logit_max_abs_diff,
            prob_max_abs_diff,
        });
    }

    let divergence = if any_token_mismatch {
        DivergenceLevel::Semantic {
            first_mismatch_step: first_mismatch_step.unwrap(),
        }
    } else if max_layer_diff > epsilon {
        DivergenceLevel::Numerical {
            max_abs_diff: max_layer_diff,
        }
    } else {
        DivergenceLevel::None
    };

    TraceDiff {
        divergence,
        first_divergence,
        layer_diffs,
        token_diffs,
        metadata_match,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::*;

    fn make_trace(id: &str, seed: u64) -> InferenceTrace {
        let mut t = InferenceTrace::new(id.into(), TraceLevel::Full, "test".into(), seed);
        t.prompt_tokens = 5;
        t.generated_tokens = 3;
        t.layers.push(LayerSnapshot {
            layer_index: 0,
            pre_norm: Some(NormStats {
                mean: 0.1,
                variance: 0.02,
                max_abs: 1.5,
            }),
            post_norm: None,
            attention_scores: Some(vec![0.5, 0.3, 0.2]),
            duration: None,
        });
        t.sampling_steps.push(SamplingSnapshot {
            step: 0,
            top_logits: vec![(10, 5.0), (20, 3.0)],
            top_probs: vec![(10, 0.8), (20, 0.15)],
            selected_token: 10,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            rng_seed: seed,
            tokens_masked: None,
        });
        t.sampling_steps.push(SamplingSnapshot {
            step: 1,
            top_logits: vec![(30, 4.0)],
            top_probs: vec![(30, 0.9)],
            selected_token: 30,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            rng_seed: seed,
            tokens_masked: None,
        });
        t
    }

    #[test]
    fn test_identical_traces_no_divergence() {
        let a = make_trace("a", 42);
        let b = make_trace("b", 42);
        let diff = diff_traces(&a, &b, 1e-6);
        assert_eq!(diff.divergence, DivergenceLevel::None);
        assert!(diff.first_divergence.is_none());
        assert!(diff.token_diffs.iter().all(|t| t.tokens_match));
    }

    #[test]
    fn test_numerical_divergence() {
        let a = make_trace("a", 42);
        let mut b = make_trace("b", 42);
        b.layers[0].pre_norm.as_mut().unwrap().mean = 0.15;
        let diff = diff_traces(&a, &b, 1e-6);
        match &diff.divergence {
            DivergenceLevel::Numerical { max_abs_diff } => {
                assert!(*max_abs_diff > 0.04);
            }
            other => panic!("expected Numerical, got {other:?}"),
        }
        assert!(diff.first_divergence.is_some());
        assert_eq!(diff.first_divergence.as_ref().unwrap().kind, "numerical");
    }

    #[test]
    fn test_semantic_divergence() {
        let a = make_trace("a", 42);
        let mut b = make_trace("b", 42);
        b.sampling_steps[1].selected_token = 99;
        let diff = diff_traces(&a, &b, 1e-6);
        match &diff.divergence {
            DivergenceLevel::Semantic {
                first_mismatch_step,
            } => {
                assert_eq!(*first_mismatch_step, 1);
            }
            other => panic!("expected Semantic, got {other:?}"),
        }
        assert!(!diff.token_diffs[1].tokens_match);
        assert_eq!(diff.token_diffs[1].token_a, 30);
        assert_eq!(diff.token_diffs[1].token_b, 99);
    }

    #[test]
    fn test_diff_serializes_to_json() {
        let a = make_trace("a", 42);
        let b = make_trace("b", 42);
        let diff = diff_traces(&a, &b, 1e-6);
        let json = serde_json::to_string(&diff).expect("serialize diff");
        assert!(json.contains("\"none\""));
    }

    #[test]
    fn test_metadata_match() {
        let a = make_trace("a", 42);
        let mut b = make_trace("b", 99);
        b.prompt_tokens = 10;
        let diff = diff_traces(&a, &b, 1e-6);
        assert!(!diff.metadata_match.seed_match);
        assert!(!diff.metadata_match.prompt_tokens_match);
        assert!(diff.metadata_match.model_id_match);
    }
}

use oxide_engine::session::InferenceSession;
use oxide_engine::trace::TraceLevel;
use oxide_kernels::SamplingParams;
use serde::{Deserialize, Serialize};

use oxide_engine::InferenceModel;

#[derive(Debug, Deserialize)]
pub struct ReplayRequest {
    pub trace_id: Option<String>,
    pub token_ids: Option<Vec<u32>>,
    pub seed: Option<u64>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct ReplayResult {
    pub matches: bool,
    pub original_tokens: Vec<u32>,
    pub replay_tokens: Vec<u32>,
    pub per_token_match: Vec<bool>,
}

pub fn replay_from_params(
    model: &dyn InferenceModel,
    token_ids: &[u32],
    seed: u64,
    max_tokens: usize,
    params: &SamplingParams,
) -> (Vec<u32>, Vec<u32>) {
    let mut session1 = InferenceSession::new(model.config(), TraceLevel::Off, seed);
    let tokens1 = model.generate(&mut session1, token_ids, max_tokens, params, None);

    let mut session2 = InferenceSession::new(model.config(), TraceLevel::Off, seed);
    let tokens2 = model.generate(&mut session2, token_ids, max_tokens, params, None);

    (tokens1, tokens2)
}

pub fn build_replay_result(original: Vec<u32>, replay: Vec<u32>) -> ReplayResult {
    let per_token_match: Vec<bool> = original
        .iter()
        .zip(replay.iter())
        .map(|(a, b)| a == b)
        .collect();
    let matches = original == replay;

    ReplayResult {
        matches,
        original_tokens: original,
        replay_tokens: replay,
        per_token_match,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_replay_result_matching() {
        let result = build_replay_result(vec![1, 2, 3], vec![1, 2, 3]);
        assert!(result.matches);
        assert_eq!(result.per_token_match, vec![true, true, true]);
    }

    #[test]
    fn test_build_replay_result_mismatch() {
        let result = build_replay_result(vec![1, 2, 3], vec![1, 9, 3]);
        assert!(!result.matches);
        assert_eq!(result.per_token_match, vec![true, false, true]);
    }
}

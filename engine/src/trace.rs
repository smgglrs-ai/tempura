use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceLevel {
    #[default]
    Off,
    Metadata,
    Summary,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    pub total_ms: f64,
    pub prompt_ms: f64,
    pub generation_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormStats {
    pub mean: f32,
    pub variance: f32,
    pub max_abs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingSnapshot {
    pub step: usize,
    pub top_logits: Vec<(u32, f32)>,
    pub top_probs: Vec<(u32, f32)>,
    pub selected_token: u32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rng_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_masked: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    pub layer_index: usize,
    pub pre_norm: Option<NormStats>,
    pub post_norm: Option<NormStats>,
    pub attention_scores: Option<Vec<f32>>,
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeSnapshot {
    pub step: usize,
    pub draft_proposed: Vec<u32>,
    pub accepted: Vec<u32>,
    pub rejected: Vec<u32>,
    pub acceptance_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTrace {
    pub trace_id: String,
    pub level: TraceLevel,
    pub model_id: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub seed: u64,
    pub timing: Option<TimingInfo>,
    pub layers: Vec<LayerSnapshot>,
    pub sampling_steps: Vec<SamplingSnapshot>,
    pub speculative_steps: Vec<SpeculativeSnapshot>,
}

impl InferenceTrace {
    pub fn new(trace_id: String, level: TraceLevel, model_id: String, seed: u64) -> Self {
        Self {
            trace_id,
            level,
            model_id,
            prompt_tokens: 0,
            generated_tokens: 0,
            seed,
            timing: None,
            layers: Vec::new(),
            sampling_steps: Vec::new(),
            speculative_steps: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_level_default_is_off() {
        assert_eq!(TraceLevel::default(), TraceLevel::Off);
    }

    #[test]
    fn test_trace_round_trip_serialization() {
        let mut trace = InferenceTrace::new(
            "trace-001".to_string(),
            TraceLevel::Full,
            "llama-2-7b".to_string(),
            42,
        );
        trace.prompt_tokens = 10;
        trace.generated_tokens = 5;
        trace.timing = Some(TimingInfo {
            total_ms: 150.0,
            prompt_ms: 50.0,
            generation_ms: 100.0,
            tokens_per_second: 50.0,
        });
        trace.layers.push(LayerSnapshot {
            layer_index: 0,
            pre_norm: Some(NormStats {
                mean: 0.1,
                variance: 0.02,
                max_abs: 1.5,
            }),
            post_norm: None,
            attention_scores: Some(vec![0.5, 0.3, 0.2]),
            duration: Some(Duration::from_micros(500)),
        });
        trace.sampling_steps.push(SamplingSnapshot {
            step: 0,
            top_logits: vec![(42, 3.5), (100, 2.1)],
            top_probs: vec![(42, 0.7), (100, 0.2)],
            selected_token: 42,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            rng_seed: 42,
            tokens_masked: None,
        });

        let json = serde_json::to_string(&trace).expect("serialize");
        let restored: InferenceTrace = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.trace_id, "trace-001");
        assert_eq!(restored.level, TraceLevel::Full);
        assert_eq!(restored.model_id, "llama-2-7b");
        assert_eq!(restored.prompt_tokens, 10);
        assert_eq!(restored.generated_tokens, 5);
        assert_eq!(restored.seed, 42);
        assert_eq!(restored.layers.len(), 1);
        assert_eq!(restored.layers[0].layer_index, 0);
        assert_eq!(restored.sampling_steps.len(), 1);
        assert_eq!(restored.sampling_steps[0].selected_token, 42);
        assert!(restored.timing.is_some());
    }

    #[test]
    fn test_trace_level_serde_snake_case() {
        let json = serde_json::to_string(&TraceLevel::Full).unwrap();
        assert_eq!(json, "\"full\"");

        let level: TraceLevel = serde_json::from_str("\"metadata\"").unwrap();
        assert_eq!(level, TraceLevel::Metadata);
    }

    #[test]
    fn test_empty_trace_round_trip() {
        let trace =
            InferenceTrace::new("empty".to_string(), TraceLevel::Off, "test".to_string(), 0);

        let json = serde_json::to_string(&trace).expect("serialize");
        let restored: InferenceTrace = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.level, TraceLevel::Off);
        assert!(restored.layers.is_empty());
        assert!(restored.sampling_steps.is_empty());
        assert!(restored.timing.is_none());
    }
}

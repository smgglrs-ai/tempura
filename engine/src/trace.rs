use serde::{Deserialize, Serialize};
use std::time::Duration;
use tempura_kernel_api::VerificationTier;

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
pub struct EdgeMetrics {
    pub joules_per_token: Option<f64>,
    pub memory_peak_bytes: Option<u64>,
    pub tokens_per_watt: Option<f64>,
    pub clock_mhz_min: Option<u32>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_verification_tier: Option<VerificationTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_metrics: Option<EdgeMetrics>,
}

pub const TRACE_MAGIC: [u8; 4] = [0x54, 0x50, 0x52, 0x41]; // "TPRA"
pub const TRACE_FORMAT_VERSION: u8 = 1;

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
            pipeline_verification_tier: None,
            edge_metrics: None,
        }
    }

    pub fn to_binary(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let payload = rmp_serde::to_vec(self)?;
        let mut buf = Vec::with_capacity(5 + payload.len());
        buf.extend_from_slice(&TRACE_MAGIC);
        buf.push(TRACE_FORMAT_VERSION);
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    pub fn from_binary(data: &[u8]) -> Result<Self, TraceDecodeError> {
        if data.len() < 5 {
            return Err(TraceDecodeError::TooShort);
        }
        if data[0..4] != TRACE_MAGIC {
            return Err(TraceDecodeError::BadMagic);
        }
        if data[4] != TRACE_FORMAT_VERSION {
            return Err(TraceDecodeError::UnsupportedVersion(data[4]));
        }
        rmp_serde::from_slice(&data[5..]).map_err(TraceDecodeError::Decode)
    }
}

impl EdgeMetrics {
    pub fn read_rapl_energy_uj() -> Option<u64> {
        std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }

    pub fn current_rss_bytes() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| {
                    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
                    Some(pages * 4096)
                })
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    pub fn from_measurement(
        energy_before_uj: Option<u64>,
        energy_after_uj: Option<u64>,
        tokens_generated: usize,
        _duration_secs: f64,
    ) -> Option<Self> {
        let joules_per_token = match (energy_before_uj, energy_after_uj) {
            (Some(before), Some(after)) if tokens_generated > 0 && after >= before => {
                let joules = (after - before) as f64 / 1_000_000.0;
                Some(joules / tokens_generated as f64)
            }
            _ => None,
        };

        let tokens_per_watt = joules_per_token.map(|jpt| if jpt > 0.0 { 1.0 / jpt } else { 0.0 });

        let memory_peak = Self::current_rss_bytes();

        Some(Self {
            joules_per_token,
            memory_peak_bytes: memory_peak,
            tokens_per_watt,
            clock_mhz_min: None,
        })
    }
}

#[derive(Debug)]
pub enum TraceDecodeError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u8),
    Decode(rmp_serde::decode::Error),
}

impl std::fmt::Display for TraceDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "trace data too short (need at least 5 bytes)"),
            Self::BadMagic => write!(f, "invalid trace magic (expected TPRA)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported trace format version {v}"),
            Self::Decode(e) => write!(f, "trace decode error: {e}"),
        }
    }
}

impl std::error::Error for TraceDecodeError {}

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

    #[test]
    fn test_binary_round_trip() {
        let mut trace = InferenceTrace::new(
            "bin-001".to_string(),
            TraceLevel::Summary,
            "test-model".to_string(),
            99,
        );
        trace.prompt_tokens = 20;
        trace.generated_tokens = 10;
        trace.timing = Some(TimingInfo {
            total_ms: 200.0,
            prompt_ms: 80.0,
            generation_ms: 120.0,
            tokens_per_second: 83.3,
        });
        trace.layers.push(LayerSnapshot {
            layer_index: 0,
            pre_norm: Some(NormStats {
                mean: 0.5,
                variance: 0.1,
                max_abs: 2.0,
            }),
            post_norm: None,
            attention_scores: None,
            duration: Some(Duration::from_micros(750)),
        });
        trace.sampling_steps.push(SamplingSnapshot {
            step: 0,
            top_logits: vec![(10, 5.0), (20, 3.0)],
            top_probs: vec![(10, 0.8), (20, 0.15)],
            selected_token: 10,
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            rng_seed: 99,
            tokens_masked: Some(5),
        });

        let binary = trace.to_binary().expect("to_binary");
        assert_eq!(&binary[0..4], &TRACE_MAGIC);
        assert_eq!(binary[4], TRACE_FORMAT_VERSION);

        let restored = InferenceTrace::from_binary(&binary).expect("from_binary");
        assert_eq!(restored.trace_id, "bin-001");
        assert_eq!(restored.level, TraceLevel::Summary);
        assert_eq!(restored.prompt_tokens, 20);
        assert_eq!(restored.generated_tokens, 10);
        assert_eq!(restored.seed, 99);
        assert_eq!(restored.layers.len(), 1);
        assert_eq!(restored.sampling_steps.len(), 1);
        assert_eq!(restored.sampling_steps[0].tokens_masked, Some(5));
        assert!(restored.timing.is_some());

        let json = serde_json::to_string(&trace).expect("json");
        assert!(binary.len() < json.len());
    }

    #[test]
    fn test_binary_bad_magic() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x01];
        assert!(matches!(
            InferenceTrace::from_binary(&data),
            Err(TraceDecodeError::BadMagic)
        ));
    }

    #[test]
    fn test_binary_too_short() {
        let data = [0x54, 0x50, 0x52];
        assert!(matches!(
            InferenceTrace::from_binary(&data),
            Err(TraceDecodeError::TooShort)
        ));
    }

    #[test]
    fn test_binary_unsupported_version() {
        let data = [0x54, 0x50, 0x52, 0x41, 0xFF];
        assert!(matches!(
            InferenceTrace::from_binary(&data),
            Err(TraceDecodeError::UnsupportedVersion(0xFF))
        ));
    }
}

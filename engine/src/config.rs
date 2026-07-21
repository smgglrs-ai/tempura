use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
    pub tensor_parallel_size: usize,
    pub quantization: Option<QuantConfig>,
    pub use_qk_norm: bool,
    pub sliding_window: Option<usize>,
    pub hidden_act: HiddenAct,
    pub rope_type: RopeType,
    pub layer_kinds: Option<Vec<LayerKind>>,
    pub deltanet_num_heads: Option<usize>,
    pub deltanet_head_dim: Option<usize>,
    pub ssm_a: Option<f32>,
    pub dt_bias: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub format: QuantFormat,
    pub bits: u8,
    pub group_size: usize,
    pub symmetric: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    None,
    Gptq,
    Awq,
    Gguf,
    Tq2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenAct {
    SiluGate,
    SquaredRelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Neox,
    Norm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    DeltaNet,
}

impl QuantConfig {
    pub fn gptq(bits: u8, group_size: usize) -> Self {
        Self {
            format: QuantFormat::Gptq,
            bits,
            group_size,
            symmetric: true,
        }
    }

    pub fn awq(bits: u8, group_size: usize) -> Self {
        Self {
            format: QuantFormat::Awq,
            bits,
            group_size,
            symmetric: false,
        }
    }

    pub fn gguf(bits: u8) -> Self {
        Self {
            format: QuantFormat::Gguf,
            bits,
            group_size: 32,
            symmetric: true,
        }
    }
}

impl ModelConfig {
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_kv_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
            max_seq_len: 4096,
            head_dim: 128,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            tensor_parallel_size: 1,
            quantization: None,
            use_qk_norm: false,
            sliding_window: None,
            hidden_act: HiddenAct::SiluGate,
            rope_type: RopeType::Neox,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
        }
    }

    pub fn micro_llama() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 5632,
            num_attention_heads: 16,
            num_kv_heads: 4,
            num_layers: 12,
            vocab_size: 32000,
            max_seq_len: 2048,
            head_dim: 64,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            tensor_parallel_size: 1,
            quantization: None,
            use_qk_norm: false,
            sliding_window: None,
            hidden_act: HiddenAct::SiluGate,
            rope_type: RopeType::Neox,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
        }
    }

    pub fn tiny_test() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
            vocab_size: 256,
            max_seq_len: 128,
            head_dim: 16,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            tensor_parallel_size: 1,
            quantization: None,
            use_qk_norm: false,
            sliding_window: None,
            hidden_act: HiddenAct::SiluGate,
            rope_type: RopeType::Neox,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
        }
    }

    pub fn from_hf_config(path: &Path) -> Result<Self, ConfigError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_hf_json(&data)
    }

    pub fn from_hf_json(json: &str) -> Result<Self, ConfigError> {
        let v: serde_json::Value = serde_json::from_str(json)?;

        let hidden_size = v["hidden_size"]
            .as_u64()
            .ok_or(ConfigError::MissingField("hidden_size"))? as usize;
        let num_attention_heads = v["num_attention_heads"]
            .as_u64()
            .ok_or(ConfigError::MissingField("num_attention_heads"))?
            as usize;
        let num_kv_heads = v
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize;
        let num_layers = v["num_hidden_layers"]
            .as_u64()
            .ok_or(ConfigError::MissingField("num_hidden_layers"))?
            as usize;
        let intermediate_size = v["intermediate_size"]
            .as_u64()
            .ok_or(ConfigError::MissingField("intermediate_size"))?
            as usize;
        let vocab_size = v["vocab_size"]
            .as_u64()
            .ok_or(ConfigError::MissingField("vocab_size"))? as usize;
        let max_seq_len = v
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let head_dim = v
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let rope_theta = v
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;
        let layer_norm_eps = v
            .get("rms_norm_eps")
            .or_else(|| v.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        let hidden_act = match v.get("hidden_act").and_then(|v| v.as_str()) {
            Some("relu2") => HiddenAct::SquaredRelu,
            _ => HiddenAct::SiluGate,
        };

        let model_type = v.get("model_type").and_then(|v| v.as_str());

        let rope_type = match model_type {
            Some("bitnet") => RopeType::Norm,
            _ => RopeType::Neox,
        };

        let quantization = match model_type {
            Some("bitnet") => Some(QuantConfig {
                format: QuantFormat::Tq2,
                bits: 2,
                group_size: 256,
                symmetric: true,
            }),
            _ => None,
        };

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            num_layers,
            vocab_size,
            max_seq_len,
            head_dim,
            rope_theta,
            layer_norm_eps,
            tensor_parallel_size: 1,
            quantization,
            use_qk_norm: false,
            sliding_window: v
                .get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            hidden_act,
            rope_type,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
        })
    }

    pub fn cache_layer_map(&self) -> Vec<Option<usize>> {
        match &self.layer_kinds {
            None => (0..self.num_layers).map(|i| Some(i)).collect(),
            Some(kinds) => {
                let mut cache_idx = 0;
                kinds
                    .iter()
                    .map(|k| match k {
                        LayerKind::Attention => {
                            let idx = cache_idx;
                            cache_idx += 1;
                            Some(idx)
                        }
                        LayerKind::DeltaNet => None,
                    })
                    .collect()
            }
        }
    }

    pub fn num_attention_layers(&self) -> usize {
        match &self.layer_kinds {
            None => self.num_layers,
            Some(kinds) => kinds.iter().filter(|k| **k == LayerKind::Attention).count(),
        }
    }

    pub fn num_deltanet_layers(&self) -> usize {
        match &self.layer_kinds {
            None => 0,
            Some(kinds) => kinds.iter().filter(|k| **k == LayerKind::DeltaNet).count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_head_dim_consistent(config: &ModelConfig) {
        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads
        );
    }

    fn assert_kv_heads_valid(config: &ModelConfig) {
        assert!(config.num_kv_heads <= config.num_attention_heads);
        assert_eq!(config.num_attention_heads % config.num_kv_heads, 0);
    }

    #[test]
    fn test_llama2_7b_preset() {
        let c = ModelConfig::llama2_7b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_kv_heads, 32);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.vocab_size, 32000);
        assert_head_dim_consistent(&c);
        assert_kv_heads_valid(&c);
        assert!(c.quantization.is_none());
    }

    #[test]
    fn test_micro_llama_preset() {
        let c = ModelConfig::micro_llama();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.num_layers, 12);
        assert_head_dim_consistent(&c);
        assert_kv_heads_valid(&c);
    }

    #[test]
    fn test_tiny_test_preset() {
        let c = ModelConfig::tiny_test();
        assert_eq!(c.hidden_size, 64);
        assert_eq!(c.num_attention_heads, 4);
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.num_layers, 2);
        assert_eq!(c.vocab_size, 256);
        assert_head_dim_consistent(&c);
        assert_kv_heads_valid(&c);
    }

    #[test]
    fn test_quant_config_gptq() {
        let q = QuantConfig::gptq(4, 128);
        assert_eq!(q.format, QuantFormat::Gptq);
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 128);
        assert!(q.symmetric);
    }

    #[test]
    fn test_quant_config_awq() {
        let q = QuantConfig::awq(4, 128);
        assert_eq!(q.format, QuantFormat::Awq);
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 128);
        assert!(!q.symmetric);
    }

    #[test]
    fn test_quant_config_gguf() {
        let q = QuantConfig::gguf(4);
        assert_eq!(q.format, QuantFormat::Gguf);
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 32);
        assert!(q.symmetric);
    }

    #[test]
    fn test_from_hf_json_llama_style() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.vocab_size, 32000);
        assert_eq!(c.max_seq_len, 4096);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.rope_theta, 500000.0);
        assert!(c.quantization.is_none());
        assert_eq!(c.tensor_parallel_size, 1);
        assert_head_dim_consistent(&c);
    }

    #[test]
    fn test_from_hf_json_no_gqa() {
        let json = r#"{
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 30522
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.num_kv_heads, 12);
        assert_eq!(c.max_seq_len, 4096);
        assert_eq!(c.rope_theta, 10000.0);
        assert_eq!(c.layer_norm_eps, 1e-5);
        assert_head_dim_consistent(&c);
        assert_kv_heads_valid(&c);
    }

    #[test]
    fn test_from_hf_json_bitnet() {
        let json = r#"{
            "model_type": "bitnet",
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "num_hidden_layers": 30,
            "vocab_size": 128256,
            "hidden_act": "relu2",
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "rope_theta": 500000.0
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.intermediate_size, 6912);
        assert_eq!(c.num_attention_heads, 20);
        assert_eq!(c.num_kv_heads, 5);
        assert_eq!(c.num_layers, 30);
        assert_eq!(c.vocab_size, 128256);
        assert_eq!(c.hidden_act, HiddenAct::SquaredRelu);
        assert!(c.quantization.is_some());
        let q = c.quantization.unwrap();
        assert_eq!(q.format, QuantFormat::Tq2);
        assert_eq!(q.bits, 2);
    }

    #[test]
    fn test_from_hf_json_missing_required_field() {
        let json = r#"{"hidden_size": 4096}"#;
        let err = ModelConfig::from_hf_json(json).unwrap_err();
        assert!(err.to_string().contains("missing required field"));
    }

    #[test]
    fn test_from_hf_json_malformed() {
        let err = ModelConfig::from_hf_json("not json").unwrap_err();
        assert!(matches!(err, ConfigError::Json(_)));
    }
}

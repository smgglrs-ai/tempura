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
    pub architecture: Architecture,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub use_layernorm: bool,
    pub use_alibi: bool,
    pub alibi_max_bias: f32,
    pub parallel_attn_ffn: bool,
    pub rmsnorm_offset: bool,
    pub attn_logit_softcapping: Option<f32>,
    pub embedding_scale: bool,
    pub layer_kinds: Option<Vec<LayerKind>>,
    pub deltanet_num_heads: Option<usize>,
    pub deltanet_head_dim: Option<usize>,
    pub ssm_a: Option<f32>,
    pub dt_bias: Option<Vec<f32>>,
    pub rope_dim: Option<usize>,
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
    Gelu,
    GeluQuick,
    Relu,
    SquaredRelu,
    GeGLU,
    ReGLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Gpt2,
    Phi,
    Gemma,
    Falcon,
    Mpt,
}

impl Architecture {
    pub fn from_gguf_name(name: &str) -> Self {
        match name {
            "llama" | "llama4" | "qwen" | "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe"
            | "mistral" | "yi" | "deci" | "baichuan" | "stablelm" | "bitnet"
            | "internlm2" | "deepseek" | "deepseek2" => Architecture::Llama,
            "gpt2" | "gptj" | "gptneox" | "starcoder" | "refact" => Architecture::Gpt2,
            "phi2" | "phi3" | "phimoe" => Architecture::Phi,
            "gemma" | "gemma2" | "gemma3" => Architecture::Gemma,
            "falcon" => Architecture::Falcon,
            "mpt" => Architecture::Mpt,
            _ => Architecture::Llama,
        }
    }
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
            architecture: Architecture::Llama,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
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
            architecture: Architecture::Llama,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
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
            architecture: Architecture::Llama,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
        }
    }

    pub fn tiny_gemma() -> Self {
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
            layer_norm_eps: 1e-6,
            tensor_parallel_size: 1,
            quantization: None,
            use_qk_norm: false,
            sliding_window: None,
            hidden_act: HiddenAct::GeGLU,
            rope_type: RopeType::Neox,
            architecture: Architecture::Gemma,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: false,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: false,
            rmsnorm_offset: true,
            attn_logit_softcapping: None,
            embedding_scale: true,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
        }
    }

    pub fn tiny_phi() -> Self {
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
            hidden_act: HiddenAct::Gelu,
            rope_type: RopeType::Neox,
            architecture: Architecture::Phi,
            n_experts: 0,
            n_experts_used: 0,
            use_layernorm: true,
            use_alibi: false,
            alibi_max_bias: 0.0,
            parallel_attn_ffn: true,
            rmsnorm_offset: false,
            attn_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
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
            Some("gelu") | Some("gelu_new") => HiddenAct::Gelu,
            Some("gelu_fast") | Some("gelu_quick") => HiddenAct::GeluQuick,
            Some("relu") => HiddenAct::Relu,
            Some("geglu") => HiddenAct::GeGLU,
            _ => HiddenAct::SiluGate,
        };

        let model_type = v.get("model_type").and_then(|v| v.as_str());

        let architecture = match model_type {
            Some(mt) => Architecture::from_gguf_name(mt),
            None => Architecture::Llama,
        };

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
            architecture,
            n_experts: v.get("num_local_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            n_experts_used: v.get("num_experts_per_tok").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            use_layernorm: matches!(architecture, Architecture::Falcon | Architecture::Mpt | Architecture::Phi),
            use_alibi: matches!(architecture, Architecture::Mpt),
            alibi_max_bias: if matches!(architecture, Architecture::Mpt) {
                v.get("attn_config").and_then(|c| c.get("alibi_bias_max")).and_then(|v| v.as_f64()).unwrap_or(8.0) as f32
            } else {
                0.0
            },
            parallel_attn_ffn: matches!(architecture, Architecture::Phi) || v.get("parallel_attn").and_then(|v| v.as_bool()).unwrap_or(false),
            rmsnorm_offset: matches!(architecture, Architecture::Gemma),
            attn_logit_softcapping: v
                .get("attn_logit_softcapping")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            embedding_scale: matches!(architecture, Architecture::Gemma),
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
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

    #[test]
    fn test_architecture_from_gguf_name() {
        assert_eq!(Architecture::from_gguf_name("llama"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_name("qwen2"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_name("qwen3"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_name("mistral"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_name("bitnet"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_name("gpt2"), Architecture::Gpt2);
        assert_eq!(Architecture::from_gguf_name("gptj"), Architecture::Gpt2);
        assert_eq!(Architecture::from_gguf_name("gptneox"), Architecture::Gpt2);
        assert_eq!(Architecture::from_gguf_name("starcoder"), Architecture::Gpt2);
        assert_eq!(Architecture::from_gguf_name("phi2"), Architecture::Phi);
        assert_eq!(Architecture::from_gguf_name("phi3"), Architecture::Phi);
        assert_eq!(Architecture::from_gguf_name("gemma"), Architecture::Gemma);
        assert_eq!(Architecture::from_gguf_name("gemma2"), Architecture::Gemma);
        assert_eq!(Architecture::from_gguf_name("falcon"), Architecture::Falcon);
        assert_eq!(Architecture::from_gguf_name("mpt"), Architecture::Mpt);
        assert_eq!(Architecture::from_gguf_name("unknown_arch"), Architecture::Llama);
    }

    #[test]
    fn test_from_hf_json_gpt2_hidden_act() {
        let json = r#"{
            "model_type": "gpt2",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 50257,
            "hidden_act": "gelu_new"
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.hidden_act, HiddenAct::Gelu);
        assert_eq!(c.architecture, Architecture::Gpt2);
    }

    #[test]
    fn test_from_hf_json_gemma_hidden_act() {
        let json = r#"{
            "model_type": "gemma",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "vocab_size": 256000,
            "hidden_act": "geglu"
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.hidden_act, HiddenAct::GeGLU);
        assert_eq!(c.architecture, Architecture::Gemma);
    }

    #[test]
    fn test_tiny_gemma_preset() {
        let c = ModelConfig::tiny_gemma();
        assert_eq!(c.architecture, Architecture::Gemma);
        assert_eq!(c.hidden_act, HiddenAct::GeGLU);
        assert!(!c.use_layernorm);
        assert!(c.rmsnorm_offset);
        assert!(c.embedding_scale);
        assert!(!c.parallel_attn_ffn);
        assert!(c.attn_logit_softcapping.is_none());
    }

    #[test]
    fn test_tiny_phi_preset() {
        let c = ModelConfig::tiny_phi();
        assert_eq!(c.architecture, Architecture::Phi);
        assert_eq!(c.hidden_act, HiddenAct::Gelu);
        assert!(c.use_layernorm);
        assert!(!c.rmsnorm_offset);
        assert!(c.parallel_attn_ffn);
    }

    #[test]
    fn test_from_hf_json_gemma_arch_fields() {
        let json = r#"{
            "model_type": "gemma",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "vocab_size": 256000,
            "hidden_act": "geglu"
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert!(c.rmsnorm_offset, "Gemma must have rmsnorm_offset");
        assert!(c.embedding_scale, "Gemma must have embedding_scale");
        assert!(!c.parallel_attn_ffn, "Gemma must NOT have parallel_attn_ffn");
        assert!(!c.use_layernorm);
    }

    #[test]
    fn test_from_hf_json_gemma2_softcapping() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "vocab_size": 256000,
            "hidden_act": "geglu",
            "attn_logit_softcapping": 50.0
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, Architecture::Gemma);
        assert!(c.rmsnorm_offset);
        assert_eq!(c.attn_logit_softcapping, Some(50.0));
    }

    #[test]
    fn test_presets_have_llama_architecture() {
        assert_eq!(ModelConfig::llama2_7b().architecture, Architecture::Llama);
        assert_eq!(ModelConfig::micro_llama().architecture, Architecture::Llama);
        assert_eq!(ModelConfig::tiny_test().architecture, Architecture::Llama);
    }

    #[test]
    fn test_cache_layer_map_all_attention() {
        let cfg = ModelConfig::tiny_test();
        let map = cfg.cache_layer_map();
        assert_eq!(map, vec![Some(0), Some(1)]);
        assert_eq!(cfg.num_attention_layers(), 2);
        assert_eq!(cfg.num_deltanet_layers(), 0);
    }

    #[test]
    fn test_cache_layer_map_hybrid() {
        let mut cfg = ModelConfig::tiny_test();
        cfg.num_layers = 4;
        cfg.layer_kinds = Some(vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
        ]);
        let map = cfg.cache_layer_map();
        assert_eq!(map, vec![None, None, None, Some(0)]);
        assert_eq!(cfg.num_attention_layers(), 1);
        assert_eq!(cfg.num_deltanet_layers(), 3);
    }

    #[test]
    fn test_cache_layer_map_interleaved() {
        let mut cfg = ModelConfig::tiny_test();
        cfg.num_layers = 8;
        cfg.layer_kinds = Some(vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
        ]);
        let map = cfg.cache_layer_map();
        assert_eq!(
            map,
            vec![None, None, None, Some(0), None, None, None, Some(1)]
        );
        assert_eq!(cfg.num_attention_layers(), 2);
        assert_eq!(cfg.num_deltanet_layers(), 6);
    }
}

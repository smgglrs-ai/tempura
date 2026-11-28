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
    pub use_f16_kv: bool,
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
    /// Gemma-2 final logit softcapping: logits = tanh(logits / cap) * cap
    /// (GGUF {arch}.final_logit_softcapping, HF final_logit_softcapping).
    /// Applied after the lm_head projection; greedy argmax is invariant
    /// under the monotonic map, but log-probs are not.
    pub final_logit_softcapping: Option<f32>,
    pub embedding_scale: bool,
    pub layer_kinds: Option<Vec<LayerKind>>,
    pub deltanet_num_heads: Option<usize>,
    pub deltanet_head_dim: Option<usize>,
    pub ssm_a: Option<f32>,
    pub dt_bias: Option<Vec<f32>>,
    pub rope_dim: Option<usize>,
    /// Gemma-3 / Llama-4 interleaved attention pattern: layer `il` is
    /// sliding-window iff `il % pattern < pattern - 1` (llama.cpp
    /// `set_swa_pattern(n, dense_first=false)`; GGUF
    /// {arch}.attention.sliding_window_pattern, HF sliding_window_pattern).
    /// Gemma-3: 6 (5 SWA + 1 global). Llama-4: 4 (3 chunked + 1 full).
    pub sliding_window_pattern: Option<usize>,
    /// RoPE base for sliding-window layers (Gemma-3: 10000 local vs 1e6
    /// global; GGUF {arch}.rope.freq_base_swa, HF rope_local_base_freq).
    pub rope_theta_swa: Option<f32>,
    /// Attention scale denominator: scale = 1/sqrt(this) instead of
    /// 1/sqrt(head_dim). Gemma-3 query_pre_attn_scalar (12B/27B use 256
    /// with head_dim 256/128... llama.cpp derives it by model type:
    /// 27B -> 1/sqrt(hidden/heads), else 1/sqrt(head_dim)).
    pub attn_scale_denominator: Option<f32>,
    /// Llama-4 no-RoPE layer step: layer `il` skips RoPE iff
    /// `(il + 1) % step == 0` (llama.cpp rope_pattern). Parsed but NOT yet
    /// wired into the forward paths (recorded gap: Llama-4 also needs
    /// chunked attention + interleaved MoE).
    pub no_rope_layer_step: Option<usize>,
    /// RoPE mscale (llama.cpp rope_attn_factor / YaRN mscale): multiplies
    /// the rope cos/sin, scaling rotated Q and K. Read UNCONDITIONALLY from
    /// {arch}.rope.scaling.attn_factor -- Phi longrope applies 1.1902 even
    /// with no rope.scaling.type key (llama.cpp reads it the same way;
    /// previously parsed into RopeScalingConfig only when a scaling type
    /// existed and was then never applied).
    pub rope_attn_factor: f32,
    pub rope_scaling: Option<RopeScalingConfig>,
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
            | "mistral" | "yi" | "deci" | "baichuan" | "stablelm" | "bitnet" | "internlm2"
            | "deepseek" | "deepseek2" => Architecture::Llama,
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
pub enum RopeScalingType {
    Linear,
    NTK,
    YaRN,
    LLaMA3,
}

#[derive(Debug, Clone)]
pub struct RopeScalingConfig {
    pub scaling_type: RopeScalingType,
    pub factor: f32,
    pub original_context_length: u32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_attn_factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
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
            use_f16_kv: false,
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
            final_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern: None,
            rope_theta_swa: None,
            attn_scale_denominator: None,
            no_rope_layer_step: None,
            rope_attn_factor: 1.0,
            rope_scaling: None,
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
            use_f16_kv: false,
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
            final_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern: None,
            rope_theta_swa: None,
            attn_scale_denominator: None,
            no_rope_layer_step: None,
            rope_attn_factor: 1.0,
            rope_scaling: None,
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
            use_f16_kv: false,
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
            final_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern: None,
            rope_theta_swa: None,
            attn_scale_denominator: None,
            no_rope_layer_step: None,
            rope_attn_factor: 1.0,
            rope_scaling: None,
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
            use_f16_kv: false,
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
            final_logit_softcapping: None,
            embedding_scale: true,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern: None,
            rope_theta_swa: None,
            attn_scale_denominator: None,
            no_rope_layer_step: None,
            rope_attn_factor: 1.0,
            rope_scaling: None,
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
            use_f16_kv: false,
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
            final_logit_softcapping: None,
            embedding_scale: false,
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern: None,
            rope_theta_swa: None,
            attn_scale_denominator: None,
            no_rope_layer_step: None,
            rope_attn_factor: 1.0,
            rope_scaling: None,
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
            // Qwen3-MoE carries only the per-expert size
            .or_else(|| v["moe_intermediate_size"].as_u64())
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
            Some("bitnet") | Some("llama4") => RopeType::Norm,
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

        let rope_scaling = v.get("rope_scaling").and_then(|rs| {
            let scaling_type = match rs.get("type").and_then(|t| t.as_str())? {
                "linear" => RopeScalingType::Linear,
                "ntk" | "dynamic" => RopeScalingType::NTK,
                "yarn" => RopeScalingType::YaRN,
                "llama3" => RopeScalingType::LLaMA3,
                _ => return None,
            };
            let factor = rs.get("factor").and_then(|f| f.as_f64()).unwrap_or(1.0) as f32;
            let original_context_length = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(max_seq_len as u64) as u32;
            Some(RopeScalingConfig {
                scaling_type,
                factor,
                original_context_length,
                yarn_beta_fast: rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32,
                yarn_beta_slow: rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                yarn_attn_factor: rs
                    .get("attn_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                low_freq_factor: rs
                    .get("low_freq_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                high_freq_factor: rs
                    .get("high_freq_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(4.0) as f32,
            })
        });

        // Qwen3: per-head Q/K RMSNorm on every layer, no attention bias,
        // SiLU FFN, NEOX rope pairing (same as qwen2). MoE variants carry
        // their expert counts under moe-specific keys with a per-expert
        // intermediate size.
        let use_qk_norm = match model_type {
            Some("qwen3") | Some("qwen3moe") => true,
            _ => false,
        };

        let (n_experts, n_experts_used, intermediate_size) = match model_type {
            Some("qwen3moe") => (
                v.get("num_experts")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                v.get("num_experts_per_tok")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                v.get("moe_intermediate_size")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(intermediate_size),
            ),
            _ => (0, 0, intermediate_size),
        };

        // Gemma-3: interleaved 5:1 sliding/global layers with a separate
        // local rope base, and the pre-attention query scale. llama.cpp
        // derives the scale denominator by model type (27B:
        // hidden/heads, else head_dim); HF carries it explicitly as
        // query_pre_attn_scalar.
        let sliding_window_pattern = match model_type {
            Some("gemma3") => v
                .get("sliding_window_pattern")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            Some("llama4") => Some(4),
            _ => None,
        };
        let rope_theta_swa = match model_type {
            Some("gemma3") => v
                .get("rope_local_base_freq")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            _ => None,
        };
        let attn_scale_denominator = match model_type {
            Some("gemma3") => {
                let scalar = v
                    .get("query_pre_attn_scalar")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as f32);
                scalar.or({
                    // llama.cpp rule when the HF key is absent (GGUF loads):
                    // 27B -> hidden/heads, else head_dim.
                    if num_layers == 62 {
                        Some(hidden_size as f32 / num_attention_heads as f32)
                    } else {
                        Some(head_dim as f32)
                    }
                })
            }
            _ => None,
        };
        let no_rope_layer_step = match model_type {
            Some("llama4") => Some(4),
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
            use_qk_norm,
            use_f16_kv: false,
            sliding_window: v
                .get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            hidden_act,
            rope_type,
            architecture,
            n_experts,
            n_experts_used,
            use_layernorm: matches!(
                architecture,
                Architecture::Falcon | Architecture::Mpt | Architecture::Phi
            ),
            use_alibi: matches!(architecture, Architecture::Mpt),
            alibi_max_bias: if matches!(architecture, Architecture::Mpt) {
                v.get("attn_config")
                    .and_then(|c| c.get("alibi_bias_max"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(8.0) as f32
            } else {
                0.0
            },
            parallel_attn_ffn: matches!(architecture, Architecture::Phi)
                || v.get("parallel_attn")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            rmsnorm_offset: matches!(architecture, Architecture::Gemma),
            attn_logit_softcapping: v
                .get("attn_logit_softcapping")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            final_logit_softcapping: v
                .get("final_logit_softcapping")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            embedding_scale: matches!(architecture, Architecture::Gemma),
            layer_kinds: None,
            deltanet_num_heads: None,
            deltanet_head_dim: None,
            ssm_a: None,
            dt_bias: None,
            rope_dim: None,
            sliding_window_pattern,
            rope_theta_swa,
            attn_scale_denominator,
            no_rope_layer_step,
            rope_attn_factor: 1.0,
            rope_scaling,
        })
    }

    pub fn cache_layer_map(&self) -> Vec<Option<usize>> {
        match &self.layer_kinds {
            None => (0..self.num_layers).map(Some).collect(),
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

    /// Layer `il` uses a sliding attention window (Gemma-3/Llama-4
    /// interleave). A plain `sliding_window` with no pattern applies to
    /// every layer.
    pub fn layer_is_sliding(&self, layer: usize) -> bool {
        if self.sliding_window.is_none() {
            return false;
        }
        match self.sliding_window_pattern {
            None => true,
            Some(n) if n > 0 => layer % n < n - 1,
            Some(_) => true,
        }
    }

    /// Effective sliding window for layer `il` (None = full attention).
    pub fn layer_sliding_window(&self, layer: usize) -> Option<usize> {
        if self.layer_is_sliding(layer) {
            self.sliding_window
        } else {
            None
        }
    }

    /// Effective RoPE base for layer `il` (SWA layers carry their own base
    /// on Gemma-3: 10000 local vs 1e6 global).
    pub fn layer_rope_theta(&self, layer: usize) -> f32 {
        if self.layer_is_sliding(layer) {
            self.rope_theta_swa.unwrap_or(self.rope_theta)
        } else {
            self.rope_theta
        }
    }

    /// Effective attention scale for layer `il`.
    pub fn layer_attn_scale(&self, head_dim: usize) -> f32 {
        let denom = self
            .attn_scale_denominator
            .unwrap_or(head_dim as f32);
        1.0 / denom.sqrt()
    }

    /// Layer `il` applies RoPE (Llama-4 skips every `step`-th layer).
    pub fn layer_uses_rope(&self, layer: usize) -> bool {
        match self.no_rope_layer_step {
            Some(step) if step > 0 => (layer + 1) % step != 0,
            _ => true,
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
        assert_eq!(
            Architecture::from_gguf_name("starcoder"),
            Architecture::Gpt2
        );
        assert_eq!(Architecture::from_gguf_name("phi2"), Architecture::Phi);
        assert_eq!(Architecture::from_gguf_name("phi3"), Architecture::Phi);
        assert_eq!(Architecture::from_gguf_name("gemma"), Architecture::Gemma);
        assert_eq!(Architecture::from_gguf_name("gemma2"), Architecture::Gemma);
        assert_eq!(Architecture::from_gguf_name("falcon"), Architecture::Falcon);
        assert_eq!(Architecture::from_gguf_name("mpt"), Architecture::Mpt);
        assert_eq!(
            Architecture::from_gguf_name("unknown_arch"),
            Architecture::Llama
        );
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
        assert!(
            !c.parallel_attn_ffn,
            "Gemma must NOT have parallel_attn_ffn"
        );
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
    fn test_from_hf_json_qwen3() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, Architecture::Llama);
        assert_eq!(c.rope_type, RopeType::Neox);
        assert!(c.use_qk_norm, "Qwen3 must enable qk_norm");
        assert_eq!(c.n_experts, 0);
        assert_eq!(c.head_dim, 128);
    }

    #[test]
    fn test_from_hf_json_qwen3moe() {
        let json = r#"{
            "model_type": "qwen3moe",
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "norm_topk_prob": true
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, Architecture::Llama);
        assert!(c.use_qk_norm);
        assert_eq!(c.n_experts, 128);
        assert_eq!(c.n_experts_used, 8);
        assert_eq!(c.intermediate_size, 768, "moe_intermediate_size fallback");
    }

    #[test]
    fn test_from_hf_json_gemma3_interleave() {
        let json = r#"{
            "model_type": "gemma3",
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_hidden_layers": 26,
            "vocab_size": 262144,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "hidden_act": "geglu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 1024,
            "sliding_window_pattern": 6
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, Architecture::Gemma);
        assert_eq!(c.sliding_window, Some(1024));
        assert_eq!(c.sliding_window_pattern, Some(6));
        assert_eq!(c.rope_theta_swa, Some(10000.0));
        assert_eq!(c.attn_scale_denominator, Some(256.0));
        assert_eq!(c.hidden_act, HiddenAct::GeGLU);
        // 5:1 interleave: layers 0-4 SWA, layer 5 global (llama.cpp
        // set_swa_pattern(n, dense_first=false): il % n < n - 1).
        for il in 0..26 {
            let expect_swa = il % 6 < 5;
            assert_eq!(
                c.layer_is_sliding(il),
                expect_swa,
                "layer {il} sliding={expect_swa}"
            );
            assert_eq!(c.layer_sliding_window(il), if expect_swa { Some(1024) } else { None });
        }
        assert_eq!(c.layer_rope_theta(0), 10000.0);
        assert_eq!(c.layer_rope_theta(5), 1000000.0);
        assert!((c.layer_attn_scale(256) - 1.0 / 256.0f32.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_from_hf_json_gemma3_27b_scale_rule() {
        // Without query_pre_attn_scalar, the 62-layer (27B) model uses
        // hidden/heads as the scale denominator (llama.cpp).
        let json = r#"{
            "model_type": "gemma3",
            "hidden_size": 5376,
            "intermediate_size": 28672,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "num_hidden_layers": 62,
            "vocab_size": 262144,
            "head_dim": 128,
            "sliding_window": 1024,
            "sliding_window_pattern": 6
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.attn_scale_denominator, Some(5376.0 / 32.0));
    }

    #[test]
    fn test_from_hf_json_llama4_pattern() {
        let json = r#"{
            "model_type": "llama4",
            "hidden_size": 5120,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "num_hidden_layers": 48,
            "vocab_size": 202048,
            "head_dim": 128,
            "interleave_moe_layer_step": 1,
            "sliding_window": 8192
        }"#;
        let c = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(c.architecture, Architecture::Llama);
        assert_eq!(c.rope_type, RopeType::Norm, "llama4 is a NORM-rope arch");
        assert_eq!(c.no_rope_layer_step, Some(4));
        // llama.cpp rope_pattern: (il + 1) % step != 0 -> layers 3, 7, 11...
        // skip RoPE.
        assert!(c.layer_uses_rope(0));
        assert!(!c.layer_uses_rope(3));
        assert!(c.layer_uses_rope(4));
        assert!(!c.layer_uses_rope(7));
        assert_eq!(c.sliding_window_pattern, Some(4));
        assert!(c.layer_is_sliding(0));
        assert!(!c.layer_is_sliding(3));
    }

    #[test]
    fn test_sliding_window_without_pattern_applies_to_all_layers() {
        let mut c = ModelConfig::tiny_test();
        c.sliding_window = Some(64);
        assert!(c.layer_is_sliding(0));
        assert!(c.layer_is_sliding(1));
        assert_eq!(c.layer_sliding_window(1), Some(64));
        // No window configured at all.
        let plain = ModelConfig::tiny_test();
        assert!(!plain.layer_is_sliding(0));
        assert_eq!(plain.layer_sliding_window(0), None);
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

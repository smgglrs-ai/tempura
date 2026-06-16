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
        }
    }
}

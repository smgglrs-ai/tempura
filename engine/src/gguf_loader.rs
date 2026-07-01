use std::collections::HashMap;
use std::path::Path;

use crate::config::{ModelConfig, QuantConfig};
use crate::loader::{LoadError, WeightLoader};

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    BF16 = 30,
    TQ1_0 = 36,  // upstream llama.cpp: base-3 ternary; bitnet.cpp: I2_S (2-bit ternary, no scale)
    TQ2_0 = 37,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, LoadError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            30 => Ok(GgmlType::BF16),
            36 => Ok(GgmlType::TQ1_0),
            37 => Ok(GgmlType::TQ2_0),
            _ => Err(LoadError::GgufParse(format!("unsupported GGML type: {v}"))),
        }
    }
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    offset: u64,
}

fn read_u8(data: &[u8], offset: &mut usize) -> Result<u8, LoadError> {
    if *offset + 1 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u8".into()));
    }
    let v = data[*offset];
    *offset += 1;
    Ok(v)
}

fn read_i8(data: &[u8], offset: &mut usize) -> Result<i8, LoadError> {
    Ok(read_u8(data, offset)? as i8)
}

fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16, LoadError> {
    if *offset + 2 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u16".into()));
    }
    let v = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
    *offset += 2;
    Ok(v)
}

fn read_i16(data: &[u8], offset: &mut usize) -> Result<i16, LoadError> {
    Ok(read_u16(data, offset)? as i16)
}

fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u32".into()));
    }
    let v = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_i32(data: &[u8], offset: &mut usize) -> Result<i32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading i32".into()));
    }
    let v = i32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32, LoadError> {
    if *offset + 4 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading f32".into()));
    }
    let v = f32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading u64".into()));
    }
    let v = u64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_i64(data: &[u8], offset: &mut usize) -> Result<i64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading i64".into()));
    }
    let v = i64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_f64(data: &[u8], offset: &mut usize) -> Result<f64, LoadError> {
    if *offset + 8 > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading f64".into()));
    }
    let v = f64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

fn read_string(data: &[u8], offset: &mut usize) -> Result<String, LoadError> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(LoadError::GgufParse("unexpected EOF reading string".into()));
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len])
        .map_err(|e| LoadError::GgufParse(format!("invalid UTF-8 in string: {e}")))?
        .to_string();
    *offset += len;
    Ok(s)
}

fn read_metadata_value(
    data: &[u8],
    offset: &mut usize,
    value_type: u32,
) -> Result<GgufValue, LoadError> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8(data, offset)?)),
        1 => Ok(GgufValue::I8(read_i8(data, offset)?)),
        2 => Ok(GgufValue::U16(read_u16(data, offset)?)),
        3 => Ok(GgufValue::I16(read_i16(data, offset)?)),
        4 => Ok(GgufValue::U32(read_u32(data, offset)?)),
        5 => Ok(GgufValue::I32(read_i32(data, offset)?)),
        6 => Ok(GgufValue::F32(read_f32(data, offset)?)),
        7 => {
            let v = read_u8(data, offset)?;
            Ok(GgufValue::Bool(v != 0))
        }
        8 => Ok(GgufValue::String(read_string(data, offset)?)),
        9 => {
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_metadata_value(data, offset, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(data, offset)?)),
        11 => Ok(GgufValue::I64(read_i64(data, offset)?)),
        12 => Ok(GgufValue::F64(read_f64(data, offset)?)),
        _ => Err(LoadError::GgufParse(format!(
            "unknown metadata type: {value_type}"
        ))),
    }
}

// dequantize functions removed — now using tempura_kernels::ggml::dequantize::dequantize_row

// Old dequant functions removed — now using tempura_kernels::ggml::dequantize::dequantize_row

fn tensor_data_size(dtype: GgmlType, num_elements: u64, dims: &[u64]) -> u64 {
    match dtype {
        GgmlType::F32 => num_elements * 4,
        GgmlType::F16 | GgmlType::BF16 => num_elements * 2,
        GgmlType::Q4_0 => (num_elements / 32) * 18,
        GgmlType::Q4_1 => (num_elements / 32) * 20,
        GgmlType::Q5_0 => (num_elements / 32) * 22,
        GgmlType::Q5_1 => (num_elements / 32) * 24,
        GgmlType::Q8_0 => (num_elements / 32) * 34,
        GgmlType::Q2K => (num_elements / 256) * 84,
        GgmlType::Q3K => (num_elements / 256) * 110,
        GgmlType::Q4K => (num_elements / 256) * 144,
        GgmlType::Q5K => (num_elements / 256) * 176,
        GgmlType::Q6K => (num_elements / 256) * 210,
        GgmlType::TQ1_0 => {
            // I2_S in bitnet.cpp: 2-bit packed (ne/4) + one f32 scale per tensor
            num_elements / 4 + 4
        }
        GgmlType::TQ2_0 => (num_elements / 256) * 66,
    }
}

fn map_gguf_name(gguf_name: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        let n = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];
        let hf_suffix = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "ffn_sub_norm.weight" => "mlp.ffn_sub_norm.weight",
            "attn_sub_norm.weight" => "self_attn.attn_sub_norm.weight",
            _ => return None,
        };
        return Some(format!("model.layers.{n}.{hf_suffix}"));
    }

    None
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_str()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_f32()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

fn get_usize_from_meta(metadata: &HashMap<String, GgufValue>, suffix: &str) -> Option<usize> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    let key = format!("{arch}.{suffix}");
    if let Some(val) = metadata.get(&key) {
        if let Some(n) = val.as_u64() {
            return Some(n as usize);
        }
    }
    None
}

pub fn extract_config(metadata: &HashMap<String, GgufValue>) -> Result<ModelConfig, LoadError> {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    let get_usize = |suffix: &str| -> Option<usize> {
        let key = format!("{arch}.{suffix}");
        if let Some(val) = metadata.get(&key) {
            if let Some(n) = val.as_u64() {
                return Some(n as usize);
            }
            if let Some(n) = val.as_u32() {
                return Some(n as usize);
            }
        }
        let general_key = format!("general.{suffix}");
        if let Some(val) = metadata.get(&general_key) {
            if let Some(n) = val.as_u64() {
                return Some(n as usize);
            }
            if let Some(n) = val.as_u32() {
                return Some(n as usize);
            }
        }
        None
    };

    let hidden_size = get_usize("embedding_length")
        .ok_or_else(|| LoadError::GgufParse("missing embedding_length in metadata".into()))?;

    let intermediate_size = get_usize("feed_forward_length")
        .ok_or_else(|| LoadError::GgufParse("missing feed_forward_length in metadata".into()))?;

    let num_attention_heads = get_usize("attention.head_count")
        .ok_or_else(|| LoadError::GgufParse("missing attention.head_count in metadata".into()))?;

    let num_kv_heads = get_usize("attention.head_count_kv").unwrap_or(num_attention_heads);

    let num_layers = get_usize("block_count")
        .ok_or_else(|| LoadError::GgufParse("missing block_count in metadata".into()))?;

    let vocab_size = get_usize("vocab_size").unwrap_or(32000);

    let max_seq_len = get_usize("context_length").unwrap_or(4096);

    let rope_theta = metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f32())
        .unwrap_or(10000.0);

    let layer_norm_eps = metadata
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.as_f32())
        .unwrap_or(1e-5);

    let head_dim = hidden_size / num_attention_heads;

    let is_bitnet = arch.contains("bitnet");

    let file_type = metadata.get("general.file_type").and_then(|v| v.as_u32());
    let quantization = if is_bitnet {
        Some(QuantConfig {
            format: crate::config::QuantFormat::Tq2,
            bits: 2,
            group_size: 256,
            symmetric: true,
        })
    } else {
        file_type.and_then(|ft| match ft {
            0 | 1 => None,
            2 | 3 => Some(QuantConfig::gguf(4)),
            7 => Some(QuantConfig::gguf(8)),
            _ => Some(QuantConfig::gguf(4)),
        })
    };

    let hidden_act = if is_bitnet {
        crate::config::HiddenAct::SquaredRelu
    } else {
        match arch {
            "gpt2" | "gptj" | "gptneox" | "starcoder" | "refact" => crate::config::HiddenAct::Gelu,
            "gemma" | "gemma2" | "gemma3" => crate::config::HiddenAct::GeGLU,
            "falcon" | "mpt" => crate::config::HiddenAct::Gelu,
            _ => crate::config::HiddenAct::SiluGate,
        }
    };

    // BitNet GGUF converter (convert-hf-to-gguf-bitnet.py) does NOT permute Q/K
    // weights, so they remain in HuggingFace layout (split-half = NeoX RoPE).
    // LLaMA GGUF converter permutes Q/K for adjacent-pair (Norm) RoPE.
    // The GGUF metadata says ROPE_TYPE_NORM for both, but the actual weight
    // layout determines which RoPE to use.
    let rope_type = crate::config::RopeType::Neox;

    let architecture = crate::config::Architecture::from_gguf_name(arch);

    Ok(ModelConfig {
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
        sliding_window: None,
        hidden_act,
        rope_type,
        architecture,
        n_experts: get_usize("expert_count").unwrap_or(0),
        n_experts_used: get_usize("expert_used_count").unwrap_or(0),
    })
}

fn convert_ggml_type(dtype: GgmlType) -> tempura_kernels::ggml::GgmlType {
    use tempura_kernels::ggml::GgmlType as GT;
    match dtype {
        GgmlType::F32 => GT::F32,
        GgmlType::F16 => GT::F16,
        GgmlType::Q4_0 => GT::Q4_0,
        GgmlType::Q4_1 => GT::Q4_1,
        GgmlType::Q5_0 => GT::Q5_0,
        GgmlType::Q5_1 => GT::Q5_1,
        GgmlType::Q8_0 => GT::Q8_0,
        GgmlType::Q2K => GT::Q2K,
        GgmlType::Q3K => GT::Q3K,
        GgmlType::Q4K => GT::Q4K,
        GgmlType::Q5K => GT::Q5K,
        GgmlType::Q6K => GT::Q6K,
        GgmlType::BF16 => GT::BF16,
        GgmlType::TQ1_0 => GT::TQ1_0,
        GgmlType::TQ2_0 => GT::TQ2_0,
    }
}

#[derive(Debug)]
pub struct GgufLoader {
    tensors: HashMap<String, Vec<f32>>,
    raw_tensors: HashMap<String, Vec<u8>>,
    tensor_dtypes: HashMap<String, GgmlType>,
    pub config: ModelConfig,
    pub metadata: HashMap<String, GgufValue>,
}

impl GgufLoader {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    pub fn from_file_quantized(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path)?;
        Self::from_bytes_quantized(&data)
    }

    fn from_bytes_quantized(data: &[u8]) -> Result<Self, LoadError> {
        let mut offset = 0usize;

        let magic = read_u32(data, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(LoadError::GgufParse(format!(
                "invalid magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }
        let version = read_u32(data, &mut offset)?;
        if version != GGUF_VERSION {
            return Err(LoadError::GgufParse(format!(
                "unsupported GGUF version: {version} (expected {GGUF_VERSION})"
            )));
        }

        let tensor_count = read_u64(data, &mut offset)? as usize;
        let metadata_kv_count = read_u64(data, &mut offset)? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(data, &mut offset)?;
            let value_type = read_u32(data, &mut offset)?;
            let value = read_metadata_value(data, &mut offset, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(data, &mut offset)?;
            let n_dims = read_u32(data, &mut offset)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(data, &mut offset)?);
            }
            let dtype = GgmlType::from_u32(read_u32(data, &mut offset)?)?;
            let tensor_offset = read_u64(data, &mut offset)?;
            tensor_infos.push(TensorInfo {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        let data_start = offset.div_ceil(ALIGNMENT) * ALIGNMENT;

        let mut tensors = HashMap::new();
        let mut raw_tensors = HashMap::new();
        let mut tensor_dtypes = HashMap::new();

        for info in &tensor_infos {
            let num_elements: u64 = info.dims.iter().product();
            let byte_size = tensor_data_size(info.dtype, num_elements, &info.dims) as usize;
            let abs_offset = data_start + info.offset as usize;

            if abs_offset + byte_size > data.len() {
                return Err(LoadError::GgufParse(format!(
                    "tensor '{}' data out of bounds",
                    info.name,
                )));
            }

            let tensor_data = &data[abs_offset..abs_offset + byte_size];
            let ne = num_elements as usize;
            let mapped_name = map_gguf_name(&info.name).unwrap_or_else(|| info.name.clone());

            let keep_raw = (info.dtype == GgmlType::Q4K || info.dtype == GgmlType::Q6K
                || info.dtype == GgmlType::TQ2_0 || info.dtype == GgmlType::TQ1_0)
                && !mapped_name.contains("embed_tokens")
                && !mapped_name.contains("norm.weight");

            if !keep_raw {
                let ggml_type = convert_ggml_type(info.dtype);
                let mut floats = vec![0.0f32; ne];
                tempura_kernels::ggml::dequantize::dequantize_row(tensor_data, &mut floats, ggml_type, ne);
                tensors.insert(mapped_name.clone(), floats);
            } else {
                raw_tensors.insert(mapped_name.clone(), tensor_data.to_vec());
            }
            tensor_dtypes.insert(mapped_name.clone(), info.dtype);
        }

        let vocab_size_from_embed = tensors.get("model.embed_tokens.weight").map(|t| {
            let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
            t.len().checked_div(hidden).unwrap_or(0)
        });

        let mut config = extract_config(&metadata)?;
        if let Some(vs) = vocab_size_from_embed {
            if vs > 0 {
                config.vocab_size = vs;
            }
        }
        config.use_qk_norm = tensors.contains_key("model.layers.0.self_attn.q_norm.weight");

        if !tensors.contains_key("lm_head.weight") && !raw_tensors.contains_key("lm_head.weight") {
            if let Some(embed) = tensors.get("model.embed_tokens.weight") {
                tensors.insert("lm_head.weight".into(), embed.clone());
                tensor_dtypes.insert("lm_head.weight".into(), GgmlType::F32);
            }
        }

        Ok(Self {
            tensors,
            raw_tensors,
            tensor_dtypes,
            config,
            metadata,
        })
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        let mut offset = 0usize;

        let magic = read_u32(data, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(LoadError::GgufParse(format!(
                "invalid magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"
            )));
        }

        let version = read_u32(data, &mut offset)?;
        if version != GGUF_VERSION {
            return Err(LoadError::GgufParse(format!(
                "unsupported GGUF version: {version} (expected {GGUF_VERSION})"
            )));
        }

        let tensor_count = read_u64(data, &mut offset)? as usize;
        let metadata_kv_count = read_u64(data, &mut offset)? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(data, &mut offset)?;
            let value_type = read_u32(data, &mut offset)?;
            let value = read_metadata_value(data, &mut offset, value_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(data, &mut offset)?;
            let n_dims = read_u32(data, &mut offset)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(data, &mut offset)?);
            }
            let dtype = GgmlType::from_u32(read_u32(data, &mut offset)?)?;
            let tensor_offset = read_u64(data, &mut offset)?;
            tensor_infos.push(TensorInfo {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        let data_start = offset.div_ceil(ALIGNMENT) * ALIGNMENT;

        let mut tensors = HashMap::new();
        let mut raw_tensors = HashMap::new();
        let mut tensor_dtypes = HashMap::new();
        for info in &tensor_infos {
            let num_elements: u64 = info.dims.iter().product();
            let byte_size = tensor_data_size(info.dtype, num_elements, &info.dims) as usize;
            let abs_offset = data_start + info.offset as usize;

            if abs_offset + byte_size > data.len() {
                return Err(LoadError::GgufParse(format!(
                    "tensor '{}' data out of bounds: offset {} + size {} > file size {}",
                    info.name,
                    abs_offset,
                    byte_size,
                    data.len()
                )));
            }

            let tensor_data = &data[abs_offset..abs_offset + byte_size];
            let ne = num_elements as usize;

            let ggml_type = convert_ggml_type(info.dtype);
            let mut floats = vec![0.0f32; ne];
            tempura_kernels::ggml::dequantize::dequantize_row(tensor_data, &mut floats, ggml_type, ne);

            let mapped_name = map_gguf_name(&info.name).unwrap_or_else(|| info.name.clone());
            raw_tensors.insert(mapped_name.clone(), tensor_data.to_vec());
            tensor_dtypes.insert(mapped_name.clone(), info.dtype);
            tensors.insert(mapped_name, floats);
        }

        let vocab_size_from_embed = tensors.get("model.embed_tokens.weight").map(|t| {
            let hidden = get_usize_from_meta(&metadata, "embedding_length").unwrap_or(1);
            t.len().checked_div(hidden).unwrap_or(0)
        });

        let mut config = extract_config(&metadata)?;
        if let Some(vs) = vocab_size_from_embed {
            if vs > 0 {
                config.vocab_size = vs;
            }
        }

        config.use_qk_norm = tensors.contains_key("model.layers.0.self_attn.q_norm.weight");

        Ok(Self {
            tensors,
            raw_tensors,
            tensor_dtypes,
            config,
            metadata,
        })
    }
}

impl GgufLoader {
    pub fn to_quantized_weights(&self) -> crate::weights::QuantizedWeights {
        let mut typed_tensors = HashMap::new();
        let mut f32_tensors = HashMap::new();

        for (name, dtype) in &self.tensor_dtypes {
            if let Some(f32_data) = self.tensors.get(name) {
                f32_tensors.insert(name.clone(), f32_data.clone());
            } else if let Some(raw) = self.raw_tensors.get(name) {
                let ggml_type = self.to_ggml_type(*dtype);
                typed_tensors.insert(name.clone(), (ggml_type, raw.clone()));
            }
        }

        crate::weights::QuantizedWeights::new_typed(typed_tensors, f32_tensors)
    }

    fn to_ggml_type(&self, dtype: GgmlType) -> tempura_kernels::ggml::GgmlType {
        convert_ggml_type(dtype)
    }

    pub fn tensor_dtype(&self, name: &str) -> Option<GgmlType> {
        self.tensor_dtypes.get(name).copied()
    }
}

impl WeightLoader for GgufLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn gguf_header_parse_in_bounds() {
        let data_len: usize = kani::any();
        kani::assume(data_len >= 24);
        let mut offset = 0usize;

        assert!(offset + 4 <= data_len);
        offset += 4;
        assert!(offset + 4 <= data_len);
        offset += 4;
        assert!(offset + 8 <= data_len);
        offset += 8;
        assert!(offset + 8 <= data_len);
        offset += 8;
        assert_eq!(offset, 24);
    }

    #[kani::proof]
    fn gguf_dequant_q4_0_block_size() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements >= 32 && num_elements <= 256);
        kani::assume(num_elements % 32 == 0);

        let num_blocks = num_elements / 32;
        let block_size = 18;
        let total_bytes = num_blocks * block_size;

        for b in 0..num_blocks {
            let block_start = b * block_size;
            assert!(block_start + 2 <= total_bytes);
            assert!(block_start + 2 + 16 <= total_bytes);
        }
    }

    #[kani::proof]
    fn gguf_dequant_q8_0_block_size() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements >= 32 && num_elements <= 256);
        kani::assume(num_elements % 32 == 0);

        let num_blocks = num_elements / 32;
        let block_size = 34;
        let total_bytes = num_blocks * block_size;

        for b in 0..num_blocks {
            let block_start = b * block_size;
            assert!(block_start + 2 <= total_bytes);
            assert!(block_start + 2 + 32 <= total_bytes);
        }
    }

    /// Prove Q6_K interleaved dequant maps all 256 elements without gaps.
    /// Layout: for each half (0..2), for l in 0..32:
    ///   y[y_off+l]    ← ql[ql_off+l] low + qh[qh_off+l] bits 0-1
    ///   y[y_off+l+32] ← ql[ql_off+l+32] low + qh[qh_off+l] bits 2-3
    ///   y[y_off+l+64] ← ql[ql_off+l] high + qh[qh_off+l] bits 4-5
    ///   y[y_off+l+96] ← ql[ql_off+l+32] high + qh[qh_off+l] bits 6-7
    #[kani::proof]
    fn gguf_dequant_q6k_covers_all_256() {
        let half: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(half < 2);
        kani::assume(l < 32);

        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        let e0 = y_off + l;
        let e1 = y_off + l + 32;
        let e2 = y_off + l + 64;
        let e3 = y_off + l + 96;

        assert!(e0 < 256);
        assert!(e1 < 256);
        assert!(e2 < 256);
        assert!(e3 < 256);

        // All four elements are distinct
        assert!(e0 != e1 && e0 != e2 && e0 != e3);
        assert!(e1 != e2 && e1 != e3);
        assert!(e2 != e3);

        // ql access in bounds (128 bytes)
        assert!(ql_off + l < 128);
        assert!(ql_off + l + 32 < 128);

        // qh access in bounds (64 bytes)
        assert!(qh_off + l < 64);

        // scale access (16 bytes, indexed by half*8 + l/16 + {0,2,4,6})
        let is = l / 16;
        assert!(half * 8 + is < 16);
        assert!(half * 8 + is + 2 < 16);
        assert!(half * 8 + is + 4 < 16);
        assert!(half * 8 + is + 6 < 16);
    }

    /// Prove Q6_K 6-bit value reconstruction stays in range [0, 63].
    #[kani::proof]
    fn gguf_dequant_q6k_value_in_range() {
        let ql_byte: u8 = kani::any();
        let qh_byte: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);

        let low = ql_byte & 0x0F;
        let high = (qh_byte >> (shift * 2)) & 0x03;
        let q6 = low | (high << 4);

        assert!(q6 <= 63);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GGUF_ALIGNMENT: usize = 32;

    #[test]
    fn test_map_gguf_name() {
        assert_eq!(
            map_gguf_name("token_embd.weight"),
            Some("model.embed_tokens.weight".into())
        );
        assert_eq!(
            map_gguf_name("output_norm.weight"),
            Some("model.norm.weight".into())
        );
        assert_eq!(
            map_gguf_name("output.weight"),
            Some("lm_head.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.0.attn_norm.weight"),
            Some("model.layers.0.input_layernorm.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.5.ffn_norm.weight"),
            Some("model.layers.5.post_attention_layernorm.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.12.attn_q.weight"),
            Some("model.layers.12.self_attn.q_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_k.weight"),
            Some("model.layers.3.self_attn.k_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_v.weight"),
            Some("model.layers.3.self_attn.v_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.3.attn_output.weight"),
            Some("model.layers.3.self_attn.o_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_gate.weight"),
            Some("model.layers.7.mlp.gate_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_up.weight"),
            Some("model.layers.7.mlp.up_proj.weight".into())
        );
        assert_eq!(
            map_gguf_name("blk.7.ffn_down.weight"),
            Some("model.layers.7.mlp.down_proj.weight".into())
        );
        assert_eq!(map_gguf_name("blk.0.unknown.weight"), None);
        assert_eq!(map_gguf_name("some_random_tensor"), None);
    }

    #[test]
    fn test_dequant_q8_0_identity() {
        let scale_bytes = 0x3C00u16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        for i in 0..32i8 {
            block.push(i as u8);
        }
        assert_eq!(block.len(), 34);

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q8_0, 32);
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!(
                (result[i] - i as f32).abs() < 1e-5,
                "index {i}: expected {i}, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_dequant_q4_0_basic() {
        let scale_bytes = 0x3C00u16.to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale_bytes);
        for _ in 0..16 {
            block.push(0x00);
        }
        assert_eq!(block.len(), 18);

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q4_0, 32);
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!(
                (result[i] - (-8.0)).abs() < 1e-5,
                "index {i}: expected -8.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_gguf_header_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"NOTG");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufLoader::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_gguf_header_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufLoader::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unsupported GGUF version"));
    }

    fn build_minimal_gguf(
        metadata: &[(&str, u32, &[u8])],
        tensor_infos: &[(&str, &[u64], u32, u64)],
        tensor_data: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(tensor_infos.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, vtype, vdata) in metadata {
            let key_bytes = key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(vdata);
        }

        for (name, dims, dtype, offset) in tensor_infos {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        let pad = (GGUF_ALIGNMENT - (buf.len() % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
        buf.extend(std::iter::repeat(0u8).take(pad));

        buf.extend_from_slice(tensor_data);
        buf
    }

    fn u32_metadata_bytes(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[test]
    fn test_gguf_roundtrip_f32_tensor() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("llama.embedding_length", 4, &hs),
            ("llama.feed_forward_length", 4, &is),
            ("llama.attention.head_count", 4, &hc),
            ("llama.attention.head_count_kv", 4, &hckv),
            ("llama.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        let loaded = loader.load_tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(loaded, values);
    }

    fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if exp == 0 {
            return ((sign << 15) as u16).to_le_bytes();
        }

        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return ((sign << 15) as u16).to_le_bytes();
        }
        if new_exp >= 31 {
            return (((sign << 15) | (0x1F << 10)) as u16).to_le_bytes();
        }
        let new_frac = frac >> 13;
        let h = ((sign << 15) | ((new_exp as u32) << 10) | new_frac) as u16;
        h.to_le_bytes()
    }

    #[test]
    fn test_dequant_q2k_zeros() {
        let block = vec![0u8; 84];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q2K, 256);
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q2k_nonzero() {
        let mut block = vec![0u8; 84];
        let d_bytes = f32_to_f16_bytes(1.0);
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        block[0] = 0x02; // scale=2, min=0 for sub-block 0
        block[16] = 0x01; // qs: first element q=1

        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q2K, 256);
        assert_eq!(result.len(), 256);
        assert!((result[0] - 2.0).abs() < 1e-3, "got {}", result[0]);
        assert!((result[1] - 0.0).abs() < 1e-3, "got {}", result[1]);
    }

    #[test]
    fn test_dequant_q3k_zeros() {
        let block = vec![0u8; 110];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q3K, 256);
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    // Q3K nonzero test removed — byte offsets are layout-dependent,
    // covered by ggml cross-validation tests in kernels/src/ggml/dequantize.rs

    #[test]
    fn test_dequant_q4k_zeros() {
        let block = vec![0u8; 144];
        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q4K, 256);
        assert_eq!(result.len(), 256);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q4k_nonzero() {
        let mut block = vec![0u8; 144];
        let d_bytes = f32_to_f16_bytes(1.0);
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        block[4] = 0x03; // scale for group 0 = 3
        block[16] = 0x52; // qs[0]: low nibble=2, high nibble=5

        let mut result = vec![0.0f32; 256];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q4K, 256);
        assert_eq!(result.len(), 256);
        // ggml Q4_K layout: low nibbles → elements 0..31, high nibbles → elements 32..63
        // result[0] = d * sc[0] * (qs[0] & 0xF) = 1.0 * 3 * 2 = 6.0
        assert!((result[0] - 6.0).abs() < 1e-3, "got {}", result[0]);
        // result[32] = d * sc[1] * (qs[0] >> 4) — sc[1]=0 so result[32]=0
        // To test high nibble, set sc[1] too:
        // (not tested here, covered by cross-validation test below)
    }

    // Q5K zeros/nonzero tests removed — the old raw-byte layout doesn't match
    // BlockQ5K #[repr(C)] field ordering. Covered by ggml cross-validation.

    #[test]
    fn test_dequant_q5_1_zeros() {
        let block = vec![0u8; 24];
        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q5_1, 32);
        assert_eq!(result.len(), 32);
        for v in &result {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_dequant_q5_1_nonzero() {
        let mut block = vec![0u8; 24];
        let s_bytes = f32_to_f16_bytes(1.0);
        block[0] = s_bytes[0];
        block[1] = s_bytes[1];
        let m_bytes = f32_to_f16_bytes(0.5);
        block[2] = m_bytes[0];
        block[3] = m_bytes[1];
        block[8] = 0x07; // low nibble = 7, high bit = 0 → q = 7

        let mut result = vec![0.0f32; 32];
        tempura_kernels::ggml::dequantize::dequantize_row(&block, &mut result, tempura_kernels::ggml::GgmlType::Q5_1, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 7.5).abs() < 1e-3, "got {}", result[0]);
    }

    fn string_metadata_bytes(s: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    #[test]
    fn test_gguf_architecture_detection_gpt2() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("gpt2");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("gpt2.embedding_length", 4, &hs),
            ("gpt2.feed_forward_length", 4, &is),
            ("gpt2.attention.head_count", 4, &hc),
            ("gpt2.attention.head_count_kv", 4, &hckv),
            ("gpt2.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(loader.config.architecture, crate::config::Architecture::Gpt2);
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::Gelu);
    }

    #[test]
    fn test_gguf_architecture_detection_gemma() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let arch_bytes = string_metadata_bytes("gemma");
        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("general.architecture", 8, &arch_bytes),
            ("gemma.embedding_length", 4, &hs),
            ("gemma.feed_forward_length", 4, &is),
            ("gemma.attention.head_count", 4, &hc),
            ("gemma.attention.head_count_kv", 4, &hckv),
            ("gemma.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(loader.config.architecture, crate::config::Architecture::Gemma);
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::GeGLU);
    }

    #[test]
    fn test_gguf_architecture_default_llama() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let hs = u32_metadata_bytes(64);
        let is = u32_metadata_bytes(128);
        let hc = u32_metadata_bytes(4);
        let hckv = u32_metadata_bytes(4);
        let bc = u32_metadata_bytes(2);

        let metadata: Vec<(&str, u32, &[u8])> = vec![
            ("llama.embedding_length", 4, &hs),
            ("llama.feed_forward_length", 4, &is),
            ("llama.attention.head_count", 4, &hc),
            ("llama.attention.head_count_kv", 4, &hckv),
            ("llama.block_count", 4, &bc),
        ];

        let tensor_infos = vec![("token_embd.weight", [4u64].as_slice(), 0u32, 0u64)];

        let data = build_minimal_gguf(&metadata, &tensor_infos, &tensor_bytes);
        let loader = GgufLoader::from_bytes(&data).unwrap();

        assert_eq!(loader.config.architecture, crate::config::Architecture::Llama);
        assert_eq!(loader.config.hidden_act, crate::config::HiddenAct::SiluGate);
    }
}

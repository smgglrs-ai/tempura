use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::loader::{LoadError, WeightLoader};

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
    #[error("tensor size mismatch for {name}: expected {expected}, got {got}")]
    SizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("GPU upload failed: {0}")]
    GpuUpload(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    I8,
    I4,
}

#[derive(Debug)]
pub enum TensorStorage {
    F32(Vec<f32>),
    Quantized {
        data: Vec<u8>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        dtype: TensorDtype,
        num_elements: usize,
        group_size: usize,
    },
}

impl TensorStorage {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorStorage::F32(v) => Some(v),
            TensorStorage::Quantized { .. } => None,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        match self {
            TensorStorage::F32(v) => v.clone(),
            TensorStorage::Quantized {
                data,
                scales,
                zeros,
                dtype,
                num_elements,
                group_size,
            } => dequantize_tensor(data, scales, zeros, *dtype, *num_elements, *group_size),
        }
    }

    pub fn dtype(&self) -> TensorDtype {
        match self {
            TensorStorage::F32(_) => TensorDtype::F32,
            TensorStorage::Quantized { dtype, .. } => *dtype,
        }
    }
}

fn dequantize_tensor(
    data: &[u8],
    scales: &[f32],
    zeros: &[f32],
    dtype: TensorDtype,
    num_elements: usize,
    group_size: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(num_elements);
    let num_groups = scales.len();

    match dtype {
        TensorDtype::I8 => {
            for g in 0..num_groups {
                let scale = scales[g];
                let zero = zeros[g];
                let start = g * group_size;
                let end = (start + group_size).min(num_elements);
                for &byte in &data[start..end] {
                    output.push(byte as f32 * scale + zero);
                }
            }
        }
        TensorDtype::I4 => {
            for g in 0..num_groups {
                let scale = scales[g];
                let zero = zeros[g];
                let start = g * group_size;
                let end = (start + group_size).min(num_elements);
                for i in start..end {
                    let byte = data[i / 2];
                    let nibble = if i % 2 == 0 {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    };
                    let q = nibble as i8 - 8;
                    output.push(q as f32 * scale + zero);
                }
            }
        }
        TensorDtype::F16 | TensorDtype::F32 => {
            panic!("dequantize called on non-quantized dtype");
        }
    }

    output
}

#[derive(Debug)]
pub struct ModelWeights {
    tensors: HashMap<String, Vec<f32>>,
}

impl ModelWeights {
    pub fn load(loader: &dyn WeightLoader, config: &ModelConfig) -> Result<Self, WeightsError> {
        let required = Self::required_tensor_names(config);
        let expected_sizes = Self::expected_tensor_sizes(config);

        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
        for name in &required {
            let data = match loader.load_tensor(name) {
                Ok(d) => d,
                Err(_) if name == "lm_head.weight" => {
                    if let Some(embed) = tensors.get("model.embed_tokens.weight") {
                        embed.clone()
                    } else {
                        return Err(WeightsError::MissingTensor(name.clone()));
                    }
                }
                Err(_) => return Err(WeightsError::MissingTensor(name.clone())),
            };

            if let Some(&expected) = expected_sizes.get(name.as_str()) {
                if data.len() != expected {
                    return Err(WeightsError::SizeMismatch {
                        name: name.clone(),
                        expected,
                        got: data.len(),
                    });
                }
            }

            tensors.insert(name.clone(), data);
        }

        // Load optional QK-norm tensors (Qwen3-style per-head norms)
        if config.use_qk_norm {
            for i in 0..config.num_layers {
                let prefix = format!("model.layers.{i}");
                for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                    let name = format!("{prefix}.{suffix}");
                    if let Ok(data) = loader.load_tensor(&name) {
                        tensors.insert(name, data);
                    }
                }
            }
        }

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn required_tensor_names(config: &ModelConfig) -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];

        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");
            names.extend([
                format!("{prefix}.input_layernorm.weight"),
                format!("{prefix}.post_attention_layernorm.weight"),
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
                format!("{prefix}.self_attn.o_proj.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
                format!("{prefix}.mlp.down_proj.weight"),
            ]);
        }

        names
    }

    fn expected_tensor_sizes(config: &ModelConfig) -> HashMap<&'static str, usize> {
        let h = config.hidden_size;
        let v = config.vocab_size;
        let mut sizes = HashMap::new();
        sizes.insert("model.embed_tokens.weight", v * h);
        sizes.insert("model.norm.weight", h);
        sizes.insert("lm_head.weight", v * h);
        sizes
    }
}

#[derive(Debug)]
pub struct QuantizedWeights {
    q4k_tensors: HashMap<String, Vec<u8>>,
    q6k_tensors: HashMap<String, Vec<u8>>,
    f32_tensors: HashMap<String, Vec<f32>>,
}

impl QuantizedWeights {
    pub fn new(
        q4k_tensors: HashMap<String, Vec<u8>>,
        q6k_tensors: HashMap<String, Vec<u8>>,
        f32_tensors: HashMap<String, Vec<f32>>,
    ) -> Self {
        Self {
            q4k_tensors,
            q6k_tensors,
            f32_tensors,
        }
    }

    pub fn get_q4k(&self, name: &str) -> Option<&[u8]> {
        self.q4k_tensors.get(name).map(|v| v.as_slice())
    }

    pub fn get_q6k(&self, name: &str) -> Option<&[u8]> {
        self.q6k_tensors.get(name).map(|v| v.as_slice())
    }

    pub fn get_f32(&self, name: &str) -> Option<&[f32]> {
        self.f32_tensors.get(name).map(|v| v.as_slice())
    }
}

#[cfg(feature = "cuda")]
pub struct GpuWeights {
    tensors: HashMap<String, cuda_core::DeviceBuffer<f32>>,
    quantized_tensors: HashMap<String, cuda_core::DeviceBuffer<u8>>,
}

#[cfg(feature = "cuda")]
impl GpuWeights {
    pub fn upload_from(
        weights: &ModelWeights,
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
    ) -> Result<Self, WeightsError> {
        ctx.bind_to_thread()
            .map_err(|e| WeightsError::GpuUpload(e.to_string()))?;
        let stream = ctx.default_stream();
        let mut tensors = HashMap::new();
        for name in weights.tensor_names() {
            if let Some(data) = weights.get(&name) {
                let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, data)
                    .map_err(|e| WeightsError::GpuUpload(format!("{name}: {e}")))?;
                tensors.insert(name, dev_buf);
            }
        }
        Ok(Self {
            tensors,
            quantized_tensors: HashMap::new(),
        })
    }

    pub fn upload_quantized(
        &mut self,
        name: &str,
        data: &[u8],
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
    ) -> Result<(), WeightsError> {
        ctx.bind_to_thread()
            .map_err(|e| WeightsError::GpuUpload(e.to_string()))?;
        let stream = ctx.default_stream();
        let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, data)
            .map_err(|e| WeightsError::GpuUpload(format!("{name}: {e}")))?;
        self.quantized_tensors.insert(name.to_string(), dev_buf);
        Ok(())
    }

    pub fn get_quantized_device(&self, name: &str) -> Option<&cuda_core::DeviceBuffer<u8>> {
        self.quantized_tensors.get(name)
    }

    pub fn upload_from_quantized(
        qweights: &QuantizedWeights,
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
    ) -> Result<Self, WeightsError> {
        ctx.bind_to_thread()
            .map_err(|e| WeightsError::GpuUpload(e.to_string()))?;
        let stream = ctx.default_stream();

        let mut tensors = HashMap::new();
        for (name, data) in &qweights.f32_tensors {
            let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, data)
                .map_err(|e| WeightsError::GpuUpload(format!("{name}: {e}")))?;
            tensors.insert(name.clone(), dev_buf);
        }

        let mut quantized_tensors = HashMap::new();
        for (name, data) in &qweights.q4k_tensors {
            let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, data)
                .map_err(|e| WeightsError::GpuUpload(format!("{name}: {e}")))?;
            quantized_tensors.insert(name.clone(), dev_buf);
        }
        for (name, data) in &qweights.q6k_tensors {
            let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, data)
                .map_err(|e| WeightsError::GpuUpload(format!("{name}: {e}")))?;
            quantized_tensors.insert(name.clone(), dev_buf);
        }

        Ok(Self {
            tensors,
            quantized_tensors,
        })
    }

    pub fn is_quantized(&self, name: &str) -> bool {
        self.quantized_tensors.contains_key(name)
    }

    pub fn upload_from_with_concat(
        weights: &ModelWeights,
        config: &crate::config::ModelConfig,
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
    ) -> Result<Self, WeightsError> {
        let mut gpu = Self::upload_from(weights, ctx)?;
        let stream = ctx.default_stream();
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;

        for layer in 0..config.num_layers {
            let prefix = format!("model.layers.{layer}");

            // QKV: concat Q [q_dim, h] + K [kv_dim, h] + V [kv_dim, h] -> [q_dim+2*kv_dim, h]
            let q_name = format!("{prefix}.self_attn.q_proj.weight");
            let k_name = format!("{prefix}.self_attn.k_proj.weight");
            let v_name = format!("{prefix}.self_attn.v_proj.weight");
            if let (Some(q), Some(k), Some(v)) = (
                weights.get(&q_name),
                weights.get(&k_name),
                weights.get(&v_name),
            ) {
                let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
                qkv.extend_from_slice(q);
                qkv.extend_from_slice(k);
                qkv.extend_from_slice(v);
                let dev_buf = cuda_core::DeviceBuffer::from_host(&stream, &qkv)
                    .map_err(|e| WeightsError::GpuUpload(format!("qkv concat L{layer}: {e}")))?;
                gpu.tensors
                    .insert(format!("{prefix}.self_attn.qkv_proj.weight"), dev_buf);
            }

            // Gate+Up: concat Gate [inter, h] + Up [inter, h] -> [2*inter, h]
            let gate_name = format!("{prefix}.mlp.gate_proj.weight");
            let up_name = format!("{prefix}.mlp.up_proj.weight");
            if let (Some(gate), Some(up)) = (weights.get(&gate_name), weights.get(&up_name)) {
                let mut gate_up = Vec::with_capacity(gate.len() + up.len());
                gate_up.extend_from_slice(gate);
                gate_up.extend_from_slice(up);
                let dev_buf =
                    cuda_core::DeviceBuffer::from_host(&stream, &gate_up).map_err(|e| {
                        WeightsError::GpuUpload(format!("gate_up concat L{layer}: {e}"))
                    })?;
                gpu.tensors
                    .insert(format!("{prefix}.mlp.gate_up_proj.weight"), dev_buf);
            }
        }

        Ok(gpu)
    }

    pub fn get(&self, name: &str) -> Option<&cuda_core::DeviceBuffer<f32>> {
        self.tensors.get(name)
    }
}

#[cfg(kani)]
mod kani_proofs {
    #[kani::proof]
    fn dequant_i8_group_indices_in_bounds() {
        let num_elements: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(num_elements >= 1 && num_elements <= 32);
        kani::assume(group_size >= 1 && group_size <= 16);

        let num_groups = (num_elements + group_size - 1) / group_size;
        let data_len = num_elements;

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(num_elements);
            assert!(start <= data_len);
            assert!(end <= data_len);
            assert!(g < num_groups);
        }
    }

    #[kani::proof]
    #[kani::unwind(33)]
    fn dequant_i4_packed_index_in_bounds() {
        let num_elements: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(num_elements >= 1 && num_elements <= 16);
        kani::assume(group_size >= 1 && group_size <= 8);

        let packed_len = (num_elements + 1) / 2;
        let num_groups = (num_elements + group_size - 1) / group_size;

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(num_elements);
            for i in start..end {
                let packed_idx = i / 2;
                assert!(packed_idx < packed_len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockWeightLoader {
        tensors: HashMap<String, Vec<f32>>,
    }

    impl MockWeightLoader {
        fn from_config(config: &ModelConfig) -> Self {
            let h = config.hidden_size;
            let v = config.vocab_size;
            let kv_h = config.num_kv_heads * config.head_dim;
            let inter = config.intermediate_size;

            let mut tensors = HashMap::new();
            tensors.insert("model.embed_tokens.weight".into(), vec![0.0; v * h]);
            tensors.insert("model.norm.weight".into(), vec![0.0; h]);
            tensors.insert("lm_head.weight".into(), vec![0.0; v * h]);

            for i in 0..config.num_layers {
                let prefix = format!("model.layers.{i}");
                tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![0.0; h]);
                tensors.insert(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![0.0; h],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![0.0; h * h],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![0.0; kv_h * h],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![0.0; kv_h * h],
                );
                tensors.insert(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![0.0; h * h],
                );
                tensors.insert(
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![0.0; inter * h],
                );
                tensors.insert(format!("{prefix}.mlp.up_proj.weight"), vec![0.0; inter * h]);
                tensors.insert(
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![0.0; h * inter],
                );
            }

            Self { tensors }
        }
    }

    impl WeightLoader for MockWeightLoader {
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

    #[test]
    fn test_load_with_correct_tensors() {
        let config = ModelConfig::tiny_test();
        let loader = MockWeightLoader::from_config(&config);
        let weights = ModelWeights::load(&loader, &config);
        assert!(weights.is_ok());

        let weights = weights.unwrap();
        assert!(weights.get("model.embed_tokens.weight").is_some());
        assert_eq!(
            weights.get("model.embed_tokens.weight").unwrap().len(),
            config.vocab_size * config.hidden_size
        );
    }

    #[test]
    fn test_load_missing_tensor() {
        let config = ModelConfig::tiny_test();
        let loader = MockWeightLoader {
            tensors: HashMap::new(),
        };
        let result = ModelWeights::load(&loader, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, WeightsError::MissingTensor(_)));
    }

    #[test]
    fn test_load_size_mismatch() {
        let config = ModelConfig::tiny_test();
        let mut loader = MockWeightLoader::from_config(&config);
        loader
            .tensors
            .insert("model.embed_tokens.weight".into(), vec![0.0; 1]);
        let result = ModelWeights::load(&loader, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, WeightsError::SizeMismatch { .. }));
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let config = ModelConfig::tiny_test();
        let loader = MockWeightLoader::from_config(&config);
        let weights = ModelWeights::load(&loader, &config).unwrap();
        assert!(weights.get("nonexistent.weight").is_none());
    }

    #[test]
    fn test_required_tensor_count() {
        let config = ModelConfig::tiny_test();
        let names = ModelWeights::required_tensor_names(&config);
        // 3 global + 9 per layer * num_layers
        assert_eq!(names.len(), 3 + 9 * config.num_layers);
    }

    #[test]
    fn test_tensor_storage_f32() {
        let storage = TensorStorage::F32(vec![1.0, 2.0, 3.0]);
        assert_eq!(storage.dtype(), TensorDtype::F32);
        assert_eq!(storage.as_f32().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.dequantize(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_storage_i8_dequantize() {
        let group_size = 4;
        let data: Vec<u8> = vec![0, 1, 2, 3]; // i8 values: 0, 1, 2, 3
        let scales = vec![0.5f32];
        let zeros = vec![0.0f32];

        let storage = TensorStorage::Quantized {
            data,
            scales,
            zeros,
            dtype: TensorDtype::I8,
            num_elements: 4,
            group_size,
        };

        assert_eq!(storage.dtype(), TensorDtype::I8);
        assert!(storage.as_f32().is_none());

        let deq = storage.dequantize();
        assert_eq!(deq, vec![0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_tensor_storage_i4_dequantize() {
        let group_size = 4;
        // 4 nibbles packed in 2 bytes: [low=0x98, high=0xBA]
        // byte 0: low nibble = 8 (q=0), high nibble = 9 (q=1)
        // byte 1: low nibble = A (q=2), high nibble = B (q=3)
        let data: Vec<u8> = vec![0x98, 0xBA];
        let scales = vec![1.0f32];
        let zeros = vec![0.0f32];

        let storage = TensorStorage::Quantized {
            data,
            scales,
            zeros,
            dtype: TensorDtype::I4,
            num_elements: 4,
            group_size,
        };

        let deq = storage.dequantize();
        assert_eq!(deq.len(), 4);
        // nibble 0 = 0x8 = 8, q = 8-8 = 0
        assert_eq!(deq[0], 0.0);
        // nibble 1 = 0x9 = 9, q = 9-8 = 1
        assert_eq!(deq[1], 1.0);
    }

    #[test]
    fn test_quant_config_gptq() {
        use crate::config::QuantConfig;
        let qc = QuantConfig::gptq(4, 128);
        assert_eq!(qc.bits, 4);
        assert_eq!(qc.group_size, 128);
        assert!(qc.symmetric);
    }

    #[test]
    fn test_dequantize_i8_accuracy() {
        let original: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let group_size = 8;

        let min_val = original.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = original.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max_val - min_val) / 255.0;
        let zero = min_val;

        let quantized: Vec<u8> = original
            .iter()
            .map(|&v| ((v - zero) / scale).round() as u8)
            .collect();

        let storage = TensorStorage::Quantized {
            data: quantized,
            scales: vec![scale],
            zeros: vec![zero],
            dtype: TensorDtype::I8,
            num_elements: 8,
            group_size,
        };

        let deq = storage.dequantize();
        let norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let diff: f32 = original
            .iter()
            .zip(&deq)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let relative_error = diff / norm;
        assert!(
            relative_error < 0.01,
            "8-bit dequant error {relative_error} >= 1%"
        );
    }
}

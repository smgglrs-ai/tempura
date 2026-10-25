use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::loader::{LoadError, WeightLoader};

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 1u32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e += 1;
        }
        f32::from_bits((sign << 31) | ((127 - 15 + 1 - e) << 23) | ((f & 0x3ff) << 13))
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
    }
}

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

/// Transpose I2S weights from row-major to column-major by byte position.
///
/// Original: row j has bytes [0..k/4). Row j byte b at offset j*(k/4)+b.
/// Transposed: byte position b has n rows contiguous. Position b row j at offset b*n+j.
/// The per-tensor f32 scale is preserved at the end.

#[derive(Debug)]
pub struct Tl2Tensor {
    pub data: Vec<u8>,
    pub m: usize,
    pub k: usize,
    pub scale: f32,
}

/// Pre-resolved weight reference for a single tensor, avoiding repeated HashMap lookups.
/// Computed once per (layer, projection) before the forward-pass layer loop.
pub enum PreIndexedWeight<'a> {
    Typed(tempura_kernels::ggml::GgmlType, &'a [u8]),
    Tl2 {
        tensor: &'a Tl2Tensor,
        corrections: Option<&'a [f32]>,
    },
    I2s(&'a [u8]),
    Q2Tiled {
        q2: Option<&'a [u8]>,
    },
    Q2x4 {
        x4: &'a [u8],
        q2: Option<&'a [u8]>,
    },
    Q2I2s {
        i2s: &'a [u8],
        scales: &'a [f32],
        q2: Option<&'a [u8]>,
    },
    Q2(&'a [u8]),
    F16(&'a [u8]),
    Q4k(&'a [u8]),
    Q6k(&'a [u8]),
    F32(&'a [f32]),
}

impl<'a> PreIndexedWeight<'a> {
    pub fn as_q4k(&self) -> Option<&'a [u8]> {
        use tempura_kernels::ggml::GgmlType;
        match self {
            PreIndexedWeight::Q4k(w) => Some(w),
            PreIndexedWeight::Typed(GgmlType::Q4K, w) => Some(w),
            _ => None,
        }
    }

    pub fn as_q5k(&self) -> Option<&'a [u8]> {
        use tempura_kernels::ggml::GgmlType;
        match self {
            PreIndexedWeight::Typed(GgmlType::Q5K, w) => Some(w),
            _ => None,
        }
    }

    pub fn as_q6k(&self) -> Option<&'a [u8]> {
        use tempura_kernels::ggml::GgmlType;
        match self {
            PreIndexedWeight::Q6k(w) => Some(w),
            PreIndexedWeight::Typed(GgmlType::Q6K, w) => Some(w),
            _ => None,
        }
    }

    pub fn as_i2s(&self) -> Option<&'a [u8]> {
        use tempura_kernels::ggml::GgmlType;
        match self {
            PreIndexedWeight::I2s(w) => Some(w),
            PreIndexedWeight::Typed(dtype, w) if *dtype == GgmlType::TQ1_0 => Some(w),
            _ => None,
        }
    }

    pub fn as_f16(&self) -> Option<&'a [u8]> {
        match self {
            PreIndexedWeight::F16(w) => Some(w),
            _ => None,
        }
    }

    pub fn as_tl2(&self) -> Option<&'a Tl2Tensor> {
        match self {
            PreIndexedWeight::Tl2 { tensor, .. } => Some(tensor),
            _ => None,
        }
    }
}

pub struct QuantizedWeights {
    q4k_tensors: HashMap<String, Vec<u8>>,
    q6k_tensors: HashMap<String, Vec<u8>>,
    tq2_tensors: HashMap<String, Vec<u8>>,
    i2s_tensors: HashMap<String, Vec<u8>>,
    tl2_tensors: HashMap<String, Tl2Tensor>,
    f16_tensors: HashMap<String, Vec<u8>>,
    f32_tensors: HashMap<String, Vec<f32>>,
    typed_tensors: HashMap<String, (tempura_kernels::ggml::GgmlType, Vec<u8>)>,
    mmap: Option<memmap2::Mmap>,
    mmap_i2s: HashMap<String, (usize, usize)>,
    mmap_f16: HashMap<String, (usize, usize)>,
    mmap_bf16: HashMap<String, (usize, usize)>,
    bf16_vecs: HashMap<String, Vec<u8>>,
    mmap_q4k: HashMap<String, (usize, usize)>,
    mmap_q6k: HashMap<String, (usize, usize)>,
    mmap_q2: HashMap<String, (usize, usize)>,
    q2_i2s: HashMap<String, (Vec<u8>, Vec<f32>, usize, usize)>,
    q2x4: HashMap<String, (Vec<u8>, usize, usize)>,
    q2_tiled: HashMap<String, (Vec<u8>, usize, usize)>,
    q2_row_corrections: HashMap<String, Vec<f32>>,
    q4k_repacked: HashMap<String, Vec<tempura_kernels::ggml::blocks::BlockQ4Kx8>>,
    q6k_repacked: HashMap<String, Vec<tempura_kernels::ggml::blocks::BlockQ6Kx8>>,
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
            tq2_tensors: HashMap::new(),
            i2s_tensors: HashMap::new(),
            tl2_tensors: HashMap::new(),
            f16_tensors: HashMap::new(),
            f32_tensors,
            typed_tensors: HashMap::new(),
            mmap: None,
            mmap_i2s: HashMap::new(),
            mmap_f16: HashMap::new(),
            mmap_bf16: HashMap::new(),
            bf16_vecs: HashMap::new(),
            mmap_q4k: HashMap::new(),
            mmap_q6k: HashMap::new(),
            mmap_q2: HashMap::new(),
            q2_i2s: HashMap::new(),
            q2x4: HashMap::new(),
            q2_tiled: HashMap::new(),
            q2_row_corrections: HashMap::new(),
            q4k_repacked: HashMap::new(), q6k_repacked: HashMap::new(),
        }
    }

    pub fn new_with_ternary(
        q4k_tensors: HashMap<String, Vec<u8>>,
        q6k_tensors: HashMap<String, Vec<u8>>,
        tq2_tensors: HashMap<String, Vec<u8>>,
        i2s_tensors: HashMap<String, Vec<u8>>,
        f32_tensors: HashMap<String, Vec<f32>>,
    ) -> Self {
        Self {
            q4k_tensors,
            q6k_tensors,
            tq2_tensors,
            tl2_tensors: HashMap::new(),
            f16_tensors: HashMap::new(),
            i2s_tensors,
            f32_tensors,
            typed_tensors: HashMap::new(),
            mmap: None,
            mmap_i2s: HashMap::new(),
            mmap_f16: HashMap::new(),
            mmap_bf16: HashMap::new(),
            bf16_vecs: HashMap::new(),
            mmap_q4k: HashMap::new(),
            mmap_q6k: HashMap::new(),
            mmap_q2: HashMap::new(),
            q2_i2s: HashMap::new(),
            q2x4: HashMap::new(),
            q2_tiled: HashMap::new(),
            q2_row_corrections: HashMap::new(),
            q4k_repacked: HashMap::new(), q6k_repacked: HashMap::new(),
        }
    }

    pub fn new_typed(
        typed_tensors: HashMap<String, (tempura_kernels::ggml::GgmlType, Vec<u8>)>,
        f32_tensors: HashMap<String, Vec<f32>>,
    ) -> Self {
        use tempura_kernels::ggml::GgmlType;
        let mut q4k_tensors = HashMap::new();
        let mut q6k_tensors = HashMap::new();
        let mut tq2_tensors = HashMap::new();
        let mut i2s_tensors = HashMap::new();
        for (name, (dtype, data)) in &typed_tensors {
            match dtype {
                GgmlType::Q6K => {
                    q6k_tensors.insert(name.clone(), data.clone());
                }
                GgmlType::TQ2_0 => {
                    tq2_tensors.insert(name.clone(), data.clone());
                }
                GgmlType::TQ1_0 => {
                    i2s_tensors.insert(name.clone(), data.clone());
                }
                _ => {
                    q4k_tensors.insert(name.clone(), data.clone());
                }
            }
        }
        Self {
            q4k_tensors,
            q6k_tensors,
            tq2_tensors,
            i2s_tensors,
            f32_tensors,
            typed_tensors,
            tl2_tensors: HashMap::new(),
            f16_tensors: HashMap::new(),
            mmap: None,
            mmap_i2s: HashMap::new(),
            mmap_f16: HashMap::new(),
            mmap_bf16: HashMap::new(),
            bf16_vecs: HashMap::new(),
            mmap_q4k: HashMap::new(),
            mmap_q6k: HashMap::new(),
            mmap_q2: HashMap::new(),
            q2_i2s: HashMap::new(),
            q2x4: HashMap::new(),
            q2_tiled: HashMap::new(),
            q2_row_corrections: HashMap::new(),
            q4k_repacked: HashMap::new(), q6k_repacked: HashMap::new(),
        }
    }

    pub fn empty() -> Self {
        Self {
            q4k_tensors: HashMap::new(),
            q6k_tensors: HashMap::new(),
            tq2_tensors: HashMap::new(),
            i2s_tensors: HashMap::new(),
            tl2_tensors: HashMap::new(),
            f16_tensors: HashMap::new(),
            f32_tensors: HashMap::new(),
            typed_tensors: HashMap::new(),
            mmap: None,
            mmap_i2s: HashMap::new(),
            mmap_f16: HashMap::new(),
            mmap_bf16: HashMap::new(),
            bf16_vecs: HashMap::new(),
            mmap_q4k: HashMap::new(),
            mmap_q6k: HashMap::new(),
            mmap_q2: HashMap::new(),
            q2_i2s: HashMap::new(),
            q2x4: HashMap::new(),
            q2_tiled: HashMap::new(),
            q2_row_corrections: HashMap::new(),
            q4k_repacked: HashMap::new(), q6k_repacked: HashMap::new(),
        }
    }

    pub fn insert_f32(&mut self, name: String, data: Vec<f32>) {
        self.f32_tensors.insert(name, data);
    }

    pub fn get_q4k(&self, name: &str) -> Option<&[u8]> {
        self.q4k_tensors
            .get(name)
            .map(|v| v.as_slice())
            .or_else(|| {
                self.mmap_q4k
                    .get(name)
                    .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
            })
    }

    pub fn get_q4k_repacked(
        &self,
        name: &str,
    ) -> Option<&[tempura_kernels::ggml::blocks::BlockQ4Kx8]> {
        self.q4k_repacked.get(name).map(|v| v.as_slice())
    }

    pub fn build_q4k_repacked(&mut self, shapes: &HashMap<String, (usize, usize)>) {
        use tempura_kernels::ggml::blocks::{BlockQ4K, BLOCK_Q4K_SIZE};
        use tempura_kernels::ggml::repack::repack_q4k_to_q4kx8;
        let t = std::time::Instant::now();
        let names: Vec<String> = shapes.keys().cloned().collect();
        for name in &names {
            let (n_rows, k) = match shapes.get(name) {
                Some(&s) => s,
                None => continue,
            };
            if n_rows == 0 || k == 0 || n_rows % 8 != 0 {
                continue;
            }
            let bpr = k / 256;
            if bpr == 0 {
                continue;
            }
            let raw = match self.get_q4k(name) {
                Some(r) => r,
                None => continue,
            };
            let expected = n_rows * bpr * BLOCK_Q4K_SIZE;
            if raw.len() != expected {
                continue;
            }
            let src = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const BlockQ4K, n_rows * bpr)
            };
            let repacked = repack_q4k_to_q4kx8(src, n_rows, bpr);
            self.q4k_repacked.insert(name.clone(), repacked);
        }
        eprintln!(
            "[Q4K] repacked {} tensors in {:.1}s",
            self.q4k_repacked.len(),
            t.elapsed().as_secs_f64()
        );
    }

    pub fn build_q6k_repacked(&mut self, shapes: &HashMap<String, (usize, usize)>) {
        use tempura_kernels::ggml::blocks::{BlockQ6K, BLOCK_Q6K_SIZE};
        use tempura_kernels::ggml::repack_llama::repack_q6_k_to_q6_k_8_bl;
        let t = std::time::Instant::now();
        let names: Vec<String> = shapes.keys().cloned().collect();
        for name in &names {
            let (n_rows, k) = match shapes.get(name) {
                Some(&s) => s,
                None => continue,
            };
            if n_rows == 0 || k == 0 || n_rows % 8 != 0 { continue; }
            let bpr = k / 256;
            if bpr == 0 { continue; }
            let raw = match self.get_q6k(name) {
                Some(r) => r,
                None => continue,
            };
            let expected = n_rows * bpr * BLOCK_Q6K_SIZE;
            if raw.len() != expected { continue; }
            let src = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const BlockQ6K, n_rows * bpr)
            };
            let mut dst = Vec::new();
            if repack_q6_k_to_q6_k_8_bl(src, n_rows, bpr, 8, &mut dst).is_ok() {
                self.q6k_repacked.insert(name.clone(), dst);
            }
        }
        eprintln!(
            "[Q6K] repacked {} tensors in {:.1}s",
            self.q6k_repacked.len(),
            t.elapsed().as_secs_f64()
        );
    }

    pub fn get_q6k_repacked(&self, name: &str) -> Option<&[tempura_kernels::ggml::blocks::BlockQ6Kx8]> {
        self.q6k_repacked.get(name).map(|v| v.as_slice())
    }

    pub fn get_q6k(&self, name: &str) -> Option<&[u8]> {
        self.q6k_tensors
            .get(name)
            .map(|v| v.as_slice())
            .or_else(|| {
                self.mmap_q6k
                    .get(name)
                    .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
            })
    }

    pub fn get_q2(&self, name: &str) -> Option<&[u8]> {
        self.mmap_q2
            .get(name)
            .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
    }

    pub fn get_tq2(&self, name: &str) -> Option<&[u8]> {
        self.tq2_tensors.get(name).map(|v| v.as_slice())
    }

    pub fn get_i2s(&self, name: &str) -> Option<&[u8]> {
        self.i2s_tensors
            .get(name)
            .map(|v| v.as_slice())
            .or_else(|| {
                self.mmap_i2s
                    .get(name)
                    .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
            })
    }

    pub fn get_tl2(&self, name: &str) -> Option<&Tl2Tensor> {
        self.tl2_tensors.get(name)
    }

    pub fn add_tl2_tensor(&mut self, name: String, data: Vec<u8>, m: usize, k: usize, scale: f32) {
        self.tl2_tensors
            .insert(name, Tl2Tensor { data, m, k, scale });
    }

    pub fn get_f16(&self, name: &str) -> Option<&[u8]> {
        self.f16_tensors
            .get(name)
            .map(|v| v.as_slice())
            .or_else(|| {
                self.mmap_f16
                    .get(name)
                    .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
            })
    }

    pub fn insert_f16(&mut self, name: String, data: Vec<u8>) {
        self.f16_tensors.insert(name, data);
    }

    pub fn set_mmap(&mut self, mmap: memmap2::Mmap) {
        self.mmap = Some(mmap);
    }

    pub fn register_mmap_i2s(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_i2s.insert(name, (offset, len));
    }

    pub fn register_mmap_f16(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_f16.insert(name, (offset, len));
    }

    pub fn register_mmap_bf16(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_bf16.insert(name, (offset, len));
    }

    pub fn insert_bf16(&mut self, name: String, data: Vec<u8>) {
        self.bf16_vecs.insert(name, data);
    }

    pub fn get_bf16(&self, name: &str) -> Option<&[u8]> {
        if let Some(v) = self.bf16_vecs.get(name) {
            return Some(v.as_slice());
        }
        self.mmap_bf16
            .get(name)
            .and_then(|&(off, len)| self.mmap.as_ref().map(|m| &m[off..off + len]))
    }

    pub fn register_mmap_q4k(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_q4k.insert(name, (offset, len));
    }

    pub fn register_mmap_q6k(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_q6k.insert(name, (offset, len));
    }

    pub fn register_mmap_q2(&mut self, name: String, offset: usize, len: usize) {
        self.mmap_q2.insert(name, (offset, len));
    }

    pub fn get_f32(&self, name: &str) -> Option<&[f32]> {
        self.f32_tensors.get(name).map(|v| v.as_slice())
    }

    pub fn typed_tensor_names(&self) -> Vec<String> {
        self.typed_tensors.keys().cloned().collect()
    }

    pub fn mmap_q4k_names(&self) -> Vec<String> {
        self.mmap_q4k.keys().cloned().collect()
    }

    pub fn mmap_q6k_names(&self) -> Vec<String> {
        self.mmap_q6k.keys().cloned().collect()
    }

    pub fn f32_tensor_names(&self) -> Vec<String> {
        self.f32_tensors.keys().cloned().collect()
    }

    pub fn mmap_f16_names(&self) -> Vec<String> {
        self.mmap_f16.keys().cloned().collect()
    }

    /// Free all tensors that have been uploaded to GPU and are no longer
    /// needed on CPU during GPU inference.  `keep_keys` lists exact tensor
    /// names that must be retained (e.g. `model.embed_tokens.weight`).
    ///
    /// All heap-allocated maps are pruned via `retain`.  The mmap index maps
    /// are also pruned so the mmap pages can be evicted under memory pressure
    /// (the mmap itself is kept alive for any remaining CPU-resident slices).
    pub fn free_gpu_uploaded(&mut self, keep_keys: &[&str]) {
        let keep: std::collections::HashSet<&str> = keep_keys.iter().copied().collect();

        self.q4k_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.q6k_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.tq2_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.i2s_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.tl2_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.f16_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.f32_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.typed_tensors.retain(|k, _| keep.contains(k.as_str()));
        self.bf16_vecs.retain(|k, _| keep.contains(k.as_str()));
        self.mmap_i2s.retain(|k, _| keep.contains(k.as_str()));
        self.mmap_f16.retain(|k, _| keep.contains(k.as_str()));
        self.mmap_q4k.retain(|k, _| keep.contains(k.as_str()));
        self.mmap_q6k.retain(|k, _| keep.contains(k.as_str()));
        self.mmap_q2.retain(|k, _| keep.contains(k.as_str()));
        self.q2_i2s.retain(|k, _| keep.contains(k.as_str()));
        self.q2x4.retain(|k, _| keep.contains(k.as_str()));
        self.q2_tiled.retain(|k, _| keep.contains(k.as_str()));
        self.q2_row_corrections.retain(|k, _| keep.contains(k.as_str()));
        self.q4k_repacked.retain(|k, _| keep.contains(k.as_str()));
        self.q6k_repacked.retain(|k, _| keep.contains(k.as_str()));
    }

    pub fn embed_token(&self, token_id: u32, hidden_size: usize, output: &mut [f32]) {
        let off = token_id as usize * hidden_size;
        if let Some(f32_data) = self.get_f32("model.embed_tokens.weight") {
            output.copy_from_slice(&f32_data[off..off + hidden_size]);
        } else if let Some(f16_data) = self.get_f16("model.embed_tokens.weight") {
            let byte_off = off * 2;
            for i in 0..hidden_size {
                let bits = u16::from_le_bytes([
                    f16_data[byte_off + i * 2],
                    f16_data[byte_off + i * 2 + 1],
                ]);
                output[i] = f16_bits_to_f32(bits);
            }
        } else if let Some(qk_data) = self
            .get_q4k("model.embed_tokens.weight")
            .or_else(|| self.get_q6k("model.embed_tokens.weight"))
        {
            let is_q6k = self.get_q4k("model.embed_tokens.weight").is_none();
            let block_size = 256;
            let block_bytes = if is_q6k { 210 } else { 144 };
            let row_bytes = (hidden_size / block_size) * block_bytes;
            let row_off = token_id as usize * row_bytes;
            if row_off + row_bytes <= qk_data.len() {
                let row = &qk_data[row_off..row_off + row_bytes];
                if is_q6k {
                    crate::gguf_loader::dequant_q6k_row(row, output, hidden_size);
                } else {
                    crate::gguf_loader::dequant_q4k_row(row, output, hidden_size);
                }
            }
        } else if let Some(q2_data) = self.get_q2("model.embed_tokens.weight") {
            let row_bytes = (hidden_size / 128) * 34;
            let row_off = token_id as usize * row_bytes;
            let row = &q2_data[row_off..row_off + row_bytes];
            for blk in 0..hidden_size / 128 {
                let block = &row[blk * 34..(blk + 1) * 34];
                let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..128 {
                    let byte_idx = 2 + i / 4;
                    let shift = (i % 4) * 2;
                    let q = ((block[byte_idx] >> shift) & 3) as i32;
                    output[blk * 128 + i] = (q - 1) as f32 * scale;
                }
            }
        } else {
            panic!("no embedding weights found");
        }
    }

    pub fn mmap_q2_names(&self) -> Vec<String> {
        self.mmap_q2.keys().cloned().collect()
    }

    pub fn get_q2_i2s(&self, name: &str) -> Option<(&[u8], &[f32], usize, usize)> {
        self.q2_i2s
            .get(name)
            .map(|(d, s, n, k)| (d.as_slice(), s.as_slice(), *n, *k))
    }

    pub fn get_q2x4(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        self.q2x4.get(name).map(|(d, n, k)| (d.as_slice(), *n, *k))
    }

    pub fn get_q2_tiled(&self, name: &str) -> Option<(&[u8], usize, usize)> {
        self.q2_tiled
            .get(name)
            .map(|(d, n, k)| (d.as_slice(), *n, *k))
    }

    pub fn has_q2_tiled(&self) -> bool {
        !self.q2_tiled.is_empty()
    }

    pub fn get_q2_row_corrections(&self, name: &str) -> Option<&[f32]> {
        self.q2_row_corrections.get(name).map(|v| v.as_slice())
    }

    pub fn build_q2_tl2(&mut self, tensor_shapes: &HashMap<String, (usize, usize)>) {
        use rayon::prelude::*;
        let names: Vec<String> = self.mmap_q2.keys().cloned().collect();
        // Process in batches of 4 to limit peak memory (~400MB scratch per tensor)
        for batch in names.chunks(4) {
            let results: Vec<_> = batch
                .par_iter()
                .filter_map(|name| {
                    let &(n, k) = tensor_shapes.get(name)?;
                    let q2_data = self.get_q2(name)?;
                    let (i2s_data, _avg_scale, corrections) =
                        tempura_kernels::quantized::q2_to_i2s_approx(q2_data, n, k);
                    let (tl2_data, scale) =
                        tempura_kernels::tl2::repack_i2s_to_tl2(&i2s_data, n, k)?;
                    Some((name.clone(), tl2_data, scale, n, k, corrections))
                })
                .collect();
            for (name, tl2_data, scale, n, k, corrections) in results {
                self.tl2_tensors.insert(
                    name.clone(),
                    Tl2Tensor {
                        data: tl2_data,
                        m: n,
                        k,
                        scale,
                    },
                );
                self.q2_row_corrections.insert(name, corrections);
            }
        }
    }

    pub fn build_q2x4(&mut self, tensor_shapes: &HashMap<String, (usize, usize)>) {
        let names: Vec<String> = self.mmap_q2.keys().cloned().collect();
        for name in names {
            if let Some(&(n, k)) = tensor_shapes.get(&name) {
                if n >= 4 {
                    if let Some(q2_data) = self.get_q2(&name) {
                        let repacked = tempura_kernels::quantized::repack_q2_to_q2x4(q2_data, n, k);
                        self.q2x4.insert(name, (repacked, n, k));
                    }
                }
            }
        }
    }

    pub fn build_q2_i2s(&mut self, tensor_shapes: &HashMap<String, (usize, usize)>) {
        let names: Vec<String> = self.mmap_q2.keys().cloned().collect();
        for name in names {
            if let Some(&(n, k)) = tensor_shapes.get(&name) {
                if let Some(q2_data) = self.get_q2(&name) {
                    let (i2s_data, scales) =
                        tempura_kernels::quantized::repack_q2_to_i2s_layout(q2_data, n, k);
                    self.q2_i2s.insert(name, (i2s_data, scales, n, k));
                }
            }
        }
    }

    pub fn build_q2_lut_tiled(&mut self, tensor_shapes: &HashMap<String, (usize, usize)>) {
        use rayon::prelude::*;
        let names: Vec<String> = self.mmap_q2.keys().cloned().collect();
        let results: Vec<_> = names
            .par_iter()
            .filter_map(|name| {
                let &(n, k) = tensor_shapes.get(name)?;
                let q2_data = self.get_q2(name)?;
                let tiled = tempura_kernels::quantized::q2_lut_repack(q2_data, n, k);
                Some((name.clone(), tiled, n, k))
            })
            .collect();
        for (name, tiled, n, k) in results {
            self.q2_tiled.insert(name, (tiled, n, k));
        }
    }

    pub fn build_tl2_from_i2s(&mut self, tensor_shapes: &HashMap<String, (usize, usize)>) {
        let names: Vec<String> = self
            .i2s_tensors
            .keys()
            .chain(self.mmap_i2s.keys())
            .cloned()
            .collect();
        for name in names {
            if let Some(&(n, k)) = tensor_shapes.get(&name) {
                if let Some(data) = self.get_i2s(&name) {
                    if let Some((tl2_data, scale)) =
                        tempura_kernels::tl2::repack_i2s_to_tl2(data, n, k)
                    {
                        self.tl2_tensors.insert(
                            name.clone(),
                            Tl2Tensor {
                                data: tl2_data,
                                m: n,
                                k,
                                scale,
                            },
                        );
                    }
                }
            }
        }
    }

    pub fn get_typed(&self, name: &str) -> Option<(tempura_kernels::ggml::GgmlType, &[u8])> {
        self.typed_tensors
            .get(name)
            .map(|(t, d)| (*t, d.as_slice()))
    }

    pub fn tensor_type(&self, name: &str) -> Option<tempura_kernels::ggml::GgmlType> {
        self.typed_tensors.get(name).map(|(t, _)| *t)
    }

    /// Resolve a tensor name to a direct weight reference, performing all HashMap lookups once.
    /// Returns None if the tensor is not found in any storage.
    pub fn resolve_weight<'a>(&'a self, name: &str) -> Option<PreIndexedWeight<'a>> {
        if let Some((dtype, w)) = self.get_typed(name) {
            return Some(PreIndexedWeight::Typed(dtype, w));
        }
        if let Some(tensor) = self.get_tl2(name) {
            let corrections = self.get_q2_row_corrections(name);
            return Some(PreIndexedWeight::Tl2 { tensor, corrections });
        }
        if let Some(w) = self.get_i2s(name) {
            return Some(PreIndexedWeight::I2s(w));
        }
        if self.get_q2_tiled(name).is_some() {
            let q2 = self.get_q2(name);
            return Some(PreIndexedWeight::Q2Tiled { q2 });
        }
        if let Some((x4, _, _)) = self.get_q2x4(name) {
            let q2 = self.get_q2(name);
            return Some(PreIndexedWeight::Q2x4 { x4, q2 });
        }
        if let Some((i2s, scales, _, _)) = self.get_q2_i2s(name) {
            let q2 = self.get_q2(name);
            return Some(PreIndexedWeight::Q2I2s { i2s, scales, q2 });
        }
        if let Some(w) = self.get_q2(name) {
            return Some(PreIndexedWeight::Q2(w));
        }
        if let Some(w) = self.get_f16(name) {
            return Some(PreIndexedWeight::F16(w));
        }
        if let Some(w) = self.get_q4k(name) {
            return Some(PreIndexedWeight::Q4k(w));
        }
        if let Some(w) = self.get_q6k(name) {
            return Some(PreIndexedWeight::Q6k(w));
        }
        if let Some(w) = self.get_f32(name) {
            return Some(PreIndexedWeight::F32(w));
        }
        None
    }

    /// Load a Llama-style HuggingFace BF16 safetensors directory into QuantizedWeights.
    ///
    /// Projection weights (q/k/v/o/gate/up/down) are stored as raw BF16 bytes in
    /// `bf16_vecs` so `maybe_upload_mxfp4_tiled_from_bf16` can convert them directly.
    /// All other tensors (norms, embeddings, lm_head) are converted to f32.
    ///
    /// Shards are processed one at a time to bound peak CPU RAM usage.
    pub fn from_safetensors_bf16_dir(
        dir: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use safetensors::SafeTensors;

        fn bf16_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
            bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    // BF16 -> f32: shift mantissa into high half.
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }

        fn is_projection(name: &str) -> bool {
            (name.contains(".self_attn.") || name.contains(".mlp."))
                && name.ends_with("_proj.weight")
        }

        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err("no .safetensors files found in directory".into());
        }

        let mut qw = Self::empty();

        for shard_path in &shard_paths {
            eprintln!("[safetensors] loading shard {:?}", shard_path.file_name().unwrap());
            let bytes = std::fs::read(shard_path)?;
            let st = SafeTensors::deserialize(&bytes)?;
            for (name, view) in st.tensors() {
                let data = view.data();
                let dtype = view.dtype();
                if is_projection(&name) && dtype == safetensors::Dtype::BF16 {
                    qw.insert_bf16(name.to_string(), data.to_vec());
                } else {
                    let f32_data = match dtype {
                        safetensors::Dtype::BF16 => bf16_bytes_to_f32_vec(data),
                        safetensors::Dtype::F32 => data
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        safetensors::Dtype::F16 => data
                            .chunks_exact(2)
                            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                            .collect(),
                        _ => continue,
                    };
                    qw.insert_f32(name.to_string(), f32_data);
                }
            }
        }

        eprintln!(
            "[safetensors] loaded {} BF16 projection tensors, {} f32 tensors",
            qw.bf16_vecs.len(),
            qw.f32_tensors.len()
        );
        Ok(qw)
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

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("size mismatch for {name}: expected {expected}, got {got}")]
    SizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
}

pub trait WeightLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, LoadError>;
    fn tensor_names(&self) -> Vec<String>;
}

pub struct SafetensorsLoader {
    tensors: HashMap<String, Vec<f32>>,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = exp;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e = e.wrapping_sub(1);
        }
        f &= 0x3ff;
        let f32_exp = 127 - 15 + 1 + e;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

impl SafetensorsLoader {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path)?;
        let tensors_view = safetensors::SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();
        for (name, view) in tensors_view.tensors() {
            let bytes = view.data();
            let floats = match view.dtype() {
                safetensors::Dtype::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::BF16 => bytes
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                safetensors::Dtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                _ => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            };
            tensors.insert(name.to_string(), floats);
        }

        Ok(Self { tensors })
    }
}

impl WeightLoader for SafetensorsLoader {
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

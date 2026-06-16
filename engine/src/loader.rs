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
    #[error("no safetensors files found in directory")]
    NoFiles,
    #[error("GGUF parse error: {0}")]
    GgufParse(String),
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
        let mut tensors = HashMap::new();
        Self::load_tensors_from_bytes(&data, &mut tensors)?;
        Ok(Self { tensors })
    }

    pub fn from_dir(dir: &Path) -> Result<Self, LoadError> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();

        if paths.is_empty() {
            return Err(LoadError::NoFiles);
        }

        paths.sort();

        let mut tensors = HashMap::new();
        for path in &paths {
            let data = std::fs::read(path)?;
            Self::load_tensors_from_bytes(&data, &mut tensors)?;
        }

        Ok(Self { tensors })
    }

    fn load_tensors_from_bytes(
        data: &[u8],
        tensors: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), LoadError> {
        let tensors_view = safetensors::SafeTensors::deserialize(data)?;
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
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn write_safetensors(dir: &Path, filename: &str, tensors: &HashMap<&str, Vec<f32>>) {
        let data: HashMap<String, safetensors::tensor::TensorView<'_>> = tensors
            .iter()
            .map(|(name, values)| {
                let bytes: &[u8] = bytemuck::cast_slice(values.as_slice());
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![values.len()],
                    bytes,
                )
                .unwrap();
                (name.to_string(), view)
            })
            .collect();
        let serialized = safetensors::tensor::serialize(&data, &None).unwrap();
        std::fs::write(dir.join(filename), serialized).unwrap();
    }

    #[test]
    fn from_dir_loads_all_shards() {
        let dir = tempfile::tempdir().unwrap();

        let mut t1 = HashMap::new();
        t1.insert("layer.0.weight", vec![1.0_f32, 2.0, 3.0]);
        write_safetensors(dir.path(), "model-00001.safetensors", &t1);

        let mut t2 = HashMap::new();
        t2.insert("layer.1.weight", vec![4.0_f32, 5.0, 6.0]);
        write_safetensors(dir.path(), "model-00002.safetensors", &t2);

        let loader = SafetensorsLoader::from_dir(dir.path()).unwrap();
        let names = loader.tensor_names();
        assert!(names.contains(&"layer.0.weight".to_string()));
        assert!(names.contains(&"layer.1.weight".to_string()));
        assert_eq!(
            loader.load_tensor("layer.0.weight").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            loader.load_tensor("layer.1.weight").unwrap(),
            vec![4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn from_dir_empty_returns_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let result = SafetensorsLoader::from_dir(dir.path());
        assert!(matches!(result, Err(LoadError::NoFiles)));
    }
}

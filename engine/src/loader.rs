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
    #[error("integrity check failed for tensor {name}: expected {expected}, got {got}")]
    IntegrityMismatch {
        name: String,
        expected: String,
        got: String,
    },
}

pub trait WeightLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, LoadError>;
    fn tensor_names(&self) -> Vec<String>;
}

pub struct SafetensorsLoader {
    tensors: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntegrityManifest {
    pub tensor_hashes: HashMap<String, String>,
}

impl IntegrityManifest {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| LoadError::GgufParse(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), LoadError> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| LoadError::GgufParse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
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

    pub fn from_file_mmap(path: &Path) -> Result<Self, LoadError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut tensors = HashMap::new();
        Self::load_tensors_from_bytes(&mmap, &mut tensors)?;
        Ok(Self { tensors })
    }

    pub fn from_dir(dir: &Path) -> Result<Self, LoadError> {
        Self::from_dir_impl(dir, false)
    }

    pub fn from_dir_mmap(dir: &Path) -> Result<Self, LoadError> {
        Self::from_dir_impl(dir, true)
    }

    fn from_dir_impl(dir: &Path, use_mmap: bool) -> Result<Self, LoadError> {
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
            if use_mmap {
                let file = std::fs::File::open(path)?;
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                Self::load_tensors_from_bytes(&mmap, &mut tensors)?;
            } else {
                let data = std::fs::read(path)?;
                Self::load_tensors_from_bytes(&data, &mut tensors)?;
            }
        }

        Ok(Self { tensors })
    }

    pub fn verify_integrity(&self, manifest: &IntegrityManifest) -> Result<(), LoadError> {
        use sha2::{Digest, Sha256};
        for (name, expected_hash) in &manifest.tensor_hashes {
            let tensor = self
                .tensors
                .get(name.as_str())
                .ok_or_else(|| LoadError::MissingTensor(name.clone()))?;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tensor.as_ptr() as *const u8, tensor.len() * 4)
            };
            let got = format!("{:x}", Sha256::digest(bytes));
            if got != *expected_hash {
                return Err(LoadError::IntegrityMismatch {
                    name: name.clone(),
                    expected: expected_hash.clone(),
                    got,
                });
            }
        }
        Ok(())
    }

    pub fn generate_manifest(&self) -> IntegrityManifest {
        use sha2::{Digest, Sha256};
        let mut hashes = HashMap::new();
        for (name, tensor) in &self.tensors {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tensor.as_ptr() as *const u8, tensor.len() * 4)
            };
            let hash = format!("{:x}", Sha256::digest(bytes));
            hashes.insert(name.clone(), hash);
        }
        IntegrityManifest {
            tensor_hashes: hashes,
        }
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

    #[test]
    fn from_file_mmap_matches_regular_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = HashMap::new();
        t.insert("weight_a", vec![1.0f32, 2.0, 3.0, 4.0]);
        t.insert("weight_b", vec![5.0f32, 6.0]);
        write_safetensors(dir.path(), "model.safetensors", &t);

        let regular = SafetensorsLoader::from_file(&dir.path().join("model.safetensors")).unwrap();
        let mmap =
            SafetensorsLoader::from_file_mmap(&dir.path().join("model.safetensors")).unwrap();

        assert_eq!(
            regular.load_tensor("weight_a").unwrap(),
            mmap.load_tensor("weight_a").unwrap()
        );
        assert_eq!(
            regular.load_tensor("weight_b").unwrap(),
            mmap.load_tensor("weight_b").unwrap()
        );
    }

    #[test]
    fn from_dir_mmap_loads_all_shards() {
        let dir = tempfile::tempdir().unwrap();
        let mut t1 = HashMap::new();
        t1.insert("a", vec![1.0f32, 2.0]);
        write_safetensors(dir.path(), "shard-01.safetensors", &t1);

        let mut t2 = HashMap::new();
        t2.insert("b", vec![3.0f32, 4.0]);
        write_safetensors(dir.path(), "shard-02.safetensors", &t2);

        let loader = SafetensorsLoader::from_dir_mmap(dir.path()).unwrap();
        assert_eq!(loader.load_tensor("a").unwrap(), vec![1.0, 2.0]);
        assert_eq!(loader.load_tensor("b").unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn integrity_manifest_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = HashMap::new();
        t.insert("w", vec![1.0f32, 2.0, 3.0]);
        write_safetensors(dir.path(), "model.safetensors", &t);

        let loader = SafetensorsLoader::from_file(&dir.path().join("model.safetensors")).unwrap();
        let manifest = loader.generate_manifest();

        assert!(manifest.tensor_hashes.contains_key("w"));

        let manifest_path = dir.path().join("manifest.json");
        manifest.save(&manifest_path).unwrap();
        let loaded = IntegrityManifest::from_file(&manifest_path).unwrap();
        assert_eq!(loaded.tensor_hashes["w"], manifest.tensor_hashes["w"]);

        assert!(loader.verify_integrity(&loaded).is_ok());
    }

    #[test]
    fn integrity_check_detects_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = HashMap::new();
        t.insert("w", vec![1.0f32, 2.0, 3.0]);
        write_safetensors(dir.path(), "model.safetensors", &t);

        let loader = SafetensorsLoader::from_file(&dir.path().join("model.safetensors")).unwrap();
        let mut manifest = loader.generate_manifest();
        manifest.tensor_hashes.insert("w".into(), "deadbeef".into());

        let result = loader.verify_integrity(&manifest);
        assert!(matches!(result, Err(LoadError::IntegrityMismatch { .. })));
    }
}

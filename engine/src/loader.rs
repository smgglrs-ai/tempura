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
}

pub trait WeightLoader {
    fn load_tensor(&self, name: &str) -> Result<Vec<f32>, LoadError>;
    fn tensor_names(&self) -> Vec<String>;
}

pub struct SafetensorsLoader {
    tensors: HashMap<String, Vec<f32>>,
}

impl SafetensorsLoader {
    pub fn from_file(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path)?;
        let tensors_view = safetensors::SafeTensors::deserialize(&data)?;

        let mut tensors = HashMap::new();
        for (name, view) in tensors_view.tensors() {
            let bytes = view.data();
            // assume f32 for now
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
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

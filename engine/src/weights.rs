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
}

#[derive(Debug)]
pub struct ModelWeights {
    tensors: HashMap<String, Vec<f32>>,
}

impl ModelWeights {
    pub fn load(loader: &dyn WeightLoader, config: &ModelConfig) -> Result<Self, WeightsError> {
        let required = Self::required_tensor_names(config);
        let expected_sizes = Self::expected_tensor_sizes(config);

        let mut tensors = HashMap::new();
        for name in &required {
            let data = loader
                .load_tensor(name)
                .map_err(|_| WeightsError::MissingTensor(name.clone()))?;

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

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
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
}

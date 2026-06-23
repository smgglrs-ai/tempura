use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub id: String,
    pub rank: usize,
    pub alpha: f32,
    pub a_matrices: HashMap<String, Vec<f32>>,
    pub b_matrices: HashMap<String, Vec<f32>>,
}

impl LoraAdapter {
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn apply_to_output(
        &self,
        layer_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let a_key = format!("{layer_name}.lora_A");
        let b_key = format!("{layer_name}.lora_B");
        let (a, b) = match (self.a_matrices.get(&a_key), self.b_matrices.get(&b_key)) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };

        let r = self.rank;
        let scale = self.scaling();

        let mut intermediate = vec![0.0f32; m * r];
        for i in 0..m {
            for j in 0..r {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += input[i * k + p] * a[j * k + p];
                }
                intermediate[i * r + j] = sum;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..r {
                    sum += intermediate[i * r + p] * b[j * r + p];
                }
                output[i * n + j] += sum * scale;
            }
        }
    }

    pub fn has_weights_for(&self, layer_name: &str) -> bool {
        let a_key = format!("{layer_name}.lora_A");
        self.a_matrices.contains_key(&a_key)
    }
}

pub struct LoraRegistry {
    adapters: RwLock<HashMap<String, Arc<LoraAdapter>>>,
}

impl LoraRegistry {
    pub fn new() -> Self {
        Self {
            adapters: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, adapter: LoraAdapter) {
        let id = adapter.id.clone();
        self.adapters
            .write()
            .unwrap()
            .insert(id, Arc::new(adapter));
    }

    pub fn get(&self, id: &str) -> Option<Arc<LoraAdapter>> {
        self.adapters.read().unwrap().get(id).cloned()
    }

    pub fn remove(&self, id: &str) -> bool {
        self.adapters.write().unwrap().remove(id).is_some()
    }

    pub fn list(&self) -> Vec<String> {
        self.adapters.read().unwrap().keys().cloned().collect()
    }

    pub fn count(&self) -> usize {
        self.adapters.read().unwrap().len()
    }
}

impl Default for LoraRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter(id: &str, rank: usize) -> LoraAdapter {
        let k = 4;
        let n = 4;
        let a = vec![0.1f32; rank * k];
        let b = vec![0.1f32; n * rank];
        LoraAdapter {
            id: id.to_string(),
            rank,
            alpha: rank as f32,
            a_matrices: HashMap::from([("proj.lora_A".into(), a)]),
            b_matrices: HashMap::from([("proj.lora_B".into(), b)]),
        }
    }

    #[test]
    fn test_lora_scaling() {
        let adapter = make_adapter("test", 8);
        assert_eq!(adapter.scaling(), 1.0);

        let adapter2 = LoraAdapter {
            alpha: 16.0,
            ..make_adapter("test2", 8)
        };
        assert_eq!(adapter2.scaling(), 2.0);
    }

    #[test]
    fn test_lora_apply() {
        let adapter = make_adapter("test", 2);
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        adapter.apply_to_output("proj", &input, &mut output, 1, 4, 4);
        assert!(output.iter().any(|&v| v != 0.0), "LoRA should modify output");
    }

    #[test]
    fn test_lora_no_weights_is_noop() {
        let adapter = make_adapter("test", 2);
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        adapter.apply_to_output("nonexistent", &input, &mut output, 1, 4, 4);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_registry() {
        let reg = LoraRegistry::new();
        assert_eq!(reg.count(), 0);

        reg.register(make_adapter("a1", 4));
        reg.register(make_adapter("a2", 8));
        assert_eq!(reg.count(), 2);

        let a1 = reg.get("a1").unwrap();
        assert_eq!(a1.rank, 4);
        assert!(reg.get("missing").is_none());

        reg.remove("a1");
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_hot_reload() {
        let reg = LoraRegistry::new();
        reg.register(make_adapter("v1", 4));
        assert_eq!(reg.get("v1").unwrap().rank, 4);

        reg.register(make_adapter("v1", 16));
        assert_eq!(reg.get("v1").unwrap().rank, 16);
    }
}

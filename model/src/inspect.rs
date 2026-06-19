use std::fs;
use std::path::Path;

use crate::ModelError;

#[derive(Debug)]
pub struct ModelInfo {
    pub format: String,
    pub architecture: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub vocab_size: Option<usize>,
    pub quantization: Option<String>,
    pub tensor_count: Option<usize>,
    pub file_size_bytes: u64,
}

pub fn inspect(path: &Path) -> Result<ModelInfo, ModelError> {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        || is_gguf_magic(path)
    {
        inspect_gguf(path)
    } else if path.is_dir() {
        inspect_dir(path)
    } else if path.extension().is_some_and(|ext| ext == "safetensors") {
        inspect_safetensors(path)
    } else {
        Err(ModelError::Format(format!(
            "unknown format: {}",
            path.display()
        )))
    }
}

fn is_gguf_magic(path: &Path) -> bool {
    fs::read(path)
        .ok()
        .map(|data| data.len() >= 4 && &data[..4] == b"GGUF")
        .unwrap_or(false)
}

fn inspect_gguf(path: &Path) -> Result<ModelInfo, ModelError> {
    let loader = tempura_engine::gguf_loader::GgufLoader::from_file(path)
        .map_err(|e| ModelError::Format(e.to_string()))?;

    let config = &loader.config;
    let arch = loader
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let quant = loader
        .metadata
        .get("general.file_type")
        .and_then(|v| v.as_u32())
        .map(|ft| format!("file_type_{ft}"));

    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    use tempura_engine::loader::WeightLoader;
    Ok(ModelInfo {
        format: "gguf".to_string(),
        architecture: arch,
        hidden_size: Some(config.hidden_size),
        num_layers: Some(config.num_layers),
        num_heads: Some(config.num_attention_heads),
        num_kv_heads: Some(config.num_kv_heads),
        vocab_size: Some(config.vocab_size),
        quantization: quant,
        tensor_count: Some(loader.tensor_names().len()),
        file_size_bytes: file_size,
    })
}

fn inspect_dir(path: &Path) -> Result<ModelInfo, ModelError> {
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Err(ModelError::Format("no config.json in directory".into()));
    }

    let config = tempura_engine::ModelConfig::from_hf_config(&config_path)
        .map_err(|e| ModelError::Format(e.to_string()))?;

    let safetensors_count = fs::read_dir(path)
        .ok()
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                .count()
        })
        .unwrap_or(0);

    let total_size: u64 = fs::read_dir(path)
        .ok()
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);

    let hf_config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&config_path).unwrap_or_default())
            .unwrap_or_default();

    let arch = hf_config
        .get("architectures")
        .and_then(|a| a.as_array()?.first()?.as_str())
        .map(|s| s.to_string());

    Ok(ModelInfo {
        format: format!("safetensors ({safetensors_count} file(s))"),
        architecture: arch,
        hidden_size: Some(config.hidden_size),
        num_layers: Some(config.num_layers),
        num_heads: Some(config.num_attention_heads),
        num_kv_heads: Some(config.num_kv_heads),
        vocab_size: Some(config.vocab_size),
        quantization: config
            .quantization
            .map(|q| format!("{:?} {}bit", q.format, q.bits)),
        tensor_count: None,
        file_size_bytes: total_size,
    })
}

fn inspect_safetensors(path: &Path) -> Result<ModelInfo, ModelError> {
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let loader = tempura_engine::loader::SafetensorsLoader::from_file(path)
        .map_err(|e| ModelError::Format(e.to_string()))?;

    use tempura_engine::loader::WeightLoader;
    let tensor_count = loader.tensor_names().len();

    Ok(ModelInfo {
        format: "safetensors".to_string(),
        architecture: None,
        hidden_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        vocab_size: None,
        quantization: None,
        tensor_count: Some(tensor_count),
        file_size_bytes: file_size,
    })
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Format:         {}", self.format)?;
        if let Some(ref arch) = self.architecture {
            writeln!(f, "Architecture:   {arch}")?;
        }
        if let Some(h) = self.hidden_size {
            writeln!(f, "Hidden size:    {h}")?;
        }
        if let Some(n) = self.num_layers {
            writeln!(f, "Layers:         {n}")?;
        }
        if let Some(n) = self.num_heads {
            write!(f, "Attention:      {n} heads")?;
            if let Some(kv) = self.num_kv_heads {
                if kv != n {
                    write!(f, " ({kv} KV heads, GQA)")?;
                }
            }
            writeln!(f)?;
        }
        if let Some(v) = self.vocab_size {
            writeln!(f, "Vocab size:     {v}")?;
        }
        if let Some(ref q) = self.quantization {
            writeln!(f, "Quantization:   {q}")?;
        }
        if let Some(t) = self.tensor_count {
            writeln!(f, "Tensors:        {t}")?;
        }
        let mb = self.file_size_bytes as f64 / 1_048_576.0;
        if mb > 1024.0 {
            writeln!(f, "Size:           {:.1} GB", mb / 1024.0)?;
        } else {
            writeln!(f, "Size:           {:.0} MB", mb)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore]
    fn inspect_tinyllama_dir() {
        let path = PathBuf::from(
            dirs::home_dir()
                .unwrap()
                .join(".local/share/navra/models/tinyllama-1.1b"),
        );
        if !path.exists() {
            eprintln!("Skipping: TinyLlama not found at {}", path.display());
            return;
        }
        let info = inspect(&path).expect("inspect should succeed");
        println!("{info}");
        assert!(info.hidden_size.is_some());
        assert!(info.num_layers.is_some());
    }
}

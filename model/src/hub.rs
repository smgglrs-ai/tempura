use std::fs;
use std::io::Write;
use std::path::Path;

use crate::store::{ModelEntry, ModelStore};
use crate::ModelError;

const HF_API_BASE: &str = "https://huggingface.co";

pub fn pull(store: &ModelStore, repo_id: &str) -> Result<ModelEntry, ModelError> {
    store
        .ensure_dir()
        .map_err(|e| ModelError::Io(e.to_string()))?;

    let name = repo_id.replace('/', "--");
    let model_dir = store.model_dir(&name);
    fs::create_dir_all(&model_dir).map_err(|e| ModelError::Io(e.to_string()))?;

    tracing::info!("Downloading {repo_id} to {}", model_dir.display());

    let files = ["config.json", "tokenizer.json", "tokenizer_config.json"];

    let client = reqwest::blocking::Client::builder()
        .user_agent("tempura-serve/0.1.0")
        .build()
        .map_err(|e| ModelError::Download(e.to_string()))?;

    for file in &files {
        let url = format!("{HF_API_BASE}/{repo_id}/resolve/main/{file}");
        match client.get(&url).send() {
            Ok(resp) if resp.status().is_success() => {
                let bytes = resp
                    .bytes()
                    .map_err(|e| ModelError::Download(e.to_string()))?;
                let dest = model_dir.join(file);
                fs::write(&dest, &bytes).map_err(|e| ModelError::Io(e.to_string()))?;
                tracing::info!("  Downloaded {file} ({} bytes)", bytes.len());
            }
            _ => {
                tracing::debug!("  {file} not found, skipping");
            }
        }
    }

    let single_url = format!("{HF_API_BASE}/{repo_id}/resolve/main/model.safetensors");
    let resp = client.head(&single_url).send();

    if resp.is_ok() && resp.unwrap().status().is_success() {
        download_file(&client, &single_url, &model_dir.join("model.safetensors"))?;
    } else {
        let index_url =
            format!("{HF_API_BASE}/{repo_id}/resolve/main/model.safetensors.index.json");
        match client.get(&index_url).send() {
            Ok(resp) if resp.status().is_success() => {
                let index_data = resp
                    .text()
                    .map_err(|e| ModelError::Download(e.to_string()))?;
                fs::write(model_dir.join("model.safetensors.index.json"), &index_data)
                    .map_err(|e| ModelError::Io(e.to_string()))?;

                let index: serde_json::Value = serde_json::from_str(&index_data)
                    .map_err(|e| ModelError::Download(e.to_string()))?;

                if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                    let mut shard_files: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect();
                    shard_files.sort();
                    shard_files.dedup();

                    for shard in &shard_files {
                        let shard_url = format!("{HF_API_BASE}/{repo_id}/resolve/main/{shard}");
                        download_file(&client, &shard_url, &model_dir.join(shard))?;
                    }
                }
            }
            _ => {
                tracing::warn!("No safetensors files found for {repo_id}");
            }
        }
    }

    let total_size = dir_size(&model_dir);

    let format = if model_dir.join("model.safetensors").exists()
        || fs::read_dir(&model_dir).ok().map(|d| {
            d.filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        }) == Some(true)
    {
        "safetensors"
    } else {
        "unknown"
    };

    let architecture = fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|data| serde_json::from_str::<serde_json::Value>(&data).ok())
        .and_then(|v| {
            v.get("architectures")?
                .as_array()?
                .first()?
                .as_str()
                .map(|s| s.to_string())
        });

    let entry = ModelEntry {
        name: name.clone(),
        repo_id: repo_id.to_string(),
        path: model_dir.clone(),
        format: format.to_string(),
        size_bytes: total_size,
        architecture,
        quantization: None,
    };

    store
        .save_manifest(&entry)
        .map_err(|e| ModelError::Io(e.to_string()))?;

    Ok(entry)
}

fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
) -> Result<(), ModelError> {
    if dest.exists() {
        tracing::info!("  {} already exists, skipping", dest.display());
        return Ok(());
    }
    tracing::info!(
        "  Downloading {}...",
        dest.file_name().unwrap_or_default().to_string_lossy()
    );
    let resp = client
        .get(url)
        .send()
        .map_err(|e| ModelError::Download(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(ModelError::Download(format!(
            "HTTP {}: {url}",
            resp.status()
        )));
    }
    let bytes = resp
        .bytes()
        .map_err(|e| ModelError::Download(e.to_string()))?;
    let mut file = fs::File::create(dest).map_err(|e| ModelError::Io(e.to_string()))?;
    file.write_all(&bytes)
        .map_err(|e| ModelError::Io(e.to_string()))?;
    tracing::info!(
        "  Saved {} ({:.1} MB)",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        bytes.len() as f64 / 1_048_576.0
    );
    Ok(())
}

fn dir_size(path: &Path) -> u64 {
    fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

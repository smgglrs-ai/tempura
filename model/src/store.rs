use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub repo_id: String,
    pub path: PathBuf,
    pub format: String,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
}

pub struct ModelStore {
    root: PathBuf,
}

// Pre-rename installs keep their models in ~/.tempura-serve; prefer that dir
// when it exists so existing downloads stay visible, otherwise use ~/.tempura.
fn models_dir(home: &Path) -> PathBuf {
    let legacy = home.join(".tempura-serve").join("models");
    if legacy.exists() {
        legacy
    } else {
        home.join(".tempura").join("models")
    }
}

impl ModelStore {
    pub fn default_path() -> PathBuf {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        models_dir(&home)
    }

    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn open() -> Self {
        Self::new(Self::default_path())
    }

    pub fn ensure_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        let mut entries = Vec::new();
        let read_dir = match fs::read_dir(&self.root) {
            Ok(d) => d,
            Err(_) => return entries,
        };

        for entry in read_dir.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let manifest_path = path.join("manifest.json");
            if let Ok(data) = fs::read_to_string(&manifest_path) {
                if let Ok(model) = serde_json::from_str::<ModelEntry>(&data) {
                    entries.push(model);
                }
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    pub fn get(&self, name: &str) -> Option<ModelEntry> {
        let manifest_path = self.model_dir(name).join("manifest.json");
        let data = fs::read_to_string(&manifest_path).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn save_manifest(&self, entry: &ModelEntry) -> std::io::Result<()> {
        let dir = self.model_dir(&entry.name);
        fs::create_dir_all(&dir)?;
        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(entry).unwrap();
        fs::write(manifest_path, json)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_fresh_install_uses_new_path() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(
            models_dir(home.path()),
            home.path().join(".tempura").join("models")
        );
    }

    #[test]
    fn models_dir_prefers_legacy_dir_when_present() {
        let home = tempfile::tempdir().unwrap();
        fs::create_dir_all(home.path().join(".tempura-serve").join("models")).unwrap();
        assert_eq!(
            models_dir(home.path()),
            home.path().join(".tempura-serve").join("models")
        );
    }

    #[test]
    fn list_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());
        store.ensure_dir().unwrap();
        assert!(store.list().is_empty());
    }

    #[test]
    fn save_and_get_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());
        store.ensure_dir().unwrap();

        let entry = ModelEntry {
            name: "test-model".to_string(),
            repo_id: "org/test-model".to_string(),
            path: dir.path().join("test-model"),
            format: "safetensors".to_string(),
            size_bytes: 1024,
            architecture: Some("LlamaForCausalLM".to_string()),
            quantization: None,
        };

        store.save_manifest(&entry).unwrap();

        let got = store.get("test-model").expect("manifest should exist");
        assert_eq!(got.repo_id, "org/test-model");
        assert_eq!(got.format, "safetensors");
        assert_eq!(got.size_bytes, 1024);

        let all = store.list();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "test-model");
    }
}

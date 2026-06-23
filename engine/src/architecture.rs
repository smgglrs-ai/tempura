use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Architecture {
    pub name: String,
    pub blocks: Vec<BlockType>,
    pub norm_type: NormType,
    pub ffn_type: FfnType,
    pub position_encoding: PositionEncoding,
    pub attention_type: AttentionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockType {
    PreNormAttention,
    PostNormAttention,
    PreNormFfn,
    PostNormFfn,
    ParallelAttentionFfn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnType {
    SiluGated,
    GeluGated,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionEncoding {
    Rope,
    RopeScaled,
    Alibi,
    Learned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    MultiHead,
    GroupedQuery,
    MultiQuery,
}

impl Architecture {
    pub fn llama() -> Self {
        Self {
            name: "llama".into(),
            blocks: vec![BlockType::PreNormAttention, BlockType::PreNormFfn],
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SiluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::GroupedQuery,
        }
    }

    pub fn mistral() -> Self {
        Self {
            name: "mistral".into(),
            blocks: vec![BlockType::PreNormAttention, BlockType::PreNormFfn],
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SiluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::GroupedQuery,
        }
    }

    pub fn phi() -> Self {
        Self {
            name: "phi".into(),
            blocks: vec![BlockType::ParallelAttentionFfn],
            norm_type: NormType::LayerNorm,
            ffn_type: FfnType::GeluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::GroupedQuery,
        }
    }

    pub fn gemma() -> Self {
        Self {
            name: "gemma".into(),
            blocks: vec![BlockType::PreNormAttention, BlockType::PreNormFfn],
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::GeluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::MultiHead,
        }
    }

    pub fn qwen2() -> Self {
        Self {
            name: "qwen2".into(),
            blocks: vec![BlockType::PreNormAttention, BlockType::PreNormFfn],
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SiluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::GroupedQuery,
        }
    }

    pub fn gpt_neox() -> Self {
        Self {
            name: "gpt-neox".into(),
            blocks: vec![BlockType::ParallelAttentionFfn],
            norm_type: NormType::LayerNorm,
            ffn_type: FfnType::GeluGated,
            position_encoding: PositionEncoding::Rope,
            attention_type: AttentionType::MultiHead,
        }
    }
}

pub fn from_model_type(model_type: &str) -> Option<Architecture> {
    match model_type {
        "llama" | "LlamaForCausalLM" => Some(Architecture::llama()),
        "mistral" | "MistralForCausalLM" => Some(Architecture::mistral()),
        "phi" | "PhiForCausalLM" | "phi3" | "Phi3ForCausalLM" => Some(Architecture::phi()),
        "gemma" | "gemma2" | "GemmaForCausalLM" | "Gemma2ForCausalLM" => {
            Some(Architecture::gemma())
        }
        "qwen2" | "Qwen2ForCausalLM" | "qwen3" | "Qwen3ForCausalLM" => Some(Architecture::qwen2()),
        "gpt_neox" | "GPTNeoXForCausalLM" => Some(Architecture::gpt_neox()),
        _ => None,
    }
}

pub fn diff_architectures(a: &Architecture, b: &Architecture) -> Vec<String> {
    let mut diffs = Vec::new();
    if a.norm_type != b.norm_type {
        diffs.push(format!("norm: {:?} vs {:?}", a.norm_type, b.norm_type));
    }
    if a.ffn_type != b.ffn_type {
        diffs.push(format!("ffn: {:?} vs {:?}", a.ffn_type, b.ffn_type));
    }
    if a.position_encoding != b.position_encoding {
        diffs.push(format!(
            "pos_enc: {:?} vs {:?}",
            a.position_encoding, b.position_encoding
        ));
    }
    if a.attention_type != b.attention_type {
        diffs.push(format!(
            "attention: {:?} vs {:?}",
            a.attention_type, b.attention_type
        ));
    }
    if a.blocks != b.blocks {
        diffs.push(format!("blocks: {:?} vs {:?}", a.blocks, b.blocks));
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_architectures() {
        let llama = Architecture::llama();
        assert_eq!(llama.norm_type, NormType::RmsNorm);
        assert_eq!(llama.ffn_type, FfnType::SiluGated);
        assert_eq!(llama.blocks.len(), 2);
    }

    #[test]
    fn test_from_model_type() {
        assert!(from_model_type("llama").is_some());
        assert!(from_model_type("LlamaForCausalLM").is_some());
        assert!(from_model_type("qwen3").is_some());
        assert!(from_model_type("unknown_model").is_none());
    }

    #[test]
    fn test_diff_identical() {
        let a = Architecture::llama();
        let b = Architecture::llama();
        assert!(diff_architectures(&a, &b).is_empty());
    }

    #[test]
    fn test_diff_different() {
        let llama = Architecture::llama();
        let phi = Architecture::phi();
        let diffs = diff_architectures(&llama, &phi);
        assert!(!diffs.is_empty());
        assert!(diffs.iter().any(|d| d.contains("norm")));
        assert!(diffs.iter().any(|d| d.contains("blocks")));
    }

    #[test]
    fn test_serializes() {
        let arch = Architecture::llama();
        let json = serde_json::to_string(&arch).unwrap();
        assert!(json.contains("\"llama\""));
        assert!(json.contains("rms_norm"));
    }

    #[test]
    fn test_all_six_architectures() {
        let archs = [
            Architecture::llama(),
            Architecture::mistral(),
            Architecture::phi(),
            Architecture::gemma(),
            Architecture::qwen2(),
            Architecture::gpt_neox(),
        ];
        assert_eq!(archs.len(), 6);
        for arch in &archs {
            assert!(!arch.name.is_empty());
            assert!(!arch.blocks.is_empty());
        }
    }
}

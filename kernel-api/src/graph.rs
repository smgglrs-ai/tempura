use crate::{Result, VerificationTier};

pub struct LayerInput<'a> {
    pub hidden_state: &'a [f32],
    pub positions: &'a [u32],
    pub k_cache: &'a [f32],
    pub v_cache: &'a [f32],
    pub seq_len: usize,
    pub kv_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
}

pub struct LayerOutput {
    pub hidden_state: Vec<f32>,
    pub new_k: Vec<f32>,
    pub new_v: Vec<f32>,
}

pub trait FusedTransformerLayer: Send + Sync {
    fn forward(
        &self,
        input: &LayerInput,
        weights: &LayerWeights,
    ) -> Result<LayerOutput>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct LayerWeights<'a> {
    pub input_layernorm: &'a [f32],
    pub post_attention_layernorm: &'a [f32],
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub gate_proj: &'a [f32],
    pub up_proj: &'a [f32],
    pub down_proj: &'a [f32],
}

pub struct PrefillInput<'a> {
    pub token_ids: &'a [u32],
    pub embeddings: &'a [f32],
    pub hidden_size: usize,
}

pub struct PrefillOutput {
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
    pub last_hidden: Vec<f32>,
}

pub trait CompiledPrefill: Send + Sync {
    fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

pub struct DecodeInput<'a> {
    pub token_embedding: &'a [f32],
    pub position: u32,
    pub k_cache: &'a [Vec<f32>],
    pub v_cache: &'a [Vec<f32>],
    pub kv_len: usize,
    pub hidden_size: usize,
}

pub struct DecodeOutput {
    pub logits: Vec<f32>,
    pub new_k: Vec<f32>,
    pub new_v: Vec<f32>,
}

pub trait CompiledDecodeStep: Send + Sync {
    fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FusedBlockTrace {
    pub block_name: String,
    pub input_norm: f32,
    pub output_norm: f32,
    pub duration_us: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_input_construction() {
        let hidden = vec![1.0f32; 64];
        let positions = vec![0u32; 4];
        let k_cache = vec![0.0f32; 256];
        let v_cache = vec![0.0f32; 256];

        let input = LayerInput {
            hidden_state: &hidden,
            positions: &positions,
            k_cache: &k_cache,
            v_cache: &v_cache,
            seq_len: 4,
            kv_len: 4,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            hidden_size: 64,
            intermediate_size: 256,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        assert_eq!(input.hidden_state.len(), 64);
        assert_eq!(input.seq_len, 4);
    }

    #[test]
    fn test_fused_block_trace_serializes() {
        let trace = FusedBlockTrace {
            block_name: "layer_0".into(),
            input_norm: 1.5,
            output_norm: 1.2,
            duration_us: Some(150),
        };
        let json = serde_json::to_string(&trace).unwrap();
        assert!(json.contains("layer_0"));
        assert!(json.contains("1.5"));
    }
}

pub mod graph;

use rand::RngCore;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, KernelError>;

// ── VerificationTier ────────────────────────────────────────────────────────

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum VerificationTier {
    Compiled,
    Tested,
    Proven,
}

// ── AttentionMode ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionMode {
    Auto,
    NaiveF32,
    NaiveF16,
    FlashF32,
    FlashF16,
}

impl Default for AttentionMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for AttentionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::NaiveF32 => write!(f, "naive-f32"),
            Self::NaiveF16 => write!(f, "naive-f16"),
            Self::FlashF32 => write!(f, "flash-f32"),
            Self::FlashF16 => write!(f, "flash-f16"),
        }
    }
}

impl std::str::FromStr for AttentionMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "naive-f32" => Ok(Self::NaiveF32),
            "naive-f16" => Ok(Self::NaiveF16),
            "flash-f32" => Ok(Self::FlashF32),
            "flash-f16" => Ok(Self::FlashF16),
            _ => Err(format!("unknown attention mode: {s}")),
        }
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub trait Attention: Send + Sync {
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn forward_windowed(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        _window_size: usize,
    ) -> Result<()> {
        self.forward(
            q,
            k,
            v,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        q: &[f32],
        k_pages: &[f32],
        v_pages: &[f32],
        output: &mut [f32],
        block_table: &[u32],
        num_blocks: usize,
        block_size: usize,
        batch: usize,
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<()> {
        let stride = num_kv_heads * head_dim;
        let mut k_contiguous = vec![0.0f32; batch * kv_len * stride];
        let mut v_contiguous = vec![0.0f32; batch * kv_len * stride];
        for b in 0..batch {
            for pos in 0..kv_len {
                let block_idx = pos / block_size;
                let block_offset = pos % block_size;
                let phys_block = block_table[b * num_blocks + block_idx] as usize;
                let page_base = phys_block * block_size * stride + block_offset * stride;
                let dst = (b * kv_len + pos) * stride;
                k_contiguous[dst..dst + stride]
                    .copy_from_slice(&k_pages[page_base..page_base + stride]);
                v_contiguous[dst..dst + stride]
                    .copy_from_slice(&v_pages[page_base..page_base + stride]);
            }
        }
        self.forward(
            q,
            &k_contiguous,
            &v_contiguous,
            output,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── MatMul ──────────────────────────────────────────────────────────────────

pub trait MatMul: Send + Sync {
    fn forward(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    fn forward_b_transposed(
        &self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────

pub trait RMSNorm: Send + Sync {
    fn forward(&self, x: &mut [f32], weight: &[f32], hidden_size: usize, eps: f32) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── RoPE ────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub trait RoPE: Send + Sync {
    fn forward(
        &self,
        x: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[u32],
        theta: f32,
    ) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── Sampler ─────────────────────────────────────────────────────────────────

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            penalty_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    pub fn apply_penalties(&self, logits: &mut [f32], prior_tokens: &[u32]) {
        if self.presence_penalty == 0.0 && self.frequency_penalty == 0.0 {
            return;
        }

        let mut counts = std::collections::HashMap::new();
        for &tok in prior_tokens {
            *counts.entry(tok).or_insert(0u32) += 1;
        }

        for (&tok, &count) in &counts {
            if (tok as usize) < logits.len() {
                logits[tok as usize] -=
                    self.frequency_penalty * count as f32 + self.presence_penalty;
            }
        }
    }
}

pub trait Sampler: Send + Sync {
    fn apply(&self, logits: &[f32], output: &mut [f32], params: &SamplingParams) -> Result<()>;
    fn sample_token(&self, probs: &[f32], rng: Option<&mut dyn RngCore>) -> u32;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── LayerNorm ───────────────────────────────────────────────────────────────

pub trait LayerNorm: Send + Sync {
    fn forward(
        &self,
        x: &mut [f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── AllReduce ───────────────────────────────────────────────────────────────

pub trait AllReduce: Send + Sync {
    fn sum_in_place(&self, data: &mut [f32]) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── KvCacheCompressor ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompressedKV {
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
    pub quantized: Vec<u8>,
    pub num_vectors: usize,
    pub head_dim: usize,
    pub group_size: usize,
    pub bits: u8,
}

impl CompressedKV {
    pub fn groups_per_vector(&self) -> usize {
        self.head_dim.div_ceil(self.group_size)
    }
}

pub trait KvCacheCompressor: Send + Sync {
    fn compress(&self, data: &[f32], head_dim: usize) -> Result<CompressedKV>;
    fn decompress(&self, compressed: &CompressedKV) -> Result<Vec<f32>>;
}

// ── MatMulQ (quantized) ─────────────────────────────────────────────────────

pub const Q4K_BLOCK_SIZE: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_GROUPS: usize = 8;
pub const Q4K_GROUP_SIZE: usize = 32;

pub const Q6K_BLOCK_SIZE: usize = 256;
pub const Q6K_BLOCK_BYTES: usize = 210;
pub const Q6K_GROUPS: usize = 16;
pub const Q6K_GROUP_SIZE: usize = 16;

pub const TQ2_BLOCK_SIZE: usize = 256;
pub const TQ2_BLOCK_BYTES: usize = 66;

pub const I2S_BLOCK_SIZE: usize = 256;
pub const I2S_BLOCK_BYTES: usize = 64;

#[allow(clippy::too_many_arguments)]
pub trait MatMulQ: Send + Sync {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    fn forward_b_transposed_tq2(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}

// ── Dequantize ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantDtype {
    Int4,
    Int8,
    Fp16,
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub dtype: QuantDtype,
    pub num_elements: usize,
    pub group_size: usize,
}

pub trait Dequantize: Send + Sync {
    fn dequantize(
        &self,
        quantized: &[u8],
        scales: &[f32],
        zeros: &[f32],
        output: &mut [f32],
        dtype: QuantDtype,
        group_size: usize,
    ) -> Result<()>;

    fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Compiled
    }
}

// ── GuidedDecoder ───────────────────────────────────────────────────────────

pub trait GuidedDecoder: Send {
    fn mask_logits(&mut self, token_ids_so_far: &[u32], logits: &mut [f32]);
    fn is_complete(&self) -> bool;
    fn tokens_masked_last_step(&self) -> usize;

    fn forced_tokens(&self) -> Option<Vec<u32>> {
        None
    }
}

// ── Verification contracts ─────────────────────────────────────────────────
//
// Shared trait contracts for struct invariants. Both the real implementation
// and the Verus verification crate implement these traits, ensuring the
// specification stays in sync with the code.
//
// The real impl uses debug_assert!(self.well_formed()) in methods.
// The Verus crate proves well_formed() is preserved for ALL inputs.

pub trait VerifiedKvCache {
    fn seq_len(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn stride(&self) -> usize { self.num_kv_heads() * self.head_dim() }
    fn layer_size(&self) -> usize { self.max_seq_len() * self.stride() }
    fn remaining_capacity(&self) -> usize { self.max_seq_len() - self.seq_len() }

    fn well_formed(&self) -> bool {
        self.seq_len() <= self.max_seq_len()
            && self.num_kv_heads() >= 1
            && self.head_dim() >= 1
            && self.max_seq_len() >= 1
    }
}

pub trait VerifiedPageAllocator {
    fn available(&self) -> usize;
    fn total(&self) -> usize;
    fn in_use(&self) -> usize { self.total() - self.available() }

    fn well_formed(&self) -> bool {
        self.available() <= self.total()
    }

    fn conservation_holds(&self) -> bool {
        self.available() + self.in_use() == self.total()
    }
}

pub trait VerifiedBlockTable {
    fn num_blocks(&self) -> usize;
    fn max_blocks(&self) -> usize;

    fn well_formed(&self) -> bool {
        self.num_blocks() <= self.max_blocks()
    }
}

pub trait VerifiedSession {
    fn position(&self) -> usize;
    fn cache_seq_len(&self) -> usize;
    fn max_seq_len(&self) -> usize;

    fn consistent(&self) -> bool {
        self.position() == self.cache_seq_len()
            && self.position() <= self.max_seq_len()
    }
}

pub trait VerifiedPrefixCache {
    fn num_entries(&self) -> usize;
    fn max_entries(&self) -> usize;
    fn tick(&self) -> u64;

    fn well_formed(&self) -> bool {
        self.num_entries() <= self.max_entries()
    }
}

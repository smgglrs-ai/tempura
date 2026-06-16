pub mod allreduce;
pub mod attention;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod layernorm;
pub mod matmul;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod turboquant;
pub mod verification_manifest;

pub use allreduce::AllReduce;
pub use attention::Attention;
pub use layernorm::LayerNorm;
pub use matmul::MatMul;
pub use rmsnorm::RMSNorm;
pub use rope::RoPE;
pub use sampling::{Sampler, SamplingParams};
pub use turboquant::KvCacheCompressor;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, KernelError>;

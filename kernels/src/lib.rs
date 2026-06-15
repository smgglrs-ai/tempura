pub mod attention;
pub mod layernorm;
pub mod matmul;
pub mod rope;
pub mod sampling;

pub use attention::Attention;
pub use layernorm::LayerNorm;
pub use matmul::MatMul;
pub use rope::RoPE;
pub use sampling::{Sampler, SamplingParams};

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, KernelError>;

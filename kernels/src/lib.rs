pub use tempura_kernel_api::{
    AllReduce, Attention, AttentionMode, CompressedKV, DeltaNet, Dequantize, GuidedDecoder,
    KernelError, KvCacheCompressor, LayerNorm, MatMul, MatMulQ, QuantDtype, QuantizedTensor,
    RMSNorm, Result, RoPE, Sampler, SamplingParams, VerificationTier, VerifiedEquivalent,
    I2S_BLOCK_BYTES,
    I2S_BLOCK_SIZE, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS, Q4K_GROUP_SIZE,
    Q5K_BLOCK_BYTES, Q5K_BLOCK_SIZE, Q6K_BLOCK_BYTES,
    Q6K_BLOCK_SIZE, Q6K_GROUPS, Q6K_GROUP_SIZE, TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE,
};

pub use ggml::kernel_table;
pub mod activations;
pub mod allreduce;
pub mod attention;
#[cfg(feature = "cutile")]
pub mod cutile_attention;
#[cfg(feature = "cutile")]
pub mod cutile_device_weights;
#[cfg(feature = "cutile")]
pub mod cutile_elementwise;
#[cfg(feature = "cutile")]
pub mod cutile_flash_attention;
#[cfg(feature = "cutile")]
pub mod cutile_matmul;
#[cfg(feature = "cutile")]
pub mod cutile_matmul_host;
#[cfg(feature = "cutile")]
pub mod cutile_quantized_gemm;
pub mod gpu_capabilities;
#[cfg(feature = "cutile")]
pub mod oxide_q2_gemv_host;
#[cfg(feature = "cutile")]
pub mod oxide_q4k_gemv_host;
#[cfg(feature = "cutile")]
pub mod q4k_gemv_v4_ptx;
#[cfg(feature = "cutile")]
pub mod q6k_gemv_v4_ptx;
#[cfg(feature = "cutile")]
pub mod oxide_q6k_gemv_host;
#[cfg(feature = "cutile")]
pub mod oxide_decode_attn_host;
#[cfg(feature = "cutile")]
pub mod kv_scatter_write;
#[cfg(feature = "cutile")]
pub mod cutile_rmsnorm;
#[cfg(feature = "cutile")]
pub mod cutile_rope;
pub mod deltanet;
pub mod dispatch;
pub mod f16_gemv;
pub mod fused;
pub mod ggml;
#[cfg(all(feature = "cuda", not(feature = "cutile")))]
pub mod gpu;
pub mod guided;
pub mod inference_pool;
pub mod layernorm;
pub mod matmul;
pub mod moe;
#[cfg(any(feature = "npu-intel", feature = "npu-qualcomm"))]
pub mod npu;
#[cfg(feature = "openmp")]
pub mod openmp_gemv;
pub mod portable_math;
#[allow(dead_code)]
pub mod quantized;
pub mod rmsnorm;
pub mod rope;
pub mod sampler_advanced;
pub mod sampler_chain;
pub mod sampling;
pub mod thread_pool;
pub mod tl2;
pub mod turboquant;
pub mod verification_manifest;

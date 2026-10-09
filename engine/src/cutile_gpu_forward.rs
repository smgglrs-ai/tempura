/// Fused GPU forward pass for the transformer.
///
/// All projection weight matrices are dequantized once at load time and uploaded to the GPU.
/// Activations flow device-to-device across all projection GEMMs within each layer.
///
/// Key invariant: whenever `gemm_d2d` is called the activation tensor MUST already have
/// exactly `dw.k_pad` columns.  Callers ensure this by uploading activations pre-padded
/// (via `to_device_padded`) so that `gemm_d2d` never has to download, re-pad, and
/// re-upload the activation.
///
/// The two remaining explicit pad steps per layer are:
///   - normed   → h_pad   (one `to_device_padded` call, used for Q, K, V)
///   - normed2  → h_pad   (one `to_device_padded` call, used for gate, up)
///   - gate×up  → inter_pad (one `to_device_padded` call, used for down)
///   - attn_out → q_pad   (one from_device + one to_device_padded, for O projection)
///
/// CPU operations that still require a host round-trip per layer: two residual
/// adds (O-proj and down-proj outputs) and one K download for the KV-cache host
/// mirror.  RMSNorm, RoPE, and SiLU-gated MLP all run entirely on device.
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use tempura_kernels::cutile_elementwise::CutileElementwise;
use tempura_kernels::cutile_flash_attention::CutileFlashAttention;
use tempura_kernels::cutile_matmul::CutileMatMul;
use tempura_kernels::cutile_quantized_gemm::{
    CutileQ2Gemv, CutileQ4Gemv, CutileQ6kGemv, Q2_0_BLOCK_BYTES, Q2_0_PACKED_U32S, Q2_0_QK,
    Q4K_PACKED_U32S, Q6K_PACKED_U32S, Q6K_QK,
};
use tempura_kernels::{Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};
use tempura_kernels::cutile_rmsnorm::CutileRmsNorm;
use tempura_kernels::cutile_rope::{CutileRoPE, RopeScratch};

use crate::config::ModelConfig;
use crate::cutile_kv_cache::CutileKvCache;
use crate::weights::QuantizedWeights;

// BK must be a multiple that divides all standard hidden / intermediate
// dimensions (4096, 11008, 14336, …).  16 matches the tiling constant used
// inside CutileMatMul::forward_bt_device_to_device.
const BK: usize = 16;

/// A projection weight tensor resident on the GPU with shape metadata.
struct DeviceWeight {
    /// Pre-uploaded f32 tensor on the GPU, shaped [n, k_pad].
    tensor: Arc<Tensor<f32>>,
    /// Number of output rows (the "N" in z[M,N] = a[M,K] @ b[N,K]^T).
    n: usize,
    #[allow(dead_code)]
    k_raw: usize,
    /// Padded K actually stored on the GPU (multiple of BK, power-of-2).
    k_pad: usize,
}

/// Q2_0 quantized weight tensor resident on the GPU.
///
/// Packed format: [n × k_blocks × Q2_0_PACKED_U32S] u32 elements.
/// Each block of 128 elements occupies 9 u32 slots:
///   slot 0      : f32 bits of block scale (f16 → f32 at upload time)
///   slots 1..=8 : 8 u32 words of 2-bit packed weights (16 values per u32)
/// Dequant in kernel: (q - 1) × scale, q ∈ {0,1,2,3}.
struct DeviceWeightQ2 {
    /// Packed u32 tensor on GPU, flat shape [n × k_blocks × Q2_0_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q2_0 blocks per row (= k / Q2_0_QK).
    k_blocks: usize,
}

/// Q6_K quantized weight tensor resident on the GPU.
///
/// Packed format: [n × k_blocks × Q6K_PACKED_U32S] u32 elements.
/// Each block of 256 elements occupies 81 u32 slots:
///   slot 0       : f32 bits of block super-scale d (f16 → f32 at upload)
///   slots 1..17  : f32 bits of 16 per-group scales (i8 → f32 at upload)
///   slots 17..81 : 64 u32 words, each holding 4 q_full bytes (values 0..63)
/// Dequant in kernel: d × scale[group] × (q_full − 32), group = element_idx / 16.
struct DeviceWeightQ6K {
    /// Packed u32 tensor on GPU, flat shape [n × k_blocks × Q6K_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q6_K blocks per row (= k / Q6K_QK).
    k_blocks: usize,
}

/// Q4_K quantized weight tensor resident on the GPU.
///
/// Packed format: [n × k_blocks × Q4K_PACKED_U32S] u32 elements.
/// Each block of 256 elements occupies 42 u32 slots:
///   slot 0       : f32 bits of d (super-scale, f16 → f32)
///   slot 1       : f32 bits of dmin (super-min, f16 → f32)
///   slots 2..6   : sc[2*j] | (sc[2*j+1]<<8) for j=0..4 (unpacked 6-bit → 8-bit scales)
///   slots 6..10  : m[2*j]  | (m[2*j+1]<<8)  for j=0..4 (unpacked 6-bit → 8-bit mins)
///   slots 10..42 : 32 u32 words of packed nibble bytes (128 bytes total)
/// Dequant in kernel: w[j*64+l] = d*sc[2*j]*lo_nibble - dmin*m[2*j]
///                    w[j*64+l+32] = d*sc[2*j+1]*hi_nibble - dmin*m[2*j+1]
struct DeviceWeightQ4K {
    /// Packed u32 tensor on GPU, flat shape [n × k_blocks × Q4K_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q4_K blocks per row (= k / Q4K_BLOCK_SIZE).
    k_blocks: usize,
}

fn next_pow2(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// IEEE 754 f16 bit pattern → f32 (used in the Q2_0 upload path).
#[inline(always)]
fn f16_bits_to_f32_inline(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = exp;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e = e.wrapping_sub(1);
        }
        f &= 0x3ff;
        let f32_exp = 127 - 15 + 1 + e;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

// Scratch-pool slot indices — one pre-allocated decode output buffer per projection.
// Each slot is reclaimed between layers because the previous caller's Arc is dropped
// at the end of each loop body before the next layer's gemm_d2d_slot call.
const SLOT_Q: usize = 0;
const SLOT_K: usize = 1;
const SLOT_V: usize = 2;
const SLOT_O: usize = 3;
const SLOT_GATE: usize = 4;
const SLOT_UP: usize = 5;
const SLOT_DOWN: usize = 6;
const SLOT_LOGITS: usize = 7;
const NUM_SLOTS: usize = 8;

/// Transformer with all projection weights pre-uploaded to the GPU.
///
/// Create once with [`CutileGpuModel::from_quantized`] at model load time, then call
/// [`CutileGpuModel::forward`] for each inference step.
pub struct CutileGpuModel {
    config: ModelConfig,
    /// next_pow2(hidden_size.div_ceil(BK) * BK) — padded column count for h-wide activations.
    h_pad: usize,
    /// next_pow2(q_dim.div_ceil(BK) * BK) — padded column count for attention output (o_proj).
    q_pad: usize,
    /// next_pow2(intermediate_size.div_ceil(BK) * BK) — padded column count for FFN activations.
    inter_pad: usize,
    device_weights: HashMap<String, DeviceWeight>,
    /// Q4_K-quantized projection weights stored in packed GPU format.
    device_weights_q4k: HashMap<String, DeviceWeightQ4K>,
    /// Q2_0-quantized projection weights stored in packed GPU format.
    device_weights_q2: HashMap<String, DeviceWeightQ2>,
    /// Q6_K-quantized projection weights stored in packed GPU format.
    device_weights_q6k: HashMap<String, DeviceWeightQ6K>,
    /// Pre-uploaded RMSNorm weight vectors (1-D, `[h_pad]` elements) keyed by
    /// their HuggingFace tensor name (e.g. `model.layers.0.input_layernorm.weight`).
    norm_weights: HashMap<String, Arc<Tensor<f32>>>,
    matmul: CutileMatMul,
    /// Q4_K on-GPU GEMV dispatcher — shares the same CUDA stream as `matmul`.
    q4k_gemv: CutileQ4Gemv,
    /// Q2_0 on-GPU GEMV dispatcher — shares the same CUDA stream as `matmul`.
    q2_gemv: CutileQ2Gemv,
    /// Q6_K on-GPU GEMV dispatcher — shares the same CUDA stream as `matmul`.
    q6k_gemv: CutileQ6kGemv,
    flash_attn: CutileFlashAttention,
    cutile_rmsnorm: CutileRmsNorm,
    cutile_rope: CutileRoPE,
    cutile_elementwise: CutileElementwise,
    /// Pre-allocated decode output scratch buffers indexed by SLOT_* constants.
    ///
    /// Eliminates the `api::zeros` allocation inside `forward_bt_device_to_device`
    /// for the 7 projection GEMMs + lm_head per decode token (m=1 only).
    /// Single-threaded inference guarantees each slot's Arc has strong_count == 1
    /// when the next layer begins, so `Arc::try_unwrap` can reclaim the Tensor
    /// without re-allocating device memory.
    scratch: RefCell<[Option<Arc<Tensor<f32>>>; NUM_SLOTS]>,
    /// Pre-allocated RoPE scratch for Q (sized for min(max_seq_len, 4096) × n_heads).
    q_rope_scratch: RefCell<RopeScratch>,
    /// Pre-allocated RoPE scratch for K (sized for min(max_seq_len, 4096) × n_kv_heads).
    k_rope_scratch: RefCell<RopeScratch>,
}

// SAFETY: CutileMatMul, CutileFlashAttention and Tensor<f32> hold raw CUDA pointers,
// but tempura uses a single-threaded inference path per model instance.
unsafe impl Send for CutileGpuModel {}
unsafe impl Sync for CutileGpuModel {}

impl CutileGpuModel {
    // ── private helpers ───────────────────────────────────────────────────────

    fn upload(
        matmul: &CutileMatMul,
        device_weights: &mut HashMap<String, DeviceWeight>,
        name: &str,
        data: &[f32],
        n: usize,
        k_raw: usize,
    ) -> Result<(), tempura_kernels::KernelError> {
        // sm_120 (Blackwell) cuTile JIT requires K to be a power-of-2.
        let k_pad = next_pow2(k_raw.div_ceil(BK) * BK);
        let padded: Arc<Vec<f32>> = if k_pad == k_raw {
            Arc::new(data.to_vec())
        } else {
            let mut v = vec![0.0f32; n * k_pad];
            for row in 0..n {
                v[row * k_pad..row * k_pad + k_raw]
                    .copy_from_slice(&data[row * k_raw..(row + 1) * k_raw]);
            }
            Arc::new(v)
        };

        let stream = matmul.stream();
        let tensor: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&padded)
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n, k_pad])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();

        device_weights.insert(
            name.to_string(),
            DeviceWeight {
                tensor,
                n,
                k_raw,
                k_pad,
            },
        );
        Ok(())
    }

    /// Dequantize a named tensor from `QuantizedWeights` to a plain f32 Vec.
    ///
    /// Returns `None` if the tensor is not present in `qw`.
    fn dequantize(qw: &QuantizedWeights, name: &str, n_elements: usize) -> Option<Vec<f32>> {
        use tempura_kernels::ggml::dequantize::dequantize_to_f32;
        use tempura_kernels::ggml::GgmlType;
        if let Some(f32_data) = qw.get_f32(name) {
            return Some(f32_data.to_vec());
        }
        if let Some((dtype, bytes)) = qw.get_typed(name) {
            return Some(dequantize_to_f32(dtype, bytes, n_elements));
        }
        // mmap-resident quantized tensors — not in typed_tensors.
        if let Some(bytes) = qw.get_q4k(name) {
            return Some(dequantize_to_f32(GgmlType::Q4K, bytes, n_elements));
        }
        if let Some(bytes) = qw.get_q6k(name) {
            return Some(dequantize_to_f32(GgmlType::Q6K, bytes, n_elements));
        }
        if let Some(bytes) = qw.get_q2(name) {
            return Some(crate::gguf_loader::dequant_q2_0_pub(bytes, n_elements));
        }
        if let Some(bytes) = qw.get_f16(name) {
            return Some(dequantize_to_f32(GgmlType::F16, bytes, n_elements));
        }
        None
    }

    /// Upload a single projection weight, dequantizing first if necessary.
    fn maybe_upload(
        matmul: &CutileMatMul,
        device_weights: &mut HashMap<String, DeviceWeight>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<(), tempura_kernels::KernelError> {
        if let Some(data) = Self::dequantize(qw, name, n * k) {
            Self::upload(matmul, device_weights, name, &data, n, k)?;
        }
        Ok(())
    }

    /// Upload a Q2_0 weight in packed GPU format (no f32 dequantization).
    ///
    /// Converts the raw Q2_0 bytes to the 9-u32-per-block GPU layout:
    ///   u32[0]    = f32 bits of block scale (f16 → f32)
    ///   u32[1..8] = 8 u32 words of packed 2-bit data
    ///
    /// Returns `false` (and does nothing) when the tensor is absent.
    fn maybe_upload_q2(
        stream: &std::sync::Arc<cuda_core::Stream>,
        device_weights_q2: &mut HashMap<String, DeviceWeightQ2>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        let bytes = match qw.get_q2(name) {
            Some(b) => b,
            None => return Ok(false),
        };

        debug_assert!(
            k % Q2_0_QK == 0,
            "k={k} must be a multiple of Q2_0_QK={Q2_0_QK}"
        );
        let k_blocks = k / Q2_0_QK;
        let total_u32s = n * k_blocks * Q2_0_PACKED_U32S;
        let mut packed: Vec<u32> = Vec::with_capacity(total_u32s);

        for row in 0..n {
            for blk in 0..k_blocks {
                let off = (row * k_blocks + blk) * Q2_0_BLOCK_BYTES;
                let block = &bytes[off..off + Q2_0_BLOCK_BYTES];

                // Slot 0: block scale (f16 → f32, stored as u32 bit pattern).
                let scale_f16_bits = u16::from_le_bytes([block[0], block[1]]);
                let scale_f32 = f16_bits_to_f32_inline(scale_f16_bits);
                packed.push(scale_f32.to_bits());

                // Slots 1-8: 32 packed bytes → 8 u32 words.
                for w in 0..8usize {
                    let b0 = block[2 + w * 4] as u32;
                    let b1 = block[3 + w * 4] as u32;
                    let b2 = block[4 + w * 4] as u32;
                    let b3 = block[5 + w * 4] as u32;
                    packed.push(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
                }
            }
        }

        debug_assert_eq!(packed.len(), total_u32s);

        let tensor: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(packed))
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n * k_blocks * Q2_0_PACKED_U32S])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();

        device_weights_q2.insert(
            name.to_string(),
            DeviceWeightQ2 { tensor, n, k_blocks },
        );
        Ok(true)
    }

    /// Upload a Q6_K weight in packed GPU format (no f32 dequantization).
    ///
    /// Converts the raw Q6_K bytes to the 81-u32-per-block GPU layout:
    ///   u32[0]      = f32 bits of block super-scale d (f16 → f32)
    ///   u32[1..17]  = 16 f32 bits of group scales (i8 → f32 preconverted)
    ///   u32[17..81] = 64 u32 words of q_full bytes (4 per word, values 0..63)
    ///
    /// Returns `false` (and does nothing) when the tensor is absent.
    fn maybe_upload_q6k(
        stream: &std::sync::Arc<cuda_core::Stream>,
        device_weights_q6k: &mut HashMap<String, DeviceWeightQ6K>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        let bytes = match qw.get_q6k(name) {
            Some(b) => b,
            None => return Ok(false),
        };

        debug_assert!(
            k % Q6K_QK == 0,
            "k={k} must be a multiple of Q6K_QK={Q6K_QK}"
        );
        let k_blocks = k / Q6K_QK;
        let total_u32s = n * k_blocks * Q6K_PACKED_U32S;
        let mut packed: Vec<u32> = Vec::with_capacity(total_u32s);

        // Q6_K on-disk block bytes (210 per block).
        const Q6K_BYTES: usize = 210;

        for row in 0..n {
            for blk in 0..k_blocks {
                let off = (row * k_blocks + blk) * Q6K_BYTES;
                let block = &bytes[off..off + Q6K_BYTES];

                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d_bits = u16::from_le_bytes([block[208], block[209]]);
                let d_f32 = f16_bits_to_f32_inline(d_bits);

                // Slot 0: block super-scale as f32 bit pattern.
                packed.push(d_f32.to_bits());

                // Slots 1..17: 16 group scales (i8 → f32).
                for k_g in 0..16usize {
                    let s_f32 = (scales[k_g] as i8) as f32;
                    packed.push(s_f32.to_bits());
                }

                // Slots 17..81: 64 u32 words of q_full bytes in output element order.
                let mut q_full = [0u8; 256];
                for half in 0..2usize {
                    for l in 0..32usize {
                        let ql_l = ql[half * 64 + l] as u32;
                        let ql_l32 = ql[half * 64 + l + 32] as u32;
                        let qh_l = qh[half * 32 + l] as u32;
                        q_full[half * 128 + l] =
                            ((ql_l & 0xF) | ((qh_l & 3) << 4)) as u8;
                        q_full[half * 128 + l + 32] =
                            ((ql_l32 & 0xF) | (((qh_l >> 2) & 3) << 4)) as u8;
                        q_full[half * 128 + l + 64] =
                            ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) as u8;
                        q_full[half * 128 + l + 96] =
                            ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) as u8;
                    }
                }
                for i in 0..64usize {
                    let w = (q_full[i * 4] as u32)
                        | ((q_full[i * 4 + 1] as u32) << 8)
                        | ((q_full[i * 4 + 2] as u32) << 16)
                        | ((q_full[i * 4 + 3] as u32) << 24);
                    packed.push(w);
                }
            }
        }

        debug_assert_eq!(packed.len(), total_u32s);

        let tensor: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(packed))
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n * k_blocks * Q6K_PACKED_U32S])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();

        device_weights_q6k.insert(
            name.to_string(),
            DeviceWeightQ6K { tensor, n, k_blocks },
        );
        Ok(true)
    }

    /// Upload a Q4_K weight in packed GPU format (no f32 dequantization on the host).
    ///
    /// Converts the raw Q4_K bytes to the 42-u32-per-block GPU layout:
    ///   u32[0]      = f32 bits of d (f16 → f32)
    ///   u32[1]      = f32 bits of dmin (f16 → f32)
    ///   u32[2+j]    = sc[2*j] | (sc[2*j+1]<<8)  for j=0..4  (unpacked 6-bit scales)
    ///   u32[6+j]    = m[2*j]  | (m[2*j+1]<<8)   for j=0..4  (unpacked 6-bit mins)
    ///   u32[10..42] = 32 u32 words holding qs[0..128] (4 bytes per u32, little-endian)
    ///
    /// Returns `false` (and does nothing) when the tensor is absent.
    fn maybe_upload_q4k(
        stream: &std::sync::Arc<cuda_core::Stream>,
        device_weights_q4k: &mut HashMap<String, DeviceWeightQ4K>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        let bytes = match qw.get_q4k(name) {
            Some(b) => b,
            None => return Ok(false),
        };

        debug_assert!(
            k % Q4K_BLOCK_SIZE == 0,
            "k={k} must be a multiple of Q4K_BLOCK_SIZE={Q4K_BLOCK_SIZE}"
        );
        let k_blocks = k / Q4K_BLOCK_SIZE;
        let total_u32s = n * k_blocks * Q4K_PACKED_U32S;
        let mut packed: Vec<u32> = Vec::with_capacity(total_u32s);

        for row in 0..n {
            for blk in 0..k_blocks {
                let off = (row * k_blocks + blk) * Q4K_BLOCK_BYTES;
                let block = &bytes[off..off + Q4K_BLOCK_BYTES];

                // u32[0]: d as f32 bits
                let d_f16 = u16::from_le_bytes([block[0], block[1]]);
                packed.push(f16_bits_to_f32_inline(d_f16).to_bits());

                // u32[1]: dmin as f32 bits
                let dmin_f16 = u16::from_le_bytes([block[2], block[3]]);
                packed.push(f16_bits_to_f32_inline(dmin_f16).to_bits());

                // Unpack the 12-byte scale blob into 8 sc and 8 m values.
                // Mirrors q4k_unpack_scales in cutile_quantized_gemm.rs.
                let ps = &block[4..16];
                let mut sc = [0u8; 8];
                let mut m = [0u8; 8];
                for i in 0..4usize {
                    sc[i] = ps[i] & 0x3F;
                    sc[i + 4] = ((ps[i + 8] & 0x0F) | ((ps[i] >> 6) << 4)) & 0x3F;
                    m[i] = ps[i + 4] & 0x3F;
                    m[i + 4] = ((ps[i + 8] >> 4) | ((ps[i + 4] >> 6) << 4)) & 0x3F;
                }

                // u32[2..6]: scale pairs sc[2*j] | (sc[2*j+1]<<8) for j=0..4
                for j in 0..4usize {
                    packed.push(sc[2 * j] as u32 | ((sc[2 * j + 1] as u32) << 8));
                }
                // u32[6..10]: min pairs m[2*j] | (m[2*j+1]<<8) for j=0..4
                for j in 0..4usize {
                    packed.push(m[2 * j] as u32 | ((m[2 * j + 1] as u32) << 8));
                }

                // u32[10..42]: 32 u32s holding qs[0..128] (128 bytes, 4 bytes per word)
                let qs = &block[16..144];
                for w in 0..32usize {
                    let b0 = qs[w * 4] as u32;
                    let b1 = qs[w * 4 + 1] as u32;
                    let b2 = qs[w * 4 + 2] as u32;
                    let b3 = qs[w * 4 + 3] as u32;
                    packed.push(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
                }
            }
        }

        debug_assert_eq!(packed.len(), total_u32s);

        let tensor: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(packed))
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n * k_blocks * Q4K_PACKED_U32S])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();

        device_weights_q4k.insert(
            name.to_string(),
            DeviceWeightQ4K { tensor, n, k_blocks },
        );
        Ok(true)
    }

    /// Upload a norm weight vector from a host slice to a 1-D device tensor of
    /// `h_pad` elements, zero-padding from `h` to `h_pad` if necessary.
    fn upload_norm_weight(
        stream: &Arc<Stream>,
        norm_weights: &mut HashMap<String, Arc<Tensor<f32>>>,
        name: &str,
        data: &[f32],
        h: usize,
        h_pad: usize,
    ) -> Result<(), tempura_kernels::KernelError> {
        let padded: Vec<f32> = if h_pad == h {
            data.to_vec()
        } else {
            let mut v = vec![0.0f32; h_pad];
            v[..h].copy_from_slice(data);
            v
        };
        let tensor: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(padded))
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();
        norm_weights.insert(name.to_string(), tensor);
        Ok(())
    }

    /// Pre-compile all cuTile GEMM kernel specializations by firing a dummy `m=1`
    /// device-to-device call for each distinct matrix shape used by this model.
    ///
    /// cuTile JIT-compiles and caches PTX specializations the first time a given
    /// (M, N, K, BM, BN, BK) tile tuple is seen.  Calling this once after weight
    /// upload means the first real decode token pays zero JIT cost.
    ///
    /// Only layer 0 weights are exercised — all layers share the same shapes.
    fn warmup_gemms(&self) {
        let dummy_h = self.to_device(&vec![0.0f32; self.h_pad], 1, self.h_pad);
        let dummy_q = self.to_device(&vec![0.0f32; self.q_pad], 1, self.q_pad);
        let dummy_inter = self.to_device(&vec![0.0f32; self.inter_pad], 1, self.inter_pad);

        // Use gemm_d2d_slot for m=1 so the coalesced GEMV path gets compiled.
        // gemm_d2d uses the tiled GEMM and would not trigger cutile_gemv_coalesced.
        let _ = self.gemm_d2d_slot(
            &dummy_h,
            "model.layers.0.self_attn.q_proj.weight",
            1,
            SLOT_Q,
        );
        let _ = self.gemm_d2d_slot(
            &dummy_h,
            "model.layers.0.self_attn.k_proj.weight",
            1,
            SLOT_K,
        );
        let _ = self.gemm_d2d_slot(
            &dummy_h,
            "model.layers.0.self_attn.v_proj.weight",
            1,
            SLOT_V,
        );
        let _ = self.gemm_d2d_slot(
            &dummy_q,
            "model.layers.0.self_attn.o_proj.weight",
            1,
            SLOT_O,
        );
        let _ = self.gemm_d2d_slot(
            &dummy_h,
            "model.layers.0.mlp.gate_proj.weight",
            1,
            SLOT_GATE,
        );
        let _ = self.gemm_d2d_slot(&dummy_h, "model.layers.0.mlp.up_proj.weight", 1, SLOT_UP);
        let _ = self.gemm_d2d_slot(
            &dummy_inter,
            "model.layers.0.mlp.down_proj.weight",
            1,
            SLOT_DOWN,
        );
        let _ = self.gemm_d2d_slot(&dummy_h, "lm_head.weight", 1, SLOT_LOGITS);

        // Also warm up the m>1 prefill path (tiled GEMM) for a representative batch.
        let dummy_h_128 = self.to_device(&vec![0.0f32; 128 * self.h_pad], 128, self.h_pad);
        let _ = self.gemm_d2d(&dummy_h_128, "model.layers.0.self_attn.q_proj.weight", 128);
        let _ = self.gemm_d2d(&dummy_h_128, "model.layers.0.mlp.gate_proj.weight", 128);
    }

    // ── constructors ──────────────────────────────────────────────────────────

    /// Dequantize all projection weights from `qw` and upload them to the GPU.
    ///
    /// Calling this once at model-load time eliminates repeated host→device weight
    /// transfers during inference.  RMSNorm weights are also pre-uploaded as 1-D
    /// device tensors so that `forward_device` never copies them per-token.
    pub fn from_quantized(
        qw: &QuantizedWeights,
        config: &ModelConfig,
    ) -> Result<Self, tempura_kernels::KernelError> {
        let matmul = CutileMatMul::new()?;
        let flash_attn = CutileFlashAttention::new()?;
        let mut dw: HashMap<String, DeviceWeight> = HashMap::new();
        let mut dw_q4k: HashMap<String, DeviceWeightQ4K> = HashMap::new();
        let mut dw_q2: HashMap<String, DeviceWeightQ2> = HashMap::new();
        let mut dw_q6k: HashMap<String, DeviceWeightQ6K> = HashMap::new();
        let mut nw: HashMap<String, Arc<Tensor<f32>>> = HashMap::new();

        let h = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let h_pad = next_pow2(h.div_ceil(BK) * BK);
        let q_pad = next_pow2(q_dim.div_ceil(BK) * BK);
        let inter_pad = next_pow2(inter.div_ceil(BK) * BK);

        // lm_head  [vocab, h] — typically not Q4_K/Q2_0; keep dequantized path.
        Self::maybe_upload(&matmul, &mut dw, qw, "lm_head.weight", vocab, h)?;

        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");

            // For each projection: try Q4_K GPU-resident first, then Q2_0, then f32.
            // The inner block avoids a closure so all three maps can be borrowed mutably.
            let mut try_proj = |name: String, n: usize, k: usize| {
                (|| -> Result<(), tempura_kernels::KernelError> {
                    // Priority: Q4_K → Q6_K → Q2_0 → f32 dequantized.
                    let uploaded_q4k =
                        Self::maybe_upload_q4k(matmul.stream(), &mut dw_q4k, qw, &name, n, k)?;
                    if uploaded_q4k {
                        return Ok(());
                    }
                    let uploaded_q6k =
                        Self::maybe_upload_q6k(matmul.stream(), &mut dw_q6k, qw, &name, n, k)?;
                    if uploaded_q6k {
                        return Ok(());
                    }
                    let uploaded_q2 =
                        Self::maybe_upload_q2(matmul.stream(), &mut dw_q2, qw, &name, n, k)?;
                    if !uploaded_q2 {
                        Self::maybe_upload(&matmul, &mut dw, qw, &name, n, k)?;
                    }
                    Ok(())
                })()
            };

            // Q / K / V projections: [q_dim/kv_dim, h]
            try_proj(format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            try_proj(format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            try_proj(format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;

            // O projection: [h, q_dim]
            try_proj(format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;

            // FFN projections: gate/up [inter, h], down [h, inter]
            try_proj(format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            try_proj(format!("{p}.mlp.up_proj.weight"), inter, h)?;
            try_proj(format!("{p}.mlp.down_proj.weight"), h, inter)?;

            // RMSNorm weights — pre-upload so forward_device never copies them per token.
            if let Some(w) = qw.get_f32(&format!("{p}.input_layernorm.weight")) {
                Self::upload_norm_weight(
                    matmul.stream(),
                    &mut nw,
                    &format!("{p}.input_layernorm.weight"),
                    w,
                    h,
                    h_pad,
                )?;
            }
            if let Some(w) = qw.get_f32(&format!("{p}.post_attention_layernorm.weight")) {
                Self::upload_norm_weight(
                    matmul.stream(),
                    &mut nw,
                    &format!("{p}.post_attention_layernorm.weight"),
                    w,
                    h,
                    h_pad,
                )?;
            }
        }

        // Final norm weight.
        if let Some(w) = qw.get_f32("model.norm.weight") {
            Self::upload_norm_weight(matmul.stream(), &mut nw, "model.norm.weight", w, h, h_pad)?;
        }

        let q4k_gemv = CutileQ4Gemv::from_stream(matmul.stream().clone());
        let q2_gemv = CutileQ2Gemv::from_stream(matmul.stream().clone());
        let q6k_gemv = CutileQ6kGemv::from_stream(matmul.stream().clone());
        let cutile_rmsnorm = CutileRmsNorm::new()?;
        let cutile_rope = CutileRoPE::new()?;
        let cutile_elementwise = CutileElementwise::new()?;

        let q4k_count = dw_q4k.len();
        let q6k_count = dw_q6k.len();
        let q2_count = dw_q2.len();
        let f32_count = dw.len();
        if q4k_count > 0 || q6k_count > 0 || q2_count > 0 {
            eprintln!(
                "[cutile] GPU-resident quantized: {q4k_count} Q4_K, {q6k_count} Q6_K, {q2_count} Q2_0, {f32_count} f32"
            );
        }

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh — prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_len×n_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        let rope_cap = 1usize;
        let rope_stream = cutile_rope.stream();
        let q_rope_scratch = RopeScratch::new(rope_stream, rope_cap * n_heads, head_dim)?;
        let k_rope_scratch = RopeScratch::new(rope_stream, rope_cap * n_kv_heads, head_dim)?;

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            device_weights_q4k: dw_q4k,
            device_weights_q2: dw_q2,
            device_weights_q6k: dw_q6k,
            norm_weights: nw,
            matmul,
            q4k_gemv,
            q2_gemv,
            q6k_gemv,
            flash_attn,
            cutile_rmsnorm,
            cutile_rope,
            cutile_elementwise,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
            q_rope_scratch: RefCell::new(q_rope_scratch),
            k_rope_scratch: RefCell::new(k_rope_scratch),
        };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        model.preallocate_scratch();
        eprintln!("[cutile] kernel warmup complete");
        Ok(model)
    }

    /// Upload all projection weights from `mw` (already f32) to the GPU.
    pub fn from_f32(
        mw: &crate::weights::ModelWeights,
        config: &ModelConfig,
    ) -> Result<Self, tempura_kernels::KernelError> {
        let matmul = CutileMatMul::new()?;
        let flash_attn = CutileFlashAttention::new()?;
        let mut dw: HashMap<String, DeviceWeight> = HashMap::new();
        let mut nw: HashMap<String, Arc<Tensor<f32>>> = HashMap::new();

        let h = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let h_pad = next_pow2(h.div_ceil(BK) * BK);
        let q_pad = next_pow2(q_dim.div_ceil(BK) * BK);
        let inter_pad = next_pow2(inter.div_ceil(BK) * BK);

        let mut try_upload = |name: &str, n: usize, k: usize| -> Result<(), _> {
            if let Some(data) = mw.get(name) {
                Self::upload(&matmul, &mut dw, name, data, n, k)?;
            }
            Ok(())
        };

        try_upload("lm_head.weight", vocab, h)?;
        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");
            try_upload(&format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            try_upload(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            try_upload(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            try_upload(&format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            try_upload(&format!("{p}.mlp.gate_proj.weight"), inter, h)?;
            try_upload(&format!("{p}.mlp.up_proj.weight"), inter, h)?;
            try_upload(&format!("{p}.mlp.down_proj.weight"), h, inter)?;

            if let Some(w) = mw.get(&format!("{p}.input_layernorm.weight")) {
                Self::upload_norm_weight(
                    matmul.stream(),
                    &mut nw,
                    &format!("{p}.input_layernorm.weight"),
                    w,
                    h,
                    h_pad,
                )?;
            }
            if let Some(w) = mw.get(&format!("{p}.post_attention_layernorm.weight")) {
                Self::upload_norm_weight(
                    matmul.stream(),
                    &mut nw,
                    &format!("{p}.post_attention_layernorm.weight"),
                    w,
                    h,
                    h_pad,
                )?;
            }
        }

        if let Some(w) = mw.get("model.norm.weight") {
            Self::upload_norm_weight(matmul.stream(), &mut nw, "model.norm.weight", w, h, h_pad)?;
        }

        let q4k_gemv = CutileQ4Gemv::from_stream(matmul.stream().clone());
        let q2_gemv = CutileQ2Gemv::from_stream(matmul.stream().clone());
        let q6k_gemv = CutileQ6kGemv::from_stream(matmul.stream().clone());
        let cutile_rmsnorm = CutileRmsNorm::new()?;
        let cutile_rope = CutileRoPE::new()?;
        let cutile_elementwise = CutileElementwise::new()?;

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh — prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_len×n_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        let rope_cap = 1usize;
        let rope_stream = cutile_rope.stream();
        let q_rope_scratch = RopeScratch::new(rope_stream, rope_cap * n_heads, head_dim)?;
        let k_rope_scratch = RopeScratch::new(rope_stream, rope_cap * n_kv_heads, head_dim)?;

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            device_weights_q4k: HashMap::new(),
            device_weights_q2: HashMap::new(),
            device_weights_q6k: HashMap::new(),
            norm_weights: nw,
            matmul,
            q4k_gemv,
            q2_gemv,
            q6k_gemv,
            flash_attn,
            cutile_rmsnorm,
            cutile_rope,
            cutile_elementwise,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
            q_rope_scratch: RefCell::new(q_rope_scratch),
            k_rope_scratch: RefCell::new(k_rope_scratch),
        };
        eprintln!("[cutile] warming up kernel specializations…");
        model.warmup_gemms();
        model.preallocate_scratch();
        eprintln!("[cutile] kernel warmup complete");
        Ok(model)
    }

    /// Return the CUDA stream used by the GPU model.
    ///
    /// Share this with `CutileKvCache::new` so that KV-cache uploads are
    /// sequenced on the same stream as the GEMM kernels.
    pub fn stream(&self) -> Arc<cuda_core::Stream> {
        self.matmul.stream().clone()
    }

    // ── device tensor helpers ─────────────────────────────────────────────────

    /// Upload a host slice as a device tensor shaped `[rows, cols]`.
    fn to_device(&self, data: &[f32], rows: usize, cols: usize) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .sync_on(stream)
            .expect("CutileGpuModel: host→device upload failed")
            .reshape(&[rows, cols])
            .expect("CutileGpuModel: reshape failed")
            .into()
    }

    /// Upload a host slice to a device tensor shaped `[rows, padded_cols]`, zero-padding
    /// each row from `real_cols` to `padded_cols` on the CPU before the transfer.
    ///
    /// When `padded_cols == real_cols` no temporary allocation is made.
    fn to_device_padded(
        &self,
        data: &[f32],
        rows: usize,
        real_cols: usize,
        padded_cols: usize,
    ) -> Arc<Tensor<f32>> {
        if padded_cols == real_cols {
            return self.to_device(data, rows, padded_cols);
        }
        let mut padded = vec![0.0f32; rows * padded_cols];
        for row in 0..rows {
            padded[row * padded_cols..row * padded_cols + real_cols]
                .copy_from_slice(&data[row * real_cols..(row + 1) * real_cols]);
        }
        self.to_device(&padded, rows, padded_cols)
    }

    fn download_device_tensor(&self, t: &Arc<Tensor<f32>>) -> Vec<f32> {
        t.to_host_vec()
            .sync_on(self.matmul.stream())
            .expect("CutileGpuModel: device→host download failed")
    }

    /// Element-wise add two device tensors of `n` elements on the GPU.
    ///
    /// Returns a new flat 1-D tensor of `n` elements (rounded up to the
    /// nearest 32 for the cuTile tile width).  Both `a` and `b` must have at
    /// least `n` elements.
    ///
    /// When `h_pad == h` for typical model sizes (power-of-2 multiples of BK)
    /// this is a pure GPU operation with zero host round-trips.
    fn add_device(&self, a: &Arc<Tensor<f32>>, b: &Arc<Tensor<f32>>, n: usize) -> Arc<Tensor<f32>> {
        const BLOCK: usize = 32;
        let stream = self.matmul.stream();
        let n_pad = n.div_ceil(BLOCK) * BLOCK;
        let out_t = api::zeros::<f32>(&[n_pad])
            .sync_on(stream)
            .expect("add_device: alloc failed")
            .partition([BLOCK]);
        let (out_t, _, _) = tempura_kernels::cutile_elementwise::kernels::cutile_vector_add(
            out_t,
            a.clone(),
            b.clone(),
        )
        .generics(vec![BLOCK.to_string()])
        .sync_on(stream)
        .expect("add_device: kernel failed");
        Arc::new(out_t.unpartition())
    }

    /// Upload a host slice to a **flat 1-D** device tensor of `rows * padded_cols`
    /// elements, zero-padding each row from `real_cols` to `padded_cols` on the CPU.
    ///
    /// Unlike `to_device_padded` this returns a 1-D tensor rather than `[rows, cols]`,
    /// which is the shape expected by the cuTile RMSNorm kernel partitioning.
    fn to_device_flat_padded(
        &self,
        data: &[f32],
        rows: usize,
        real_cols: usize,
        padded_cols: usize,
    ) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();
        let v: Arc<Vec<f32>> = if padded_cols == real_cols {
            Arc::new(data.to_vec())
        } else {
            let mut v = vec![0.0f32; rows * padded_cols];
            for row in 0..rows {
                v[row * padded_cols..row * padded_cols + real_cols]
                    .copy_from_slice(&data[row * real_cols..(row + 1) * real_cols]);
            }
            Arc::new(v)
        };
        Arc::new(
            api::copy_host_vec_to_device(&v)
                .sync_on(stream)
                .expect("CutileGpuModel: host→device upload (flat) failed"),
        )
    }

    /// Device-to-device GEMM: `output[m, n] = a[m, k_pad] @ b[n, k_pad]^T`.
    ///
    /// The caller MUST ensure `a` has exactly `dw.k_pad` columns.  No download
    /// or re-padding is performed here — the activation must already be padded.
    ///
    /// Falls back silently (returns a zero tensor) if the weight is absent.
    fn gemm_d2d(&self, a: &Arc<Tensor<f32>>, weight_name: &str, m: usize) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[m, 1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        self.matmul
            .forward_bt_device_to_device(a, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("CutileGpuModel::gemm_d2d failed")
    }

    /// Pre-allocate decode output scratch buffers for each projection slot.
    ///
    /// Called once after construction.  Each slot holds a 1-D flat tensor of
    /// n elements (the projection output width for m=1 decode).  On the first
    /// decode forward pass these buffers are zeroed and reused via
    /// `forward_bt_d2d_into` instead of calling `api::zeros` per GEMM.
    fn preallocate_scratch(&self) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        let stream = self.matmul.stream();
        let alloc = |n: usize| -> Option<Arc<Tensor<f32>>> {
            Some(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(stream)
                    .expect("preallocate_scratch: device allocation failed"),
            ))
        };

        let mut pool = self.scratch.borrow_mut();
        pool[SLOT_Q] = alloc(q_dim);
        pool[SLOT_K] = alloc(kv_dim);
        pool[SLOT_V] = alloc(kv_dim);
        pool[SLOT_O] = alloc(h);
        pool[SLOT_GATE] = alloc(inter);
        pool[SLOT_UP] = alloc(inter);
        pool[SLOT_DOWN] = alloc(h);
        pool[SLOT_LOGITS] = alloc(vocab);
    }

    /// Non-blocking variant of `gemm_d2d_slot`.
    ///
    /// Enqueues the GEMM kernel without waiting for it to finish. The caller
    /// MUST call `self.matmul.sync_stream()` before reading the result on the CPU.
    ///
    /// For the pooled decode path (m == 1) the pre-allocated scratch buffer is
    /// reclaimed exactly as in `gemm_d2d_slot`. Prefill (m > 1) or a pool miss
    /// falls back to the synchronous `forward_bt_device_to_device` path.
    ///
    /// # Safety
    ///
    /// The returned tensor's device memory is being written asynchronously. Do
    /// not read it on the CPU until the stream has been synced.
    unsafe fn submit_gemm_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[m, 1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };

        if m == 1 {
            let maybe_tensor = {
                let mut pool = self.scratch.borrow_mut();
                match pool[slot].take() {
                    Some(arc) => match Arc::try_unwrap(arc) {
                        Ok(tensor) => Some(tensor),
                        Err(arc_back) => {
                            pool[slot] = Some(arc_back);
                            None
                        }
                    },
                    None => None,
                }
            };

            if dw.k_pad % 128 == 0 {
                if let Some(tensor) = maybe_tensor {
                    // SAFETY: caller will sync_stream before any CPU read.
                    if let Ok(result) = unsafe {
                        self.matmul
                            .submit_gemv_coalesced_d2d_into(a, &dw.tensor, tensor, dw.n, dw.k_pad)
                    } {
                        self.scratch.borrow_mut()[slot] = Some(result.clone());
                        return result;
                    }
                }
            } else if let Some(tensor) = maybe_tensor {
                // k_pad not divisible by 128 — use strided GEMV (async).
                if let Ok(result) = unsafe {
                    self.matmul
                        .submit_bt_d2d_into(a, &dw.tensor, tensor, m, dw.n, dw.k_pad)
                } {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            }
        }

        // Prefill (m > 1) or pool miss: fall back to the synchronous path.
        self.matmul
            .forward_bt_device_to_device(a, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("submit_gemm_d2d_slot fallback failed")
    }

    /// Device-to-device GEMM/GEMV reusing a pre-allocated scratch buffer.
    ///
    /// For m == 1 (decode): uses `forward_gemv_d2d_into` / `forward_gemv_device_to_device`
    /// which dispatch through `cutile_gemv` — a split-K kernel that avoids the
    /// ~500× MMA slowdown caused by hardware refusing BM=1 tiles.  Each block
    /// handles N_BLOCK_GEMV=64 output neurons via element-wise multiply +
    /// reduce_sum over [64×64] weight tiles, fully utilising warp-level vector
    /// lanes (~target: ~1 TFLOPS vs. ~1.8 GFLOPS scalar fallback).
    ///
    /// For m > 1 (prefill): uses `forward_bt_d2d_into` / `forward_bt_device_to_device`
    /// with the square BM×BN GEMM as before — MMA is efficient for M >= 16.
    ///
    /// When the pool slot holds a uniquely-owned `Arc<Tensor<f32>>`, reclaims
    /// the tensor and reuses the device allocation.  Falls back to a fresh
    /// allocation if the Arc is shared or the slot is empty.
    fn gemm_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[m, 1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };

        // Pooled path: try to reclaim the pre-allocated scratch tensor.
        let maybe_tensor = {
            let mut pool = self.scratch.borrow_mut();
            match pool[slot].take() {
                Some(arc) => match Arc::try_unwrap(arc) {
                    Ok(tensor) => Some(tensor),
                    Err(arc_back) => {
                        pool[slot] = Some(arc_back);
                        None
                    }
                },
                None => None,
            }
        };

        if m == 1 {
            // Decode: use coalesced GEMV kernel — one program per output element,
            // loading BK=128 contiguous elements per iteration from one row of the
            // weight matrix. Falls back to the strided GEMV when k_pad is not
            // divisible by 128 (only for very small K values, uncommon in practice).
            let use_coalesced = dw.k_pad % 128 == 0;
            if use_coalesced {
                if let Some(tensor) = maybe_tensor {
                    if let Ok(result) = self
                        .matmul
                        .forward_gemv_coalesced_d2d_into(a, &dw.tensor, tensor, dw.n, dw.k_pad)
                    {
                        self.scratch.borrow_mut()[slot] = Some(result.clone());
                        return result;
                    }
                }
                return self
                    .matmul
                    .forward_gemv_coalesced_d2d(a, &dw.tensor, dw.n, dw.k_pad)
                    .expect("CutileGpuModel::gemm_d2d_slot coalesced GEMV failed");
            }
            // k_pad not divisible by 128 — fall back to strided GEMV.
            if let Some(tensor) = maybe_tensor {
                if let Ok(result) = self
                    .matmul
                    .forward_gemv_d2d_into(a, &dw.tensor, tensor, dw.n, dw.k_pad)
                {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            }
            return self
                .matmul
                .forward_gemv_device_to_device(a, &dw.tensor, dw.n, dw.k_pad)
                .expect("CutileGpuModel::gemm_d2d_slot GEMV failed");
        }

        // Prefill (m > 1): use square tiled GEMM.
        if let Some(tensor) = maybe_tensor {
            if let Ok(result) = self
                .matmul
                .forward_bt_d2d_into(a, &dw.tensor, tensor, m, dw.n, dw.k_pad)
            {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }
        self.matmul
            .forward_bt_device_to_device(a, &dw.tensor, m, dw.n, dw.k_pad)
            .expect("CutileGpuModel::gemm_d2d_slot GEMM failed")
    }

    /// Device-to-device GEMV for Q2_0 quantized weights (decode, m=1 only).
    ///
    /// Dispatches `cutile_q2_0_gemv` which dequantizes on-GPU. Falls back
    /// silently to a zero tensor when the weight is absent in `device_weights_q2`.
    fn gemm_q2_d2d(&self, a: &Arc<Tensor<f32>>, weight_name: &str) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q2.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        // normed_dev is rank-2 [seq_len, h_pad] from RMSNorm. The Q2_0 GEMV
        // kernel expects rank-1 x. Reshape [1, h_pad] → [h_pad] as a zero-copy
        // metadata operation (no D→H→D round-trip). The kernel reads exactly
        // k_blocks * Q2_0_QK elements, which is ≤ h_pad; the padding is ignored.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q2_0_QK)])
            .expect("gemm_q2_d2d: reshape [1, stride] → [stride] failed");
        self.q2_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("CutileGpuModel::gemm_q2_d2d failed")
    }

    /// Device-to-device GEMV for Q6_K quantized weights (decode, m=1 only).
    ///
    /// Dispatches `cutile_q6k_gemv` which dequantizes on-GPU. Falls back
    /// silently to a zero tensor when the weight is absent in `device_weights_q6k`.
    fn gemm_q6k_d2d(&self, a: &Arc<Tensor<f32>>, weight_name: &str) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q6k.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        // Zero-copy reshape [1, h_pad] → [h_pad]; kernel reads only k_blocks * Q6K_QK.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q6K_QK)])
            .expect("gemm_q6k_d2d: reshape [1, stride] → [stride] failed");
        self.q6k_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("CutileGpuModel::gemm_q6k_d2d failed")
    }

    /// Device-to-device GEMV for Q4_K quantized weights (decode, m=1 only).
    ///
    /// Dispatches `cutile_q4k_gemv` which dequantizes on-GPU.  Falls back
    /// silently to a zero tensor when the weight is absent in `device_weights_q4k`.
    fn gemm_q4k_d2d(&self, a: &Arc<Tensor<f32>>, weight_name: &str) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q4k.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        // Zero-copy reshape [1, h_pad] → [h_pad]; kernel reads only k_blocks * Q4K_BLOCK_SIZE.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::Q4K_BLOCK_SIZE)])
            .expect("gemm_q4k_d2d: reshape [1, stride] → [stride] failed");
        self.q4k_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("CutileGpuModel::gemm_q4k_d2d failed")
    }

    /// Device-to-device Q2_0 GEMV reusing a pre-allocated scratch buffer.
    ///
    /// When the pool slot holds a uniquely-owned `Arc<Tensor<f32>>`, reclaims
    /// the tensor and enqueues the kernel asynchronously (no blocking sync).
    /// Falls back to synchronous `forward_d2d` on a pool miss.
    ///
    /// Because all quantized GEMVs share the matmul CUDA stream, ordering with
    /// subsequent GPU operations (RoPE, etc.) on the same stream is guaranteed.
    /// CPU reads are gated by the first `sync_on` call that follows.
    fn gemm_q2_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q2.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q2_0_QK)])
            .expect("gemm_q2_d2d_slot: reshape failed");

        let maybe_tensor = {
            let mut pool = self.scratch.borrow_mut();
            match pool[slot].take() {
                Some(arc) => match Arc::try_unwrap(arc) {
                    Ok(tensor) => Some(tensor),
                    Err(arc_back) => {
                        pool[slot] = Some(arc_back);
                        None
                    }
                },
                None => None,
            }
        };

        if let Some(tensor) = maybe_tensor {
            // SAFETY: subsequent GPU ops on the same stream see the completed output
            // before any CPU read; the first sync_on on this stream provides the barrier.
            if let Ok(result) = unsafe {
                self.q2_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        self.q2_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("gemm_q2_d2d_slot failed")
    }

    /// Device-to-device Q6_K GEMV reusing a pre-allocated scratch buffer.
    ///
    /// Same async-on-pool-hit / sync-on-miss strategy as `gemm_q2_d2d_slot`.
    fn gemm_q6k_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q6k.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q6K_QK)])
            .expect("gemm_q6k_d2d_slot: reshape failed");

        let maybe_tensor = {
            let mut pool = self.scratch.borrow_mut();
            match pool[slot].take() {
                Some(arc) => match Arc::try_unwrap(arc) {
                    Ok(tensor) => Some(tensor),
                    Err(arc_back) => {
                        pool[slot] = Some(arc_back);
                        None
                    }
                },
                None => None,
            }
        };

        if let Some(tensor) = maybe_tensor {
            if let Ok(result) = unsafe {
                self.q6k_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        self.q6k_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("gemm_q6k_d2d_slot failed")
    }

    /// Device-to-device Q4_K GEMV reusing a pre-allocated scratch buffer.
    ///
    /// Same async-on-pool-hit / sync-on-miss strategy as `gemm_q2_d2d_slot`.
    fn gemm_q4k_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let dw = match self.device_weights_q4k.get(weight_name) {
            Some(d) => d,
            None => {
                let stream = self.matmul.stream();
                return Arc::new(
                    api::zeros::<f32>(&[1])
                        .sync_on(stream)
                        .expect("zeros failed"),
                );
            }
        };
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::Q4K_BLOCK_SIZE)])
            .expect("gemm_q4k_d2d_slot: reshape failed");

        let maybe_tensor = {
            let mut pool = self.scratch.borrow_mut();
            match pool[slot].take() {
                Some(arc) => match Arc::try_unwrap(arc) {
                    Ok(tensor) => Some(tensor),
                    Err(arc_back) => {
                        pool[slot] = Some(arc_back);
                        None
                    }
                },
                None => None,
            }
        };

        if let Some(tensor) = maybe_tensor {
            if let Ok(result) = unsafe {
                self.q4k_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        self.q4k_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("gemm_q4k_d2d_slot failed")
    }

    /// Unified GEMV/GEMM that transparently selects Q4_K, Q6_K, Q2_0 (decode only), or f32.
    ///
    /// For m == 1: checks `device_weights_q4k` first, then `device_weights_q6k`,
    /// then `device_weights_q2`, then falls through to `gemm_d2d_slot` for f32 weights.
    /// The slotted quantized paths use pre-allocated scratch buffers with async dispatch,
    /// eliminating the two `sync_on` calls (allocation + kernel) per GEMV.
    /// For m > 1 (prefill): quantized GEMV is not yet batched; fall back to
    /// host-side dequantization + f32 GEMM (slower but correct; prefill is infrequent).
    fn gemm_any_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        // ── Decode (m == 1): on-GPU dequant GEMV with pre-allocated scratch ───
        if m == 1 && self.device_weights_q4k.contains_key(weight_name) {
            return self.gemm_q4k_d2d_slot(a, weight_name, slot);
        }
        if m == 1 && self.device_weights_q6k.contains_key(weight_name) {
            return self.gemm_q6k_d2d_slot(a, weight_name, slot);
        }
        if m == 1 && self.device_weights_q2.contains_key(weight_name) {
            return self.gemm_q2_d2d_slot(a, weight_name, slot);
        }

        // ── Prefill (m > 1): host dequant + f32 GEMM ─────────────────────────
        if m > 1 {
            // Q4_K prefill: reconstruct f32 from packed GPU format.
            if let Some(dw) = self.device_weights_q4k.get(weight_name) {
                let k = dw.k_blocks * Q4K_BLOCK_SIZE;
                let packed_host: Vec<u32> = Arc::clone(&dw.tensor)
                    .to_host_vec()
                    .sync_on(self.matmul.stream())
                    .expect("Q4_K packed download failed");
                let mut f32_weights = vec![0.0f32; dw.n * k];
                for row in 0..dw.n {
                    for blk in 0..dw.k_blocks {
                        let base = (row * dw.k_blocks + blk) * Q4K_PACKED_U32S;
                        let d = f32::from_bits(packed_host[base]);
                        let dmin = f32::from_bits(packed_host[base + 1]);
                        let k_base = blk * Q4K_BLOCK_SIZE;
                        for j in 0..4usize {
                            let sc_pair = packed_host[base + 2 + j];
                            let m_pair = packed_host[base + 6 + j];
                            let sc_lo = (sc_pair & 0xFF) as f32;
                            let sc_hi = ((sc_pair >> 8) & 0xFF) as f32;
                            let m_lo = (m_pair & 0xFF) as f32;
                            let m_hi = ((m_pair >> 8) & 0xFF) as f32;
                            let d1 = d * sc_lo;
                            let m1 = dmin * m_lo;
                            let d2 = d * sc_hi;
                            let m2 = dmin * m_hi;
                            for l in 0..32usize {
                                let qs_word_idx = (j * 32 + l) / 4;
                                let byte_shift = ((j * 32 + l) % 4) * 8;
                                let qs_word = packed_host[base + 10 + qs_word_idx];
                                let byte_val = (qs_word >> byte_shift) & 0xFF;
                                let lo = (byte_val & 0xF) as f32;
                                let hi = (byte_val >> 4) as f32;
                                f32_weights[row * k + k_base + j * 64 + l] = d1 * lo - m1;
                                f32_weights[row * k + k_base + j * 64 + 32 + l] = d2 * hi - m2;
                            }
                        }
                    }
                }
                let k_pad = next_pow2(k.div_ceil(BK) * BK);
                let w_dev = self.to_device_padded(&f32_weights, dw.n, k, k_pad);
                return self.matmul
                    .forward_bt_device_to_device(a, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q4_K prefill GEMM failed");
            }

            // Q6_K prefill: reconstruct f32 from packed GPU format.
            if let Some(dw) = self.device_weights_q6k.get(weight_name) {
                let k = dw.k_blocks * Q6K_QK;
                let packed_host: Vec<u32> = Arc::clone(&dw.tensor)
                    .to_host_vec()
                    .sync_on(self.matmul.stream())
                    .expect("Q6K packed download failed");
                let mut f32_weights = vec![0.0f32; dw.n * k];
                for row in 0..dw.n {
                    for blk in 0..dw.k_blocks {
                        let base = (row * dw.k_blocks + blk) * Q6K_PACKED_U32S;
                        let d = f32::from_bits(packed_host[base]);
                        let k_base = blk * Q6K_QK;
                        for group in 0..16usize {
                            let sc = f32::from_bits(packed_host[base + 1 + group]);
                            for word in 0..4usize {
                                let word_data = packed_host[base + 17 + group * 4 + word];
                                for byte_pos in 0..4usize {
                                    let q = ((word_data >> (byte_pos * 8)) & 0xFF) as f32 - 32.0;
                                    let idx = row * k + k_base + group * 16 + word * 4 + byte_pos;
                                    f32_weights[idx] = d * sc * q;
                                }
                            }
                        }
                    }
                }
                let k_pad = next_pow2(k.div_ceil(BK) * BK);
                let w_dev = self.to_device_padded(&f32_weights, dw.n, k, k_pad);
                return self.matmul
                    .forward_bt_device_to_device(a, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q6K prefill GEMM failed");
            }

            // Q2_0 prefill: reconstruct f32 from packed GPU format.
            if let Some(dw) = self.device_weights_q2.get(weight_name) {
                let k = dw.k_blocks * Q2_0_QK;
                let packed_host: Vec<u32> = Arc::clone(&dw.tensor)
                    .to_host_vec()
                    .sync_on(self.matmul.stream())
                    .expect("Q2_0 packed download failed");
                let mut f32_weights = vec![0.0f32; dw.n * k];
                for n in 0..dw.n {
                    for blk in 0..dw.k_blocks {
                        let base = (n * dw.k_blocks + blk) * Q2_0_PACKED_U32S;
                        let scale = f32::from_bits(packed_host[base]);
                        let k_base = blk * Q2_0_QK;
                        for word in 0..8usize {
                            let word_data = packed_host[base + 1 + word];
                            for bit_pair in 0..16usize {
                                let q = ((word_data >> (bit_pair * 2)) & 3) as i32 - 1;
                                let idx = n * k + k_base + word * 16 + bit_pair;
                                f32_weights[idx] = q as f32 * scale;
                            }
                        }
                    }
                }
                let k_pad = next_pow2(k.div_ceil(BK) * BK);
                let w_dev = self.to_device_padded(&f32_weights, dw.n, k, k_pad);
                return self.matmul
                    .forward_bt_device_to_device(a, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q2_0 prefill GEMM failed");
            }
        }
        self.gemm_d2d_slot(a, weight_name, m, slot)
    }

    // ── inference ─────────────────────────────────────────────────────────────

    /// Full autoregressive forward pass with GPU-resident KV cache.
    ///
    /// Activations flow device-to-device across all projection GEMMs.
    /// RMSNorm and residual adds run entirely on device.  SiLU still runs on CPU.
    ///
    /// KV cache: K/V are held in the host mirror and passed directly to
    /// `flash_attn.forward`, skipping the redundant upload + immediate re-download
    /// that occurred when using `forward_device`.
    ///
    /// `hidden_dev` is kept on device throughout the layer loop — uploaded once
    /// before the loop and downloaded once after to extract the last token for
    /// the final norm + lm_head.
    ///
    /// The caller is responsible for calling `gpu_kv_cache.advance(token_ids.len())`
    /// and incrementing the session position after this returns.
    pub fn forward(
        &self,
        position: usize,
        token_ids: &[u32],
        qweights: &QuantizedWeights,
        gpu_kv_cache: &mut CutileKvCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let seq_len = token_ids.len();
        let q_dim = n_heads * head_dim;
        let inter = cfg.intermediate_size;

        // ── 1. Embedding lookup (CPU) ─────────────────────────────────────────
        let embed = qweights
            .get_f32("model.embed_tokens.weight")
            .expect("model.embed_tokens.weight not found");
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &tok) in token_ids.iter().enumerate() {
            hidden[t * h..(t + 1) * h]
                .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
        }
        if cfg.embedding_scale {
            let scale = (h as f32).sqrt();
            for v in &mut hidden {
                *v *= scale;
            }
        }

        let positions: Vec<u32> = (position..position + seq_len).map(|p| p as u32).collect();

        // Pre-build weight name strings to avoid allocations in the hot loop.
        #[allow(clippy::type_complexity)]
        let layer_names: Vec<(
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
        )> = (0..cfg.num_layers)
            .map(|l| {
                let p = format!("model.layers.{l}");
                (
                    format!("{p}.input_layernorm.weight"),
                    format!("{p}.post_attention_layernorm.weight"),
                    format!("{p}.self_attn.q_proj.weight"),
                    format!("{p}.self_attn.k_proj.weight"),
                    format!("{p}.self_attn.v_proj.weight"),
                    format!("{p}.self_attn.o_proj.weight"),
                    format!("{p}.mlp.gate_proj.weight"),
                    format!("{p}.mlp.up_proj.weight"),
                    format!("{p}.mlp.down_proj.weight"),
                )
            })
            .collect();

        // ── 2. Layer loop ──────────────────────────────────────────────────────
        // Upload hidden once before the loop.  It stays on device via add_device
        // across all layers; no per-layer re-upload is needed.
        let mut hidden_dev = self.to_device_flat_padded(&hidden, seq_len, h, self.h_pad);

        for layer in 0..cfg.num_layers {
            let (
                ref input_ln_name,
                ref post_ln_name,
                ref q_w_name,
                ref k_w_name,
                ref v_w_name,
                ref o_w_name,
                ref gate_w_name,
                ref up_w_name,
                ref down_w_name,
            ) = layer_names[layer];

            // ── 2a. Pre-attention RMSNorm (GPU) ───────────────────────────────
            // hidden_dev is already on device — no upload needed.
            let input_ln_w_dev = self
                .norm_weights
                .get(input_ln_name)
                .expect("CutileGpuModel: input_layernorm weight not pre-uploaded");
            let normed_dev = self
                .cutile_rmsnorm
                .forward_device_fused(
                    self.matmul.stream(),
                    hidden_dev.clone(),
                    input_ln_w_dev,
                    seq_len,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("CutileGpuModel: pre-attention RMSNorm failed");

            // ── 2b. QKV projections — device-to-device ───────────────────────
            // For Q2_0 models: gemm_any_d2d_slot dequantizes on-GPU (m=1 only).
            // For f32 models: same path as before, synchronous.
            // For prefill (m>1): Q2_0 falls back to dequantized f32 GEMM.
            let q_dev = self.gemm_any_d2d_slot(&normed_dev, q_w_name, seq_len, SLOT_Q);
            let k_dev = self.gemm_any_d2d_slot(&normed_dev, k_w_name, seq_len, SLOT_K);
            let v_dev = self.gemm_any_d2d_slot(&normed_dev, v_w_name, seq_len, SLOT_V);

            // ── 2c. RoPE (GPU) — rotate Q and K entirely on device ───────────
            // cos/sin tables (tiny: seq_len × n_heads × head_dim/2 floats each)
            // are computed on CPU and uploaded.  Q and K are deinterleaved and
            // rotated on the GPU using pre-allocated scratch buffers, eliminating
            // the 5 × 2 api::zeros calls (280 blocking allocs per token at 28 layers).
            let (q_rot_dev, k_rot_dev) = if !cfg.use_alibi {
                let qr = self
                    .cutile_rope
                    .forward_device_scratch(
                        &q_dev,
                        &positions,
                        seq_len,
                        n_heads,
                        head_dim,
                        cfg.rope_theta,
                        &mut self.q_rope_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: GPU RoPE for Q failed");
                let kr = self
                    .cutile_rope
                    .forward_device_scratch(
                        &k_dev,
                        &positions,
                        seq_len,
                        n_kv_heads,
                        head_dim,
                        cfg.rope_theta,
                        &mut self.k_rope_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: GPU RoPE for K failed");
                (qr, kr)
            } else {
                // ALiBi: no rotation — keep Q/K tensors on device as-is.
                (q_dev.clone(), k_dev.clone())
            };

            // ── 2d. GPU KV-cache append (host mirror) ────────────────────────
            // V does not need RoPE, download from device to update host mirror.
            // K was rotated on device; one download for the rotated K host mirror.
            let k_rot_host = self.download_device_tensor(&k_rot_dev);
            let v_host = self.download_device_tensor(&v_dev);
            gpu_kv_cache.append_host(layer, &k_rot_host, &v_host, seq_len);

            // kv_len includes the tokens just appended (seq_len new tokens on
            // top of whatever was already in the cache).
            let kv_len = gpu_kv_cache.seq_len() + seq_len;

            // ── 2e. Attention ─────────────────────────────────────────────────
            let q_rot_host = self.download_device_tensor(&q_rot_dev);
            let (k_slice, v_slice) = gpu_kv_cache.get_kv_host_slices(layer, kv_len);
            let mut attn_output = vec![0.0f32; seq_len * n_heads * head_dim];

            if seq_len == 1 {
                // Decode: 1-token GQA attention is O(kv_len × head_dim) on CPU.
                // Using the GPU flash attention kernel for seq_len=1 costs ~896
                // sync_on CUDA calls per token (32 head H2D/D2H × 28 layers),
                // each with driver overhead that dominates at small tensor sizes.
                // CPU GQA for 1 token: ~500K multiply-adds — essentially free.
                let scale = 1.0 / (head_dim as f32).sqrt();
                let heads_per_group = n_heads / n_kv_heads;
                let mut scores = vec![0.0f32; kv_len];
                for h in 0..n_heads {
                    let kv_h = h / heads_per_group;
                    let q_head = &q_rot_host[h * head_dim..(h + 1) * head_dim];
                    let kv_k = &k_slice[kv_h * head_dim..];
                    let kv_v = &v_slice[kv_h * head_dim..];

                    // scores[t] = Q · K[t] * scale
                    for t in 0..kv_len {
                        let k_tok = &kv_k[t * n_kv_heads * head_dim..t * n_kv_heads * head_dim + head_dim];
                        scores[t] = q_head.iter().zip(k_tok).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }

                    // softmax in-place
                    let max_s = scores[..kv_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = scores[..kv_len].iter_mut().map(|s| { *s = (*s - max_s).exp(); *s }).sum();
                    for s in &mut scores[..kv_len] { *s /= sum; }

                    // out[h] = sum_t scores[t] * V[t]
                    let out_head = &mut attn_output[h * head_dim..(h + 1) * head_dim];
                    for v_d in 0..head_dim { out_head[v_d] = 0.0; }
                    for t in 0..kv_len {
                        let v_tok = &kv_v[t * n_kv_heads * head_dim..t * n_kv_heads * head_dim + head_dim];
                        for d in 0..head_dim {
                            out_head[d] += scores[t] * v_tok[d];
                        }
                    }
                }
            } else {
                // Prefill: use GPU flash attention (amortises CUDA overhead over many rows).
                self.flash_attn
                    .forward_batched(
                        &q_rot_host,
                        k_slice,
                        v_slice,
                        &mut attn_output,
                        1,
                        seq_len,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        1.0 / (head_dim as f32).sqrt(),
                    )
                    .expect("CutileGpuModel: flash attention failed");
            }

            // ── 2f. O projection — device-to-device ──────────────────────────
            // Upload attn_output padded to q_pad for O projection.
            let attn_padded_dev = self.to_device_padded(&attn_output, seq_len, q_dim, self.q_pad);
            let o_dev = self.gemm_any_d2d_slot(&attn_padded_dev, o_w_name, seq_len, SLOT_O);

            // ── 2g. Residual add (GPU) ────────────────────────────────────────
            // o_dev has h columns; pad to h_pad when h_pad > h (rare edge case
            // e.g. h=5120 → h_pad=8192).  For models where h_pad == h (the
            // common case) this is a zero-cost clone.
            let o_padded = if self.h_pad == h {
                o_dev.clone()
            } else {
                let o_host = self.download_device_tensor(&o_dev);
                self.to_device_flat_padded(&o_host, seq_len, h, self.h_pad)
            };
            hidden_dev = self.add_device(&hidden_dev, &o_padded, seq_len * self.h_pad);

            // ── 2h. Post-attention RMSNorm (GPU) ─────────────────────────────
            // hidden_dev is already updated on device — no upload needed.
            let post_ln_w_dev = self
                .norm_weights
                .get(post_ln_name)
                .expect("CutileGpuModel: post_attention_layernorm weight not pre-uploaded");
            let normed2_dev = self
                .cutile_rmsnorm
                .forward_device_fused(
                    self.matmul.stream(),
                    hidden_dev.clone(),
                    post_ln_w_dev,
                    seq_len,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("CutileGpuModel: post-attention RMSNorm failed");

            // ── 2i. FFN gate + up — device-to-device ─────────────────────────
            // For Q2_0 models gemm_any_d2d_slot dispatches on-GPU dequant (synchronous).
            // For f32 models the synchronous gemm_d2d_slot path is used.
            let gate_dev = self.gemm_any_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
            let up_dev = self.gemm_any_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);

            // ── 2j. SiLU gate × up (GPU) ─────────────────────────────────────
            // gate_dev and up_dev already reside on the GPU.  silu_mul_device
            // passes them directly to the GPU kernel — 0 D→H + 0 H→D when
            // inter is B-aligned (true for all standard models).
            // The result has seq_len × inter elements; pad to inter_pad if needed.
            let gate_silu_n_dev = self
                .cutile_elementwise
                .silu_mul_device(&gate_dev, &up_dev, seq_len, inter)
                .expect("CutileGpuModel: GPU SiLU-mul failed");

            // ── 2k. FFN down — device-to-device ──────────────────────────────
            // Pad gate_silu from inter to inter_pad for the down_proj GEMM.
            // When inter == inter_pad (pow-2 models) this is a free reshape.
            let gate_silu_dev = if self.inter_pad == inter {
                gate_silu_n_dev
            } else {
                let gate_silu_host = self.download_device_tensor(&gate_silu_n_dev);
                self.to_device_padded(&gate_silu_host, seq_len, inter, self.inter_pad)
            };
            let down_dev = self.gemm_any_d2d_slot(&gate_silu_dev, down_w_name, seq_len, SLOT_DOWN);

            // ── 2l. Residual add (GPU) ────────────────────────────────────────
            let down_padded = if self.h_pad == h {
                down_dev.clone()
            } else {
                let down_host = self.download_device_tensor(&down_dev);
                self.to_device_flat_padded(&down_host, seq_len, h, self.h_pad)
            };
            hidden_dev = self.add_device(&hidden_dev, &down_padded, seq_len * self.h_pad);
        }

        // ── 3. Final norm + lm_head ────────────────────────────────────────────
        // Download hidden_dev once after the loop.  Each row is h_pad elements;
        // the last token occupies hidden_final[(seq_len-1)*h_pad..seq_len*h_pad].
        let hidden_final = self.download_device_tensor(&hidden_dev);
        let last_flat = Arc::new(
            api::copy_host_vec_to_device(&Arc::new(
                hidden_final[(seq_len - 1) * self.h_pad..seq_len * self.h_pad].to_vec(),
            ))
            .sync_on(self.matmul.stream())
            .expect("CutileGpuModel: last-token upload failed"),
        );
        let final_norm_w_dev = self
            .norm_weights
            .get("model.norm.weight")
            .expect("CutileGpuModel: model.norm.weight not pre-uploaded");
        let last_dev = self
            .cutile_rmsnorm
            .forward_device_fused(
                self.matmul.stream(),
                last_flat,
                final_norm_w_dev,
                1,
                h,
                self.h_pad,
                cfg.layer_norm_eps,
            )
            .expect("CutileGpuModel: final RMSNorm failed");

        let logits_dev = self.gemm_d2d_slot(&last_dev, "lm_head.weight", 1, SLOT_LOGITS);
        self.download_device_tensor(&logits_dev)
    }
}

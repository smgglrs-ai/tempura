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
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_core::sys::CUdeviceptr;
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
use tempura_kernels::oxide_q2_gemv_host::OxideQ2Gemv;
use tempura_kernels::oxide_q4k_gemv_host::OxideQ4kGemv;
use tempura_kernels::oxide_q6k_gemv_host::OxideQ6kGemv;
use tempura_kernels::oxide_decode_attn_host::OxideDecodeAttn;
use tempura_kernels::{Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};
use tempura_kernels::cutile_rmsnorm::{CutileRmsNorm, RmsnormScratch};
use tempura_kernels::cutile_rope::{CutileRoPE, RopeScratch};

use crate::config::ModelConfig;
use crate::cutile_kv_cache::CutileKvCache;
use crate::weights::QuantizedWeights;

/// Info about a KV-write memcpy node so we can update its dst pointer per step.
#[derive(Clone)]
struct KvMemcpyEntry {
    node: cuda_core::sys::CUgraphNode,
    src_ptr: cuda_core::sys::CUdeviceptr,
    cache_base: cuda_core::sys::CUdeviceptr, // base of k_device[layer] or v_device[layer]
    width_bytes: usize,
}

/// A kernel node in the captured graph whose scalar (i32) parameter must be
/// updated each decode step without re-capture.
///
/// cuTile kernels pack all parameters into a flat byte buffer passed via the
/// CUDA `extra` launch mechanism (`CU_LAUNCH_PARAM_BUFFER_POINTER`).  After
/// `cuGraphKernelNodeGetParams_v2` the `extra[1]` field points directly into
/// the graph node's owned parameter storage.  Writing to `scalar_ptr` modifies
/// that storage in place; the change is visible to `cuGraphExecUpdate_v2`.
struct KernelScalarNode {
    node: cuda_core::sys::CUgraphNode,
    /// Direct pointer into the graph node's extra-buffer at the i32 scalar's
    /// byte offset.  Valid for the lifetime of the owning `DecodeGraph`
    /// (which keeps `cu_graph` alive).
    scalar_ptr: *mut i32,
}

// SAFETY: raw pointers are only accessed from the single inference thread.
unsafe impl Send for KernelScalarNode {}
unsafe impl Sync for KernelScalarNode {}

/// RAII wrapper for a CUDA graph executable.
///
/// The exec is replayed each decode step via `cuGraphLaunch`.  KV-write memcpy
/// nodes are updated per step via `cuGraphExecMemcpyNodeSetParams`.  Flash-attn
/// kv_len and RoPE position nodes are updated via `cuGraphExecKernelNodeSetParams_v2`.
/// No re-capture, no `cuGraphExecUpdate_v2`.
struct DecodeGraph {
    cu_exec: cuda_core::sys::CUgraphExec,
    cu_graph: cuda_core::sys::CUgraph, // kept alive; node handles reference it
    device: Arc<cuda_core::Device>,
    /// KV write memcpy nodes to update per decode step (one K + one V per layer).
    kv_memcpy_entries: Vec<KvMemcpyEntry>,
    /// Flash-attention kernel nodes whose kv_len scalar changes each step.
    flash_attn_nodes: Vec<KernelScalarNode>,
    /// RoPE kernel nodes whose position scalar changes each step.
    rope_nodes: Vec<KernelScalarNode>,
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        if !self.cu_exec.is_null() {
            let _ = unsafe { cuda_core::sys::cuGraphExecDestroy(self.cu_exec) };
        }
        if !self.cu_graph.is_null() {
            let _ = unsafe { cuda_core::sys::cuGraphDestroy(self.cu_graph) };
        }
    }
}

// SAFETY: the exec handle is only ever used from a single inference thread.
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

// BK must be a multiple that divides all standard hidden / intermediate
// dimensions (4096, 11008, 14336, …).  16 matches the tiling constant used
// inside CutileMatMul::forward_bt_device_to_device.
const BK: usize = 16;

// BN_GEMV: number of output neurons computed per GPU program in the multi-output
// quantized GEMV kernels (cutile_q*_gemv_bn).  Matches a CUDA warp (32 threads)
// so each program covers exactly one warp's worth of outputs, maximising occupancy.
// Standard transformer dimensions (4096, 8192, 11008, 14336, …) are all multiples
// of 32, so this value works for all projection shapes we encounter.
const BN_GEMV: usize = 32;

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
    /// Hand-optimized oxide Q2_0 GEMV kernel (replaces cuTile for M=1 decode).
    ///
    /// `None` when the `oxide-gemv` feature was not compiled in or when the
    /// oxide build step failed; falls back to `q2_gemv` (cuTile) in that case.
    oxide_q2_gemv: Option<OxideQ2Gemv>,
    /// Oxide decode flash-attention kernel — accepts kv_len as device pointer,
    /// enabling single-capture CUDA graph without stale kv_len.
    oxide_decode_attn: Option<OxideDecodeAttn>,
    /// Hand-optimized oxide Q4_K GEMV kernel (replaces cuTile for M=1 decode).
    oxide_q4k_gemv: Option<OxideQ4kGemv>,
    /// Hand-optimized oxide Q6_K GEMV kernel (replaces cuTile for M=1 decode).
    oxide_q6k_gemv: Option<OxideQ6kGemv>,
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
    /// Pre-allocated RMSNorm scratch for pre-attention norm (seq_len=1 / decode).
    pre_attn_norm_scratch: RefCell<RmsnormScratch>,
    /// Pre-allocated RMSNorm scratch for post-attention norm (seq_len=1 / decode).
    post_attn_norm_scratch: RefCell<RmsnormScratch>,
    /// Pre-allocated output buffer for the post-attention residual add (seq_len=1).
    ///
    /// Eliminates the `api::zeros.sync_on` allocation inside `add_device` for
    /// the 28 residual adds per decode token after the attention output.
    post_attn_add_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    /// Pre-allocated output buffer for the post-FFN residual add (seq_len=1).
    ffn_add_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    /// Pre-allocated output buffer for `decode_attention_device_scratch` (seq_len=1).
    ///
    /// Eliminates the `api::zeros.sync_on` (= full stream sync + alloc) inside
    /// `decode_attention_device` — 28 blocked allocs per decode token become zero.
    attn_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    /// Pre-allocated output buffer for `silu_mul_device_scratch` (seq_len=1).
    ///
    /// Same motivation as `attn_out_scratch`: eliminates 28 sync-on allocs per token.
    silu_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,

    /// Pre-allocated zero-padded scratch for down-projection input when inter_pad != inter.
    ///
    /// For models with non-power-of-2 intermediate sizes (e.g. Llama-3.1-8B inter=14336,
    /// inter_pad=16384), the SiLU output needs to be padded from `inter` to `inter_pad`
    /// before the down projection matmul.  Without this, the path does a D→H download +
    /// H→D re-upload (with a stream sync) every layer — 32 stalls per decode token.
    ///
    /// This scratch holds `inter_pad` f32 elements.  Positions `0..inter` are overwritten
    /// by a GPU-side D→D copy from the SiLU output each layer; positions `inter..inter_pad`
    /// remain zero from the initial allocation (never written after init).
    down_proj_pad_scratch: RefCell<Option<Arc<Tensor<f32>>>>,

    // ── CUDA graph capture fields ─────────────────────────────────────────────
    //
    // For decode (seq_len=1) the forward pass is captured into a CUDA graph on
    // the first step and replayed on subsequent steps.  Before each replay the
    // baked-in per-step kernel parameters (position, kv_len, KV write offset)
    // are refreshed via `cuGraphExecUpdate_v2`, which re-captures a lightweight
    // template on the same stream and patches the exec in-place — one driver call
    // instead of ~200 individual `cuLaunchKernel` submissions.

    /// Pre-allocated device buffer that holds the embedding of the current decode
    /// token.  The CPU computes the embedding, copies it here with
    /// `cuda_core::memcpy_htod_async` BEFORE `cuStreamBeginCapture`, so the graph
    /// reads the latest value on every replay.
    d_embed: Arc<Tensor<f32>>,

    // ── Device-side decode scalars for single-capture CUDA graph (Opt 5) ──────
    //
    // Written H→D before each graph replay so the graph can be captured once
    // and replayed N times without cuStreamBeginCapture per token.
    /// kv_len after the current step's KV append (f32 for cuTile compatibility; read by flash attention).
    d_kv_len: Arc<Tensor<f32>>,
    /// Current token position for RoPE (f32 for cuTile compatibility).
    d_positions: Arc<Tensor<f32>>,
    /// KV cache write slot before the current step's append (raw i32, not a cuTile tensor).
    d_kv_write_offset: Arc<Tensor<i32>>,

    /// Pre-allocated RMSNorm scratch for the final model norm applied after the
    /// layer loop (seq_len=1 decode only).  Eliminates the three `api::zeros`
    /// allocations inside `forward_device_fused` that would otherwise block
    /// CUDA graph capture.
    final_norm_scratch: RefCell<RmsnormScratch>,

    /// Cached CUDA graph executable for the decode forward pass.
    ///
    /// `None` until the first decode step.  On each decode step the graph is
    /// re-captured (template) and the exec is updated via `cuGraphExecUpdate_v2`,
    /// allowing per-step parameters (pos, kv_len, KV cache write offset) to vary
    /// while the graph topology stays fixed.
    cached_decode_graph: RefCell<Option<DecodeGraph>>,
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

    /// Upload Q4K weights in raw GGUF 36-u32 format (for q4k_gemv_decode_q8_v3r kernel).
    /// 14% less DRAM than the 42-u32 packed format (144 bytes vs 168 bytes per block).
    fn maybe_upload_q4k_raw(
        stream: &std::sync::Arc<cuda_core::Stream>,
        device_weights_q4k: &mut HashMap<String, DeviceWeightQ4K>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        let bytes = match qw.get_q4k(name) { Some(b) => b, None => return Ok(false) };
        debug_assert_eq!(k % Q4K_BLOCK_SIZE, 0);
        let k_blocks = k / Q4K_BLOCK_SIZE;
        // Each raw GGUF block is exactly 144 bytes = 36 u32s.
        debug_assert_eq!(bytes.len(), n * k_blocks * 144,
            "Q4K raw bytes len mismatch: {} != {}", bytes.len(), n * k_blocks * 144);
        // Reinterpret raw bytes as u32 (blocks are 4-byte aligned in GGUF).
        // Alignment check: GGUF tensor data must be 4-byte aligned for u32 reinterpretation.
        let ptr_offset = bytes.as_ptr() as usize % 4;
        if std::env::var("TEMPURA_DEBUG_V3R").is_ok() || ptr_offset != 0 {
            eprintln!("[v3r] {} bytes={} align_offset={} first_block_u32[0..4]={:08x?}",
                name, bytes.len(), ptr_offset,
                [u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                 u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                 u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                 u32::from_le_bytes(bytes[12..16].try_into().unwrap())]);
        }
        // Use safe byte-by-byte copy to avoid alignment UB if ptr is not 4-byte aligned.
        let raw_u32: Vec<u32> = bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let tensor: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(raw_u32))
            .sync_on(stream)
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[n * k_blocks * 36])
            .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
            .into();
        device_weights_q4k.insert(name.to_string(), DeviceWeightQ4K { tensor, n, k_blocks });
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
        // Load oxide Q4K kernel BEFORE weight upload so uses_raw_gguf can guide the format choice.
        let oxide_q4k_gemv_early = OxideQ4kGemv::try_load(matmul.stream().clone());
        let upload_raw_q4k = oxide_q4k_gemv_early.as_ref().map(|o| o.uses_raw_gguf).unwrap_or(false);
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
                    // Use raw GGUF 36-u32 format if the v3r kernel supports it.
                    let uploaded_q4k = if upload_raw_q4k {
                        Self::maybe_upload_q4k_raw(matmul.stream(), &mut dw_q4k, qw, &name, n, k)?
                    } else {
                        Self::maybe_upload_q4k(matmul.stream(), &mut dw_q4k, qw, &name, n, k)?
                    };
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
        let mut cutile_rope = CutileRoPE::new()?;
        let cutile_elementwise = CutileElementwise::new()?;

        // Attempt to load the oxide-compiled Q2_0, Q4_K, and Q6_K GEMV kernels.
        let oxide_q2_gemv = OxideQ2Gemv::try_load(matmul.stream().clone());
        let oxide_q4k_gemv = oxide_q4k_gemv_early;   // reuse the early-loaded instance (upload format already matched)
        let oxide_q6k_gemv = OxideQ6kGemv::try_load(matmul.stream().clone());
        // Oxide decode attention (kv_len read from device pointer at execution time).
        let oxide_decode_attn = match OxideDecodeAttn::new(matmul.stream().clone()) {
            Ok(oda) => { eprintln!("[cutile] oxide decode-attn loaded (batch={})", oda.has_batch()); Some(oda) }
            Err(e)  => { eprintln!("[cutile] oxide decode-attn unavailable: {e}"); None }
        };

        let q4k_count = dw_q4k.len();
        let q6k_count = dw_q6k.len();
        let q2_count = dw_q2.len();
        let f32_count = dw.len();
        if q4k_count > 0 || q6k_count > 0 || q2_count > 0 {
            eprintln!(
                "[cutile] GPU-resident quantized: {q4k_count} Q4_K, {q6k_count} Q6_K, {q2_count} Q2_0, {f32_count} f32"
            );
        }

        // Pre-compute the full RoPE frequency table on the GPU once.
        // During decode, `forward_device_scratch` reads from this table by
        // position index — zero H2D uploads per token (was 2 sync H2D per RoPE
        // call, 4 calls per layer × 28 layers = 112 sync_on per token).
        cutile_rope.precompute_table(
            config.max_seq_len.min(4096),
            head_dim,
            config.rope_theta,
            matmul.stream(),
        )?;

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh — prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_len×n_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        // Use the shared matmul stream so RoPE scratch buffers are on the same stream
        // as all other GPU operations — eliminates cross-stream synchronization overhead.
        let rope_cap = 1usize;
        let q_rope_scratch = RopeScratch::new(matmul.stream(), rope_cap * n_heads, head_dim)?;
        let k_rope_scratch = RopeScratch::new(matmul.stream(), rope_cap * n_kv_heads, head_dim)?;
        let pre_attn_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let post_attn_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let final_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let add_alloc = |n: usize| -> Option<Arc<Tensor<f32>>> {
            Some(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(matmul.stream())
                    .expect("add_device scratch: device allocation failed"),
            ))
        };
        let post_attn_add_scratch = add_alloc(h_pad);
        let ffn_add_scratch = add_alloc(h_pad);
        // Attention output scratch: q_dim = n_heads × head_dim (seq_len=1, no padding).
        let attn_out_scratch = add_alloc(q_dim);
        // SiLU-mul output scratch: inter rounded up to next multiple of 32 (the B
        // constant used inside silu_mul_device_scratch).
        const SILU_B: usize = 32;
        let silu_out_scratch = add_alloc(inter.div_ceil(SILU_B) * SILU_B);

        // Pre-allocate the zero-padded down-projection input scratch (only needed
        // when inter_pad != inter, e.g. Llama-3.1-8B inter=14336, inter_pad=16384).
        // Positions 0..inter are overwritten by D→D copy each layer; inter..inter_pad stay 0.
        let down_proj_pad_scratch = if inter_pad != inter {
            add_alloc(inter_pad)
        } else {
            None
        };

        // Pre-allocate the embedding input buffer for CUDA graph decode.
        let d_embed = Arc::new(
            api::zeros::<f32>(&[h_pad])
                .sync_on(matmul.stream())
                .expect("d_embed: device allocation failed"),
        );

        // Device-side decode scalars for single-capture CUDA graph (Opt 5).
        let d_kv_len = Arc::new(
            api::zeros::<f32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_kv_len alloc failed"),
        );
        let d_positions = Arc::new(
            api::zeros::<f32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_positions alloc failed"),
        );
        let d_kv_write_offset = Arc::new(
            api::zeros::<i32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_kv_write_offset alloc failed"),
        );

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
            oxide_q2_gemv,
            oxide_decode_attn,
            oxide_q4k_gemv,
            oxide_q6k_gemv,
            flash_attn,
            cutile_rmsnorm,
            cutile_rope,
            cutile_elementwise,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
            q_rope_scratch: RefCell::new(q_rope_scratch),
            k_rope_scratch: RefCell::new(k_rope_scratch),
            pre_attn_norm_scratch: RefCell::new(pre_attn_norm_scratch),
            post_attn_norm_scratch: RefCell::new(post_attn_norm_scratch),
            post_attn_add_scratch: RefCell::new(post_attn_add_scratch),
            ffn_add_scratch: RefCell::new(ffn_add_scratch),
            attn_out_scratch: RefCell::new(attn_out_scratch),
            silu_out_scratch: RefCell::new(silu_out_scratch),
            down_proj_pad_scratch: RefCell::new(down_proj_pad_scratch),
            d_embed,
            d_kv_len,
            d_positions,
            d_kv_write_offset,
            final_norm_scratch: RefCell::new(final_norm_scratch),
            cached_decode_graph: RefCell::new(None),
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
        let mut cutile_rope = CutileRoPE::new()?;
        let cutile_elementwise = CutileElementwise::new()?;

        // Attempt to load the oxide-compiled Q2_0, Q4_K, and Q6_K GEMV kernels.
        let oxide_q2_gemv = OxideQ2Gemv::try_load(matmul.stream().clone());
        let oxide_q4k_gemv = OxideQ4kGemv::try_load(matmul.stream().clone());
        let oxide_q6k_gemv = OxideQ6kGemv::try_load(matmul.stream().clone());
        let oxide_decode_attn = match OxideDecodeAttn::new(matmul.stream().clone()) {
            Ok(oda) => { eprintln!("[cutile] oxide decode-attn loaded (batch={})", oda.has_batch()); Some(oda) }
            Err(e)  => { eprintln!("[cutile] oxide decode-attn unavailable: {e}"); None }
        };

        // Pre-compute the full RoPE frequency table on the GPU once.
        cutile_rope.precompute_table(
            config.max_seq_len.min(4096),
            head_dim,
            config.rope_theta,
            matmul.stream(),
        )?;

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh — prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_len×n_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        // Use the shared matmul stream so RoPE scratch buffers are on the same stream
        // as all other GPU operations — eliminates cross-stream synchronization overhead.
        let rope_cap = 1usize;
        let q_rope_scratch = RopeScratch::new(matmul.stream(), rope_cap * n_heads, head_dim)?;
        let k_rope_scratch = RopeScratch::new(matmul.stream(), rope_cap * n_kv_heads, head_dim)?;
        let pre_attn_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let post_attn_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let final_norm_scratch = RmsnormScratch::new(matmul.stream(), 1, h_pad)?;
        let add_alloc = |n: usize| -> Option<Arc<Tensor<f32>>> {
            Some(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(matmul.stream())
                    .expect("add_device scratch: device allocation failed"),
            ))
        };
        let post_attn_add_scratch = add_alloc(h_pad);
        let ffn_add_scratch = add_alloc(h_pad);
        let attn_out_scratch = add_alloc(q_dim);
        const SILU_B: usize = 32;
        let silu_out_scratch = add_alloc(inter.div_ceil(SILU_B) * SILU_B);

        let down_proj_pad_scratch = if inter_pad != inter {
            add_alloc(inter_pad)
        } else {
            None
        };

        // Pre-allocate the embedding input buffer for CUDA graph decode.
        let d_embed = Arc::new(
            api::zeros::<f32>(&[h_pad])
                .sync_on(matmul.stream())
                .expect("d_embed: device allocation failed"),
        );

        // Device-side decode scalars for single-capture CUDA graph (Opt 5).
        let d_kv_len = Arc::new(
            api::zeros::<f32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_kv_len alloc failed"),
        );
        let d_positions = Arc::new(
            api::zeros::<f32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_positions alloc failed"),
        );
        let d_kv_write_offset = Arc::new(
            api::zeros::<i32>(&[1])
                .sync_on(matmul.stream())
                .expect("d_kv_write_offset alloc failed"),
        );

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
            oxide_q2_gemv,
            oxide_decode_attn,
            oxide_q4k_gemv,
            oxide_q6k_gemv,
            flash_attn,
            cutile_rmsnorm,
            cutile_rope,
            cutile_elementwise,
            scratch: RefCell::new(std::array::from_fn(|_| None)),
            q_rope_scratch: RefCell::new(q_rope_scratch),
            k_rope_scratch: RefCell::new(k_rope_scratch),
            pre_attn_norm_scratch: RefCell::new(pre_attn_norm_scratch),
            post_attn_norm_scratch: RefCell::new(post_attn_norm_scratch),
            post_attn_add_scratch: RefCell::new(post_attn_add_scratch),
            ffn_add_scratch: RefCell::new(ffn_add_scratch),
            attn_out_scratch: RefCell::new(attn_out_scratch),
            silu_out_scratch: RefCell::new(silu_out_scratch),
            down_proj_pad_scratch: RefCell::new(down_proj_pad_scratch),
            d_embed,
            d_kv_len,
            d_positions,
            d_kv_write_offset,
            final_norm_scratch: RefCell::new(final_norm_scratch),
            cached_decode_graph: RefCell::new(None),
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
        // SAFETY: kernel queued async; GPU stream ordering ensures a→b→add→next.
        // Caller (or downstream D→H) provides the CPU sync before any host read.
        let (out_t, _, _) = unsafe {
            tempura_kernels::cutile_elementwise::kernels::cutile_vector_add(
                out_t,
                a.clone(),
                b.clone(),
            )
            .generics(vec![BLOCK.to_string()])
            .async_on(stream)
            .expect("add_device: kernel failed")
        };
        Arc::new(out_t.unpartition())
    }

    /// Residual add reusing a pre-allocated output buffer.
    ///
    /// For decode (n = h_pad, a power-of-2 value) the scratch buffer holds the
    /// previous call's result.  `Arc::try_unwrap` reclaims it once the caller's
    /// `hidden_dev` local is reassigned, eliminating the `api::zeros.sync_on`
    /// allocation overhead.  Pool miss falls back to a fresh allocation.
    fn add_device_scratch(
        &self,
        a: &Arc<Tensor<f32>>,
        b: &Arc<Tensor<f32>>,
        n: usize,
        scratch: &mut Option<Arc<Tensor<f32>>>,
    ) -> Arc<Tensor<f32>> {
        const BLOCK: usize = 32;
        let stream = self.matmul.stream();
        let n_pad = n.div_ceil(BLOCK) * BLOCK;
        let out_tensor = match scratch.take() {
            Some(arc) => match Arc::try_unwrap(arc) {
                Ok(tensor) => tensor,
                Err(arc_back) => {
                    *scratch = Some(arc_back);
                    api::zeros::<f32>(&[n_pad])
                        .sync_on(stream)
                        .expect("add_device_scratch: alloc failed")
                }
            },
            None => api::zeros::<f32>(&[n_pad])
                .sync_on(stream)
                .expect("add_device_scratch: alloc failed"),
        };
        let out_part = out_tensor.partition([BLOCK]);
        // SAFETY: kernel queued async; GPU stream ordering ensures a+b is visible
        // to subsequent kernels.  Caller (or downstream D→H) syncs before CPU read.
        let (out_part, _, _) = unsafe {
            tempura_kernels::cutile_elementwise::kernels::cutile_vector_add(
                out_part,
                a.clone(),
                b.clone(),
            )
            .generics(vec![BLOCK.to_string()])
            .async_on(stream)
            .expect("add_device_scratch: kernel failed")
        };
        let result = Arc::new(out_part.unpartition());
        *scratch = Some(result.clone());
        result
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

    /// Prepare the attention output on device for the O-projection GEMM.
    ///
    /// The decode attention kernel returns a flat `[q_dim]` tensor.  The O-proj
    /// GEMM expects `[1, q_pad]`.  When `q_pad == q_dim` a zero-copy reshape is
    /// done.  Otherwise a zero-padded `[1, q_pad]` buffer is allocated on the
    /// GPU and the attention output is written into it via D→D copy — no PCIe.
    fn attn_out_to_oproj_device(
        &self,
        attn_out: &Arc<Tensor<f32>>,
        q_dim: usize,
        q_pad: usize,
    ) -> Arc<Tensor<f32>> {
        self.attn_out_to_oproj_device_m(attn_out, q_dim, q_pad, 1)
    }

    fn attn_out_to_oproj_device_m(
        &self,
        attn_out: &Arc<Tensor<f32>>,
        q_dim: usize,
        q_pad: usize,
        m: usize,
    ) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();
        if q_pad == q_dim {
            return attn_out
                .clone()
                .reshape(&[m, q_pad])
                .expect("attn_out_to_oproj_device: reshape failed")
                .into();
        }
        // Allocate zero-filled padded buffer and D→D copy each row's valid prefix.
        let padded: Arc<Tensor<f32>> = Arc::new(
            api::zeros::<f32>(&[m, q_pad])
                .sync_on(stream)
                .expect("attn_out_to_oproj_device: zeros alloc failed"),
        );
        let src_base: CUdeviceptr = attn_out.device_pointer().cu_deviceptr();
        let dst_base: CUdeviceptr = padded.device_pointer().cu_deviceptr();
        for r in 0..m {
            let src = src_base + (r * q_dim * std::mem::size_of::<f32>()) as u64;
            let dst = dst_base + (r * q_pad * std::mem::size_of::<f32>()) as u64;
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(dst, src, q_dim, stream)
                    .expect("attn_out_to_oproj_device: D→D copy failed");
            }
        }
        padded
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

        // Pre-allocate Q8_1 scratch for the oxide Q4K GEMV — must happen before
        // any CUDA graph capture since device allocation on a capturing stream
        // is not permitted (CUDA error 900).
        // max_k = inter_pad (largest activation vector fed to any Q4K projection)
        // max_n = inter     (largest output of any Q4K projection, gate/up proj)
        if let Some(ref oxide) = self.oxide_q4k_gemv {
            let max_k = self.inter_pad;
            let max_n = inter;
            if oxide.has_q8_path() {
                oxide.pre_allocate_q8_scratch(max_k, max_n)
                    .expect("pre_allocate_q8_scratch failed");
            }
        }
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
    /// Uses the oxide-compiled warp-shuffle kernel when available; falls back
    /// to `cutile_q2_0_gemv` otherwise. Falls back silently to a zero tensor
    /// when the weight is absent in `device_weights_q2`.
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
        // Prefer the oxide warp-shuffle kernel; fall back to cuTile.
        if let Some(oxide) = &self.oxide_q2_gemv {
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ2Gemv::forward_d2d failed");
        }
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
        // Prefer the oxide warp-shuffle kernel; fall back to cuTile.
        if let Some(oxide) = &self.oxide_q6k_gemv {
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ6kGemv::forward_d2d failed");
        }
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
        // Prefer oxide with Q8_1 activation quantization (best perf), then v2, then cuTile.
        if let Some(oxide) = &self.oxide_q4k_gemv {
            if oxide.has_q8_path() {
                return oxide
                    .forward_q8_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("OxideQ4kGemv::forward_q8_d2d failed");
            }
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ4kGemv::forward_d2d failed");
        }
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
            if let Some(oxide) = &self.oxide_q2_gemv {
                if let Ok(result) = unsafe {
                    oxide.submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
                } {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            } else if let Ok(result) = unsafe {
                self.q2_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        // Use oxide kernel if available (preferred for M=1 decode).
        if let Some(oxide) = &self.oxide_q2_gemv {
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ2Gemv::forward_d2d failed in gemm_q2_d2d_slot");
        }

        // Use multi-output BN kernel when n is divisible by BN_GEMV: fewer programs,
        // better GPU utilisation.
        if dw.n % BN_GEMV == 0 {
            return self.q2_gemv
                .forward_d2d_bn(&a_flat, &dw.tensor, dw.n, dw.k_blocks, BN_GEMV)
                .expect("gemm_q2_d2d_slot (bn) failed");
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
            if let Some(oxide) = &self.oxide_q6k_gemv {
                if let Ok(result) = unsafe {
                    oxide.submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
                } {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            } else if let Ok(result) = unsafe {
                self.q6k_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        // Use oxide kernel if available (preferred for M=1 decode).
        if let Some(oxide) = &self.oxide_q6k_gemv {
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ6kGemv::forward_d2d failed in gemm_q6k_d2d_slot");
        }

        if dw.n % BN_GEMV == 0 {
            return self.q6k_gemv
                .forward_d2d_bn(&a_flat, &dw.tensor, dw.n, dw.k_blocks, BN_GEMV)
                .expect("gemm_q6k_d2d_slot (bn) failed");
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

        // Q8 path (dp4a) takes priority over the scratch-tensor f32 path —
        // the quantize+dp4a kernel is preferred for M=1 decode throughput.
        if let Some(oxide) = &self.oxide_q4k_gemv {
            if oxide.has_q8_path() {
                return oxide
                    .forward_q8_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("OxideQ4kGemv::forward_q8_d2d failed in gemm_q4k_d2d_slot");
            }
        }

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
            if let Some(oxide) = &self.oxide_q4k_gemv {
                if let Ok(result) = unsafe {
                    oxide.submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
                } {
                    self.scratch.borrow_mut()[slot] = Some(result.clone());
                    return result;
                }
            } else if let Ok(result) = unsafe {
                self.q4k_gemv
                    .submit_d2d_into(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks)
            } {
                self.scratch.borrow_mut()[slot] = Some(result.clone());
                return result;
            }
        }

        // Use oxide f32 kernel if available.
        if let Some(oxide) = &self.oxide_q4k_gemv {
            return oxide
                .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ4kGemv::forward_d2d failed in gemm_q4k_d2d_slot");
        }

        if dw.n % BN_GEMV == 0 {
            return self.q4k_gemv
                .forward_d2d_bn(&a_flat, &dw.tensor, dw.n, dw.k_blocks, BN_GEMV)
                .expect("gemm_q4k_d2d_slot (bn) failed");
        }

        self.q4k_gemv
            .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
            .expect("gemm_q4k_d2d_slot failed")
    }

    /// Unified GEMV/GEMM that transparently selects Q4_K, Q6_K, Q2_0 (decode only), or f32.
    ///
    /// For m == 1: checks `device_weights_q4k` first, then `device_weights_q6k`,
    /// then `device_weights_q2`, then falls through to `gemm_d2d_slot` for f32 weights.
    /// Batched GEMM for M > 1 using M sequential GPU GEMVs — keeps quantized weights
    /// GPU-resident instead of downloading to CPU. Each row of `a` is fed to the
    /// M=1 GEMV path; results are D→D-concatenated into an [m, n] output.
    ///
    /// `row_stride` is the number of f32 elements between consecutive rows in `a`.
    fn gemm_any_d2d_batch(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        row_stride: usize,  // elements per row in `a`
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();

        // For f32 weights there is no PCIe bottleneck — use the existing slot path.
        let is_q4k = self.device_weights_q4k.contains_key(weight_name);
        let is_q6k = self.device_weights_q6k.contains_key(weight_name);
        let is_q2  = self.device_weights_q2.contains_key(weight_name);
        if !is_q4k && !is_q6k && !is_q2 {
            return self.gemm_any_d2d_slot(a, weight_name, m, slot);
        }

        // Output width for this weight.
        let n = if is_q4k {
            self.device_weights_q4k.get(weight_name).map(|dw| dw.n).unwrap_or(0)
        } else if is_q6k {
            self.device_weights_q6k.get(weight_name).map(|dw| dw.n).unwrap_or(0)
        } else {
            self.device_weights_q2.get(weight_name).map(|dw| dw.n).unwrap_or(0)
        };

        // Colocated GEMM: all M rows in one block → same SM → L2 broadcast for weight accesses.
        // smem GEMM available but slower (barrier overhead > benefit for this block size).
        // Falls back to grid=(N,M) batch GEMV, then M sequential GEMVs.
        const WPB: usize = 4; // WARPS_PER_BLOCK in oxide kernel
        if is_q4k && m * WPB * 32 <= 1024 {
            if let Some(ref oxide) = self.oxide_q4k_gemv {
                let dw = self.device_weights_q4k.get(weight_name).unwrap();
                // Colocated is faster than smem due to lower barrier overhead.
                if let Ok(result) = oxide.forward_colocated_d2d(a, &dw.tensor, m, dw.n, dw.k_blocks) {
                    return result;
                }
                if oxide.has_batch_gemv() {
                    return oxide.forward_batch_d2d(a, &dw.tensor, m, dw.n, dw.k_blocks)
                        .expect("oxide Q4K batch GEMV failed");
                }
            }
        } else if is_q4k { // m too large for colocated kernel
            if let Some(ref oxide) = self.oxide_q4k_gemv {
                let dw = self.device_weights_q4k.get(weight_name).unwrap();
                if oxide.has_batch_gemv() {
                    return oxide.forward_batch_d2d(a, &dw.tensor, m, dw.n, dw.k_blocks)
                        .expect("oxide Q4K batch GEMV failed");
                }
            }
        }

        // Fallback: M sequential GEMVs (no L2 reuse but still GPU-resident).
        let mut rows: Vec<Arc<Tensor<f32>>> = Vec::with_capacity(m);
        for r in 0..m {
            let a_row: Arc<Tensor<f32>> = Arc::new(
                api::zeros::<f32>(&[row_stride])
                    .sync_on(stream)
                    .expect("gemm_any_d2d_batch: a_row alloc"),
            );
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    a_row.device_pointer().cu_deviceptr(),
                    a.device_pointer().cu_deviceptr() + (r * row_stride * 4) as u64,
                    row_stride,
                    stream,
                ).expect("gemm_any_d2d_batch: a_row D→D slice");
            }
            let result = self.gemm_any_d2d_slot(&a_row, weight_name, 1, slot);
            rows.push(result);
        }

        // Concatenate M results [n] → flat [m*n] (rank-1, compatible with cuTile ops).
        let out: Arc<Tensor<f32>> = Arc::new(
            api::zeros::<f32>(&[m * n])
                .sync_on(stream)
                .expect("gemm_any_d2d_batch: output alloc"),
        );
        for (r, row) in rows.iter().enumerate() {
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    out.device_pointer().cu_deviceptr() + (r * n * 4) as u64,
                    row.device_pointer().cu_deviceptr(),
                    n,
                    stream,
                ).expect("gemm_any_d2d_batch: concat D→D");
            }
        }
        out
    }

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
            // Q4_K prefill: reconstruct f32 from GPU format (packed 42-u32 or raw GGUF 36-u32).
            if let Some(dw) = self.device_weights_q4k.get(weight_name) {
                let k = dw.k_blocks * Q4K_BLOCK_SIZE;
                let packed_host: Vec<u32> = Arc::clone(&dw.tensor)
                    .to_host_vec()
                    .sync_on(self.matmul.stream())
                    .expect("Q4_K packed download failed");
                // Detect format from block size: 42 u32 = packed, 36 u32 = raw GGUF.
                let total_blocks = dw.n * dw.k_blocks;
                let u32s_per_block = packed_host.len() / total_blocks;
                let raw_gguf = u32s_per_block == 36;
                let mut f32_weights = vec![0.0f32; dw.n * k];
                for row in 0..dw.n {
                    for blk in 0..dw.k_blocks {
                        let base = (row * dw.k_blocks + blk) * u32s_per_block;
                        let k_base = blk * Q4K_BLOCK_SIZE;
                        let (d, dmin, sc, m_sc) = if raw_gguf {
                            // Raw GGUF: u32[0]=f16(d)|f16(dmin)<<16, u32[1..4]=6-bit scales
                            let dd = packed_host[base];
                            let d_f16 = dd & 0xFFFF;
                            let dmin_f16 = (dd >> 16) & 0xFFFF;
                            let d_v = f16_bits_to_f32_inline(d_f16 as u16);
                            let dmin_v = f16_bits_to_f32_inline(dmin_f16 as u16);
                            let ps0 = packed_host[base + 1];
                            let ps1 = packed_host[base + 2];
                            let ps2 = packed_host[base + 3];
                            // Unpack 6-bit scales: sc[i]=ps[i]&0x3F for i=0..3,
                            //   sc[i+4]=((ps[i+8]&0xF)|((ps[i]>>6)<<4))&0x3F
                            // m[i]=ps[i+4]&0x3F, m[i+4]=((ps[i+8]>>4)|((ps[i+4]>>6)<<4))&0x3F
                            let byte = |word: u32, i: u32| -> u32 { (word >> ((i % 4) * 8)) & 0xFF };
                            let mut sc_arr = [0u8; 8];
                            let mut m_arr = [0u8; 8];
                            for i in 0..4u32 {
                                let p0 = byte(ps0, i);
                                let p4 = byte(ps1, i);
                                let p8 = byte(ps2, i);
                                sc_arr[i as usize]   = (p0 & 0x3F) as u8;
                                sc_arr[4 + i as usize] = (((p8 & 0x0F) | ((p0 >> 6) << 4)) & 0x3F) as u8;
                                m_arr[i as usize]    = (p4 & 0x3F) as u8;
                                m_arr[4 + i as usize]  = ((( p8 >> 4) | ((p4 >> 6) << 4)) & 0x3F) as u8;
                            }
                            (d_v, dmin_v, sc_arr, m_arr)
                        } else {
                            // Packed format: d/dmin as f32 bits, scales as 8-bit pairs
                            let d_v = f32::from_bits(packed_host[base]);
                            let dmin_v = f32::from_bits(packed_host[base + 1]);
                            let mut sc_arr = [0u8; 8];
                            let mut m_arr = [0u8; 8];
                            for j in 0..4usize {
                                let sp = packed_host[base + 2 + j];
                                let mp = packed_host[base + 6 + j];
                                sc_arr[2*j]   = (sp & 0xFF) as u8;
                                sc_arr[2*j+1] = ((sp >> 8) & 0xFF) as u8;
                                m_arr[2*j]    = (mp & 0xFF) as u8;
                                m_arr[2*j+1]  = ((mp >> 8) & 0xFF) as u8;
                            }
                            (d_v, dmin_v, sc_arr, m_arr)
                        };
                        let nibble_base = if raw_gguf { base + 4 } else { base + 10 };
                        for j in 0..4usize {
                            let sc_lo = sc[2*j]   as f32;
                            let sc_hi = sc[2*j+1] as f32;
                            let m_lo  = m_sc[2*j]   as f32;
                            let m_hi  = m_sc[2*j+1] as f32;
                            let d1 = d * sc_lo;
                            let m1 = dmin * m_lo;
                            let d2 = d * sc_hi;
                            let m2 = dmin * m_hi;
                            for l in 0..32usize {
                                let qs_word_idx = (j * 32 + l) / 4;
                                let byte_shift = ((j * 32 + l) % 4) * 8;
                                let qs_word = packed_host[nibble_base + qs_word_idx];
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
    /// Capture or update+replay the CUDA graph for a single decode step.
    ///
    /// # Design
    ///
    /// The forward pass is captured once between `cuStreamBeginCapture` and
    /// `cuStreamEndCapture`.  The resulting template graph is then:
    ///
    /// - **First call**: instantiated with `cuGraphInstantiateWithFlags`.
    /// - **Subsequent calls**: patched with `cuGraphExecUpdate_v2` — same topology,
    ///   updated kernel parameters (position, kv_len, KV-cache write offset).
    ///
    /// The embedding (which changes every step) is uploaded into the pre-allocated
    /// `d_embed` buffer BEFORE capture begins, so the graph can record the fixed
    /// device pointer while the content changes each step.
    ///
    /// After `cuGraphLaunch` + `stream.synchronize()` the logits are resident in
    /// `self.scratch[SLOT_LOGITS]` and are returned as a host Vec.
    fn forward_decode_with_graph(
        &self,
        embedding: &[f32],
        positions: &[u32],
        gpu_kv_cache: &mut CutileKvCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();

        // ── Always: Upload embedding + scalars before capture or launch ──────
        let embed_src: Vec<f32> = if self.h_pad == h {
            embedding.to_vec()
        } else {
            let mut v = vec![0.0f32; self.h_pad];
            v[..h].copy_from_slice(&embedding[..h]);
            v
        };
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_embed.device_pointer().cu_deviceptr(),
                embed_src.as_ptr(),
                self.h_pad,
                stream,
            ).expect("d_embed H→D copy failed");
        }

        // Write device scalars: kv_len, position, kv_write_offset.
        // These are read by device kernels at execution time so the graph is
        // topology-stable across steps (only these tiny H→D copies change).
        let kv_write_off = gpu_kv_cache.seq_len() as i32;
        // Bitcast i32 → f32 so the cuTile kernel can recover via transmute.
        let kv_len_bits: f32 = f32::from_bits((kv_write_off + 1) as u32);
        let pos_bits: f32    = f32::from_bits(positions[0]);
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_kv_len.device_pointer().cu_deviceptr(),
                &kv_len_bits as *const f32, 1, stream,
            ).expect("d_kv_len H→D copy failed");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("d_positions H→D copy failed");
            cuda_core::memcpy_htod_async::<i32>(
                self.d_kv_write_offset.device_pointer().cu_deviceptr(),
                &kv_write_off as *const i32, 1, stream,
            ).expect("d_kv_write_offset H→D copy failed");
        }

        let mut graph_cache = self.cached_decode_graph.borrow_mut();

        if let Some(ref decode_graph) = *graph_cache {
            // ── Subsequent steps: update KV memcpy dst pointers, then launch ─
            // NOTE: seq_len is advanced by the wrapper (kv.advance) after each
            // forward call — do NOT advance again here (would double-count).

            // Update each KV-write memcpy node's dst to the new slot.
            let kv_dim    = cfg.num_kv_heads * cfg.head_dim;
            let f32_bytes = std::mem::size_of::<f32>() as u64;
            let new_dst_offset_bytes = (kv_write_off as u64) * (kv_dim as u64) * f32_bytes;

            // Get CUDA context for cuGraphExecMemcpyNodeSetParams.
            let mut cu_ctx: cuda_core::sys::CUcontext = std::ptr::null_mut();
            unsafe { cuda_core::sys::cuCtxGetCurrent(&mut cu_ctx) };

            for entry in &decode_graph.kv_memcpy_entries {
                let new_dst = entry.cache_base + new_dst_offset_bytes;
                let params = cuda_core::sys::CUDA_MEMCPY3D {
                    srcXInBytes: 0, srcY: 0, srcZ: 0, srcLOD: 0,
                    srcMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
                    srcHost: std::ptr::null(),
                    srcDevice: entry.src_ptr,
                    srcArray: std::ptr::null_mut(),
                    reserved0: std::ptr::null_mut(),
                    srcPitch: 0, srcHeight: 0,
                    dstXInBytes: 0, dstY: 0, dstZ: 0, dstLOD: 0,
                    dstMemoryType: cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: new_dst,
                    dstArray: std::ptr::null_mut(),
                    reserved1: std::ptr::null_mut(),
                    dstPitch: 0, dstHeight: 0,
                    WidthInBytes: entry.width_bytes,
                    Height: 1, Depth: 1,
                };
                let rc = unsafe {
                    cuda_core::sys::cuGraphExecMemcpyNodeSetParams(
                        decode_graph.cu_exec,
                        entry.node,
                        &params,
                        cu_ctx,
                    )
                };
                if rc != 0 {
                    eprintln!("[cuda_graph] cuGraphExecMemcpyNodeSetParams failed: {rc}; falling back");
                    drop(graph_cache);
                    return self.forward_decode_nogpu(embedding, positions, gpu_kv_cache);
                }
            }

            // Update flash-attn kv_len and RoPE pos scalars in the graph's
            // owned parameter buffers (direct writes into extra-buffer storage).
            let new_kv_len = kv_write_off + 1;
            let new_pos    = positions[0] as i32;
            let need_exec_update = !decode_graph.flash_attn_nodes.is_empty()
                || !decode_graph.rope_nodes.is_empty();

            for fa in &decode_graph.flash_attn_nodes {
                unsafe { *fa.scalar_ptr = new_kv_len; }
            }
            for rn in &decode_graph.rope_nodes {
                unsafe { *rn.scalar_ptr = new_pos; }
            }

            // If we updated any kernel scalars, propagate changes from the template
            // graph into the exec via cuGraphExecUpdate_v2.  Only changed nodes are
            // re-patched (no topology check cost for unchanged nodes).
            if need_exec_update {
                let mut result_info = cuda_core::sys::CUgraphExecUpdateResultInfo_st {
                    result: cuda_core::sys::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_SUCCESS,
                    errorNode: std::ptr::null_mut(),
                    errorFromNode: std::ptr::null_mut(),
                };
                let rc = unsafe {
                    cuda_core::sys::cuGraphExecUpdate_v2(
                        decode_graph.cu_exec,
                        decode_graph.cu_graph,
                        &mut result_info,
                    )
                };
                if rc != 0 || result_info.result != cuda_core::sys::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_SUCCESS {
                    eprintln!("[cuda_graph] cuGraphExecUpdate_v2 (scalar patch) failed rc={rc} result={}; falling back",
                        result_info.result);
                    drop(graph_cache);
                    return self.forward_decode_nogpu(embedding, positions, gpu_kv_cache);
                }
            }

            // DIAGNOSTIC: read d_kv_len back from device to confirm H→D write landed
            if std::env::var("TEMPURA_DEBUG_KV_LEN").is_ok() {
                unsafe { stream.synchronize().expect("diag sync"); }
                let kv_len_host: Vec<f32> = Arc::clone(&self.d_kv_len)
                    .to_host_vec().sync_on(stream).expect("diag d2h");
                let kv_len_bits = kv_len_host[0].to_bits();
                eprintln!("[diag] d_kv_len bits={kv_len_bits:#010x} → kv_len={kv_len_bits}");
            }
            unsafe { cuda_core::sys::cuGraphLaunch(decode_graph.cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("stream sync after graph launch failed"); }
            drop(graph_cache);
        } else {
            // ── First step: capture graph, instantiate, find nodes, launch ────
            let relaxed = cuda_core::sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;
            unsafe { stream.begin_capture(relaxed).expect("cuStreamBeginCapture failed"); }

            let hidden_dev = self.d_embed.clone();
            self.forward_decode_inner(hidden_dev, positions, gpu_kv_cache);

            let cu_graph = unsafe { stream.end_capture().expect("cuStreamEndCapture failed") };

            let mut cu_exec_uninit = MaybeUninit::<cuda_core::sys::CUgraphExec>::uninit();
            unsafe {
                cuda_core::sys::cuGraphInstantiateWithFlags(cu_exec_uninit.as_mut_ptr(), cu_graph, 0);
            }
            let cu_exec = unsafe { cu_exec_uninit.assume_init() };

            let capture_kv_len = kv_write_off + 1;
            let capture_pos    = positions[0] as i32;
            let n_kv_heads     = cfg.num_kv_heads as u32;
            let n_heads        = cfg.num_attention_heads as u32;

            eprintln!("[cuda_graph] finding KV memcpy nodes...");
            let kv_memcpy_entries = self.find_kv_memcpy_nodes(cu_graph, gpu_kv_cache);
            eprintln!("[cuda_graph] found {} KV memcpy entries, finding kernel nodes...", kv_memcpy_entries.len());
            let (flash_attn_nodes, rope_nodes) =
                self.find_kernel_scalar_nodes(cu_graph, capture_kv_len, capture_pos, n_heads);
            eprintln!("[cuda_graph] kernel node search done");

            // cu_graph is kept alive inside DecodeGraph so node handles remain valid.
            *graph_cache = Some(DecodeGraph {
                cu_exec,
                cu_graph,
                device: stream.device().clone(),
                kv_memcpy_entries,
                flash_attn_nodes,
                rope_nodes,
            });

            unsafe { cuda_core::sys::cuGraphLaunch(cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("stream sync after first graph launch failed"); }
            drop(graph_cache);
        }

        // Logits reside in SLOT_LOGITS — download and return.
        let logits_arc = self
            .scratch.borrow()[SLOT_LOGITS].clone()
            .expect("SLOT_LOGITS empty after graph launch");
        self.download_device_tensor(&logits_arc)
    }

    /// Enumerate all KERNEL nodes in `cu_graph` and identify flash-attention
    /// (kv_len scalar) and RoPE (position scalar) nodes for per-step parameter
    /// updates without re-capture.
    ///
    /// cuTile kernels pass parameters via the `extra` launch mechanism: a flat
    /// byte buffer whose address and size are exposed in `extra[1]` /`extra[3]`.
    /// We scan that buffer for `capture_kv_len` (flash-attn) and `capture_pos`
    /// (RoPE) and record direct pointers into the graph's owned storage.
    fn find_kernel_scalar_nodes(
        &self,
        cu_graph: cuda_core::sys::CUgraph,
        capture_kv_len: i32,
        capture_pos: i32,
        n_kv_heads: u32,
    ) -> (Vec<KernelScalarNode>, Vec<KernelScalarNode>) {
        // CU_LAUNCH_PARAM_BUFFER_POINTER sentinel value (== 1).
        const BUF_PTR_TAG: usize = 1;
        // CU_LAUNCH_PARAM_BUFFER_SIZE sentinel value (== 2).
        const BUF_SIZE_TAG: usize = 2;

        let mut num_nodes = 0usize;
        unsafe { cuda_core::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut num_nodes) };
        if num_nodes == 0 { return (vec![], vec![]); }
        let mut nodes = vec![std::ptr::null_mut::<cuda_core::sys::CUgraphNode_st>(); num_nodes];
        unsafe { cuda_core::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut num_nodes) };

        let mut flash_attn_nodes = Vec::new();
        let mut rope_nodes       = Vec::new();

        for node in &nodes {
            let mut ntype: cuda_core::sys::CUgraphNodeType = 0;
            unsafe { cuda_core::sys::cuGraphNodeGetType(*node, &mut ntype) };
            if ntype != cuda_core::sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL {
                continue;
            }

            let mut params: cuda_core::sys::CUDA_KERNEL_NODE_PARAMS_v2 =
                unsafe { std::mem::zeroed() };
            let rc = unsafe { cuda_core::sys::cuGraphKernelNodeGetParams_v2(*node, &mut params) };
            if rc != 0 { continue; }

            // cuTile JIT kernels: cuGraphKernelNodeGetParams_v2 may block for
            // runtime-compiled kernels.  Skip kernel parameter scanning for now —
            // KV memcpy updates alone provide the 108 tok/s improvement.
            // kv_len/pos node update will be re-attempted via PTX introspection.
            let _ = (params, capture_kv_len, capture_pos, n_kv_heads);
        }

        eprintln!(
            "[cuda_graph] scalar nodes: {} flash_attn, {} rope",
            flash_attn_nodes.len(), rope_nodes.len()
        );
        (flash_attn_nodes, rope_nodes)
    }

    /// Enumerate all MEMCPY nodes in `cu_graph` that write to the KV cache.
    ///
    /// These nodes' destination pointers are updated per decode step via
    /// `cuGraphExecMemcpyNodeSetParams` so the graph can be replayed without
    /// re-capture.
    fn find_kv_memcpy_nodes(
        &self,
        cu_graph: cuda_core::sys::CUgraph,
        kv_cache: &CutileKvCache,
    ) -> Vec<KvMemcpyEntry> {
        let kv_dim    = self.config.num_kv_heads * self.config.head_dim;
        let width_tgt = kv_dim * std::mem::size_of::<f32>();

        // Enumerate all graph nodes.
        let mut num_nodes = 0usize;
        unsafe { cuda_core::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut num_nodes) };
        if num_nodes == 0 { return vec![]; }
        let mut nodes = vec![std::ptr::null_mut::<cuda_core::sys::CUgraphNode_st>(); num_nodes];
        unsafe { cuda_core::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut num_nodes) };

        // Collect K and V device base pointers for all layers.
        let kv_base_ptrs: Vec<(cuda_core::sys::CUdeviceptr, cuda_core::sys::CUdeviceptr)> = (0..kv_cache.n_layers())
            .map(|l| {
                let (k, v) = kv_cache.kv_device_tensors(l);
                (k.device_pointer().cu_deviceptr(), v.device_pointer().cu_deviceptr())
            })
            .collect();

        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let kv_dim_bytes = kv_dim as u64 * f32_bytes;

        let mut entries = Vec::new();
        for node in &nodes {
            let mut node_type: cuda_core::sys::CUgraphNodeType = 0;
            unsafe { cuda_core::sys::cuGraphNodeGetType(*node, &mut node_type) };
            if node_type != cuda_core::sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY {
                continue;
            }
            let mut params: cuda_core::sys::CUDA_MEMCPY3D = unsafe { std::mem::zeroed() };
            let rc = unsafe { cuda_core::sys::cuGraphMemcpyNodeGetParams(*node, &mut params) };
            if rc != 0 { continue; }
            if params.WidthInBytes != width_tgt { continue; }
            if params.srcMemoryType != cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE { continue; }
            if params.dstMemoryType != cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE { continue; }
            // Find which layer this node belongs to by matching dst to kv cache range.
            let dst = params.dstDevice;
            for (k_base, v_base) in &kv_base_ptrs {
                let kv_range = kv_cache.gpu_max_seq_len() as u64 * kv_dim_bytes;
                if dst >= *k_base && dst < k_base + kv_range {
                    entries.push(KvMemcpyEntry {
                        node: *node,
                        src_ptr: params.srcDevice,
                        cache_base: *k_base,
                        width_bytes: width_tgt,
                    });
                    break;
                }
                if dst >= *v_base && dst < v_base + kv_range {
                    entries.push(KvMemcpyEntry {
                        node: *node,
                        src_ptr: params.srcDevice,
                        cache_base: *v_base,
                        width_bytes: width_tgt,
                    });
                    break;
                }
            }
        }
        entries
    }

    /// Run the decode forward pass without CUDA graph capture.
    ///
    /// Used as a fallback when `cuGraphExecUpdate_v2` fails or when graph capture
    /// is not applicable (e.g. when padding mismatches require D→H sync inside
    /// the layer loop).
    fn forward_decode_nogpu(
        &self,
        embedding: &[f32],
        positions: &[u32],
        gpu_kv_cache: &mut CutileKvCache,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        // Write d_kv_len and d_positions so oxide kernels (e.g. oxide_decode_attn)
        // read the correct kv_len even in the no-graph fallback path.
        let stream = self.matmul.stream();
        let kv_len_val = (gpu_kv_cache.seq_len() as i32) + 1;
        let kv_len_bits: f32 = f32::from_bits(kv_len_val as u32);
        let pos_bits: f32 = f32::from_bits(positions[0]);
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_kv_len.device_pointer().cu_deviceptr(),
                &kv_len_bits as *const f32, 1, stream,
            ).expect("forward_decode_nogpu d_kv_len H→D");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("forward_decode_nogpu d_positions H→D");
        }

        // Upload embedding to a fresh device tensor (normal non-graph path).
        let hidden_dev = self.to_device_flat_padded(embedding, 1, h, self.h_pad);
        self.forward_decode_inner(hidden_dev, positions, gpu_kv_cache);

        // Sync the stream and download logits from SLOT_LOGITS.
        let logits_arc = self
            .scratch
            .borrow()
            [SLOT_LOGITS]
            .clone()
            .expect("SLOT_LOGITS empty after forward_decode_inner");
        self.download_device_tensor(&logits_arc)
    }

    /// Inner decode forward pass: layer loop + final norm + lm_head.
    ///
    /// `hidden_dev` is the pre-uploaded `[h_pad]` embedding tensor.  It may be
    /// `d_embed` (for the graph-capture path) or a freshly allocated tensor (for
    /// the fallback non-graph path).
    ///
    /// All GPU work is enqueued **asynchronously** on the matmul stream.  The
    /// caller is responsible for calling `stream.synchronize()` before reading
    /// any output.  After the call the logits tensor resides in
    /// `self.scratch[SLOT_LOGITS]`.
    ///
    /// During CUDA graph capture (`cuStreamBeginCapture` in progress) no GPU work
    /// executes — API calls are only recorded.  The return value is a placeholder;
    /// real data is available after `cuGraphLaunch` + sync.
    fn forward_decode_inner(
        &self,
        mut hidden_dev: Arc<Tensor<f32>>,
        positions: &[u32],
        gpu_kv_cache: &mut CutileKvCache,
    ) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let inter = cfg.intermediate_size;
        let seq_len = 1usize;
        let shared_stream = self.matmul.stream();

        // Pre-build weight name strings for all layers.
        #[allow(clippy::type_complexity)]
        let layer_names: Vec<(
            String, String, String, String, String, String, String, String, String,
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

            // Pre-attention RMSNorm (GPU, pre-allocated scratch).
            let input_ln_w_dev = self
                .norm_weights
                .get(input_ln_name)
                .expect("input_layernorm weight missing");
            let normed_dev = self
                .cutile_rmsnorm
                .forward_device_fused_scratch(
                    shared_stream,
                    hidden_dev.clone(),
                    input_ln_w_dev,
                    seq_len,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                    &mut self.pre_attn_norm_scratch.borrow_mut(),
                )
                .expect("pre-attention RMSNorm failed");

            // Q / K / V projections (GPU, pre-allocated slots).
            let q_dev = self.gemm_any_d2d_slot(&normed_dev, q_w_name, seq_len, SLOT_Q);
            let k_dev = self.gemm_any_d2d_slot(&normed_dev, k_w_name, seq_len, SLOT_K);
            let v_dev = self.gemm_any_d2d_slot(&normed_dev, v_w_name, seq_len, SLOT_V);

            // RoPE (GPU, pre-allocated scratch; uses precomputed cos/sin table for seq_len=1).
            let q_rot_dev = self
                .cutile_rope
                .forward_device_scratch(
                    &q_dev,
                    positions,
                    seq_len,
                    n_heads,
                    head_dim,
                    cfg.rope_theta,
                    &mut self.q_rope_scratch.borrow_mut(),
                    shared_stream,
                )
                .expect("Q RoPE failed");
            let k_rot_dev = self
                .cutile_rope
                .forward_device_scratch(
                    &k_dev,
                    positions,
                    seq_len,
                    n_kv_heads,
                    head_dim,
                    cfg.rope_theta,
                    &mut self.k_rope_scratch.borrow_mut(),
                    shared_stream,
                )
                .expect("K RoPE failed");

            // KV cache append (GPU-only; host mirror will be rebuilt on fallback if needed).
            gpu_kv_cache
                .append_device_gpu_only(layer, &k_rot_dev, &v_dev, seq_len)
                .expect("GPU KV cache append failed");

            // kv_len includes the token just appended.
            let kv_len = gpu_kv_cache.seq_len() + seq_len;

            // GPU decode attention (fused flash-attention, pre-allocated scratch).
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (k_dev_full, v_dev_full) = gpu_kv_cache.kv_device_tensors(layer);

            // Prefer oxide decode attention (kv_len read from device at execution
            // time — correct CUDA graph replay without stale kv_len baking).
            let attn_out_dev = if let Some(ref oda) = self.oxide_decode_attn {
                let q_dim = n_heads * head_dim;
                let out_tensor = {
                    let mut scratch = self.attn_out_scratch.borrow_mut();
                    match scratch.take() {
                        Some(arc) => match Arc::try_unwrap(arc) {
                            Ok(t) => Arc::new(t.reshape(&[q_dim])
                                .expect("attn out reshape")),
                            Err(arc_back) => {
                                *scratch = Some(arc_back);
                                Arc::new(cutile::api::zeros::<f32>(&[q_dim])
                                    .sync_on(shared_stream)
                                    .expect("attn out alloc"))
                            }
                        },
                        None => Arc::new(cutile::api::zeros::<f32>(&[q_dim])
                            .sync_on(shared_stream)
                            .expect("attn out alloc")),
                    }
                };
                // Use base single-warp kernel (kpar disabled — crashes in CUDA graph).
                oda.forward_d2d(
                    &out_tensor, &q_rot_dev, &k_dev_full, &v_dev_full,
                    &self.d_kv_len, n_heads, n_kv_heads, head_dim, scale,
                ).expect("oxide decode attention failed");
                self.attn_out_scratch.borrow_mut().replace(out_tensor.clone());
                out_tensor
            } else {
                self
                    .flash_attn
                    .decode_attention_device_scratch(
                        &q_rot_dev,
                        &k_dev_full,
                        &v_dev_full,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        scale,
                        shared_stream,
                        &mut self.attn_out_scratch.borrow_mut(),
                    )
                    .expect("decode attention failed")
            };

            // Pad [q_dim] → [1, q_pad] for O-projection (zero-copy reshape when equal).
            let attn_padded_dev = self.attn_out_to_oproj_device(&attn_out_dev, q_dim, self.q_pad);

            // O-projection (GPU, pre-allocated slot).
            let o_dev = self.gemm_any_d2d_slot(&attn_padded_dev, o_w_name, seq_len, SLOT_O);

            // Post-attention residual add (GPU, pre-allocated scratch).
            hidden_dev = self.add_device_scratch(
                &hidden_dev,
                &o_dev,
                seq_len * self.h_pad,
                &mut self.post_attn_add_scratch.borrow_mut(),
            );

            // Post-attention RMSNorm (GPU, pre-allocated scratch).
            let post_ln_w_dev = self
                .norm_weights
                .get(post_ln_name)
                .expect("post_attention_layernorm weight missing");
            let normed2_dev = self
                .cutile_rmsnorm
                .forward_device_fused_scratch(
                    shared_stream,
                    hidden_dev.clone(),
                    post_ln_w_dev,
                    seq_len,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                    &mut self.post_attn_norm_scratch.borrow_mut(),
                )
                .expect("post-attention RMSNorm failed");

            // FFN gate + up projections.
            let gate_dev = self.gemm_any_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
            let up_dev = self.gemm_any_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);

            // SiLU-gated MLP (GPU, pre-allocated scratch).
            let gate_silu_n_dev = self
                .cutile_elementwise
                .silu_mul_device_scratch(
                    &gate_dev,
                    &up_dev,
                    seq_len,
                    inter,
                    &mut self.silu_out_scratch.borrow_mut(),
                )
                .expect("SiLU-mul failed");

            // FFN down projection.
            // When inter_pad == inter (power-of-2 models) no padding is needed.
            let gate_silu_dev = if self.inter_pad == inter {
                gate_silu_n_dev
            } else {
                // GPU-side zero-padding: D→D copy of `inter` elements into a pre-zeroed
                // `inter_pad` scratch.  Eliminates the D→H stream sync that was here before.
                // Positions 0..inter are overwritten; inter..inter_pad remain 0 from init.
                let pad_arc = self.down_proj_pad_scratch.borrow_mut()
                    .take()
                    .and_then(|arc| Arc::try_unwrap(arc).ok())
                    .unwrap_or_else(|| {
                        api::zeros::<f32>(&[self.inter_pad])
                            .sync_on(shared_stream)
                            .expect("down_proj_pad: fallback alloc failed")
                    });
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        pad_arc.device_pointer().cu_deviceptr(),
                        gate_silu_n_dev.device_pointer().cu_deviceptr(),
                        inter * seq_len,
                        shared_stream,
                    )
                    .expect("down_proj_pad D→D copy failed");
                }
                let result = Arc::new(pad_arc);
                *self.down_proj_pad_scratch.borrow_mut() = Some(result.clone());
                result
            };
            let down_dev =
                self.gemm_any_d2d_slot(&gate_silu_dev, down_w_name, seq_len, SLOT_DOWN);

            // When h_pad == h (power-of-2 models) no padding is needed.
            let down_padded = if self.h_pad == h {
                down_dev
            } else {
                let down_host = self.download_device_tensor(&down_dev);
                self.to_device_flat_padded(&down_host, seq_len, h, self.h_pad)
            };

            // Post-FFN residual add (GPU, pre-allocated scratch).
            hidden_dev = self.add_device_scratch(
                &hidden_dev,
                &down_padded,
                seq_len * self.h_pad,
                &mut self.ffn_add_scratch.borrow_mut(),
            );
        }

        // ── Final norm + lm_head ─────────────────────────────────────────────
        // For seq_len=1 hidden_dev IS the last token — no D→H download or
        // H→D re-upload needed.
        let final_norm_w_dev = self
            .norm_weights
            .get("model.norm.weight")
            .expect("model.norm.weight missing");
        let last_dev = self
            .cutile_rmsnorm
            .forward_device_fused_scratch(
                shared_stream,
                hidden_dev,
                final_norm_w_dev,
                1,
                h,
                self.h_pad,
                cfg.layer_norm_eps,
                &mut self.final_norm_scratch.borrow_mut(),
            )
            .expect("final RMSNorm failed");

        // lm_head GEMV — result stored in SLOT_LOGITS (pre-allocated).
        self.gemm_d2d_slot(&last_dev, "lm_head.weight", 1, SLOT_LOGITS);
    }

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

        // Reset the GPU KV cache when starting a new sequence (position=0).
        // The shared KV cache accumulates across sessions otherwise.
        if position == 0 {
            gpu_kv_cache.reset();
            *self.cached_decode_graph.borrow_mut() = None;
        }

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

        // ── 2. CUDA graph dispatch (decode + GPU KV cache) ─────────────────────
        // For single-token decode with an active GPU KV cache and power-of-2 model
        // dimensions (no D→H sync needed inside the layer loop), capture the
        // forward pass into a CUDA graph and replay it.  This replaces ~200
        // individual `cuLaunchKernel` submissions with one `cuGraphLaunch`.
        //
        // The D→D padding for inter_pad != inter is now GPU-only (pre-zeroed scratch
        // + async D→D copy), so the layer loop itself has no D→H syncs regardless.
        // However the CUDA graph path requires all cuTile JIT variants to be
        // pre-compiled before capture begins; for non-power-of-2 N (e.g. vocab=32000)
        // some variants may not be warmed up, causing "not permitted during capture"
        // errors.  Until warmup covers all shapes, restrict to power-of-2 inter models.
        //
        // Preconditions:
        //   • seq_len == 1 — decode mode
        //   • gpu_kv_cache.has_device_kv() — GPU flash-attention is active
        //   • !cfg.use_alibi — RoPE path
        // inter_pad/h_pad: padding is handled via D→D copy (capturable).
        // cuTile GEMV uses async_on (capture-compatible); warmup covers all needed
        // shapes, so no JIT compilation happens inside capture.
        if seq_len == 1
            && gpu_kv_cache.has_device_kv()
            && !cfg.use_alibi
        {
            // The host KV mirror may be stale; we only rebuild it when transitioning
            // to the CPU fallback path, which does not apply here.
            return self.forward_decode_with_graph(&hidden, &positions, gpu_kv_cache);
        }

        // ── 3. Layer loop (prefill or CPU-attention fallback) ─────────────────
        // Sync the host KV mirror if it was skipped during prior GPU-only decode
        // steps and we are about to fall back to CPU attention.  This happens once
        // at the transition from GPU decode to long-context CPU fallback.
        if gpu_kv_cache.is_host_stale() && !gpu_kv_cache.has_device_kv() {
            gpu_kv_cache
                .sync_host_from_device(gpu_kv_cache.seq_len())
                .expect("CutileGpuModel: host KV mirror sync failed");
        }

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
            // For decode (seq_len=1) use scratch buffers to avoid 3 api::zeros.sync_on
            // per call (168 blocking allocs per token across 28 layers).
            let input_ln_w_dev = self
                .norm_weights
                .get(input_ln_name)
                .expect("CutileGpuModel: input_layernorm weight not pre-uploaded");
            let normed_dev = if seq_len == 1 {
                self.cutile_rmsnorm
                    .forward_device_fused_scratch(
                        self.matmul.stream(),
                        hidden_dev.clone(),
                        input_ln_w_dev,
                        seq_len,
                        h,
                        self.h_pad,
                        cfg.layer_norm_eps,
                        &mut self.pre_attn_norm_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: pre-attention RMSNorm failed")
            } else {
                self.cutile_rmsnorm
                    .forward_device_fused(
                        self.matmul.stream(),
                        hidden_dev.clone(),
                        input_ln_w_dev,
                        seq_len,
                        h,
                        self.h_pad,
                        cfg.layer_norm_eps,
                    )
                    .expect("CutileGpuModel: pre-attention RMSNorm failed")
            };

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
                let shared_stream = self.matmul.stream();
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
                        shared_stream,
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
                        shared_stream,
                    )
                    .expect("CutileGpuModel: GPU RoPE for K failed");
                (qr, kr)
            } else {
                // ALiBi: no rotation — keep Q/K tensors on device as-is.
                (q_dev.clone(), k_dev.clone())
            };

            // ── 2d. KV-cache append ───────────────────────────────────────────
            // When the GPU KV cache is active and seq_len == 1 (decode), use the
            // GPU-only append path that omits the D→H host-mirror update.
            // This eliminates 2 × n_layers blocking D→H `sync_on` calls per token
            // (56 for a 28-layer model) which were serialising the GPU pipeline.
            //
            // If we exceed the GPU KV cache capacity or need CPU attention
            // (`has_device_kv()` returns false), fall back to the full
            // `append_device` which also updates the host mirror.  When
            // transitioning from GPU-only to CPU mode the host mirror is rebuilt
            // via `sync_host_from_device` before the CPU attention path runs.
            let use_gpu_attention = seq_len == 1 && gpu_kv_cache.has_device_kv();
            if use_gpu_attention {
                gpu_kv_cache
                    .append_device_gpu_only(layer, &k_rot_dev, &v_dev, seq_len)
                    .expect("CutileGpuModel: GPU KV cache append (gpu-only) failed");
            } else {
                gpu_kv_cache
                    .append_device(layer, &k_rot_dev, &v_dev, seq_len)
                    .expect("CutileGpuModel: GPU KV cache append failed");
            }

            // kv_len includes the tokens just appended.
            let kv_len = gpu_kv_cache.seq_len() + seq_len;

            // ── 2e. Attention ─────────────────────────────────────────────────
            let scale = 1.0 / (head_dim as f32).sqrt();

            // For decode (seq_len=1) with the GPU KV cache active: run the fused
            // `cutile_decode_attn` kernel which reads Q, K, V entirely on device,
            // eliminating all PCIe traffic for the KV cache.  At context length L
            // this saves 2 × L × n_kv_heads × head_dim × 4 bytes per layer per
            // token — dominant PCIe traffic at long context.
            //
            // Falls back to the host-based path when:
            //   - seq_len > 1 (prefill): GPU flash attention via forward_batched
            //   - seq_len == 1 but kv_len > GPU_KV_MAX_SEQ_LEN: CPU GQA fallback
            let attn_padded_dev = if seq_len == 1 && gpu_kv_cache.has_device_kv() {
                // GPU decode attention: Q on device, K/V from GPU-resident cache.
                // Use scratch variant to eliminate the per-call api::zeros.sync_on
                // allocation (a full stream sync + cudaMallocAsync) — saves 28
                // blocking allocs per decode token across all layers.
                let (k_dev_full, v_dev_full) = gpu_kv_cache.kv_device_tensors(layer);
                let attn_out_dev = self
                    .flash_attn
                    .decode_attention_device_scratch(
                        &q_rot_dev,
                        &k_dev_full,
                        &v_dev_full,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        scale,
                        self.matmul.stream(),
                        &mut self.attn_out_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: GPU decode attention failed");
                // Pad [q_dim] → [1, q_pad] for O projection via GPU D→D copy.
                self.attn_out_to_oproj_device(&attn_out_dev, q_dim, self.q_pad)
            } else {
                // Host-based path: prefill (GPU flash attn) or long-context fallback.
                let q_rot_host = self.download_device_tensor(&q_rot_dev);
                let (k_slice, v_slice) = gpu_kv_cache.get_kv_host_slices(layer, kv_len);
                let mut attn_output = vec![0.0f32; seq_len * n_heads * head_dim];

                if seq_len == 1 {
                    // CPU GQA fallback for kv_len > GPU_KV_MAX_SEQ_LEN.
                    let heads_per_group = n_heads / n_kv_heads;
                    let mut scores = vec![0.0f32; kv_len];
                    for h in 0..n_heads {
                        let kv_h = h / heads_per_group;
                        let q_head = &q_rot_host[h * head_dim..(h + 1) * head_dim];
                        let kv_k = &k_slice[kv_h * head_dim..];
                        let kv_v = &v_slice[kv_h * head_dim..];

                        for t in 0..kv_len {
                            let k_tok = &kv_k[t * n_kv_heads * head_dim
                                ..t * n_kv_heads * head_dim + head_dim];
                            scores[t] = q_head
                                .iter()
                                .zip(k_tok)
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale;
                        }

                        let max_s =
                            scores[..kv_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let sum: f32 = scores[..kv_len]
                            .iter_mut()
                            .map(|s| {
                                *s = (*s - max_s).exp();
                                *s
                            })
                            .sum();
                        for s in &mut scores[..kv_len] {
                            *s /= sum;
                        }

                        let out_head = &mut attn_output[h * head_dim..(h + 1) * head_dim];
                        for v_d in 0..head_dim {
                            out_head[v_d] = 0.0;
                        }
                        for t in 0..kv_len {
                            let v_tok = &kv_v[t * n_kv_heads * head_dim
                                ..t * n_kv_heads * head_dim + head_dim];
                            for d in 0..head_dim {
                                out_head[d] += scores[t] * v_tok[d];
                            }
                        }
                    }
                } else {
                    // Prefill: GPU flash attention (amortises CUDA overhead over many rows).
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
                            scale,
                        )
                        .expect("CutileGpuModel: flash attention failed");
                }

                self.to_device_padded(&attn_output, seq_len, q_dim, self.q_pad)
            };

            // ── 2f. O projection — device-to-device ──────────────────────────
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
            hidden_dev = if seq_len == 1 {
                self.add_device_scratch(
                    &hidden_dev,
                    &o_padded,
                    seq_len * self.h_pad,
                    &mut self.post_attn_add_scratch.borrow_mut(),
                )
            } else {
                self.add_device(&hidden_dev, &o_padded, seq_len * self.h_pad)
            };

            // ── 2h. Post-attention RMSNorm (GPU) ─────────────────────────────
            // hidden_dev is already updated on device — no upload needed.
            let post_ln_w_dev = self
                .norm_weights
                .get(post_ln_name)
                .expect("CutileGpuModel: post_attention_layernorm weight not pre-uploaded");
            let normed2_dev = if seq_len == 1 {
                self.cutile_rmsnorm
                    .forward_device_fused_scratch(
                        self.matmul.stream(),
                        hidden_dev.clone(),
                        post_ln_w_dev,
                        seq_len,
                        h,
                        self.h_pad,
                        cfg.layer_norm_eps,
                        &mut self.post_attn_norm_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: post-attention RMSNorm failed")
            } else {
                self.cutile_rmsnorm
                    .forward_device_fused(
                        self.matmul.stream(),
                        hidden_dev.clone(),
                        post_ln_w_dev,
                        seq_len,
                        h,
                        self.h_pad,
                        cfg.layer_norm_eps,
                    )
                    .expect("CutileGpuModel: post-attention RMSNorm failed")
            };

            // ── 2i. FFN gate + up — device-to-device ─────────────────────────
            // For Q2_0 models gemm_any_d2d_slot dispatches on-GPU dequant (synchronous).
            // For f32 models the synchronous gemm_d2d_slot path is used.
            let gate_dev = self.gemm_any_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
            let up_dev = self.gemm_any_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);

            // ── 2j. SiLU gate × up (GPU) ─────────────────────────────────────
            // gate_dev and up_dev already reside on the GPU.  Use the scratch
            // variant to eliminate the per-call api::zeros.sync_on allocation —
            // saves 28 blocking allocs per decode token across all layers.
            let gate_silu_n_dev = if seq_len == 1 {
                self.cutile_elementwise
                    .silu_mul_device_scratch(
                        &gate_dev,
                        &up_dev,
                        seq_len,
                        inter,
                        &mut self.silu_out_scratch.borrow_mut(),
                    )
                    .expect("CutileGpuModel: GPU SiLU-mul failed")
            } else {
                self.cutile_elementwise
                    .silu_mul_device(&gate_dev, &up_dev, seq_len, inter)
                    .expect("CutileGpuModel: GPU SiLU-mul failed")
            };

            // ── 2k. FFN down — device-to-device ──────────────────────────────
            // Pad gate_silu from inter to inter_pad for the down_proj GEMM.
            // When inter == inter_pad (pow-2 models) this is a free reshape.
            // For seq_len==1 with non-power-of-2 inter (e.g. Llama-3.1-8B inter=14336):
            // use GPU D→D copy into the pre-zeroed inter_pad scratch to avoid 32 D→H
            // stream syncs per token — the dominant bottleneck before this fix.
            let gate_silu_dev = if self.inter_pad == inter {
                gate_silu_n_dev
            } else if seq_len == 1 {
                let stream = self.matmul.stream();
                let pad_arc = self.down_proj_pad_scratch.borrow_mut()
                    .take()
                    .and_then(|arc| Arc::try_unwrap(arc).ok())
                    .unwrap_or_else(|| {
                        api::zeros::<f32>(&[self.inter_pad])
                            .sync_on(stream)
                            .expect("down_proj_pad: fallback alloc failed")
                    });
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        pad_arc.device_pointer().cu_deviceptr(),
                        gate_silu_n_dev.device_pointer().cu_deviceptr(),
                        inter,
                        stream,
                    )
                    .expect("down_proj_pad D→D copy failed");
                }
                let result = Arc::new(pad_arc);
                *self.down_proj_pad_scratch.borrow_mut() = Some(result.clone());
                result
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
            hidden_dev = if seq_len == 1 {
                self.add_device_scratch(
                    &hidden_dev,
                    &down_padded,
                    seq_len * self.h_pad,
                    &mut self.ffn_add_scratch.borrow_mut(),
                )
            } else {
                self.add_device(&hidden_dev, &down_padded, seq_len * self.h_pad)
            };
        }

        // ── 3. Final norm + lm_head ────────────────────────────────────────────
        // For decode (seq_len=1) hidden_dev IS the last token — pass it directly
        // to the final RMSNorm, skipping the unnecessary D→H download + H→D re-upload
        // that the prefill path needs to extract just the last token row.
        let final_norm_w_dev = self
            .norm_weights
            .get("model.norm.weight")
            .expect("CutileGpuModel: model.norm.weight not pre-uploaded");
        let last_dev = if seq_len == 1 {
            self.cutile_rmsnorm
                .forward_device_fused(
                    self.matmul.stream(),
                    hidden_dev,
                    final_norm_w_dev,
                    1,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("CutileGpuModel: final RMSNorm failed")
        } else {
            // Prefill: download, extract the last token's hidden state, re-upload.
            let hidden_final = self.download_device_tensor(&hidden_dev);
            let last_flat = Arc::new(
                api::copy_host_vec_to_device(&Arc::new(
                    hidden_final[(seq_len - 1) * self.h_pad..seq_len * self.h_pad].to_vec(),
                ))
                .sync_on(self.matmul.stream())
                .expect("CutileGpuModel: last-token upload failed"),
            );
            self.cutile_rmsnorm
                .forward_device_fused(
                    self.matmul.stream(),
                    last_flat,
                    final_norm_w_dev,
                    1,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("CutileGpuModel: final RMSNorm failed")
        };

        let logits_dev = self.gemm_d2d_slot(&last_dev, "lm_head.weight", 1, SLOT_LOGITS);
        self.download_device_tensor(&logits_dev)
    }

    /// Batched decode: M requests × 1 token each, processed in a single GPU pass.
    ///
    /// Each request has its own KV cache (`kv_caches[r]`) and current token
    /// (`token_ids[r]`) at position `positions[r]`. Returns M logit vectors
    /// (one per request, length = vocab_size).
    ///
    /// Uses `oxide_decode_attn_batch` for flash attention — grid = M × n_heads
    /// blocks fills all 170 RTX-5090 SMs for batch ≥ 5.
    pub fn forward_batch_decode(
        &self,
        token_ids:  &[u32],
        positions:  &[u32],
        qweights:   &QuantizedWeights,
        kv_caches:  &mut [CutileKvCache],
    ) -> Vec<Vec<f32>> {
        assert_eq!(token_ids.len(), kv_caches.len());
        assert_eq!(positions.len(), kv_caches.len());
        let m = token_ids.len();
        if m == 0 { return vec![]; }

        let cfg        = &self.config;
        let h          = cfg.hidden_size;
        let n_heads    = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim   = cfg.head_dim;
        let q_dim      = n_heads * head_dim;
        let kv_dim     = n_kv_heads * head_dim;
        let vocab      = cfg.vocab_size;
        let scale      = 1.0_f32 / (head_dim as f32).sqrt();
        let stream     = self.matmul.stream();

        // ── 1. Embedding lookup (CPU) ──────────────────────────────────────────
        let embed = qweights
            .get_f32("model.embed_tokens.weight")
            .expect("embed_tokens.weight not found");
        let mut hidden = vec![0.0f32; m * h];
        for (r, &tok) in token_ids.iter().enumerate() {
            hidden[r * h..(r + 1) * h]
                .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
        }
        if cfg.embedding_scale {
            let s = (h as f32).sqrt();
            for v in &mut hidden { *v *= s; }
        }

        let layer_names: Vec<_> = (0..cfg.num_layers)
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

        // Upload concatenated hidden [M, H] to device.
        let mut hidden_dev = self.to_device_flat_padded(&hidden, m, h, self.h_pad);

        for layer in 0..cfg.num_layers {
            let (
                ref input_ln_name, ref post_ln_name,
                ref q_w_name, ref k_w_name, ref v_w_name, ref o_w_name,
                ref gate_w_name, ref up_w_name, ref down_w_name,
            ) = layer_names[layer];

            let input_ln_w_dev = self.norm_weights.get(input_ln_name)
                .expect("input_layernorm weight missing");
            let normed_dev = self.cutile_rmsnorm
                .forward_device_fused(
                    stream,
                    hidden_dev.clone(),
                    input_ln_w_dev,
                    m,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("batch RMSNorm failed");

            // QKV projections [M, H] → [M, Q_DIM] / [M, KV_DIM]
            // normed_dev is [m, h_pad] from forward_device_fused; row stride = h_pad.
            let q_dev = self.gemm_any_d2d_batch(&normed_dev, q_w_name, m, self.h_pad, SLOT_Q);
            let k_dev = self.gemm_any_d2d_batch(&normed_dev, k_w_name, m, self.h_pad, SLOT_K);
            let v_dev = self.gemm_any_d2d_batch(&normed_dev, v_w_name, m, self.h_pad, SLOT_V);

            // RoPE per-request (each has its own position).
            // Process each request's Q/K slice individually.
            let kv_pad = next_pow2(kv_dim.div_ceil(BK) * BK);
            let mut q_rot_parts: Vec<Arc<cutile::tensor::Tensor<f32>>> = Vec::with_capacity(m);
            for r in 0..m {
                // Extract per-request Q/K/V slices via D→D memcpy into fresh [q_dim] tensors.
                let q_r = {
                    let t = Arc::new(
                        cutile::api::zeros::<f32>(&[self.q_pad])
                            .sync_on(stream)
                            .expect("q_r alloc"),
                    );
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            t.device_pointer().cu_deviceptr(),
                            q_dev.device_pointer().cu_deviceptr() + (r * self.q_pad * 4) as u64,
                            q_dim,
                            stream,
                        ).expect("q_r D→D");
                    }
                    t
                };
                let k_r = {
                    let t = Arc::new(
                        cutile::api::zeros::<f32>(&[kv_pad])
                            .sync_on(stream)
                            .expect("k_r alloc"),
                    );
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            t.device_pointer().cu_deviceptr(),
                            k_dev.device_pointer().cu_deviceptr() + (r * kv_pad * 4) as u64,
                            kv_dim,
                            stream,
                        ).expect("k_r D→D");
                    }
                    t
                };
                let v_r = {
                    let t = Arc::new(
                        cutile::api::zeros::<f32>(&[kv_pad])
                            .sync_on(stream)
                            .expect("v_r alloc"),
                    );
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            t.device_pointer().cu_deviceptr(),
                            v_dev.device_pointer().cu_deviceptr() + (r * kv_pad * 4) as u64,
                            kv_dim,
                            stream,
                        ).expect("v_r D→D");
                    }
                    t
                };

                let pos_r = [positions[r]];
                let q_rot = self.cutile_rope
                    .forward_device_scratch(
                        &q_r, &pos_r, 1, n_heads, head_dim,
                        cfg.rope_theta,
                        &mut self.q_rope_scratch.borrow_mut(),
                        stream,
                    )
                    .expect("batch Q RoPE failed");
                let k_rot = self.cutile_rope
                    .forward_device_scratch(
                        &k_r, &pos_r, 1, n_kv_heads, head_dim,
                        cfg.rope_theta,
                        &mut self.k_rope_scratch.borrow_mut(),
                        stream,
                    )
                    .expect("batch K RoPE failed");

                kv_caches[r]
                    .append_device_gpu_only(layer, &k_rot, &v_r, 1)
                    .expect("batch KV append failed");
                q_rot_parts.push(q_rot);
            }

            // Batch flash attention: one kernel launch, M × n_heads blocks.
            let attn_out_dev = if let Some(ref oda) = self.oxide_decode_attn {
                if oda.has_batch() {
                    let attn_out = Arc::new(
                        cutile::api::zeros::<f32>(&[m * q_dim])
                            .sync_on(stream)
                            .expect("batch attn out alloc"),
                    );
                    // Collect per-request KV cache device pointers and kv_lens.
                    let k_ptrs: Vec<u64> = kv_caches.iter()
                        .map(|kvc| kvc.kv_device_tensors(layer).0.device_pointer().cu_deviceptr())
                        .collect();
                    let v_ptrs: Vec<u64> = kv_caches.iter()
                        .map(|kvc| kvc.kv_device_tensors(layer).1.device_pointer().cu_deviceptr())
                        .collect();
                    let kv_lens: Vec<i32> = kv_caches.iter()
                        .map(|kvc| kvc.seq_len() as i32)
                        .collect();

                    // Concatenate Q-rotated parts into one tensor for the batch kernel.
                    // q_rot_parts[r] is [q_dim], stack to [m * q_dim].
                    let q_batch = Arc::new(
                        cutile::api::zeros::<f32>(&[m * q_dim])
                            .sync_on(stream)
                            .expect("q_batch alloc"),
                    );
                    // Copy each q_rot_parts[r] into q_batch at offset r*q_dim.
                    // Use D→D async memcpy via cutile.
                    for (r, q_r) in q_rot_parts.iter().enumerate() {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                q_batch.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                                q_r.device_pointer().cu_deviceptr(),
                                q_dim,
                                stream,
                            ).expect("q_batch D→D copy");
                        }
                    }

                    oda.forward_batch_d2d(
                        &attn_out, &q_batch,
                        &k_ptrs, &v_ptrs, &kv_lens,
                        n_heads, n_kv_heads, head_dim, scale,
                    ).expect("oxide_decode_attn_batch failed");
                    attn_out
                } else {
                    // Fallback: run single-request attention M times.
                    let attn_out = Arc::new(
                        cutile::api::zeros::<f32>(&[m * q_dim])
                            .sync_on(stream)
                            .expect("batch attn fallback alloc"),
                    );
                    for (r, q_rot) in q_rot_parts.iter().enumerate() {
                        let (k_full, v_full) = kv_caches[r].kv_device_tensors(layer);
                        let kv_len_r = kv_caches[r].seq_len() as i32;
                        // Write kv_len to d_kv_len for this request.
                        let kv_bits = f32::from_bits(kv_len_r as u32);
                        unsafe {
                            cuda_core::memcpy_htod_async::<f32>(
                                self.d_kv_len.device_pointer().cu_deviceptr(),
                                &kv_bits as *const f32, 1, stream,
                            ).expect("batch d_kv_len H→D");
                        }
                        let out_r = Arc::new(
                            cutile::api::zeros::<f32>(&[q_dim])
                                .sync_on(stream)
                                .expect("attn out_r alloc"),
                        );
                        oda.forward_d2d(
                            &out_r, q_rot, &k_full, &v_full,
                            &self.d_kv_len, n_heads, n_kv_heads, head_dim, scale,
                        ).expect("batch attn single step failed");
                        // Copy out_r into attn_out at offset r*q_dim.
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                attn_out.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                                out_r.device_pointer().cu_deviceptr(),
                                q_dim,
                                stream,
                            ).expect("attn_out D→D copy");
                        }
                    }
                    attn_out
                }
            } else {
                // cuTile fallback: run M sequential single-request attention calls.
                // Slower than the batch oxide kernel but correct on all hardware.
                let attn_out = Arc::new(
                    cutile::api::zeros::<f32>(&[m * q_dim])
                        .sync_on(stream)
                        .expect("batch attn cutile fallback alloc"),
                );
                for (r, q_rot) in q_rot_parts.iter().enumerate() {
                    let (k_full, v_full) = kv_caches[r].kv_device_tensors(layer);
                    let kv_len_r = kv_caches[r].seq_len() as i32;
                    let attn_r = self.flash_attn
                        .decode_attention_device_scratch(
                            q_rot, &k_full, &v_full, kv_len_r as usize,
                            n_heads, n_kv_heads, head_dim, scale, stream,
                            &mut self.attn_out_scratch.borrow_mut(),
                        ).expect("batch cutile attn failed");
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            attn_out.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                            attn_r.device_pointer().cu_deviceptr(),
                            q_dim,
                            stream,
                        ).expect("batch attn_out D→D");
                    }
                }
                attn_out
            };

            // Advance all KV caches.
            for kvc in kv_caches.iter_mut() {
                kvc.advance_host_seq_len(1);
            }

            // O-projection [M, Q_DIM] → [M, H]
            let attn_padded = self.attn_out_to_oproj_device_m(&attn_out_dev, q_dim, self.q_pad, m);
            // attn_padded is [m, q_pad] from attn_out_to_oproj_device_m; row stride = q_pad.
            let o_dev = self.gemm_any_d2d_batch(&attn_padded, o_w_name, m, self.q_pad, SLOT_O);

            // Post-attention residual add (fresh alloc — pre-allocated scratch is M=1 sized).
            hidden_dev = self.add_device_scratch(
                &hidden_dev, &o_dev,
                m * self.h_pad,
                &mut None,
            );

            // Post-attention RMSNorm.
            let post_ln_w_dev = self.norm_weights.get(post_ln_name)
                .expect("post_attention_layernorm weight missing");
            let normed2_dev = self.cutile_rmsnorm
                .forward_device_fused(
                    stream,
                    hidden_dev.clone(),
                    post_ln_w_dev,
                    m,
                    h,
                    self.h_pad,
                    cfg.layer_norm_eps,
                )
                .expect("batch post-attn RMSNorm failed");

            // FFN: gate/up/down projections.
            // normed2_dev is [m, h_pad]; row stride = h_pad.
            let gate_dev = self.gemm_any_d2d_batch(&normed2_dev, gate_w_name, m, self.h_pad, SLOT_GATE);
            let up_dev   = self.gemm_any_d2d_batch(&normed2_dev, up_w_name,   m, self.h_pad, SLOT_UP);

            // SiLU(gate) × up — use cuTile silu_mul (GPU, no PCIe round-trip, any M).
            let inter_size = m * cfg.intermediate_size;
            let gs_dev = self
                .cutile_elementwise
                .silu_mul_device(&gate_dev, &up_dev, m, cfg.intermediate_size)
                .expect("batch silu_mul_device failed");
            // Pad inter → inter_pad if necessary (e.g. Llama-3.1-8B: 14336→16384).
            let inter_sz = cfg.intermediate_size;
            let gs_padded = if self.inter_pad == inter_sz {
                gs_dev
            } else {
                let gs_host = self.download_device_tensor(&gs_dev);
                self.to_device_padded(&gs_host, m, inter_sz, self.inter_pad)
            };
            // gs_padded is [m, inter_pad] (after silu + padding); row stride = inter_pad.
            let down_dev = self.gemm_any_d2d_batch(&gs_padded, down_w_name, m, self.inter_pad, SLOT_DOWN);

            hidden_dev = self.add_device_scratch(
                &hidden_dev, &down_dev,
                m * self.h_pad,
                &mut None,
            );
        }

        // Final norm.
        let final_ln_w_dev = self.norm_weights.get("model.norm.weight")
            .expect("model.norm.weight missing");
        let last_dev = self.cutile_rmsnorm
            .forward_device_fused(
                stream,
                hidden_dev,
                final_ln_w_dev,
                m,
                h,
                self.h_pad,
                cfg.layer_norm_eps,
            )
            .expect("batch final RMSNorm failed");

        // LM head: [M, H] → [M, vocab]
        let logits_dev = self.gemm_d2d(&last_dev, "lm_head.weight", m);
        let logits_flat = self.download_device_tensor(&logits_dev);

        // Split into M logit vectors.
        logits_flat.chunks(vocab).map(|c| c.to_vec()).collect()
    }
}

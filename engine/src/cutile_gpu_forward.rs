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
///   - normed   -> h_pad   (one `to_device_padded` call, used for Q, K, V)
///   - normed2  -> h_pad   (one `to_device_padded` call, used for gate, up)
///   - gatexup  -> inter_pad (one `to_device_padded` call, used for down)
///   - attn_out -> q_pad   (one from_device + one to_device_padded, for O projection)
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
use tempura_kernels::kv_scatter_write::KvScatterWrite;
use tempura_kernels::mxfp4_dispatch::MxfpGemm;

use crate::config::ModelConfig;
use crate::cutile_kv_cache::CutileKvCache;
use crate::weights::QuantizedWeights;

/// RAII wrapper for a CUDA graph executable.
///
/// The exec is replayed each decode step via `cuGraphLaunch`.  All per-step
/// scalars (kv_len, position, KV write offset) are read from DEVICE memory at
/// execution time -- the oxide attention kernels read `d_kv_len`, the
/// scatter-write kernel reads `d_kv_write_offset`, and the RoPE row gather
/// reads `d_positions` -- so no graph node parameters ever need patching and
/// no `cuGraphExecUpdate_v2` is required.
///
/// KV-cache device pointers are baked at capture and validated on every
/// replay: a different cache object (new session, fresh-cache bench pass)
/// invalidates the graph and forces a re-capture.
struct DecodeGraph {
    cu_exec: cuda_core::sys::CUgraphExec,
    cu_graph: cuda_core::sys::CUgraph, // kept alive; node handles reference it
    device: Arc<cuda_core::Device>,
    /// KV-cache device pointers captured at graph creation, per layer.
    captured_k_ptrs: Vec<u64>,
    captured_v_ptrs: Vec<u64>,
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
// dimensions (4096, 11008, 14336, ...).  16 matches the tiling constant used
// inside CutileMatMul::forward_bt_device_to_device.
const BK: usize = 16;

// BN_GEMV: number of output neurons computed per GPU program in the multi-output
// quantized GEMV kernels (cutile_q*_gemv_bn).  Matches a CUDA warp (32 threads)
// so each program covers exactly one warp's worth of outputs, maximising occupancy.
// Standard transformer dimensions (4096, 8192, 11008, 14336, ...) are all multiples
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
/// Packed format: [n x k_blocks x Q2_0_PACKED_U32S] u32 elements.
/// Each block of 128 elements occupies 9 u32 slots:
///   slot 0      : f32 bits of block scale (f16 -> f32 at upload time)
///   slots 1..=8 : 8 u32 words of 2-bit packed weights (16 values per u32)
/// Dequant in kernel: (q - 1) x scale, q ??? {0,1,2,3}.
struct DeviceWeightQ2 {
    /// Packed u32 tensor on GPU, flat shape [n x k_blocks x Q2_0_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q2_0 blocks per row (= k / Q2_0_QK).
    k_blocks: usize,
}

/// Q6_K quantized weight tensor resident on the GPU.
///
/// Packed format: [n x k_blocks x Q6K_PACKED_U32S] u32 elements (53 u32/block, native ql+qh).
/// Each block of 256 elements occupies 81 u32 slots:
///   slot 0       : f32 bits of block super-scale d (f16 -> f32 at upload)
///   slots 1..17  : f32 bits of 16 per-group scales (i8 -> f32 at upload)
///   slots 17..81 : 64 u32 words, each holding 4 q_full bytes (values 0..63)
/// Dequant in kernel: d x scale[group] x (q_full ??? 32), group = element_idx / 16.
struct DeviceWeightQ6K {
    /// Packed u32 tensor on GPU, flat shape [n x k_blocks x Q6K_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q6_K blocks per row (= k / Q6K_QK).
    k_blocks: usize,
}

/// Q4_K quantized weight tensor resident on the GPU.
///
/// Packed format: [n x k_blocks x Q4K_PACKED_U32S] u32 elements.
/// Each block of 256 elements occupies 42 u32 slots:
///   slot 0       : f32 bits of d (super-scale, f16 -> f32)
///   slot 1       : f32 bits of dmin (super-min, f16 -> f32)
///   slots 2..6   : sc[2*j] | (sc[2*j+1]<<8) for j=0..4 (unpacked 6-bit -> 8-bit scales)
///   slots 6..10  : m[2*j]  | (m[2*j+1]<<8)  for j=0..4 (unpacked 6-bit -> 8-bit mins)
///   slots 10..42 : 32 u32 words of packed nibble bytes (128 bytes total)
/// Dequant in kernel: w[j*64+l] = d*sc[2*j]*lo_nibble - dmin*m[2*j]
///                    w[j*64+l+32] = d*sc[2*j+1]*hi_nibble - dmin*m[2*j+1]
struct DeviceWeightQ4K {
    /// Packed u32 tensor on GPU, flat shape [n x k_blocks x Q4K_PACKED_U32S].
    tensor: Arc<Tensor<u32>>,
    /// Number of output neurons.
    n: usize,
    /// Number of Q4_K blocks per row (= k / Q4K_BLOCK_SIZE).
    k_blocks: usize,
}

/// MXFP4-quantized projection weight in E2M1 packed nibble + UE8M0 block-scale format.
///
/// Each row has `k_blocks32` blocks of 32 elements:
///   - 16 packed E2M1 bytes per block (2 nibbles/byte, lo nibble first)
///   - 1 UE8M0 scale byte per block (scale = 2^(e-127))
///
/// GPU layout:
///   - `packed` tensor shape: [n * k_blocks32 * 16] u8 (E2M1 nibbles)
///   - `scales` tensor shape: [n * k_blocks32]      u8 (UE8M0 scales)
///
/// Per 256-element superblock: 128 + 8 = 136 bytes (vs Q4K 168 bytes, 19% smaller).
struct DeviceWeightMxfp4 {
    /// Packed E2M1 nibbles, flat shape [n * k_blocks32 * 16].
    packed: Arc<Tensor<u8>>,
    /// UE8M0 block scales, flat shape [n * k_blocks32].
    scales: Arc<Tensor<u8>>,
    /// Number of output neurons.
    n: usize,
    /// Number of 32-element blocks per row (= k / 32).
    k_blocks32: usize,
}

/// MXFP4-quantized projection weight in tiled kernel layout for mxf4_batch_gemm_tiled.
///
/// Packed E2M1 nibbles: [N][K/2] bytes, row-major.
/// Tile scales (b_scales): [(N/8)][(K/32)] bytes = [n8_tiles][k32_blocks].
///   Each scale covers all 8 n-rows x 32 k-cols in the tile.
///   Kernel index: b_scale_base = tile_n * k32_blocks + k32_idx.
struct DeviceWeightMxfp4Tiled {
    /// Packed E2M1 nibbles, flat [n * k/2] u8.
    packed: Arc<Tensor<u8>>,
    /// UE8M0 tile scales, flat [(n/8) * (k/32)] u8.
    tiled_scales: Arc<Tensor<u8>>,
    /// Number of output neurons (N).
    n: usize,
    /// Input feature dimension (K).
    k: usize,
}

fn next_pow2(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// IEEE 754 f16 bit pattern -> f32 (used in the Q2_0 upload path).
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

// Scratch-pool slot indices -- one pre-allocated decode output buffer per projection.
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
    /// next_pow2(hidden_size.div_ceil(BK) * BK) -- padded column count for h-wide activations.
    h_pad: usize,
    /// next_pow2(q_dim.div_ceil(BK) * BK) -- padded column count for attention output (o_proj).
    q_pad: usize,
    /// next_pow2(intermediate_size.div_ceil(BK) * BK) -- padded column count for FFN activations.
    inter_pad: usize,
    device_weights: HashMap<String, DeviceWeight>,
    /// Q4_K-quantized projection weights stored in packed GPU format.
    device_weights_q4k: HashMap<String, DeviceWeightQ4K>,
    /// Q2_0-quantized projection weights stored in packed GPU format.
    device_weights_q2: HashMap<String, DeviceWeightQ2>,
    /// Q6_K-quantized projection weights stored in packed GPU format.
    device_weights_q6k: HashMap<String, DeviceWeightQ6K>,
    /// MXFP4-quantized projection weights (E2M1 packed + UE8M0 block scales).
    /// Populated when original BF16/F16 safetensors weights are available.
    device_weights_mxfp4: HashMap<String, DeviceWeightMxfp4>,
    /// MXFP4-quantized weights in tiled kernel layout for mxf4_batch_gemm_tiled.
    /// Populated at model load time by dequanting Q4K weights to f32, then to MXFP4 tiled format.
    device_weights_mxfp4_tiled: HashMap<String, DeviceWeightMxfp4Tiled>,
    /// MXFP4 tiled batch GEMM dispatcher (sm_120a only, None on other hardware).
    mxfp4_gemm: Option<MxfpGemm>,
    /// Pre-allocated device buffer for MXFP4 activation packed nibbles (batch decode).
    /// Sized for [max_batch_graph * h_pad / 2] u8. Reused across layers.
    mxfp4_act_packed: Option<Arc<Tensor<u8>>>,
    /// Pre-allocated device buffer for MXFP4 activation tile scales (batch decode).
    /// Sized for [(max_batch_graph/16) * (h_pad/32)] u8.
    mxfp4_act_scales: Option<Arc<Tensor<u8>>>,
    /// Pre-allocated device buffer for MXFP4 output buffers (batch decode, inter-width).
    /// Sized for [max_batch_graph * inter] f32. Reused for gate/up/down projections.
    mxfp4_out_inter: Option<Arc<Tensor<f32>>>,
    /// Pre-allocated device buffer for MXFP4 output (h-width projections: Q, K, V, O, final).
    mxfp4_out_h: Option<Arc<Tensor<f32>>>,
    /// Pre-uploaded RMSNorm weight vectors (1-D, `[h_pad]` elements) keyed by
    /// their HuggingFace tensor name (e.g. `model.layers.0.input_layernorm.weight`).
    norm_weights: HashMap<String, Arc<Tensor<f32>>>,
    matmul: CutileMatMul,
    /// Q4_K on-GPU GEMV dispatcher -- shares the same CUDA stream as `matmul`.
    q4k_gemv: CutileQ4Gemv,
    /// Q2_0 on-GPU GEMV dispatcher -- shares the same CUDA stream as `matmul`.
    q2_gemv: CutileQ2Gemv,
    /// Hand-optimized oxide Q2_0 GEMV kernel (replaces cuTile for M=1 decode).
    ///
    /// `None` when the `oxide-gemv` feature was not compiled in or when the
    /// oxide build step failed; falls back to `q2_gemv` (cuTile) in that case.
    oxide_q2_gemv: Option<OxideQ2Gemv>,
    /// Oxide decode flash-attention kernel -- accepts kv_len as device pointer,
    /// enabling single-capture CUDA graph without stale kv_len.
    oxide_decode_attn: Option<OxideDecodeAttn>,
    /// Stable-address device scratch for the split-K decode attention
    /// (`OxideDecodeAttn::splitk_scratch_len` f32 elements). Allocated before
    /// any CUDA graph capture so the baked-in pointer stays valid for every
    /// replay of the graph's lifetime. `None` when the split-K pair is absent
    /// from the PTX (falls back to kpar) or allocation failed.
    attn_splitk_scratch: Option<Arc<Tensor<f32>>>,
    /// TEMPURA_DEBUG_NAN=1: run decode steps WITHOUT CUDA graph capture and
    /// D2H-check every intermediate tensor, eprinting the first op that
    /// produces NaN (numerical bisection of the decode pipeline).
    debug_nan_trace: bool,
    /// Hand-optimized oxide Q4_K GEMV kernel (replaces cuTile for M=1 decode).
    oxide_q4k_gemv: Option<OxideQ4kGemv>,
    /// Hand-optimized oxide Q6_K GEMV kernel (replaces cuTile for M=1 decode).
    oxide_q6k_gemv: Option<OxideQ6kGemv>,
    /// Q6_K on-GPU GEMV dispatcher -- shares the same CUDA stream as `matmul`.
    q6k_gemv: CutileQ6kGemv,
    flash_attn: CutileFlashAttention,
    cutile_rmsnorm: CutileRmsNorm,
    cutile_rope: CutileRoPE,
    cutile_elementwise: CutileElementwise,
    /// Scatter-write kernel for KV cache updates inside the CUDA graph.
    ///
    /// Reads `d_kv_write_offset` from device at execution time, eliminating
    /// 64 `cuGraphExecMemcpyNodeSetParams` calls per decode step.
    kv_scatter_write: KvScatterWrite,
    /// Pre-allocated decode output scratch buffers indexed by SLOT_* constants.
    ///
    /// Eliminates the `api::zeros` allocation inside `forward_bt_device_to_device`
    /// for the 7 projection GEMMs + lm_head per decode token (m=1 only).
    /// Single-threaded inference guarantees each slot's Arc has strong_count == 1
    /// when the next layer begins, so `Arc::try_unwrap` can reclaim the Tensor
    /// without re-allocating device memory.
    scratch: RefCell<[Option<Arc<Tensor<f32>>>; NUM_SLOTS]>,
    /// Pre-allocated RoPE scratch for Q (sized for min(max_seq_len, 4096) x n_heads).
    q_rope_scratch: RefCell<RopeScratch>,
    /// Pre-allocated RoPE scratch for K (sized for min(max_seq_len, 4096) x n_kv_heads).
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
    /// `decode_attention_device` -- 28 blocked allocs per decode token become zero.
    attn_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    /// Pre-allocated output buffer for `silu_mul_device_scratch` (seq_len=1).
    ///
    /// Same motivation as `attn_out_scratch`: eliminates 28 sync-on allocs per token.
    silu_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,

    /// Pre-allocated zero-padded scratch for down-projection input when inter_pad != inter.
    ///
    /// For models with non-power-of-2 intermediate sizes (e.g. Llama-3.1-8B inter=14336,
    /// inter_pad=16384), the SiLU output needs to be padded from `inter` to `inter_pad`
    /// before the down projection matmul.  Without this, the path does a D->H download +
    /// H->D re-upload (with a stream sync) every layer -- 32 stalls per decode token.
    ///
    /// This scratch holds `inter_pad` f32 elements.  Positions `0..inter` are overwritten
    /// by a GPU-side D->D copy from the SiLU output each layer; positions `inter..inter_pad`
    /// remain zero from the initial allocation (never written after init).
    down_proj_pad_scratch: RefCell<Option<Arc<Tensor<f32>>>>,

    /// Prefill (m>1) GEMM output buffers indexed by SLOT_* constants.
    ///
    /// The decode `scratch` pool is sized for m=1 and its tensors are handed
    /// to CUDA-graph-captured decode paths, so prefill keeps a separate pool:
    /// grown on demand to `[m_pad * n]` for the largest chunk seen, reused
    /// across layers (single-stream ordering makes overwrite-after-consume
    /// safe). Never referenced by any captured graph -- prefill runs eager.
    prefill_scratch: RefCell<[Option<Arc<Tensor<f32>>>; NUM_SLOTS]>,
    /// MXFP4 activation buffers for prefill, grown on demand to the largest
    /// `[m_pad, k]` seen. Kept separate from `mxfp4_act_packed`/`_scales`
    /// because those are pointer-baked into the captured batch-decode graph
    /// and must never be reallocated.
    prefill_mxfp4_packed: RefCell<Option<Arc<Tensor<u8>>>>,
    prefill_mxfp4_scales: RefCell<Option<Arc<Tensor<u8>>>>,

    // -- CUDA graph capture fields ---------------------------------------------
    //
    // For decode (seq_len=1) the forward pass is captured into a CUDA graph on
    // the first step and replayed on subsequent steps.  Before each replay the
    // baked-in per-step kernel parameters (position, kv_len, KV write offset)
    // are refreshed via `cuGraphExecUpdate_v2`, which re-captures a lightweight
    // template on the same stream and patches the exec in-place -- one driver call
    // instead of ~200 individual `cuLaunchKernel` submissions.

    /// Pre-allocated device buffer that holds the embedding of the current decode
    /// token.  The CPU computes the embedding, copies it here with
    /// `cuda_core::memcpy_htod_async` BEFORE `cuStreamBeginCapture`, so the graph
    /// reads the latest value on every replay.
    d_embed: Arc<Tensor<f32>>,

    // -- Device-side decode scalars for single-capture CUDA graph (Opt 5) ------
    //
    // Written H->D before each graph replay so the graph can be captured once
    // and replayed N times without cuStreamBeginCapture per token.
    /// kv_len after the current step's KV append (f32 for cuTile compatibility; read by flash attention).
    d_kv_len: Arc<Tensor<f32>>,
    /// Current token position for RoPE (f32 for cuTile compatibility).
    d_positions: Arc<Tensor<f32>>,
    /// KV cache write slot before the current step's append (raw i32, not a cuTile tensor).
    d_kv_write_offset: Arc<Tensor<i32>>,

    // -- CUDA-graph-stable RoPE rows --------------------------------------------
    //
    // cutile_rope_from_table takes `pos` as a host launch scalar, which a CUDA
    // graph bakes at capture time and replays STALE (every decoded token after
    // the first rotated at the capture position). Instead of patching graph
    // nodes, the captured graph gathers the current cos/sin table row into
    // these one-row buffers with `d2d_gather_row`, which reads the position
    // from `d_positions` at EXECUTION time (same pattern as the KV write
    // offset). The rope kernel then runs with pos=0 against the row buffers.
    /// Current cos table row `[head_dim/2]`, refreshed in-graph per step.
    d_rope_cos_row: Arc<Tensor<f32>>,
    /// Current sin table row `[head_dim/2]`, refreshed in-graph per step.
    d_rope_sin_row: Arc<Tensor<f32>>,

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

    // -- Multi-stream QKV concurrency ------------------------------------------
    //
    // K [kv_dim x h] and V [kv_dim x h] are independent of Q [q_dim x h] and
    // can execute concurrently: for GQA models K and V are 4x smaller and hide
    // behind Q's latency.  During CUDA graph capture in RELAXED mode, operations
    // on `stream_kv` that wait on `event_fork` (recorded on `stream_main`) are
    // automatically included in the graph as concurrent nodes.

    /// Secondary CUDA stream for concurrent K and V GEMVs in decode graph.
    stream_kv: Arc<cuda_core::Stream>,
    /// Fork event: recorded on stream_main after normed_dev is ready; waited on
    /// by stream_kv to join the capture graph and establish the dependency edge.
    event_fork: cuda_core::Event,
    /// Join event: recorded on stream_kv after V GEMV; waited on by stream_main
    /// before RoPE to ensure K and V results are visible on the main timeline.
    event_kv_done: cuda_core::Event,

    // -- Batch decode CUDA graph (M > 1 simultaneous sequences) ---------------
    //
    // Pre-allocated device buffers sized for MAX_BATCH_GRAPH sequences.
    // Written H->D before each graph replay; the graph reads from stable
    // device pointers -- topology-stable across steps.
    //
    // Requirements for capture:
    //   (a) All Q4K weights -> sm120 batch kernel (no dynamic allocation).
    //   (b) All M sequences synchronized (same position, same kv_len).
    //   (c) q_pad == q_dim (no attn-output padding needed; true for all standard
    //       LLaMA/Qwen models where n_heads * head_dim is already a power-of-2).

    /// Maximum number of sequences the batch graph is sized for.
    max_batch_graph: usize,

    /// One [h_pad] embedding buffer per sequence (H->D written before graph launch).
    d_embed_batch: Vec<Arc<Tensor<f32>>>,

    /// Working hidden state [max_batch_graph x h_pad] used throughout the layer loop.
    /// Initialised from the M embeddings (D->D concat) before graph launch.
    batch_hidden: Arc<Tensor<f32>>,

    /// Pre-attention normed hidden [max_batch_graph x h_pad].
    batch_normed1: Arc<Tensor<f32>>,
    /// Post-attention normed hidden [max_batch_graph x h_pad].
    batch_normed2: Arc<Tensor<f32>>,

    /// Batch GEMV outputs -- pre-allocated to avoid allocation inside graph capture.
    batch_gemv_q:    Arc<Tensor<f32>>,   // [max_batch_graph x q_dim]
    batch_gemv_k:    Arc<Tensor<f32>>,   // [max_batch_graph x kv_dim]
    batch_gemv_v:    Arc<Tensor<f32>>,   // [max_batch_graph x kv_dim]
    batch_gemv_gate: Arc<Tensor<f32>>,   // [max_batch_graph x inter]
    batch_gemv_up:   Arc<Tensor<f32>>,   // [max_batch_graph x inter]
    batch_gemv_o:    Arc<Tensor<f32>>,   // [max_batch_graph x h_pad] (O-proj output)
    batch_gemv_down: Arc<Tensor<f32>>,   // [max_batch_graph x h_pad] (down-proj output)

    /// Per-sequence Q slices for RoPE [max_batch_graph][q_dim].
    batch_q_seqs: Vec<Arc<Tensor<f32>>>,
    /// Per-sequence K slices for RoPE [max_batch_graph][kv_dim].
    batch_k_seqs: Vec<Arc<Tensor<f32>>>,
    /// Per-sequence V slices (stable device ptrs for KV scatter-write) [max_batch_graph][kv_dim].
    batch_v_seqs: Vec<Arc<Tensor<f32>>>,

    /// Stable K-after-RoPE output per sequence [max_batch_graph][kv_dim].
    /// D->D copied from RoPE output each layer so the KV scatter-write kernel
    /// reads from a known, fixed device address baked in at graph capture.
    batch_k_rot: Vec<Arc<Tensor<f32>>>,
    /// Stable Q-after-RoPE output per sequence [max_batch_graph][q_pad].
    batch_q_rot: Vec<Arc<Tensor<f32>>>,

    /// Per-sequence attention output [max_batch_graph][q_dim].
    batch_attn_out: Vec<Arc<Tensor<f32>>>,

    /// Concatenated O-proj input [max_batch_graph x q_pad] (= batch_attn_out for q_pad==q_dim).
    batch_oproj_in: Arc<Tensor<f32>>,

    /// Scratch for batch residual adds.
    batch_post_attn_add_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    batch_ffn_add_scratch:       RefCell<Option<Arc<Tensor<f32>>>>,

    /// SiLU-gated multiply scratch [max_batch_graph x inter].
    batch_silu_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,

    /// Compact down-proj input [max_batch_graph x inter] (used when inter_pad != inter).
    batch_down_compact: Option<Arc<Tensor<f32>>>,

    /// RMSNorm scratch for batch pre-attn / post-attn / final norm.
    batch_pre_attn_norm_scratch:  RefCell<RmsnormScratch>,
    batch_post_attn_norm_scratch: RefCell<RmsnormScratch>,
    batch_final_norm_scratch:     RefCell<RmsnormScratch>,

    /// Q8_1 quantisation scratch for h-width Q4K projections (Q, K, V, O, gate, up).
    /// Sized [max_batch_graph x h_pad / 4] (qs) and [max_batch_graph x h_q8_blocks] (ds).
    batch_q8_qs_h: Arc<Tensor<u32>>,
    batch_q8_ds_h: Arc<Tensor<u32>>,

    /// Q8_1 quantisation scratch for inter-width Q4K projection (down).
    /// Sized [max_batch_graph x inter / 4] (qs) and [max_batch_graph x inter_q8_blocks] (ds).
    batch_q8_qs_i: Arc<Tensor<u32>>,
    batch_q8_ds_i: Arc<Tensor<u32>>,

    /// Per-sequence silu-output row buffers [max_batch_graph][inter].
    /// Used as pre-allocated temporaries when down_proj is Q6K (Q4_K_M models),
    /// so that the sequential Q6K GEMV calls can be captured in the CUDA graph.
    batch_silu_row_bufs: Vec<Arc<Tensor<f32>>>,

    /// Device pointer arrays [max_batch_graph x u64] for batch KV scatter-write.
    /// Stored as raw CUdeviceptr (cuMemAlloc'd) because cuTile Tensor<u64> is
    /// not a supported type.  The memory holds `max_batch_graph x 8` bytes.
    /// Repurposed as K/V cache base pointer arrays for oxide_decode_attn_batch.
    batch_src_k_ptrs_dev: cuda_core::sys::CUdeviceptr,
    batch_src_v_ptrs_dev: cuda_core::sys::CUdeviceptr,

    /// Per-sequence kv_len device array [max_batch_graph x i32].
    /// Updated H->D before each batch graph replay with current seq_len+1 values.
    batch_attn_kv_lens_dev: cuda_core::sys::CUdeviceptr,

    /// Per-layer K/V cache base pointers [MAX_ATTN_LAYERS x max_batch_graph x u64].
    /// Filled once before graph capture (stable KV cache addresses); never updated
    /// inside the CUDA graph itself.
    batch_attn_all_k_ptrs: cuda_core::sys::CUdeviceptr,
    batch_attn_all_v_ptrs: cuda_core::sys::CUdeviceptr,

    /// Cached batch CUDA graph executable (dp4a path).
    cached_batch_graph: RefCell<Option<BatchDecodeGraph>>,

    /// Cached MXFP4 batch CUDA graph executable.
    /// Separate from cached_batch_graph so both dp4a and MXFP4 graphs can coexist.
    /// Captured with forward_batch_mxfp4_inner_graph (MXFP4 unconditional).
    cached_mxfp4_batch_graph: RefCell<Option<BatchDecodeGraph>>,

    /// True when every projection weight in every layer is Q4K -- a precondition
    /// for the batch CUDA graph path (which only handles Q4K projections).
    all_proj_q4k: bool,
}

/// RAII wrapper for the batch-decode CUDA graph executable.
struct BatchDecodeGraph {
    cu_exec:          cuda_core::sys::CUgraphExec,
    cu_graph:         cuda_core::sys::CUgraph,
    device:           Arc<cuda_core::Device>,
    m:                usize,
    /// KV-cache device pointers captured at graph instantiation time [layer][seq].
    /// Checked on every replay: if any pointer changes (fresh-per-call bench pattern),
    /// the graph is invalidated and re-captured on the next call.
    captured_k_ptrs: Vec<Vec<u64>>,
    captured_v_ptrs: Vec<Vec<u64>>,
}

impl Drop for BatchDecodeGraph {
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

unsafe impl Send for BatchDecodeGraph {}
unsafe impl Sync for BatchDecodeGraph {}

// SAFETY: CutileMatMul, CutileFlashAttention and Tensor<f32> hold raw CUDA pointers,
// but tempura uses a single-threaded inference path per model instance.
unsafe impl Send for CutileGpuModel {}
unsafe impl Sync for CutileGpuModel {}

/// Query free device VRAM using cuMemGetInfo_v2.
/// Returns 0 on failure (caller treats it as "no headroom").
fn query_free_vram_bytes() -> usize {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let rc = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) };
    if rc != 0 {
        eprintln!("[cutile] cuMemGetInfo_v2 failed (rc={rc}); falling back to max_batch=2");
        return 0;
    }
    free
}

/// Compute the maximum batch size the GPU can sustain for the batch decode graph.
///
/// Scratch tensors sized `[mb x dim]` and KV cache slots dominate VRAM per extra
/// sequence.  We query free VRAM after model weights are uploaded and divide by
/// the per-sequence cost, then clamp to [2, 16].
fn compute_max_batch(
    config: &ModelConfig,
    h_pad: usize,
    q_dim: usize,
    kv_dim: usize,
    inter: usize,
) -> usize {
    let gpu_max_seq_len = config.max_seq_len.min(crate::cutile_kv_cache::GPU_KV_MAX_SEQ_LEN);

    // Scratch tensors allocated once for the whole batch (flat [mb x dim], f32).
    // batch_hidden, batch_normed1, batch_normed2   -> 3 x h_pad
    // batch_gemv_q                                 -> q_dim
    // batch_gemv_k, batch_gemv_v                   -> 2 x kv_dim
    // batch_gemv_gate, batch_gemv_up               -> 2 x inter
    // batch_gemv_o, batch_gemv_down                -> 2 x h_pad
    // batch_oproj_in                               -> q_dim  (q_pad ~ q_dim)
    // batch_post_attn_add, batch_ffn_add           -> 2 x h_pad
    // batch_silu_out                               -> inter
    // batch_down_compact (worst case)              -> inter
    // Q8 scratch (qs_h + ds_h + qs_i + ds_i)      -> h_pad/4 + h_pad/32 + inter/4 + inter/32 (f32 elements in u32)
    // RmsnormScratch x3, silu_row_bufs, d_embed    -> approximated as h_pad
    // Total approximation (in f32 elements per extra sequence):
    let f32_per_extra = 3 * h_pad          // hidden, normed1, normed2
        + q_dim                            // gemv_q
        + 2 * kv_dim                       // gemv_k, gemv_v
        + 2 * inter                        // gemv_gate, gemv_up
        + 2 * h_pad                        // gemv_o, gemv_down
        + q_dim                            // oproj_in
        + 2 * h_pad                        // post_attn_add, ffn_add
        + inter                            // silu_out
        + inter                            // down_compact (worst case)
        + h_pad;                           // per-seq vecs + rmsnorm + silu_row

    let scratch_bytes_per_extra = f32_per_extra * std::mem::size_of::<f32>();

    // KV cache per extra sequence (f32 per layer per K and V, gpu_max_seq_len slots).
    let kv_cache_bytes_per_extra =
        config.num_layers * 2 * gpu_max_seq_len * kv_dim * std::mem::size_of::<f32>();

    let vram_per_extra = scratch_bytes_per_extra + kv_cache_bytes_per_extra;

    let free_vram = query_free_vram_bytes();
    // Reserve 1 GB headroom for cuTile workspace and driver overhead.
    let usable = free_vram.saturating_sub(1 << 30);

    let extra = if vram_per_extra > 0 { usable / vram_per_extra } else { 0 };
    let mb = (1 + extra).clamp(2, 16);

    eprintln!(
        "[cutile] batch graph max_batch={mb} \
        (scratch={:.1}MiB/seq, kv_cache={:.0}MiB/seq, free_vram={:.1}GB)",
        scratch_bytes_per_extra as f64 / (1 << 20) as f64,
        kv_cache_bytes_per_extra as f64 / (1 << 20) as f64,
        free_vram as f64 / (1u64 << 30) as f64,
    );

    mb
}

impl CutileGpuModel {
    // -- private helpers -------------------------------------------------------

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
        // mmap-resident quantized tensors -- not in typed_tensors.
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
    ///   u32[0]    = f32 bits of block scale (f16 -> f32)
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

                // Slot 0: block scale (f16 -> f32, stored as u32 bit pattern).
                let scale_f16_bits = u16::from_le_bytes([block[0], block[1]]);
                let scale_f32 = f16_bits_to_f32_inline(scale_f16_bits);
                packed.push(scale_f32.to_bits());

                // Slots 1-8: 32 packed bytes -> 8 u32 words.
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
    /// Converts the raw Q6_K bytes to the 53-u32-per-block compact GPU layout:
    ///   u32[0]      = f32 bits of block super-scale d (f16 -> f32)
    ///   u32[1..5]   = 4 u32 words: 16 group scales as bias-128 u8 (4 per u32)
    ///                 stored_byte = (scale_i8 as i32 + 128) as u8
    ///   u32[5..37]  = 32 u32 words: ql[0..128] in native GGUF order (4 per u32)
    ///   u32[37..53] = 16 u32 words: qh[0..64] in native GGUF order (4 per u32)
    ///
    /// Saves 35% DRAM vs the old 81-u32 expanded q_full format.
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

                // Slots 1..5: 16 group scales as bias-128 u8, 4 per u32.
                for w in 0..4usize {
                    let b0 = ((scales[w * 4] as i8) as i32 + 128) as u8;
                    let b1 = ((scales[w * 4 + 1] as i8) as i32 + 128) as u8;
                    let b2 = ((scales[w * 4 + 2] as i8) as i32 + 128) as u8;
                    let b3 = ((scales[w * 4 + 3] as i8) as i32 + 128) as u8;
                    packed.push(
                        (b0 as u32)
                            | ((b1 as u32) << 8)
                            | ((b2 as u32) << 16)
                            | ((b3 as u32) << 24),
                    );
                }

                // Slots 5..37: ql[0..128] packed 4 u8 per u32 (native GGUF order).
                for i in 0..32usize {
                    let w = (ql[i * 4] as u32)
                        | ((ql[i * 4 + 1] as u32) << 8)
                        | ((ql[i * 4 + 2] as u32) << 16)
                        | ((ql[i * 4 + 3] as u32) << 24);
                    packed.push(w);
                }

                // Slots 37..53: qh[0..64] packed 4 u8 per u32 (native GGUF order).
                for i in 0..16usize {
                    let w = (qh[i * 4] as u32)
                        | ((qh[i * 4 + 1] as u32) << 8)
                        | ((qh[i * 4 + 2] as u32) << 16)
                        | ((qh[i * 4 + 3] as u32) << 24);
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
    ///   u32[0]      = f32 bits of d (f16 -> f32)
    ///   u32[1]      = f32 bits of dmin (f16 -> f32)
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
        let expected_bytes = n * k_blocks * Q4K_BLOCK_BYTES;
        if bytes.len() != expected_bytes {
            eprintln!(
                "[q4k] {name}: expected {expected_bytes} bytes (n={n} k={k}), got {} -- \
                 shape mismatch (hybrid SSM model?), skipping Q4K upload",
                bytes.len()
            );
            return Ok(false);
        }
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
        let expected_bytes = n * k_blocks * 144;
        if bytes.len() != expected_bytes {
            // Shape mismatch -- e.g. hybrid models (qwen35/Mamba) where attention layers
            // store a non-standard Q dim (12288 vs expected 6144 for n_heads*head_dim).
            // Fall back to f32 dequantized path rather than crashing with a cuTile reshape error.
            eprintln!(
                "[q4k_raw] {name}: expected {expected_bytes} bytes (n={n} k={k} k_blocks={k_blocks}), \
                 got {} -- likely unsupported architecture (hybrid SSM or non-standard Q dim), skipping Q4K upload",
                bytes.len()
            );
            return Ok(false);
        }
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

    /// Quantize BF16 or F16 weights to MXFP4 (E2M1 packed nibbles + UE8M0 block scales)
    /// and upload the result to the GPU as two separate u8 tensors.
    ///
    /// `dtype` controls the source interpretation: `"bf16"` or `"f16"`.
    ///
    /// Returns `false` (and does nothing) when the tensor is absent in `qw`.
    fn maybe_upload_mxfp4(
        stream: &Arc<Stream>,
        device_weights_mxfp4: &mut HashMap<String, DeviceWeightMxfp4>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
        dtype: &str,
    ) -> Result<bool, tempura_kernels::KernelError> {
        use tempura_kernels::mxfp4::{
            bf16_bytes_to_f32, f16_bytes_to_f32, quantize_row_to_mxfp4, MXFP4_BLOCK_SIZE,
        };

        let raw_bytes: &[u8] = match dtype {
            "bf16" => match qw.get_f16(name) {
                Some(b) if b.len() == n * k * 2 => b,
                Some(_) | None => return Ok(false),
            },
            "f16" => match qw.get_f16(name) {
                Some(b) if b.len() == n * k * 2 => b,
                Some(_) | None => return Ok(false),
            },
            "f32" => {
                // f32 path: go through get_f32 and convert directly.
                let f32_data = match qw.get_f32(name) {
                    Some(d) if d.len() == n * k => d,
                    Some(_) | None => return Ok(false),
                };
                if k % MXFP4_BLOCK_SIZE != 0 {
                    return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                        "MXFP4 requires k={k} to be a multiple of block_size={MXFP4_BLOCK_SIZE}"
                    )));
                }
                let k_blocks32 = k / MXFP4_BLOCK_SIZE;
                let mut all_packed: Vec<u8> = Vec::with_capacity(n * k / 2);
                let mut all_scales: Vec<u8> = Vec::with_capacity(n * k_blocks32);
                for row in 0..n {
                    let row_data = &f32_data[row * k..(row + 1) * k];
                    let (p, s) = quantize_row_to_mxfp4(row_data);
                    all_packed.extend_from_slice(&p);
                    all_scales.extend_from_slice(&s);
                }
                let packed_tensor: Arc<Tensor<u8>> =
                    api::copy_host_vec_to_device(&Arc::new(all_packed))
                        .sync_on(stream)
                        .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                        .into();
                let scales_tensor: Arc<Tensor<u8>> =
                    api::copy_host_vec_to_device(&Arc::new(all_scales))
                        .sync_on(stream)
                        .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                        .into();
                device_weights_mxfp4.insert(
                    name.to_string(),
                    DeviceWeightMxfp4 {
                        packed: packed_tensor,
                        scales: scales_tensor,
                        n,
                        k_blocks32,
                    },
                );
                return Ok(true);
            }
            other => {
                return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                    "maybe_upload_mxfp4: unsupported dtype '{other}'; use 'bf16', 'f16', or 'f32'"
                )));
            }
        };

        if k % MXFP4_BLOCK_SIZE != 0 {
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "MXFP4 requires k={k} to be a multiple of block_size={MXFP4_BLOCK_SIZE}"
            )));
        }
        let k_blocks32 = k / MXFP4_BLOCK_SIZE;

        // Convert bytes to f32 according to the declared dtype.
        let f32_vals: Vec<f32> = match dtype {
            "bf16" => bf16_bytes_to_f32(raw_bytes),
            _ => f16_bytes_to_f32(raw_bytes),
        };

        debug_assert_eq!(f32_vals.len(), n * k);

        let mut all_packed: Vec<u8> = Vec::with_capacity(n * k / 2);
        let mut all_scales: Vec<u8> = Vec::with_capacity(n * k_blocks32);
        for row in 0..n {
            let row_data = &f32_vals[row * k..(row + 1) * k];
            let (p, s) = quantize_row_to_mxfp4(row_data);
            all_packed.extend_from_slice(&p);
            all_scales.extend_from_slice(&s);
        }

        debug_assert_eq!(all_packed.len(), n * k / 2);
        debug_assert_eq!(all_scales.len(), n * k_blocks32);

        let packed_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(all_packed))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();
        let scales_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(all_scales))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();

        device_weights_mxfp4.insert(
            name.to_string(),
            DeviceWeightMxfp4 {
                packed: packed_tensor,
                scales: scales_tensor,
                n,
                k_blocks32,
            },
        );
        Ok(true)
    }

    /// Dequantize a Q4K weight row from GGUF bytes to f32, then quantize to MXFP4 tiled
    /// kernel layout and upload to GPU.
    ///
    /// `n` and `k` are the weight matrix shape [N, K].
    /// `k` must be a multiple of 64, `n` must be a multiple of 8.
    ///
    /// Returns `false` when the tensor is absent in `qw` or dimensions are incompatible.
    fn maybe_upload_mxfp4_tiled_from_q4k(
        stream: &Arc<Stream>,
        dw_mxfp4_tiled: &mut HashMap<String, DeviceWeightMxfp4Tiled>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        use tempura_kernels::ggml::blocks::{BlockQ4K, BLOCK_Q4K_SIZE};
        use tempura_kernels::ggml::dequantize::dequantize_row_q4_k;
        use tempura_kernels::mxfp4::quantize_matrix_to_mxfp4_tiled;

        if n % 8 != 0 || k % 64 != 0 {
            return Ok(false);
        }
        let k_blocks = k / 256; // Q4K block size is 256 elements
        if k_blocks == 0 {
            return Ok(false);
        }
        let raw = match qw.get_q4k(name) {
            Some(b) if b.len() == n * k_blocks * BLOCK_Q4K_SIZE => b,
            _ => return Ok(false),
        };

        // Dequantize all N rows from Q4K to f32.
        let mut f32_weights = vec![0.0f32; n * k];
        let blocks_per_row = k_blocks;
        for row in 0..n {
            let block_start = row * blocks_per_row;
            let blocks = unsafe {
                std::slice::from_raw_parts(
                    raw.as_ptr().add(block_start * BLOCK_Q4K_SIZE) as *const BlockQ4K,
                    blocks_per_row,
                )
            };
            dequantize_row_q4_k(blocks, &mut f32_weights[row * k..(row + 1) * k]);
        }

        // Quantize to MXFP4 tiled kernel layout.
        let (packed, tiled_scales) = quantize_matrix_to_mxfp4_tiled(&f32_weights, n, k);

        let packed_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(packed))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();
        let scales_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(tiled_scales))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();

        dw_mxfp4_tiled.insert(
            name.to_string(),
            DeviceWeightMxfp4Tiled { packed: packed_tensor, tiled_scales: scales_tensor, n, k },
        );
        Ok(true)
    }

    /// Load BF16 weights from a GGUF mmap, convert BF16->f32->MXFP4, and upload
    /// the tiled kernel layout to the GPU.
    ///
    /// BF16 is an exact truncation of f32 (upper 16 bits), so conversion is lossless
    /// within the f32 range -- no quantization error beyond the original BF16 precision.
    ///
    /// `n` and `k` are the weight matrix shape [N, K].
    /// `k` must be a multiple of 64, `n` must be a multiple of 8.
    ///
    /// Returns `false` when the tensor is absent in `qw` or dimensions are incompatible.
    fn maybe_upload_mxfp4_tiled_from_bf16(
        stream: &Arc<Stream>,
        dw_mxfp4_tiled: &mut HashMap<String, DeviceWeightMxfp4Tiled>,
        qw: &QuantizedWeights,
        name: &str,
        n: usize,
        k: usize,
    ) -> Result<bool, tempura_kernels::KernelError> {
        use tempura_kernels::mxfp4::{bf16_bytes_to_f32, quantize_matrix_to_mxfp4_tiled};

        if n % 8 != 0 || k % 64 != 0 {
            return Ok(false);
        }
        let raw = match qw.get_bf16(name) {
            Some(b) if b.len() == n * k * 2 => b,
            _ => return Ok(false),
        };

        // BF16 -> f32: exact upcast, no information loss beyond original BF16 precision.
        let f32_weights = bf16_bytes_to_f32(raw);
        debug_assert_eq!(f32_weights.len(), n * k);

        let (packed, tiled_scales) = quantize_matrix_to_mxfp4_tiled(&f32_weights, n, k);

        let packed_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(packed))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();
        let scales_tensor: Arc<Tensor<u8>> =
            api::copy_host_vec_to_device(&Arc::new(tiled_scales))
                .sync_on(stream)
                .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?
                .into();

        dw_mxfp4_tiled.insert(
            name.to_string(),
            DeviceWeightMxfp4Tiled { packed: packed_tensor, tiled_scales: scales_tensor, n, k },
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
    /// Only layer 0 weights are exercised -- all layers share the same shapes.
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

    // -- constructors ----------------------------------------------------------

    /// Dequantize all projection weights from `qw` and upload them to the GPU.
    ///
    /// Calling this once at model-load time eliminates repeated host->device weight
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
        let mut dw_mxfp4_tiled: HashMap<String, DeviceWeightMxfp4Tiled> = HashMap::new();
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

        // lm_head  [vocab, h] -- typically not Q4_K/Q2_0; keep dequantized path.
        Self::maybe_upload(&matmul, &mut dw, qw, "lm_head.weight", vocab, h)?;

        // TEMPURA_NO_MXFP4=1 skips MXFP4 tiling of the Q4K/BF16 projections so
        // the batch decode graph dispatches the original Q4K/Q6K GEMV kernels
        // instead of the MXFP4 tensor-core path -- numerical bisect switch for
        // the batch pipeline (TEMPURA_NO_OXIDE_ATTN pattern). The tiled map
        // stays empty and every dispatch site falls back naturally.
        let mxfp4_tiling_disabled = std::env::var("TEMPURA_NO_MXFP4")
            .map(|v| v == "1")
            .unwrap_or(false);
        if mxfp4_tiling_disabled {
            eprintln!("[mxfp4] tiled upload disabled (TEMPURA_NO_MXFP4=1)");
        }

        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");

            // For each projection: try Q4_K GPU-resident first, then Q2_0, then f32.
            // The inner block avoids a closure so all three maps can be borrowed mutably.
            let mut try_proj = |name: String, n: usize, k: usize| {
                (|| -> Result<(), tempura_kernels::KernelError> {
                    // Priority: Q4_K -> Q6_K -> Q2_0 -> f32 dequantized.
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

            // MXFP4 tiled upload: dequant Q4K -> f32 -> MXFP4 tiled format for sm_120a.
            // Best-effort; silently skips if shape is incompatible or Q4K data absent.
            if !mxfp4_tiling_disabled {
                for (wname, n_dim, k_dim) in [
                    (format!("{p}.self_attn.q_proj.weight"),    q_dim, h),
                    (format!("{p}.self_attn.k_proj.weight"),    kv_dim, h),
                    (format!("{p}.self_attn.v_proj.weight"),    kv_dim, h),
                    (format!("{p}.self_attn.o_proj.weight"),    h,      q_dim),
                    (format!("{p}.mlp.gate_proj.weight"),       inter,  h),
                    (format!("{p}.mlp.up_proj.weight"),         inter,  h),
                    (format!("{p}.mlp.down_proj.weight"),       h,      inter),
                ] {
                    // Try Q4K first (quantized GGUF), then BF16 (full-precision GGUF).
                    // Only one will succeed; the other returns false without touching GPU state.
                    let q4k_ok = match Self::maybe_upload_mxfp4_tiled_from_q4k(
                        matmul.stream(), &mut dw_mxfp4_tiled, qw, &wname, n_dim, k_dim,
                    ) {
                        Ok(ok) => ok,
                        Err(e) => { eprintln!("[mxfp4] tiled q4k {wname}: {e}"); false }
                    };
                    if !q4k_ok {
                        if let Err(e) = Self::maybe_upload_mxfp4_tiled_from_bf16(
                            matmul.stream(), &mut dw_mxfp4_tiled, qw, &wname, n_dim, k_dim,
                        ) {
                            eprintln!("[mxfp4] tiled bf16 {wname}: {e}");
                        }
                    }
                }
            }

            // RMSNorm weights -- pre-upload so forward_device never copies them per token.
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
        // TEMPURA_NO_OXIDE_ATTN=1 disables the module (attention falls back to
        // cuTile flash-attn, SiLU to the cutile elementwise path) -- numerical
        // bisect switch for the decode pipeline.
        let oxide_decode_attn = if std::env::var("TEMPURA_NO_OXIDE_ATTN").map(|v| v == "1").unwrap_or(false) {
            eprintln!("[cutile] oxide decode-attn disabled (TEMPURA_NO_OXIDE_ATTN=1)");
            None
        } else {
            match OxideDecodeAttn::new(matmul.stream().clone()) {
                Ok(oda) => { eprintln!("[cutile] oxide decode-attn loaded (batch={}, splitk={})", oda.has_batch(), oda.has_splitk()); Some(oda) }
                Err(e)  => { eprintln!("[cutile] oxide decode-attn unavailable: {e}"); None }
            }
        };
        // Split-K decode attention scratch: one stable-address buffer for the
        // CUDA graph's lifetime (pointer baked in at capture, never reallocated).
        let attn_splitk_scratch = oxide_decode_attn.as_ref()
            .filter(|oda| oda.has_splitk())
            .map(|_| {
                Arc::new(
                    api::zeros::<f32>(&[OxideDecodeAttn::splitk_scratch_len(n_heads, head_dim)])
                        .sync_on(matmul.stream())
                        .expect("splitk attn scratch: device allocation failed"),
                )
            });
        let debug_nan_trace = std::env::var("TEMPURA_DEBUG_NAN").is_ok();
        let kv_scatter_write = KvScatterWrite::new()
            .expect("KvScatterWrite: failed to load scatter-write PTX kernel");

        // Load MXFP4 tiled batch GEMM (sm_120a only, None on other hardware).
        let mxfp4_gemm = MxfpGemm::try_load(matmul.stream().clone());
        let mxfp4_count = dw_mxfp4_tiled.len();
        if mxfp4_count > 0 {
            eprintln!("[mxfp4] {mxfp4_count} MXFP4 tiled weights ready; gemm={}", mxfp4_gemm.is_some());
        }

        // Pre-allocate MXFP4 activation scratch for batch decode CUDA graph.
        // m_pad=16 always: max_batch_graph<=16 and (x+15)&!15 == 16 for all x in 1..=16.
        // Sized for max(h_pad64, inter_pad64) so one buffer covers h-width and inter-width
        // projections. Reused across layers (serial kernel launches in CUDA graph).
        let (mxfp4_batch_act_packed, mxfp4_batch_act_scales) =
            if mxfp4_gemm.is_some() && !dw_mxfp4_tiled.is_empty() {
                let m_pad       = 16usize;
                let h_pad64     = (h     + 63) & !63;
                let inter_pad64 = (inter + 63) & !63;
                let max_k       = h_pad64.max(inter_pad64);
                let packed_sz   = m_pad * max_k / 2;
                let scales_sz   = (m_pad / 16) * (max_k / 32); // = max_k/32
                let alloc_u8 = |sz: usize| -> Arc<Tensor<u8>> {
                    Arc::new(
                        api::zeros::<u8>(&[sz])
                            .sync_on(matmul.stream())
                            .expect("mxfp4 batch act alloc"),
                    )
                };
                eprintln!(
                    "[mxfp4] batch act buffers: packed={packed_sz}B scales={scales_sz}B \
                     (m_pad={m_pad} max_k={max_k})"
                );
                (Some(alloc_u8(packed_sz)), Some(alloc_u8(scales_sz)))
            } else {
                (None, None)
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
        // position index -- zero H2D uploads per token (was 2 sync H2D per RoPE
        // call, 4 calls per layer x 28 layers = 112 sync_on per token).
        cutile_rope.precompute_table(
            config.max_seq_len.min(4096),
            head_dim,
            config.rope_theta,
            matmul.stream(),
        )?;

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh -- prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_lenxn_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        // Use the shared matmul stream so RoPE scratch buffers are on the same stream
        // as all other GPU operations -- eliminates cross-stream synchronization overhead.
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
        // Attention output scratch: q_dim = n_heads x head_dim (seq_len=1, no padding).
        let attn_out_scratch = add_alloc(q_dim);
        // SiLU-mul output scratch: inter rounded up to next multiple of 32 (the B
        // constant used inside silu_mul_device_scratch).
        const SILU_B: usize = 32;
        let silu_out_scratch = add_alloc(inter.div_ceil(SILU_B) * SILU_B);

        // Pre-allocate the zero-padded down-projection input scratch (only needed
        // when inter_pad != inter, e.g. Llama-3.1-8B inter=14336, inter_pad=16384).
        // Positions 0..inter are overwritten by D->D copy each layer; inter..inter_pad stay 0.
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
        // One-row cos/sin buffers for CUDA-graph-stable RoPE (gathered in-graph
        // by d2d_gather_row reading the position from d_positions).
        let rope_half = config.head_dim / 2;
        let d_rope_cos_row = Arc::new(
            api::zeros::<f32>(&[rope_half])
                .sync_on(matmul.stream())
                .expect("d_rope_cos_row alloc failed"),
        );
        let d_rope_sin_row = Arc::new(
            api::zeros::<f32>(&[rope_half])
                .sync_on(matmul.stream())
                .expect("d_rope_sin_row alloc failed"),
        );

        // Extract device before moving `matmul` into the struct.
        let device = matmul.stream().device().clone();
        let stream_kv = device.new_stream()
            .expect("CutileGpuModel: stream_kv creation failed");
        let event_fork = device.new_event()
            .expect("CutileGpuModel: event_fork creation failed");
        let event_kv_done = device.new_event()
            .expect("CutileGpuModel: event_kv_done creation failed");

        // -- Batch decode graph pre-allocations --------------------------------
        // Check whether q, k, o, gate, up projections are Q4K in every layer.
        // v_proj and down_proj may be Q4K or Q6K (Q4_K_M uses Q6K for both);
        // they are handled via sequential per-seq Q6K GEMVs in the inner loop.
        let all_proj_q4k = (0..config.num_layers).all(|l| {
            let p = format!("model.layers.{l}");
            ["self_attn.q_proj", "self_attn.k_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"]
                .iter()
                .all(|w| dw_q4k.contains_key(&format!("{p}.{w}.weight")))
        });
        eprintln!("[batch_graph] all_proj_q4k={all_proj_q4k}");

        // Dynamic max_batch: query free VRAM after weight upload.
        let max_batch_graph = compute_max_batch(config, h_pad, q_dim, kv_dim, inter);

        let (
            max_batch_graph, d_embed_batch, batch_hidden, batch_normed1, batch_normed2,
            batch_gemv_q, batch_gemv_k, batch_gemv_v, batch_gemv_gate, batch_gemv_up,
            batch_gemv_o, batch_gemv_down, batch_q_seqs, batch_k_seqs, batch_v_seqs,
            batch_k_rot, batch_q_rot, batch_attn_out, batch_oproj_in,
            batch_post_attn_add_scratch, batch_ffn_add_scratch, batch_silu_out_scratch,
            batch_down_compact, batch_pre_attn_norm_scratch, batch_post_attn_norm_scratch,
            batch_final_norm_scratch, batch_q8_qs_h, batch_q8_ds_h, batch_q8_qs_i, batch_q8_ds_i,
            batch_silu_row_bufs,
            batch_src_k_ptrs_dev, batch_src_v_ptrs_dev, batch_attn_kv_lens_dev,
            batch_attn_all_k_ptrs, batch_attn_all_v_ptrs,
        ) = Self::alloc_batch_fields(matmul.stream(), max_batch_graph, h_pad, q_pad, q_dim, kv_dim, inter, inter_pad)?;

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            device_weights_q4k: dw_q4k,
            device_weights_q2: dw_q2,
            device_weights_q6k: dw_q6k,
            device_weights_mxfp4: HashMap::new(),
            device_weights_mxfp4_tiled: dw_mxfp4_tiled,
            mxfp4_gemm,
            mxfp4_act_packed: mxfp4_batch_act_packed,
            mxfp4_act_scales: mxfp4_batch_act_scales,
            mxfp4_out_inter: None,
            mxfp4_out_h: None,
            norm_weights: nw,
            matmul,
            q4k_gemv,
            q2_gemv,
            q6k_gemv,
            oxide_q2_gemv,
            oxide_decode_attn,
            attn_splitk_scratch,
            debug_nan_trace,
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
            prefill_scratch: RefCell::new(std::array::from_fn(|_| None)),
            prefill_mxfp4_packed: RefCell::new(None),
            prefill_mxfp4_scales: RefCell::new(None),
            d_embed,
            d_kv_len,
            d_positions,
            d_kv_write_offset,
            d_rope_cos_row,
            d_rope_sin_row,
            kv_scatter_write,
            final_norm_scratch: RefCell::new(final_norm_scratch),
            cached_decode_graph: RefCell::new(None),
            stream_kv,
            event_fork,
            event_kv_done,
            // Batch graph fields
            max_batch_graph,
            d_embed_batch,
            batch_hidden,
            batch_normed1,
            batch_normed2,
            batch_gemv_q,
            batch_gemv_k,
            batch_gemv_v,
            batch_gemv_gate,
            batch_gemv_up,
            batch_gemv_o,
            batch_gemv_down,
            batch_q_seqs,
            batch_k_seqs,
            batch_v_seqs,
            batch_k_rot,
            batch_q_rot,
            batch_attn_out,
            batch_oproj_in,
            batch_post_attn_add_scratch,
            batch_ffn_add_scratch,
            batch_silu_out_scratch,
            batch_down_compact,
            batch_pre_attn_norm_scratch,
            batch_post_attn_norm_scratch,
            batch_final_norm_scratch,
            batch_q8_qs_h,
            batch_q8_ds_h,
            batch_q8_qs_i,
            batch_q8_ds_i,
            batch_silu_row_bufs,
            batch_src_k_ptrs_dev,
            batch_src_v_ptrs_dev,
            batch_attn_kv_lens_dev,
            batch_attn_all_k_ptrs,
            batch_attn_all_v_ptrs,
            cached_batch_graph: RefCell::new(None),
            cached_mxfp4_batch_graph: RefCell::new(None),
            all_proj_q4k,
        };
        eprintln!("[cutile] warming up kernel specializations...");
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
            Ok(oda) => { eprintln!("[cutile] oxide decode-attn loaded (batch={}, splitk={})", oda.has_batch(), oda.has_splitk()); Some(oda) }
            Err(e)  => { eprintln!("[cutile] oxide decode-attn unavailable: {e}"); None }
        };
        // Split-K decode attention scratch: one stable-address buffer for the
        // CUDA graph's lifetime (pointer baked in at capture, never reallocated).
        let attn_splitk_scratch = oxide_decode_attn.as_ref()
            .filter(|oda| oda.has_splitk())
            .map(|_| {
                Arc::new(
                    api::zeros::<f32>(&[OxideDecodeAttn::splitk_scratch_len(n_heads, head_dim)])
                        .sync_on(matmul.stream())
                        .expect("splitk attn scratch: device allocation failed"),
                )
            });
        let debug_nan_trace = std::env::var("TEMPURA_DEBUG_NAN").is_ok();
        let kv_scatter_write = KvScatterWrite::new()
            .expect("KvScatterWrite: failed to load scatter-write PTX kernel");

        // Pre-compute the full RoPE frequency table on the GPU once.
        cutile_rope.precompute_table(
            config.max_seq_len.min(4096),
            head_dim,
            config.rope_theta,
            matmul.stream(),
        )?;

        // Scratch for exactly one decode step (seq_len=1). Prefill falls back to
        // forward_device which allocates fresh -- prefill is infrequent so the
        // allocation cost there is amortised. Sizing at n_heads (not max_seq_lenxn_heads)
        // avoids the out-of-bounds launch that occurs when cap_num_sh > actual num_sh.
        // Use the shared matmul stream so RoPE scratch buffers are on the same stream
        // as all other GPU operations -- eliminates cross-stream synchronization overhead.
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
        // One-row cos/sin buffers for CUDA-graph-stable RoPE (gathered in-graph
        // by d2d_gather_row reading the position from d_positions).
        let rope_half = config.head_dim / 2;
        let d_rope_cos_row = Arc::new(
            api::zeros::<f32>(&[rope_half])
                .sync_on(matmul.stream())
                .expect("d_rope_cos_row alloc failed"),
        );
        let d_rope_sin_row = Arc::new(
            api::zeros::<f32>(&[rope_half])
                .sync_on(matmul.stream())
                .expect("d_rope_sin_row alloc failed"),
        );

        // Extract device before moving `matmul` into the struct.
        let device = matmul.stream().device().clone();
        let stream_kv = device.new_stream()
            .expect("CutileGpuModel: stream_kv creation failed");
        let event_fork = device.new_event()
            .expect("CutileGpuModel: event_fork creation failed");
        let event_kv_done = device.new_event()
            .expect("CutileGpuModel: event_kv_done creation failed");

        // Dynamic max_batch: query free VRAM after weight upload.
        let max_batch_graph = compute_max_batch(config, h_pad, q_dim, kv_dim, inter);

        let (
            max_batch_graph, d_embed_batch, batch_hidden, batch_normed1, batch_normed2,
            batch_gemv_q, batch_gemv_k, batch_gemv_v, batch_gemv_gate, batch_gemv_up,
            batch_gemv_o, batch_gemv_down, batch_q_seqs, batch_k_seqs, batch_v_seqs,
            batch_k_rot, batch_q_rot, batch_attn_out, batch_oproj_in,
            batch_post_attn_add_scratch, batch_ffn_add_scratch, batch_silu_out_scratch,
            batch_down_compact, batch_pre_attn_norm_scratch, batch_post_attn_norm_scratch,
            batch_final_norm_scratch, batch_q8_qs_h, batch_q8_ds_h, batch_q8_qs_i, batch_q8_ds_i,
            batch_silu_row_bufs,
            batch_src_k_ptrs_dev, batch_src_v_ptrs_dev, batch_attn_kv_lens_dev,
            batch_attn_all_k_ptrs, batch_attn_all_v_ptrs,
        ) = Self::alloc_batch_fields(matmul.stream(), max_batch_graph, h_pad, q_pad, q_dim, kv_dim, inter, inter_pad)?;

        let model = Self {
            config: config.clone(),
            h_pad,
            q_pad,
            inter_pad,
            device_weights: dw,
            device_weights_q4k: HashMap::new(),
            device_weights_q2: HashMap::new(),
            device_weights_q6k: HashMap::new(),
            device_weights_mxfp4: HashMap::new(),
            device_weights_mxfp4_tiled: HashMap::new(),
            mxfp4_gemm: None,
            mxfp4_act_packed: None,
            mxfp4_act_scales: None,
            mxfp4_out_inter: None,
            mxfp4_out_h: None,
            norm_weights: nw,
            matmul,
            q4k_gemv,
            q2_gemv,
            q6k_gemv,
            oxide_q2_gemv,
            oxide_decode_attn,
            attn_splitk_scratch,
            debug_nan_trace,
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
            prefill_scratch: RefCell::new(std::array::from_fn(|_| None)),
            prefill_mxfp4_packed: RefCell::new(None),
            prefill_mxfp4_scales: RefCell::new(None),
            d_embed,
            d_kv_len,
            d_positions,
            d_kv_write_offset,
            d_rope_cos_row,
            d_rope_sin_row,
            kv_scatter_write,
            final_norm_scratch: RefCell::new(final_norm_scratch),
            cached_decode_graph: RefCell::new(None),
            stream_kv,
            event_fork,
            event_kv_done,
            // Batch graph fields
            max_batch_graph,
            d_embed_batch,
            batch_hidden,
            batch_normed1,
            batch_normed2,
            batch_gemv_q,
            batch_gemv_k,
            batch_gemv_v,
            batch_gemv_gate,
            batch_gemv_up,
            batch_gemv_o,
            batch_gemv_down,
            batch_q_seqs,
            batch_k_seqs,
            batch_v_seqs,
            batch_k_rot,
            batch_q_rot,
            batch_attn_out,
            batch_oproj_in,
            batch_post_attn_add_scratch,
            batch_ffn_add_scratch,
            batch_silu_out_scratch,
            batch_down_compact,
            batch_pre_attn_norm_scratch,
            batch_post_attn_norm_scratch,
            batch_final_norm_scratch,
            batch_q8_qs_h,
            batch_q8_ds_h,
            batch_q8_qs_i,
            batch_q8_ds_i,
            batch_silu_row_bufs,
            batch_src_k_ptrs_dev,
            batch_src_v_ptrs_dev,
            batch_attn_kv_lens_dev,
            batch_attn_all_k_ptrs,
            batch_attn_all_v_ptrs,
            cached_batch_graph: RefCell::new(None),
            cached_mxfp4_batch_graph: RefCell::new(None),
            all_proj_q4k: false, // from_f32 uses f32 weights, not Q4K
        };
        eprintln!("[cutile] warming up kernel specializations...");
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

    /// Returns true if MXFP4 tiled batch GEMM is available (sm_120a hardware + weights loaded).
    pub fn has_mxfp4(&self) -> bool {
        self.mxfp4_gemm.is_some() && !self.device_weights_mxfp4_tiled.is_empty()
    }

    /// Number of MXFP4 tiled weight tensors currently loaded.
    pub fn mxfp4_weight_count(&self) -> usize {
        self.device_weights_mxfp4_tiled.len()
    }

    /// Number of MXFP4 tiled weight tensors (alias for bench reporting).
    pub fn mxfp4_tiled_count(&self) -> usize {
        self.device_weights_mxfp4_tiled.len()
    }

    /// True if the batch decode CUDA graph will use MXFP4 tensor-core dispatch.
    pub fn mxfp4_active(&self) -> bool {
        self.mxfp4_gemm.is_some()
            && self.mxfp4_act_packed.is_some()
            && !self.device_weights_mxfp4_tiled.is_empty()
    }

    /// Returns true if ALL Q4K projection layers have MXFP4 tiled weights loaded,
    /// making the MXFP4 batch graph eligible. This check is stable across all replays:
    /// device_weights_mxfp4_tiled is write-once at model load and never mutated during
    /// forward passes (proven by Kani: mxfp4_graph_eligibility_is_deterministic).
    fn mxfp4_eligible_for_batch_graph(&self) -> bool {
        if !self.mxfp4_active() {
            return false;
        }
        let cfg = &self.config;
        (0..cfg.num_layers).all(|l| {
            let p = format!("model.layers.{l}");
            let qw   = format!("{p}.self_attn.q_proj.weight");
            let kw   = format!("{p}.self_attn.k_proj.weight");
            let ow   = format!("{p}.self_attn.o_proj.weight");
            let gatew = format!("{p}.mlp.gate_proj.weight");
            let upw  = format!("{p}.mlp.up_proj.weight");
            // V and down projections stay as Q4K/Q6K (Q4_K_M format).
            [qw.as_str(), kw.as_str(), ow.as_str(), gatew.as_str(), upw.as_str()]
                .iter()
                .all(|n| self.device_weights_mxfp4_tiled.contains_key(*n))
        })
    }

    /// Maximum batch size supported by the CUDA graph decode path.
    pub fn max_batch_graph(&self) -> usize {
        self.max_batch_graph
    }

    /// Run all 7 MXFP4 GEMMs for one transformer layer and return elapsed seconds.
    ///
    /// Uses a pre-quantized random MXFP4 activation [m, h] as input.
    /// The output is discarded. Useful for benchmarking raw MXFP4 GEMM throughput
    /// without the overhead of activation quantization.
    ///
    /// Returns `None` when MXFP4 is not available or the model is not Q4K.
    pub fn bench_mxfp4_one_layer(
        &self,
        layer: usize,
        m: usize,
    ) -> Option<std::time::Duration> {
        use tempura_kernels::mxfp4::quantize_activation_to_mxfp4_tiled;

        let mxfp4 = self.mxfp4_gemm.as_ref()?;
        if self.device_weights_mxfp4_tiled.is_empty() {
            return None;
        }

        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let p = format!("model.layers.{layer}");
        let stream = self.matmul.stream();

        // Pad m to nearest multiple of 16 (m16-tiles requirement).
        let m_pad = (m + 15) & !15;
        // Pad k to nearest multiple of 64 (k64-tiles requirement).
        let h_pad64 = (h + 63) & !63;
        let inter_pad64 = (inter + 63) & !63;

        // Build a random f32 activation [m_pad, h_pad64] and quantize to MXFP4.
        // All m_pad rows carry live data here (bench input), so m_valid = m_pad.
        let act_h_f32: Vec<f32> = (0..m_pad * h_pad64).map(|i| (i as f32) * 0.001).collect();
        let (act_h_packed, act_h_scales) =
            quantize_activation_to_mxfp4_tiled(&act_h_f32, m_pad, h_pad64, m_pad);

        let act_i_f32: Vec<f32> = (0..m_pad * inter_pad64).map(|i| (i as f32) * 0.001).collect();
        let (act_i_packed, act_i_scales) =
            quantize_activation_to_mxfp4_tiled(&act_i_f32, m_pad, inter_pad64, m_pad);

        // Upload activation MXFP4 tensors to GPU.
        let a_h_packed: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&Arc::new(act_h_packed))
            .sync_on(stream).ok()?.into();
        let a_h_scales: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&Arc::new(act_h_scales))
            .sync_on(stream).ok()?.into();
        let a_i_packed: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&Arc::new(act_i_packed))
            .sync_on(stream).ok()?.into();
        let a_i_scales: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&Arc::new(act_i_scales))
            .sync_on(stream).ok()?.into();

        let mk_out_f32 = |rows: usize, cols: usize| -> Arc<Tensor<f32>> {
            Arc::new(
                api::zeros::<f32>(&[rows * cols])
                    .sync_on(stream)
                    .expect("bench_mxfp4_one_layer: output alloc"),
            )
        };

        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        // Pre-allocate output tensors.
        let out_q    = mk_out_f32(m_pad, q_dim);
        let out_k    = mk_out_f32(m_pad, kv_dim);
        let out_v    = mk_out_f32(m_pad, kv_dim);
        let out_o    = mk_out_f32(m_pad, h);
        let out_gate = mk_out_f32(m_pad, inter);
        let out_up   = mk_out_f32(m_pad, inter);
        let out_down = mk_out_f32(m_pad, h);

        let run_gemm = |a_packed: &Arc<Tensor<u8>>, a_scales: &Arc<Tensor<u8>>,
                         wname: &str, out: &Arc<Tensor<f32>>| {
            let dw = self.device_weights_mxfp4_tiled.get(wname)?;
            let k64_tiles = (dw.k / 64) as u32;
            mxfp4.forward(
                a_packed, a_scales,
                &dw.packed, &dw.tiled_scales,
                out,
                m_pad as u32, dw.n as u32, k64_tiles,
            ).ok()?;
            Some(())
        };

        // Warmup pass (one layer).
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.q_proj.weight"), &out_q);
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.k_proj.weight"), &out_k);
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.v_proj.weight"), &out_v);
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.o_proj.weight"), &out_o);
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.mlp.gate_proj.weight"), &out_gate);
        run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.mlp.up_proj.weight"), &out_up);
        run_gemm(&a_i_packed, &a_i_scales, &format!("{p}.mlp.down_proj.weight"), &out_down);
        unsafe { stream.synchronize().ok()? };

        // Timed pass.
        let t0 = std::time::Instant::now();
        const ITERS: usize = 50;
        for _ in 0..ITERS {
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.q_proj.weight"), &out_q);
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.k_proj.weight"), &out_k);
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.v_proj.weight"), &out_v);
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.self_attn.o_proj.weight"), &out_o);
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.mlp.gate_proj.weight"), &out_gate);
            run_gemm(&a_h_packed, &a_h_scales, &format!("{p}.mlp.up_proj.weight"), &out_up);
            run_gemm(&a_i_packed, &a_i_scales, &format!("{p}.mlp.down_proj.weight"), &out_down);
        }
        unsafe { stream.synchronize().ok()? };
        Some(t0.elapsed() / ITERS as u32)
    }

    // -- device tensor helpers -------------------------------------------------

    /// Upload a host slice as a device tensor shaped `[rows, cols]`.
    fn to_device(&self, data: &[f32], rows: usize, cols: usize) -> Arc<Tensor<f32>> {
        let stream = self.matmul.stream();
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .sync_on(stream)
            .expect("CutileGpuModel: host->device upload failed")
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
            .expect("CutileGpuModel: device->host download failed")
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
        // SAFETY: kernel queued async; GPU stream ordering ensures a->b->add->next.
        // Caller (or downstream D->H) provides the CPU sync before any host read.
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
        // to subsequent kernels.  Caller (or downstream D->H) syncs before CPU read.
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

    /// Allocate all pre-allocated device tensors required by the batch decode CUDA graph.
    ///
    /// Called once during model construction; the returned values are stored as struct fields.
    /// `mb` is the dynamic max-batch size computed from available GPU VRAM.
    #[allow(clippy::type_complexity)]
    fn alloc_batch_fields(
        stream:   &Arc<cuda_core::Stream>,
        mb:       usize,
        h_pad:    usize,
        q_pad:    usize,
        q_dim:    usize,
        kv_dim:   usize,
        inter:    usize,
        inter_pad: usize,
    ) -> Result<(
        usize,                             // max_batch_graph
        Vec<Arc<Tensor<f32>>>,             // d_embed_batch
        Arc<Tensor<f32>>,                  // batch_hidden
        Arc<Tensor<f32>>,                  // batch_normed1
        Arc<Tensor<f32>>,                  // batch_normed2
        Arc<Tensor<f32>>,                  // batch_gemv_q
        Arc<Tensor<f32>>,                  // batch_gemv_k
        Arc<Tensor<f32>>,                  // batch_gemv_v
        Arc<Tensor<f32>>,                  // batch_gemv_gate
        Arc<Tensor<f32>>,                  // batch_gemv_up
        Arc<Tensor<f32>>,                  // batch_gemv_o
        Arc<Tensor<f32>>,                  // batch_gemv_down
        Vec<Arc<Tensor<f32>>>,             // batch_q_seqs
        Vec<Arc<Tensor<f32>>>,             // batch_k_seqs
        Vec<Arc<Tensor<f32>>>,             // batch_v_seqs
        Vec<Arc<Tensor<f32>>>,             // batch_k_rot
        Vec<Arc<Tensor<f32>>>,             // batch_q_rot
        Vec<Arc<Tensor<f32>>>,             // batch_attn_out
        Arc<Tensor<f32>>,                  // batch_oproj_in
        RefCell<Option<Arc<Tensor<f32>>>>, // batch_post_attn_add_scratch
        RefCell<Option<Arc<Tensor<f32>>>>, // batch_ffn_add_scratch
        RefCell<Option<Arc<Tensor<f32>>>>, // batch_silu_out_scratch
        Option<Arc<Tensor<f32>>>,          // batch_down_compact
        RefCell<RmsnormScratch>,           // batch_pre_attn_norm_scratch
        RefCell<RmsnormScratch>,           // batch_post_attn_norm_scratch
        RefCell<RmsnormScratch>,           // batch_final_norm_scratch
        Arc<Tensor<u32>>,                  // batch_q8_qs_h
        Arc<Tensor<u32>>,                  // batch_q8_ds_h
        Arc<Tensor<u32>>,                  // batch_q8_qs_i
        Arc<Tensor<u32>>,                  // batch_q8_ds_i
        Vec<Arc<Tensor<f32>>>,             // batch_silu_row_bufs
        cuda_core::sys::CUdeviceptr,       // batch_src_k_ptrs_dev
        cuda_core::sys::CUdeviceptr,       // batch_src_v_ptrs_dev
        cuda_core::sys::CUdeviceptr,       // batch_attn_kv_lens_dev
        cuda_core::sys::CUdeviceptr,       // batch_attn_all_k_ptrs
        cuda_core::sys::CUdeviceptr,       // batch_attn_all_v_ptrs
    ), tempura_kernels::KernelError> {
        // mb is passed in from compute_max_batch (dynamic VRAM-based sizing).
        let af32 = |n: usize| -> Result<Arc<Tensor<f32>>, tempura_kernels::KernelError> {
            Ok(Arc::new(
                api::zeros::<f32>(&[n])
                    .sync_on(stream)
                    .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?,
            ))
        };
        let au32 = |n: usize| -> Result<Arc<Tensor<u32>>, tempura_kernels::KernelError> {
            Ok(Arc::new(
                api::zeros::<u32>(&[n])
                    .sync_on(stream)
                    .map_err(|e| tempura_kernels::KernelError::InvalidConfig(e.to_string()))?,
            ))
        };
        let vec_af32 = |n: usize| -> Result<Vec<Arc<Tensor<f32>>>, tempura_kernels::KernelError> {
            (0..mb).map(|_| af32(n)).collect()
        };

        let d_embed_batch  = vec_af32(h_pad)?;
        let batch_hidden   = af32(mb * h_pad)?;
        let batch_normed1  = af32(mb * h_pad)?;
        let batch_normed2  = af32(mb * h_pad)?;
        let batch_gemv_q    = af32(mb * q_dim)?;
        let batch_gemv_k    = af32(mb * kv_dim)?;
        let batch_gemv_v    = af32(mb * kv_dim)?;
        let batch_gemv_gate = af32(mb * inter)?;
        let batch_gemv_up   = af32(mb * inter)?;
        let batch_gemv_o    = af32(mb * h_pad)?;
        let batch_gemv_down = af32(mb * h_pad)?;
        let batch_q_seqs  = vec_af32(q_dim)?;
        let batch_k_seqs  = vec_af32(kv_dim)?;
        let batch_v_seqs  = vec_af32(kv_dim)?;
        let batch_k_rot   = vec_af32(kv_dim)?;
        let batch_q_rot   = vec_af32(q_pad)?;
        let batch_attn_out = vec_af32(q_dim)?;
        let batch_oproj_in = af32(mb * q_pad)?;

        const SILU_B: usize = 32;
        let batch_post_attn_add_scratch = RefCell::new(Some(af32(mb * h_pad)?));
        let batch_ffn_add_scratch       = RefCell::new(Some(af32(mb * h_pad)?));
        let batch_silu_out_scratch      = RefCell::new(Some(af32(mb * inter.div_ceil(SILU_B) * SILU_B)?));
        let batch_down_compact = if inter_pad != inter {
            Some(af32(mb * inter)?)
        } else {
            None
        };

        let batch_pre_attn_norm_scratch  = RefCell::new(
            RmsnormScratch::new(stream, mb, h_pad)?
        );
        let batch_post_attn_norm_scratch = RefCell::new(
            RmsnormScratch::new(stream, mb, h_pad)?
        );
        let batch_final_norm_scratch     = RefCell::new(
            RmsnormScratch::new(stream, mb, h_pad)?
        );

        // Q8_1 scratch for h-width projections (Q, K, V, O, gate, up).
        // k = h_pad (rounded up to h_pad which is a power of 2, divisible by 256).
        let h_k_total      = mb * h_pad;
        let h_q8_blocks    = h_k_total.div_ceil(32);
        let batch_q8_qs_h  = au32(h_k_total / 4)?;
        let batch_q8_ds_h  = au32(h_q8_blocks)?;

        // Q8_1 scratch for inter-width down projection.
        // k = inter (actual K, not padded; Q8_1 quantise sees k_total = mb * inter).
        let i_k_total      = mb * inter;
        let i_q8_blocks    = i_k_total.div_ceil(32);
        let batch_q8_qs_i  = au32(i_k_total / 4)?;
        let batch_q8_ds_i  = au32(i_q8_blocks)?;

        // Raw device pointer arrays [mb x u64] for batch KV scatter-write.
        // Allocated as raw cuMemAlloc'd memory (not a cuTile Tensor since u64 is unsupported).
        let ptr_bytes = (mb * std::mem::size_of::<u64>()) as usize;
        let mut batch_src_k_ptrs_dev: cuda_core::sys::CUdeviceptr = 0;
        let mut batch_src_v_ptrs_dev: cuda_core::sys::CUdeviceptr = 0;
        let rc = unsafe { cuda_core::sys::cuMemAlloc_v2(&mut batch_src_k_ptrs_dev, ptr_bytes) };
        if rc != 0 {
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "alloc_batch_fields: cuMemAlloc batch_src_k_ptrs failed: {rc}"
            )));
        }
        let rc = unsafe { cuda_core::sys::cuMemAlloc_v2(&mut batch_src_v_ptrs_dev, ptr_bytes) };
        if rc != 0 {
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_k_ptrs_dev) };
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "alloc_batch_fields: cuMemAlloc batch_src_v_ptrs failed: {rc}"
            )));
        }

        // Per-sequence kv_len array [mb x i32] for oxide_decode_attn_batch kernel.
        let kv_len_bytes = mb * std::mem::size_of::<i32>();
        let mut batch_attn_kv_lens_dev: cuda_core::sys::CUdeviceptr = 0;
        let rc = unsafe { cuda_core::sys::cuMemAlloc_v2(&mut batch_attn_kv_lens_dev, kv_len_bytes) };
        if rc != 0 {
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_k_ptrs_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_v_ptrs_dev) };
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "alloc_batch_fields: cuMemAlloc batch_attn_kv_lens failed: {rc}"
            )));
        }

        // Per-layer K/V cache pointer arrays for oxide_decode_attn_batch.
        // MAX_ATTN_LAYERS=64 covers any realistic model; actual layers checked at runtime.
        const MAX_ATTN_LAYERS: usize = 64;
        let all_ptrs_bytes = MAX_ATTN_LAYERS * mb * std::mem::size_of::<u64>();
        let mut batch_attn_all_k_ptrs: cuda_core::sys::CUdeviceptr = 0;
        let mut batch_attn_all_v_ptrs: cuda_core::sys::CUdeviceptr = 0;
        let rc = unsafe { cuda_core::sys::cuMemAlloc_v2(&mut batch_attn_all_k_ptrs, all_ptrs_bytes) };
        if rc != 0 {
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_k_ptrs_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_v_ptrs_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_attn_kv_lens_dev) };
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "alloc_batch_fields: cuMemAlloc batch_attn_all_k_ptrs failed: {rc}"
            )));
        }
        let rc = unsafe { cuda_core::sys::cuMemAlloc_v2(&mut batch_attn_all_v_ptrs, all_ptrs_bytes) };
        if rc != 0 {
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_k_ptrs_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_src_v_ptrs_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_attn_kv_lens_dev) };
            let _ = unsafe { cuda_core::sys::cuMemFree_v2(batch_attn_all_k_ptrs) };
            return Err(tempura_kernels::KernelError::InvalidConfig(format!(
                "alloc_batch_fields: cuMemAlloc batch_attn_all_v_ptrs failed: {rc}"
            )));
        }

        // Sized inter_pad (not inter) because gemm_q6k_d2d_slot reshapes input to
        // next_pow2(k_blocks * Q6K_QK) = next_pow2(inter), which equals inter_pad.
        let batch_silu_row_bufs: Vec<Arc<Tensor<f32>>> = (0..mb)
            .map(|_| af32(inter_pad))
            .collect::<Result<_, _>>()?;

        Ok((
            mb, d_embed_batch, batch_hidden, batch_normed1, batch_normed2,
            batch_gemv_q, batch_gemv_k, batch_gemv_v, batch_gemv_gate, batch_gemv_up,
            batch_gemv_o, batch_gemv_down, batch_q_seqs, batch_k_seqs, batch_v_seqs,
            batch_k_rot, batch_q_rot, batch_attn_out, batch_oproj_in,
            batch_post_attn_add_scratch, batch_ffn_add_scratch, batch_silu_out_scratch,
            batch_down_compact, batch_pre_attn_norm_scratch, batch_post_attn_norm_scratch,
            batch_final_norm_scratch, batch_q8_qs_h, batch_q8_ds_h, batch_q8_qs_i, batch_q8_ds_i,
            batch_silu_row_bufs,
            batch_src_k_ptrs_dev, batch_src_v_ptrs_dev, batch_attn_kv_lens_dev,
            batch_attn_all_k_ptrs, batch_attn_all_v_ptrs,
        ))
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
                .expect("CutileGpuModel: host->device upload (flat) failed"),
        )
    }

    /// Prepare the attention output on device for the O-projection GEMM.
    ///
    /// The decode attention kernel returns a flat `[q_dim]` tensor.  The O-proj
    /// GEMM expects `[1, q_pad]`.  When `q_pad == q_dim` a zero-copy reshape is
    /// done.  Otherwise a zero-padded `[1, q_pad]` buffer is allocated on the
    /// GPU and the attention output is written into it via D->D copy -- no PCIe.
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
        // Allocate zero-filled padded buffer and D->D copy each row's valid prefix.
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
                    .expect("attn_out_to_oproj_device: D->D copy failed");
            }
        }
        padded
    }

    /// Device-to-device GEMM: `output[m, n] = a[m, k_pad] @ b[n, k_pad]^T`.
    ///
    /// The caller MUST ensure `a` has exactly `dw.k_pad` columns.  No download
    /// or re-padding is performed here -- the activation must already be padded.
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

        // Pre-allocate Q8_1 scratch for the oxide Q4K GEMV -- must happen before
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

        // Pre-allocate kpar output scratch for the Q6K sm120_mbar GEMV.
        // max_n = inter (largest Q6K output, e.g. gate/up_proj for MLP FFN expand).
        if let Some(ref oxide) = self.oxide_q6k_gemv {
            if oxide.has_kpar() {
                oxide.pre_allocate_kpar_output_scratch(inter)
                    .expect("oxide Q6K: pre_allocate_kpar_output_scratch failed");
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
                // k_pad not divisible by 128 -- use strided GEMV (async).
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
    /// which dispatch through `cutile_gemv` -- a split-K kernel that avoids the
    /// ~500x MMA slowdown caused by hardware refusing BM=1 tiles.  Each block
    /// handles N_BLOCK_GEMV=64 output neurons via element-wise multiply +
    /// reduce_sum over [64x64] weight tiles, fully utilising warp-level vector
    /// lanes (~target: ~1 TFLOPS vs. ~1.8 GFLOPS scalar fallback).
    ///
    /// For m > 1 (prefill): uses `forward_bt_d2d_into` / `forward_bt_device_to_device`
    /// with the square BMxBN GEMM as before -- MMA is efficient for M >= 16.
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
            // Decode: use coalesced GEMV kernel -- one program per output element,
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
            // k_pad not divisible by 128 -- fall back to strided GEMV.
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
        // kernel expects rank-1 x. Reshape [1, h_pad] -> [h_pad] as a zero-copy
        // metadata operation (no D->H->D round-trip). The kernel reads exactly
        // k_blocks * Q2_0_QK elements, which is <= h_pad; the padding is ignored.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q2_0_QK)])
            .expect("gemm_q2_d2d: reshape [1, stride] -> [stride] failed");
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
        // Zero-copy reshape [1, h_pad] -> [h_pad]; kernel reads only k_blocks * Q6K_QK.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::cutile_quantized_gemm::Q6K_QK)])
            .expect("gemm_q6k_d2d: reshape [1, stride] -> [stride] failed");
        // Prefer kpar (v3, 8 warps/block -> 100% SM occupancy) when available,
        // else fall back to v2, then cuTile.
        if let Some(oxide) = &self.oxide_q6k_gemv {
            return oxide
                .forward_kpar_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                .expect("OxideQ6kGemv::forward_kpar_d2d failed");
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
        // Zero-copy reshape [1, h_pad] -> [h_pad]; kernel reads only k_blocks * Q4K_BLOCK_SIZE.
        let a_flat = Arc::clone(a)
            .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::Q4K_BLOCK_SIZE)])
            .expect("gemm_q4k_d2d: reshape [1, stride] -> [stride] failed");
        // Prefer oxide with Q8_1 activation quantization (best perf), then v2, then cuTile.
        // For large projections (n >= 4096: Q proj, O proj), use the best kpar variant:
        //   - v4 (sm_120 PTX, native GGUF 36-u32): Blackwell (RTX 5090), 14% less DRAM
        //   - v3_kpar (sm_80 oxide PTX, repacked 42-u32): fallback for Ampere/Ada
        // When native format (uses_raw_gguf=true) is active, ALL Q4K tensors must use
        // kpar regardless of n size (the old q8/f32 paths use repacked 42-u32 format).
        if let Some(oxide) = &self.oxide_q4k_gemv {
            let min_kpar_n = if oxide.uses_raw_gguf { 0 } else { 4096 };
            if oxide.has_any_kpar() && dw.n >= min_kpar_n {
                return oxide
                    .forward_best_kpar_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("OxideQ4kGemv::forward_best_kpar_d2d failed");
            }
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

        // Prefer kpar (sm120_mbar or dp4a_kpar8) for best throughput.
        // kpar allocates its own output, so it must be called outside graph capture.
        // For CUDA-graph-captured paths, the slot-hit below uses submit_kpar_d2d_into_on.
        // NOTE: this check MUST run before the pool take below -- taking the slot
        // and returning here leaked it (Q6K down_projs drained pool[SLOT_DOWN],
        // which the Q4K kpar path of a mixed-quant model then found empty).
        if let Some(oxide) = &self.oxide_q6k_gemv {
            if oxide.has_kpar() {
                return oxide
                    .forward_kpar_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("OxideQ6kGemv::forward_kpar_d2d failed in gemm_q6k_d2d_slot");
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

        // Use oxide baseline if available (no kpar).
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

        // Q8 kpar path: quantize + kpar GEMV into the PER-SLOT output tensor.
        // The previous shared-scratch output (forward_best_kpar_d2d) made
        // consecutive GEMV outputs ALIAS: gate and up both wrote q8_out_scratch,
        // so the silu read silu(up)*up with gate's values clobbered (found via
        // bench-batch fresh-cache replay + per-op bits_sum diffing).
        //
        // The pool Arc is borrowed, never taken: the slot's device memory is
        // reused every call (stream-ordered writes make this safe) and callers
        // get clones. This cannot miss, so the fresh-allocation fallback below
        // is unreachable during CUDA graph capture (a sync alloc there is a
        // hard error).
        if let Some(oxide) = &self.oxide_q4k_gemv {
            let min_kpar_n = if oxide.uses_raw_gguf { 0 } else { 4096 };
            if oxide.has_any_kpar() && dw.n >= min_kpar_n {
                let slot_arc = self.scratch.borrow()[slot].clone();
                if let Some(slot_arc) = slot_arc {
                    let submitted = unsafe {
                        if oxide.uses_raw_gguf {
                            oxide.submit_sm120_kpar_into_slot(
                                &a_flat, &dw.tensor, &slot_arc, dw.n, dw.k_blocks,
                            )
                        } else {
                            oxide.submit_dp4a_kpar_into_slot(
                                &a_flat, &dw.tensor, &slot_arc, dw.n, dw.k_blocks,
                            )
                        }
                    };
                    match submitted {
                        Ok(result) => return result,
                        Err(e) => panic!(
                            "gemm_q4k_d2d_slot: kpar submit failed for {weight_name} \
                             (n={}, k_blocks={}): {e}",
                            dw.n, dw.k_blocks
                        ),
                    }
                } else {
                    // Pool slot not pre-allocated. This should be impossible
                    // after preallocate_scratch; include the pool state in the
                    // panic to diagnose any recurrence.
                    let pool_state: Vec<bool> =
                        self.scratch.borrow().iter().map(|s| s.is_some()).collect();
                    panic!(
                        "gemm_q4k_d2d_slot: pool slot {slot} empty for weight \
                         {weight_name} (n={}, k_blocks={}); pool occupied: {pool_state:?}",
                        dw.n, dw.k_blocks
                    );
                }
            }
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

    /// Dispatch Q, K, V projections with multi-stream concurrency for decode (seq_len=1).
    ///
    /// When oxide GEMVs are available for the weight format and seq_len==1, K and V
    /// are dispatched on `stream_kv` concurrently with Q on `stream_main`, connected by
    /// CUDA events (fork after normed_dev, join before RoPE).  This lets K [kv_dim x h]
    /// and V [kv_dim x h] hide behind Q [q_dim x h] on the GPU.
    ///
    /// K and V use the oxide f32 GEMV (not the Q8 dp4a path) so each writes into its own
    /// pre-allocated slot (SLOT_K / SLOT_V) without touching the Q8 shared scratch buffer
    /// that Q may be using on stream_main simultaneously.
    ///
    /// Falls back to the sequential path when oxide GEMVs are unavailable or seq_len > 1.
    fn dispatch_qkv_multistream(
        &self,
        normed: &Arc<Tensor<f32>>,
        q_name: &str,
        k_name: &str,
        v_name: &str,
        seq_len: usize,
    ) -> (Arc<Tensor<f32>>, Arc<Tensor<f32>>, Arc<Tensor<f32>>) {
        // Multi-stream path: oxide f32 GEMV available for Q4K, Q6K, or Q2 K/V weights.
        // TEMPURA_NO_MULTISTREAM=1 bisects the stream_kv fork/join off (decode
        // numerics debugging -- see bench-quality / bench-decode-ops).
        if seq_len == 1
            && std::env::var("TEMPURA_NO_MULTISTREAM").as_deref() != Ok("1")
        {
            let has_q4k_kv = self.oxide_q4k_gemv.is_some()
                && (self.device_weights_q4k.contains_key(k_name)
                    || self.device_weights_q6k.contains_key(k_name)
                    || self.device_weights_q2.contains_key(k_name));

            if has_q4k_kv {
                return self.dispatch_qkv_oxide_multistream(normed, q_name, k_name, v_name);
            }
        }

        // Sequential fallback (prefill, cuTile-only, or f32 weights).
        let q_dev = self.gemm_any_d2d_slot(normed, q_name, seq_len, SLOT_Q);
        let k_dev = self.gemm_any_d2d_slot(normed, k_name, seq_len, SLOT_K);
        let v_dev = self.gemm_any_d2d_slot(normed, v_name, seq_len, SLOT_V);
        (q_dev, k_dev, v_dev)
    }

    /// Inner multi-stream QKV dispatch: fork stream_kv for K/V, Q on stream_main.
    ///
    /// Called only when oxide f32 GEMVs are confirmed available for K/V weights.
    fn dispatch_qkv_oxide_multistream(
        &self,
        normed: &Arc<Tensor<f32>>,
        q_name: &str,
        k_name: &str,
        v_name: &str,
    ) -> (Arc<Tensor<f32>>, Arc<Tensor<f32>>, Arc<Tensor<f32>>) {
        // Fork: normed_dev is ready on stream_main. Record a fork event and have
        // stream_kv wait on it so the GPU graph includes a dependency edge.
        self.event_fork
            .record(&self.matmul.stream())
            .expect("event_fork record failed");
        self.stream_kv
            .wait_event(&self.event_fork)
            .expect("stream_kv wait_event(fork) failed");

        // -- K on stream_kv -----------------------------------------------------
        let cu_kv = self.stream_kv.cu_stream();
        let k_dev = self.dispatch_kv_on_stream(normed, k_name, SLOT_K, cu_kv);
        let v_dev = self.dispatch_kv_on_stream(normed, v_name, SLOT_V, cu_kv);

        // -- Q on stream_main (normal path, may use Q8 dp4a) --------------------
        let q_dev = self.gemm_any_d2d_slot(normed, q_name, 1, SLOT_Q);

        // Join: stream_main waits for K and V to complete before RoPE.
        self.event_kv_done
            .record(&self.stream_kv)
            .expect("event_kv_done record failed");
        self.matmul
            .stream()
            .wait_event(&self.event_kv_done)
            .expect("stream_main wait_event(kv_done) failed");

        (q_dev, k_dev, v_dev)
    }

    /// Dispatch a single K or V GEMV on `cu_stream` using the oxide f32 kernel.
    ///
    /// Takes the pre-allocated slot tensor, submits the GEMV into it, puts it
    /// back in the pool, and returns a clone of the Arc -- identical lifecycle to
    /// `gemm_q4k_d2d_slot` but on an explicit stream instead of `self.stream`.
    ///
    /// # Safety
    /// `cu_stream` must be a valid CUstream that outlives this call (it is captured
    /// into the CUDA graph, so it lives as long as the graph does).
    fn dispatch_kv_on_stream(
        &self,
        normed: &Arc<Tensor<f32>>,
        weight_name: &str,
        slot: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Arc<Tensor<f32>> {
        // Q4K path.
        if let Some(dw) = self.device_weights_q4k.get(weight_name) {
            if let Some(oxide) = &self.oxide_q4k_gemv {
                let a_flat = Arc::clone(normed)
                    .reshape(&[next_pow2(dw.k_blocks * tempura_kernels::Q4K_BLOCK_SIZE)])
                    .expect("dispatch_kv_on_stream: Q4K reshape failed");
                let maybe_tensor = {
                    let mut pool = self.scratch.borrow_mut();
                    pool[slot].take().and_then(|arc| Arc::try_unwrap(arc).ok())
                };
                if let Some(tensor) = maybe_tensor {
                    // Native format: use sm120_kpar with per-KV-stream Q8_1 scratch.
                    // This avoids the data race with Q's Q8 scratch on stream_main.
                    if oxide.uses_raw_gguf && oxide.has_sm120_kpar() {
                        if let Ok(result) = unsafe {
                            oxide.submit_sm120_kpar_kv_into_slot(
                                &a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks, cu_stream,
                            )
                        } {
                            self.scratch.borrow_mut()[slot] = Some(result.clone());
                            return result;
                        }
                        // sm120_kpar failed -- fall through to pool-miss path below.
                    } else {
                        // Repacked format: use the basic f32 GEMV (reads 42-u32).
                        if let Ok(result) = unsafe {
                            oxide.submit_d2d_into_on(
                                &a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks, cu_stream,
                            )
                        } {
                            self.scratch.borrow_mut()[slot] = Some(result.clone());
                            return result;
                        }
                    }
                }
                // Pool miss or submit failed: synchronous fallback (not during graph capture).
                let result = oxide
                    .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("dispatch_kv_on_stream: Q4K forward_d2d failed");
                return result;
            }
        }

        // Q6K path.
        if let Some(dw) = self.device_weights_q6k.get(weight_name) {
            if let Some(oxide) = &self.oxide_q6k_gemv {
                let a_flat = Arc::clone(normed)
                    .reshape(&[next_pow2(dw.k_blocks * Q6K_QK)])
                    .expect("dispatch_kv_on_stream: Q6K reshape failed");
                let maybe_tensor = {
                    let mut pool = self.scratch.borrow_mut();
                    pool[slot].take().and_then(|arc| Arc::try_unwrap(arc).ok())
                };
                if let Some(tensor) = maybe_tensor {
                    if let Ok(result) = unsafe {
                        oxide.submit_d2d_into_on(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks, cu_stream)
                    } {
                        self.scratch.borrow_mut()[slot] = Some(result.clone());
                        return result;
                    }
                }
                return oxide
                    .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("dispatch_kv_on_stream: Q6K forward_d2d failed");
            }
        }

        // Q2 path.
        if let Some(dw) = self.device_weights_q2.get(weight_name) {
            if let Some(oxide) = &self.oxide_q2_gemv {
                let a_flat = Arc::clone(normed)
                    .reshape(&[next_pow2(dw.k_blocks * Q2_0_QK)])
                    .expect("dispatch_kv_on_stream: Q2 reshape failed");
                let maybe_tensor = {
                    let mut pool = self.scratch.borrow_mut();
                    pool[slot].take().and_then(|arc| Arc::try_unwrap(arc).ok())
                };
                if let Some(tensor) = maybe_tensor {
                    if let Ok(result) = unsafe {
                        oxide.submit_d2d_into_on(&a_flat, &dw.tensor, tensor, dw.n, dw.k_blocks, cu_stream)
                    } {
                        self.scratch.borrow_mut()[slot] = Some(result.clone());
                        return result;
                    }
                }
                return oxide
                    .forward_d2d(&a_flat, &dw.tensor, dw.n, dw.k_blocks)
                    .expect("dispatch_kv_on_stream: Q2 forward_d2d failed");
            }
        }

        // Fallback: f32 or cuTile-only -- sequential on stream_main.
        self.gemm_any_d2d_slot(normed, weight_name, 1, slot)
    }

    /// Unified GEMV/GEMM that transparently selects Q4_K, Q6_K, Q2_0 (decode only), or f32.
    ///
    /// For m == 1: checks `device_weights_q4k` first, then `device_weights_q6k`,
    /// then `device_weights_q2`, then falls through to `gemm_d2d_slot` for f32 weights.
    /// Batched GEMM for M > 1 using M sequential GPU GEMVs -- keeps quantized weights
    /// GPU-resident instead of downloading to CPU. Each row of `a` is fed to the
    /// M=1 GEMV path; results are D->D-concatenated into an [m, n] output.
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

        // For f32 weights there is no PCIe bottleneck -- use the existing slot path.
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

        // sm_120 batch Q4K kernel: hand-written PTX with static smem, avoids error 719.
        // Only available when uses_raw_gguf=true (weights in native 36-u32 GGUF format).
        if is_q4k {
            if let Some(oxide) = &self.oxide_q4k_gemv {
                if oxide.has_sm120_batch() && oxide.uses_raw_gguf {
                    let dw = self.device_weights_q4k.get(weight_name).unwrap();
                    // `a` is flat [m * row_stride]; the batch kernel reads it row-major.
                    // k = k_blocks * 256; row_stride may be padded -- pass just k elements
                    // by building a trimmed [m * k] view if needed.
                    let k = dw.k_blocks * 256;
                    let a_trimmed: Arc<Tensor<f32>> = if row_stride == k {
                        a.clone()
                    } else {
                        // row_stride != k: copy each row's k elements into a compact [m*k] tensor.
                        let compact = Arc::new(
                            api::zeros::<f32>(&[m * k])
                                .sync_on(stream)
                                .expect("gemm_any_d2d_batch sm120: compact alloc"),
                        );
                        for r in 0..m {
                            unsafe {
                                cuda_core::memcpy_dtod_async::<f32>(
                                    compact.device_pointer().cu_deviceptr() + (r * k * 4) as u64,
                                    a.device_pointer().cu_deviceptr() + (r * row_stride * 4) as u64,
                                    k,
                                    stream,
                                ).expect("gemm_any_d2d_batch sm120: row copy");
                            }
                        }
                        compact
                    };
                    return oxide
                        .forward_sm120_batch_d2d(&a_trimmed, &dw.tensor, m, dw.n, dw.k_blocks)
                        .expect("sm120_batch Q4K GEMV failed");
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
                ).expect("gemm_any_d2d_batch: a_row D->D slice");
            }
            let result = self.gemm_any_d2d_slot(&a_row, weight_name, 1, slot);
            rows.push(result);
        }

        // Concatenate M results [n] -> flat [m*n] (rank-1, compatible with cuTile ops).
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
                ).expect("gemm_any_d2d_batch: concat D->D");
            }
        }
        out
    }

    /// Prefill (m > 1) GEMM via the MXFP4 tensor-core path: on-GPU activation
    /// quantization + `mxf4_batch_gemm_tiled` (stream-K persistent, arbitrary M).
    ///
    /// Engagement guard: `a` must be a compact row-major `[m, k]` activation
    /// (`a.size() == m * dw.k`), e.g. RMSNorm rows when `h_pad == h`. Strided or
    /// padded activations (h_pad > h) are not layout-compatible and fall back.
    ///
    /// The quant kernel never loads rows >= m_valid, so the compact input needs
    /// no staged padding rows; rows `m..m_pad` of the output are zeros and are
    /// trimmed so callers always receive exactly `[m, n]`.
    fn gemm_mxfp4_prefill(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Option<Arc<Tensor<f32>>> {
        let mxfp4 = self.mxfp4_gemm.as_ref()?;
        let dw = self.device_weights_mxfp4_tiled.get(weight_name)?;
        let k = dw.k;
        if k % 64 != 0 || a.size() != m * k {
            return None;
        }
        let n = dw.n;
        let m_pad = m.div_ceil(16) * 16;
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();

        // Grow-on-demand activation buffers. Reused across layers and chunks at
        // whatever [m_pad, k] stride the current call uses (each call's quant +
        // GEMM agree on the stride; no state crosses calls). The alloc's
        // stream sync guarantees prior consumers have finished before the old
        // buffer is dropped.
        let packed_sz = m_pad * k / 2;
        let scales_sz = (m_pad / 16) * (k / 32);
        {
            let mut buf = self.prefill_mxfp4_packed.borrow_mut();
            if buf.as_ref().is_none_or(|t| t.size() < packed_sz) {
                *buf = Some(Arc::new(
                    api::zeros::<u8>(&[packed_sz])
                        .sync_on(stream)
                        .expect("prefill mxfp4 packed buffer alloc"),
                ));
            }
        }
        {
            let mut buf = self.prefill_mxfp4_scales.borrow_mut();
            if buf.as_ref().is_none_or(|t| t.size() < scales_sz) {
                *buf = Some(Arc::new(
                    api::zeros::<u8>(&[scales_sz])
                        .sync_on(stream)
                        .expect("prefill mxfp4 scales buffer alloc"),
                ));
            }
        }
        let packed = self.prefill_mxfp4_packed.borrow().as_ref().unwrap().clone();
        let scales = self.prefill_mxfp4_scales.borrow().as_ref().unwrap().clone();
        mxfp4
            .quantize_activation(a, &packed, &scales, m_pad as u32, k as u32, m as u32, cu_stream)
            .unwrap_or_else(|e| panic!("mxfp4 prefill quant {weight_name}: {e}"));
        if self.debug_nan_trace {
            let ab: Vec<f32> = Arc::clone(a).to_host_vec().sync_on(stream).expect("dbg a");
            let pb: Vec<u8> = Arc::clone(&packed).to_host_vec().sync_on(stream).expect("dbg packed");
            let sb: Vec<u8> = Arc::clone(&scales).to_host_vec().sync_on(stream).expect("dbg scales");
            let asum: u64 = ab.iter().map(|v| v.to_bits() as u64).sum();
            let fresh_p = m_pad * k / 2;
            let fresh_s = (m_pad / 16) * (k / 32);
            let psum: u64 = pb.iter().map(|&b| b as u64).sum();
            let pfr: u64 = pb[..fresh_p.min(pb.len())].iter().map(|&b| b as u64).sum();
            let ssum: u64 = sb.iter().map(|&b| b as u64).sum();
            let sfr: u64 = sb[..fresh_s.min(sb.len())].iter().map(|&b| b as u64).sum();
            eprintln!(
                "[nan-trace] {weight_name} quant: a_sum={asum} packed_fresh={pfr} (len={} fresh={fresh_p}) \
                 scales_fresh={sfr} (len={} fresh={fresh_s} full={ssum}) (m={m} k={k})",
                pb.len(),
                sb.len()
            );
        }

        // Output: [m_pad, n] from the per-slot pool. Every element is written
        // by the GEMM, so a grown-then-reused buffer needs no re-zeroing.
        let out = {
            let mut pool = self.prefill_scratch.borrow_mut();
            if pool[slot].as_ref().is_none_or(|t| t.size() < m_pad * n) {
                pool[slot] = Some(Arc::new(
                    api::zeros::<f32>(&[m_pad * n])
                        .sync_on(stream)
                        .expect("prefill output buffer alloc"),
                ));
            }
            pool[slot].as_ref().unwrap().clone()
        };
        // Single launch at full m_pad. (An earlier revision chunked this at
        // m=16 rows believing the cutlass-mlir backend miscompiled larger
        // grids; the real bug was our own accumulator store mapping writing
        // up to 7 rows past every m16 tile into the cudaMallocAsync pool --
        // silent corruption at small grids, illegal access at large ones.
        // Fixed and validated against a random-data CPU reference at every
        // shape up to (128, 4096, 4096) in tools/mxf4-gemv.)
        mxfp4
            .forward(
                &packed,
                &scales,
                &dw.packed,
                &dw.tiled_scales,
                &out,
                m_pad as u32,
                n as u32,
                (k / 64) as u32,
            )
            .unwrap_or_else(|e| panic!("mxfp4 prefill gemm {weight_name}: {e}"));
        if self.debug_nan_trace {
            let ob: Vec<f32> = Arc::clone(&out).to_host_vec().sync_on(stream).expect("dbg out");
            let osum: u64 = ob[..(m_pad * n).min(ob.len())].iter().map(|v| v.to_bits() as u64).sum();
            // Per-row zero census: localizes which m_pad rows the GEMM left
            // unwritten or zeroed (prefill m < m_pad bisection).
            let zero_rows: Vec<usize> = (0..m_pad)
                .filter(|&r| {
                    ob[r * n..(r + 1) * n]
                        .iter()
                        .all(|v| v.to_bits() == 0)
                })
                .collect();
            let nan_rows = ob
                .chunks(n)
                .filter(|c| c.iter().any(|v| v.is_nan()))
                .count();
            eprintln!(
                "[nan-trace] {weight_name} mxfp4-gemm (16-row chunks): out_fresh={osum} \
                 (len={} m_pad*n={}) zero_rows={}/{} first_zero={:?} nan_rows={nan_rows}",
                ob.len(),
                m_pad * n,
                zero_rows.len(),
                m_pad,
                zero_rows.first(),
            );
        }

        if m_pad == m {
            Some(out)
        } else {
            // Trim [m_pad, n] -> [m, n]: fresh tensor + one D->D row copy so
            // downstream consumers see the exact shape the fallback path
            // returns (they all index rows by the explicit seq_len/m param,
            // but shape-exactness keeps every assert and size check honest).
            let trimmed = Arc::new(
                api::zeros::<f32>(&[m * n])
                    .sync_on(stream)
                    .expect("prefill trim alloc"),
            );
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    trimmed.device_pointer().cu_deviceptr(),
                    out.device_pointer().cu_deviceptr(),
                    m * n,
                    stream,
                )
                .expect("prefill mxfp4 trim copy");
            }
            Some(trimmed)
        }
    }

    /// Prefill (m > 1) Q6_K GEMM via the sm120 M-row batch GEMV, chunked over
    /// rows: one launch per <= 16 rows, each reading the packed weights once
    /// into shared memory and writing its rows of the `[m, n]` output.
    fn gemm_q6k_prefill_batch(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Option<Arc<Tensor<f32>>> {
        if m < 2 {
            return None;
        }
        let oxide = self.oxide_q6k_gemv.as_ref()?;
        if !oxide.has_sm120_batch() {
            return None;
        }
        let dw = self.device_weights_q6k.get(weight_name)?;
        let k = dw.k_blocks * 256;
        if a.size() != m * k {
            return None;
        }
        let n = dw.n;
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();
        let out = {
            let mut pool = self.prefill_scratch.borrow_mut();
            if pool[slot].as_ref().is_none_or(|t| t.size() < m * n) {
                pool[slot] = Some(Arc::new(
                    api::zeros::<f32>(&[m * n])
                        .sync_on(stream)
                        .expect("q6k prefill output buffer alloc"),
                ));
            }
            pool[slot].as_ref().unwrap().clone()
        };
        // Chunks of 2..=16 rows: the kernel's shared-memory reduction area caps
        // one launch at 16 rows and it rejects m < 2.
        let mut row = 0usize;
        for &rows in &crate::pipeline::capped_batch_chunks(m, 2, 16) {
            oxide
                .forward_sm120_batch_rows(
                    a,
                    row,
                    &dw.tensor,
                    &out,
                    row,
                    n,
                    dw.k_blocks,
                    rows,
                    cu_stream,
                )
                .unwrap_or_else(|e| {
                    panic!("q6k prefill batch {weight_name} rows {row}..{}: {e}", row + rows)
                });
            row += rows;
        }
        Some(out)
    }

    /// Prepare an m>1 activation for the host-dequant f32 fallback GEMM.
    ///
    /// The fallback GEMM runs at `k_pad` (next-pow2) columns with zero padding.
    /// Prefill callers feed the GPU fast paths COMPACT `[m, k]` activations
    /// (the FFN silu output, so the q6k batch kernel can consume it directly);
    /// when the fast paths decline, pad here on the host. An already-padded
    /// `[m, k_pad]` activation (attention/hidden rows) passes through.
    fn pad_prefill_activation_for_f32_gemm(
        &self,
        a: &Arc<Tensor<f32>>,
        m: usize,
        k: usize,
        k_pad: usize,
    ) -> Arc<Tensor<f32>> {
        if k_pad == k || a.size() == m * k_pad {
            return a.clone();
        }
        debug_assert_eq!(
            a.size(),
            m * k,
            "prefill activation is neither [m, k] nor [m, k_pad]"
        );
        let compact = self.download_device_tensor(a);
        self.to_device_padded(&compact, m, k, k_pad)
    }

    /// The slotted quantized paths use pre-allocated scratch buffers with async dispatch,
    /// eliminating the two `sync_on` calls (allocation + kernel) per GEMV.
    /// For m > 1 (prefill): MXFP4 tensor-core GEMM (tiled Q4K/BF16 weights) or
    /// chunked sm120 q6k batch GEMV when the activation is layout-compatible;
    /// otherwise fall back to host-side dequantization + f32 GEMM.
    /// TEMPURA_NO_GPU_PREFILL=1 bisects both GPU fast paths off.
    fn gemm_any_d2d_slot(
        &self,
        a: &Arc<Tensor<f32>>,
        weight_name: &str,
        m: usize,
        slot: usize,
    ) -> Arc<Tensor<f32>> {
        // -- Decode (m == 1): on-GPU dequant GEMV with pre-allocated scratch ---
        // TEMPURA_DECODE_F32_GEMV=1 bisects the quantized GEMVs off: the step
        // runs through the exact f32 host-dequant GEMM (prefill-style) instead.
        let f32_decode = m == 1
            && std::env::var("TEMPURA_DECODE_F32_GEMV").as_deref() == Ok("1");
        if m == 1 && !f32_decode && self.device_weights_q4k.contains_key(weight_name) {
            return self.gemm_q4k_d2d_slot(a, weight_name, slot);
        }
        if m == 1 && self.device_weights_q6k.contains_key(weight_name) {
            return self.gemm_q6k_d2d_slot(a, weight_name, slot);
        }
        if m == 1 && self.device_weights_q2.contains_key(weight_name) {
            return self.gemm_q2_d2d_slot(a, weight_name, slot);
        }

        // -- Prefill (m > 1): GPU fast paths, then host-dequant fallback ------
        // Q6K weights go through the q6k batch GEMV FIRST (matching the batch-
        // decode dispatch, which keeps down_proj at Q6K precision); tiled Q4K
        // projections then take the MXFP4 tensor-core path.
        if m > 1 && std::env::var("TEMPURA_NO_GPU_PREFILL").as_deref() != Ok("1") {
            if let Some(out) = self.gemm_q6k_prefill_batch(a, weight_name, m, slot) {
                return out;
            }
            if let Some(out) = self.gemm_mxfp4_prefill(a, weight_name, m, slot) {
                return out;
            }
        }
        if m > 1 || f32_decode {
            // Q4_K prefill: reconstruct f32 from GPU format (packed 42-u32 or raw GGUF 36-u32).
            if let Some(dw) = self.device_weights_q4k.get(weight_name) {                let k = dw.k_blocks * Q4K_BLOCK_SIZE;
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
                let a_gemm = if m == 1 {
                    // f32_decode bisect: decode activations arrive rank-1.
                    let host = self.download_device_tensor(a);
                    self.to_device_padded(&host, 1, k, k_pad)
                } else {
                    self.pad_prefill_activation_for_f32_gemm(a, m, k, k_pad)
                };
                let out = self.matmul
                    .forward_bt_device_to_device(&a_gemm, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q4_K prefill GEMM failed");
                if m == 1 {
                    // f32_decode bisect: decode consumers expect a rank-1 [n]
                    // tensor; the f32 GEMM returns rank-2 with partition stride
                    // metadata that survives Arc-level reshape. Flatten via the
                    // host (eager mode only -- this switch requires DEBUG_NAN).
                    let host = self.download_device_tensor(&out);
                    return self.to_device_flat_padded(&host, 1, dw.n, dw.n);
                }
                return out;
            }

            // Q6_K prefill: reconstruct f32 from packed GPU format (53 u32/block, ql+qh).
            // (f32_decode bisect skips this: the m=1 activation arrives rank-1
            // and the q6k kpar GEMV is independently validated.)
            if m > 1
            {
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
                        for within in 0..256usize {
                            let half  = within / 128;
                            let sub   = (within % 128) / 32;
                            let l     = within % 32;
                            let group = within / 16;

                            let sc_word = packed_host[base + 1 + group / 4];
                            let sc_byte = ((sc_word >> ((group % 4) * 8)) & 0xFF) as u8;
                            let sc = sc_byte as f32 - 128.0;

                            let ql_byte_idx = half * 64 + l + (sub % 2) * 32;
                            let ql_u32 = packed_host[base + 5 + ql_byte_idx / 4];
                            let ql_byte = ((ql_u32 >> ((ql_byte_idx % 4) * 8)) & 0xFF) as u8;
                            let lo = (ql_byte >> ((sub / 2) * 4)) & 0xF;

                            let qh_byte_idx = half * 32 + l;
                            let qh_u32 = packed_host[base + 37 + qh_byte_idx / 4];
                            let qh_byte = ((qh_u32 >> ((qh_byte_idx % 4) * 8)) & 0xFF) as u8;
                            let hi = (qh_byte >> (sub * 2)) & 0x3;

                            let q6 = lo | (hi << 4);
                            let idx = row * k + k_base + within;
                            f32_weights[idx] = d * sc * (q6 as f32 - 32.0);
                        }
                    }
                }
                let k_pad = next_pow2(k.div_ceil(BK) * BK);
                let w_dev = self.to_device_padded(&f32_weights, dw.n, k, k_pad);
                let a_gemm = self.pad_prefill_activation_for_f32_gemm(a, m, k, k_pad);
                return self.matmul
                    .forward_bt_device_to_device(&a_gemm, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q6K prefill GEMM failed");
                }
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
                let a_gemm = self.pad_prefill_activation_for_f32_gemm(a, m, k, k_pad);
                return self.matmul
                    .forward_bt_device_to_device(&a_gemm, &w_dev, m, dw.n, k_pad)
                    .expect("gemm_any_d2d_slot Q2_0 prefill GEMM failed");
            }
        }
        self.gemm_d2d_slot(a, weight_name, m, slot)
    }

    // -- inference -------------------------------------------------------------

    /// Full autoregressive forward pass with GPU-resident KV cache.
    ///
    /// Activations flow device-to-device across all projection GEMMs.
    /// RMSNorm and residual adds run entirely on device.  SiLU still runs on CPU.
    ///
    /// KV cache: K/V are held in the host mirror and passed directly to
    /// `flash_attn.forward`, skipping the redundant upload + immediate re-download
    /// that occurred when using `forward_device`.
    ///
    /// `hidden_dev` is kept on device throughout the layer loop -- uploaded once
    /// Capture or update+replay the CUDA graph for a single decode step.
    ///
    /// # Design
    ///
    /// The forward pass is captured once between `cuStreamBeginCapture` and
    /// `cuStreamEndCapture`.  The resulting template graph is then:
    ///
    /// - **First call**: instantiated with `cuGraphInstantiateWithFlags`.
    /// - **Subsequent calls**: patched with `cuGraphExecUpdate_v2` -- same topology,
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

        // -- Always: Upload embedding + scalars before capture or launch ------
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
            ).expect("d_embed H->D copy failed");
        }

        // Write device scalars: kv_len, position, kv_write_offset.
        // These are read by device kernels at execution time so the graph is
        // topology-stable across steps (only these tiny H->D copies change).
        let kv_write_off = gpu_kv_cache.seq_len() as i32;
        // Bitcast i32 -> f32 so the cuTile kernel can recover via transmute.
        let kv_len_bits: f32 = f32::from_bits((kv_write_off + 1) as u32);
        let pos_bits: f32    = f32::from_bits(positions[0]);
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_kv_len.device_pointer().cu_deviceptr(),
                &kv_len_bits as *const f32, 1, stream,
            ).expect("d_kv_len H->D copy failed");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("d_positions H->D copy failed");
            cuda_core::memcpy_htod_async::<i32>(
                self.d_kv_write_offset.device_pointer().cu_deviceptr(),
                &kv_write_off as *const i32, 1, stream,
            ).expect("d_kv_write_offset H->D copy failed");
        }

        let mut graph_cache = self.cached_decode_graph.borrow_mut();

        if self.debug_nan_trace {
            // Numerical bisection mode: execute the step WITHOUT graph capture
            // so per-op D2H NaN checks inside forward_decode_inner are legal
            // (async D2H during stream capture is not).
            drop(graph_cache);
            let hidden_dev = self.d_embed.clone();
            self.forward_decode_inner(hidden_dev, positions, gpu_kv_cache);
            unsafe { stream.synchronize().expect("nan-trace stream sync failed"); }
            let logits_arc = self
                .scratch
                .borrow()
                [SLOT_LOGITS]
                .clone()
                .expect("SLOT_LOGITS empty after nan-trace step");
            return self.download_device_tensor(&logits_arc);
        }

        // Validate KV-cache device pointers before replay: a different cache
        // object (new session, fresh-cache bench verification pass) must
        // re-capture -- the graph bakes the cache pointers at capture time.
        {
            let kv_ptrs_stale = match *graph_cache {
                Some(ref dg) => (0..cfg.num_layers).any(|l| {
                    let (k, v) = gpu_kv_cache.kv_device_tensors(l);
                    k.device_pointer().cu_deviceptr() != dg.captured_k_ptrs[l]
                        || v.device_pointer().cu_deviceptr() != dg.captured_v_ptrs[l]
                }),
                None => false,
            };
            if kv_ptrs_stale {
                eprintln!("[cuda_graph] KV cache ptrs changed -- re-capturing");
                *graph_cache = None;
            }
        }

        if let Some(ref decode_graph) = *graph_cache {
            // -- Subsequent steps: plain replay ---------------------------------
            // KV writes are handled by the scatter-write kernel inside the graph
            // (reads d_kv_write_offset at execution time -- no driver calls needed).
            // Attention kv_len is read from d_kv_len and the RoPE cos/sin row is
            // gathered by d2d_gather_row reading d_positions -- both at execution
            // time, so no kernel-node parameter patching is needed here.
            // NOTE: seq_len is advanced by the wrapper (kv.advance) after each
            // forward call -- do NOT advance again here (would double-count).

            // DIAGNOSTIC: read d_kv_len back from device to confirm H->D write landed
            if std::env::var("TEMPURA_DEBUG_KV_LEN").is_ok() {
                unsafe { stream.synchronize().expect("diag sync"); }
                let kv_len_host: Vec<f32> = Arc::clone(&self.d_kv_len)
                    .to_host_vec().sync_on(stream).expect("diag d2h");
                let kv_len_bits = kv_len_host[0].to_bits();
                eprintln!("[diag] d_kv_len bits={kv_len_bits:#010x} -> kv_len={kv_len_bits}");
            }
            unsafe { cuda_core::sys::cuGraphLaunch(decode_graph.cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("stream sync after graph launch failed"); }
            drop(graph_cache);
        } else {
            // -- First step: capture graph, instantiate, launch ----------------
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

            // Record the KV-cache device pointers this graph was captured with.
            let captured_k_ptrs: Vec<u64> = (0..cfg.num_layers)
                .map(|l| gpu_kv_cache.kv_device_tensors(l).0.device_pointer().cu_deviceptr())
                .collect();
            let captured_v_ptrs: Vec<u64> = (0..cfg.num_layers)
                .map(|l| gpu_kv_cache.kv_device_tensors(l).1.device_pointer().cu_deviceptr())
                .collect();

            // cu_graph is kept alive inside DecodeGraph so node handles remain valid.
            *graph_cache = Some(DecodeGraph {
                cu_exec,
                cu_graph,
                device: stream.device().clone(),
                captured_k_ptrs,
                captured_v_ptrs,
            });

            unsafe { cuda_core::sys::cuGraphLaunch(cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("stream sync after first graph launch failed"); }
            drop(graph_cache);
        }

        // Logits reside in SLOT_LOGITS -- download and return.
        let logits_arc = self
            .scratch.borrow()[SLOT_LOGITS].clone()
            .expect("SLOT_LOGITS empty after graph launch");
        self.download_device_tensor(&logits_arc)
    }

    /// Refresh the CUDA-graph-stable RoPE cos/sin row buffers for the CURRENT
    /// position and report whether the graph rope path is usable.
    ///
    /// Enqueues two `d2d_gather_row` kernels on the matmul stream that copy
    /// table row `pos` into `d_rope_cos_row` / `d_rope_sin_row`, where `pos`
    /// is read from `d_positions` at EXECUTION time. Inside a CUDA graph
    /// capture these kernels are recorded once and re-gather the right row on
    /// every replay -- unlike `cutile_rope_from_table`'s `pos` host launch
    /// scalar, which the graph bakes at capture time and replays stale.
    ///
    /// Returns `false` (and gathers nothing) when the precomputed table does
    /// not cover `pos_hint` or the gather fails; the caller must then use the
    /// standard `forward_device_scratch` rope path. Callers that capture this
    /// into a graph must gate replay on `positions[0] < table_max_pos()` so
    /// replay-time positions stay in range (the gather clamps as defense in
    /// depth).
    fn rope_rows_graph_ready(&self, pos_hint: u32) -> bool {
        let Some((cos_ptr, sin_ptr, half, max_pos)) = self.cutile_rope.table_ptrs() else {
            return false;
        };
        if pos_hint as usize >= max_pos {
            return false;
        }
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();
        let pos_ptr = self.d_positions.device_pointer().cu_deviceptr();
        let cos_dst = self.d_rope_cos_row.device_pointer().cu_deviceptr();
        let sin_dst = self.d_rope_sin_row.device_pointer().cu_deviceptr();
        let res = self
            .kv_scatter_write
            .gather_row_by_device_pos(cos_ptr, cos_dst, pos_ptr, half as u32, max_pos as u32, cu_stream)
            .and_then(|_| {
                self.kv_scatter_write.gather_row_by_device_pos(
                    sin_ptr, sin_dst, pos_ptr, half as u32, max_pos as u32, cu_stream,
                )
            });
        if let Err(e) = res {
            eprintln!("[rope] gather_row_by_device_pos failed: {e}; using standard rope");
            return false;
        }
        true
    }

    /// Run the decode forward pass without CUDA graph capture.
    ///
    /// Used as a fallback when `cuGraphExecUpdate_v2` fails or when graph capture
    /// is not applicable (e.g. when padding mismatches require D->H sync inside
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
            ).expect("forward_decode_nogpu d_kv_len H->D");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("forward_decode_nogpu d_positions H->D");
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

    /// TEMPURA_DEBUG_NAN helper: D2H a tensor and eprintln a NaN report.
    /// Only called when `debug_nan_trace` is set (never during graph capture).
    ///
    /// Also prints the NaN-tolerant bits_sum of the tensor: with
    /// TEMPURA_DEBUG_NAN=1, diffing the per-op sums between two otherwise
    /// identical runs localizes the first op whose output is
    /// allocation-history-dependent (read-before-write scratch).
    fn nan_report(&self, label: &str, t: &Arc<Tensor<f32>>) {
        let stream = self.matmul.stream();
        let v = t.to_host_vec().sync_on(stream).expect("nan-trace download");
        let n = v.iter().filter(|x| x.is_nan()).count();
        let bits_sum: u64 = v.iter().map(|x| x.to_bits() as u64).sum();
        if n > 0 {
            let first = v.iter().position(|x| x.is_nan()).unwrap_or(0);
            eprintln!("[nan-trace] {label}: {n}/{} NaN (first at idx {first}) bits_sum={bits_sum}", v.len());
        } else {
            eprintln!("[nan-trace] {label}: ok bits_sum={bits_sum}");
        }
        // Per-1k-element slice sums: localizes WHICH region of a diverging
        // tensor moves (stale tail vs whole-tensor difference).
        if std::env::var("TEMPURA_DEBUG_SLICES").is_ok() && v.len() > 1024 {
            let mut out = String::new();
            for (si, chunk) in v.chunks(1024).enumerate() {
                let s: u64 = chunk.iter().map(|x| x.to_bits() as u64).sum();
                out.push_str(&format!(" {si}:{s}"));
            }
            eprintln!("[nan-trace] {label} slices:{out}");
        }
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
    /// executes -- API calls are only recorded.  The return value is a placeholder;
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

        // Refresh the graph-stable RoPE cos/sin rows once per step. The two
        // gather kernels read the position from d_positions at EXECUTION time,
        // so inside a CUDA graph they are captured once and re-gather the
        // correct table row on every replay. When the table does not cover the
        // position (or gathering fails), fall back to the standard rope path.
        let rope_graph = self.rope_rows_graph_ready(positions[0]);

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
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} input-hidden"), &hidden_dev);
                self.nan_report(&format!("L{layer} pre-attn-norm"), &normed_dev);
            }

            // Q / K / V projections (GPU, pre-allocated slots).
            //
            // Decode (seq_len=1) with oxide GEMVs: K and V run on `stream_kv` concurrently
            // with Q on `stream_main`.  For GQA models K/V are 4x smaller than Q and hide
            // behind Q's latency.  The fork/join events are captured by the CUDA graph in
            // RELAXED mode, so the concurrent topology is replayed on every graph execution.
            //
            // K and V use the f32 oxide GEMV (not the Q8 dp4a path) so that each writes into
            // its own pre-allocated slot tensor (SLOT_K / SLOT_V), avoiding the shared Q8
            // scratch buffer that would be clobbered by concurrent Q execution on stream_main.
            let (q_dev, k_dev, v_dev) = self.dispatch_qkv_multistream(
                &normed_dev, q_w_name, k_w_name, v_w_name, seq_len,
            );
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} q-proj"), &q_dev);
                self.nan_report(&format!("L{layer} k-proj"), &k_dev);
                self.nan_report(&format!("L{layer} v-proj"), &v_dev);
            }

            // RoPE (GPU, pre-allocated scratch; uses precomputed cos/sin table for seq_len=1).
            // Graph-stable path: the cos/sin row for the CURRENT position is
            // gathered into the one-row buffers by d2d_gather_row (reads
            // d_positions at execution time), so a captured graph replays with
            // the correct rotation for every position. `rope_graph` was
            // computed once before the layer loop.
            let q_rot_dev = if rope_graph {
                self.cutile_rope
                    .forward_device_scratch_graph(
                        &q_dev,
                        n_heads,
                        head_dim,
                        &self.d_rope_cos_row,
                        &self.d_rope_sin_row,
                        &mut self.q_rope_scratch.borrow_mut(),
                        shared_stream,
                    )
                    .expect("Q RoPE (graph rows) failed")
            } else {
                self.cutile_rope
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
                    .expect("Q RoPE failed")
            };
            let k_rot_dev = if rope_graph {
                self.cutile_rope
                    .forward_device_scratch_graph(
                        &k_dev,
                        n_kv_heads,
                        head_dim,
                        &self.d_rope_cos_row,
                        &self.d_rope_sin_row,
                        &mut self.k_rope_scratch.borrow_mut(),
                        shared_stream,
                    )
                    .expect("K RoPE (graph rows) failed")
            } else {
                self.cutile_rope
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
                    .expect("K RoPE failed")
            };

            // Scatter-write K and V into the KV cache at the slot indicated by
            // d_kv_write_offset (read from device at execution time).  This is
            // graph-stable: the src/cache pointers are captured once; only the
            // value at d_kv_write_offset changes (updated via H->D before each step).
            let kv_dim_u32 = (n_kv_heads * head_dim) as u32;
            let (k_cache_arc, v_cache_arc) = gpu_kv_cache.kv_device_tensors(layer);
            let write_off_ptr = self.d_kv_write_offset.device_pointer().cu_deviceptr();
            let cu_stream_raw = shared_stream.cu_stream();
            self.kv_scatter_write
                .copy_with_device_offset(
                    k_rot_dev.device_pointer().cu_deviceptr(),
                    k_cache_arc.device_pointer().cu_deviceptr(),
                    write_off_ptr,
                    kv_dim_u32,
                    cu_stream_raw,
                )
                .expect("KV scatter-write K failed");
            self.kv_scatter_write
                .copy_with_device_offset(
                    v_dev.device_pointer().cu_deviceptr(),
                    v_cache_arc.device_pointer().cu_deviceptr(),
                    write_off_ptr,
                    kv_dim_u32,
                    cu_stream_raw,
                )
                .expect("KV scatter-write V failed");
            gpu_kv_cache.mark_host_stale();

            // kv_len includes the token just appended.
            let kv_len = gpu_kv_cache.seq_len() + seq_len;

            // GPU decode attention (fused flash-attention, pre-allocated scratch).
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (k_dev_full, v_dev_full) = gpu_kv_cache.kv_device_tensors(layer);

            // Prefer oxide decode attention (kv_len read from device at execution
            // time -- correct CUDA graph replay without stale kv_len baking).
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
                // Split-K variant: grid (n_heads x 16) fills all SMs; the
                // kpar fallback still beats the single-warp scan.
                if let Some(ref splitk_scratch) = self.attn_splitk_scratch {
                    oda.forward_splitk_d2d(
                        &out_tensor, &q_rot_dev, &k_dev_full, &v_dev_full,
                        &self.d_kv_len, splitk_scratch, n_heads, n_kv_heads, head_dim, scale,
                    ).expect("oxide decode attention splitk failed");
                } else {
                    oda.forward_kpar_d2d(
                        &out_tensor, &q_rot_dev, &k_dev_full, &v_dev_full,
                        &self.d_kv_len, n_heads, n_kv_heads, head_dim, scale,
                    ).expect("oxide decode attention kpar failed");
                }
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

            // Pad [q_dim] -> [1, q_pad] for O-projection (zero-copy reshape when equal).
            let attn_padded_dev = self.attn_out_to_oproj_device(&attn_out_dev, q_dim, self.q_pad);
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} attn-out"), &attn_out_dev);
                self.nan_report(&format!("L{layer} q-rot"), &q_rot_dev);
            }

            // O-projection (GPU, pre-allocated slot).
            let o_dev = self.gemm_any_d2d_slot(&attn_padded_dev, o_w_name, seq_len, SLOT_O);

            // Post-attention residual add (GPU, pre-allocated scratch).
            hidden_dev = self.add_device_scratch(
                &hidden_dev,
                &o_dev,
                seq_len * self.h_pad,
                &mut self.post_attn_add_scratch.borrow_mut(),
            );
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} o-proj"), &o_dev);
                self.nan_report(&format!("L{layer} post-attn-hidden"), &hidden_dev);
            }

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
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} post-attn-norm"), &normed2_dev);
            }

            // FFN gate + up projections.
            // Decode path: try fused gate+up GEMV with inline SiLU (halves activation DRAM).
            // Prefill path (seq_len > 1) or non-Q4K: fall through to two separate GEMVs.
            // TEMPURA_NO_FUSED_SILU=1 bisects the fused kernel off (separate GEMVs).
            let fused_silu_disabled = std::env::var("TEMPURA_NO_FUSED_SILU")
                .map(|v| v == "1")
                .unwrap_or(false);
            let gate_silu_n_dev = if seq_len == 1
                && !fused_silu_disabled
                && self.device_weights_q4k.contains_key(gate_w_name.as_str())
                && self.device_weights_q4k.contains_key(up_w_name.as_str())
                && self.oxide_q4k_gemv.as_ref().map(|o| o.has_sm120_fused_silu() || o.has_fused_silu()).unwrap_or(false)
            {
                let oxide = self.oxide_q4k_gemv.as_ref().unwrap();
                let gdw = self.device_weights_q4k.get(gate_w_name.as_str()).unwrap();
                let udw = self.device_weights_q4k.get(up_w_name.as_str()).unwrap();
                let a_flat = Arc::clone(&normed2_dev)
                    .reshape(&[next_pow2(gdw.k_blocks * tempura_kernels::Q4K_BLOCK_SIZE)])
                    .expect("fused silu reshape failed");
                // Prefer sm_120 hand-tuned PTX (direct Blackwell codegen) over oxide sm_80.
                if oxide.has_sm120_fused_silu() {
                    oxide
                        .forward_sm120_fused_silu_d2d(&a_flat, &gdw.tensor, &udw.tensor, gdw.n, gdw.k_blocks)
                        .expect("q4k_fused_silu_sm120 failed")
                } else {
                    oxide
                        .forward_fused_silu_dp4a_d2d(&a_flat, &gdw.tensor, &udw.tensor, gdw.n, gdw.k_blocks)
                        .expect("q4k_fused_silu_dp4a failed")
                }
            } else {
                let gate_dev = self.gemm_any_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
                let up_dev = self.gemm_any_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);
                if self.debug_nan_trace {
                    self.nan_report(&format!("L{layer} gate-dev"), &gate_dev);
                    self.nan_report(&format!("L{layer} up-dev"), &up_dev);
                }
                self.cutile_elementwise
                    .silu_mul_device_scratch(
                        &gate_dev,
                        &up_dev,
                        seq_len,
                        inter,
                        &mut self.silu_out_scratch.borrow_mut(),
                        &shared_stream,
                    )
                    .expect("SiLU-mul failed")
            };
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} gate-silu"), &gate_silu_n_dev);
            }

            // FFN down projection.
            // When inter_pad == inter (power-of-2 models) no padding is needed.
            let gate_silu_dev = if self.inter_pad == inter {
                gate_silu_n_dev
            } else {
                // GPU-side zero-padding: D->D copy of `inter` elements into a pre-zeroed
                // `inter_pad` scratch.  Eliminates the D->H stream sync that was here before.
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
                    .expect("down_proj_pad D->D copy failed");
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
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} down-proj"), &down_padded);
                self.nan_report(&format!("L{layer} post-ffn-hidden"), &hidden_dev);
            }
        }

        // -- Final norm + lm_head ---------------------------------------------
        // For seq_len=1 hidden_dev IS the last token -- no D->H download or
        // H->D re-upload needed.
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
        if self.debug_nan_trace {
            self.nan_report("final-norm", &last_dev);
        }

        // lm_head GEMV -- result stored in SLOT_LOGITS (pre-allocated).
        self.gemm_d2d_slot(&last_dev, "lm_head.weight", 1, SLOT_LOGITS);
        if self.debug_nan_trace {
            let logits_arc = self
                .scratch
                .borrow()
                [SLOT_LOGITS]
                .clone()
                .expect("SLOT_LOGITS empty after lm_head");
            self.nan_report("lm_head", &logits_arc);
        }
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

        // -- 1. Embedding lookup (CPU) -----------------------------------------
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

        // -- 2. CUDA graph dispatch (decode + GPU KV cache) ---------------------
        // For single-token decode with an active GPU KV cache and power-of-2 model
        // dimensions (no D->H sync needed inside the layer loop), capture the
        // forward pass into a CUDA graph and replay it.  This replaces ~200
        // individual `cuLaunchKernel` submissions with one `cuGraphLaunch`.
        //
        // The D->D padding for inter_pad != inter is now GPU-only (pre-zeroed scratch
        // + async D->D copy), so the layer loop itself has no D->H syncs regardless.
        // However the CUDA graph path requires all cuTile JIT variants to be
        // pre-compiled before capture begins; for non-power-of-2 N (e.g. vocab=32000)
        // some variants may not be warmed up, causing "not permitted during capture"
        // errors.  Until warmup covers all shapes, restrict to power-of-2 inter models.
        //
        // Preconditions:
        //   * seq_len == 1 -- decode mode
        //   * gpu_kv_cache.has_device_kv() -- GPU flash-attention is active
        //   * !cfg.use_alibi -- RoPE path
        //   * oxide_decode_attn present -- the captured attention must read
        //     kv_len from device memory; the cuTile flash-attn fallback takes
        //     kv_len as a host launch scalar which the graph would bake at
        //     capture and replay stale
        //   * position < rope table_max_pos -- the captured RoPE gathers the
        //     cos/sin row by device-side position; beyond the table the graph
        //     would clamp to the last row (wrong rotation)
        // inter_pad/h_pad: padding is handled via D->D copy (capturable).
        // cuTile GEMV uses async_on (capture-compatible); warmup covers all needed
        // shapes, so no JIT compilation happens inside capture.
        // TEMPURA_FORCE_PREFILL_M1=1: bisect switch that routes a single-token
        // forward through the prefill loop (host-staged attention, prefill
        // GEMMs) instead of the decode path -- used with TEMPURA_DEBUG_NAN to
        // diff per-op bits_sums between two m=1 runs (decode bug hunting).
        // TEMPURA_DEBUG_NAN prints the gate values here when set.
        if seq_len == 1
            && gpu_kv_cache.has_device_kv()
            && !cfg.use_alibi
            && std::env::var("TEMPURA_FORCE_PREFILL_M1").as_deref() != Ok("1")
        {
            if self.oxide_decode_attn.is_some()
                && position < self.cutile_rope.table_max_pos()
            {
                // The host KV mirror may be stale; we only rebuild it when transitioning
                // to the CPU fallback path, which does not apply here.
                return self.forward_decode_with_graph(&hidden, &positions, gpu_kv_cache);
            }
            // Graph-capture correctness gate failed: run the same decode inner
            // WITHOUT a graph.  The cuTile flash-attn gets a fresh host kv_len
            // every call and the standard RoPE path handles any position, so
            // both cases stay correct -- just slower.
            return self.forward_decode_nogpu(&hidden, &positions, gpu_kv_cache);
        }

        // -- 3. Layer loop (prefill or CPU-attention fallback) -----------------
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

            // -- 2a. Pre-attention RMSNorm (GPU) -------------------------------
            // hidden_dev is already on device -- no upload needed.
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

            // -- 2b. QKV projections -- device-to-device -----------------------
            // For Q2_0 models: gemm_any_d2d_slot dequantizes on-GPU (m=1 only).
            // For f32 models: same path as before, synchronous.
            // For prefill (m>1): Q2_0 falls back to dequantized f32 GEMM.
            let q_dev = self.gemm_any_d2d_slot(&normed_dev, q_w_name, seq_len, SLOT_Q);
            let k_dev = self.gemm_any_d2d_slot(&normed_dev, k_w_name, seq_len, SLOT_K);
            let v_dev = self.gemm_any_d2d_slot(&normed_dev, v_w_name, seq_len, SLOT_V);

            // -- 2c. RoPE (GPU) -- rotate Q and K entirely on device -----------
            // cos/sin tables (tiny: seq_len x n_heads x head_dim/2 floats each)
            // are computed on CPU and uploaded.  Q and K are deinterleaved and
            // rotated on the GPU using pre-allocated scratch buffers, eliminating
            // the 5 x 2 api::zeros calls (280 blocking allocs per token at 28 layers).
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
                // ALiBi: no rotation -- keep Q/K tensors on device as-is.
                (q_dev.clone(), k_dev.clone())
            };
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} pre-attn-norm"), &normed_dev);
                self.nan_report(&format!("L{layer} q-proj"), &q_dev);
                self.nan_report(&format!("L{layer} k-proj"), &k_dev);
                self.nan_report(&format!("L{layer} v-proj"), &v_dev);
                self.nan_report(&format!("L{layer} q-rot"), &q_rot_dev);
                self.nan_report(&format!("L{layer} k-rot"), &k_rot_dev);
            }

            // -- 2d. KV-cache append -------------------------------------------
            // When the GPU KV cache is active and seq_len == 1 (decode), use the
            // GPU-only append path that omits the D->H host-mirror update.
            // This eliminates 2 x n_layers blocking D->H `sync_on` calls per token
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

            // -- 2e. Attention -------------------------------------------------
            let scale = 1.0 / (head_dim as f32).sqrt();

            // For decode (seq_len=1) with the GPU KV cache active: run the fused
            // `cutile_decode_attn` kernel which reads Q, K, V entirely on device,
            // eliminating all PCIe traffic for the KV cache.  At context length L
            // this saves 2 x L x n_kv_heads x head_dim x 4 bytes per layer per
            // token -- dominant PCIe traffic at long context.
            //
            // Falls back to the host-based path when:
            //   - seq_len > 1 (prefill): GPU flash attention via forward_batched
            //   - seq_len == 1 but kv_len > GPU_KV_MAX_SEQ_LEN: CPU GQA fallback
            let attn_padded_dev = if seq_len == 1 && gpu_kv_cache.has_device_kv() {
                // GPU decode attention: Q on device, K/V from GPU-resident cache.
                // Use scratch variant to eliminate the per-call api::zeros.sync_on
                // allocation (a full stream sync + cudaMallocAsync) -- saves 28
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
                // Pad [q_dim] -> [1, q_pad] for O projection via GPU D->D copy.
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

            // -- 2f. O projection -- device-to-device --------------------------
            let o_dev = self.gemm_any_d2d_slot(&attn_padded_dev, o_w_name, seq_len, SLOT_O);
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} attn-out"), &attn_padded_dev);
                self.nan_report(&format!("L{layer} o-proj"), &o_dev);
            }

            // -- 2g. Residual add (GPU) ----------------------------------------
            // o_dev has h columns; pad to h_pad when h_pad > h (rare edge case
            // e.g. h=5120 -> h_pad=8192).  For models where h_pad == h (the
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

            // -- 2h. Post-attention RMSNorm (GPU) -----------------------------
            // hidden_dev is already updated on device -- no upload needed.
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

            // -- 2i. FFN gate + up -- device-to-device -------------------------
            // For Q2_0 models gemm_any_d2d_slot dispatches on-GPU dequant (synchronous).
            // For f32 models the synchronous gemm_d2d_slot path is used.
            let gate_dev = self.gemm_any_d2d_slot(&normed2_dev, gate_w_name, seq_len, SLOT_GATE);
            let up_dev = self.gemm_any_d2d_slot(&normed2_dev, up_w_name, seq_len, SLOT_UP);
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} post-attn-hidden"), &hidden_dev);
                self.nan_report(&format!("L{layer} post-attn-norm"), &normed2_dev);
                self.nan_report(&format!("L{layer} gate-dev"), &gate_dev);
                self.nan_report(&format!("L{layer} up-dev"), &up_dev);
            }

            // -- 2j. SiLU gate x up (GPU) -------------------------------------
            // gate_dev and up_dev already reside on the GPU.  Use the scratch
            // variant to eliminate the per-call api::zeros.sync_on allocation --
            // saves 28 blocking allocs per decode token across all layers.
            let gate_silu_n_dev = if seq_len == 1 {
                let silu_stream = self.matmul.stream();
                self.cutile_elementwise
                    .silu_mul_device_scratch(
                        &gate_dev,
                        &up_dev,
                        seq_len,
                        inter,
                        &mut self.silu_out_scratch.borrow_mut(),
                        &silu_stream,
                    )
                    .expect("CutileGpuModel: GPU SiLU-mul failed")
            } else {
                let silu_stream = self.matmul.stream();
                self.cutile_elementwise
                    .silu_mul_device(&gate_dev, &up_dev, seq_len, inter, &silu_stream)
                    .expect("CutileGpuModel: GPU SiLU-mul failed")
            };

            // -- 2k. FFN down -- device-to-device ------------------------------
            // Pad gate_silu from inter to inter_pad for the down_proj GEMM.
            // When inter == inter_pad (pow-2 models) this is a free reshape.
            // For seq_len==1 with non-power-of-2 inter (e.g. Llama-3.1-8B inter=14336):
            // use GPU D->D copy into the pre-zeroed inter_pad scratch to avoid 32 D->H
            // stream syncs per token -- the dominant bottleneck before this fix.
            // For seq_len>1: pass the COMPACT [seq_len, inter] silu output -- the
            // q6k sm120 batch fast path in gemm_any_d2d_slot consumes it directly
            // (same layout the batch-decode path feeds the same kernel). The
            // host-dequant f32 fallback pads to inter_pad internally; the old
            // D->H->D pad here cost a full download+upload round-trip per layer.
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
                    .expect("down_proj_pad D->D copy failed");
                }
                let result = Arc::new(pad_arc);
                *self.down_proj_pad_scratch.borrow_mut() = Some(result.clone());
                result
            } else {
                gate_silu_n_dev
            };
            let down_dev = self.gemm_any_d2d_slot(&gate_silu_dev, down_w_name, seq_len, SLOT_DOWN);
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} gate-silu"), &gate_silu_dev);
                self.nan_report(&format!("L{layer} down-proj"), &down_dev);
            }

            // -- 2l. Residual add (GPU) ----------------------------------------
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
            if self.debug_nan_trace {
                self.nan_report(&format!("L{layer} post-ffn-hidden"), &hidden_dev);
            }
        }

        // -- 3. Final norm + lm_head --------------------------------------------
        // For decode (seq_len=1) hidden_dev IS the last token -- pass it directly
        // to the final RMSNorm, skipping the unnecessary D->H download + H->D re-upload
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

    // -- Batch decode CUDA graph ----------------------------------------------

    /// Inner layer-loop for batch decode CUDA graph capture.
    ///
    /// All intermediate tensors are pre-allocated struct fields; no device
    /// allocation or stream synchronisation happens inside this function.
    /// The function body is captured as a CUDA graph on the first call.
    ///
    /// # Assumptions
    /// - All M sequences are **synchronised**: same `kv_write_off` and same
    ///   `position` for every sequence at each decode step.
    /// - All projection weights are Q4K (quantised).
    /// - `q_pad == q_dim` (standard LLaMA/Qwen geometry; no attn-output padding).
    fn forward_batch_decode_inner_graph(
        &self,
        m:            usize,
        positions:    &[u32],          // length m (same value for all in synchronised batch)
        kv_caches:    &mut [CutileKvCache],
        use_mxfp4:    bool,            // if true: MXFP4 tensor-core path for Q/K/O/gate/up
    ) {
        let cfg        = &self.config;
        let h          = cfg.hidden_size;
        let n_heads    = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let head_dim   = cfg.head_dim;
        let q_dim      = n_heads * head_dim;
        let kv_dim     = n_kv_heads * head_dim;
        let inter      = cfg.intermediate_size;
        let inter_pad  = self.inter_pad;
        let stream     = self.matmul.stream();
        let cu_stream  = stream.cu_stream();

        // oxide_q4k batch is required for Q4K projection paths but not for all-MXFP4 models.
        // When use_mxfp4=true and all projections are MXFP4, oxide is never called.
        let oxide = self.oxide_q4k_gemv.as_ref().filter(|o| {
            o.has_streamk_batch() || o.has_mma_batch() || o.has_sm120_batch()
        });
        if oxide.is_none() && !use_mxfp4 {
            panic!("forward_batch_decode_inner_graph: oxide Q4K sm120 batch required for non-MXFP4 path");
        }

        // Shared write offset (same for all M sequences -- synchronised batch).
        let write_off_ptr = self.d_kv_write_offset.device_pointer().cu_deviceptr();
        let kv_dim_u32 = kv_dim as u32;

        // hidden_dev starts as batch_hidden (pre-loaded with M embeddings before capture).
        // We do NOT borrow_mut here (would conflict); instead we alias via Arc::clone.
        let mut hidden_dev: Arc<Tensor<f32>> = self.batch_hidden.clone();

        // MXFP4 path: sm_120a tensor-core GEMM + on-GPU activation quantization.
        // Gated by use_mxfp4 which is set at graph-capture time and baked into the graph.
        // Two separate graphs: dp4a (use_mxfp4=false) and MXFP4 (use_mxfp4=true).
        let mxfp4_active = use_mxfp4
            && self.mxfp4_gemm.is_some()
            && self.mxfp4_act_packed.is_some()
            && self.mxfp4_act_scales.is_some()
            && !self.device_weights_mxfp4_tiled.is_empty();
        // m_pad: nearest multiple of 16 >= m. For max_batch_graph<=16, always 16.
        let mxfp4_m_pad = 16u32;

        #[allow(clippy::type_complexity)]
        let layer_names: Vec<(String,String,String,String,String,String,String,String,String)> =
            (0..cfg.num_layers).map(|l| {
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
            }).collect();

        // Refresh the graph-stable RoPE cos/sin rows once per step (gather
        // kernels read the position from d_positions at execution time; inside
        // the CUDA graph they re-gather the right row on every replay). All M
        // sequences share one position (synchronised batch -- graph gate).
        let rope_graph = self.rope_rows_graph_ready(positions[0]);

        for layer in 0..cfg.num_layers {
            let (ref input_ln, ref post_ln,
                 ref qw, ref kw, ref vw, ref ow,
                 ref gate_w, ref up_w, ref down_w) = layer_names[layer];

            // -- Pre-attention RMSNorm -----------------------------------------
            let input_ln_dev = self.norm_weights.get(input_ln).expect("input_layernorm");
            let normed1 = self.cutile_rmsnorm
                .forward_device_fused_scratch(
                    stream, hidden_dev.clone(), input_ln_dev,
                    m, h, self.h_pad, cfg.layer_norm_eps,
                    &mut self.batch_pre_attn_norm_scratch.borrow_mut(),
                )
                .expect("batch pre-attn RMSNorm");
            // normed1 is [m, h_pad]; trim to [m * h_pad] for GEMV
            let normed1_flat: Arc<Tensor<f32>> = normed1;

            // -- Q / K / V batch GEMM (MMA tensor-core preferred, dp4a fallback) -
            // gemv_q4k_h is only called when the weight is Q4K; oxide must be Some then.
            let gemv_q4k_h = |a: &Arc<Tensor<f32>>, wname: &str, out: &Arc<Tensor<f32>>,
                               n: usize, k_blocks: usize| {
                let dw = self.device_weights_q4k.get(wname)
                    .unwrap_or_else(|| panic!("batch inner: Q4K weight {wname} missing"));
                let ox = oxide.unwrap_or_else(|| panic!("batch inner: oxide Q4K required for Q4K path"));
                if ox.has_streamk_batch() {
                    ox.forward_streamk_batch_d2d_into(
                        a, &dw.tensor, m, n, k_blocks,
                        &self.batch_q8_qs_h, &self.batch_q8_ds_h, out,
                    ).unwrap_or_else(|e| panic!("batch inner stream-K GEMM {wname}: {e}"));
                } else if ox.has_mma_batch() {
                    ox.forward_mma_batch_d2d_into(
                        a, &dw.tensor, m, n, k_blocks,
                        &self.batch_q8_qs_h, &self.batch_q8_ds_h, out,
                    ).unwrap_or_else(|e| panic!("batch inner MMA GEMM {wname}: {e}"));
                } else {
                    ox.forward_sm120_batch_d2d_into(
                        a, &dw.tensor, m, n, k_blocks,
                        &self.batch_q8_qs_h, &self.batch_q8_ds_h, out,
                    ).unwrap_or_else(|e| panic!("batch inner GEMV {wname}: {e}"));
                }
            };

            // Trim normed1 to compact [m * k] if h_pad != k (k = hidden_size for Q/K/V).
            // dw_q/dw_k are None for all-MXFP4 models (BF16 safetensors); k_h from MXFP4 then.
            let dw_q = self.device_weights_q4k.get(qw.as_str());
            let dw_k = self.device_weights_q4k.get(kw.as_str());

            let k_h = if let Some(dw) = dw_q.as_ref() {
                dw.k_blocks * 256
            } else if let Some(dw_mx) = self.device_weights_mxfp4_tiled.get(qw.as_str()) {
                dw_mx.k
            } else {
                panic!("batch inner: q_proj weight {qw} missing from Q4K and MXFP4");
            };
            let act_h = if self.h_pad == k_h {
                normed1_flat.clone()
            } else {
                let compact = self.batch_normed1.clone();
                for r in 0..m {
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            compact.device_pointer().cu_deviceptr() + (r * k_h * 4) as u64,
                            normed1_flat.device_pointer().cu_deviceptr() + (r * self.h_pad * 4) as u64,
                            k_h, stream,
                        ).expect("batch inner: normed compact D->D");
                    }
                }
                compact
            };

            // -- Q / K / V projection dispatch ---------------------------------
            // For Q4_K_M models, Q and K are Q4K (-> MXFP4 available) while
            // V is Q6K (-> no MXFP4, uses per-seq Q6K fallback). Quantize once,
            // then dispatch each projection independently.
            //
            // Hybrid path: if any of Q/K has MXFP4, quantize h-width activation
            // on GPU (no PCIe) and GEMM those projections via tensor cores.
            // V always uses Q4K or Q6K (whichever is available).
            let has_mxfp4_q = mxfp4_active
                && self.device_weights_mxfp4_tiled.contains_key(qw.as_str());
            let has_mxfp4_k = mxfp4_active
                && self.device_weights_mxfp4_tiled.contains_key(kw.as_str());

            // Quantize h-width activation once if any h-width projection uses MXFP4.
            let h_act_quantized = (has_mxfp4_q || has_mxfp4_k) && {
                // Ensure batch_normed1 has valid data for all 16 rows (m_pad=16).
                // When h_pad == k_h, act_h is normed1_flat (m rows only); copy into
                // batch_normed1 so we can quantize with m_pad=16.
                // When h_pad != k_h, act_h IS batch_normed1 (already compacted).
                let act_ptr   = act_h.device_pointer().cu_deviceptr();
                let norm1_ptr = self.batch_normed1.device_pointer().cu_deviceptr();
                if act_ptr != norm1_ptr {
                    for r in 0..m {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                norm1_ptr + (r * k_h * 4) as u64,
                                act_ptr   + (r * k_h * 4) as u64,
                                k_h, stream,
                            ).expect("mxfp4: normed1 -> batch_normed1");
                        }
                    }
                }
                let mxfp4    = self.mxfp4_gemm.as_ref().unwrap();
                let packed    = self.mxfp4_act_packed.as_ref().unwrap();
                let scales    = self.mxfp4_act_scales.as_ref().unwrap();
                let k_h_pad64 = ((k_h + 63) & !63) as u32;
                // m_valid=m: rows m..16 of batch_normed1 are stale and MUST NOT
                // contribute to the shared m16k32 tile-max scale.
                mxfp4.quantize_activation(
                    &self.batch_normed1, packed, scales,
                    mxfp4_m_pad, k_h_pad64, m as u32, cu_stream,
                ).unwrap_or_else(|e| panic!("mxfp4 h-act quant L{layer}: {e}"));
                true
            };

            // V projection: MXFP4 / Q4K batch GEMV / Q6K sequential GEMVs.
            // BF16 (all-MXFP4) models have V in device_weights_mxfp4_tiled.
            // Q4_K_M models have V as Q6K; MXFP4 check first.
            if mxfp4_active && self.device_weights_mxfp4_tiled.contains_key(vw.as_str()) {
                // MXFP4 V: reuse already-quantized h-width activation (same packed/scales as Q/K).
                let mxfp4  = self.mxfp4_gemm.as_ref().unwrap();
                let packed  = self.mxfp4_act_packed.as_ref().unwrap();
                let scales  = self.mxfp4_act_scales.as_ref().unwrap();
                if !h_act_quantized {
                    let k_h_pad64 = ((k_h + 63) & !63) as u32;
                    // Ensure batch_normed1 has valid data for all 16 rows.
                    let act_ptr   = act_h.device_pointer().cu_deviceptr();
                    let norm1_ptr = self.batch_normed1.device_pointer().cu_deviceptr();
                    if act_ptr != norm1_ptr {
                        for r in 0..m {
                            unsafe {
                                cuda_core::memcpy_dtod_async::<f32>(
                                    norm1_ptr + (r * k_h * 4) as u64,
                                    act_ptr   + (r * k_h * 4) as u64,
                                    k_h, stream,
                                ).expect("mxfp4 V: normed1 -> batch_normed1");
                            }
                        }
                    }
                    mxfp4.quantize_activation(
                        &self.batch_normed1, packed, scales,
                        mxfp4_m_pad, k_h_pad64, m as u32, cu_stream,
                    ).unwrap_or_else(|e| panic!("mxfp4 V quant L{layer}: {e}"));
                }
                let dw_v_mx = self.device_weights_mxfp4_tiled.get(vw.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_v_mx.packed, &dw_v_mx.tiled_scales,
                    &self.batch_gemv_v,
                    mxfp4_m_pad, dw_v_mx.n as u32, (dw_v_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 V L{layer}: {e}"));
                for r in 0..m {
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            self.batch_v_seqs[r].device_pointer().cu_deviceptr(),
                            self.batch_gemv_v.device_pointer().cu_deviceptr() + (r * kv_dim * 4) as u64,
                            kv_dim, stream,
                        ).expect("batch inner: v_seqs MXFP4 slice");
                    }
                }
            } else if let Some(dw_v) = self.device_weights_q4k.get(vw.as_str()) {
                gemv_q4k_h(&act_h, vw, &self.batch_gemv_v, dw_v.n, dw_v.k_blocks);
                for r in 0..m {
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            self.batch_v_seqs[r].device_pointer().cu_deviceptr(),
                            self.batch_gemv_v.device_pointer().cu_deviceptr() + (r * kv_dim * 4) as u64,
                            kv_dim, stream,
                        ).expect("batch inner: v_seqs Q4K slice");
                    }
                }
            } else if let Some(dw_v) = self.device_weights_q6k.get(vw.as_str()) {
                // Q6K V: sequential per-seq GEMVs using graph-safe submit_d2d_arc.
                // batch_q_seqs[r] ([q_dim=hidden] elements) is used as the normed-row
                // input temp; batch_v_seqs[r] receives the output directly.
                for r in 0..m {
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            self.batch_q_seqs[r].device_pointer().cu_deviceptr(),
                            act_h.device_pointer().cu_deviceptr() + (r * k_h * 4) as u64,
                            k_h, stream,
                        ).expect("batch inner: Q6K V normed row copy");
                    }
                    if let Some(ref oxide_q6k) = self.oxide_q6k_gemv {
                        oxide_q6k.submit_d2d_arc_on(
                            &self.batch_q_seqs[r], &dw_v.tensor,
                            &self.batch_v_seqs[r], dw_v.n, dw_v.k_blocks,
                            cu_stream,
                        ).expect("batch inner: Q6K V GEMV");
                    } else {
                        panic!("batch inner: oxide_q6k_gemv required for Q6K V path");
                    }
                }
            } else {
                panic!("batch inner: v_proj {} not in MXFP4, Q4K, or Q6K maps", vw);
            }

            // Q projection: MXFP4 tensor-core or Q4K stream-K.
            if has_mxfp4_q && h_act_quantized {
                let mxfp4   = self.mxfp4_gemm.as_ref().unwrap();
                let packed   = self.mxfp4_act_packed.as_ref().unwrap();
                let scales   = self.mxfp4_act_scales.as_ref().unwrap();
                let dw_q_mx = self.device_weights_mxfp4_tiled.get(qw.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_q_mx.packed, &dw_q_mx.tiled_scales,
                    &self.batch_gemv_q,
                    mxfp4_m_pad, dw_q_mx.n as u32, (dw_q_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 Q L{layer}: {e}"));
            } else if let Some(dq) = dw_q.as_ref() {
                gemv_q4k_h(&act_h, qw, &self.batch_gemv_q, dq.n, dq.k_blocks);
            } else {
                panic!("batch inner: q_proj {qw} not in MXFP4 or Q4K");
            }

            // K projection: MXFP4 tensor-core or Q4K stream-K.
            if has_mxfp4_k && h_act_quantized {
                let mxfp4   = self.mxfp4_gemm.as_ref().unwrap();
                let packed   = self.mxfp4_act_packed.as_ref().unwrap();
                let scales   = self.mxfp4_act_scales.as_ref().unwrap();
                let dw_k_mx = self.device_weights_mxfp4_tiled.get(kw.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_k_mx.packed, &dw_k_mx.tiled_scales,
                    &self.batch_gemv_k,
                    mxfp4_m_pad, dw_k_mx.n as u32, (dw_k_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 K L{layer}: {e}"));
            } else if let Some(dk) = dw_k.as_ref() {
                gemv_q4k_h(&act_h, kw, &self.batch_gemv_k, dk.n, dk.k_blocks);
            } else {
                panic!("batch inner: k_proj {kw} not in MXFP4 or Q4K");
            }

            // -- Per-sequence D->D slice, RoPE, and stable copy -----------------
            for r in 0..m {
                // Slice Q and K rows from batch outputs into per-seq tensors.
                // (batch_q_seqs[r] may have been overwritten by Q6K V path above -- OK,
                //  we overwrite it here with the correct Q slice.)
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_q_seqs[r].device_pointer().cu_deviceptr(),
                        self.batch_gemv_q.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                        q_dim, stream,
                    ).expect("batch inner: q_seqs slice");
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_k_seqs[r].device_pointer().cu_deviceptr(),
                        self.batch_gemv_k.device_pointer().cu_deviceptr() + (r * kv_dim * 4) as u64,
                        kv_dim, stream,
                    ).expect("batch inner: k_seqs slice");
                }

                // Apply RoPE to Q and K for sequence r. Graph-stable path uses
                // the one-row cos/sin buffers gathered once per step (position
                // read from d_positions at execution time); all sequences share
                // the same position (synchronised batch -- graph gate).
                let q_rope_tmp = if rope_graph {
                    self.cutile_rope
                        .forward_device_scratch_graph(
                            &self.batch_q_seqs[r], n_heads, head_dim,
                            &self.d_rope_cos_row, &self.d_rope_sin_row,
                            &mut self.q_rope_scratch.borrow_mut(), stream,
                        )
                        .expect("batch inner: Q RoPE (graph rows)")
                } else {
                    let pos_r = [positions[r]];
                    self.cutile_rope
                        .forward_device_scratch(
                            &self.batch_q_seqs[r], &pos_r, 1, n_heads, head_dim,
                            cfg.rope_theta, &mut self.q_rope_scratch.borrow_mut(), stream,
                        )
                        .expect("batch inner: Q RoPE")
                };
                let k_rope_tmp = if rope_graph {
                    self.cutile_rope
                        .forward_device_scratch_graph(
                            &self.batch_k_seqs[r], n_kv_heads, head_dim,
                            &self.d_rope_cos_row, &self.d_rope_sin_row,
                            &mut self.k_rope_scratch.borrow_mut(), stream,
                        )
                        .expect("batch inner: K RoPE (graph rows)")
                } else {
                    let pos_r = [positions[r]];
                    self.cutile_rope
                        .forward_device_scratch(
                            &self.batch_k_seqs[r], &pos_r, 1, n_kv_heads, head_dim,
                            cfg.rope_theta, &mut self.k_rope_scratch.borrow_mut(), stream,
                        )
                        .expect("batch inner: K RoPE")
                };

                // D->D copy to stable pre-allocated tensors (fixed device ptrs for graph).
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_q_rot[r].device_pointer().cu_deviceptr(),
                        q_rope_tmp.device_pointer().cu_deviceptr(),
                        q_dim, stream,
                    ).expect("batch inner: q_rot stable copy");
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_k_rot[r].device_pointer().cu_deviceptr(),
                        k_rope_tmp.device_pointer().cu_deviceptr(),
                        kv_dim, stream,
                    ).expect("batch inner: k_rot stable copy");
                }
            }

            // -- KV scatter-write: M sequential single-seq writes (graph-stable) --
            // The per-sequence KV cache device pointers are baked into the scatter-write
            // kernel nodes at capture time. They remain valid as long as the SAME KV
            // cache objects are used (checked by the pointer validation in the outer wrapper).
            for r in 0..m {
                let (k_cache, v_cache) = kv_caches[r].kv_device_tensors(layer);
                self.kv_scatter_write
                    .copy_with_device_offset(
                        self.batch_k_rot[r].device_pointer().cu_deviceptr(),
                        k_cache.device_pointer().cu_deviceptr(),
                        write_off_ptr,
                        kv_dim_u32,
                        cu_stream,
                    )
                    .expect("batch inner: scatter K seq");
                self.kv_scatter_write
                    .copy_with_device_offset(
                        self.batch_v_seqs[r].device_pointer().cu_deviceptr(),
                        v_cache.device_pointer().cu_deviceptr(),
                        write_off_ptr,
                        kv_dim_u32,
                        cu_stream,
                    )
                    .expect("batch inner: scatter V seq");
                // mark_host_stale() is a host-side call -- handled by outer wrapper
                // after graph launch to avoid advancing seq_len 32x during capture.
            }

            // -- Attention: batch (single kernel) or sequential fallback -----------
            let kv_len_after = kv_caches[0].seq_len() + 1; // same for all (synchronized)
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            // Concatenate Q-rot into batch_oproj_in (also serves as Q batch buffer).
            for r in 0..m {
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_oproj_in.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                        self.batch_q_rot[r].device_pointer().cu_deviceptr(),
                        q_dim, stream,
                    ).expect("batch inner: q_rot to oproj_in");
                }
            }

            // K/V cache base pointers for this layer (filled before graph capture,
            // stored in batch_attn_all_{k,v}_ptrs at offset layer * m * 8).
            let layer_k_ptrs_dev = self.batch_attn_all_k_ptrs + (layer * m * 8) as u64;
            let layer_v_ptrs_dev = self.batch_attn_all_v_ptrs + (layer * m * 8) as u64;

            let used_batch_attn = if let Some(ref oda) = self.oxide_decode_attn {
                if oda.has_batch() && self.batch_attn_all_k_ptrs != 0 {
                    // Q is already concatenated in batch_oproj_in at stride q_dim.
                    // Reuse batch_gemv_q as the flat output buffer [m * q_dim].
                    let batch_q_flat = &self.batch_oproj_in;
                    let batch_out_flat = &self.batch_gemv_q;
                    oda.forward_batch_graph_d2d(
                        batch_out_flat, batch_q_flat,
                        layer_k_ptrs_dev,
                        layer_v_ptrs_dev,
                        self.batch_attn_kv_lens_dev,
                        m, n_heads, n_kv_heads, head_dim, scale,
                    ).expect("batch inner: batch attention");
                    // Scatter batch output into stable per-seq attn_out[r] tensors.
                    for r in 0..m {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                self.batch_attn_out[r].device_pointer().cu_deviceptr(),
                                batch_out_flat.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                                q_dim, stream,
                            ).expect("batch inner: scatter attn_out");
                        }
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if !used_batch_attn {
                for r in 0..m {
                    let (k_full, v_full) = kv_caches[r].kv_device_tensors(layer);
                    if let Some(ref oda) = self.oxide_decode_attn {
                        let attn_out = &self.batch_attn_out[r];
                        oda.forward_kpar_d2d(
                            attn_out, &self.batch_q_rot[r], &k_full, &v_full,
                            &self.d_kv_len, n_heads, n_kv_heads, head_dim, scale,
                        ).expect("batch inner: kpar attention");
                    } else {
                        let attn_out = &self.batch_attn_out[r];
                        let _ = self.flash_attn
                            .decode_attention_device_scratch(
                                &self.batch_q_rot[r], &k_full, &v_full,
                                kv_len_after, n_heads, n_kv_heads, head_dim, scale,
                                stream, &mut self.attn_out_scratch.borrow_mut(),
                            )
                            .expect("batch inner: cutile attention");
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                attn_out.device_pointer().cu_deviceptr(),
                                self.attn_out_scratch.borrow().as_ref().unwrap().device_pointer().cu_deviceptr(),
                                q_dim, stream,
                            ).expect("batch inner: attn_out copy");
                        }
                    }
                }
            }

            // Note: advance_host_seq_len is a host-side call and is NOT captured in the
            // CUDA graph. It is called ONCE per step by forward_batch_decode_with_graph
            // after the graph launch, not per-layer here.

            // -- Concatenate attn outputs -> batch_oproj_in ---------------------
            for r in 0..m {
                unsafe {
                    cuda_core::memcpy_dtod_async::<f32>(
                        self.batch_oproj_in.device_pointer().cu_deviceptr() + (r * self.q_pad * 4) as u64,
                        self.batch_attn_out[r].device_pointer().cu_deviceptr(),
                        q_dim, stream,
                    ).expect("batch inner: attn_out concat");
                }
            }

            // -- O-projection dispatch -----------------------------------------
            let dw_o = self.device_weights_q4k.get(ow.as_str());
            let k_o = if let Some(dw) = dw_o.as_ref() {
                dw.k_blocks * 256
            } else if let Some(dw_mx) = self.device_weights_mxfp4_tiled.get(ow.as_str()) {
                dw_mx.k
            } else {
                q_dim // fallback: q_dim is the input width for o_proj
            };
            // MXFP4 path: quantize batch_oproj_in [mb * q_pad] (rows 0..m valid),
            // then tensor-core GEMM. Requires q_pad == k_o (no stride gap); if not,
            // fall back to Q4K compact path.
            let use_mxfp4_o = mxfp4_active
                && self.q_pad == k_o   // contiguous rows (no stride gap)
                && self.device_weights_mxfp4_tiled.contains_key(ow.as_str());
            if use_mxfp4_o {
                let mxfp4  = self.mxfp4_gemm.as_ref().unwrap();
                let packed  = self.mxfp4_act_packed.as_ref().unwrap();
                let scales  = self.mxfp4_act_scales.as_ref().unwrap();
                let k_o_pad64 = ((k_o + 63) & !63) as u32;
                // batch_oproj_in is [mb * q_pad] = [16 * q_dim]; rows m..15 are
                // stale. They never reach the GEMM output consumers, but the
                // shared tile-max scale WOULD be poisoned by them without m_valid.
                mxfp4.quantize_activation(
                    &self.batch_oproj_in, packed, scales,
                    mxfp4_m_pad, k_o_pad64, m as u32, cu_stream,
                ).unwrap_or_else(|e| panic!("mxfp4 O quant L{layer}: {e}"));
                let dw_o_mx = self.device_weights_mxfp4_tiled.get(ow.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_o_mx.packed, &dw_o_mx.tiled_scales,
                    &self.batch_gemv_o,
                    mxfp4_m_pad, dw_o_mx.n as u32, (dw_o_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 O L{layer}: {e}"));
            } else {
                // Q4K path (compact if q_pad != k_o).
                let act_o = if self.q_pad == k_o {
                    self.batch_oproj_in.clone()
                } else {
                    let compact = self.batch_gemv_q.clone();
                    for r in 0..m {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                compact.device_pointer().cu_deviceptr() + (r * k_o * 4) as u64,
                                self.batch_oproj_in.device_pointer().cu_deviceptr() + (r * self.q_pad * 4) as u64,
                                k_o, stream,
                            ).expect("batch inner: oproj compact");
                        }
                    }
                    compact
                };
                if let Some(dw) = dw_o.as_ref() {
                    gemv_q4k_h(&act_o, ow, &self.batch_gemv_o, dw.n, dw.k_blocks);
                } else {
                    panic!("batch inner: o_proj {ow} not in MXFP4 or Q4K");
                }
            }

            // -- Post-attention residual add ------------------------------------
            hidden_dev = self.add_device_scratch(
                &hidden_dev, &self.batch_gemv_o, m * self.h_pad,
                &mut self.batch_post_attn_add_scratch.borrow_mut(),
            );

            // -- Post-attention RMSNorm ----------------------------------------
            let post_ln_dev = self.norm_weights.get(post_ln).expect("post_layernorm");
            let normed2 = self.cutile_rmsnorm
                .forward_device_fused_scratch(
                    stream, hidden_dev.clone(), post_ln_dev,
                    m, h, self.h_pad, cfg.layer_norm_eps,
                    &mut self.batch_post_attn_norm_scratch.borrow_mut(),
                )
                .expect("batch post-attn RMSNorm");

            // Compact normed2 for gate/up GEMV (same k as Q/K/V).
            let dw_gate = self.device_weights_q4k.get(gate_w.as_str());
            let k_g = if let Some(dw) = dw_gate.as_ref() {
                dw.k_blocks * 256
            } else if let Some(dw_mx) = self.device_weights_mxfp4_tiled.get(gate_w.as_str()) {
                dw_mx.k
            } else {
                h // fallback: gate uses hidden_size input
            };
            let act_g = if self.h_pad == k_g {
                normed2.clone()
            } else {
                let compact = self.batch_normed2.clone();
                for r in 0..m {
                    unsafe {
                        cuda_core::memcpy_dtod_async::<f32>(
                            compact.device_pointer().cu_deviceptr() + (r * k_g * 4) as u64,
                            normed2.device_pointer().cu_deviceptr() + (r * self.h_pad * 4) as u64,
                            k_g, stream,
                        ).expect("batch inner: normed2 compact");
                    }
                }
                compact
            };

            // -- gate/up dispatch -----------------------------------------------
            let dw_up = self.device_weights_q4k.get(up_w.as_str());
            let use_mxfp4_gateup = mxfp4_active
                && self.device_weights_mxfp4_tiled.contains_key(gate_w.as_str())
                && self.device_weights_mxfp4_tiled.contains_key(up_w.as_str());
            if use_mxfp4_gateup {
                // Ensure batch_normed2 has valid data for m_pad=16 rows.
                // When h_pad == k_g, act_g is normed2 (m rows only); copy into batch_normed2.
                // When h_pad != k_g, act_g IS batch_normed2 (already compacted).
                let act_g_ptr  = act_g.device_pointer().cu_deviceptr();
                let norm2_ptr  = self.batch_normed2.device_pointer().cu_deviceptr();
                if act_g_ptr != norm2_ptr {
                    for r in 0..m {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                norm2_ptr + (r * k_g * 4) as u64,
                                act_g_ptr + (r * k_g * 4) as u64,
                                k_g, stream,
                            ).expect("mxfp4 gateup: normed2 -> batch_normed2");
                        }
                    }
                }
                let mxfp4  = self.mxfp4_gemm.as_ref().unwrap();
                let packed  = self.mxfp4_act_packed.as_ref().unwrap();
                let scales  = self.mxfp4_act_scales.as_ref().unwrap();
                let k_g_pad64 = ((k_g + 63) & !63) as u32;
                mxfp4.quantize_activation(
                    &self.batch_normed2, packed, scales,
                    mxfp4_m_pad, k_g_pad64, m as u32, cu_stream,
                ).unwrap_or_else(|e| panic!("mxfp4 gateup quant L{layer}: {e}"));
                let dw_gate_mx = self.device_weights_mxfp4_tiled.get(gate_w.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_gate_mx.packed, &dw_gate_mx.tiled_scales,
                    &self.batch_gemv_gate,
                    mxfp4_m_pad, dw_gate_mx.n as u32, (dw_gate_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 gate L{layer}: {e}"));
                let dw_up_mx = self.device_weights_mxfp4_tiled.get(up_w.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_up_mx.packed, &dw_up_mx.tiled_scales,
                    &self.batch_gemv_up,
                    mxfp4_m_pad, dw_up_mx.n as u32, (dw_up_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 up L{layer}: {e}"));
            } else if let (Some(dg), Some(du)) = (dw_gate.as_ref(), dw_up.as_ref()) {
                gemv_q4k_h(&act_g, gate_w, &self.batch_gemv_gate, dg.n, dg.k_blocks);
                gemv_q4k_h(&act_g, up_w,   &self.batch_gemv_up,   du.n, du.k_blocks);
            } else {
                panic!("batch inner: gate/up proj not in MXFP4 or Q4K for layer {layer}");
            }

            // -- SiLU(gate) x up -----------------------------------------------
            let silu_out = self.cutile_elementwise
                .silu_mul_device_scratch(
                    &self.batch_gemv_gate, &self.batch_gemv_up, m, inter,
                    &mut self.batch_silu_out_scratch.borrow_mut(),
                    &stream,
                )
                .expect("batch inner: SiLU-mul");

            // -- Down projection dispatch (inter -> h): MXFP4 / Q4K / Q6K ------
            // silu_out points to batch_silu_out_scratch [mb * inter_rounded] (mb=16).
            // Rows 0..m have valid gate*up data; rows m..mb are stale (unused).
            let use_mxfp4_down = mxfp4_active
                && self.device_weights_mxfp4_tiled.contains_key(down_w.as_str());
            if use_mxfp4_down {
                let mxfp4  = self.mxfp4_gemm.as_ref().unwrap();
                let packed  = self.mxfp4_act_packed.as_ref().unwrap();
                let scales  = self.mxfp4_act_scales.as_ref().unwrap();
                // inter must be multiple of 64 (holds for all standard LLaMA/Qwen models).
                let inter_pad64 = ((inter + 63) & !63) as u32;
                // silu_out holds only m*inter elements (m < 16): m_valid both
                // masks the scale and stops the kernel from reading past the
                // end of the tensor (the old all-NaN-at-M=4 OOB read).
                mxfp4.quantize_activation(
                    &silu_out, packed, scales,
                    mxfp4_m_pad, inter_pad64, m as u32, cu_stream,
                ).unwrap_or_else(|e| panic!("mxfp4 down quant L{layer}: {e}"));
                let dw_down_mx = self.device_weights_mxfp4_tiled.get(down_w.as_str()).unwrap();
                mxfp4.forward(
                    packed, scales,
                    &dw_down_mx.packed, &dw_down_mx.tiled_scales,
                    &self.batch_gemv_down,
                    mxfp4_m_pad, dw_down_mx.n as u32, (dw_down_mx.k / 64) as u32,
                ).unwrap_or_else(|e| panic!("mxfp4 down L{layer}: {e}"));
            } else if let Some(dw_down) = self.device_weights_q4k.get(down_w.as_str()) {
                // Q4K down_proj: use batch GEMV.
                let k_down = dw_down.k_blocks * 256;
                let act_down = if silu_out.size() == m * k_down {
                    silu_out
                } else if let Some(ref compact_buf) = self.batch_down_compact {
                    for r in 0..m {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                compact_buf.device_pointer().cu_deviceptr() + (r * k_down * 4) as u64,
                                silu_out.device_pointer().cu_deviceptr() + (r * inter * 4) as u64,
                                k_down, stream,
                            ).expect("batch inner: down Q4K compact");
                        }
                    }
                    compact_buf.clone()
                } else {
                    silu_out
                };
                let ox_down = oxide.expect("batch inner: oxide Q4K required for Q4K down path");
                if ox_down.has_streamk_batch() {
                    ox_down.forward_streamk_batch_d2d_into(
                        &act_down, &dw_down.tensor, m, dw_down.n, dw_down.k_blocks,
                        &self.batch_q8_qs_i, &self.batch_q8_ds_i, &self.batch_gemv_down,
                    ).expect("batch inner: down Q4K stream-K GEMM");
                } else if ox_down.has_mma_batch() {
                    ox_down.forward_mma_batch_d2d_into(
                        &act_down, &dw_down.tensor, m, dw_down.n, dw_down.k_blocks,
                        &self.batch_q8_qs_i, &self.batch_q8_ds_i, &self.batch_gemv_down,
                    ).expect("batch inner: down Q4K MMA GEMM");
                } else {
                    ox_down.forward_sm120_batch_d2d_into(
                        &act_down, &dw_down.tensor, m, dw_down.n, dw_down.k_blocks,
                        &self.batch_q8_qs_i, &self.batch_q8_ds_i, &self.batch_gemv_down,
                    ).expect("batch inner: down Q4K GEMV");
                }
            } else if let Some(dw_down) = self.device_weights_q6k.get(down_w.as_str()) {
                // Q6K down_proj (Q4_K_M models).
                let h_out  = dw_down.n;
                let k_down = dw_down.k_blocks * Q6K_QK;
                let oxide_q6k = self.oxide_q6k_gemv.as_ref()
                    .expect("batch inner: oxide_q6k_gemv required for Q6K down path");
                if m > 1 && oxide_q6k.has_sm120_batch() && silu_out.size() == m * k_down {
                    oxide_q6k.forward_sm120_batch_d2d_into(
                        &silu_out, &dw_down.tensor, &self.batch_gemv_down,
                        h_out, dw_down.k_blocks, m, cu_stream,
                    ).expect("batch inner: Q6K batch down_proj GEMV");
                } else {
                    for r in 0..m {
                        let copy_len = k_down.min(inter);
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                self.batch_silu_row_bufs[r].device_pointer().cu_deviceptr(),
                                silu_out.device_pointer().cu_deviceptr() + (r * inter * 4) as u64,
                                copy_len, stream,
                            ).expect("batch inner: Q6K silu row D->D");
                        }
                        oxide_q6k.submit_d2d_arc_on(
                            &self.batch_silu_row_bufs[r], &dw_down.tensor,
                            &self.batch_q_seqs[0], h_out, dw_down.k_blocks,
                            cu_stream,
                        ).expect("batch inner: Q6K down_proj GEMV");
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                self.batch_gemv_down.device_pointer().cu_deviceptr() + (r * h_out * 4) as u64,
                                self.batch_q_seqs[0].device_pointer().cu_deviceptr(),
                                h_out, stream,
                            ).expect("batch inner: Q6K down_proj concat");
                        }
                    }
                }
            } else {
                panic!("batch inner: down_proj weight {down_w} not found in Q4K or Q6K maps");
            }

            // -- FFN residual add -----------------------------------------------
            hidden_dev = self.add_device_scratch(
                &hidden_dev, &self.batch_gemv_down, m * self.h_pad,
                &mut self.batch_ffn_add_scratch.borrow_mut(),
            );
        }

        // Layer loop done -- `hidden_dev` is the final residual add output.
        // Store it into batch_hidden so the caller can read it after graph launch.
        // D->D copy only if it's not already at batch_hidden's address.
        let dst_ptr = self.batch_hidden.device_pointer().cu_deviceptr();
        let src_ptr = hidden_dev.device_pointer().cu_deviceptr();
        if src_ptr != dst_ptr {
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    dst_ptr, src_ptr, m * self.h_pad, stream,
                ).expect("batch inner: final hidden copy");
            }
        }
        // batch_hidden now holds the layer-loop output; caller applies final norm + lm_head.
    }

    /// Inner body for the MXFP4 batch CUDA graph: MXFP4 tensor-core GEMM unconditionally
    /// for Q/K/O/gate/up projections. V stays Q4K, down stays Q6K.
    ///
    /// Eligibility must be verified by the caller (mxfp4_eligible_for_batch_graph()).
    /// Called exclusively between cuStreamBeginCapture and cuStreamEndCapture to bake
    /// MXFP4 kernel launches into the CUDA graph with no runtime branching.
    fn forward_batch_mxfp4_inner_graph(
        &self,
        m:         usize,
        positions: &[u32],
        kv_caches: &mut [CutileKvCache],
    ) {
        // Delegates to the shared inner graph with use_mxfp4=true.
        // At capture time, mxfp4_active = true, so all Q/K/O/gate/up dispatch uses
        // MXFP4 tensor cores. The captured graph nodes are MXFP4-only (no dp4a fallback).
        self.forward_batch_decode_inner_graph(m, positions, kv_caches, true);
    }

    /// MXFP4 batch CUDA graph: captures forward_batch_mxfp4_inner_graph once,
    /// replays on subsequent calls. Uses a separate graph cache from the dp4a path.
    ///
    /// Returns None if MXFP4 is not eligible (caller falls back to dp4a graph).
    fn forward_batch_mxfp4_with_graph(
        &self,
        embeddings: &[Vec<f32>],
        positions:  &[u32],
        qweights:   &QuantizedWeights,
        kv_caches:  &mut [CutileKvCache],
    ) -> Option<Vec<Vec<f32>>> {
        if !self.mxfp4_eligible_for_batch_graph() {
            return None;
        }
        let m = embeddings.len();
        let cfg    = &self.config;
        let h      = cfg.hidden_size;
        let vocab  = cfg.vocab_size;
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();

        let kv_write_off = kv_caches[0].seq_len() as i32;
        let kv_len_bits: f32 = f32::from_bits((kv_write_off + 1) as u32);
        let pos_bits: f32    = f32::from_bits(positions[0]);

        // H->D: upload M embeddings and decode scalars.
        for (r, emb) in embeddings.iter().enumerate() {
            let src: Vec<f32> = if self.h_pad == h {
                emb.to_vec()
            } else {
                let mut v = vec![0.0f32; self.h_pad];
                v[..h].copy_from_slice(&emb[..h]);
                v
            };
            unsafe {
                cuda_core::memcpy_htod_async::<f32>(
                    self.d_embed_batch[r].device_pointer().cu_deviceptr(),
                    src.as_ptr(), self.h_pad, stream,
                ).expect("mxfp4 batch graph: d_embed_batch H->D");
            }
        }
        for (r, _emb) in embeddings.iter().enumerate() {
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    self.batch_hidden.device_pointer().cu_deviceptr() + (r * self.h_pad * 4) as u64,
                    self.d_embed_batch[r].device_pointer().cu_deviceptr(),
                    self.h_pad, stream,
                ).expect("mxfp4 batch graph: embedding concat D->D");
            }
        }
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_kv_len.device_pointer().cu_deviceptr(),
                &kv_len_bits as *const f32, 1, stream,
            ).expect("mxfp4 batch graph: d_kv_len H->D");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("mxfp4 batch graph: d_positions H->D");
            cuda_core::memcpy_htod_async::<i32>(
                self.d_kv_write_offset.device_pointer().cu_deviceptr(),
                &kv_write_off as *const i32, 1, stream,
            ).expect("mxfp4 batch graph: d_kv_write_offset H->D");
        }
        if self.batch_attn_kv_lens_dev != 0 {
            let kv_len_after = kv_write_off + 1;
            let kv_lens: Vec<i32> = (0..m).map(|_| kv_len_after).collect();
            unsafe {
                cuda_core::sys::cuMemcpyHtoDAsync_v2(
                    self.batch_attn_kv_lens_dev,
                    kv_lens.as_ptr() as *const std::ffi::c_void,
                    m * 4,
                    cu_stream,
                );
            }
        }

        let mut graph_cache = self.cached_mxfp4_batch_graph.borrow_mut();

        // Validate KV-cache device pointers before replay.
        {
            let need_invalidate = match *graph_cache {
                Some(ref dg) => {
                    if dg.m != m {
                        eprintln!("[mxfp4_graph] batch size changed ({} -> {m}) -- re-capturing", dg.m);
                        true
                    } else {
                        !(0..cfg.num_layers).all(|l| {
                            (0..m).all(|r| {
                                let (k, v) = kv_caches[r].kv_device_tensors(l);
                                k.device_pointer().cu_deviceptr() == dg.captured_k_ptrs[l][r]
                                    && v.device_pointer().cu_deviceptr() == dg.captured_v_ptrs[l][r]
                            })
                        })
                    }
                }
                None => false,
            };
            if need_invalidate {
                eprintln!("[mxfp4_graph] KV cache ptrs changed -- re-capturing");
                *graph_cache = None;
            }
        }

        if let Some(ref decode_graph) = *graph_cache {
            // Subsequent steps: plain replay. kv_lens for the batch attention
            // are read from batch_attn_kv_lens_dev (H->D above), the KV write
            // offset from d_kv_write_offset, and the RoPE cos/sin row is
            // re-gathered from d_positions -- all at execution time.
            unsafe { cuda_core::sys::cuGraphLaunch(decode_graph.cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("mxfp4 batch graph sync"); }
            drop(graph_cache);
        } else {
            // First call: fill K/V ptr arrays, then capture MXFP4 graph.
            if self.batch_attn_all_k_ptrs != 0 {
                for l in 0..cfg.num_layers {
                    let k_ptrs: Vec<u64> = (0..m)
                        .map(|r| kv_caches[r].kv_device_tensors(l).0.device_pointer().cu_deviceptr())
                        .collect();
                    let v_ptrs: Vec<u64> = (0..m)
                        .map(|r| kv_caches[r].kv_device_tensors(l).1.device_pointer().cu_deviceptr())
                        .collect();
                    unsafe {
                        cuda_core::sys::cuMemcpyHtoD_v2(
                            self.batch_attn_all_k_ptrs + (l * m * 8) as u64,
                            k_ptrs.as_ptr() as *const std::ffi::c_void,
                            m * 8,
                        );
                        cuda_core::sys::cuMemcpyHtoD_v2(
                            self.batch_attn_all_v_ptrs + (l * m * 8) as u64,
                            v_ptrs.as_ptr() as *const std::ffi::c_void,
                            m * 8,
                        );
                    }
                }
            }

            // Pre-size batch scratch tensors before capture.
            {
                const SILU_B: usize = 32;
                let inter = cfg.intermediate_size;
                let inter_pad = inter.div_ceil(SILU_B) * SILU_B;
                let h_pad = self.h_pad;
                let alloc_flat = |n: usize| -> Arc<Tensor<f32>> {
                    Arc::new(
                        api::zeros::<f32>(&[n])
                            .sync_on(stream)
                            .expect("mxfp4 batch_graph: scratch pre-size"),
                    )
                };
                self.batch_pre_attn_norm_scratch.borrow_mut().out_t  = Some(alloc_flat(m * h_pad));
                self.batch_post_attn_norm_scratch.borrow_mut().out_t = Some(alloc_flat(m * h_pad));
                self.batch_final_norm_scratch.borrow_mut().out_t     = Some(alloc_flat(m * h_pad));
                *self.batch_silu_out_scratch.borrow_mut() = Some(alloc_flat(m * inter_pad));
            }

            let relaxed = cuda_core::sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;
            unsafe { stream.begin_capture(relaxed).expect("mxfp4 batch graph: begin_capture"); }

            // Capture MXFP4 path (use_mxfp4=true). All Q/K/O/gate/up projections use
            // mxfp4_quantize_activation_tiled + mxf4_batch_gemm_tiled kernel nodes.
            self.forward_batch_mxfp4_inner_graph(m, positions, kv_caches);

            let cu_graph = unsafe { stream.end_capture().expect("mxfp4 batch graph: end_capture") };

            let mut cu_exec_uninit = std::mem::MaybeUninit::<cuda_core::sys::CUgraphExec>::uninit();
            unsafe {
                cuda_core::sys::cuGraphInstantiateWithFlags(cu_exec_uninit.as_mut_ptr(), cu_graph, 0);
            }
            let cu_exec = unsafe { cu_exec_uninit.assume_init() };

            let captured_k_ptrs: Vec<Vec<u64>> = (0..cfg.num_layers)
                .map(|l| (0..m).map(|r| kv_caches[r].kv_device_tensors(l).0.device_pointer().cu_deviceptr()).collect())
                .collect();
            let captured_v_ptrs: Vec<Vec<u64>> = (0..cfg.num_layers)
                .map(|l| (0..m).map(|r| kv_caches[r].kv_device_tensors(l).1.device_pointer().cu_deviceptr()).collect())
                .collect();

            *graph_cache = Some(BatchDecodeGraph {
                cu_exec,
                cu_graph,
                device: stream.device().clone(),
                m,
                captured_k_ptrs,
                captured_v_ptrs,
            });

            unsafe { cuda_core::sys::cuGraphLaunch(cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("mxfp4 batch graph: first sync"); }
            drop(graph_cache);
        }

        // Advance KV cache host-side seq_len once per step.
        for kvc in kv_caches.iter_mut().take(m) {
            kvc.advance_host_seq_len(1);
            kvc.mark_host_stale();
        }

        // Outside graph: final norm + lm_head + D->H.
        let final_ln_dev = self.norm_weights.get("model.norm.weight")
            .expect("model.norm.weight missing");
        let last_dev = self.cutile_rmsnorm
            .forward_device_fused_scratch(
                stream,
                self.batch_hidden.clone(),
                final_ln_dev,
                m, h, self.h_pad, cfg.layer_norm_eps,
                &mut self.batch_final_norm_scratch.borrow_mut(),
            )
            .expect("mxfp4 batch graph: final RMSNorm");

        let logits_dev = self.gemm_d2d(&last_dev, "lm_head.weight", m);
        let logits_flat = self.download_device_tensor(&logits_dev);
        Some(logits_flat.chunks(vocab).map(|c| c.to_vec()).collect())
    }

    /// CUDA-graph-accelerated batch decode: M synchronised sequences, 1 token each.
    ///
    /// Captures on the first call, replays on subsequent calls.
    /// Before each graph launch the embeddings and decode scalars are written H->D.
    /// After launch, final norm + lm_head + D->H are run outside the graph.
    ///
    /// Falls back to `forward_batch_decode` (no-graph) when:
    ///   - Q4K sm120 batch kernel is unavailable
    ///   - M > max_batch_graph
    ///   - KV caches do not have device KV
    fn forward_batch_decode_with_graph(
        &self,
        embeddings: &[Vec<f32>],          // [m][h]
        positions:  &[u32],               // [m] -- must all be equal (synchronised)
        qweights:   &QuantizedWeights,
        kv_caches:  &mut [CutileKvCache],
    ) -> Vec<Vec<f32>> {
        let m = embeddings.len();
        assert!(m >= 2 && m <= self.max_batch_graph,
            "batch graph: m={m} must be in [2, {}]", self.max_batch_graph);

        // Try MXFP4 tensor-core path first: separate graph cache, MXFP4-unconditional
        // inner body. When MXFP4 weights are loaded for all Q4K projections, this path
        // captures a graph with mxfp4_quantize_activation_tiled + mxf4_batch_gemm_tiled
        // nodes (no dp4a fallback nodes). Returns None if ineligible.
        if let Some(result) = self.forward_batch_mxfp4_with_graph(embeddings, positions, qweights, kv_caches) {
            if !result.is_empty() {
                return result;
            }
            // Empty result means MXFP4 graph update failed; fall through to dp4a.
        }
        let cfg    = &self.config;
        let h      = cfg.hidden_size;
        let vocab  = cfg.vocab_size;
        let stream = self.matmul.stream();
        let cu_stream = stream.cu_stream();

        // Write offset (same for all M sequences).
        let kv_write_off = kv_caches[0].seq_len() as i32;
        let kv_len_bits: f32 = f32::from_bits((kv_write_off + 1) as u32);
        let pos_bits: f32    = f32::from_bits(positions[0]);

        // -- H->D: upload M embeddings and decode scalars -----------------------
        for (r, emb) in embeddings.iter().enumerate() {
            let src: Vec<f32> = if self.h_pad == h {
                emb.to_vec()
            } else {
                let mut v = vec![0.0f32; self.h_pad];
                v[..h].copy_from_slice(&emb[..h]);
                v
            };
            unsafe {
                cuda_core::memcpy_htod_async::<f32>(
                    self.d_embed_batch[r].device_pointer().cu_deviceptr(),
                    src.as_ptr(), self.h_pad, stream,
                ).expect("batch graph: d_embed_batch H->D");
            }
        }
        // Concatenate embeddings into batch_hidden (single H->D copy of M rows).
        // Upload each row separately via D->D within the stream (already uploaded above).
        for (r, _emb) in embeddings.iter().enumerate() {
            unsafe {
                cuda_core::memcpy_dtod_async::<f32>(
                    self.batch_hidden.device_pointer().cu_deviceptr() + (r * self.h_pad * 4) as u64,
                    self.d_embed_batch[r].device_pointer().cu_deviceptr(),
                    self.h_pad, stream,
                ).expect("batch graph: embedding concat D->D");
            }
        }

        // Update decode scalars (same for all M sequences).
        unsafe {
            cuda_core::memcpy_htod_async::<f32>(
                self.d_kv_len.device_pointer().cu_deviceptr(),
                &kv_len_bits as *const f32, 1, stream,
            ).expect("batch graph: d_kv_len H->D");
            cuda_core::memcpy_htod_async::<f32>(
                self.d_positions.device_pointer().cu_deviceptr(),
                &pos_bits as *const f32, 1, stream,
            ).expect("batch graph: d_positions H->D");
            cuda_core::memcpy_htod_async::<i32>(
                self.d_kv_write_offset.device_pointer().cu_deviceptr(),
                &kv_write_off as *const i32, 1, stream,
            ).expect("batch graph: d_kv_write_offset H->D");
        }

        // Update kv_len array for oxide_decode_attn_batch kernel.
        // All M sequences are synchronized, so all entries are the same value.
        if self.batch_attn_kv_lens_dev != 0 {
            let kv_len_after = kv_write_off + 1;
            let kv_lens: Vec<i32> = (0..m).map(|_| kv_len_after).collect();
            unsafe {
                cuda_core::sys::cuMemcpyHtoDAsync_v2(
                    self.batch_attn_kv_lens_dev,
                    kv_lens.as_ptr() as *const std::ffi::c_void,
                    m * 4,
                    cu_stream,
                );
            }
        }

        let mut graph_cache = self.cached_batch_graph.borrow_mut();

        // Validate KV-cache device pointers before replay.
        // If they differ (fresh-per-call bench: new cache objects each step),
        // invalidate the cached graph so the else-branch re-captures.
        {
            let need_invalidate = match *graph_cache {
                Some(ref dg) => {
                    if dg.m != m {
                        // Batch size changed: re-capture for new m.
                        eprintln!("[batch_graph] batch size changed ({} -> {m}) -- re-capturing", dg.m);
                        true
                    } else {
                        !(0..cfg.num_layers).all(|l| {
                            (0..m).all(|r| {
                                let (k, v) = kv_caches[r].kv_device_tensors(l);
                                k.device_pointer().cu_deviceptr() == dg.captured_k_ptrs[l][r]
                                    && v.device_pointer().cu_deviceptr() == dg.captured_v_ptrs[l][r]
                            })
                        })
                    }
                }
                None => false,
            };
            if need_invalidate {
                eprintln!("[batch_graph] KV cache ptrs changed -- re-capturing");
                *graph_cache = None;
            }
        }

        if let Some(ref decode_graph) = *graph_cache {
            // -- Subsequent steps: plain replay --------------------------------
            // Per-step scalars are all read from device memory at execution
            // time (batch_attn_kv_lens_dev, d_kv_write_offset, d_positions).
            unsafe { cuda_core::sys::cuGraphLaunch(decode_graph.cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("batch graph sync"); }
            drop(graph_cache);
        } else {
            // -- First call: fill per-layer K/V ptr arrays, then capture graph --
            // Upload K/V cache base pointers for all layers to the stable device
            // arrays BEFORE begin_capture.  These pointers are stable (same KV
            // cache objects throughout a generation); they are baked into the
            // captured kernel launch parameters and never updated inside the graph.
            if self.batch_attn_all_k_ptrs != 0 {
                for l in 0..cfg.num_layers {
                    let k_ptrs: Vec<u64> = (0..m)
                        .map(|r| kv_caches[r].kv_device_tensors(l).0.device_pointer().cu_deviceptr())
                        .collect();
                    let v_ptrs: Vec<u64> = (0..m)
                        .map(|r| kv_caches[r].kv_device_tensors(l).1.device_pointer().cu_deviceptr())
                        .collect();
                    unsafe {
                        cuda_core::sys::cuMemcpyHtoD_v2(
                            self.batch_attn_all_k_ptrs + (l * m * 8) as u64,
                            k_ptrs.as_ptr() as *const std::ffi::c_void,
                            m * 8,
                        );
                        cuda_core::sys::cuMemcpyHtoD_v2(
                            self.batch_attn_all_v_ptrs + (l * m * 8) as u64,
                            v_ptrs.as_ptr() as *const std::ffi::c_void,
                            m * 8,
                        );
                    }
                }
            }

            // Pre-size batch scratch tensors for seq_len=m before CUDA graph capture.
            // Scratch was pre-allocated for max_batch at construction; the reshape inside
            // forward_device_fused_scratch requires the correct element count.  Synchronous
            // alloc is safe here because we are not yet in capture mode.
            {
                const SILU_B: usize = 32;
                let inter = cfg.intermediate_size;
                let inter_pad = inter.div_ceil(SILU_B) * SILU_B;
                let h_pad = self.h_pad;
                let alloc_flat = |n: usize| -> Arc<Tensor<f32>> {
                    Arc::new(
                        api::zeros::<f32>(&[n])
                            .sync_on(stream)
                            .expect("batch_graph: scratch pre-size"),
                    )
                };
                self.batch_pre_attn_norm_scratch.borrow_mut().out_t  = Some(alloc_flat(m * h_pad));
                self.batch_post_attn_norm_scratch.borrow_mut().out_t = Some(alloc_flat(m * h_pad));
                self.batch_final_norm_scratch.borrow_mut().out_t     = Some(alloc_flat(m * h_pad));
                *self.batch_silu_out_scratch.borrow_mut() = Some(alloc_flat(m * inter_pad));
            }

            let relaxed = cuda_core::sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED;
            unsafe { stream.begin_capture(relaxed).expect("batch graph: begin_capture"); }

            // Capture dp4a path (use_mxfp4=false). MXFP4 is captured in a separate graph
            // via forward_batch_mxfp4_with_graph when MXFP4 weights are eligible.
            self.forward_batch_decode_inner_graph(m, positions, kv_caches, false);

            let cu_graph = unsafe { stream.end_capture().expect("batch graph: end_capture") };

            let mut cu_exec_uninit = std::mem::MaybeUninit::<cuda_core::sys::CUgraphExec>::uninit();
            unsafe {
                cuda_core::sys::cuGraphInstantiateWithFlags(cu_exec_uninit.as_mut_ptr(), cu_graph, 0);
            }
            let cu_exec = unsafe { cu_exec_uninit.assume_init() };

            // Record KV cache device pointers so replay can detect staleness.
            let captured_k_ptrs: Vec<Vec<u64>> = (0..cfg.num_layers)
                .map(|l| (0..m).map(|r| kv_caches[r].kv_device_tensors(l).0.device_pointer().cu_deviceptr()).collect())
                .collect();
            let captured_v_ptrs: Vec<Vec<u64>> = (0..cfg.num_layers)
                .map(|l| (0..m).map(|r| kv_caches[r].kv_device_tensors(l).1.device_pointer().cu_deviceptr()).collect())
                .collect();

            *graph_cache = Some(BatchDecodeGraph {
                cu_exec,
                cu_graph,
                device: stream.device().clone(),
                m,
                captured_k_ptrs,
                captured_v_ptrs,
            });

            unsafe { cuda_core::sys::cuGraphLaunch(cu_exec, cu_stream); }
            unsafe { stream.synchronize().expect("batch graph: first sync"); }
            drop(graph_cache);
        }

        // Advance KV cache host-side seq_len once per step (NOT inside the inner function
        // to avoid running 32x during graph capture).
        for kvc in kv_caches.iter_mut().take(m) {
            kvc.advance_host_seq_len(1);
            kvc.mark_host_stale();
        }

        // -- Outside graph: final norm + lm_head + D->H ------------------------
        let final_ln_dev = self.norm_weights.get("model.norm.weight")
            .expect("model.norm.weight missing");
        let last_dev = self.cutile_rmsnorm
            .forward_device_fused_scratch(
                stream,
                self.batch_hidden.clone(),
                final_ln_dev,
                m, h, self.h_pad, cfg.layer_norm_eps,
                &mut self.batch_final_norm_scratch.borrow_mut(),
            )
            .expect("batch graph: final RMSNorm");

        // lm_head: [M, H] -> [M, vocab]  -- run outside graph (allocation OK here)
        let logits_dev = self.gemm_d2d(&last_dev, "lm_head.weight", m);
        let logits_flat = self.download_device_tensor(&logits_dev);
        logits_flat.chunks(vocab).map(|c| c.to_vec()).collect()
    }

    /// Batched decode: M requests x 1 token each, processed in a single GPU pass.
    ///
    /// Each request has its own KV cache (`kv_caches[r]`) and current token
    /// (`token_ids[r]`) at position `positions[r]`. Returns M logit vectors
    /// (one per request, length = vocab_size).
    ///
    /// Uses `oxide_decode_attn_batch` for flash attention -- grid = M x n_heads
    /// blocks fills all 170 RTX-5090 SMs for batch >= 5.
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

        // Single request: delegate to the verified single-sequence CUDA-graph
        // path. m=1 never qualifies for the batch graph (requires m >= 2), and
        // the generic no-graph fallback below is slow (per-token allocations)
        // and was never output-verified (the old "M=1 fallback broken" bug).
        // forward() leaves seq_len advancement to its caller, so advance here
        // to preserve forward_batch_decode's stateful contract.
        if m == 1 {
            let logits = self.forward(
                positions[0] as usize,
                &token_ids[..1],
                qweights,
                &mut kv_caches[0],
            );
            kv_caches[0].advance_host_seq_len(1);
            kv_caches[0].mark_host_stale();
            return vec![logits];
        }

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

        // -- 1. Embedding lookup (CPU) ------------------------------------------
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

        // -- Batch CUDA graph path ---------------------------------------------
        // Use the pre-captured CUDA graph when:
        //   (a) All KV caches have device-resident KV (GPU flash attention).
        //   (b) All Q4K sm120 batch kernel is available.
        //   (c) M <= max_batch_graph.
        //   (d) !use_alibi.
        //   (e) All positions are equal (synchronised batch decode).
        //   (f) oxide_decode_attn present -- the captured attention must read
        //       kv_len from device memory; the cuTile flash-attn fallback takes
        //       kv_len as a host launch scalar the graph would bake and replay
        //       stale.
        //   (g) position < RoPE table_max_pos -- the captured RoPE gathers the
        //       cos/sin row by device-side position; beyond the table the
        //       gather would clamp to the last row (wrong rotation).
        // Batch graph only for M >= 2 (M=1 uses the single-seq graph via forward()).
        // All M sequences must have the same position (synchronised batch decode).
        // All projection weights must be Q4K (checked once at model load time).
        let positions_equal = positions.windows(2).all(|w| w[0] == w[1]);
        let kv_all_device  = kv_caches.iter().all(|kvc| kvc.has_device_kv());
        let base_ok = m >= 2
            && m <= self.max_batch_graph
            && kv_all_device
            && !cfg.use_alibi
            && positions_equal
            && self.oxide_decode_attn.is_some()
            && (positions[0] as usize) < self.cutile_rope.table_max_pos();
        // Q4K dp4a/sm120 batch graph: requires oxide Q4K batch kernel.
        let can_q4k_graph = base_ok
            && self.all_proj_q4k
            && self.oxide_q4k_gemv.as_ref().map(|o| o.has_streamk_batch() || o.has_mma_batch() || o.has_sm120_batch()).unwrap_or(false);
        // MXFP4 tensor-core batch graph: Q/K/O/gate/up in MXFP4 tiled; no Q4K batch kernel needed.
        let can_mxfp4_graph = base_ok && self.mxfp4_eligible_for_batch_graph();
        let can_graph = can_q4k_graph || can_mxfp4_graph;

        if can_graph {
            let embeddings: Vec<Vec<f32>> = (0..m)
                .map(|r| hidden[r * h..(r + 1) * h].to_vec())
                .collect();
            let results = self.forward_batch_decode_with_graph(&embeddings, positions, qweights, kv_caches);
            if !results.is_empty() {
                return results;
            }
            // Empty result means graph update failed -- fall through to no-graph path.
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

            // QKV projections [M, H] -> [M, Q_DIM] / [M, KV_DIM]
            // normed_dev is [m, h_pad] from forward_device_fused; row stride = h_pad.
            let q_dev = self.gemm_any_d2d_batch(&normed_dev, q_w_name, m, self.h_pad, SLOT_Q);
            let k_dev = self.gemm_any_d2d_batch(&normed_dev, k_w_name, m, self.h_pad, SLOT_K);
            let v_dev = self.gemm_any_d2d_batch(&normed_dev, v_w_name, m, self.h_pad, SLOT_V);

            // RoPE per-request (each has its own position).
            // Process each request's Q/K slice individually.
            let kv_pad = next_pow2(kv_dim.div_ceil(BK) * BK);
            let mut q_rot_parts: Vec<Arc<cutile::tensor::Tensor<f32>>> = Vec::with_capacity(m);
            for r in 0..m {
                // Extract per-request Q/K/V slices via D->D memcpy into fresh [q_dim] tensors.
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
                        ).expect("q_r D->D");
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
                        ).expect("k_r D->D");
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
                        ).expect("v_r D->D");
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

            // Batch flash attention: one kernel launch, M x n_heads blocks.
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
                    // Use D->D async memcpy via cutile.
                    for (r, q_r) in q_rot_parts.iter().enumerate() {
                        unsafe {
                            cuda_core::memcpy_dtod_async::<f32>(
                                q_batch.device_pointer().cu_deviceptr() + (r * q_dim * 4) as u64,
                                q_r.device_pointer().cu_deviceptr(),
                                q_dim,
                                stream,
                            ).expect("q_batch D->D copy");
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
                            ).expect("batch d_kv_len H->D");
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
                            ).expect("attn_out D->D copy");
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
                        ).expect("batch attn_out D->D");
                    }
                }
                attn_out
            };

            // Advance all KV caches.
            for kvc in kv_caches.iter_mut() {
                kvc.advance_host_seq_len(1);
            }

            // O-projection [M, Q_DIM] -> [M, H]
            let attn_padded = self.attn_out_to_oproj_device_m(&attn_out_dev, q_dim, self.q_pad, m);
            // attn_padded is [m, q_pad] from attn_out_to_oproj_device_m; row stride = q_pad.
            let o_dev = self.gemm_any_d2d_batch(&attn_padded, o_w_name, m, self.q_pad, SLOT_O);

            // Post-attention residual add (fresh alloc -- pre-allocated scratch is M=1 sized).
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

            // SiLU(gate) x up -- use cuTile silu_mul (GPU, no PCIe round-trip, any M).
            let inter_size = m * cfg.intermediate_size;
            let gs_dev = self
                .cutile_elementwise
                .silu_mul_device(&gate_dev, &up_dev, m, cfg.intermediate_size, &stream)
                .expect("batch silu_mul_device failed");
            // Pad inter -> inter_pad if necessary (e.g. Llama-3.1-8B: 14336->16384).
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

        // LM head: [M, H] -> [M, vocab]
        let logits_dev = self.gemm_d2d(&last_dev, "lm_head.weight", m);
        let logits_flat = self.download_device_tensor(&logits_dev);

        // Split into M logit vectors.
        logits_flat.chunks(vocab).map(|c| c.to_vec()).collect()
    }
}

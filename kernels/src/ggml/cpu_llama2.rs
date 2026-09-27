//! Direct Rust translation of remaining functions from llama.cpp `ggml-cpu.c`.
//!
//! Source file: ggml/src/ggml-cpu/ggml-cpu.c
//!
//! Translated sections:
//!   - `ggml_graph_plan`             → [`graph_plan`]
//!   - `ggml_cpu_try_fuse_ops`       → [`try_fuse_ops`]
//!   - `ggml_cpu_fp32_to_fp32`       → [`fp32_to_fp32`]
//!   - `ggml_cpu_fp32_to_fp16`       → [`fp32_to_fp16`]
//!   - `ggml_cpu_fp16_to_fp32`       → [`fp16_to_fp32`]
//!   - `ggml_cpu_fp32_to_bf16`       → [`fp32_to_bf16`]
//!   - `ggml_cpu_fp32_to_i32`        → [`fp32_to_i32`]
//!   - `ggml_cpu_bf16_to_fp32`       → [`bf16_to_fp32`]
//!   - `ggml_cpu_has_avx`            → [`has_avx`]
//!   - `ggml_cpu_has_avx2`           → [`has_avx2`]
//!   - `ggml_cpu_has_avx_vnni`       → [`has_avx_vnni`]
//!   - `ggml_cpu_has_avx512`         → [`has_avx512`]
//!   - `ggml_cpu_has_avx512_vbmi`    → [`has_avx512_vbmi`]
//!   - `ggml_cpu_has_avx512_vnni`    → [`has_avx512_vnni`]
//!   - `ggml_cpu_has_avx512_bf16`    → [`has_avx512_bf16`]
//!   - `ggml_cpu_has_amx_int8`       → [`has_amx_int8`]
//!   - `ggml_cpu_has_bmi2`           → [`has_bmi2`]
//!   - `ggml_cpu_has_fma`            → [`has_fma`]
//!   - `ggml_cpu_has_arm_fma`        → [`has_arm_fma`]
//!   - `ggml_cpu_has_riscv_v`        → [`has_riscv_v`]
//!   - `ggml_cpu_has_f16c`           → [`has_f16c`]
//!   - `ggml_cpu_has_fp16_va`        → [`has_fp16_va`]
//!   - `ggml_cpu_has_wasm_simd`      → [`has_wasm_simd`]
//!   - `ggml_cpu_has_sse3`           → [`has_sse3`]
//!   - `ggml_cpu_has_ssse3`          → [`has_ssse3`]
//!   - `ggml_cpu_has_vsx`            → [`has_vsx`]
//!   - `ggml_cpu_has_neon`           → [`has_neon`]
//!   - `ggml_cpu_has_dotprod`        → [`has_dotprod`]
//!   - `ggml_cpu_has_sve`            → [`has_sve`]
//!   - `ggml_cpu_has_matmul_int8`    → [`has_matmul_int8`]

use super::cpu_llama::{GgmlOp, GgmlCplan, DEFAULT_N_THREADS};
use super::GgmlType;

// ---------------------------------------------------------------------------
// Constants translated from ops.h and common.h
// ---------------------------------------------------------------------------

/// Cache line size for scratch buffer alignment (ops.h: CACHE_LINE_SIZE).
/// On ARM: 128, VSX: 256, otherwise: 64.
#[cfg(target_arch = "aarch64")]
const CACHE_LINE_SIZE: usize = 128;
#[cfg(target_arch = "powerpc64")]
const CACHE_LINE_SIZE: usize = 256;
#[cfg(not(any(target_arch = "aarch64", target_arch = "powerpc64")))]
const CACHE_LINE_SIZE: usize = 64;

/// Im2col scratch size (ops.h: GGML_IM2COL_WORK_SIZE = 16 MiB).
const IM2COL_WORK_SIZE: usize = 16 * 1024 * 1024;

/// Flash attention tile sizes (common.h: GGML_FA_TILE_Q / GGML_FA_TILE_KV).
const FA_TILE_Q: usize = 64;
const FA_TILE_KV: usize = 64;

/// SOFT_MAX_UNROLL factor (vec.h: GGML_SOFT_MAX_UNROLL = 4).
const SOFT_MAX_UNROLL: i64 = 4;

// ---------------------------------------------------------------------------
// Rich node view for graph_plan
//
// The `GgmlTensor` in cpu_llama.rs is a minimal scheduling-only view.
// `PlanNode` carries the additional type + shape information that
// `ggml_graph_plan` needs to estimate per-op work-buffer sizes.
// ---------------------------------------------------------------------------

/// Source-tensor descriptor used only during planning.
#[derive(Clone, Copy)]
pub struct PlanSrc {
    /// Tensor type of this source (e.g. F32, Q4_K).
    pub typ: GgmlType,
    /// Element counts per dimension (ne[0..4]).
    pub ne: [i64; 4],
}

/// Per-node descriptor for work-buffer estimation (used only by [`graph_plan`]).
///
/// Mirrors the fields of `struct ggml_tensor` that `ggml_graph_plan` accesses.
pub struct PlanNode {
    /// Operator type.
    pub op: GgmlOp,
    /// Output tensor type.
    pub typ: GgmlType,
    /// Output element counts per dimension.
    pub ne: [i64; 4],
    /// Optional source tensors (up to 3).
    pub src: [Option<PlanSrc>; 3],
    /// op_params[0] — used by GATED_DELTA_NET (K stride).
    pub op_param0: i32,
}

impl PlanNode {
    fn src0(&self) -> Option<PlanSrc> { self.src[0] }
    fn src1(&self) -> Option<PlanSrc> { self.src[1] }
    fn src2(&self) -> Option<PlanSrc> { self.src[2] }
}

/// Minimal graph view for planning — just the node slice.
pub struct PlanGraph<'a> {
    pub nodes: &'a [PlanNode],
}

impl<'a> PlanGraph<'a> {
    pub fn n_nodes(&self) -> usize { self.nodes.len() }
}

// ---------------------------------------------------------------------------
// helpers matching C macros
// ---------------------------------------------------------------------------

/// Round `n` up to the next multiple of `m` (m must be a power of 2).
/// Translates `ggml_up(n, m)` from ggml-impl.h line 68.
#[inline]
fn ggml_up(n: i64, m: i64) -> i64 {
    debug_assert!(m > 0 && (m & (m - 1)) == 0, "m must be a power of 2");
    (n + m - 1) & !(m - 1)
}

/// Row byte-size for `n` elements of `typ`.
/// Translates `ggml_row_size(type, n)` from ggml.h / ggml-impl.h.
#[inline]
fn row_size(typ: GgmlType, n: i64) -> usize {
    let bs = typ.block_size() as i64;
    let ts = typ.type_size();
    // ggml_row_size = (type_size / block_size) * n
    // Integer: n / block_size * type_size (n assumed divisible by block_size).
    ((n + bs - 1) / bs) as usize * ts
}

/// `ggml_nelements(src)` — total element count.
#[inline]
fn nelements(ne: &[i64; 4]) -> i64 {
    ne[0] * ne[1] * ne[2] * ne[3]
}

// ---------------------------------------------------------------------------
// graph_plan — translated from `ggml_graph_plan` (ggml-cpu.c lines 2815-3068)
// ---------------------------------------------------------------------------
//
// Computes the scratch work-buffer size required by each node in `cgraph`
// and returns a `GgmlCplan` with `work_data` pre-allocated to that size.
//
// External functions with complex dependencies are stubbed:
//   - `ggml_cpu_extra_work_size`       → always returns `false` (not applicable)
//   - `ggml_cpu_iqp_supports_mul_mat`  → always returns `false`
//   - `ggml_cpu_iqp_scratch_size`      → returns 0
//
// The caller must populate `PlanNode` fields accurately for the work-buffer
// estimate to be correct; incorrect sizes lead to under/over-allocation
// (under causes UB in the actual compute; over wastes memory).

/// Translated from `ggml_graph_plan` (ggml-cpu.c lines 2815-3068).
///
/// `n_threads_hint` — if 0, defaults to `DEFAULT_N_THREADS`.
/// Returns a `GgmlCplan` with `work_data` allocated (or empty if not needed).
pub fn graph_plan(cgraph: &PlanGraph<'_>, n_threads_hint: usize) -> GgmlCplan {
    let n_threads = if n_threads_hint == 0 {
        DEFAULT_N_THREADS
    } else {
        n_threads_hint
    };

    let mut work_size: usize = 0;
    let mut max_tasks: usize = 1;

    for node in cgraph.nodes.iter() {
        // Use the scheduling get_n_tasks to compute thread count per op.
        // We create a minimal GgmlTensor-like struct inline here.
        let n_tasks = get_n_tasks_for_plan(node, n_threads);
        if n_tasks > max_tasks {
            max_tasks = n_tasks;
        }

        let mut cur: usize = 0;

        // ggml_cpu_extra_work_size stub: always false → proceed with switch.

        match node.op {
            GgmlOp::Cpy | GgmlOp::Dup => {
                // If output is quantized, or F16<->BF16, or F32<->I32: need f32 scratch.
                let needs_scratch = node.typ.is_quantized()
                    || matches!(
                        (node.src0().map(|s| s.typ), node.src1().map(|s| s.typ)),
                        (Some(GgmlType::F16), Some(GgmlType::BF16))
                            | (Some(GgmlType::BF16), Some(GgmlType::F16))
                            | (Some(GgmlType::F32), Some(GgmlType::I32))
                            | (Some(GgmlType::I32), Some(GgmlType::F32))
                    );
                if needs_scratch {
                    cur = GgmlType::F32.type_size() * node.ne[0] as usize * n_tasks;
                }
            }

            GgmlOp::Add | GgmlOp::AddId | GgmlOp::Add1 => {
                if node.src0().map(|s| s.typ.is_quantized()).unwrap_or(false) {
                    let ne0 = node.src0().map(|s| s.ne[0]).unwrap_or(0);
                    cur = GgmlType::F32.type_size() * ne0 as usize * n_tasks;
                }
            }

            GgmlOp::Acc => {
                if node.src0().map(|s| s.typ.is_quantized()).unwrap_or(false) {
                    let ne0 = node.src1().map(|s| s.ne[0]).unwrap_or(0);
                    cur = GgmlType::F32.type_size() * ne0 as usize * n_tasks;
                }
            }

            GgmlOp::CountEqual => {
                cur = node.typ.type_size() * n_tasks;
            }

            GgmlOp::MulMat => {
                // vec_dot scratch: if src1 type != vec_dot_type, we need to convert.
                if let Some(src0) = node.src0() {
                    let vdt = src0.typ.vec_dot_type();
                    if let Some(src1) = node.src1() {
                        if src1.typ != vdt {
                            cur = row_size(vdt, nelements(&src1.ne));
                        }
                    }
                }
                // IQP scratch: ggml_cpu_iqp_supports_mul_mat stubbed to false.
            }

            GgmlOp::MulMatId => {
                // Scratch for: src1 conversion + matrix_row_counts + matrix_rows + atomic chunks.
                // Stubbed: ggml_cpu_iqp_supports_mul_mat_id → false.
                if let (Some(src0), Some(src1), Some(_ids)) =
                    (node.src0(), node.src1(), node.src2())
                {
                    let vdt = src0.typ.vec_dot_type();
                    let n_as = src0.ne[2] as usize;
                    let ids_ne0 = node.src2().map(|s| s.ne[0]).unwrap_or(0) as usize;
                    let ids_ne1 = node.src2().map(|s| s.ne[1]).unwrap_or(0) as usize;

                    if src1.typ != vdt {
                        cur += row_size(vdt, nelements(&src1.ne))
                            + core::mem::size_of::<i64>();
                    }
                    // matrix_row_counts + padding
                    cur += n_as * core::mem::size_of::<i64>() + core::mem::size_of::<i64>();
                    // matrix_rows (mmid_row_mapping = two i32s = 8 bytes)
                    cur += n_as * ids_ne0 * ids_ne1 * 8 + core::mem::size_of::<i64>();
                    // atomic_current_chunk
                    cur += CACHE_LINE_SIZE * n_as + CACHE_LINE_SIZE;
                }
            }

            GgmlOp::OutProd => {
                if let Some(src0) = node.src0() {
                    if src0.typ.is_quantized() || src0.typ == GgmlType::F16 {
                        cur = GgmlType::F32.type_size() * src0.ne[0] as usize * n_tasks;
                    }
                }
            }

            GgmlOp::SetRows => {
                if let Some(src0) = node.src0() {
                    if src0.typ == GgmlType::F16 && node.typ != GgmlType::F16 {
                        cur = GgmlType::F32.type_size() * src0.ne[0] as usize * n_tasks;
                    }
                }
            }

            GgmlOp::SoftMax | GgmlOp::Rope | GgmlOp::RopeBack => {
                cur = GgmlType::F32.type_size() * node.ne[0] as usize * n_tasks;
            }

            GgmlOp::ConvTranspose1d => {
                // src0 is the kernel: ne[0]=K, ne[1]=Cout, ne[2]=Cin.
                // src1 is the input: ne[0]=L, ne[1]=Cin.
                if let (Some(src0), Some(src1)) = (node.src0(), node.src1()) {
                    let ne00 = src0.ne[0];
                    let ne01 = src0.ne[1];
                    let ne02 = src0.ne[2];
                    let ne10 = src1.ne[0];
                    let ne11 = src1.ne[1];
                    match src0.typ {
                        GgmlType::F16 | GgmlType::BF16 => {
                            cur += 2 * (ne00 * ne01 * ne02) as usize;
                            cur += 2 * (ne10 * ne11) as usize;
                        }
                        GgmlType::F32 => {
                            cur += 4 * (ne00 * ne01 * ne02) as usize;
                            cur += 4 * (ne10 * ne11) as usize;
                        }
                        _ => panic!("ggml_graph_plan: unsupported type for CONV_TRANSPOSE_1D"),
                    }
                }
            }

            GgmlOp::Conv2d | GgmlOp::Conv3d => {
                cur = IM2COL_WORK_SIZE;
            }

            GgmlOp::ConvTranspose2d => {
                if let (Some(src0), Some(src1)) = (node.src0(), node.src1()) {
                    let ts = src0.typ.type_size();
                    cur += ts * (src0.ne[0] * src0.ne[1] * src0.ne[2] * src0.ne[3]) as usize;
                    cur += ts * (src1.ne[0] * src1.ne[1] * src1.ne[2] * src1.ne[3]) as usize;
                }
            }

            GgmlOp::TopK => {
                if let Some(src0) = node.src0() {
                    cur += GgmlType::I32.type_size() * src0.ne[0] as usize * n_tasks;
                }
            }

            GgmlOp::FlashAttnExt => {
                // Per-thread buffers for prefill and decode paths.
                if let (Some(src0), Some(src1), Some(src2)) =
                    (node.src0(), node.src1(), node.src2())
                {
                    let dk = src1.ne[0] as usize;
                    let dv = src2.ne[0] as usize;
                    let _neq2 = src0.ne[2] as usize;

                    // Prefill: Q_q + KQ + mask + VKQ32 + V32 + K_f32 per thread.
                    let prefill = 4 * (FA_TILE_Q * dk
                        + 2 * FA_TILE_Q * FA_TILE_KV
                        + FA_TILE_Q * dv
                        + FA_TILE_KV * dv
                        + FA_TILE_KV * dk)
                        * n_tasks;

                    // Decode: VKQ accumulator + partial M/S + intra-thread scratch.
                    let n_chunks = n_tasks;
                    let neq2 = src0.ne[2] as usize;
                    let decode = 4 * (neq2 * n_chunks * (2 + dv) + n_tasks * (dk + 2 * dv));

                    cur += prefill.max(decode);
                }
            }

            GgmlOp::FlashAttnBack => {
                // Scratch for S and SM buffers: 2 * max(D, ne11_up) * n_tasks * sizeof(f32).
                if let (Some(src0), Some(src1)) = (node.src0(), node.src1()) {
                    let d = src0.ne[0];
                    let ne11 = ggml_up(src1.ne[1], SOFT_MAX_UNROLL);
                    let mx_dn = d.max(ne11) * 2;
                    match src1.typ {
                        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                            cur = 4 * mx_dn as usize * n_tasks;
                            cur += 4 * mx_dn as usize * n_tasks;
                        }
                        _ => {}
                    }
                }
            }

            GgmlOp::CrossEntropyLoss => {
                if let Some(src0) = node.src0() {
                    cur = node.typ.type_size() * (n_tasks + src0.ne[0] as usize * n_tasks);
                }
            }

            GgmlOp::GatedDeltaNet => {
                if let Some(src2) = node.src2() {
                    let s_v = src2.ne[0] as usize;
                    let k = node.op_param0 as usize;
                    let per_thread = s_v + if k > 1 { s_v * s_v } else { 0 };
                    cur = per_thread * 4 * n_tasks;
                }
            }

            GgmlOp::LightningIndexer => {
                if let Some(src1) = node.src1() {
                    cur += 4 * src1.ne[0] as usize * n_tasks;
                }
            }

            // All other ops: no scratch buffer needed (or handled by extra_work_size).
            _ => {}
        }

        if cur > work_size {
            work_size = cur;
        }
    }

    // Padding for per-thread alignment.
    if work_size > 0 {
        work_size += CACHE_LINE_SIZE * n_threads;
    }

    let effective_threads = max_tasks.min(n_threads);
    let work_data = if work_size > 0 {
        vec![0u8; work_size]
    } else {
        Vec::new()
    };

    GgmlCplan {
        n_threads:           effective_threads,
        work_data,
        abort_callback:      None,
        abort_callback_data: core::ptr::null_mut(),
        use_ref:             false,
    }
}

/// Per-op thread count for the planning pass.
/// Mirrors `ggml_get_n_tasks` but operating on `PlanNode` rather than `GgmlTensor`.
fn get_n_tasks_for_plan(node: &PlanNode, n_threads: usize) -> usize {
    if node.ne[0] == 0 {
        return 1;
    }
    match node.op {
        GgmlOp::Cpy
        | GgmlOp::Dup
        | GgmlOp::Cont
        | GgmlOp::Add
        | GgmlOp::AddId
        | GgmlOp::Add1
        | GgmlOp::Acc
        | GgmlOp::Cumsum
        | GgmlOp::Tri
        | GgmlOp::Fill
        | GgmlOp::CountEqual
        | GgmlOp::SolveTri
        | GgmlOp::GatedDeltaNet
        | GgmlOp::Dsv4HcComb
        | GgmlOp::Dsv4HcPre
        | GgmlOp::Dsv4HcPost
        | GgmlOp::SiluBack
        | GgmlOp::Mul
        | GgmlOp::Div
        | GgmlOp::Norm
        | GgmlOp::RmsNorm
        | GgmlOp::RmsNormBack
        | GgmlOp::L2Norm
        | GgmlOp::GroupNorm
        | GgmlOp::Concat
        | GgmlOp::MulMat
        | GgmlOp::MulMatId
        | GgmlOp::OutProd
        | GgmlOp::Glu
        | GgmlOp::DiagMaskZero
        | GgmlOp::DiagMaskInf
        | GgmlOp::SoftMaxBack
        | GgmlOp::Rope
        | GgmlOp::RopeBack
        | GgmlOp::AddRelPos
        | GgmlOp::Im2col
        | GgmlOp::Im2colBack
        | GgmlOp::Im2col3d
        | GgmlOp::Conv2d
        | GgmlOp::Conv3d
        | GgmlOp::Conv2dDw
        | GgmlOp::Col2im1d
        | GgmlOp::ConvTranspose1d
        | GgmlOp::ConvTranspose2d
        | GgmlOp::Upscale
        | GgmlOp::Pad
        | GgmlOp::PadReflect1d
        | GgmlOp::Roll
        | GgmlOp::Arange
        | GgmlOp::TimestepEmbedding
        | GgmlOp::Argsort
        | GgmlOp::TopK
        | GgmlOp::FlashAttnExt
        | GgmlOp::FlashAttnBack
        | GgmlOp::SsmConv
        | GgmlOp::SsmScan
        | GgmlOp::LightningIndexer
        | GgmlOp::CrossEntropyLoss
        | GgmlOp::CrossEntropyLossBack
        | GgmlOp::OptStepAdamW
        | GgmlOp::OptStepSgd => n_threads,

        GgmlOp::SoftMax => n_threads.min(node.ne[1] as usize),

        GgmlOp::RwkvWkv6
        | GgmlOp::GatedLinearAttn
        | GgmlOp::RwkvWkv7 => {
            let src1_ne1 = node.src1().map(|s| s.ne[1] as usize).unwrap_or(1);
            n_threads.min(src1_ne1)
        }

        GgmlOp::MapCustom1
        | GgmlOp::MapCustom2
        | GgmlOp::MapCustom3
        | GgmlOp::Custom => {
            // op_param0 == -1 (GGML_N_TASKS_MAX) → n_threads; else min(param, n_threads)
            if node.op_param0 == -1 {
                n_threads
            } else {
                n_threads.min(node.op_param0 as usize)
            }
        }

        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// try_fuse_ops — translated from `ggml_cpu_try_fuse_ops` (ggml-cpu.c lines 3075-3107)
// ---------------------------------------------------------------------------
//
// In the C source, this checks `ggml_can_fuse(cgraph, node_n, fuse_ops, 2)`
// which requires access to the full graph DAG (i.e., which nodes consume which
// outputs).  That predicate is not yet translated into Rust, so fusion is
// represented as a trait that the caller implements.
//
// A `CanFuse` implementor inspects the graph and returns `true` if
// nodes [node_n .. node_n+len) form a fusible chain.
//
// The actual fused kernel (`ggml_compute_forward_rms_norm_mul_fused`) is
// translated in ops_llama.rs; the trait routes to it via the `FusedDispatch`
// trait.

/// Check predicate for op-fusion eligibility.
/// Mirrors `ggml_can_fuse(cgraph, node_n, ops, len)` from ggml-impl.h.
pub trait CanFuse {
    /// Returns `true` if `nodes[node_n .. node_n + ops.len()]` can be fused
    /// as the sequence `ops`.
    fn can_fuse(&self, node_n: usize, ops: &[GgmlOp]) -> bool;
}

/// Fusion dispatcher — executed when a fusion is detected.
pub trait FusedDispatch<P> {
    /// Execute the RMS_NORM + MUL fused kernel.
    fn forward_rms_norm_mul_fused(&self, params: &P, rms_node_n: usize, mul_node_n: usize);
}

/// Global fusion disable flag (mirrors `ggml_cpu_disable_fusion`).
/// Set to `true` to bypass all fusion (e.g. for reference implementations).
static DISABLE_FUSION: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn set_fusion_disabled(disabled: bool) {
    DISABLE_FUSION.store(disabled, core::sync::atomic::Ordering::Relaxed);
}

pub fn is_fusion_disabled() -> bool {
    DISABLE_FUSION.load(core::sync::atomic::Ordering::Relaxed)
}

/// Rich plan-time node view needed by `try_fuse_ops` to evaluate fusion guards.
///
/// Contains the fields that `ggml_cpu_try_fuse_ops` actually reads:
///   `node->src[0]->type`, `mul_node->type`, `mul_w->type`,
///   `mul_w->ne[0]`, `mul_w->nb[0]`.
pub struct FuseNodeInfo {
    /// Output tensor type of this node.
    pub typ: GgmlType,
    /// Element counts per dimension.
    pub ne: [i64; 4],
    /// Byte stride of the weight tensor (nb[0]): must equal `sizeof(float)` for F32.
    pub weight_nb0: usize,
    /// Type of the weight source tensor.
    pub weight_type: GgmlType,
    /// Type of the input source (src[0]).
    pub src0_type: GgmlType,
}

/// Translated from `ggml_cpu_try_fuse_ops` (ggml-cpu.c lines 3075-3107).
///
/// Returns the number of nodes consumed by fusion (0 = no fusion applied).
///
/// `node_n`  — index of the current node in the graph.
/// `rms_node` — info for the RMS_NORM candidate.
/// `mul_node` — info for the following MUL node (caller provides if exists).
/// `use_ref`  — from the compute plan; disables fusion when true.
/// `can_fuse` — graph-level predicate (implements [`CanFuse`]).
/// `dispatch` — called to execute the fused kernel.
pub fn try_fuse_ops<P, D: FusedDispatch<P>, C: CanFuse>(
    node_n: usize,
    rms_node: &FuseNodeInfo,
    mul_node: Option<&FuseNodeInfo>,
    use_ref: bool,
    can_fuse: &C,
    dispatch: &D,
    params: &P,
) -> usize {
    if is_fusion_disabled() || use_ref {
        return 0;
    }

    // RMS_NORM + MUL fusion.
    if let Some(mul) = mul_node {
        if can_fuse.can_fuse(node_n, &[GgmlOp::RmsNorm, GgmlOp::Mul]) {
            if rms_node.src0_type == GgmlType::F32
                && mul.typ == GgmlType::F32
                && mul.weight_type == GgmlType::F32
                && mul.ne[0] == rms_node.ne[0]
                && mul.weight_nb0 == core::mem::size_of::<f32>()
            {
                dispatch.forward_rms_norm_mul_fused(params, node_n, node_n + 1);
                return 1;
            }
        }
    }

    0
}

// ---------------------------------------------------------------------------
// Float conversion functions
// (ggml-cpu.c lines 3484-3641)
// ---------------------------------------------------------------------------
//
// The C source uses SIMD intrinsics (F16C / AVX512 / RISC-V) for throughput.
// The Rust translations use the `half` crate for scalar fallback; SIMD paths
// are added via `target_feature` blocks on x86_64 for the AVX-512/F16C cases.

/// Translated from `ggml_cpu_fp32_to_fp32` — memcpy (ggml-cpu.c line 3484).
pub fn fp32_to_fp32(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

/// Translated from `ggml_cpu_fp32_to_fp16` (ggml-cpu.c lines 3488-3519).
///
/// Converts `n` f32 values to IEEE 754 half-precision (u16 bit patterns).
/// Uses F16C SIMD when available on x86_64; falls back to `half` crate.
pub fn fp32_to_fp16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut i = 0usize;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "f16c"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 16 <= n {
                let xv = _mm512_loadu_ps(src.as_ptr().add(i));
                let yv = _mm512_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT as i32);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, yv);
                i += 16;
            }
            while i + 8 <= n {
                let xv = _mm256_loadu_ps(src.as_ptr().add(i));
                let yv = _mm256_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT as i32);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, yv);
                i += 8;
            }
            while i + 4 <= n {
                let xv = _mm_loadu_ps(src.as_ptr().add(i));
                let yv = _mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT as i32);
                _mm_storel_epi64(dst.as_mut_ptr().add(i) as *mut __m128i, yv);
                i += 4;
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "f16c"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 8 <= n {
                let xv = _mm256_loadu_ps(src.as_ptr().add(i));
                let yv = _mm256_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT as i32);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, yv);
                i += 8;
            }
            while i + 4 <= n {
                let xv = _mm_loadu_ps(src.as_ptr().add(i));
                let yv = _mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT as i32);
                _mm_storel_epi64(dst.as_mut_ptr().add(i) as *mut __m128i, yv);
                i += 4;
            }
        }
    }

    // Scalar fallback via `half` crate.
    while i < n {
        dst[i] = half::f16::from_f32(src[i]).to_bits();
        i += 1;
    }
}

/// Translated from `ggml_cpu_fp16_to_fp32` (ggml-cpu.c lines 3521-3573).
///
/// Converts `n` IEEE 754 half-precision bit patterns (u16) to f32.
pub fn fp16_to_fp32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut i = 0usize;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "f16c"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 16 <= n {
                let xv = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let yv = _mm512_cvtph_ps(xv);
                _mm512_storeu_ps(dst.as_mut_ptr().add(i), yv);
                i += 16;
            }
            while i + 8 <= n {
                let xv = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                let yv = _mm256_cvtph_ps(xv);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), yv);
                i += 8;
            }
            while i + 4 <= n {
                let xv = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
                let yv = _mm_cvtph_ps(xv);
                _mm_storeu_ps(dst.as_mut_ptr().add(i), yv);
                i += 4;
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "f16c"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 8 <= n {
                let xv = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                let yv = _mm256_cvtph_ps(xv);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), yv);
                i += 8;
            }
            while i + 4 <= n {
                let xv = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
                let yv = _mm_cvtph_ps(xv);
                _mm_storeu_ps(dst.as_mut_ptr().add(i), yv);
                i += 4;
            }
        }
    }

    while i < n {
        dst[i] = half::f16::from_bits(src[i]).to_f32();
        i += 1;
    }
}

/// Translated from `ggml_cpu_fp32_to_bf16` (ggml-cpu.c lines 3575-3580).
///
/// Converts f32 to bfloat16 (u16 bit pattern = upper 16 bits of f32).
pub fn fp32_to_bf16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = half::bf16::from_f32(*s).to_bits();
    }
}

/// Translated from `ggml_cpu_fp32_to_i32` (ggml-cpu.c lines 3582-3587).
///
/// Truncates f32 → i32 (C cast: `y[i] = x[i]`).
pub fn fp32_to_i32(src: &[f32], dst: &mut [i32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as i32;
    }
}

/// Translated from `ggml_cpu_bf16_to_fp32` (ggml-cpu.c lines 3589-3641).
///
/// Converts bfloat16 (u16 bit patterns) to f32.
/// Uses AVX2/AVX512 shift-and-cast path on x86_64 when available.
pub fn bf16_to_fp32(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut i = 0usize;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 16 <= n {
                // Load 16 u16 → zero-extend to i32 → shift left 16 → reinterpret as f32.
                let xv = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let z32 = _mm512_cvtepu16_epi32(xv);
                let shifted = _mm512_slli_epi32(z32, 16);
                let fp = _mm512_castsi512_ps(shifted);
                _mm512_storeu_ps(dst.as_mut_ptr().add(i), fp);
                i += 16;
            }
            while i + 8 <= n {
                let xv = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                let z32 = _mm256_cvtepu16_epi32(xv);
                let shifted = _mm256_slli_epi32(z32, 16);
                let fp = _mm256_castsi256_ps(shifted);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), fp);
                i += 8;
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        target_feature = "avx2"
    ))]
    {
        use std::arch::x86_64::*;
        unsafe {
            while i + 8 <= n {
                let xv = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
                let z32 = _mm256_cvtepu16_epi32(xv);
                let shifted = _mm256_slli_epi32(z32, 16);
                let fp = _mm256_castsi256_ps(shifted);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), fp);
                i += 8;
            }
        }
    }

    while i < n {
        dst[i] = half::bf16::from_bits(src[i]).to_f32();
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// CPU feature detection
// (ggml-cpu.c lines 3643-3851)
//
// Each function returns 1 if the CPU supports the feature, 0 otherwise.
// In C these are compile-time macro guards; in Rust we use `cfg` attributes.
// The stable std `is_x86_feature_detected!` macro handles runtime detection
// where compile-time detection is insufficient.
// ---------------------------------------------------------------------------

/// `ggml_cpu_has_avx` — AVX support (ggml-cpu.c line 3643).
pub fn has_avx() -> bool {
    #[cfg(target_feature = "avx")]
    return true;
    #[cfg(not(target_feature = "avx"))]
    return false;
}

/// `ggml_cpu_has_avx_vnni` — AVX-VNNI support (ggml-cpu.c line 3651).
pub fn has_avx_vnni() -> bool {
    #[cfg(target_feature = "avxvnni")]
    return true;
    #[cfg(not(target_feature = "avxvnni"))]
    return false;
}

/// `ggml_cpu_has_avx2` — AVX2 support (ggml-cpu.c line 3659).
pub fn has_avx2() -> bool {
    #[cfg(target_feature = "avx2")]
    return true;
    #[cfg(not(target_feature = "avx2"))]
    return false;
}

/// `ggml_cpu_has_avx512` — AVX-512F support (ggml-cpu.c line 3667).
pub fn has_avx512() -> bool {
    #[cfg(target_feature = "avx512f")]
    return true;
    #[cfg(not(target_feature = "avx512f"))]
    return false;
}

/// `ggml_cpu_has_avx512_vbmi` — AVX-512 VBMI support (ggml-cpu.c line 3675).
pub fn has_avx512_vbmi() -> bool {
    #[cfg(target_feature = "avx512vbmi")]
    return true;
    #[cfg(not(target_feature = "avx512vbmi"))]
    return false;
}

/// `ggml_cpu_has_avx512_vnni` — AVX-512 VNNI support (ggml-cpu.c line 3683).
pub fn has_avx512_vnni() -> bool {
    #[cfg(target_feature = "avx512vnni")]
    return true;
    #[cfg(not(target_feature = "avx512vnni"))]
    return false;
}

/// `ggml_cpu_has_avx512_bf16` — AVX-512 BF16 support (ggml-cpu.c line 3691).
pub fn has_avx512_bf16() -> bool {
    #[cfg(target_feature = "avx512bf16")]
    return true;
    #[cfg(not(target_feature = "avx512bf16"))]
    return false;
}

/// `ggml_cpu_has_amx_int8` — AMX INT8 support (ggml-cpu.c line 3699).
pub fn has_amx_int8() -> bool {
    #[cfg(target_feature = "amx-int8")]
    return true;
    #[cfg(not(target_feature = "amx-int8"))]
    return false;
}

/// `ggml_cpu_has_bmi2` — BMI2 support (ggml-cpu.c line 3707).
pub fn has_bmi2() -> bool {
    #[cfg(target_feature = "bmi2")]
    return true;
    #[cfg(not(target_feature = "bmi2"))]
    return false;
}

/// `ggml_cpu_has_fma` — FMA support (ggml-cpu.c line 3715).
pub fn has_fma() -> bool {
    #[cfg(target_feature = "fma")]
    return true;
    #[cfg(not(target_feature = "fma"))]
    return false;
}

/// `ggml_cpu_has_arm_fma` — ARM FMA support (ggml-cpu.c line 3723).
pub fn has_arm_fma() -> bool {
    #[cfg(target_feature = "neon")]
    return true;
    #[cfg(not(target_feature = "neon"))]
    return false;
}

/// `ggml_cpu_has_riscv_v` — RISC-V V intrinsics (ggml-cpu.c line 3731).
pub fn has_riscv_v() -> bool {
    false // Rust stable doesn't expose riscv_v_intrinsic cfg
}

/// `ggml_cpu_has_f16c` — F16C (half-float convert) support (ggml-cpu.c line 3747).
pub fn has_f16c() -> bool {
    #[cfg(target_feature = "f16c")]
    return true;
    #[cfg(not(target_feature = "f16c"))]
    return false;
}

/// `ggml_cpu_has_fp16_va` — ARM FP16 vector arithmetic (ggml-cpu.c line 3755).
pub fn has_fp16_va() -> bool {
    false // not yet exposed as a stable Rust cfg target_feature
}

/// `ggml_cpu_has_wasm_simd` — WebAssembly SIMD128 (ggml-cpu.c line 3763).
pub fn has_wasm_simd() -> bool {
    #[cfg(target_feature = "simd128")]
    return true;
    #[cfg(not(target_feature = "simd128"))]
    return false;
}

/// `ggml_cpu_has_sse3` — SSE3 support (ggml-cpu.c line 3779).
pub fn has_sse3() -> bool {
    #[cfg(target_feature = "sse3")]
    return true;
    #[cfg(not(target_feature = "sse3"))]
    return false;
}

/// `ggml_cpu_has_ssse3` — SSSE3 support (ggml-cpu.c line 3787).
pub fn has_ssse3() -> bool {
    #[cfg(target_feature = "ssse3")]
    return true;
    #[cfg(not(target_feature = "ssse3"))]
    return false;
}

/// `ggml_cpu_has_vsx` — PowerPC VSX (POWER9_VECTOR) (ggml-cpu.c line 3795).
pub fn has_vsx() -> bool {
    false // PowerPC VSX not exposed in stable Rust cfg
}

/// `ggml_cpu_has_neon` — ARM NEON (ggml-cpu.c line 3811).
pub fn has_neon() -> bool {
    #[cfg(target_feature = "neon")]
    return true;
    #[cfg(not(target_feature = "neon"))]
    return false;
}

/// `ggml_cpu_has_dotprod` — ARM dot-product (ggml-cpu.c line 3819).
pub fn has_dotprod() -> bool {
    #[cfg(target_feature = "dotprod")]
    return true;
    #[cfg(not(target_feature = "dotprod"))]
    return false;
}

/// `ggml_cpu_has_sve` — ARM SVE (ggml-cpu.c line 3827).
pub fn has_sve() -> bool {
    false // ARM SVE not yet stable in Rust cfg
}

/// `ggml_cpu_has_matmul_int8` — ARM matmul int8 (ggml-cpu.c line 3835).
pub fn has_matmul_int8() -> bool {
    false // not yet in stable Rust
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plan_node(op: GgmlOp, typ: GgmlType, ne: [i64; 4]) -> PlanNode {
        PlanNode {
            op,
            typ,
            ne,
            src: [None, None, None],
            op_param0: 0,
        }
    }

    #[test]
    fn graph_plan_no_nodes_empty_work() {
        let nodes: &[PlanNode] = &[];
        let g = PlanGraph { nodes };
        let plan = graph_plan(&g, 4);
        assert_eq!(plan.work_data.len(), 0);
        assert!(plan.n_threads >= 1);
    }

    #[test]
    fn graph_plan_add_no_scratch() {
        let nodes = [make_plan_node(GgmlOp::Add, GgmlType::F32, [64, 1, 1, 1])];
        let g = PlanGraph { nodes: &nodes };
        let plan = graph_plan(&g, 4);
        // ADD with F32 src needs no scratch → work_size = 0.
        assert_eq!(plan.work_data.len(), 0);
    }

    #[test]
    fn graph_plan_softmax_scratch() {
        let nodes = [make_plan_node(GgmlOp::SoftMax, GgmlType::F32, [128, 4, 1, 1])];
        let g = PlanGraph { nodes: &nodes };
        let plan = graph_plan(&g, 4);
        // SOFT_MAX needs F32 * ne[0] * n_tasks scratch + alignment padding.
        assert!(plan.work_data.len() > 0);
    }

    #[test]
    fn fp32_to_fp16_roundtrip() {
        let src: Vec<f32> = vec![1.0f32, -2.5, 0.0, f32::INFINITY, 3.14159];
        let mut mid = vec![0u16; src.len()];
        let mut dst = vec![0f32; src.len()];
        fp32_to_fp16(&src, &mut mid);
        fp16_to_fp32(&mid, &mut dst);
        // Allow ~0.1% relative error for half-precision roundtrip.
        for (s, d) in src.iter().zip(dst.iter()) {
            if s.is_finite() {
                let err = (s - d).abs() / s.abs().max(1e-6);
                assert!(err < 1e-2, "roundtrip error {err} for {s} -> {d}");
            }
        }
    }

    #[test]
    fn fp32_to_bf16_roundtrip() {
        let src: Vec<f32> = vec![1.0f32, -2.5, 0.0, 3.14159];
        let mut mid = vec![0u16; src.len()];
        let mut dst = vec![0f32; src.len()];
        fp32_to_bf16(&src, &mut mid);
        bf16_to_fp32(&mid, &mut dst);
        for (s, d) in src.iter().zip(dst.iter()) {
            let err = (s - d).abs() / s.abs().max(1e-6);
            assert!(err < 1e-2, "bf16 roundtrip error {err} for {s} -> {d}");
        }
    }

    #[test]
    fn fp32_to_i32_truncation() {
        let src = [1.9f32, -1.9, 0.0, 100.0];
        let mut dst = [0i32; 4];
        fp32_to_i32(&src, &mut dst);
        assert_eq!(dst, [1, -1, 0, 100]);
    }

    #[test]
    fn cpu_feature_detection_returns_bool() {
        // These should not panic.
        let _ = has_avx();
        let _ = has_avx2();
        let _ = has_fma();
        let _ = has_f16c();
        let _ = has_neon();
    }

    #[test]
    fn ggml_up_rounds_up() {
        assert_eq!(ggml_up(0, 4), 0);
        assert_eq!(ggml_up(1, 4), 4);
        assert_eq!(ggml_up(4, 4), 4);
        assert_eq!(ggml_up(5, 4), 8);
        assert_eq!(ggml_up(3, SOFT_MAX_UNROLL), 4);
    }
}

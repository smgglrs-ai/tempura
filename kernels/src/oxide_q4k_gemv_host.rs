/// Host-side wrapper for the oxide-compiled Q4_K GEMV kernel.
///
/// The kernel is compiled with `cargo oxide build` from `tools/oxide-gemv-q4k/`.
/// `build.rs` extracts the embedded PTX from the compiled binary and writes
/// it to `$OUT_DIR/oxide_q4k_gemv.ptx`. This module embeds that PTX at
/// compile time and loads it into the CUDA driver at runtime.
///
/// ## Kernel design
///
/// ```text
/// Grid:  ceil(N / WARPS_PER_BLOCK) blocks
/// Block: WARPS_PER_BLOCK * 32 = 128 threads
/// ```
///
/// Each 32-thread warp computes ONE output neuron via stride-32 strided
/// accumulation followed by a butterfly warp-reduce (5x `shfl.sync.bfly`).
/// Lane 0 of each warp writes the final sum to `out[n_out]`.
///
/// ## PTX parameter layout
///
/// ```text
/// param_0: .u64 .ptr  -- out pointer (DisjointSlice data ptr)
/// param_1: .u64       -- out len (u64)
/// param_2: .u64 .ptr  -- x pointer
/// param_3: .u64       -- x len (u64)
/// param_4: .u64 .ptr  -- packed pointer
/// param_5: .u64       -- packed len (u64)
/// param_6: .u32       -- n (u32)
/// param_7: .u32       -- k_blocks (u32)
/// ```

use crate::{KernelError, Result};
use crate::gpu_capabilities::{GpuCapabilities, KernelVariant};
use crate::q4k_gemv_v4_ptx;
use crate::q4k_fused_silu_v4_ptx;
use crate::q4k_batch_sm120_ptx;
use crate::q4k_mma_batch_sm120_ptx;
use cuda_core::{launch_kernel, Stream};
use cuda_core::sys::{CUfunction, CUmodule};
use cutile::{api, prelude::DeviceOp, tensor::Tensor};
use std::ffi::{c_void, CString};
use std::sync::Arc;

/// The PTX source compiled by `cargo oxide build tools/oxide-gemv-q4k`.
#[cfg(feature = "oxide-gemv")]
static OXIDE_Q4K_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_q4k_gemv.ptx"));

const KERNEL_ENTRY: &str = "q4k_gemv_dp4a_base";
const KERNEL_BATCH: &str = "q4k_gemv_dp4a_base_batch";
const KERNEL_COLOCATED: &str = "q4k_gemm_colocated";
const KERNEL_SMEM: &str = "q4k_gemm_smem";
const KERNEL_Q8_QUANTIZE: &str = "quantize_q8_1";
const KERNEL_Q8_GEMV: &str = "q4k_gemv_decode_q8";
const KERNEL_Q8_GEMV_V2: &str = "q4k_gemv_dp4a_q8_v2";
const KERNEL_Q8_GEMV_V3: &str = "q4k_gemv_dp4a_q8";
const KERNEL_Q8_GEMV_V3R: &str = "q4k_gemv_dp4a_q8_raw"; // raw GGUF 36-u32 format
const KERNEL_Q8_GEMV_V3_KPAR: &str = "q4k_gemv_dp4a_kpar8"; // 8 warps/block for Q/O proj
const KERNEL_FUSED_SILU: &str = "q4k_fused_silu_dp4a"; // fused gate+up+SiLU
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128
const WARPS_PER_BLOCK_KPAR: u32 = 8;
const BLOCK_DIM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 32; // 256


/// Launches the oxide-compiled Q4_K GEMV kernel for M=1 decode.
pub struct OxideQ4kGemv {
    cu_module:      CUmodule,
    cu_function:    CUfunction,
    /// Optional Q8_1 quantize kernel (present after `cargo oxide build` with new kernels).
    cu_q8_quantize:    Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV kernel (mini-warp only, no K-parallelism).
    cu_q8_gemv:        Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV dp4a_kpar4 kernel (K-parallelism + mini-warp dp4a combined).
    cu_q8_dp4a_kpar4:     Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV dp4a kernel (VDR=2 double-block K-loop).
    cu_q8_dp4a:     Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV dp4a_raw kernel (dp4a + raw GGUF 36-u32 format, 14% less DRAM).
    cu_q8_dp4a_raw:    Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV dp4a_kpar8 kernel (8 warps/block -> 100% SM occupancy for large N).
    cu_q8_dp4a_kpar8: Option<CUfunction>,
    /// Optional batched Q4K GEMV: grid=(N,M,1), L2-shared weight blocks.
    cu_batch_gemv:     Option<CUfunction>,
    /// Optional colocated GEMM: all M rows in one block, L2 broadcast for weights.
    cu_colocated:      Option<CUfunction>,
    /// Optional smem-tiled GEMM: row-0 loads weights into smem, all M rows compute from smem.
    cu_smem:           Option<CUfunction>,
    /// Optional fused gate+up GEMV with inline SiLU: single launch for both FFN projections.
    cu_fused_silu:     Option<CUfunction>,
    /// Optional sm_120-native kpar kernel (native 36-u32 GGUF format).
    /// Loaded from a SEPARATE CUDA module (hand-written PTX, not the oxide binary).
    cu_sm120_module:      Option<CUmodule>,
    cu_sm120_dp4a_kpar8:        Option<CUfunction>,
    /// Optional sm_120-native fused gate+up+SiLU kernel (hand-written PTX).
    /// Faster than oxide sm_80 fused silu: direct Blackwell codegen, no JIT overhead.
    cu_sm120_fused_module:      Option<CUmodule>,
    cu_sm120_fused_silu:        Option<CUfunction>,
    /// Optional sm_120-native batch GEMV kernel (hand-written PTX, static smem only).
    /// Replaces the oxide sm_80 batch kernels that crash on Blackwell (error 719).
    cu_sm120_batch_module:      Option<CUmodule>,
    cu_sm120_batch:             Option<CUfunction>,
    /// Optional sm_120 INT8 tensor-core batch GEMM (mma.sync.aligned.m16n8k32).
    /// Uses ~4000 TOPS INT8 tensor cores vs ~840 TOPS dp4a SIMD.
    /// Grid (N/8, 1, 1), Block (32, 1, 1). Requires sm_120 + uses_raw_gguf.
    cu_mma_batch_module:        Option<CUmodule>,
    cu_mma_batch:               Option<CUfunction>,
    /// Selected Q4K+Q8_1 kpar dispatch function: sm120 or dp4a_kpar8 (sm_80) fallback.
    /// Populated by the KernelVariant selection at construction time.  Always Some when
    /// at least one kpar variant loaded.  None only when no kpar kernel is available at all.
    pub kpar_dispatch: Option<KernelVariant>,
    stream:         Arc<Stream>,
    /// True when the raw-GGUF 36-u32 kernel is active (needs matching upload format).
    pub uses_raw_gguf: bool,
    /// Pre-allocated Q8_1 scratch buffers (set by pre_allocate_q8_scratch before graph capture).
    /// These avoid device allocation on a capturing stream (CUDA error 900).
    q8_qs_scratch:     std::cell::RefCell<Option<Arc<Tensor<u32>>>>,  // [max_k / 4]
    q8_ds_scratch:     std::cell::RefCell<Option<Arc<Tensor<u32>>>>,  // [max_k / 32]
    q8_out_scratch:    std::cell::RefCell<Option<Arc<Tensor<f32>>>>,  // [max_n]
    /// Pre-allocated output for the fused gate+up SiLU kernel.
    fused_silu_scratch: std::cell::RefCell<Option<Arc<Tensor<f32>>>>, // [max_n]
    /// Separate Q8_1 activation scratch for the KV stream (stream_kv).
    /// The main q8_qs_scratch / q8_ds_scratch are used by Q projection on stream_main.
    /// K and V run concurrently on stream_kv -- they need their own scratch to avoid
    /// data races when uses_raw_gguf=true (both use quantize_q8_1 via submit_sm120_kpar_kv_into_slot).
    kv_q8_qs_scratch: std::cell::RefCell<Option<Arc<Tensor<u32>>>>,
    kv_q8_ds_scratch: std::cell::RefCell<Option<Arc<Tensor<u32>>>>,
}

unsafe impl Send for OxideQ4kGemv {}
unsafe impl Sync for OxideQ4kGemv {}

impl Drop for OxideQ4kGemv {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
        }
        if let Some(m) = self.cu_sm120_module {
            if !m.is_null() {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
        }
        if let Some(m) = self.cu_sm120_fused_module {
            if !m.is_null() {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
        }
        if let Some(m) = self.cu_sm120_batch_module {
            if !m.is_null() {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
        }
        if let Some(m) = self.cu_mma_batch_module {
            if !m.is_null() {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
        }
    }
}

impl OxideQ4kGemv {
    /// Load the oxide Q4_K GEMV kernel from the embedded PTX.
    ///
    /// Returns `None` when `oxide-gemv` feature is absent or loading fails.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        #[cfg(not(feature = "oxide-gemv"))]
        return None;

        #[cfg(feature = "oxide-gemv")]
        {
            if OXIDE_Q4K_PTX.is_empty() {
                return None;
            }
            let ptx_str = std::str::from_utf8(OXIDE_Q4K_PTX).ok()?;
            let c_ptx = CString::new(ptx_str).ok()?;
            let entry = CString::new(KERNEL_ENTRY).ok()?;

            let _ = stream.device().bind_to_thread().ok()?;

            const CUDA_SUCCESS: u32 = 0;

            let cu_module = unsafe {
                let mut module = std::mem::MaybeUninit::uninit();
                let ret = cuda_core::sys::cuModuleLoadData(
                    module.as_mut_ptr(),
                    c_ptx.as_ptr() as *const _,
                );
                if ret != CUDA_SUCCESS {
                    eprintln!(
                        "oxide-gemv-q4k: cuModuleLoadData failed (code={ret}); \
                         falling back to cuTile Q4_K kernel"
                    );
                    return None;
                }
                module.assume_init()
            };

            let cu_function = unsafe {
                let mut func = std::mem::MaybeUninit::uninit();
                let ret = cuda_core::sys::cuModuleGetFunction(
                    func.as_mut_ptr(),
                    cu_module,
                    entry.as_ptr(),
                );
                if ret != CUDA_SUCCESS {
                    let _ = cuda_core::sys::cuModuleUnload(cu_module);
                    eprintln!(
                        "oxide-gemv-q4k: cuModuleGetFunction('{KERNEL_ENTRY}') failed (code={ret})"
                    );
                    return None;
                }
                func.assume_init()
            };

            // Opportunistically load the Q8_1 kernels -- they are absent until the
            // user rebuilds with `cargo oxide build` after adding the new kernels.
            let load_fn = |name: &str| -> Option<CUfunction> {
                let entry = CString::new(name).ok()?;
                let mut func = std::mem::MaybeUninit::uninit();
                let ret = unsafe {
                    cuda_core::sys::cuModuleGetFunction(func.as_mut_ptr(), cu_module, entry.as_ptr())
                };
                if ret != CUDA_SUCCESS { None } else { Some(unsafe { func.assume_init() }) }
            };
            let cu_q8_quantize = load_fn(KERNEL_Q8_QUANTIZE);
            let cu_q8_gemv     = load_fn(KERNEL_Q8_GEMV);
            let cu_q8_dp4a_kpar4  = load_fn(KERNEL_Q8_GEMV_V2);
            let cu_q8_dp4a      = load_fn(KERNEL_Q8_GEMV_V3);
            let cu_q8_dp4a_raw     = load_fn(KERNEL_Q8_GEMV_V3R);
            let cu_q8_dp4a_kpar8 = load_fn(KERNEL_Q8_GEMV_V3_KPAR);
            let cu_batch_gemv  = load_fn(KERNEL_BATCH);
            let cu_colocated   = load_fn(KERNEL_COLOCATED);
            let cu_smem        = load_fn(KERNEL_SMEM);
            let cu_fused_silu  = load_fn(KERNEL_FUSED_SILU);

            // -- Try to load sm_120 v4 PTX (Blackwell-native, native GGUF format) ---
            // Query device compute capability. Only attempt the sm_120 PTX when the
            // device is Blackwell (cc >= 1200); other GPUs get the sm_80 oxide fallback.
            let caps = GpuCapabilities::query(stream.device().cu_device());
            // TEMPURA_NO_SM120_KPAR=1 disables sm120 for debugging (uses dp4a_kpar8 fallback).
            let sm120_disabled = std::env::var("TEMPURA_NO_SM120_KPAR").map(|v| v == "1").unwrap_or(false);
            let (cu_sm120_module, cu_sm120_dp4a_kpar8) = if !sm120_disabled && caps.cc() >= q4k_gemv_v4_ptx::MIN_CC {
                let ptx_cstr = CString::new(q4k_gemv_v4_ptx::PTX).unwrap_or_default();
                let mut v4_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        v4_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let v4_mod = unsafe { v4_mod.assume_init() };
                    let entry = CString::new(q4k_gemv_v4_ptx::ENTRY).unwrap_or_default();
                    let mut v4_fn = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(v4_fn.as_mut_ptr(), v4_mod, entry.as_ptr())
                    };
                    if rc2 == CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q4k: loaded sm_120 v4 PTX ({} bytes) -- native GGUF format, direct Blackwell codegen",
                            q4k_gemv_v4_ptx::PTX.len(),
                        );
                        (Some(v4_mod), Some(unsafe { v4_fn.assume_init() }))
                    } else {
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(v4_mod) };
                        eprintln!("oxide-gemv-q4k: cuModuleGetFunction(sm120_dp4a_kpar8) failed (rc={rc2}), using dp4a_kpar8");
                        (None, None)
                    }
                } else {
                    eprintln!(
                        "oxide-gemv-q4k: cuModuleLoadData(sm_120 PTX) failed (rc={rc}), cc={}, using dp4a_kpar8",
                        caps.cc(),
                    );
                    (None, None)
                }
            } else {
                // Non-Blackwell GPU: skip sm120 entirely, use oxide sm_80 dp4a_kpar8.
                (None, None)
            };

            // -- Load sm_120 fused gate+up+SiLU PTX (separate module from kpar) -
            // Reuses the same CUDA module as the sm_120 kpar if already loaded,
            // or loads a second module. Currently: separate module, same cc guard.
            let mut cu_sm120_fused_mod: Option<CUmodule> = None;
            let cu_sm120_fused_silu: Option<CUfunction> = if !sm120_disabled && caps.cc() >= q4k_fused_silu_v4_ptx::MIN_CC {
                let ptx_cstr = CString::new(q4k_fused_silu_v4_ptx::PTX).unwrap_or_default();
                let mut fused_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        fused_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let fused_mod = unsafe { fused_mod.assume_init() };
                    let entry = CString::new(q4k_fused_silu_v4_ptx::ENTRY).unwrap_or_default();
                    let mut fused_fn = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(fused_fn.as_mut_ptr(), fused_mod, entry.as_ptr())
                    };
                    if rc2 == CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q4k: loaded sm_120 fused silu PTX ({} bytes) -- direct Blackwell codegen",
                            q4k_fused_silu_v4_ptx::PTX.len(),
                        );
                        cu_sm120_fused_mod = Some(fused_mod);
                        Some(unsafe { fused_fn.assume_init() })
                    } else {
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(fused_mod) };
                        eprintln!("oxide-gemv-q4k: cuModuleGetFunction(sm120_fused_silu) failed (rc={rc2}), using oxide fused silu");
                        None
                    }
                } else {
                    eprintln!(
                        "oxide-gemv-q4k: cuModuleLoadData(sm120_fused_silu PTX) failed (rc={rc}), using oxide fused silu",
                    );
                    None
                }
            } else {
                None
            };

            // -- Load sm_120 batch GEMV PTX (static smem, no JIT crash on Blackwell) -
            // The oxide sm_80 batch kernels (q4k_gemv_dp4a_base_batch, q4k_gemm_colocated)
            // crash with CUDA error 719 on Blackwell due to dynamic smem JIT incompatibility.
            // This hand-written sm_120 PTX uses static smem and works correctly on RTX 5090.
            let mut cu_sm120_batch_mod: Option<CUmodule> = None;
            let cu_sm120_batch: Option<CUfunction> = if !sm120_disabled && caps.cc() >= q4k_batch_sm120_ptx::MIN_CC {
                let ptx_cstr = CString::new(q4k_batch_sm120_ptx::PTX).unwrap_or_default();
                let mut batch_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        batch_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let batch_mod = unsafe { batch_mod.assume_init() };
                    let entry = CString::new(q4k_batch_sm120_ptx::ENTRY).unwrap_or_default();
                    let mut batch_fn = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(batch_fn.as_mut_ptr(), batch_mod, entry.as_ptr())
                    };
                    if rc2 == CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q4k: loaded sm_120 batch PTX ({} bytes) -- static smem, Blackwell batch GEMV",
                            q4k_batch_sm120_ptx::PTX.len(),
                        );
                        cu_sm120_batch_mod = Some(batch_mod);
                        Some(unsafe { batch_fn.assume_init() })
                    } else {
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(batch_mod) };
                        eprintln!("oxide-gemv-q4k: cuModuleGetFunction(sm120_batch) failed (rc={rc2}), batch decode uses sequential fallback");
                        None
                    }
                } else {
                    eprintln!(
                        "oxide-gemv-q4k: cuModuleLoadData(sm120_batch PTX) failed (rc={rc}), batch decode uses sequential fallback",
                    );
                    None
                }
            } else {
                None
            };

            // -- Load sm_120 MMA batch GEMM PTX (INT8 tensor cores, m16n8k32) ---
            let mut cu_mma_batch_mod: Option<CUmodule> = None;
            let cu_mma_batch: Option<CUfunction> = if !sm120_disabled && caps.cc() >= q4k_mma_batch_sm120_ptx::MIN_CC {
                let ptx_cstr = CString::new(q4k_mma_batch_sm120_ptx::PTX).unwrap_or_default();
                let mut mma_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        mma_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let mma_mod = unsafe { mma_mod.assume_init() };
                    let entry = CString::new(q4k_mma_batch_sm120_ptx::ENTRY).unwrap_or_default();
                    let mut mma_fn = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(mma_fn.as_mut_ptr(), mma_mod, entry.as_ptr())
                    };
                    if rc2 == CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q4k: loaded sm_120 MMA batch PTX ({} bytes) -- INT8 tensor cores m16n8k32",
                            q4k_mma_batch_sm120_ptx::PTX.len(),
                        );
                        cu_mma_batch_mod = Some(mma_mod);
                        Some(unsafe { mma_fn.assume_init() })
                    } else {
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(mma_mod) };
                        eprintln!("oxide-gemv-q4k: cuModuleGetFunction(mma_batch) failed (rc={rc2}), falling back to dp4a batch");
                        None
                    }
                } else {
                    eprintln!("oxide-gemv-q4k: cuModuleLoadData(mma_batch PTX) failed (rc={rc}), falling back to dp4a batch");
                    None
                }
            } else {
                None
            };

            // -- Build KernelVariant table and select best kpar dispatch ---------
            let mut kpar_variants: Vec<KernelVariant> = Vec::new();
            if let Some(f) = cu_q8_dp4a_kpar8 {
                kpar_variants.push(KernelVariant { cu_fn: f, min_cc: 800, name: "q4k_gemv_dp4a_kpar8" });
            }
            if let Some(f) = cu_sm120_dp4a_kpar8 {
                kpar_variants.push(KernelVariant { cu_fn: f, min_cc: 1200, name: "q4k_gemv_sm120_dp4a_kpar8" });
            }
            let kpar_dispatch = KernelVariant::select_best(&kpar_variants, &caps).cloned();
            // v4 uses native 36-u32 GGUF format in the v4 kernel itself, but the production
            // engine uses `dispatch_kv_on_stream` for K/V projections which calls the BASIC f32
            // GEMV kernel -- that reads repacked 42-u32 format. If we set uses_raw_gguf=true,
            // K/V weights would be in native format but the basic f32 GEMV reads them as
            // repacked -> wrong values -> NaN -> error 719 in graph replay.
            //
            // TEMPORARY WORKAROUND: Keep uses_raw_gguf=false (repacked format) until the engine
            // properly handles native format for all dispatch paths (including dispatch_kv_on_stream).
            // sm120 kpar ALSO accepts repacked 42-u32 by adjusting its parameter layout.
            // The sm_120 codegen benefit is retained; the 14% DRAM reduction requires further work.
            //
            // TODO: once engine uses per-slot output tensors for K/V, enable native format
            // by setting uses_raw_gguf = uses_sm120 and updating dispatch_kv_on_stream to use sm120 path.
            let uses_sm120 = kpar_dispatch.as_ref().map(|v| v.min_cc >= 1200).unwrap_or(false);

            eprintln!(
                "oxide-gemv-q4k: loaded (oxide PTX {} bytes, entry={}, q8={}, q8_dp4a_kpar4={}, q8_dp4a={}, q8_dp4a_raw={}, q8_dp4a_kpar8={}, sm120_dp4a_kpar8={}, kpar_dispatch={}, fused_silu={}, sm120_fused_silu={}, batch={}, sm120_batch={}, mma_batch={})",
                OXIDE_Q4K_PTX.len(),
                KERNEL_ENTRY,
                cu_q8_gemv.is_some(),
                cu_q8_dp4a_kpar4.is_some(),
                cu_q8_dp4a.is_some(),
                cu_q8_dp4a_raw.is_some(),
                cu_q8_dp4a_kpar8.is_some(),
                cu_sm120_dp4a_kpar8.is_some(),
                kpar_dispatch.as_ref().map(|v| v.name).unwrap_or("none"),
                cu_fused_silu.is_some(),
                cu_sm120_fused_silu.is_some(),
                cu_batch_gemv.is_some(),
                cu_sm120_batch.is_some(),
                cu_mma_batch.is_some(),
            );
            // dp4a_raw: raw GGUF format for the oxide sm_80 path -- benchmarked and DISABLED.
            // uses_raw_gguf=true when sm120 is selected: sm120 reads native 36-u32 GGUF format.
            // dispatch_kv_on_stream uses submit_sm120_kpar_kv_into_slot (with per-KV-stream
            // Q8_1 scratch) when uses_raw_gguf=true, so K/V weights can also be native.
            let uses_raw_gguf = uses_sm120;
            Some(Self {
                cu_module, cu_function, cu_q8_quantize, cu_q8_gemv, cu_q8_dp4a_kpar4, cu_q8_dp4a, cu_q8_dp4a_raw, cu_q8_dp4a_kpar8, cu_batch_gemv, cu_colocated, cu_smem, cu_fused_silu, cu_sm120_module, cu_sm120_dp4a_kpar8, cu_sm120_fused_module: cu_sm120_fused_mod, cu_sm120_fused_silu, cu_sm120_batch_module: cu_sm120_batch_mod, cu_sm120_batch, cu_mma_batch_module: cu_mma_batch_mod, cu_mma_batch, kpar_dispatch, uses_raw_gguf, stream,
                q8_qs_scratch:      std::cell::RefCell::new(None),
                q8_ds_scratch:      std::cell::RefCell::new(None),
                q8_out_scratch:     std::cell::RefCell::new(None),
                fused_silu_scratch: std::cell::RefCell::new(None),
                kv_q8_qs_scratch:   std::cell::RefCell::new(None),
                kv_q8_ds_scratch:   std::cell::RefCell::new(None),
            })
        }
    }

    /// Device-to-device Q4_K GEMV: `out[n] = sum_k dequant(packed[n,k]) * x[k]`.
    pub fn forward_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let out = api::zeros::<f32>(&[n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        self.launch(x, packed, &out, n, k_blocks)?;

        unsafe {
            self.stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(Arc::new(out))
    }

    /// Returns true if the tiled batch GEMV kernel is compiled into the PTX.
    pub fn has_batch_gemv(&self) -> bool {
        self.cu_batch_gemv.is_some()
    }

    /// Batched Q4_K GEMV: `out[m, n] = dequant(packed[n,k]) x x[m,k]`.
    ///
    /// Grid = (N, M, 1) -- each block handles one (n_out, row) pair.
    /// L2-cache reuse: M blocks for the same n_out share the weight data.
    pub fn forward_batch_d2d(
        &self,
        x: &Arc<Tensor<f32>>,        // [m, k] activations (flat m*k elements)
        packed: &Arc<Tensor<u32>>,    // [n, k_blocks, 42] packed Q4K weights
        m: usize,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let func = match &self.cu_batch_gemv {
            Some(f) => *f,
            None => return Err(KernelError::InvalidConfig("batch GEMV not compiled".into())),
        };

        let out = api::zeros::<f32>(&[m * n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = (m * n) as u64;
        // Pass x as a raw pointer (batch kernel uses *const f32 for row arithmetic).
        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        let mut params: [*mut c_void; 9] = [
            &out_ptr     as *const _ as *mut c_void,
            &out_len     as *const _ as *mut c_void,
            &x_ptr       as *const _ as *mut c_void,  // *const f32 raw pointer
            &packed_ptr  as *const _ as *mut c_void,
            &packed_len  as *const _ as *mut c_void,
            &m_u32       as *const _ as *mut c_void,
            &n_u32       as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
            std::ptr::null_mut(),  // sentinel (no extra param)
        ];
        // Trim to 8 params (drop the sentinel).
        let mut params8: [*mut c_void; 8] = [
            params[0], params[1], params[2], params[3],
            params[4], params[5], params[6], params[7],
        ];

        let cu_stream = self.stream.cu_stream();
        unsafe {
            launch_kernel(
                func,
                (n_u32, m_u32, 1),
                (BLOCK_DIM, 1, 1),
                WARPS_PER_BLOCK * 4,
                cu_stream,
                &mut params8,
            )
        }.map_err(|e| KernelError::InvalidConfig(format!("batch q4k gemv: {e}")))?;

        unsafe {
            self.stream.synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }
        Ok(Arc::new(out))
    }

    /// Colocated Q4K GEMM: all M rows in ONE block for L2 weight broadcast.
    ///
    /// Grid = (ceil(N / WARPS_PER_BLOCK), 1, 1)
    /// Block = (WARPS_PER_BLOCK * 32 * M, 1, 1)   flat encoding: row = (tid/32) / WARPS_PER_BLOCK
    /// smem = M * WARPS_PER_BLOCK * 4 bytes
    pub fn forward_colocated_d2d(
        &self,
        x: &Arc<Tensor<f32>>,        // [m, k] activations
        packed: &Arc<Tensor<u32>>,    // [n, k_blocks, 42]
        m: usize,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let func = match &self.cu_colocated {
            Some(f) => *f,
            None => return Err(KernelError::InvalidConfig("colocated GEMM not compiled".into())),
        };
        // Max block size check: WARPS_PER_BLOCK * 32 * m <= 1024
        let block_x = (WARPS_PER_BLOCK * 32) as usize * m;
        if block_x > 1024 {
            return Err(KernelError::InvalidConfig(
                format!("colocated GEMM: block_dim_x={block_x} > 1024 for m={m}")
            ));
        }

        let out = api::zeros::<f32>(&[m * n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = (m * n) as u64;
        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        let mut params8: [*mut c_void; 8] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &m_u32        as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
        ];

        let grid_x = (n_u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        let smem_bytes = m_u32 * WARPS_PER_BLOCK * 4;
        let cu_stream = self.stream.cu_stream();
        unsafe {
            launch_kernel(
                func,
                (grid_x, 1, 1),
                (block_x as u32, 1, 1),
                smem_bytes,
                cu_stream,
                &mut params8,
            )
        }.map_err(|e| KernelError::InvalidConfig(format!("colocated q4k gemm: {e}")))?;

        unsafe {
            self.stream.synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }
        Ok(Arc::new(out))
    }

    pub fn has_smem_gemm(&self) -> bool { self.cu_smem.is_some() }

    /// Smem-tiled GEMM: row-0 warps load 42 u32s of weight data into smem, all M rows
    /// then compute FMAs from smem -- eliminating (M-1)/M of weight DRAM traffic.
    pub fn forward_smem_d2d(
        &self,
        x: &Arc<Tensor<f32>>,        // [m, k] activations
        packed: &Arc<Tensor<u32>>,    // [n, k_blocks, 42]
        m: usize,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let func = match &self.cu_smem {
            Some(f) => *f,
            None => return self.forward_colocated_d2d(x, packed, m, n, k_blocks),
        };
        let block_x = (WARPS_PER_BLOCK * 32) as usize * m;
        if block_x > 1024 {
            return Err(KernelError::InvalidConfig(
                format!("smem GEMM: block_dim_x={block_x} > 1024 for m={m}")
            ));
        }

        let out = api::zeros::<f32>(&[m * n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = (m * n) as u64;
        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        let mut params8: [*mut c_void; 8] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &m_u32        as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
        ];

        let grid_x = (n_u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        // smem: [WARPS_PER_BLOCK * 42 weight u32s] + [M * WARPS_PER_BLOCK partial-sum f32s]
        let smem_bytes = (WARPS_PER_BLOCK * 42 + m_u32 * WARPS_PER_BLOCK) * 4;
        let cu_stream = self.stream.cu_stream();
        unsafe {
            launch_kernel(
                func,
                (grid_x, 1, 1),
                (block_x as u32, 1, 1),
                smem_bytes,
                cu_stream,
                &mut params8,
            )
        }.map_err(|e| KernelError::InvalidConfig(format!("smem q4k gemm: {e}")))?;

        unsafe {
            self.stream.synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }
        Ok(Arc::new(out))
    }

    /// Non-blocking variant: re-uses a pre-allocated output tensor.
    pub unsafe fn submit_d2d_into(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        self.launch(x, packed, &out, n, k_blocks)?;
        Ok(Arc::new(out))
    }

    /// Launch quantize_q8_1 + sm120_dp4a_kpar8 on a specified CUDA stream, writing to a
    /// caller-supplied slot tensor.  Uses separate `kv_q8_qs_scratch` /
    /// `kv_q8_ds_scratch` to avoid racing with the main stream's Q8 scratch
    /// (which Q projection uses concurrently on stream_main).
    ///
    /// Called by `dispatch_kv_on_stream` when `uses_raw_gguf=true` (native 36-u32
    /// format).  `packed` must be in native 36-u32 format.
    ///
    /// # Safety
    /// `cu_kv_stream` must be a valid CUstream for the duration of the launches.
    pub unsafe fn submit_sm120_kpar_kv_into_slot(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
        cu_kv_stream: cuda_core::sys::CUstream,
    ) -> Result<Arc<Tensor<f32>>> {
        let kpar_fn = match &self.kpar_dispatch {
            Some(v) if v.min_cc >= 1200 => v.cu_fn,
            _ => return Err(KernelError::InvalidConfig("sm120_dp4a_kpar8 not available for KV dispatch".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_q8_blocks = k.div_ceil(32);

        // Use KV-specific activation scratch (separate from main-stream Q8 scratch).
        let x_qs = match &*self.kv_q8_qs_scratch.borrow() {
            Some(t) if t.size() >= k / 4 => t.clone(),
            _ => return Err(KernelError::InvalidConfig(
                "kv_q8_qs_scratch not pre-allocated or too small".into(),
            )),
        };
        let x_ds = match &*self.kv_q8_ds_scratch.borrow() {
            Some(t) if t.size() >= k_q8_blocks => t.clone(),
            _ => return Err(KernelError::InvalidConfig(
                "kv_q8_ds_scratch not pre-allocated or too small".into(),
            )),
        };

        // -- quantize_q8_1 on cu_kv_stream --------------------------------------
        {
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k / 4) as u64;
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k as u32;
            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_kv_stream, &mut params)
                .map_err(|e| KernelError::InvalidConfig(format!("quantize_q8_1 (kv stream v4): {e}")))?;
        }

        // -- sm120_dp4a_kpar8 on cu_kv_stream, writing into `out` ----------------------
        {
            let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ou_ptr: u64 = out.device_pointer().cu_deviceptr();
            let n_u32:  u32 = n as u32;
            let kb_u32: u32 = k_blocks as u32;
            let mut params: [*mut c_void; 6] = [
                &wt_ptr as *const _ as *mut c_void,
                &qs_ptr as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ou_ptr as *const _ as *mut c_void,
                &n_u32  as *const _ as *mut c_void,
                &kb_u32 as *const _ as *mut c_void,
            ];
            launch_kernel(kpar_fn, (n as u32, 1, 1), (BLOCK_DIM_KPAR, 1, 1), 128, cu_kv_stream, &mut params)
                .map_err(|e| KernelError::InvalidConfig(format!("sm120_dp4a_kpar8 KV slot: {e}")))?;
        }

        Ok(Arc::new(out))
    }

    /// Like `submit_d2d_into` but launches on an explicit CUDA stream.
    ///
    /// Used by the multi-stream QKV path to run K and V GEMVs on `stream_kv`
    /// concurrently with the Q GEMV on `stream_main`.  Only the f32 kernel is
    /// used (not the Q8 dp4a path) so that the per-weight output is written
    /// directly into the pre-allocated slot tensor instead of shared Q8 scratch.
    ///
    /// # Safety
    /// `cu_stream` must be a valid CUstream for the duration of the kernel.
    pub unsafe fn submit_d2d_into_on(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<Arc<Tensor<f32>>> {
        self.launch_on(x, packed, &out, n, k_blocks, cu_stream)?;
        Ok(Arc::new(out))
    }

    fn launch_on(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: &Tensor<f32>,
        n: usize,
        k_blocks: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<()> {
        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = n as u64;
        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let x_len: u64 = x.size() as u64;
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        let mut params: [*mut c_void; 8] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &x_len        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
        ];

        let grid_x = n as u32;
        unsafe {
            launch_kernel(
                self.cu_function,
                (grid_x, 1, 1),
                (BLOCK_DIM, 1, 1),
                32 * 4,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q4k_gemv_decode_v2 (kv stream): {e}")))?;
        Ok(())
    }

    fn launch(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: &Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<()> {
        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = n as u64;

        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let x_len: u64 = x.size() as u64;

        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;

        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        let mut params: [*mut c_void; 8] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &x_len        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
        ];

        // Multi-warp: one block per output neuron. All WARPS_PER_BLOCK warps in a
        // block cooperate on the same neuron, each covering K/WARPS_PER_BLOCK blocks.
        let grid_x = n as u32;
        let cu_stream = self.stream.cu_stream();

        unsafe {
            launch_kernel(
                self.cu_function,
                (grid_x, 1, 1),
                (BLOCK_DIM, 1, 1),
                32 * 4,  // 128 bytes: 32 smem f32 slots needed on Blackwell (all warp-0 lanes read smem[lane])
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q4k_gemv_decode_v2: {e}")))?;

        Ok(())
    }

    /// Pre-allocate Q8_1 scratch buffers outside of CUDA graph capture.
    ///
    /// Must be called before the first `cuStreamBeginCapture` -- allocations on
    /// a capturing stream are not permitted (CUDA error 900).  The buffers are
    /// sized for the largest K and N seen across all Q4K projections in the model.
    pub fn pre_allocate_q8_scratch(&self, max_k: usize, max_n: usize) -> Result<()> {
        if self.cu_q8_quantize.is_none() { return Ok(()); }
        let k_q8_blocks = max_k.div_ceil(32);
        *self.q8_qs_scratch.borrow_mut() = Some(Arc::new(
            api::zeros::<u32>(&[max_k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        ));
        *self.q8_ds_scratch.borrow_mut() = Some(Arc::new(
            api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        ));
        *self.q8_out_scratch.borrow_mut() = Some(Arc::new(
            api::zeros::<f32>(&[max_n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        ));
        if self.cu_fused_silu.is_some() {
            *self.fused_silu_scratch.borrow_mut() = Some(Arc::new(
                api::zeros::<f32>(&[max_n])
                    .sync_on(&self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ));
        }
        // When sm120 is active (uses_raw_gguf=true), K/V projections use submit_sm120_kpar_kv_into_slot
        // on stream_kv concurrently with Q on stream_main. They need their own Q8_1 scratch to
        // avoid data races. Size: same as main scratch (max_k covers all projection dims).
        if self.uses_raw_gguf {
            *self.kv_q8_qs_scratch.borrow_mut() = Some(Arc::new(
                api::zeros::<u32>(&[max_k / 4])
                    .sync_on(&self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ));
            *self.kv_q8_ds_scratch.borrow_mut() = Some(Arc::new(
                api::zeros::<u32>(&[k_q8_blocks])
                    .sync_on(&self.stream)
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
            ));
        }
        Ok(())
    }

    /// Returns true when the fused gate+up SiLU kernel is compiled into the PTX AND
    /// the current upload format is repacked 42-u32 (fused silu requires repacked format).
    ///
    /// When `uses_raw_gguf == true` (v4 sm_120 kernel active), fused silu is disabled
    /// and the engine falls back to two separate GEMVs + element-wise SiLU-mul.
    pub fn has_fused_silu(&self) -> bool {
        !self.uses_raw_gguf && self.cu_fused_silu.is_some() && self.cu_q8_quantize.is_some()
    }

    /// Returns true when the sm_120 hand-tuned fused gate+up+SiLU kernel is available.
    /// Preferred over `has_fused_silu` when both are present (direct Blackwell codegen).
    pub fn has_sm120_fused_silu(&self) -> bool {
        self.cu_sm120_fused_silu.is_some() && self.cu_q8_quantize.is_some()
    }

    /// Returns true when the Q8_1 quantize + integer GEMV path is compiled into the PTX.
    pub fn has_q8_path(&self) -> bool {
        self.cu_q8_quantize.is_some()
            && (self.cu_q8_dp4a.is_some() || self.cu_q8_dp4a_kpar4.is_some() || self.cu_q8_gemv.is_some())
    }

    /// Returns true when the sm_120 kpar kernel is loaded and selected.
    pub fn has_sm120_kpar(&self) -> bool {
        self.cu_sm120_dp4a_kpar8.is_some()
            && self.kpar_dispatch.as_ref().map(|v| v.min_cc >= 1200).unwrap_or(false)
    }

    /// Returns true when the sm_120 batch GEMV kernel is loaded (static smem, Blackwell-safe).
    pub fn has_sm120_batch(&self) -> bool {
        self.cu_sm120_batch.is_some() && self.cu_q8_quantize.is_some()
    }

    /// Returns true when the sm_120 MMA batch GEMM kernel is loaded (INT8 tensor cores).
    /// Currently disabled: the simple (N/8,1,1) grid has low SM occupancy vs dp4a.
    /// Re-enable when stream-K variant (grid=SM_COUNT, fixup pass) is implemented.
    pub fn has_mma_batch(&self) -> bool {
        false
    }

    /// Batched Q4K+Q8_1 GEMV on sm_120: quantize M rows then launch grid=(N,M,1).
    ///
    /// Avoids the CUDA error 719 crash from oxide sm_80 dynamic-smem batch kernels on Blackwell.
    /// The M activation rows in `x` are quantized in one `quantize_q8_1` call (k_total=M*K),
    /// producing contiguous act_qs/act_ds arrays that the batch PTX kernel strides into by row.
    ///
    /// `x` must be a flat [m * k] f32 tensor (row-major, k = k_blocks * 256).
    pub fn forward_sm120_batch_d2d(
        &self,
        x: &Arc<Tensor<f32>>,         // [m * k] row-major activations
        packed: &Arc<Tensor<u32>>,     // [n * k_blocks * 36] native GGUF Q4K weights
        m: usize,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let batch_fn = match self.cu_sm120_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("sm120_batch not loaded".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_total = m * k;
        let k_q8_blocks_total = k_total.div_ceil(32); // m * k_blocks * 8

        let cu_stream = self.stream.cu_stream();

        // Allocate Q8_1 scratch for M rows.
        let x_qs = Arc::new(
            api::zeros::<u32>(&[k_total / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        );
        let x_ds = Arc::new(
            api::zeros::<u32>(&[k_q8_blocks_total])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        );
        let out = Arc::new(
            api::zeros::<f32>(&[m * n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?,
        );

        // Quantize M*K activations in one call.
        {
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k_total / 4) as u64;
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks_total as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k_total as u32;
            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks_total.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("quantize_q8_1 batch: {e}")))?;
        }

        // Launch sm120 batch kernel v2: grid=(n, 1, 1) -- one block per output
        // neuron, weights read ONCE and shared across all M activation rows.
        {
            let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ou_ptr: u64 = out.device_pointer().cu_deviceptr();
            let n_u32:  u32 = n as u32;
            let kb_u32: u32 = k_blocks as u32;
            let m_u32:  u32 = m as u32;
            let mut params: [*mut c_void; 7] = [
                &wt_ptr as *const _ as *mut c_void,
                &qs_ptr as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ou_ptr as *const _ as *mut c_void,
                &n_u32  as *const _ as *mut c_void,
                &kb_u32 as *const _ as *mut c_void,
                &m_u32  as *const _ as *mut c_void,
            ];
            unsafe {
                launch_kernel(
                    batch_fn,
                    (n_u32, 1, 1),
                    (BLOCK_DIM_KPAR, 1, 1),
                    128,
                    cu_stream,
                    &mut params,
                )
            }.map_err(|e| KernelError::InvalidConfig(format!("sm120_batch q4k gemv: {e}")))?;
        }

        unsafe {
            self.stream.synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }
        Ok(out)
    }

    /// Graph-capture-safe variant of `forward_sm120_batch_d2d`.
    ///
    /// Accepts pre-allocated Q8_1 scratch tensors and a pre-allocated output
    /// tensor -- no device allocations or stream synchronisations are performed
    /// inside this method, so it can be called within a `cuStreamBeginCapture`
    /// region.
    ///
    /// `x`      -- flat `[m * k]` f32 activations (row-major, k = k_blocks * 256).
    /// `packed` -- `[n * k_blocks * 36]` native GGUF Q4K weights.
    /// `x_qs`   -- pre-allocated `[m * k / 4]` u32 Q8_1 quantise scratch.
    /// `x_ds`   -- pre-allocated `[ceil(m*k/32)]` u32 Q8_1 scale scratch.
    /// `out`    -- pre-allocated `[m * n]` f32 output tensor (written in-place).
    pub fn forward_sm120_batch_d2d_into(
        &self,
        x:       &Arc<Tensor<f32>>,
        packed:  &Arc<Tensor<u32>>,
        m:       usize,
        n:       usize,
        k_blocks: usize,
        x_qs:    &Arc<Tensor<u32>>,
        x_ds:    &Arc<Tensor<u32>>,
        out:     &Arc<Tensor<f32>>,
    ) -> Result<()> {
        let batch_fn = match self.cu_sm120_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("sm120_batch not loaded".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_total = m * k;
        let k_q8_blocks_total = k_total.div_ceil(32);

        let cu_stream = self.stream.cu_stream();

        // Quantize M*K activations in one call (async -- no stream sync).
        {
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k_total / 4) as u64;
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks_total as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k_total as u32;
            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks_total.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("quantize_q8_1 batch _into: {e}")))?;
        }

        // Launch sm120 batch kernel v2: grid=(n, 1, 1) -- one block per output
        // neuron, weights read ONCE and shared across all M activation rows.
        {
            let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ou_ptr: u64 = out.device_pointer().cu_deviceptr();
            let n_u32:  u32 = n as u32;
            let kb_u32: u32 = k_blocks as u32;
            let m_u32:  u32 = m as u32;
            let mut params: [*mut c_void; 7] = [
                &wt_ptr as *const _ as *mut c_void,
                &qs_ptr as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ou_ptr as *const _ as *mut c_void,
                &n_u32  as *const _ as *mut c_void,
                &kb_u32 as *const _ as *mut c_void,
                &m_u32  as *const _ as *mut c_void,
            ];
            unsafe {
                launch_kernel(
                    batch_fn,
                    (n_u32, 1, 1),
                    (BLOCK_DIM_KPAR, 1, 1),
                    128,
                    cu_stream,
                    &mut params,
                )
            }.map_err(|e| KernelError::InvalidConfig(format!("sm120_batch q4k gemv _into: {e}")))?;
        }

        // No stream.synchronize() -- caller owns the sync (graph launch or explicit sync).
        Ok(())
    }

    /// Graph-capture-safe INT8 tensor-core batch GEMM using mma.sync.aligned.m16n8k32.
    ///
    /// Requires pre-quantized Q8_1 scratch (`x_qs`, `x_ds`) and pre-allocated output `out`.
    /// No device allocations or stream synchronisations inside -- safe for CUDA graph capture.
    ///
    /// `x`       -- flat [m * k] f32 activations (row-major, k = k_blocks * 256).
    /// `packed`  -- [n * k_blocks * 36] native GGUF Q4K weights (uses_raw_gguf=true required).
    /// `x_qs`    -- pre-allocated [m * k / 4] u32 Q8_1 INT8 scratch (from q8_quantize).
    /// `x_ds`    -- pre-allocated [ceil(m*k/32)] u32 Q8_1 scale scratch.
    /// `out`     -- pre-allocated [m * n] f32 output (written in-place, MUST be zeroed before
    ///              graph capture since the kernel accumulates via smem, not direct global write).
    pub fn forward_mma_batch_d2d_into(
        &self,
        x:        &Arc<Tensor<f32>>,
        packed:   &Arc<Tensor<u32>>,
        m:        usize,
        n:        usize,
        k_blocks: usize,
        x_qs:     &Arc<Tensor<u32>>,
        x_ds:     &Arc<Tensor<u32>>,
        out:      &Arc<Tensor<f32>>,
    ) -> Result<()> {
        let mma_fn = match self.cu_mma_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("mma_batch not loaded".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_total = m * k;
        let k_q8_blocks_total = k_total.div_ceil(32);

        let cu_stream = self.stream.cu_stream();

        // Quantize M*K activations in one call (async).
        {
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k_total / 4) as u64;
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks_total as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k_total as u32;
            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks_total.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("quantize_q8_1 mma_batch: {e}")))?;
        }

        // Launch MMA batch kernel:
        // Grid: (ceil(n/8), 1, 1), Block: (32, 1, 1) -- 1 warp per n8 tile.
        {
            let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ou_ptr: u64 = out.device_pointer().cu_deviceptr();
            let n_u32:  u32 = n as u32;
            let kb_u32: u32 = k_blocks as u32;
            let m_u32:  u32 = m as u32;
            let mut params: [*mut c_void; 7] = [
                &wt_ptr as *const _ as *mut c_void,
                &qs_ptr as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ou_ptr as *const _ as *mut c_void,
                &n_u32  as *const _ as *mut c_void,
                &kb_u32 as *const _ as *mut c_void,
                &m_u32  as *const _ as *mut c_void,
            ];
            let grid_x = n_u32.div_ceil(8);
            // Warp count: largest divisor of k_blocks*8 that maximises sm_120 occupancy.
            // Occupancy formula: floor(65536 / (regs * W * 32)) * W.
            // W=4 (regs~39): floor(65536/4992)*4 = 52 warps/SM.
            // W=8 (regs~42): floor(65536/10752)*8 = 48 warps/SM.
            // W=4 wins; k_blocks*8 is always divisible by 4 for k_blocks >= 1.
            let warps = if k_blocks * 8 % 4 == 0 { 4u32 } else { 2u32 };
            let block_dim = warps * 32;
            unsafe {
                launch_kernel(
                    mma_fn,
                    (grid_x, 1, 1),
                    (block_dim, 1, 1),
                    0,            // no dynamic smem (uses static smem in PTX)
                    cu_stream,
                    &mut params,
                )
            }.map_err(|e| KernelError::InvalidConfig(format!("mma_batch q4k gemm: {e}")))?;
        }

        // No stream.synchronize() -- caller owns the sync.
        Ok(())
    }

    /// Returns true when ANY kpar dispatch kernel is available (sm120 or dp4a_kpar8 fallback).
    pub fn has_any_kpar(&self) -> bool {
        self.cu_q8_quantize.is_some() && self.kpar_dispatch.is_some()
    }

    /// Returns true when the dp4a_kpar8 kernel (8 warps/block) is compiled into the PTX.
    pub fn has_dp4a_kpar8(&self) -> bool {
        self.cu_q8_quantize.is_some() && self.cu_q8_dp4a_kpar8.is_some()
    }

    /// K-parallel Q4K+Q8_1 GEMV using 8 warps/block for 100% SM occupancy on large projections.
    ///
    /// Quantizes `x` to Q8_1 then launches `q4k_gemv_decode_q8_v3_kpar` (PTX name) with 256 threads/block.
    /// Use for n_rows >= 4096 (Q proj, O proj) where the 8-warp config saturates SM warp slots.
    ///
    /// Grid = (N, 1, 1). Block = (256, 1, 1). smem = 128 bytes.
    pub fn forward_dp4a_kpar8_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };
        let kpar_fn = match self.cu_q8_dp4a_kpar8 {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("dp4a_kpar8 kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_q8_blocks = k.div_ceil(32);

        let use_scratch = self.q8_qs_scratch.borrow().as_ref()
            .map(|t| t.size() >= k / 4)
            .unwrap_or(false);

        let x_qs_arc: Arc<Tensor<u32>>;
        let x_ds_arc: Arc<Tensor<u32>>;
        let out_arc: Arc<Tensor<f32>>;

        if use_scratch {
            x_qs_arc = self.q8_qs_scratch.borrow().as_ref().unwrap().clone();
            x_ds_arc = self.q8_ds_scratch.borrow().as_ref().unwrap().clone();
            out_arc  = self.q8_out_scratch.borrow().as_ref().unwrap().clone();
        } else {
            x_qs_arc = Arc::new(api::zeros::<u32>(&[k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            x_ds_arc = Arc::new(api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            out_arc  = Arc::new(api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        }
        let x_qs = x_qs_arc.as_ref();
        let x_ds = x_ds_arc.as_ref();
        let out  = out_arc.as_ref();

        let cu_stream = self.stream.cu_stream();

        // -- Launch quantize_q8_1 --------------------------------------------
        {
            let qs_ptr:  u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len:  u64 = (k / 4) as u64;
            let ds_ptr:  u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len:  u64 = k_q8_blocks as u64;
            let x_ptr:   u64 = x.device_pointer().cu_deviceptr();
            let x_len:   u64 = x.size() as u64;
            let k_u32:   u32 = k as u32;

            let mut params: [*mut c_void; 7] = [
                &qs_ptr  as *const _ as *mut c_void,
                &qs_len  as *const _ as *mut c_void,
                &ds_ptr  as *const _ as *mut c_void,
                &ds_len  as *const _ as *mut c_void,
                &x_ptr   as *const _ as *mut c_void,
                &x_len   as *const _ as *mut c_void,
                &k_u32   as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(
                    q8_quant_fn,
                    (grid_x, 1, 1),
                    (BLOCK_DIM, 1, 1),
                    0,
                    cu_stream,
                    &mut params,
                )
            }
            .map_err(|e| KernelError::InvalidConfig(format!("oxide quantize_q8_1 (kpar): {e}")))?;
        }

        // -- Launch q4k_gemv_dp4a_kpar8 (PTX: q4k_gemv_decode_q8_v3_kpar) -------------------------------
        {
            let out_ptr:  u64 = out.device_pointer().cu_deviceptr();
            let out_len:  u64 = n as u64;
            let qs_ptr:   u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len:   u64 = (k / 4) as u64;
            let ds_ptr:   u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len:   u64 = k_q8_blocks as u64;
            let pk_ptr:   u64 = packed.device_pointer().cu_deviceptr();
            let pk_len:   u64 = packed.size() as u64;
            let n_u32:    u32 = n as u32;
            let kb_u32:   u32 = k_blocks as u32;

            let mut params: [*mut c_void; 10] = [
                &out_ptr as *const _ as *mut c_void,
                &out_len as *const _ as *mut c_void,
                &qs_ptr  as *const _ as *mut c_void,
                &qs_len  as *const _ as *mut c_void,
                &ds_ptr  as *const _ as *mut c_void,
                &ds_len  as *const _ as *mut c_void,
                &pk_ptr  as *const _ as *mut c_void,
                &pk_len  as *const _ as *mut c_void,
                &n_u32   as *const _ as *mut c_void,
                &kb_u32  as *const _ as *mut c_void,
            ];
            // smem: 32 f32 slots = 128 bytes (8 warps x 4 mw slots; Blackwell predicated-smem safe).
            let smem_bytes: u32 = 32 * 4;
            unsafe {
                launch_kernel(
                    kpar_fn,
                    (n as u32, 1, 1),
                    (BLOCK_DIM_KPAR, 1, 1),
                    smem_bytes,
                    cu_stream,
                    &mut params,
                )
            }
            .map_err(|e| KernelError::InvalidConfig(format!("oxide q4k_gemv_dp4a_kpar8: {e}")))?;
        }

        Ok(out_arc)
    }

    /// Unified kpar dispatch: uses sm120_dp4a_kpar8 (sm_120, native GGUF) when loaded, else dp4a_kpar8 (sm_80).
    ///
    /// The best variant was selected at construction time via `kpar_dispatch`.  The caller
    /// must ensure `packed` has the matching format: native 36-u32 for sm120, repacked 42-u32
    /// for dp4a_kpar8.  This is guaranteed by `uses_raw_gguf` driving the upload format.
    ///
    /// Grid = (N, 1, 1). Block = (256, 1, 1). smem = 128 bytes.
    pub fn forward_best_kpar_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };
        let kv = match &self.kpar_dispatch {
            Some(v) => v,
            None => return Err(KernelError::InvalidConfig("no kpar kernel available".into())),
        };

        let k = k_blocks * 256;
        let k_q8_blocks = k.div_ceil(32);
        let use_scratch = self.q8_qs_scratch.borrow().as_ref()
            .map(|t| t.size() >= k / 4)
            .unwrap_or(false);

        let x_qs_arc: Arc<Tensor<u32>>;
        let x_ds_arc: Arc<Tensor<u32>>;
        let out_arc: Arc<Tensor<f32>>;
        // With uses_raw_gguf=false (repacked format), K/V are dispatched via
        // dispatch_kv_on_stream (basic f32 GEMV to slot tensors), not through this
        // function. The q8_out_scratch aliasing issue doesn't arise.
        if use_scratch {
            x_qs_arc = self.q8_qs_scratch.borrow().as_ref().unwrap().clone();
            x_ds_arc = self.q8_ds_scratch.borrow().as_ref().unwrap().clone();
            out_arc  = self.q8_out_scratch.borrow().as_ref().unwrap().clone();
        } else {
            x_qs_arc = Arc::new(api::zeros::<u32>(&[k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            x_ds_arc = Arc::new(api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            out_arc  = Arc::new(api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        }
        let x_qs = x_qs_arc.as_ref();
        let x_ds = x_ds_arc.as_ref();
        let out  = out_arc.as_ref();
        let cu_stream = self.stream.cu_stream();

        // -- quantize_q8_1 ---------------------------------------------------
        {
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k / 4) as u64;
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k as u32;
            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("quantize_q8_1 (best_kpar): {e}")))?;
        }

        // -- Dispatch: sm120_dp4a_kpar8 (6 raw-ptr params) or dp4a_kpar8 (10 slice params) ----
        if kv.min_cc >= 1200 {
            // sm120_dp4a_kpar8: weights, act_qs, act_ds, out, n_rows, k_blocks  (6 raw-ptr params)
            let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
            let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
            let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
            let ou_ptr: u64 = out.device_pointer().cu_deviceptr();
            let n_u32:  u32 = n as u32;
            let kb_u32: u32 = k_blocks as u32;
            let mut params: [*mut c_void; 6] = [
                &wt_ptr as *const _ as *mut c_void,
                &qs_ptr as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ou_ptr as *const _ as *mut c_void,
                &n_u32  as *const _ as *mut c_void,
                &kb_u32 as *const _ as *mut c_void,
            ];
            // sm120_dp4a_kpar8 uses STATIC shared memory (.shared .align 4 .b8 smem_buf[128]).
            // Pass sharedMemBytes=128 to match dp4a_kpar8 and ensure CUDA graph captures
            // the correct smem footprint (static+dynamic = 128+128 = 256 total, all OK).
            unsafe {
                launch_kernel(kv.cu_fn, (n as u32, 1, 1), (BLOCK_DIM_KPAR, 1, 1), 128, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("q4k_gemv_sm120_dp4a_kpar8: {e}")))?;
        } else {
            // dp4a_kpar8: out (ptr+len), x_qs (ptr+len), x_ds (ptr+len), packed (ptr+len), n, k_blocks
            let out_ptr: u64 = out.device_pointer().cu_deviceptr();
            let out_len: u64 = n as u64;
            let qs_ptr:  u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len:  u64 = (k / 4) as u64;
            let ds_ptr:  u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len:  u64 = k_q8_blocks as u64;
            let pk_ptr:  u64 = packed.device_pointer().cu_deviceptr();
            let pk_len:  u64 = packed.size() as u64;
            let n_u32:   u32 = n as u32;
            let kb_u32:  u32 = k_blocks as u32;
            let mut params: [*mut c_void; 10] = [
                &out_ptr as *const _ as *mut c_void,
                &out_len as *const _ as *mut c_void,
                &qs_ptr  as *const _ as *mut c_void,
                &qs_len  as *const _ as *mut c_void,
                &ds_ptr  as *const _ as *mut c_void,
                &ds_len  as *const _ as *mut c_void,
                &pk_ptr  as *const _ as *mut c_void,
                &pk_len  as *const _ as *mut c_void,
                &n_u32   as *const _ as *mut c_void,
                &kb_u32  as *const _ as *mut c_void,
            ];
            unsafe {
                launch_kernel(kv.cu_fn, (n as u32, 1, 1), (BLOCK_DIM_KPAR, 1, 1), 128, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("q4k_gemv_dp4a_kpar8 (fallback): {e}")))?;
        }

        Ok(out_arc)
    }

    /// Quantize `x` (f32 K-vector) -> Q8_1 packed int8 + scales, then run Q4K GEMV.
    ///
    /// Allocates temporary Q8_1 buffers, chains two kernel launches on the stream,
    /// syncs, and returns the output tensor.
    pub fn forward_q8_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        // Prefer dp4a_raw (raw GGUF, 14% less DRAM) > dp4a (VDR=2) > dp4a_kpar4 > v1.
        let (q8_quant_fn, q8_gemv_fn, kernel_variant) = match &self.cu_q8_quantize {
            Some(q) => {
                if self.uses_raw_gguf {
                    if let Some(g3r) = &self.cu_q8_dp4a_raw {
                        (*q, *g3r, 4u8)
                    } else if let Some(g3) = &self.cu_q8_dp4a {
                        (*q, *g3, 3u8)
                    } else if let Some(g2) = &self.cu_q8_dp4a_kpar4 {
                        (*q, *g2, 2u8)
                    } else if let Some(g) = &self.cu_q8_gemv {
                        (*q, *g, 1u8)
                    } else {
                        return Err(KernelError::InvalidConfig("Q8 GEMV kernel not loaded".into()));
                    }
                } else if let Some(g3) = &self.cu_q8_dp4a {
                    (*q, *g3, 3u8)
                } else if let Some(g2) = &self.cu_q8_dp4a_kpar4 {
                    (*q, *g2, 2u8)
                } else if let Some(g) = &self.cu_q8_gemv {
                    (*q, *g, 1u8)
                } else {
                    return Err(KernelError::InvalidConfig("Q8 GEMV kernel not loaded".into()));
                }
            }
            _ => return Err(KernelError::InvalidConfig("Q8 kernels not loaded".into())),
        };
        let use_v2 = kernel_variant >= 2;

        let k = k_blocks * 256;  // total activation elements
        let k_q8_blocks = k.div_ceil(32);  // number of Q8_1 blocks

        // Use pre-allocated scratch when available (required during CUDA graph capture --
        // allocating on a capturing stream causes CUDA error 900).  Fall back to fresh
        // allocation only when pre_allocate_q8_scratch has not been called yet.
        let use_scratch = self.q8_qs_scratch.borrow().as_ref()
            .map(|t| t.size() >= k / 4)
            .unwrap_or(false);

        let x_qs_arc: Arc<Tensor<u32>>;
        let x_ds_arc: Arc<Tensor<u32>>;
        let out_arc: Arc<Tensor<f32>>;

        if use_scratch {
            x_qs_arc = self.q8_qs_scratch.borrow().as_ref().unwrap().clone();
            x_ds_arc = self.q8_ds_scratch.borrow().as_ref().unwrap().clone();
            out_arc  = self.q8_out_scratch.borrow().as_ref().unwrap().clone();
        } else {
            x_qs_arc = Arc::new(api::zeros::<u32>(&[k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            x_ds_arc = Arc::new(api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            out_arc  = Arc::new(api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        }
        let x_qs = x_qs_arc.as_ref();
        let x_ds = x_ds_arc.as_ref();
        let out  = out_arc.as_ref();

        let cu_stream = self.stream.cu_stream();

        // -- Launch quantize_q8_1 --------------------------------------------
        {
            let qs_ptr:  u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len:  u64 = (k / 4) as u64;
            let ds_ptr:  u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len:  u64 = k_q8_blocks as u64;
            let x_ptr:   u64 = x.device_pointer().cu_deviceptr();
            let x_len:   u64 = x.size() as u64;
            let k_u32:   u32 = k as u32;

            let mut params: [*mut c_void; 7] = [
                &qs_ptr  as *const _ as *mut c_void,
                &qs_len  as *const _ as *mut c_void,
                &ds_ptr  as *const _ as *mut c_void,
                &ds_len  as *const _ as *mut c_void,
                &x_ptr   as *const _ as *mut c_void,
                &x_len   as *const _ as *mut c_void,
                &k_u32   as *const _ as *mut c_void,
            ];

            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(
                    q8_quant_fn,
                    (grid_x, 1, 1),
                    (BLOCK_DIM, 1, 1),
                    0,
                    cu_stream,
                    &mut params,
                )
            }
            .map_err(|e| KernelError::InvalidConfig(format!("oxide quantize_q8_1: {e}")))?;
        }

        // -- Launch q4k_gemv_decode_q8 ---------------------------------------
        {
            let out_ptr:  u64 = out.device_pointer().cu_deviceptr();
            let out_len:  u64 = n as u64;
            let qs_ptr:   u64 = x_qs.device_pointer().cu_deviceptr();
            let qs_len:   u64 = (k / 4) as u64;
            let ds_ptr:   u64 = x_ds.device_pointer().cu_deviceptr();
            let ds_len:   u64 = k_q8_blocks as u64;
            let pk_ptr:   u64 = packed.device_pointer().cu_deviceptr();
            let pk_len:   u64 = packed.size() as u64;
            let n_u32:    u32 = n as u32;
            let kb_u32:   u32 = k_blocks as u32;

            let mut params: [*mut c_void; 10] = [
                &out_ptr as *const _ as *mut c_void,
                &out_len as *const _ as *mut c_void,
                &qs_ptr  as *const _ as *mut c_void,
                &qs_len  as *const _ as *mut c_void,
                &ds_ptr  as *const _ as *mut c_void,
                &ds_len  as *const _ as *mut c_void,
                &pk_ptr  as *const _ as *mut c_void,
                &pk_len  as *const _ as *mut c_void,
                &n_u32   as *const _ as *mut c_void,
                &kb_u32  as *const _ as *mut c_void,
            ];

            // v2: one block per output neuron (all WARPS_PER_BLOCK warps cooperate on K).
            // v1: ceil(N/WARPS_PER_BLOCK) blocks (one warp per neuron).
            // smem: 32 f32 slots = 128 bytes. The cross-warp reduce uses lanes 0..3 of warp 0
            // to read smem[(w*4+lane)] for w=0..3, lane=0..3 (max index=15, 16 slots needed).
            // BUT on Blackwell sm_120, predicated-but-still-executing lanes 4..31 of warp 0
            // also execute the smem load for their lane index (4..31), requiring 32 slots = 128
            // bytes to avoid trapping on OOB shared-memory access. Use 128 bytes for all
            // v2/v3/v3r kernels to be safe on Blackwell hardware.
            let (grid_x, smem_bytes) = if use_v2 {
                (n as u32, 32 * 4)  // 128 bytes: 32 f32 slots (Blackwell predicated-smem safety)
            } else {
                (n.div_ceil(WARPS_PER_BLOCK as usize) as u32, 0)
            };
            let kernel_name = match kernel_variant {
                3 => "q4k_gemv_decode_q8_v3",
                2 => "q4k_gemv_decode_q8_v2",
                _ => "q4k_gemv_decode_q8",
            };
            unsafe {
                launch_kernel(
                    q8_gemv_fn,
                    (grid_x, 1, 1),
                    (BLOCK_DIM, 1, 1),
                    smem_bytes,
                    cu_stream,
                    &mut params,
                )
            }
            .map_err(|e| KernelError::InvalidConfig(format!("oxide {kernel_name}: {e}")))?;
        }

        // No stream.synchronize() here -- the caller (CUDA graph or direct dispatch)
        // is responsible for synchronization.  Adding a sync inside capture mode
        // would cause CUDA error 900 (operation not permitted when stream is capturing).

        Ok(out_arc)
    }

    /// Fused gate+up GEMV with inline SiLU: one kernel for both FFN projections.
    ///
    /// Quantizes `x` into Q8_1 once, then launches `q4k_gemv_fused_silu_mul` which
    /// computes `out[i] = silu(gate_dot[i]) * up_dot[i]` in a single pass.  Halves
    /// activation DRAM and kernel launches vs two `forward_q8_d2d` + SiLU-mul calls.
    ///
    /// Returns `Err` when the fused kernel is not compiled into the PTX (caller should
    /// fall back to two separate GEMVs + SiLU-mul).
    pub fn forward_fused_silu_dp4a_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        gate_packed: &Arc<Tensor<u32>>,
        up_packed:   &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let fused_fn = match self.cu_fused_silu {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("fused silu kernel not loaded".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_q8_blocks = k.div_ceil(32);

        // Reuse pre-allocated Q8_1 scratch (required during CUDA graph capture).
        let use_scratch = self.q8_qs_scratch.borrow().as_ref()
            .map(|t| t.size() >= k / 4)
            .unwrap_or(false);

        let x_qs_arc: Arc<Tensor<u32>>;
        let x_ds_arc: Arc<Tensor<u32>>;
        let out_arc: Arc<Tensor<f32>>;

        if use_scratch && self.fused_silu_scratch.borrow().is_some() {
            x_qs_arc = self.q8_qs_scratch.borrow().as_ref().unwrap().clone();
            x_ds_arc = self.q8_ds_scratch.borrow().as_ref().unwrap().clone();
            out_arc  = self.fused_silu_scratch.borrow().as_ref().unwrap().clone();
        } else {
            x_qs_arc = Arc::new(api::zeros::<u32>(&[k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            x_ds_arc = Arc::new(api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            out_arc  = Arc::new(api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        }

        let cu_stream = self.stream.cu_stream();

        // -- Launch quantize_q8_1 --------------------------------------------
        {
            let qs_ptr: u64 = x_qs_arc.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k / 4) as u64;
            let ds_ptr: u64 = x_ds_arc.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k as u32;

            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("oxide quantize_q8_1 (fused): {e}")))?;
        }

        // -- Launch q4k_gemv_fused_silu_mul ----------------------------------
        {
            let out_ptr:  u64 = out_arc.device_pointer().cu_deviceptr();
            let out_len:  u64 = n as u64;
            let gate_ptr: u64 = gate_packed.device_pointer().cu_deviceptr();
            let gate_len: u64 = gate_packed.size() as u64;
            let up_ptr:   u64 = up_packed.device_pointer().cu_deviceptr();
            let up_len:   u64 = up_packed.size() as u64;
            let qs_ptr:   u64 = x_qs_arc.device_pointer().cu_deviceptr();
            let qs_len:   u64 = (k / 4) as u64;
            let ds_ptr:   u64 = x_ds_arc.device_pointer().cu_deviceptr();
            let ds_len:   u64 = k_q8_blocks as u64;
            let n_u32:    u32 = n as u32;
            let kb_u32:   u32 = k_blocks as u32;

            let mut params: [*mut c_void; 12] = [
                &out_ptr  as *const _ as *mut c_void,
                &out_len  as *const _ as *mut c_void,
                &gate_ptr as *const _ as *mut c_void,
                &gate_len as *const _ as *mut c_void,
                &up_ptr   as *const _ as *mut c_void,
                &up_len   as *const _ as *mut c_void,
                &qs_ptr   as *const _ as *mut c_void,
                &qs_len   as *const _ as *mut c_void,
                &ds_ptr   as *const _ as *mut c_void,
                &ds_len   as *const _ as *mut c_void,
                &n_u32    as *const _ as *mut c_void,
                &kb_u32   as *const _ as *mut c_void,
            ];
            // 32 f32 smem slots = 128 bytes (Blackwell predicated-smem safety, same as v3)
            let smem_bytes: u32 = 32 * 4;
            unsafe {
                launch_kernel(fused_fn, (n as u32, 1, 1), (BLOCK_DIM, 1, 1), smem_bytes, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("oxide q4k_gemv_fused_silu_mul: {e}")))?;
        }

        Ok(out_arc)
    }

    /// sm_120 hand-tuned fused gate+up GEMV with inline SiLU.
    ///
    /// Runs Q8_1 quantization (reusing pre-allocated scratch), then launches
    /// `q4k_fused_silu_sm120_dp4a` which computes `out[i] = silu(gate[i]) * up[i]`
    /// in a single pass. Preferred over `forward_fused_silu_dp4a_d2d` on Blackwell.
    pub fn forward_sm120_fused_silu_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        gate_packed: &Arc<Tensor<u32>>,
        up_packed:   &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let fused_fn = match self.cu_sm120_fused_silu {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("sm120_fused_silu not loaded".into())),
        };
        let q8_quant_fn = match self.cu_q8_quantize {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q8 quantize kernel not loaded".into())),
        };

        let k = k_blocks * 256;
        let k_q8_blocks = k.div_ceil(32);

        // Reuse pre-allocated scratch tensors (required during CUDA graph capture).
        let use_scratch = self.q8_qs_scratch.borrow().as_ref()
            .map(|t| t.size() >= k / 4)
            .unwrap_or(false);

        let x_qs_arc: Arc<Tensor<u32>>;
        let x_ds_arc: Arc<Tensor<u32>>;
        let out_arc: Arc<Tensor<f32>>;

        if use_scratch && self.fused_silu_scratch.borrow().is_some() {
            x_qs_arc = self.q8_qs_scratch.borrow().as_ref().unwrap().clone();
            x_ds_arc = self.q8_ds_scratch.borrow().as_ref().unwrap().clone();
            out_arc  = self.fused_silu_scratch.borrow().as_ref().unwrap().clone();
        } else {
            x_qs_arc = Arc::new(api::zeros::<u32>(&[k / 4])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            x_ds_arc = Arc::new(api::zeros::<u32>(&[k_q8_blocks])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
            out_arc  = Arc::new(api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?);
        }

        let cu_stream = self.stream.cu_stream();

        // -- Launch quantize_q8_1 (same as oxide fused silu path) -----------
        {
            let qs_ptr: u64 = x_qs_arc.device_pointer().cu_deviceptr();
            let qs_len: u64 = (k / 4) as u64;
            let ds_ptr: u64 = x_ds_arc.device_pointer().cu_deviceptr();
            let ds_len: u64 = k_q8_blocks as u64;
            let x_ptr:  u64 = x.device_pointer().cu_deviceptr();
            let x_len:  u64 = x.size() as u64;
            let k_u32:  u32 = k as u32;

            let mut params: [*mut c_void; 7] = [
                &qs_ptr as *const _ as *mut c_void,
                &qs_len as *const _ as *mut c_void,
                &ds_ptr as *const _ as *mut c_void,
                &ds_len as *const _ as *mut c_void,
                &x_ptr  as *const _ as *mut c_void,
                &x_len  as *const _ as *mut c_void,
                &k_u32  as *const _ as *mut c_void,
            ];
            let grid_x = k_q8_blocks.div_ceil(WARPS_PER_BLOCK as usize) as u32;
            unsafe {
                launch_kernel(q8_quant_fn, (grid_x, 1, 1), (BLOCK_DIM, 1, 1), 0, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("oxide quantize_q8_1 (sm120_fused): {e}")))?;
        }

        // -- Launch q4k_fused_silu_sm120_dp4a --------------------------------
        // Parameters: gate, up, act_qs, act_ds, out, n, k_blocks (7 params, no lengths).
        {
            let gate_ptr: u64 = gate_packed.device_pointer().cu_deviceptr();
            let up_ptr:   u64 = up_packed.device_pointer().cu_deviceptr();
            let qs_ptr:   u64 = x_qs_arc.device_pointer().cu_deviceptr();
            let ds_ptr:   u64 = x_ds_arc.device_pointer().cu_deviceptr();
            let out_ptr:  u64 = out_arc.device_pointer().cu_deviceptr();
            let n_u32:    u32 = n as u32;
            let kb_u32:   u32 = k_blocks as u32;

            let mut params: [*mut c_void; 7] = [
                &gate_ptr as *const _ as *mut c_void,
                &up_ptr   as *const _ as *mut c_void,
                &qs_ptr   as *const _ as *mut c_void,
                &ds_ptr   as *const _ as *mut c_void,
                &out_ptr  as *const _ as *mut c_void,
                &n_u32    as *const _ as *mut c_void,
                &kb_u32   as *const _ as *mut c_void,
            ];
            // smem: 32 f32 slots = 128 bytes (gate[16] + up[16])
            let smem_bytes: u32 = 32 * 4;
            unsafe {
                launch_kernel(fused_fn, (n as u32, 1, 1), (BLOCK_DIM, 1, 1), smem_bytes, cu_stream, &mut params)
            }.map_err(|e| KernelError::InvalidConfig(format!("sm120_fused_silu: {e}")))?;
        }

        Ok(out_arc)
    }
}

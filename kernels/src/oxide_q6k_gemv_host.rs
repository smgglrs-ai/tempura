/// Host-side wrapper for the oxide-compiled Q6_K GEMV kernel.
///
/// The kernel is compiled with `cargo oxide build` from `tools/oxide-gemv-q6k/`.
/// `build.rs` extracts the embedded PTX from the compiled binary and writes
/// it to `$OUT_DIR/oxide_q6k_gemv.ptx`. This module embeds that PTX at
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
use cuda_core::{launch_kernel, Stream};
use cuda_core::sys::{CUfunction, CUmodule};
use cutile::{api, prelude::DeviceOp, tensor::Tensor};
use std::cell::RefCell;
use std::ffi::{c_void, CString};
use std::sync::Arc;

/// The PTX source compiled by `cargo oxide build tools/oxide-gemv-q6k`.
#[cfg(feature = "oxide-gemv")]
static OXIDE_Q6K_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_q6k_gemv.ptx"));

const KERNEL_ENTRY: &str = "q6k_gemv_dp4a";
const KERNEL_ENTRY_KPAR: &str = "q6k_gemv_dp4a_kpar8";
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128
const WARPS_PER_BLOCK_KPAR: u32 = 8;
const BLOCK_DIM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 32; // 256
const SMEM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 4; // 32 bytes
/// Launches the oxide-compiled Q6_K GEMV kernel for M=1 or M=2 decode.
pub struct OxideQ6kGemv {
    cu_module:              CUmodule,
    cu_function:            CUfunction,
    cu_function_kpar:       Option<CUfunction>,
    /// sm_120 hand-written PTX module (separate from oxide module).
    cu_sm120_module:        Option<CUmodule>,
    /// q6k_gemv_sm120_kpar8: cp.async triple-buffer + register-exchange
    /// variant (sm_120+). Default since 2026-10-29.
    cu_function_sm120_mbar: Option<CUfunction>,
    /// q6k_gemv_sm120_kpar8_db: dual-block DIRECT-LOAD variant (same module,
    /// second entry) -- no cp.async/smem/barriers in the loop, 2x MLP, L2
    /// prefetch. Preferred over kpar8 for ANY k_blocks since 2026-10-29
    /// (TEMPURA_NO_Q6K_DB=1 bisects back to the cp.async variant).
    cu_function_sm120_db:   Option<CUfunction>,
    /// sm_120 batch module: q6k_gemv_sm120_batch (M rows, shared weight reads).
    cu_sm120_batch_module:  Option<CUmodule>,
    /// q6k_gemv_sm120_batch: true-batch kernel, one block per output neuron. (sm_120+).
    cu_function_sm120_batch: Option<CUfunction>,
    /// sm_120 db-batch module: q6k_gemv_sm120_kpar8_db_batch (the kpar8_db
    /// warp structure with an m-loop -- bit-identical per row to the per-row
    /// db GEMV; the no-graph serving batch path).
    cu_sm120_db_batch_module: Option<CUmodule>,
    cu_function_sm120_db_batch: Option<CUfunction>,
    /// q6k_mma128_batch_sm120: LARGE-m INT8 tensor-core batch (m <= 128,
    /// weights once, Q8_1 activations).
    cu_mma128_module: Option<CUmodule>,
    cu_mma128_batch: Option<CUfunction>,
    /// q6k_mmq_batch_sm120: the llama.cpp-mmq-structured LARGE-m INT8 batch
    /// (128 neurons x 32 tokens per CTA, cp.async smem staging, f16 (d*sc)
    /// pre-extraction; the 2026-11-20 port of the q4k mmq winner).
    cu_mmq_module: Option<CUmodule>,
    cu_mmq_batch: Option<CUfunction>,
    cu_mmq_batch_j64: Option<CUfunction>,
    cu_mmq_batch_j128: Option<CUfunction>,
    /// q6k_mmq_split_reduce: the fixed-order grid.z partials combine (the
    /// 2026-11-21 q6k k-split port).
    cu_mmq_reduce: Option<CUfunction>,
    /// Pre-allocated kpar output scratch -- avoids allocation during CUDA graph capture.
    /// Set via `pre_allocate_kpar_output_scratch(max_n)` before first graph capture.
    kpar_out_scratch: RefCell<Option<Arc<Tensor<f32>>>>,
    stream:                 Arc<Stream>,
}

unsafe impl Send for OxideQ6kGemv {}
unsafe impl Sync for OxideQ6kGemv {}

impl Drop for OxideQ6kGemv {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
        }
        if let Some(m) = self.cu_sm120_module.take() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
        }
        if let Some(m) = self.cu_sm120_batch_module.take() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
        }
        if let Some(m) = self.cu_sm120_db_batch_module.take() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
        }
        if let Some(m) = self.cu_mma128_module.take() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
        }
        if let Some(m) = self.cu_mmq_module.take() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
        }
    }
}

impl OxideQ6kGemv {
    /// Opt-in dynamic shared memory ceiling for the sm120 batch kernel
    /// (set via `cuFuncSetAttribute` at load). 64 KiB covers k_blocks <= 305
    /// (down_proj K up to ~78k); larger shapes fall back to per-row GEMVs.
    const MAX_BATCH_DYNAMIC_SMEM: usize = 64 * 1024;

    /// Load Q6_K GEMV kernels.
    ///
    /// Attempts to load the oxide-compiled PTX (baseline + kpar8) and, on sm_120+,
    /// the hand-written cp.async kernel (`q6k_gemv_sm120_kpar8`).
    /// Returns `None` only when no kernel variant could be loaded at all.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        let _ = stream.context().bind_to_thread().ok()?;
        const CUDA_SUCCESS: u32 = 0;

        // -- Try oxide-compiled PTX (feature-gated) -----------------------------
        #[cfg(not(feature = "oxide-gemv"))]
        let (cu_module, cu_function, cu_function_kpar): (CUmodule, CUfunction, Option<CUfunction>) =
            (std::ptr::null_mut(), std::ptr::null_mut(), None);

        #[cfg(feature = "oxide-gemv")]
        let (cu_module, cu_function, cu_function_kpar) = {
            let oxide_ok = !OXIDE_Q6K_PTX.is_empty();
            if !oxide_ok {
                eprintln!("oxide-gemv-q6k: embedded PTX is empty; oxide variants unavailable");
                (std::ptr::null_mut(), std::ptr::null_mut(), None)
            } else {
                let ptx_str = match std::str::from_utf8(OXIDE_Q6K_PTX) {
                    Ok(s) => s,
                    Err(_) => {
                        eprintln!("oxide-gemv-q6k: PTX is not valid UTF-8");
                        return None;
                    }
                };
                let c_ptx = match CString::new(ptx_str) {
                    Ok(c) => c,
                    Err(_) => return None,
                };
                let cu_module = unsafe {
                    let mut module = std::mem::MaybeUninit::uninit();
                    let ret = cuda_core::sys::cuModuleLoadData(
                        module.as_mut_ptr(),
                        c_ptx.as_ptr() as *const _,
                    );
                    if ret != CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q6k: cuModuleLoadData failed (code={ret}); \
                             oxide variants unavailable"
                        );
                        std::ptr::null_mut()
                    } else {
                        module.assume_init()
                    }
                };
                if cu_module.is_null() {
                    (std::ptr::null_mut(), std::ptr::null_mut(), None)
                } else {
                    let entry = match CString::new(KERNEL_ENTRY) {
                        Ok(c) => c,
                        Err(_) => return None,
                    };
                    let cu_function = unsafe {
                        let mut func = std::mem::MaybeUninit::uninit();
                        let ret = cuda_core::sys::cuModuleGetFunction(
                            func.as_mut_ptr(),
                            cu_module,
                            entry.as_ptr(),
                        );
                        if ret != CUDA_SUCCESS {
                            eprintln!(
                                "oxide-gemv-q6k: cuModuleGetFunction('{KERNEL_ENTRY}') \
                                 failed (code={ret}); oxide baseline unavailable"
                            );
                            std::ptr::null_mut()
                        } else {
                            func.assume_init()
                        }
                    };
                    let cu_function_kpar =
                        CString::new(KERNEL_ENTRY_KPAR).ok().and_then(|entry_kpar| {
                            let mut func = std::mem::MaybeUninit::uninit();
                            let ret = unsafe {
                                cuda_core::sys::cuModuleGetFunction(
                                    func.as_mut_ptr(),
                                    cu_module,
                                    entry_kpar.as_ptr(),
                                )
                            };
                            if ret == CUDA_SUCCESS {
                                eprintln!(
                                    "oxide-gemv-q6k: loaded kpar variant (entry={})",
                                    KERNEL_ENTRY_KPAR
                                );
                                Some(unsafe { func.assume_init() })
                            } else {
                                eprintln!(
                                    "oxide-gemv-q6k: kpar variant not found \
                                     (code={ret}); using dp4a baseline"
                                );
                                None
                            }
                        });
                    (cu_module, cu_function, cu_function_kpar)
                }
            }
        };

        // -- Try sm_120 cp.async PTX (not feature-gated, any sm_120 GPU) ------
        // Requires Blackwell GPU (cc >= 1200). Falls back to cu_function_kpar
        // on older GPUs.
        //
        // DEFAULT ON since 2026-10-29 (register-exchange rewrite): the
        // kernel's loop no longer contains any CTA-wide barriers. Each lane
        // reads only the smem words it copied itself (per-thread
        // wait_group + fence.proxy.async), cross-lane values arrive via
        // shfl.sync.idx register exchange, and the cross-warp partial
        // handoff is a generic-proxy release/acquire counter. The prior
        // variants' races (elided warp-uniform mbarrier pairs,
        // DEFER_BLOCKING bar.sync, the kb=16 residual) all lived in the
        // cross-lane async->generic smem publication this design removes.
        // TEMPURA_NO_SM120_Q6K=1 opts out to the dp4a kpar baseline.
        let sm120_force_off = std::env::var("TEMPURA_NO_SM120_Q6K")
            .map(|v| v == "1")
            .unwrap_or(false);
        let sm120_disabled = sm120_force_off;

        let sm120_cc = {
            let caps = crate::gpu_capabilities::GpuCapabilities::query(
                stream.context().device().cu_device()
            );
            caps.cc()
        };
        let sm120_cc_ok = sm120_cc >= crate::q6k_gemv_v4_ptx::MIN_CC;

        let (cu_sm120_module, cu_function_sm120_mbar, cu_function_sm120_db_raw) =
            if !sm120_disabled && sm120_cc_ok {
            let ptx_cstr = CString::new(crate::q6k_gemv_v4_ptx::PTX)
                .unwrap_or_default();
            let mut sm120_mod = std::mem::MaybeUninit::uninit();
            let rc = unsafe {
                cuda_core::sys::cuModuleLoadData(
                    sm120_mod.as_mut_ptr(),
                    ptx_cstr.as_ptr() as *const _,
                )
            };
            if rc == CUDA_SUCCESS {
                let sm120_mod = unsafe { sm120_mod.assume_init() };
                let entry_cstr = CString::new(crate::q6k_gemv_v4_ptx::ENTRY)
                    .unwrap_or_default();
                let mut fn_out = std::mem::MaybeUninit::uninit();
                let rc2 = unsafe {
                    cuda_core::sys::cuModuleGetFunction(
                        fn_out.as_mut_ptr(),
                        sm120_mod,
                        entry_cstr.as_ptr(),
                    )
                };
                if rc2 == CUDA_SUCCESS {
                    let kpar_fn = unsafe { fn_out.assume_init() };
                    eprintln!(
                        "oxide-gemv-q6k: loaded sm120 kpar8 variant \
                         (entry={}, cc={sm120_cc})",
                        crate::q6k_gemv_v4_ptx::ENTRY,
                    );
                    // Second entry in the same module: the dual-block
                    // direct-load variant. Best-effort -- select_kpar_fn
                    // falls back to the cp.async variant when absent.
                    let db_fn = {
                        let db_cstr = CString::new(crate::q6k_gemv_v4_ptx::ENTRY_DB)
                            .unwrap_or_default();
                        let mut db_out = std::mem::MaybeUninit::uninit();
                        let rc3 = unsafe {
                            cuda_core::sys::cuModuleGetFunction(
                                db_out.as_mut_ptr(),
                                sm120_mod,
                                db_cstr.as_ptr(),
                            )
                        };
                        if rc3 == CUDA_SUCCESS {
                            eprintln!(
                                "oxide-gemv-q6k: loaded sm120 kpar8_db variant \
                                 (entry={}, cc={sm120_cc})",
                                crate::q6k_gemv_v4_ptx::ENTRY_DB,
                            );
                            Some(unsafe { db_out.assume_init() })
                        } else {
                            eprintln!(
                                "oxide-gemv-q6k: cuModuleGetFunction(kpar8_db) \
                                 failed (rc={rc3}); using the cp.async variant"
                            );
                            None
                        }
                    };
                    (Some(sm120_mod), Some(kpar_fn), db_fn)
                } else {
                    eprintln!(
                        "oxide-gemv-q6k: cuModuleGetFunction(sm120_mbar) \
                         failed (rc={rc2}); using kpar8 baseline"
                    );
                    let _ = unsafe { cuda_core::sys::cuModuleUnload(sm120_mod) };
                    (None, None, None)
                }
            } else {
                eprintln!(
                    "oxide-gemv-q6k: cuModuleLoadData(sm120 PTX) \
                     failed (rc={rc}); using kpar8 baseline"
                );
                (None, None, None)
            }
        } else if sm120_force_off {
            eprintln!("oxide-gemv-q6k: TEMPURA_NO_SM120_Q6K=1, skipping sm120 kpar8");
            (None, None, None)
        } else {
            (None, None, None)
        };

        // -- Try sm_120 batch PTX (q6k_gemv_sm120_batch, M rows 2..=16) ----------
        // Loads independently of the kpar8 GEMV above: the batch kernel is a
        // different code path (two-phase, verified by bench-batch replay) and
        // must stay available even when the kpar8 GEMV is disabled.
        let (cu_sm120_batch_module, cu_function_sm120_batch) =
            if !sm120_force_off && sm120_cc_ok {
                let ptx_cstr = CString::new(crate::q6k_batch_sm120_ptx::PTX)
                    .unwrap_or_default();
                let mut batch_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        batch_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let batch_mod = unsafe { batch_mod.assume_init() };
                    let entry_cstr = CString::new(crate::q6k_batch_sm120_ptx::ENTRY)
                        .unwrap_or_default();
                    let mut fn_out = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(
                            fn_out.as_mut_ptr(),
                            batch_mod,
                            entry_cstr.as_ptr(),
                        )
                    };
                    if rc2 == CUDA_SUCCESS {
                        let batch_fn = unsafe { fn_out.assume_init() };
                        // Raise the dynamic-smem opt-in ceiling for the batch
                        // kernel (weights + reduction area live in dynamic smem).
                        // Best-effort: on failure the kernel still works for
                        // shapes under the default 48 KiB limit.
                        let rc_attr = unsafe {
                            cuda_core::sys::cuFuncSetAttribute(
                                batch_fn,
                                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                Self::MAX_BATCH_DYNAMIC_SMEM as i32,
                            )
                        };
                        if rc_attr != CUDA_SUCCESS {
                            eprintln!(
                                "oxide-gemv-q6k: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) \
                                 failed (rc={rc_attr}); batch kernel limited to 48 KiB smem"
                            );
                        }
                        eprintln!(
                            "oxide-gemv-q6k: loaded sm120_batch variant (entry={})",
                            crate::q6k_batch_sm120_ptx::ENTRY,
                        );
                        (Some(batch_mod), Some(batch_fn))
                    } else {
                        eprintln!(
                            "oxide-gemv-q6k: cuModuleGetFunction(sm120_batch) \
                             failed (rc={rc2}); batch variant unavailable"
                        );
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(batch_mod) };
                        (None, None)
                    }
                } else {
                    eprintln!(
                        "oxide-gemv-q6k: cuModuleLoadData(sm120_batch PTX) \
                         failed (rc={rc}); batch variant unavailable"
                    );
                    (None, None)
                }
            } else {
                (None, None)
            };

        // -- Try sm_120 db-batch PTX (q6k_gemv_sm120_kpar8_db_batch) ----------
        // The kpar8_db warp structure with an m-loop: bit-identical per row
        // to the per-row db GEMV (the no-graph serving batch path; see
        // bench-probe-q6k-batch). Loads independently of the kernels above.
        let (cu_sm120_db_batch_module, cu_function_sm120_db_batch) =
            if !sm120_force_off && sm120_cc_ok {
                let ptx_cstr = CString::new(crate::q6k_gemv_v4_ptx::PTX_DB_BATCH)
                    .unwrap_or_default();
                let mut db_batch_mod = std::mem::MaybeUninit::uninit();
                let rc = unsafe {
                    cuda_core::sys::cuModuleLoadData(
                        db_batch_mod.as_mut_ptr(),
                        ptx_cstr.as_ptr() as *const _,
                    )
                };
                if rc == CUDA_SUCCESS {
                    let db_batch_mod = unsafe { db_batch_mod.assume_init() };
                    let entry_cstr = CString::new(crate::q6k_gemv_v4_ptx::ENTRY_DB_BATCH)
                        .unwrap_or_default();
                    let mut fn_out = std::mem::MaybeUninit::uninit();
                    let rc2 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(
                            fn_out.as_mut_ptr(),
                            db_batch_mod,
                            entry_cstr.as_ptr(),
                        )
                    };
                    if rc2 == CUDA_SUCCESS {
                        let db_batch_fn = unsafe { fn_out.assume_init() };
                        eprintln!(
                            "oxide-gemv-q6k: loaded sm120_db_batch variant (entry={})",
                            crate::q6k_gemv_v4_ptx::ENTRY_DB_BATCH,
                        );
                        (Some(db_batch_mod), Some(db_batch_fn))
                    } else {
                        eprintln!(
                            "oxide-gemv-q6k: cuModuleGetFunction(sm120_db_batch) \
                             failed (rc={rc2}); db_batch variant unavailable"
                        );
                        let _ = unsafe { cuda_core::sys::cuModuleUnload(db_batch_mod) };
                        (None, None)
                    }
                } else {
                    eprintln!(
                        "oxide-gemv-q6k: cuModuleLoadData(sm120_db_batch PTX) \
                         failed (rc={rc}); db_batch variant unavailable"
                    );
                    (None, None)
                }
            } else {
                (None, None)
            };

        // -- Try sm_120 LARGE-m MMA batch PTX (q6k_mma128_batch_sm120) --------
        // INT8 tensor cores, m <= 128 rows per launch, weights once, Q8_1
        // activations (llama.cpp Q6_K x Q8_K class -- NOT the f32-activation
        // class of the kernels above; used only by the exact-prefill route).
        let (cu_mma128_module, cu_mma128_batch) = if !sm120_force_off && sm120_cc_ok {
            let ptx_cstr =
                CString::new(crate::q6k_mma128_batch_sm120_ptx::PTX).unwrap_or_default();
            let mut m = std::mem::MaybeUninit::uninit();
            let rc = unsafe {
                cuda_core::sys::cuModuleLoadData(m.as_mut_ptr(), ptx_cstr.as_ptr() as *const _)
            };
            if rc == CUDA_SUCCESS {
                let m = unsafe { m.assume_init() };
                let entry_cstr =
                    CString::new(crate::q6k_mma128_batch_sm120_ptx::ENTRY).unwrap_or_default();
                let mut f = std::mem::MaybeUninit::uninit();
                let rc2 = unsafe {
                    cuda_core::sys::cuModuleGetFunction(f.as_mut_ptr(), m, entry_cstr.as_ptr())
                };
                if rc2 == CUDA_SUCCESS {
                    eprintln!(
                        "oxide-gemv-q6k: loaded sm_120 MMA128 batch PTX ({} bytes) -- INT8 tensor cores, m<=128, weights once, Q8_1 activations",
                        crate::q6k_mma128_batch_sm120_ptx::PTX.len(),
                    );
                    (Some(m), Some(unsafe { f.assume_init() }))
                } else {
                    eprintln!(
                        "oxide-gemv-q6k: cuModuleGetFunction(mma128_batch) failed (rc={rc2})"
                    );
                    let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
                    (None, None)
                }
            } else {
                eprintln!(
                    "oxide-gemv-q6k: cuModuleLoadData(mma128_batch PTX) failed (rc={rc})"
                );
                (None, None)
            }
        } else {
            (None, None)
        };

        // -- Try sm_120 MMQ-structured batch PTX (q6k_mmq_batch_sm120) -------
        // The 2026-11-20 port of the q4k mmq winner: 128 neurons x 32 tokens
        // per CTA, cp.async smem staging, f16 (d*sc) pre-extraction --
        // different numerics class from mma128 (the f16 scale roundings);
        // gated by the mmq probe arm (CPU mirror + class band) + MMLU/ppl.
        let (cu_mmq_module, cu_mmq_batch, cu_mmq_reduce, cu_mmq_j64, cu_mmq_j128) =
            if !sm120_force_off && sm120_cc_ok {
            let ptx_cstr =
                CString::new(crate::q6k_mmq_batch_sm120_ptx::ptx()).unwrap_or_default();
            let mut m = std::mem::MaybeUninit::uninit();
            let rc = unsafe {
                cuda_core::sys::cuModuleLoadData(m.as_mut_ptr(), ptx_cstr.as_ptr() as *const _)
            };
            if rc == CUDA_SUCCESS {
                let m = unsafe { m.assume_init() };
                let entry = CString::new(crate::q6k_mmq_batch_sm120_ptx::ENTRY).unwrap_or_default();
                let mut f = std::mem::MaybeUninit::uninit();
                let rc2 = unsafe {
                    cuda_core::sys::cuModuleGetFunction(f.as_mut_ptr(), m, entry.as_ptr())
                };
                if rc2 == CUDA_SUCCESS {
                    let mmq_fn = unsafe { f.assume_init() };
                    // 37888 B dynamic smem is under the 48 KiB static limit,
                    // but the opt-in matches the q4k mmq pattern (best-effort;
                    // on failure the launch errors and callers fall back).
                    let rc_attr = unsafe {
                        cuda_core::sys::cuFuncSetAttribute(
                            mmq_fn,
                            cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            crate::q6k_mmq_batch_sm120_ptx::DYNAMIC_SMEM as i32,
                        )
                    };
                    if rc_attr != CUDA_SUCCESS {
                        eprintln!(
                            "oxide-gemv-q6k: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) \
                             failed for mmq_batch (rc={rc_attr})"
                        );
                    }
                    eprintln!(
                        "oxide-gemv-q6k: loaded sm_120 MMQ batch PTX ({} bytes) -- INT8 tensor cores, 128 neurons x 32 tokens/block, cp.async staging, f16 scales",
                        ptx_cstr.as_bytes().len(),
                    );
                    // Fixed-order split-reduce entry (same module; optional --
                    // has_mmq_split_batch gates the split path on it).
                    let reduce_entry =
                        CString::new(crate::q6k_mmq_batch_sm120_ptx::ENTRY_REDUCE)
                            .unwrap_or_default();
                    let mut rf = std::mem::MaybeUninit::uninit();
                    let rc3 = unsafe {
                        cuda_core::sys::cuModuleGetFunction(rf.as_mut_ptr(), m, reduce_entry.as_ptr())
                    };
                    let cu_mmq_reduce = if rc3 == CUDA_SUCCESS {
                        Some(unsafe { rf.assume_init() })
                    } else {
                        eprintln!(
                            "oxide-gemv-q6k: cuModuleGetFunction(mmq_split_reduce) failed (rc={rc3})"
                        );
                        None
                    };
                    // J=64 / J=128 wide-token variants (same module,
                    // 2026-11-22): missing entries just disable them.
                    let get_j = |entry: &str, smem: usize| -> Option<CUfunction> {
                        let c = CString::new(entry).unwrap_or_default();
                        let mut jf = std::mem::MaybeUninit::uninit();
                        let rc = unsafe {
                            cuda_core::sys::cuModuleGetFunction(jf.as_mut_ptr(), m, c.as_ptr())
                        };
                        if rc != CUDA_SUCCESS {
                            return None;
                        }
                        let jf = unsafe { jf.assume_init() };
                        let rc_attr = unsafe {
                            cuda_core::sys::cuFuncSetAttribute(
                                jf,
                                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                smem as i32,
                            )
                        };
                        if rc_attr != CUDA_SUCCESS {
                            return None;
                        }
                        Some(jf)
                    };
                    let j64 = get_j(
                        crate::q6k_mmq_batch_sm120_ptx::ENTRY_J64,
                        crate::q6k_mmq_batch_sm120_ptx::dynamic_smem(64),
                    );
                    let j128 = get_j(
                        crate::q6k_mmq_batch_sm120_ptx::ENTRY_J128,
                        crate::q6k_mmq_batch_sm120_ptx::dynamic_smem(128),
                    );
                    (Some(m), Some(mmq_fn), cu_mmq_reduce, j64, j128)
                } else {
                    let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
                    eprintln!(
                        "oxide-gemv-q6k: cuModuleGetFunction(mmq_batch) failed (rc={rc2})"
                    );
                    (None, None, None, None, None)
                }
            } else {
                eprintln!(
                    "oxide-gemv-q6k: cuModuleLoadData(mmq_batch PTX) failed (rc={rc})"
                );
                (None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None)
        };

        // Return Some only when at least one kernel is available.
        let any_loaded = !cu_function.is_null()
            || cu_function_kpar.is_some()
            || cu_function_sm120_mbar.is_some();
        if !any_loaded {
            // Unload the oxide module if it was loaded (sm120 failed too).
            if !cu_module.is_null() {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(cu_module) };
            }
            if let Some(m) = cu_sm120_batch_module {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
            if let Some(m) = cu_sm120_db_batch_module {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
            if let Some(m) = cu_mma128_module {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
            if let Some(m) = cu_mmq_module {
                let _ = unsafe { cuda_core::sys::cuModuleUnload(m) };
            }
            eprintln!("oxide-gemv-q6k: no Q6K kernel variant available; falling back to cuTile");
            return None;
        }

        eprintln!(
            "oxide-gemv-q6k: Q6K kernels loaded (oxide_fn={}, kpar={}, sm120_mbar={}, sm120_batch={})",
            !cu_function.is_null(),
            cu_function_kpar.is_some(),
            cu_function_sm120_mbar.is_some(),
            cu_function_sm120_batch.is_some(),
        );
        Some(Self {
            cu_module,
            cu_function,
            cu_function_kpar,
            cu_sm120_module,
            cu_function_sm120_mbar,
            cu_function_sm120_db: if std::env::var("TEMPURA_NO_Q6K_DB")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                eprintln!("oxide-gemv-q6k: TEMPURA_NO_Q6K_DB=1, using the cp.async kpar8 variant");
                None
            } else {
                cu_function_sm120_db_raw
            },
            cu_sm120_batch_module,
            cu_function_sm120_batch,
            cu_sm120_db_batch_module,
            cu_function_sm120_db_batch,
            cu_mma128_module,
            cu_mma128_batch,
            cu_mmq_module,
            cu_mmq_batch,
            cu_mmq_batch_j64: cu_mmq_j64,
            cu_mmq_batch_j128: cu_mmq_j128,
            cu_mmq_reduce,
            kpar_out_scratch: RefCell::new(None),
            stream,
        })
    }

    /// Returns `true` when the kpar kernel (dp4a_kpar8 or sm120_mbar) is available.
    pub fn has_kpar(&self) -> bool {
        self.cu_function_sm120_mbar.is_some() || self.cu_function_kpar.is_some()
    }

    /// Returns `true` when the sm_120 cp.async mbar kpar variant was loaded.
    pub fn has_sm120_mbar(&self) -> bool {
        self.cu_function_sm120_mbar.is_some()
    }

    /// Returns `true` when the sm_120 batch variant (M rows, 2..=16) was loaded.
    pub fn has_sm120_batch(&self) -> bool {
        self.cu_function_sm120_batch.is_some()
    }

    /// Returns `true` when the sm_120 db-batch variant (the kpar8_db warp
    /// structure with an m-loop, bit-identical per row to the per-row db
    /// GEMV) was loaded.
    pub fn has_sm120_db_batch(&self) -> bool {
        self.cu_function_sm120_db_batch.is_some()
    }

    /// Returns `true` when the sm_120 LARGE-m MMA batch kernel
    /// (q6k_mma128_batch_sm120) was loaded. NOTE: this kernel takes Q8_1-
    /// quantized activations (llama.cpp's Q6_K x Q8_K class), NOT raw f32
    /// like the other batch kernels -- see
    /// [`Self::forward_mma128_batch_d2d_rows`].
    pub fn has_mma128_batch(&self) -> bool {
        self.cu_mma128_batch.is_some()
    }

    /// Returns `true` when the sm_120 MMQ-structured LARGE-m batch kernel
    /// (`q6k_mmq_batch_sm120`) was loaded. Same activation contract as
    /// `forward_mma128_batch_d2d_rows` (Q8_1-quantized, llama.cpp's
    /// Q6_K x Q8_K class) but the f16 (d*sc) scale-product numerics class
    /// (see [`Self::forward_mmq_batch_d2d_rows`]).
    pub fn has_mmq_split_batch(&self) -> bool {
        self.cu_mmq_batch.is_some() && self.cu_mmq_reduce.is_some()
    }

    pub fn has_mmq_batch(&self) -> bool {
        self.cu_mmq_batch.is_some()
    }

    /// sm_120 M-row batch db GEMV (native GGUF Q6K) with row-window offsets:
    /// computes rows `[x_row, x_row + m)` of X into rows `[out_row, out_row +
    /// m)` of OUT against the SAME packed weights.
    ///
    /// The kernel reads each weight block ONCE for all m rows and is
    /// BIT-IDENTICAL per row to the per-row `q6k_gemv_sm120_kpar8_db` GEMV
    /// (see [`q6k_gemv_v4_ptx::PTX_DB_BATCH`]), so chunking a batch of
    /// M > 16 rows into ceil-chunks of <= 16 rows leaves every row's
    /// numerics unchanged -- no batch-composition dependence. `m` must be
    /// in 1..=16 (local-memory row budget).
    ///
    /// `x` is flat [m_total * k] f32 row-major (the window starts at
    /// `x_row`); `packed` is native GGUF Q6K [n * k_blocks * 53 u32];
    /// `out` is [m_total * n] f32 (rows at `out_row`).
    pub fn forward_sm120_db_batch_d2d_rows(
        &self,
        x: &Arc<Tensor<f32>>,
        x_row: usize,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        out_row: usize,
        m: usize,
        n: usize,
        k_blocks: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<()> {
        if m == 0 || m > 16 {
            return Err(KernelError::InvalidConfig(format!(
                "forward_sm120_db_batch_d2d_rows: m must be in 1..=16 (64 B local row \
                 budget); got {m}"
            )));
        }
        let k = k_blocks * 256;
        let row_bytes = k * 4;
        let x_row_bytes = x_row * row_bytes;
        let out_row_bytes = out_row * n * 4;
        if x_row_bytes + m * row_bytes > x.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "forward_sm120_db_batch_d2d_rows: x slice [{x_row}, {x_row}+{m}) x {k} \
                 exceeds tensor ({} f32)",
                x.size()
            )));
        }
        if out_row_bytes + m * n * 4 > out.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "forward_sm120_db_batch_d2d_rows: out slice [{out_row}, {out_row}+{m}) x {n} \
                 exceeds tensor ({} f32)",
                out.size()
            )));
        }
        let db_batch_fn = self
            .cu_function_sm120_db_batch
            .ok_or_else(|| KernelError::InvalidConfig("q6k_gemv_sm120_kpar8_db_batch not loaded".into()))?;

        let out_ptr: u64 = out.device_pointer().cu_deviceptr() as u64 + out_row_bytes as u64;
        let out_len: u64 = out.size() as u64 - (out_row_bytes / 4) as u64;
        let x_ptr: u64 = x.device_pointer().cu_deviceptr() as u64 + x_row_bytes as u64;
        let x_len: u64 = x.size() as u64 - (x_row_bytes / 4) as u64;
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;
        let m_u32 = m as u32;

        let mut params: [*mut c_void; 9] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &x_len        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
            &m_u32        as *const _ as *mut c_void,
        ];

        unsafe {
            launch_kernel(
                db_batch_fn,
                (n as u32, 1, 1),
                (BLOCK_DIM_KPAR, 1, 1), // 8 warps (kpar8_db structure)
                0,                      // static smem in PTX (520 bytes)
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k_gemv_sm120_kpar8_db_batch: {e}")))?;
        Ok(())
    }

    /// Batch Q6K GEMV for M-row decode (2..=16): reads each weight block ONCE
    /// from DRAM/L2 (into shared memory) and computes all M dot products.
    ///
    /// Grid: (n, 1, 1). Block: (256, 1, 1). Dynamic smem: k_blocks*212 + 512.
    ///
    /// - `x_m`: `[M * k_blocks * 256]` f32 -- M activation rows, row-major.
    /// - `packed`: `[n * k_blocks * 53]` u32 -- Q6K weight blocks.
    /// - `out_m`: `[M * n]` f32 -- row r at `[r*n .. (r+1)*n)`.
    /// - `n`: output neurons per row.
    /// - `k_blocks`: Q6K blocks per activation row (any value; no %8 constraint).
    /// - `m`: batch rows, 2..=16 (rows above 16 would overflow the reduction area).
    /// - `cu_stream`: CUDA stream (must be the graph-capture stream when inside capture).
    ///
    /// No allocation; safe inside `cuStreamBeginCapture` / `cuStreamEndCapture`.
    pub fn forward_sm120_batch_d2d_into(
        &self,
        x_m: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out_m: &Arc<Tensor<f32>>,
        n: usize,
        k_blocks: usize,
        m: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<()> {
        if m < 2 || m > 16 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch requires m in 2..=16 (got {m})"
            )));
        }
        let smem_bytes = crate::q6k_batch_sm120_ptx::dynamic_smem_bytes(k_blocks);
        if smem_bytes > Self::MAX_BATCH_DYNAMIC_SMEM {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch needs {smem_bytes} B dynamic smem (> {}); \
                 falling back to per-row GEMVs",
                Self::MAX_BATCH_DYNAMIC_SMEM
            )));
        }
        let batch_fn = self.cu_function_sm120_batch.ok_or_else(|| {
            KernelError::InvalidConfig("q6k_gemv_sm120_batch not loaded".into())
        })?;

        let out_ptr: u64 = out_m.device_pointer().cu_deviceptr();
        let out_len: u64 = out_m.size() as u64;
        let x_ptr: u64 = x_m.device_pointer().cu_deviceptr();
        let x_len: u64 = x_m.size() as u64;
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;
        let m_u32 = m as u32;

        let mut params: [*mut c_void; 9] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &x_len        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
            &m_u32        as *const _ as *mut c_void,
        ];

        unsafe {
            launch_kernel(
                batch_fn,
                (n as u32, 1, 1),
                (BLOCK_DIM_KPAR, 1, 1),
                smem_bytes as u32,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k_gemv_sm120_batch: {e}")))?;
        Ok(())
    }

    /// LARGE-m MMA batch GEMM (m in 1..=128): the m16n8k32 tensor-core path
    /// with the weight matrix read ONCE per launch (8 statically-unrolled
    /// m16 tiles, register accumulators). Q8_1-ACTIVATION class (llama.cpp
    /// Q6_K x Q8_K) -- NOT raw f32 like [`Self::forward_sm120_batch_rows`];
    /// the caller quantizes via the q4k host's `quantize_q8_1_window` (the
    /// quantize kernel lives in the q4k PTX module).
    ///
    /// Per-row numerics are independent of the row window (rows predicated
    /// on the tile guard), so chunking a batch of M > 128 rows into
    /// ceil(M/128) launches is batch-composition-invariant.
    ///
    /// `x` is flat [m_total * k] f32 row-major (window at `x_row`, used
    /// ONLY for bounds validation -- the math reads `x_qs`/`x_ds`);
    /// `packed` is the packed Q6K [n * k_blocks * 53 u32]; `x_qs`/`x_ds`
    /// are Q8_1 scratch for the window's m rows (m * k / 4 and m * k / 32
    /// u32); `out` is [m_total * n] f32 (rows at `out_row`).
    pub fn forward_mma128_batch_d2d_rows(
        &self,
        x: &Arc<Tensor<f32>>,
        x_row: usize,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        out_row: usize,
        m: usize,
        n: usize,
        k_blocks: usize,
        x_qs: &Arc<Tensor<u32>>,
        x_ds: &Arc<Tensor<u32>>,
        j: usize,
    ) -> Result<()> {
        if m == 0 || m > 128 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mma128_batch_d2d_rows: m must be in 1..=128 (8 m16 tiles); got {m}"
            )));
        }
        let k = k_blocks * 256;
        let row_bytes = k * 4;
        let x_row_bytes = x_row * row_bytes;
        let out_row_bytes = out_row * n * 4;
        if x_row_bytes + m * row_bytes > x.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mma128: x slice [{x_row}, {}+{m}) x {k} exceeds tensor ({} f32)",
                x_row,
                x.size()
            )));
        }
        if out_row_bytes + m * n * 4 > out.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mma128: out slice [{out_row}, {}+{m}) x {n} exceeds tensor ({} f32)",
                out_row,
                out.size()
            )));
        }
        if x_qs.size() < m * k / 4 || x_ds.size() < m * k.div_ceil(32) {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mma128: Q8_1 scratch too small (qs {} < {}, ds {} < {})",
                x_qs.size(),
                m * k / 4,
                x_ds.size(),
                m * k.div_ceil(32)
            )));
        }
        let mma_fn = match self.cu_mma128_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig("q6k mma128_batch not loaded".into())),
        };
        let cu_stream = self.stream.cu_stream();
        let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
        let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
        let ou_ptr: u64 = out.device_pointer().cu_deviceptr() + out_row_bytes as u64;
        let n_u32: u32 = n as u32;
        let kb_u32: u32 = k_blocks as u32;
        let m_u32: u32 = m as u32;
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
        unsafe {
            launch_kernel(
                mma_fn,
                (grid_x, 1, 1),
                (256, 1, 1),  // 8 warps, static smem in PTX
                0,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k mma128_batch gemm rows: {e}")))?;
        Ok(())
    }

    /// MMQ-structured LARGE-m batch GEMM (`q6k_mmq_batch_sm120`): 128
    /// neurons x 32 tokens per CTA, cp.async smem staging, weight scale
    /// products pre-extracted as f16 (the q4k mmq numerics class ported to
    /// Q6K's d-only per-16-element scales). NOT bit-identical to mma128
    /// (the f16 scale roundings) -- the q6k mmq probe arm gates it against
    /// a CPU mirror of the exact op sequence plus a class band vs the f64
    /// reference; end-to-end quality is gated by MMLU/ppl. Requires
    /// `n % 128 == 0`; `x_qs`/`x_ds` must already hold the Q8_1 window for
    /// rows `[x_row, x_row + m)` (same contract as
    /// `forward_mma128_batch_d2d_rows`).
    /// Resolve the mmq kernel function for a J variant.
    fn mmq_fn_for_j(&self, j: usize) -> Result<CUfunction> {
        let f = match j {
            32 => self.cu_mmq_batch,
            64 => self.cu_mmq_batch_j64,
            128 => self.cu_mmq_batch_j128,
            _ => {
                return Err(KernelError::InvalidConfig(format!(
                    "q6k mmq_fn_for_j: J must be 32/64/128; got {j}"
                )))
            }
        };
        f.ok_or_else(|| {
            KernelError::InvalidConfig(format!("q6k mmq_fn_for_j: J={j} variant not loaded"))
        })
    }

    pub fn forward_mmq_batch_d2d_rows(
        &self,
        _x: &Arc<Tensor<f32>>,
        x_row: usize,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        out_row: usize,
        m: usize,
        n: usize,
        k_blocks: usize,
        x_qs: &Arc<Tensor<u32>>,
        x_ds: &Arc<Tensor<u32>>,
        j: usize,
    ) -> Result<()> {
        if m == 0 || m > 128 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mmq_batch_d2d_rows: m must be in 1..=128; got {m}"
            )));
        }
        if n % 128 != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mmq_batch_d2d_rows: n must be a multiple of 128; got {n}"
            )));
        }
        let k = k_blocks * 256;
        let row_bytes = k * 4;
        let x_row_bytes = x_row * row_bytes;
        let out_row_bytes = out_row * n * 4;
        if x_row_bytes + m * row_bytes > _x.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq: x slice [{x_row}, {}+{m}) x {k} exceeds tensor ({} f32)",
                x_row,
                _x.size()
            )));
        }
        if out_row_bytes + m * n * 4 > out.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq: out slice [{out_row}, {}+{m}) x {n} exceeds tensor ({} f32)",
                out_row,
                out.size()
            )));
        }
        if x_qs.size() < m * k / 4 || x_ds.size() < m * k.div_ceil(32) {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq: Q8_1 scratch too small (qs {} < {}, ds {} < {})",
                x_qs.size(),
                m * k / 4,
                x_ds.size(),
                m * k.div_ceil(32)
            )));
        }
        let mmq_fn = self.mmq_fn_for_j(j)?;
        let cu_stream = self.stream.cu_stream();
        let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
        let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
        let ou_ptr: u64 = out.device_pointer().cu_deviceptr() + out_row_bytes as u64;
        let n_u32: u32 = n as u32;
        let kb_u32: u32 = k_blocks as u32;
        let m_u32: u32 = m as u32;
        // splits=1 / null partials: the unsplit direct-store path.
        let splits_u32: u32 = 1;
        let partials_ptr: u64 = 0;
        let mut params: [*mut c_void; 9] = [
            &wt_ptr as *const _ as *mut c_void,
            &qs_ptr as *const _ as *mut c_void,
            &ds_ptr as *const _ as *mut c_void,
            &ou_ptr as *const _ as *mut c_void,
            &n_u32  as *const _ as *mut c_void,
            &kb_u32 as *const _ as *mut c_void,
            &m_u32  as *const _ as *mut c_void,
            &splits_u32 as *const _ as *mut c_void,
            &partials_ptr as *const _ as *mut c_void,
        ];
        // grid (n/128, ceil(m/J)): 128 neurons x J tokens per CTA.
        let grid_x = n_u32 / 128;
        let grid_y = m_u32.div_ceil(j as u32);
        unsafe {
            launch_kernel(
                mmq_fn,
                (grid_x, grid_y, 1),
                (256, 1, 1),
                crate::q6k_mmq_batch_sm120_ptx::dynamic_smem(j) as u32,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k mmq_batch gemm rows: {e}")))?;
        Ok(())
    }

    /// MMQ batch GEMM with a grid.z K-SPLIT for grid fill on the 128-CTA
    /// down_proj shape (the 2026-11-21 port of the q4k split lever:
    /// n=4096 runs (4096/128)*4 = 128 CTAs on 170 SMs = 75% fill). CTA
    /// (x, y, z) computes k-blocks [z*kb/S, (z+1)*kb/S) and writes its f32
    /// partial to `partials[z*m*n ..]`; the fixed-order `q6k_mmq_split_reduce`
    /// entry combines them (out = p0 + p1 + ..., ascending z).
    ///
    /// Numerics vs the unsplit kernel: the per-split f32 accumulation is
    /// identical (same (kb, gw, sub) order within the range); the combine
    /// changes the SUM ORDER -- the split arms in bench-probe-q6k-batch
    /// gate this against a mirror of the exact split op sequence.
    /// `splits == 1` delegates to the unsplit kernel (identical path).
    /// Activations must be pre-quantized (same contract as
    /// [`Self::forward_mmq_batch_d2d_rows`]).
    ///
    /// `partials` must hold `splits * m * n` f32 (only touched when
    /// splits > 1).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mmq_split_batch_d2d_rows(
        &self,
        x: &Arc<Tensor<f32>>,
        x_row: usize,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        out_row: usize,
        m: usize,
        n: usize,
        k_blocks: usize,
        x_qs: &Arc<Tensor<u32>>,
        x_ds: &Arc<Tensor<u32>>,
        splits: usize,
        partials: &Arc<Tensor<f32>>,
        j: usize,
    ) -> Result<()> {
        if splits <= 1 {
            return self.forward_mmq_batch_d2d_rows(
                x, x_row, packed, out, out_row, m, n, k_blocks, x_qs, x_ds, j,
                );
        }
        if m == 0 || m > 128 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mmq_split_batch_d2d_rows: m must be in 1..=128; got {m}"
            )));
        }
        if n % 128 != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mmq_split_batch_d2d_rows: n must be a multiple of 128; got {n}"
            )));
        }
        if splits > k_blocks {
            return Err(KernelError::InvalidConfig(format!(
                "q6k forward_mmq_split_batch_d2d_rows: splits {splits} > k_blocks {k_blocks}"
            )));
        }
        if partials.size() < splits * m * n {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq split: partials holds {} f32, need {splits}*{m}*{n}",
                partials.size()
            )));
        }
        let mmq_fn = self.mmq_fn_for_j(j)?;
        let reduce_fn = match self.cu_mmq_reduce {
            Some(f) => f,
            None => {
                return Err(KernelError::InvalidConfig(
                    "q6k mmq_split_reduce not loaded".into(),
                ))
            }
        };
        let k = k_blocks * 256;
        let row_bytes = k * 4;
        let x_row_bytes = x_row * row_bytes;
        let out_row_bytes = out_row * n * 4;
        if x_row_bytes + m * row_bytes > x.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq split: x slice [{x_row}, {}+{m}) x {k} exceeds tensor ({} f32)",
                x_row,
                x.size()
            )));
        }
        if out_row_bytes + m * n * 4 > out.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq split: out slice [{out_row}, {}+{m}) x {n} exceeds tensor ({} f32)",
                out_row,
                out.size()
            )));
        }
        if x_qs.size() < m * k / 4 || x_ds.size() < m * k.div_ceil(32) {
            return Err(KernelError::InvalidConfig(format!(
                "q6k mmq split: Q8_1 scratch too small (qs {} < {}, ds {} < {})",
                x_qs.size(),
                m * k / 4,
                x_ds.size(),
                m * k.div_ceil(32)
            )));
        }
        let cu_stream = self.stream.cu_stream();
        let wt_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let qs_ptr: u64 = x_qs.device_pointer().cu_deviceptr();
        let ds_ptr: u64 = x_ds.device_pointer().cu_deviceptr();
        let ou_ptr: u64 = out.device_pointer().cu_deviceptr() + out_row_bytes as u64;
        let pt_ptr: u64 = partials.device_pointer().cu_deviceptr();
        let n_u32: u32 = n as u32;
        let kb_u32: u32 = k_blocks as u32;
        let m_u32: u32 = m as u32;
        let splits_u32: u32 = splits as u32;
        let mut params: [*mut c_void; 9] = [
            &wt_ptr as *const _ as *mut c_void,
            &qs_ptr as *const _ as *mut c_void,
            &ds_ptr as *const _ as *mut c_void,
            &ou_ptr as *const _ as *mut c_void,
            &n_u32  as *const _ as *mut c_void,
            &kb_u32 as *const _ as *mut c_void,
            &m_u32  as *const _ as *mut c_void,
            &splits_u32 as *const _ as *mut c_void,
            &pt_ptr as *const _ as *mut c_void,
        ];
        let grid_x = n_u32 / 128;
        let grid_y = m_u32.div_ceil(j as u32);
        unsafe {
            launch_kernel(
                mmq_fn,
                (grid_x, grid_y, splits_u32),
                (256, 1, 1),
                crate::q6k_mmq_batch_sm120_ptx::dynamic_smem(j) as u32,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k mmq_split gemm rows: {e}")))?;
        {
            let total_u32: u32 = (m * n) as u32;
            let mut params: [*mut c_void; 4] = [
                &ou_ptr as *const _ as *mut c_void,
                &pt_ptr as *const _ as *mut c_void,
                &total_u32 as *const _ as *mut c_void,
                &splits_u32 as *const _ as *mut c_void,
            ];
            let grid_x = (total_u32 + 255) / 256;
            unsafe {
                launch_kernel(
                    reduce_fn,
                    (grid_x, 1, 1),
                    (256, 1, 1),
                    0,
                    cu_stream,
                    &mut params,
                )
            }
            .map_err(|e| KernelError::InvalidConfig(format!("q6k mmq_split_reduce: {e}")))?;
        }
        Ok(())
    }

    /// Row-offset variant of `forward_sm120_batch_d2d_into` for prefill chunking.
    ///
    /// Computes rows `[x_row, x_row + m)` of X into rows `[out_row, out_row + m)`
    /// of OUT, reading the SAME packed weights. Lets a prefill chunk of M > 16
    /// rows run as ceil-chunks of <= 16 rows against one [M * n] output tensor
    /// while the kernel still reads each weight block once per launch.
    ///
    /// The kernel loads only the pointer/shape params (the `*_len` guards are
    /// host-side), so base-pointer arithmetic is the only change.
    pub fn forward_sm120_batch_rows(
        &self,
        x_m: &Arc<Tensor<f32>>,
        x_row: usize,
        packed: &Arc<Tensor<u32>>,
        out_m: &Arc<Tensor<f32>>,
        out_row: usize,
        n: usize,
        k_blocks: usize,
        m: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<()> {
        if m < 2 || m > 16 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch_rows requires m in 2..=16 (got {m})"
            )));
        }
        let k = k_blocks * 256;
        let row_bytes = k * 4;
        let x_row_bytes = x_row * row_bytes;
        let out_row_bytes = out_row * n * 4;
        if x_row_bytes + m * row_bytes > x_m.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch_rows: x slice [{x_row}, {x_row}+{m}) x {k} \
                 exceeds tensor ({} f32)",
                x_m.size()
            )));
        }
        if out_row_bytes + m * n * 4 > out_m.size() * 4 {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch_rows: out slice [{out_row}, {out_row}+{m}) x {n} \
                 exceeds tensor ({} f32)",
                out_m.size()
            )));
        }
        let smem_bytes = crate::q6k_batch_sm120_ptx::dynamic_smem_bytes(k_blocks);
        if smem_bytes > Self::MAX_BATCH_DYNAMIC_SMEM {
            return Err(KernelError::InvalidConfig(format!(
                "q6k_gemv_sm120_batch_rows needs {smem_bytes} B dynamic smem (> {})",
                Self::MAX_BATCH_DYNAMIC_SMEM
            )));
        }
        let batch_fn = self.cu_function_sm120_batch.ok_or_else(|| {
            KernelError::InvalidConfig("q6k_gemv_sm120_batch not loaded".into())
        })?;

        let out_ptr: u64 = out_m.device_pointer().cu_deviceptr() as u64 + out_row_bytes as u64;
        let out_len: u64 = out_m.size() as u64 - (out_row_bytes / 4) as u64;
        let x_ptr: u64 = x_m.device_pointer().cu_deviceptr() as u64 + x_row_bytes as u64;
        let x_len: u64 = x_m.size() as u64 - (x_row_bytes / 4) as u64;
        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr() as u64;
        let packed_len: u64 = packed.size() as u64;
        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;
        let m_u32 = m as u32;

        let mut params: [*mut c_void; 9] = [
            &out_ptr      as *const _ as *mut c_void,
            &out_len      as *const _ as *mut c_void,
            &x_ptr        as *const _ as *mut c_void,
            &x_len        as *const _ as *mut c_void,
            &packed_ptr   as *const _ as *mut c_void,
            &packed_len   as *const _ as *mut c_void,
            &n_u32        as *const _ as *mut c_void,
            &k_blocks_u32 as *const _ as *mut c_void,
            &m_u32        as *const _ as *mut c_void,
        ];

        unsafe {
            launch_kernel(
                batch_fn,
                (n as u32, 1, 1),
                (BLOCK_DIM_KPAR, 1, 1),
                smem_bytes as u32,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("q6k_gemv_sm120_batch_rows: {e}")))?;
        Ok(())
    }

    /// Pre-allocate a kpar output scratch tensor of size `max_n`.
    ///
    /// Must be called BEFORE CUDA graph capture starts (e.g., in the engine's
    /// pre-allocation step after warmup). Ensures `forward_kpar_d2d` never
    /// allocates on a capturing stream (CUDA error 900).
    pub fn pre_allocate_kpar_output_scratch(&self, max_n: usize) -> Result<()> {
        if !self.has_kpar() {
            return Ok(());
        }
        let tensor = api::zeros::<f32>(&[max_n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        *self.kpar_out_scratch.borrow_mut() = Some(Arc::new(tensor));
        Ok(())
    }

    /// K-parallel GEMV: grid=(n,1,1), block=(256,1,1).
    ///
    /// Prefers sm120_mbar (cp.async double-buffer, sm_120+) when available;
    /// falls back to dp4a_kpar8 (oxide, sm_80), then to dp4a baseline.
    ///
    /// Uses the pre-allocated scratch tensor when it matches `n` (safe during CUDA
    /// graph capture). Falls back to `api::zeros` allocation otherwise (not
    /// capture-safe -- call `pre_allocate_kpar_output_scratch(n)` before capture).
    pub fn forward_kpar_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let (kpar_fn, is_sm120) = self.select_kpar_fn(k_blocks)
            .ok_or_else(|| KernelError::InvalidConfig("no kpar variant loaded".into()))?;

        // Use pre-allocated scratch to avoid allocation during CUDA graph capture.
        // If scratch is large enough, use it (write to its GPU buffer); the caller
        // gets a clone of the same Arc, pointing to the same GPU memory.
        // This is safe for CUDA graph replay -- the captured write address is constant.
        let use_scratch = self.kpar_out_scratch.borrow()
            .as_ref()
            .map(|t| t.size() >= n)
            .unwrap_or(false);

        // Scratch path: no allocation, no synchronize (safe during CUDA graph capture).
        // Non-scratch path: allocate + sync (only outside graph capture).
        if use_scratch {
            let scratch = self.kpar_out_scratch.borrow().as_ref().unwrap().clone();
            let out_ref: &Tensor<f32> = &*scratch;
            self.launch_kpar_on(kpar_fn, is_sm120, x, packed, out_ref, n, k_blocks,
                                 self.stream.cu_stream())?;
            Ok(scratch)
        } else {
            let out = api::zeros::<f32>(&[n])
                .sync_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            self.launch_kpar_on(kpar_fn, is_sm120, x, packed, &out, n, k_blocks,
                                 self.stream.cu_stream())?;
            unsafe {
                self.stream
                    .synchronize()
                    .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
            }
            Ok(Arc::new(out))
        }
    }

    /// Non-blocking kpar variant on an explicit CUDA stream.
    ///
    /// Prefers sm120_mbar when available.
    ///
    /// # Safety
    /// `cu_stream` must be a valid CUstream for the duration of the kernel.
    pub unsafe fn submit_kpar_d2d_into_on(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<Arc<Tensor<f32>>> {
        let (kpar_fn, is_sm120) = match self.select_kpar_fn(k_blocks) {
            Some(v) => v,
            None => return self.submit_d2d_into_on(x, packed, out, n, k_blocks, cu_stream),
        };
        self.launch_kpar_on(kpar_fn, is_sm120, x, packed, &out, n, k_blocks, cu_stream)?;
        Ok(Arc::new(out))
    }

    /// Select the best available kpar function:
    /// sm120_kpar8_db > sm120_kpar8 > dp4a_kpar8 > None.
    /// - The dual-block direct-load variant (`_db`) has no CTA loop barriers,
    ///   so it works for ANY k_blocks (its B-block is simply invalid on the
    ///   ragged tail).
    /// - The cp.async variant requires `k_blocks % 8 == 0`: its loop-tail
    ///   publication point uses CTA-wide `bar.sync 0` (load-bearing on this
    ///   hw -- see the kernel doc), which is only deadlock-free when all 8
    ///   warps run the same number of loop iterations.
    fn select_kpar_fn(&self, k_blocks: usize) -> Option<(CUfunction, bool)> {
        if let Some(f) = self.cu_function_sm120_db {
            return Some((f, true));
        }
        if let Some(f) = self.cu_function_sm120_mbar {
            if k_blocks % 8 == 0 {
                return Some((f, true));
            }
        }
        self.cu_function_kpar.map(|f| (f, false))
    }

    fn launch_kpar_on(
        &self,
        kpar_fn: CUfunction,
        is_sm120_mbar: bool,
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

        // sm120 kpar8 declares 5384 bytes of static shared memory; pass 0 dynamic.
        // dp4a_kpar8 uses SMEM_KPAR (32 bytes) dynamic shared memory.
        let smem = if is_sm120_mbar { 0 } else { SMEM_KPAR };
        let label = if is_sm120_mbar { "q6k_gemv_sm120_kpar8" } else { "oxide q6k_gemv_dp4a_kpar8" };

        unsafe {
            launch_kernel(
                kpar_fn,
                (n as u32, 1, 1),
                (BLOCK_DIM_KPAR, 1, 1),
                smem,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("{label}: {e}")))?;
        Ok(())
    }

    /// Device-to-device Q6_K GEMV: `out[n] = sum_k dequant(packed[n,k]) * x[k]`.
    pub fn forward_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        if self.cu_function.is_null() {
            return Err(KernelError::InvalidConfig(
                "oxide q6k_gemv_dp4a baseline not loaded (oxide-gemv PTX missing entry)".into(),
            ));
        }
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

    /// Non-blocking, graph-safe variant: launches the baseline Q6K GEMV and writes
    /// into a caller-owned `Arc<Tensor<f32>>`.  No allocation, no stream sync.
    /// Safe to call inside `cuStreamBeginCapture` / `cuStreamEndCapture` regions.
    ///
    /// Uses the oxide stream internally.  For explicit stream control (needed when
    /// the calling code runs on a different stream than the oxide stream), use
    /// `submit_d2d_arc_on`.
    pub fn submit_d2d_arc(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<()> {
        unsafe { self.launch(x, packed, out, n, k_blocks) }
    }

    /// Like `submit_d2d_arc` but launches on an explicit `cu_stream`.
    ///
    /// Use this inside CUDA graph capture regions to ensure the kernel is captured
    /// on the same stream as the surrounding operations.
    pub fn submit_d2d_arc_on(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: &Arc<Tensor<f32>>,
        n: usize,
        k_blocks: usize,
        cu_stream: cuda_core::sys::CUstream,
    ) -> Result<()> {
        unsafe { self.launch_on(x, packed, out, n, k_blocks, cu_stream) }
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

    /// Like `submit_d2d_into` but launches on an explicit CUDA stream.
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
        if self.cu_function.is_null() {
            return Err(KernelError::InvalidConfig(
                "oxide q6k_gemv_dp4a baseline not loaded".into(),
            ));
        }
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

        let grid_x = n.div_ceil(WARPS_PER_BLOCK as usize) as u32;
        unsafe {
            launch_kernel(
                self.cu_function,
                (grid_x, 1, 1),
                (BLOCK_DIM, 1, 1),
                0,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q6k_gemv_decode (kv stream): {e}")))?;
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
        if self.cu_function.is_null() {
            return Err(KernelError::InvalidConfig(
                "oxide q6k_gemv_dp4a baseline not loaded".into(),
            ));
        }
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

        let grid_x = n.div_ceil(WARPS_PER_BLOCK as usize) as u32;
        let cu_stream = self.stream.cu_stream();

        unsafe {
            launch_kernel(
                self.cu_function,
                (grid_x, 1, 1),
                (BLOCK_DIM, 1, 1),
                0,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q6k_gemv_decode: {e}")))?;

        Ok(())
    }
}

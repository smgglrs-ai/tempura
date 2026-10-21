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

/// Launches the oxide-compiled Q6_K GEMV kernel for M=1 decode.
pub struct OxideQ6kGemv {
    cu_module:              CUmodule,
    cu_function:            CUfunction,
    cu_function_kpar:       Option<CUfunction>,
    /// sm_120 hand-written PTX module (separate from oxide module).
    cu_sm120_module:        Option<CUmodule>,
    /// q6k_gemv_sm120_mbar_kpar8: cp.async double-buffer variant (sm_120+).
    cu_function_sm120_mbar: Option<CUfunction>,
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
    }
}

impl OxideQ6kGemv {
    /// Load Q6_K GEMV kernels.
    ///
    /// Attempts to load the oxide-compiled PTX (baseline + kpar8) and, on sm_120+,
    /// the hand-written cp.async double-buffer kernel (`q6k_gemv_sm120_mbar_kpar8`).
    /// Returns `None` only when no kernel variant could be loaded at all.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        let _ = stream.device().bind_to_thread().ok()?;
        const CUDA_SUCCESS: u32 = 0;

        // -- Try oxide-compiled PTX (feature-gated) -----------------------------
        #[cfg(not(feature = "oxide-gemv"))]
        let (cu_module, cu_function, cu_function_kpar) =
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

        // -- Try sm_120 cp.async mbar PTX (not feature-gated, any sm_120 GPU) ------
            // Requires Blackwell GPU (cc >= 1200). Falls back to cu_function_kpar on older GPUs.
            let sm120_disabled = std::env::var("TEMPURA_NO_SM120_Q6K")
                .map(|v| v == "1")
                .unwrap_or(false);

            let (cu_sm120_module, cu_function_sm120_mbar) = if !sm120_disabled {
                let caps = crate::gpu_capabilities::GpuCapabilities::query(
                    stream.device().cu_device()
                );
                if caps.cc() >= crate::q6k_gemv_v4_ptx::MIN_CC {
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
                            eprintln!(
                                "oxide-gemv-q6k: loaded sm120_mbar variant \
                                 (entry={}, cc={})",
                                crate::q6k_gemv_v4_ptx::ENTRY,
                                caps.cc(),
                            );
                            (Some(sm120_mod), Some(unsafe { fn_out.assume_init() }))
                        } else {
                            eprintln!(
                                "oxide-gemv-q6k: cuModuleGetFunction(sm120_mbar) \
                                 failed (rc={rc2}); using kpar8 baseline"
                            );
                            let _ = unsafe { cuda_core::sys::cuModuleUnload(sm120_mod) };
                            (None, None)
                        }
                    } else {
                        eprintln!(
                            "oxide-gemv-q6k: cuModuleLoadData(sm120 PTX) \
                             failed (rc={rc}); using kpar8 baseline"
                        );
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                eprintln!("oxide-gemv-q6k: TEMPURA_NO_SM120_Q6K=1, skipping sm120_mbar");
                (None, None)
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
            eprintln!("oxide-gemv-q6k: no Q6K kernel variant available; falling back to cuTile");
            return None;
        }

        eprintln!(
            "oxide-gemv-q6k: Q6K kernels loaded (oxide_fn={}, kpar={}, sm120_mbar={})",
            !cu_function.is_null(),
            cu_function_kpar.is_some(),
            cu_function_sm120_mbar.is_some(),
        );
        Some(Self {
            cu_module,
            cu_function,
            cu_function_kpar,
            cu_sm120_module,
            cu_function_sm120_mbar,
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
        let (kpar_fn, is_sm120) = self.select_kpar_fn()
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
        let (kpar_fn, is_sm120) = match self.select_kpar_fn() {
            Some(v) => v,
            None => return self.submit_d2d_into_on(x, packed, out, n, k_blocks, cu_stream),
        };
        self.launch_kpar_on(kpar_fn, is_sm120, x, packed, &out, n, k_blocks, cu_stream)?;
        Ok(Arc::new(out))
    }

    /// Select the best available kpar function: sm120_mbar > dp4a_kpar8 > None.
    fn select_kpar_fn(&self) -> Option<(CUfunction, bool)> {
        if let Some(f) = self.cu_function_sm120_mbar {
            return Some((f, true));
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

        // sm120_mbar declares 3424 bytes of static shared memory; pass 0 dynamic.
        // dp4a_kpar8 uses SMEM_KPAR (32 bytes) dynamic shared memory.
        let smem = if is_sm120_mbar { 0 } else { SMEM_KPAR };
        let label = if is_sm120_mbar { "q6k_gemv_sm120_mbar_kpar8" } else { "oxide q6k_gemv_dp4a_kpar8" };

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

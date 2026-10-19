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
/// accumulation followed by a butterfly warp-reduce (5× `shfl.sync.bfly`).
/// Lane 0 of each warp writes the final sum to `out[n_out]`.
///
/// ## PTX parameter layout
///
/// ```text
/// param_0: .u64 .ptr  — out pointer (DisjointSlice data ptr)
/// param_1: .u64       — out len (u64)
/// param_2: .u64 .ptr  — x pointer
/// param_3: .u64       — x len (u64)
/// param_4: .u64 .ptr  — packed pointer
/// param_5: .u64       — packed len (u64)
/// param_6: .u32       — n (u32)
/// param_7: .u32       — k_blocks (u32)
/// ```

use crate::{KernelError, Result};
use cuda_core::{launch_kernel, Stream};
use cuda_core::sys::{CUfunction, CUmodule};
use cutile::{api, prelude::DeviceOp, tensor::Tensor};
use std::ffi::{c_void, CString};
use std::sync::Arc;

/// The PTX source compiled by `cargo oxide build tools/oxide-gemv-q6k`.
#[cfg(feature = "oxide-gemv")]
static OXIDE_Q6K_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_q6k_gemv.ptx"));

const KERNEL_ENTRY: &str = "q6k_gemv_decode_v2";
const KERNEL_ENTRY_KPAR: &str = "q6k_gemv_decode_v3_kpar";
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128
const WARPS_PER_BLOCK_KPAR: u32 = 8;
const BLOCK_DIM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 32; // 256
const SMEM_KPAR: u32 = WARPS_PER_BLOCK_KPAR * 4; // 32 bytes

/// Launches the oxide-compiled Q6_K GEMV kernel for M=1 decode.
pub struct OxideQ6kGemv {
    cu_module:        CUmodule,
    cu_function:      CUfunction,
    cu_function_kpar: Option<CUfunction>,
    stream:           Arc<Stream>,
}

unsafe impl Send for OxideQ6kGemv {}
unsafe impl Sync for OxideQ6kGemv {}

impl Drop for OxideQ6kGemv {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
        }
    }
}

impl OxideQ6kGemv {
    /// Load the oxide Q6_K GEMV kernel from the embedded PTX.
    ///
    /// Returns `None` when `oxide-gemv` feature is absent or loading fails.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        #[cfg(not(feature = "oxide-gemv"))]
        return None;

        #[cfg(feature = "oxide-gemv")]
        {
            if OXIDE_Q6K_PTX.is_empty() {
                return None;
            }
            let ptx_str = std::str::from_utf8(OXIDE_Q6K_PTX).ok()?;
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
                        "oxide-gemv-q6k: cuModuleLoadData failed (code={ret}); \
                         falling back to cuTile Q6_K kernel"
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
                        "oxide-gemv-q6k: cuModuleGetFunction('{KERNEL_ENTRY}') failed (code={ret})"
                    );
                    return None;
                }
                func.assume_init()
            };

            // Load kpar variant; optional — fall back to v2 if absent.
            let cu_function_kpar = CString::new(KERNEL_ENTRY_KPAR).ok().and_then(|entry_kpar| {
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
                        "oxide-gemv-q6k: kpar variant not found (code={ret}); using v2"
                    );
                    None
                }
            });

            eprintln!(
                "oxide-gemv-q6k: loaded Q6_K GEMV kernel (PTX {} bytes, entry={})",
                OXIDE_Q6K_PTX.len(),
                KERNEL_ENTRY
            );
            Some(Self { cu_module, cu_function, cu_function_kpar, stream })
        }
    }

    /// Returns `true` when the kpar kernel (v3) was successfully loaded.
    pub fn has_kpar(&self) -> bool {
        self.cu_function_kpar.is_some()
    }

    /// K-parallel GEMV: grid=(n,1,1), block=(256,1,1), smem=32 bytes.
    ///
    /// Preferred over `forward_d2d` for large k_blocks (e.g. down_proj k_blocks=56)
    /// because 8 warps per block → 100% SM occupancy vs. 37.5% for v2.
    pub fn forward_kpar_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let kpar = match self.cu_function_kpar {
            Some(f) => f,
            None => return self.forward_d2d(x, packed, n, k_blocks),
        };
        let out = api::zeros::<f32>(&[n])
            .sync_on(&self.stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        self.launch_kpar_on(kpar, x, packed, &out, n, k_blocks, self.stream.cu_stream())?;

        unsafe {
            self.stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(Arc::new(out))
    }

    /// Non-blocking kpar variant on an explicit CUDA stream.
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
        let kpar = match self.cu_function_kpar {
            Some(f) => f,
            None => return self.submit_d2d_into_on(x, packed, out, n, k_blocks, cu_stream),
        };
        self.launch_kpar_on(kpar, x, packed, &out, n, k_blocks, cu_stream)?;
        Ok(Arc::new(out))
    }

    fn launch_kpar_on(
        &self,
        kpar_fn: CUfunction,
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

        unsafe {
            launch_kernel(
                kpar_fn,
                (n as u32, 1, 1),
                (BLOCK_DIM_KPAR, 1, 1),
                SMEM_KPAR,
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q6k_gemv_decode_v3_kpar: {e}")))?;
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

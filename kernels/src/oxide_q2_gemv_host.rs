/// Host-side wrapper for the oxide-compiled Q2_0 GEMV kernel.
///
/// The kernel is compiled with `cargo oxide build` from `tools/oxide-gemv/`.
/// `build.rs` extracts the embedded PTX from the compiled binary and writes
/// it to `$OUT_DIR/oxide_q2_gemv.ptx`. This module embeds that PTX at
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
/// ## PTX parameter layout (from tools/oxide-gemv PTX inspection)
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

/// The PTX source compiled by `cargo oxide build tools/oxide-gemv`.
///
/// Embedded at compile time by build.rs which extracts the `.oxart` section
/// from `tools/oxide-gemv/target/release/oxide-gemv`.
///
/// When the `oxide-gemv` feature is absent (non-CUDA builds or CI without
/// GPU), `try_load` returns `None` gracefully.
#[cfg(feature = "oxide-gemv")]
static OXIDE_Q2_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_q2_gemv.ptx"));

const KERNEL_ENTRY: &str = "q2_0_gemv_decode";
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128

/// Launches the oxide-compiled Q2_0 GEMV kernel for M=1 decode.
///
/// Uses raw CUDA driver APIs to load the PTX into the device's primary context
/// (the same context used by the cutile / matmul stream). Avoids the
/// `CudaContext`/`CudaStream` simt-API type mismatch.
///
/// Falls back to `None` when the `oxide-gemv` feature is absent.
pub struct OxideQ2Gemv {
    cu_module:   CUmodule,
    cu_function: CUfunction,
    stream:      Arc<Stream>,
}

// SAFETY: CUDA module/function handles are thread-safe for launching
// (multiple host threads can submit work to different streams on the same device).
unsafe impl Send for OxideQ2Gemv {}
unsafe impl Sync for OxideQ2Gemv {}

impl Drop for OxideQ2Gemv {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            // Ignore errors during drop (device may already be gone).
            let _ = unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
        }
    }
}

impl OxideQ2Gemv {
    /// Load the oxide Q2_0 GEMV kernel from the embedded PTX.
    ///
    /// Loads into the current CUDA context (the primary context of the device
    /// bound to the calling thread). Must be called after the cutile/matmul
    /// stream is set up so the context is already current.
    ///
    /// Returns `None` when `oxide-gemv` feature is absent or loading fails.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        #[cfg(not(feature = "oxide-gemv"))]
        return None;

        #[cfg(feature = "oxide-gemv")]
        {
            if OXIDE_Q2_PTX.is_empty() {
                return None;
            }
            let ptx_str = std::str::from_utf8(OXIDE_Q2_PTX).ok()?;
            let c_ptx = CString::new(ptx_str).ok()?;
            let entry = CString::new(KERNEL_ENTRY).ok()?;

            // Bind the stream's device context to this thread before loading.
            let _ = stream.device().bind_to_thread().ok()?;

            // CUDA_SUCCESS == 0 for all CUDA driver API calls.
            const CUDA_SUCCESS: u32 = 0;

            let cu_module = unsafe {
                let mut module = std::mem::MaybeUninit::uninit();
                let ret = cuda_core::sys::cuModuleLoadData(
                    module.as_mut_ptr(),
                    c_ptx.as_ptr() as *const _,
                );
                if ret != CUDA_SUCCESS {
                    eprintln!(
                        "oxide-gemv: cuModuleLoadData failed (code={ret}); \
                         falling back to cuTile Q2_0 kernel"
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
                        "oxide-gemv: cuModuleGetFunction('{KERNEL_ENTRY}') failed (code={ret})"
                    );
                    return None;
                }
                func.assume_init()
            };

            eprintln!(
                "oxide-gemv: loaded Q2_0 GEMV kernel (PTX {} bytes, entry={})",
                OXIDE_Q2_PTX.len(),
                KERNEL_ENTRY
            );
            Some(Self { cu_module, cu_function, stream })
        }
    }

    /// Device-to-device Q2_0 GEMV: `out[n] = sum_k dequant(packed[n,k]) * x[k]`.
    ///
    /// `x`        — activation tensor [K = k_blocks * 128]
    /// `packed`   — packed Q2_0 weights [N * k_blocks * 9] u32
    /// `n`        — number of output neurons
    /// `k_blocks` — number of Q2_0 blocks per output row
    ///
    /// Returns a new device tensor of shape [n].
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

        // Synchronize so the caller can safely read the tensor.
        unsafe {
            self.stream
                .synchronize()
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        }

        Ok(Arc::new(out))
    }

    /// Non-blocking variant: re-uses a pre-allocated output tensor.
    ///
    /// # Safety
    ///
    /// The returned tensor's device memory is being written asynchronously.
    /// The caller MUST ensure the stream is synced before reading on CPU.
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

    fn launch(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: &Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<()> {
        // Device pointers via the public DevicePointer API (cuda_async).
        let out_ptr: u64 = out.device_pointer().cu_deviceptr();
        let out_len: u64 = n as u64;

        let x_ptr: u64 = x.device_pointer().cu_deviceptr();
        let x_len: u64 = x.size() as u64;

        let packed_ptr: u64 = packed.device_pointer().cu_deviceptr();
        let packed_len: u64 = packed.size() as u64;

        let n_u32 = n as u32;
        let k_blocks_u32 = k_blocks as u32;

        // cuLaunchKernel takes &mut [*mut c_void] where each element points to
        // the actual argument value. The driver reads the values before returning.
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

        // SAFETY: grid/block dims valid; device ptrs live for kernel duration;
        // param types match the PTX ABI (confirmed via PTX inspection of
        // tools/oxide-gemv/target/release/oxide-gemv).
        unsafe {
            launch_kernel(
                self.cu_function,
                (grid_x, 1, 1),
                (BLOCK_DIM, 1, 1),
                0, // no dynamic shared memory
                cu_stream,
                &mut params,
            )
        }
        .map_err(|e| KernelError::InvalidConfig(format!("oxide q2_0_gemv_decode: {e}")))?;

        Ok(())
    }
}

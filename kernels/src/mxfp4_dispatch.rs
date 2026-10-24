//! Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM.
//!
//! Kernel: `mxf4_batch_gemm_tiled` -- stream-K persistent, grid=(sm_count,1,1),
//! block=(128,1,1) = 4 warps. Computes C[M][N] = A_mxfp4[M][K] x B_mxfp4[N][K]
//! with f32 output using SM120a m16n8k64 MMA tensor cores.
//!
//! The cubin is compiled from `tools/mxf4-gemv/` with:
//!   CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build --arch sm_120a
//!
//! Returns `None` on non-sm_120a hardware (cuModuleLoadData fails gracefully).

use crate::{KernelError, Result};
use cuda_core::{sys, Stream};
use cutile::tensor::Tensor;
use std::ffi::CString;
use std::sync::Arc;

/// Cubin bytes compiled for sm_120a (RTX 5090 Blackwell accelerated).
/// Path relative to this source file: tools/mxf4-gemv/mxf4_gemv.cubin
static MXFP4_CUBIN: &[u8] = include_bytes!("../../tools/mxf4-gemv/mxf4_gemv.cubin");

const KERNEL_NAME: &str = "mxf4_batch_gemm_tiled";

/// Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM.
///
/// Loaded via cuModuleLoadData from the embedded cubin. Gracefully returns
/// `None` on non-sm_120a hardware where cuModuleLoadData rejects the ELF.
pub struct MxfpGemm {
    cu_module:   sys::CUmodule,
    cu_function: sys::CUfunction,
    sm_count:    u32,
    stream:      Arc<Stream>,
}

impl MxfpGemm {
    /// Try to load the MXFP4 batch GEMM kernel from the embedded cubin.
    ///
    /// Returns `None` when the cubin is rejected (non-sm_120a hardware) or
    /// the kernel symbol is absent.
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        let mut module = std::mem::MaybeUninit::uninit();
        let rc = unsafe {
            sys::cuModuleLoadData(
                module.as_mut_ptr(),
                MXFP4_CUBIN.as_ptr() as *const std::ffi::c_void,
            )
        };
        if rc != 0 {
            eprintln!("[mxfp4] cuModuleLoadData failed (rc={rc}): requires sm_120a (RTX 5090)");
            return None;
        }
        let module = unsafe { module.assume_init() };

        let name = CString::new(KERNEL_NAME).unwrap_or_default();
        let mut func = std::mem::MaybeUninit::uninit();
        let rc = unsafe {
            sys::cuModuleGetFunction(func.as_mut_ptr(), module, name.as_ptr())
        };
        if rc != 0 {
            eprintln!("[mxfp4] cuModuleGetFunction({KERNEL_NAME}) failed (rc={rc})");
            unsafe { sys::cuModuleUnload(module) };
            return None;
        }
        let func = unsafe { func.assume_init() };

        let mut sm_count = 0i32;
        unsafe {
            use sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SM_COUNT;
            sys::cuDeviceGetAttribute(&mut sm_count, SM_COUNT, 0);
        }
        let sm_count = sm_count.max(1) as u32;

        eprintln!(
            "[mxfp4] MxfpGemm loaded: {KERNEL_NAME} ({} bytes cubin, sm_count={sm_count})",
            MXFP4_CUBIN.len()
        );
        Some(Self { cu_module: module, cu_function: func, sm_count, stream })
    }

    /// Launch MXFP4 tiled batch GEMM: A[m, k] x B[n, k]^T -> C[m, n] (f32).
    ///
    /// - `a_packed`:  [m * k/2] u8, E2M1 nibbles row-major [M][K/2]
    /// - `a_scales`:  [(m/16) * (k/32)] u8, UE8M0 tile scales [m16_tiles][k32_blocks]
    /// - `b_packed`:  [n * k/2] u8, E2M1 nibbles row-major [N][K/2]
    /// - `b_scales`:  [(n/8) * (k/32)] u8, UE8M0 tile scales [n8_tiles][k32_blocks]
    /// - `c`:         [m * n] f32, output row-major [M][N]
    /// - `k64_tiles`: k / 64
    ///
    /// `m` must be multiple of 16, `n` multiple of 8, `k` multiple of 64.
    /// Launch is async on the stored stream; caller syncs if needed.
    pub fn forward(
        &self,
        a_packed:  &Arc<Tensor<u8>>,
        a_scales:  &Arc<Tensor<u8>>,
        b_packed:  &Arc<Tensor<u8>>,
        b_scales:  &Arc<Tensor<u8>>,
        c:         &Arc<Tensor<f32>>,
        m:         u32,
        n:         u32,
        k64_tiles: u32,
    ) -> Result<()> {
        let cu_stream = self.stream.cu_stream();

        let mut p_ap = a_packed.device_pointer().cu_deviceptr() as u64;
        let mut p_as = a_scales.device_pointer().cu_deviceptr() as u64;
        let mut p_bp = b_packed.device_pointer().cu_deviceptr() as u64;
        let mut p_bs = b_scales.device_pointer().cu_deviceptr() as u64;
        let mut p_c  = c.device_pointer().cu_deviceptr() as u64;
        let mut pm   = m;
        let mut pn   = n;
        let mut pk   = k64_tiles;

        let mut params: [*mut std::ffi::c_void; 8] = [
            &mut p_ap as *mut _ as *mut std::ffi::c_void,
            &mut p_as as *mut _ as *mut std::ffi::c_void,
            &mut p_bp as *mut _ as *mut std::ffi::c_void,
            &mut p_bs as *mut _ as *mut std::ffi::c_void,
            &mut p_c  as *mut _ as *mut std::ffi::c_void,
            &mut pm   as *mut _ as *mut std::ffi::c_void,
            &mut pn   as *mut _ as *mut std::ffi::c_void,
            &mut pk   as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function,
                self.sm_count, 1, 1,
                128, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "mxf4_batch_gemm_tiled cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// SM count used as grid X dimension.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }
}

impl Drop for MxfpGemm {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.cu_module) };
    }
}

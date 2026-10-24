//! Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM and the
//! sm_120 GPU-side activation quantization kernel.
//!
//! GEMM kernel: `mxf4_batch_gemm_tiled` -- stream-K persistent, grid=(sm_count,1,1),
//! block=(128,1,1) = 4 warps. Computes C[M][N] = A_mxfp4[M][K] x B_mxfp4[N][K]
//! with f32 output using SM120a m16n8k64 MMA tensor cores.
//! Compiled from `tools/mxf4-gemv/` with:
//!   CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build --arch sm_120a
//!
//! Quant kernel: `mxfp4_quantize_activation_tiled` -- sm_120 PTX, one block per
//! [16-row x 32-col] tile. Eliminates 22ms/token D->H->quant->H->D round-trip.
//!
//! Returns `None` on non-sm_120a hardware (cuModuleLoadData for the cubin fails
//! gracefully). The quant kernel loads separately and works on any sm_120+ device.

use crate::{mxfp4_quant_ptx, KernelError, Result};
use cuda_core::{sys, Stream};
use cutile::tensor::Tensor;
use std::ffi::CString;
use std::sync::Arc;

/// Cubin bytes compiled for sm_120a (RTX 5090 Blackwell accelerated).
/// Path relative to this source file: tools/mxf4-gemv/mxf4_gemv.cubin
static MXFP4_CUBIN: &[u8] = include_bytes!("../../tools/mxf4-gemv/mxf4_gemv.cubin");

const KERNEL_NAME: &str = "mxf4_batch_gemm_tiled";

/// Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM and the
/// sm_120 GPU activation quantization kernel.
///
/// GEMM: loaded via cuModuleLoadData from the embedded cubin. Returns `None` on
/// non-sm_120a hardware. Quant: loaded from inline PTX; available on sm_120+.
pub struct MxfpGemm {
    cu_module:     sys::CUmodule,
    cu_function:   sys::CUfunction,
    /// GPU activation quantization: f32 -> E2M1+UE8M0 tiled. None if unsupported.
    cu_quant_mod:  Option<sys::CUmodule>,
    cu_quant_fn:   Option<sys::CUfunction>,
    sm_count:      u32,
    stream:        Arc<Stream>,
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

        // Load GPU activation quantization kernel from inline sm_120 PTX.
        let quant_ptx = CString::new(mxfp4_quant_ptx::PTX).unwrap_or_default();
        let mut qmod = std::mem::MaybeUninit::uninit();
        let qrc = unsafe {
            sys::cuModuleLoadData(
                qmod.as_mut_ptr(),
                quant_ptx.as_ptr() as *const std::ffi::c_void,
            )
        };
        let (cu_quant_mod, cu_quant_fn) = if qrc == 0 {
            let qmod = unsafe { qmod.assume_init() };
            let qname = CString::new(mxfp4_quant_ptx::ENTRY).unwrap_or_default();
            let mut qfn = std::mem::MaybeUninit::uninit();
            let qfrc = unsafe {
                sys::cuModuleGetFunction(qfn.as_mut_ptr(), qmod, qname.as_ptr())
            };
            if qfrc == 0 {
                eprintln!("[mxfp4] quant kernel loaded: {} ({} bytes PTX)",
                    mxfp4_quant_ptx::ENTRY, mxfp4_quant_ptx::PTX.len());
                (Some(qmod), Some(unsafe { qfn.assume_init() }))
            } else {
                eprintln!("[mxfp4] cuModuleGetFunction({}) failed (rc={qfrc}), GPU quant unavailable",
                    mxfp4_quant_ptx::ENTRY);
                unsafe { sys::cuModuleUnload(qmod) };
                (None, None)
            }
        } else {
            eprintln!("[mxfp4] quant PTX load failed (rc={qrc}), GPU quant unavailable");
            (None, None)
        };

        eprintln!(
            "[mxfp4] MxfpGemm loaded: {KERNEL_NAME} ({} bytes cubin, sm_count={sm_count})",
            MXFP4_CUBIN.len()
        );
        Some(Self { cu_module: module, cu_function: func, cu_quant_mod, cu_quant_fn, sm_count, stream })
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

    /// GPU-side activation quantization: f32[M*K] -> (packed_e2m1, scales_ue8m0).
    ///
    /// Launches `mxfp4_quantize_activation_tiled` entirely on-GPU; no PCIe transfer.
    /// One block per [16-row x 32-col] tile (matches quantize_activation_to_mxfp4_tiled).
    ///
    /// - `src`:        [m * k] f32, row-major activation
    /// - `packed`:     [m * k / 2] u8, output E2M1 nibbles (lo nibble = first elem)
    /// - `scales`:     [(m/16) * (k/32)] u8, output UE8M0 tile scales
    /// - `m`, `k`:     dimensions (m must be multiple of 16, k multiple of 32)
    /// - `stream`:     CUDA stream for async launch
    ///
    /// Returns `Err` if the quant kernel was not loaded (non-sm_120 hardware or
    /// load failure). Caller must sync the stream before reading `packed`/`scales`.
    pub fn quantize_activation(
        &self,
        src:    &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u8>>,
        scales: &Arc<Tensor<u8>>,
        m:      u32,
        k:      u32,
        stream: sys::CUstream,
    ) -> Result<()> {
        let qfn = self.cu_quant_fn.ok_or_else(|| {
            KernelError::InvalidConfig("mxfp4 quant kernel not loaded".into())
        })?;

        debug_assert_eq!(m % 16, 0, "m must be multiple of 16");
        debug_assert_eq!(k % 32, 0, "k must be multiple of 32");

        let k32_blocks = k / 32;
        let m16_tiles  = m / 16;
        let grid_x     = m16_tiles * k32_blocks;  // one block per tile

        let mut p_src    = src.device_pointer().cu_deviceptr() as u64;
        let mut p_packed = packed.device_pointer().cu_deviceptr() as u64;
        let mut p_scales = scales.device_pointer().cu_deviceptr() as u64;
        let mut pm       = m;
        let mut pk       = k;
        let mut pk32     = k32_blocks;

        let mut params: [*mut std::ffi::c_void; 6] = [
            &mut p_src    as *mut _ as *mut std::ffi::c_void,
            &mut p_packed as *mut _ as *mut std::ffi::c_void,
            &mut p_scales as *mut _ as *mut std::ffi::c_void,
            &mut pm       as *mut _ as *mut std::ffi::c_void,
            &mut pk       as *mut _ as *mut std::ffi::c_void,
            &mut pk32     as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                qfn,
                grid_x, 1, 1,
                256, 1, 1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "mxfp4_quantize_activation_tiled cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Returns true if GPU activation quantization is available.
    pub fn has_gpu_quant(&self) -> bool {
        self.cu_quant_fn.is_some()
    }

    /// SM count used as grid X dimension.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }
}

impl Drop for MxfpGemm {
    fn drop(&mut self) {
        if let Some(qmod) = self.cu_quant_mod {
            unsafe { sys::cuModuleUnload(qmod) };
        }
        unsafe { sys::cuModuleUnload(self.cu_module) };
    }
}

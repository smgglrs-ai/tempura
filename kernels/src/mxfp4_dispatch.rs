//! Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM and the
//! sm_120 GPU-side activation quantization kernel.
//!
//! GEMM kernel: `mxf4_batch_gemm_tiled` -- one m16n8 output tile per warp on a
//! 2D grid (ceil(n_tiles/4), m_tiles, 1), block=(128,1,1) = 4 warps. Computes
//! C[M][N] = A_mxfp4[M][K] x B_mxfp4[N][K] with f32 output using SM120a
//! m16n8k64 MMA tensor cores. Validated against a random-data CPU reference
//! in tools/mxf4-gemv (uniform-data tests cannot catch fragment-mapping
//! bugs: any store permutation "passes" and NaN comparisons are false).
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
const KERNEL_NAME_RB4: &str = "mxf4_batch_gemm_tiled_rb4";
const KERNEL_NAME_RB4M32: &str = "mxf4_batch_gemm_tiled_rb4m32";

/// Minimum rb4 grid size (blocks) for the register-blocked kernel to beat
/// the m16n8 kernel: rb4 quarters the block count (each warp covers 4 n8
/// tiles), so it only wins on large-n shapes that still fill the SMs --
/// measured on the prefill m=128 shapes (tools/mxf4-gemv benchmark,
/// 2026-10-29): (128,14336,4096) 122.4 -> 106.2 us (141.6 TFLOPS, 896
/// blocks) vs (128,4096,4096) 35.8 vs 38.8 us (256 blocks) and
/// (128,1024,4096) 11.7 vs 37.0 us (64 blocks).
const RB4_MIN_BLOCKS: u32 = 512;

/// Minimum rb4m32 grid size. The m32 variant also halves the B re-read
/// traffic across m-tiles (the binding limit for large-n GEMMs after rb4):
/// (128,14336,4096) 123.3 -> 57.8 us (259.9 TFLOPS, 448 blocks) and
/// (128,4096,14336) 127.4 -> 114.4 us (131.5 TFLOPS, 128 blocks) vs
/// (128,1024,4096) 11.7 vs 34.5 us (32 blocks).
const RB4M32_MIN_BLOCKS: u32 = 128;

/// Host-side dispatch for the cute-rs sm_120a MXFP4 tiled batch GEMM and the
/// sm_120 GPU activation quantization kernel.
///
/// GEMM: loaded via cuModuleLoadData from the embedded cubin. Returns `None` on
/// non-sm_120a hardware. Quant: loaded from inline PTX; available on sm_120+.
pub struct MxfpGemm {
    cu_module:     sys::CUmodule,
    cu_function:   sys::CUfunction,
    /// Register-blocked variant (m16n32 per warp, 4 MMAs sharing the A
    /// fragment): bit-identical output, preferred on large-n grids.
    /// None when the cubin predates it.
    cu_function_rb4: Option<sys::CUfunction>,
    /// m32 register-blocked variant (m32n32 per warp, 8 MMAs; B fragments
    /// shared across the two m16 halves, halving the B re-read traffic):
    /// bit-identical output, preferred when m % 32 == 0 on grids >=
    /// RB4M32_MIN_BLOCKS. None when the cubin predates it.
    cu_function_rb4m32: Option<sys::CUfunction>,
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

        // Best-effort: the register-blocked variants (same cubin, extra
        // entries). Older cubins without them keep working.
        let load_entry = |name: &str| -> Option<sys::CUfunction> {
            let cstr = CString::new(name).unwrap_or_default();
            let mut fn_out = std::mem::MaybeUninit::uninit();
            let rc = unsafe { sys::cuModuleGetFunction(fn_out.as_mut_ptr(), module, cstr.as_ptr()) };
            if rc == 0 {
                Some(unsafe { fn_out.assume_init() })
            } else {
                eprintln!("[mxfp4] cuModuleGetFunction({name}) failed (rc={rc})");
                None
            }
        };
        let cu_function_rb4 = load_entry(KERNEL_NAME_RB4);
        let cu_function_rb4m32 = load_entry(KERNEL_NAME_RB4M32);

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
        Some(Self { cu_module: module, cu_function: func, cu_function_rb4, cu_function_rb4m32, cu_quant_mod, cu_quant_fn, sm_count, stream })
    }

    /// Row-offset chunk variant of [`MxfpGemm::forward`].
    ///
    /// Computes rows `[a_row, a_row + m)` of A into rows `[c_row, c_row + m)`
    /// of C against the SAME B weights. Kept as a utility for callers that
    /// want to bound a single launch's grid (e.g. feeding a kernel whose
    /// output tile coverage is still being validated); the prefill path
    /// launches at full m_pad since the accumulator store mapping was fixed
    /// (2026-10-27) and every shape up to (128, 4096, 4096) is validated
    /// against a random-data CPU reference in tools/mxf4-gemv.
    pub fn forward_offset(
        &self,
        a_packed:  &Arc<Tensor<u8>>,
        a_scales:  &Arc<Tensor<u8>>,
        b_packed:  &Arc<Tensor<u8>>,
        b_scales:  &Arc<Tensor<u8>>,
        c:         &Arc<Tensor<f32>>,
        m:         u32,
        n:         u32,
        k64_tiles: u32,
        a_row:     u32,
        c_row:     u32,
    ) -> Result<()> {
        let cu_stream = self.stream.cu_stream();

        let n_tiles = n / 8;
        let m_tiles = m / 16;
        debug_assert!(n_tiles >= 1, "n must be >= 8 (got {n})");
        debug_assert!(m_tiles >= 1, "m must be >= 16 (got {m})");

        // Byte offsets: a packed row = k64_tiles * 32 bytes; the m16 tile
        // scale rows advance every 16 packed rows; a C row = n f32.
        let a_row_bytes = (a_row as u64) * (k64_tiles as u64) * 32;
        let a_scales_bytes = ((a_row / 16) as u64) * (k64_tiles as u64) * 2;
        let c_row_bytes = (c_row as u64) * (n as u64) * 4;

        // Kernel choice (all bit-identical; measured winners per grid):
        //   rb4m32 (m32n32/warp, B re-reads halved) when m % 32 == 0 and
        //     its grid >= RB4M32_MIN_BLOCKS,
        //   rb4 (m16n32/warp) when m % 32 != 0 but large-n,
        //   m16n8 otherwise.
        let rb4m32_eligible = m % 32 == 0;
        let rb4m32_grid_x = n_tiles.div_ceil(16);
        if let Some(rb_fn) = self.cu_function_rb4m32 {
            if rb4m32_eligible && rb4m32_grid_x * (m / 32) >= RB4M32_MIN_BLOCKS {
                let mut p_ap = a_packed.device_pointer().cu_deviceptr() as u64 + a_row_bytes;
                let mut p_as = a_scales.device_pointer().cu_deviceptr() as u64 + a_scales_bytes;
                let mut p_bp = b_packed.device_pointer().cu_deviceptr() as u64;
                let mut p_bs = b_scales.device_pointer().cu_deviceptr() as u64;
                let mut p_c  = c.device_pointer().cu_deviceptr() as u64 + c_row_bytes;
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
                        rb_fn,
                        rb4m32_grid_x, m / 32, 1,
                        128, 1, 1,
                        0,
                        cu_stream,
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                if rc != 0 {
                    return Err(KernelError::InvalidConfig(format!(
                        "{KERNEL_NAME_RB4M32} cuLaunchKernel failed: {rc}"
                    )));
                }
                return Ok(());
            }
        }

        // Register-blocked variant on large-n grids (see RB4_MIN_BLOCKS).
        // Bit-identical output; same 8-parameter signature.
        if let Some(rb4_fn) = self.cu_function_rb4 {
            let rb4_grid_x = n_tiles.div_ceil(16);
            if rb4_grid_x * m_tiles >= RB4_MIN_BLOCKS {
                let mut p_ap = a_packed.device_pointer().cu_deviceptr() as u64 + a_row_bytes;
                let mut p_as = a_scales.device_pointer().cu_deviceptr() as u64 + a_scales_bytes;
                let mut p_bp = b_packed.device_pointer().cu_deviceptr() as u64;
                let mut p_bs = b_scales.device_pointer().cu_deviceptr() as u64;
                let mut p_c  = c.device_pointer().cu_deviceptr() as u64 + c_row_bytes;
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
                        rb4_fn,
                        rb4_grid_x, m_tiles, 1,
                        128, 1, 1,
                        0,
                        cu_stream,
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                if rc != 0 {
                    return Err(KernelError::InvalidConfig(format!(
                        "{KERNEL_NAME_RB4} cuLaunchKernel failed: {rc}"
                    )));
                }
                return Ok(());
            }
        }

        let grid_x = n_tiles.div_ceil(4);

        let mut p_ap = a_packed.device_pointer().cu_deviceptr() as u64 + a_row_bytes;
        let mut p_as = a_scales.device_pointer().cu_deviceptr() as u64 + a_scales_bytes;
        let mut p_bp = b_packed.device_pointer().cu_deviceptr() as u64;
        let mut p_bs = b_scales.device_pointer().cu_deviceptr() as u64;
        let mut p_c  = c.device_pointer().cu_deviceptr() as u64 + c_row_bytes;
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
                grid_x, m_tiles, 1,
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
    /// Grid: (ceil(n_tiles/4), m_tiles, 1) -- one m16n8 output tile per warp.
    /// The kernel is validated against a random-data CPU reference for every
    /// shape in tools/mxf4-gemv (including (128, 4096, 4096) at 1024 blocks)
    /// plus a bitwise double-launch determinism check.
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
        self.forward_offset(a_packed, a_scales, b_packed, b_scales, c, m, n, k64_tiles, 0, 0)
    }

    /// GPU-side activation quantization: f32[M*K] -> (packed_e2m1, scales_ue8m0).
    ///
    /// Launches `mxfp4_quantize_activation_tiled` entirely on-GPU; no PCIe transfer.
    /// One block per [16-row x 32-col] tile (matches quantize_activation_to_mxfp4_tiled).
    ///
    /// - `src`:        [m * k] f32, row-major activation (may hold only m_valid rows
    ///                 when m_valid < m -- padding rows are never loaded)
    /// - `packed`:     [m * k / 2] u8, output E2M1 nibbles (lo nibble = first elem)
    /// - `scales`:     [(m/16) * (k/32)] u8, output UE8M0 tile scales
    /// - `m`, `k`:     dimensions (m must be multiple of 16, k multiple of 32)
    /// - `m_valid`:    live rows [0, m_valid); rows >= m_valid are masked to zero so
    ///                 the shared tile-max scale ignores stale buffer contents
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
        m_valid: u32,
        stream: sys::CUstream,
    ) -> Result<()> {
        let qfn = self.cu_quant_fn.ok_or_else(|| {
            KernelError::InvalidConfig("mxfp4 quant kernel not loaded".into())
        })?;

        debug_assert_eq!(m % 16, 0, "m must be multiple of 16");
        debug_assert_eq!(k % 32, 0, "k must be multiple of 32");
        debug_assert!(m_valid <= m, "m_valid={m_valid} must be <= m={m}");

        let k32_blocks = k / 32;
        let m16_tiles  = m / 16;
        let grid_x     = m16_tiles * k32_blocks;  // one block per tile

        let mut p_src    = src.device_pointer().cu_deviceptr() as u64;
        let mut p_packed = packed.device_pointer().cu_deviceptr() as u64;
        let mut p_scales = scales.device_pointer().cu_deviceptr() as u64;
        let mut pm       = m;
        let mut pk       = k;
        let mut pk32     = k32_blocks;
        let mut pmv      = m_valid;

        let mut params: [*mut std::ffi::c_void; 7] = [
            &mut p_src    as *mut _ as *mut std::ffi::c_void,
            &mut p_packed as *mut _ as *mut std::ffi::c_void,
            &mut p_scales as *mut _ as *mut std::ffi::c_void,
            &mut pm       as *mut _ as *mut std::ffi::c_void,
            &mut pk       as *mut _ as *mut std::ffi::c_void,
            &mut pk32     as *mut _ as *mut std::ffi::c_void,
            &mut pmv      as *mut _ as *mut std::ffi::c_void,
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

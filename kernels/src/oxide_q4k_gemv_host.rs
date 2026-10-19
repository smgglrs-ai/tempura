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

/// The PTX source compiled by `cargo oxide build tools/oxide-gemv-q4k`.
#[cfg(feature = "oxide-gemv")]
static OXIDE_Q4K_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_q4k_gemv.ptx"));

const KERNEL_ENTRY: &str = "q4k_gemv_decode_v2";
const KERNEL_BATCH: &str = "q4k_gemv_decode_v2_batch";
const KERNEL_COLOCATED: &str = "q4k_gemm_colocated";
const KERNEL_SMEM: &str = "q4k_gemm_smem";
const KERNEL_Q8_QUANTIZE: &str = "quantize_q8_1";
const KERNEL_Q8_GEMV: &str = "q4k_gemv_decode_q8";
const KERNEL_Q8_GEMV_V2: &str = "q4k_gemv_decode_q8_v2";
const KERNEL_Q8_GEMV_V3: &str = "q4k_gemv_decode_q8_v3";
const KERNEL_Q8_GEMV_V3R: &str = "q4k_gemv_decode_q8_v3r"; // raw GGUF 36-u32 format
const KERNEL_FUSED_SILU: &str = "q4k_gemv_fused_silu_mul"; // fused gate+up+SiLU
const WARPS_PER_BLOCK: u32 = 4;
const BLOCK_DIM: u32 = WARPS_PER_BLOCK * 32; // 128

/// Launches the oxide-compiled Q4_K GEMV kernel for M=1 decode.
pub struct OxideQ4kGemv {
    cu_module:      CUmodule,
    cu_function:    CUfunction,
    /// Optional Q8_1 quantize kernel (present after `cargo oxide build` with new kernels).
    cu_q8_quantize:    Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV kernel (mini-warp only, no K-parallelism).
    cu_q8_gemv:        Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV v2 kernel (K-parallelism + mini-warp dp4a combined).
    cu_q8_gemv_v2:     Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV v3 kernel (VDR=2 double-block K-loop).
    cu_q8_gemv_v3:     Option<CUfunction>,
    /// Optional Q4K+Q8_1 GEMV v3r kernel (v3 + raw GGUF 36-u32 format, 14% less DRAM).
    cu_q8_gemv_v3r:    Option<CUfunction>,
    /// Optional batched Q4K GEMV: grid=(N,M,1), L2-shared weight blocks.
    cu_batch_gemv:     Option<CUfunction>,
    /// Optional colocated GEMM: all M rows in one block, L2 broadcast for weights.
    cu_colocated:      Option<CUfunction>,
    /// Optional smem-tiled GEMM: row-0 loads weights into smem, all M rows compute from smem.
    cu_smem:           Option<CUfunction>,
    /// Optional fused gate+up GEMV with inline SiLU: single launch for both FFN projections.
    cu_fused_silu:     Option<CUfunction>,
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
}

unsafe impl Send for OxideQ4kGemv {}
unsafe impl Sync for OxideQ4kGemv {}

impl Drop for OxideQ4kGemv {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
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

            // Opportunistically load the Q8_1 kernels — they are absent until the
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
            let cu_q8_gemv_v2  = load_fn(KERNEL_Q8_GEMV_V2);
            let cu_q8_gemv_v3  = load_fn(KERNEL_Q8_GEMV_V3);
            let cu_q8_gemv_v3r = load_fn(KERNEL_Q8_GEMV_V3R);
            let cu_batch_gemv  = load_fn(KERNEL_BATCH);
            let cu_colocated   = load_fn(KERNEL_COLOCATED);
            let cu_smem        = load_fn(KERNEL_SMEM);
            let cu_fused_silu  = load_fn(KERNEL_FUSED_SILU);

            eprintln!(
                "oxide-gemv-q4k: loaded Q4_K GEMV kernel (PTX {} bytes, entry={}, q8_path={}, q8v2={}, q8v3={}, q8v3r={}, fused_silu={}, uses_raw_gguf={}, batch={})",
                OXIDE_Q4K_PTX.len(),
                KERNEL_ENTRY,
                cu_q8_gemv.is_some(),
                cu_q8_gemv_v2.is_some(),
                cu_q8_gemv_v3.is_some(),
                cu_q8_gemv_v3r.is_some(),
                cu_fused_silu.is_some(),
                false, // uses_raw_gguf — see note below
                cu_batch_gemv.is_some(),
            );
            // v3r: 36-u32 raw GGUF format. Benchmarked and DISABLED (performance regression).
            // uses_raw_gguf=true benchmarked at 101 tok/s vs 107 tok/s with packed format.
            // Root cause: on-the-fly 6-bit scale unpacking in v3r adds compute overhead that
            // exceeds the 14% DRAM bandwidth savings (144 vs 168 bytes/block).
            // Correctness is verified (tail and paired loops both pass, prefill format-detection
            // in gemm_any_d2d_slot handles 36-u32 format correctly), but v3 is faster.
            let uses_raw_gguf = false;
            Some(Self {
                cu_module, cu_function, cu_q8_quantize, cu_q8_gemv, cu_q8_gemv_v2, cu_q8_gemv_v3, cu_q8_gemv_v3r, cu_batch_gemv, cu_colocated, cu_smem, cu_fused_silu, uses_raw_gguf, stream,
                q8_qs_scratch:      std::cell::RefCell::new(None),
                q8_ds_scratch:      std::cell::RefCell::new(None),
                q8_out_scratch:     std::cell::RefCell::new(None),
                fused_silu_scratch: std::cell::RefCell::new(None),
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

    /// Batched Q4_K GEMV: `out[m, n] = dequant(packed[n,k]) × x[m,k]`.
    ///
    /// Grid = (N, M, 1) — each block handles one (n_out, row) pair.
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
    /// then compute FMAs from smem — eliminating (M-1)/M of weight DRAM traffic.
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
    /// Must be called before the first `cuStreamBeginCapture` — allocations on
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
        Ok(())
    }

    /// Returns true when the fused gate+up SiLU kernel is compiled into the PTX.
    pub fn has_fused_silu(&self) -> bool {
        self.cu_fused_silu.is_some() && self.cu_q8_quantize.is_some()
    }

    /// Returns true when the Q8_1 quantize + integer GEMV path is compiled into the PTX.
    pub fn has_q8_path(&self) -> bool {
        self.cu_q8_quantize.is_some()
            && (self.cu_q8_gemv_v3.is_some() || self.cu_q8_gemv_v2.is_some() || self.cu_q8_gemv.is_some())
    }

    /// Quantize `x` (f32 K-vector) → Q8_1 packed int8 + scales, then run Q4K GEMV.
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
        // Prefer v3r (raw GGUF, 14% less DRAM) > v3 (VDR=2) > v2 > v1.
        let (q8_quant_fn, q8_gemv_fn, kernel_variant) = match &self.cu_q8_quantize {
            Some(q) => {
                if self.uses_raw_gguf {
                    if let Some(g3r) = &self.cu_q8_gemv_v3r {
                        (*q, *g3r, 4u8)
                    } else if let Some(g3) = &self.cu_q8_gemv_v3 {
                        (*q, *g3, 3u8)
                    } else if let Some(g2) = &self.cu_q8_gemv_v2 {
                        (*q, *g2, 2u8)
                    } else if let Some(g) = &self.cu_q8_gemv {
                        (*q, *g, 1u8)
                    } else {
                        return Err(KernelError::InvalidConfig("Q8 GEMV kernel not loaded".into()));
                    }
                } else if let Some(g3) = &self.cu_q8_gemv_v3 {
                    (*q, *g3, 3u8)
                } else if let Some(g2) = &self.cu_q8_gemv_v2 {
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

        // Use pre-allocated scratch when available (required during CUDA graph capture —
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

        // ── Launch quantize_q8_1 ────────────────────────────────────────────
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

        // ── Launch q4k_gemv_decode_q8 ───────────────────────────────────────
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

        // No stream.synchronize() here — the caller (CUDA graph or direct dispatch)
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
    pub fn forward_fused_silu_mul_d2d(
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

        // ── Launch quantize_q8_1 ────────────────────────────────────────────
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

        // ── Launch q4k_gemv_fused_silu_mul ──────────────────────────────────
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
}

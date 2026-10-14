/// Host-side wrapper for the oxide decode attention kernel.
///
/// Unlike `CutileFlashAttention::decode_attention_device_scratch`, this accepts
/// `kv_len_dev` as a device tensor whose VALUE is read at kernel execution time.
/// The POINTER is baked into the CUDA graph at capture; the value is updated H→D
/// each step. This eliminates the stale-kv_len problem.
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, Stream};
use cutile::tensor::Tensor;

use crate::{KernelError, Result};

const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_decode_attn.ptx"));
const KERNEL_NAME:      &str = "oxide_decode_attn";
const KERNEL_KPAR_NAME: &str = "oxide_decode_attn_kpar";
const WARPS_PER_BLOCK_KPAR: u32 = 8;

pub struct OxideDecodeAttn {
    cu_module:         cuda_core::sys::CUmodule,
    cu_function:       cuda_core::sys::CUfunction,
    cu_function_kpar:  Option<cuda_core::sys::CUfunction>,
    stream:            Arc<Stream>,
}

unsafe impl Send for OxideDecodeAttn {}
unsafe impl Sync for OxideDecodeAttn {}

impl OxideDecodeAttn {
    /// Load PTX from the embedded build artifact.
    pub fn new(stream: Arc<Stream>) -> Result<Self> {
        let mut cu_module = std::ptr::null_mut::<cuda_core::sys::CUmod_st>();
        let rc = unsafe {
            cuda_core::sys::cuModuleLoadData(
                &mut cu_module,
                PTX.as_ptr() as *const std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn: cuModuleLoadData failed: {rc}"
            )));
        }

        let name = std::ffi::CString::new(KERNEL_NAME).unwrap();
        let mut cu_function = std::ptr::null_mut::<cuda_core::sys::CUfunc_st>();
        let rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(&mut cu_function, cu_module, name.as_ptr())
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn: cuModuleGetFunction({KERNEL_NAME}) failed: {rc}"
            )));
        }

        // Try to load the K-parallel variant (optional — falls back to single-warp).
        let kpar_name = std::ffi::CString::new(KERNEL_KPAR_NAME).unwrap();
        let mut cu_function_kpar = std::ptr::null_mut::<cuda_core::sys::CUfunc_st>();
        let kpar_rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(&mut cu_function_kpar, cu_module, kpar_name.as_ptr())
        };
        let cu_function_kpar = if kpar_rc == 0 { Some(cu_function_kpar) } else { None };

        eprintln!("[oxide] loaded decode-attn kernel (PTX {} bytes, kpar={})",
            PTX.len(), cu_function_kpar.is_some());
        Ok(Self { cu_module, cu_function, cu_function_kpar, stream })
    }

    /// Run decode attention with kv_len read from a device tensor at execution time.
    ///
    /// `kv_len_dev` must be a 1-element i32 (or f32 bit-encoded i32) device tensor.
    /// The engine writes the current step's kv_len H→D before each graph replay.
    pub fn forward_d2d(
        &self,
        out:        &Arc<Tensor<f32>>,   // [n_heads * head_dim] — written in-place
        q:          &Arc<Tensor<f32>>,   // [n_heads * head_dim]
        k_cache:    &Arc<Tensor<f32>>,   // [gpu_max_seq_len * kv_dim]
        v_cache:    &Arc<Tensor<f32>>,   // [gpu_max_seq_len * kv_dim]
        kv_len_dev: &Arc<Tensor<f32>>,   // 1-element; value = kv_len (bits match i32)
        n_heads:    usize,
        n_kv_heads: usize,
        head_dim:   usize,
        scale:      f32,
    ) -> Result<()> {
        let heads_per_group = (n_heads / n_kv_heads) as u32;
        let kv_len_ptr = kv_len_dev.device_pointer().cu_deviceptr() as u64;

        // Build raw parameter pointers for cuLaunchKernel.
        let mut out_ptr   = out.device_pointer().cu_deviceptr() as u64;
        let mut q_ptr     = q.device_pointer().cu_deviceptr() as u64;
        let mut k_ptr     = k_cache.device_pointer().cu_deviceptr() as u64;
        let mut v_ptr     = v_cache.device_pointer().cu_deviceptr() as u64;
        let mut kv_p      = kv_len_ptr;
        let mut sc        = scale;
        let mut nkv       = n_kv_heads as u32;
        let mut hd        = head_dim as u32;
        let mut hpg       = heads_per_group;

        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut q_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut k_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut v_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut kv_p    as *mut _ as *mut std::ffi::c_void,
            &mut sc      as *mut _ as *mut std::ffi::c_void,
            &mut nkv     as *mut _ as *mut std::ffi::c_void,
            &mut hd      as *mut _ as *mut std::ffi::c_void,
            &mut hpg     as *mut _ as *mut std::ffi::c_void,
        ];

        let shared_bytes = (2 * head_dim * 4) as u32;
        let rc = unsafe {
            cuda_core::sys::cuLaunchKernel(
                self.cu_function,
                n_heads as u32, 1, 1,   // grid
                32, 1, 1,                // block (one warp)
                shared_bytes,
                self.stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn::forward_d2d cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Whether the K-parallel variant is available.
    pub fn has_kpar(&self) -> bool { self.cu_function_kpar.is_some() }

    /// K-parallel decode attention (WARPS_PER_BLOCK_KPAR warps per head).
    pub fn forward_kpar_d2d(
        &self,
        out:        &Arc<Tensor<f32>>,
        q:          &Arc<Tensor<f32>>,
        k_cache:    &Arc<Tensor<f32>>,
        v_cache:    &Arc<Tensor<f32>>,
        kv_len_dev: &Arc<Tensor<f32>>,
        n_heads:    usize,
        n_kv_heads: usize,
        head_dim:   usize,
        scale:      f32,
    ) -> Result<()> {
        let f = match self.cu_function_kpar {
            Some(f) => f,
            None => return self.forward_d2d(out, q, k_cache, v_cache, kv_len_dev, n_heads, n_kv_heads, head_dim, scale),
        };
        let heads_per_group = (n_heads / n_kv_heads) as u32;
        let kv_len_ptr = kv_len_dev.device_pointer().cu_deviceptr() as u64;

        let mut out_ptr = out.device_pointer().cu_deviceptr() as u64;
        let mut q_ptr   = q.device_pointer().cu_deviceptr() as u64;
        let mut k_ptr   = k_cache.device_pointer().cu_deviceptr() as u64;
        let mut v_ptr   = v_cache.device_pointer().cu_deviceptr() as u64;
        let mut kv_p    = kv_len_ptr;
        let mut sc      = scale;
        let mut nkv     = n_kv_heads as u32;
        let mut hd      = head_dim as u32;
        let mut hpg     = heads_per_group;

        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut q_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut k_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut v_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut kv_p    as *mut _ as *mut std::ffi::c_void,
            &mut sc      as *mut _ as *mut std::ffi::c_void,
            &mut nkv     as *mut _ as *mut std::ffi::c_void,
            &mut hd      as *mut _ as *mut std::ffi::c_void,
            &mut hpg     as *mut _ as *mut std::ffi::c_void,
        ];

        // smem: (head_dim*(W+1) + 2*W) * 4 bytes
        let w = WARPS_PER_BLOCK_KPAR as usize;
        let shared_bytes = ((head_dim * (w + 1) + 2 * w) * 4) as u32;
        let block_dim = WARPS_PER_BLOCK_KPAR * 32;

        let rc = unsafe {
            cuda_core::sys::cuLaunchKernel(
                f,
                n_heads as u32, 1, 1,
                block_dim, 1, 1,
                shared_bytes,
                self.stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn::forward_kpar_d2d cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }
}

impl Drop for OxideDecodeAttn {
    fn drop(&mut self) {
        unsafe { cuda_core::sys::cuModuleUnload(self.cu_module) };
    }
}

/// Host-side wrapper for the oxide decode attention kernel.
///
/// Unlike `CutileFlashAttention::decode_attention_device_scratch`, this accepts
/// `kv_len_dev` as a device tensor whose VALUE is read at kernel execution time.
/// The POINTER is baked into the CUDA graph at capture; the value is updated H->D
/// each step. This eliminates the stale-kv_len problem.
use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer, Stream};
use cutile::tensor::Tensor;

use crate::{KernelError, Result};

const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oxide_decode_attn.ptx"));
const KERNEL_NAME:       &str = "oxide_decode_attn";
const KERNEL_KPAR_NAME:  &str = "oxide_decode_attn_kpar";
const KERNEL_BATCH_NAME: &str = "oxide_decode_attn_batch";
const WARPS_PER_BLOCK_KPAR: u32 = 8;

pub struct OxideDecodeAttn {
    cu_module:          cuda_core::sys::CUmodule,
    cu_function:        cuda_core::sys::CUfunction,
    cu_function_kpar:   Option<cuda_core::sys::CUfunction>,
    cu_function_batch:  Option<cuda_core::sys::CUfunction>,
    stream:             Arc<Stream>,
}

unsafe impl Send for OxideDecodeAttn {}
unsafe impl Sync for OxideDecodeAttn {}

impl OxideDecodeAttn {
    /// Load PTX from the embedded build artifact.
    pub fn new(stream: Arc<Stream>) -> Result<Self> {
        // cuModuleLoadData for PTX text format requires a null-terminated string.
        // include_bytes! does NOT add a null terminator, so we must append one.
        let ptx_nul: Vec<u8> = PTX.iter().copied().chain(std::iter::once(0u8)).collect();
        let mut cu_module = std::ptr::null_mut::<cuda_core::sys::CUmod_st>();
        let rc = unsafe {
            cuda_core::sys::cuModuleLoadData(
                &mut cu_module,
                ptx_nul.as_ptr() as *const std::ffi::c_void,
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

        // Try to load the K-parallel variant (optional -- falls back to single-warp).
        let kpar_name = std::ffi::CString::new(KERNEL_KPAR_NAME).unwrap();
        let mut cu_function_kpar = std::ptr::null_mut::<cuda_core::sys::CUfunc_st>();
        let kpar_rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(&mut cu_function_kpar, cu_module, kpar_name.as_ptr())
        };
        let cu_function_kpar = if kpar_rc == 0 { Some(cu_function_kpar) } else { None };

        // Try to load the batch variant.
        let batch_name = std::ffi::CString::new(KERNEL_BATCH_NAME).unwrap();
        let mut cu_function_batch = std::ptr::null_mut::<cuda_core::sys::CUfunc_st>();
        let batch_rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(&mut cu_function_batch, cu_module, batch_name.as_ptr())
        };
        let cu_function_batch = if batch_rc == 0 { Some(cu_function_batch) } else { None };

        eprintln!("[oxide] loaded decode-attn kernel (PTX {} bytes, kpar={}, batch={})",
            PTX.len(), cu_function_kpar.is_some(), cu_function_batch.is_some());
        Ok(Self { cu_module, cu_function, cu_function_kpar, cu_function_batch, stream })
    }

    /// Run decode attention with kv_len read from a device tensor at execution time.
    ///
    /// `kv_len_dev` must be a 1-element i32 (or f32 bit-encoded i32) device tensor.
    /// The engine writes the current step's kv_len H->D before each graph replay.
    pub fn forward_d2d(
        &self,
        out:        &Arc<Tensor<f32>>,   // [n_heads * head_dim] -- written in-place
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

        // PTX param layout (9 params) matching oxide_decode_attn kernel signature:
        // 0:out(*mut f32), 1:q, 2:k_cache, 3:v_cache, 4:kv_len_ptr,
        // 5:scale(f32), 6:n_kv_heads(u32), 7:head_dim(u32), 8:heads_per_group(u32)
        // n_heads is the grid X dimension; kv_stride = n_kv_heads*head_dim computed in kernel.
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
            &mut out_ptr as *mut _ as *mut std::ffi::c_void,  // param 0: out (*mut f32)
            &mut q_ptr   as *mut _ as *mut std::ffi::c_void,  // param 1: q
            &mut k_ptr   as *mut _ as *mut std::ffi::c_void,  // param 2: k_cache
            &mut v_ptr   as *mut _ as *mut std::ffi::c_void,  // param 3: v_cache
            &mut kv_p    as *mut _ as *mut std::ffi::c_void,  // param 4: kv_len_ptr
            &mut sc      as *mut _ as *mut std::ffi::c_void,  // param 5: scale
            &mut nkv     as *mut _ as *mut std::ffi::c_void,  // param 6: n_kv_heads
            &mut hd      as *mut _ as *mut std::ffi::c_void,  // param 7: head_dim
            &mut hpg     as *mut _ as *mut std::ffi::c_void,  // param 8: heads_per_group
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
            &mut out_ptr as *mut _ as *mut std::ffi::c_void,  // param 0: out (*mut f32)
            &mut q_ptr   as *mut _ as *mut std::ffi::c_void,  // param 1: q
            &mut k_ptr   as *mut _ as *mut std::ffi::c_void,  // param 2: k_cache
            &mut v_ptr   as *mut _ as *mut std::ffi::c_void,  // param 3: v_cache
            &mut kv_p    as *mut _ as *mut std::ffi::c_void,  // param 4: kv_len_ptr
            &mut sc      as *mut _ as *mut std::ffi::c_void,  // param 5: scale
            &mut nkv     as *mut _ as *mut std::ffi::c_void,  // param 6: n_kv_heads
            &mut hd      as *mut _ as *mut std::ffi::c_void,  // param 7: head_dim
            &mut hpg     as *mut _ as *mut std::ffi::c_void,  // param 8: heads_per_group
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

    /// Whether the batch variant is available.
    pub fn has_batch(&self) -> bool { self.cu_function_batch.is_some() }

    /// Batch decode attention for CUDA graph capture (pre-allocated device arrays).
    ///
    /// Unlike `forward_batch_d2d`, does NOT call cuMemAllocAsync.
    /// All device arrays must be pre-allocated with stable addresses.
    /// `kv_lens_dev` is updated H->D BEFORE each graph launch by the caller;
    /// `k_ptrs_dev` and `v_ptrs_dev` are stable (baked in at graph capture).
    pub fn forward_batch_graph_d2d(
        &self,
        out:         &Arc<Tensor<f32>>,  // [M * n_heads * head_dim]
        q:           &Arc<Tensor<f32>>,  // [M * n_heads * head_dim]
        k_ptrs_dev:  u64,                // CUdeviceptr to [M x u64] K-cache pointers
        v_ptrs_dev:  u64,                // CUdeviceptr to [M x u64] V-cache pointers
        kv_lens_dev: u64,                // CUdeviceptr to [M x i32] kv_len values
        m:           usize,
        n_heads:     usize,
        n_kv_heads:  usize,
        head_dim:    usize,
        scale:       f32,
    ) -> Result<()> {
        let f = match self.cu_function_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig(
                "oxide_decode_attn_batch not available in this PTX".into()
            )),
        };

        let kv_stride = (n_kv_heads * head_dim) as u32;

        let mut out_ptr  = out.device_pointer().cu_deviceptr() as u64;
        let mut q_ptr    = q.device_pointer().cu_deviceptr() as u64;
        let mut kp       = k_ptrs_dev;
        let mut vp       = v_ptrs_dev;
        let mut kvlp     = kv_lens_dev;
        let mut bs       = m as u32;
        let mut nh       = n_heads as u32;
        let mut nkv      = n_kv_heads as u32;
        let mut hd       = head_dim as u32;
        let mut kv_str   = kv_stride;
        let mut sc       = scale;

        let mut params: [*mut std::ffi::c_void; 11] = [
            &mut out_ptr as *mut _ as *mut _,
            &mut q_ptr   as *mut _ as *mut _,
            &mut kp      as *mut _ as *mut _,
            &mut vp      as *mut _ as *mut _,
            &mut kvlp    as *mut _ as *mut _,
            &mut bs      as *mut _ as *mut _,
            &mut nh      as *mut _ as *mut _,
            &mut nkv     as *mut _ as *mut _,
            &mut hd      as *mut _ as *mut _,
            &mut kv_str  as *mut _ as *mut _,
            &mut sc      as *mut _ as *mut _,
        ];

        let shared_bytes = (2 * head_dim * 4) as u32;
        let grid_x = (m * n_heads) as u32;
        let stream = self.stream.cu_stream();

        let rc = unsafe {
            cuda_core::sys::cuLaunchKernel(
                f,
                grid_x, 1, 1,
                32, 1, 1,
                shared_bytes,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn::forward_batch_graph_d2d cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Batched decode attention: one block per (request, head) pair.
    ///
    /// `k_cache_ptrs` and `v_cache_ptrs` are CUdeviceptr values (device pointers
    /// to each request's KV cache) stored in host Vecs -- uploaded to device as a
    /// small array before the kernel launch.
    pub fn forward_batch_d2d(
        &self,
        out:          &Arc<Tensor<f32>>,   // [M * n_heads * head_dim]
        q:            &Arc<Tensor<f32>>,   // [M * n_heads * head_dim]
        k_cache_ptrs: &[u64],              // [M] CUdeviceptr per request
        v_cache_ptrs: &[u64],              // [M] CUdeviceptr per request
        kv_lens:      &[i32],              // [M] current kv_len per request
        n_heads:      usize,
        n_kv_heads:   usize,
        head_dim:     usize,
        scale:        f32,
    ) -> Result<()> {
        let f = match self.cu_function_batch {
            Some(f) => f,
            None => return Err(KernelError::InvalidConfig(
                "oxide_decode_attn_batch not available in this PTX".into()
            )),
        };
        let batch_size = kv_lens.len();

        // Upload pointer arrays to device (tiny H->D, ~8 x batch_size bytes each).
        let stream = self.stream.cu_stream();
        let k_ptrs_dev = unsafe {
            let mut dev_ptr: u64 = 0;
            cuda_core::sys::cuMemAllocAsync(
                &mut dev_ptr as *mut u64,
                (batch_size * 8) as usize,
                stream,
            );
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                dev_ptr,
                k_cache_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_size * 8,
                stream,
            );
            dev_ptr
        };
        let v_ptrs_dev = unsafe {
            let mut dev_ptr: u64 = 0;
            cuda_core::sys::cuMemAllocAsync(
                &mut dev_ptr as *mut u64,
                (batch_size * 8) as usize,
                stream,
            );
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                dev_ptr,
                v_cache_ptrs.as_ptr() as *const std::ffi::c_void,
                batch_size * 8,
                stream,
            );
            dev_ptr
        };
        let kv_lens_dev = unsafe {
            let mut dev_ptr: u64 = 0;
            cuda_core::sys::cuMemAllocAsync(
                &mut dev_ptr as *mut u64,
                (batch_size * 4) as usize,
                stream,
            );
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                dev_ptr,
                kv_lens.as_ptr() as *const std::ffi::c_void,
                batch_size * 4,
                stream,
            );
            dev_ptr
        };

        let kv_stride = (n_kv_heads * head_dim) as u32;
        let heads_per_group = (n_heads / n_kv_heads) as u32;

        let mut out_ptr   = out.device_pointer().cu_deviceptr() as u64;
        let mut q_ptr     = q.device_pointer().cu_deviceptr() as u64;
        let mut kp        = k_ptrs_dev;
        let mut vp        = v_ptrs_dev;
        let mut kvlp      = kv_lens_dev;
        let mut bs        = batch_size as u32;
        let mut nh        = n_heads as u32;
        let mut nkv       = n_kv_heads as u32;
        let mut hd        = head_dim as u32;
        let mut kv_str    = kv_stride;
        let mut sc        = scale;

        let mut params: [*mut std::ffi::c_void; 11] = [
            &mut out_ptr as *mut _ as *mut _,
            &mut q_ptr   as *mut _ as *mut _,
            &mut kp      as *mut _ as *mut _,
            &mut vp      as *mut _ as *mut _,
            &mut kvlp    as *mut _ as *mut _,
            &mut bs      as *mut _ as *mut _,
            &mut nh      as *mut _ as *mut _,
            &mut nkv     as *mut _ as *mut _,
            &mut hd      as *mut _ as *mut _,
            &mut kv_str  as *mut _ as *mut _,
            &mut sc      as *mut _ as *mut _,
        ];

        let shared_bytes = (2 * head_dim * 4) as u32;
        let grid_x = (batch_size * n_heads) as u32;

        let rc = unsafe {
            cuda_core::sys::cuLaunchKernel(
                f,
                grid_x, 1, 1,
                32, 1, 1,
                shared_bytes,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        // Free temp device allocations asynchronously.
        unsafe {
            cuda_core::sys::cuMemFreeAsync(k_ptrs_dev, stream);
            cuda_core::sys::cuMemFreeAsync(v_ptrs_dev, stream);
            cuda_core::sys::cuMemFreeAsync(kv_lens_dev, stream);
        }

        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn::forward_batch_d2d cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Elementwise SiLU(gate) * up -- fully on GPU, no PCIe traffic.
    /// `n` = batch_size * inter_pad (total elements).
    pub fn silu_mul_d2d(
        &self,
        out:  &Arc<Tensor<f32>>,
        gate: &Arc<Tensor<f32>>,
        up:   &Arc<Tensor<f32>>,
        n:    usize,
    ) -> Result<()> {
        let name = std::ffi::CString::new("oxide_silu_mul").unwrap();
        let mut cu_fn = std::ptr::null_mut::<cuda_core::sys::CUfunc_st>();
        let rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(&mut cu_fn, self.cu_module, name.as_ptr())
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn: cuModuleGetFunction(oxide_silu_mul) failed: {rc}"
            )));
        }

        let block_dim: u32 = 256;
        let grid_dim  = ((n as u32) + block_dim - 1) / block_dim;

        let mut out_ptr  = out.device_pointer().cu_deviceptr()  as u64;
        let mut gate_ptr = gate.device_pointer().cu_deviceptr() as u64;
        let mut up_ptr   = up.device_pointer().cu_deviceptr()   as u64;
        let mut n_u32    = n as u32;

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut out_ptr  as *mut _ as *mut std::ffi::c_void,
            &mut gate_ptr as *mut _ as *mut std::ffi::c_void,
            &mut up_ptr   as *mut _ as *mut std::ffi::c_void,
            &mut n_u32    as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            cuda_core::sys::cuLaunchKernel(
                cu_fn,
                grid_dim, 1, 1,
                block_dim, 1, 1,
                0,
                self.stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "OxideDecodeAttn::silu_mul_d2d cuLaunchKernel failed: {rc}"
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

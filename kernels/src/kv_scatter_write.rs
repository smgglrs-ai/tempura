/// Scatter-write kernel for the KV cache: copies `kv_dim` f32 elements from
/// a source tensor into the KV cache at the row given by a device-side offset.
///
/// The device pointer `write_off_ptr` is captured once in the CUDA graph and
/// never changes.  The *value* at that address is updated H→D each step (4
/// bytes).  Because the kernel reads the offset at execution time there are no
/// `cuGraphExecMemcpyNodeSetParams` calls — the graph is topology-stable.
use crate::{KernelError, Result};
use cuda_core::sys;

// PTX scatter-write kernel.
//
// Grid:  ceil(kv_dim / BLOCK_SIZE) blocks.
// Block: BLOCK_SIZE threads (one thread per f32 element).
//
// Parameters:
//   param_src          – device pointer to the source tensor   (kv_dim f32 elements)
//   param_cache_base   – device pointer to the KV cache row 0  ([gpu_max_seq_len × kv_dim] f32)
//   param_write_off_ptr– device pointer to a single i32 holding the target row index
//   param_kv_dim       – number of f32 elements per row (u32)
// PTX ISA 8.7 / sm_120 (Blackwell consumer, RTX 5090).
// Numbered register style (%r<N>, %rd<N>) is required by the CUDA 13 ptxas.
const PTX_SOURCE: &str = r#"
.version 8.7
.target sm_120
.address_size 64

.visible .entry kv_scatter_write(
    .param .u64 param_src,
    .param .u64 param_cache_base,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim
)
{
    // %r0  = param_kv_dim
    // %r1  = ctaid.x   %r2 = ntid.x   %r3 = global tid
    // %r4  = s32 write_off read from device
    // %rd0 = src ptr   %rd1 = cache_base   %rd2 = write_off device ptr
    // %rd3 = row (sign-extended)  %rd4 = kv_dim64  %rd5 = tid64
    // %rd6 = dst element index    %rd7 = dst byte addr  %rd8 = src byte addr
    .reg .u32   %r<8>;
    .reg .u64   %rd<10>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_cache_base];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];

    // global thread index = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    // bounds check: if tid >= kv_dim, exit
    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done;

    // load write_offset from device (sign-extend to u64; always >= 0)
    ld.global.s32 %r4,  [%rd2];
    cvt.u64.s32   %rd3, %r4;
    cvt.u64.u32   %rd4, %r0;
    cvt.u64.u32   %rd5, %r3;

    // dst element index = row * kv_dim + tid
    mul.lo.u64  %rd6, %rd3, %rd4;
    add.u64     %rd6, %rd6, %rd5;

    // dst byte address = cache_base + dst_idx * 4
    shl.b64  %rd7, %rd6, 2;
    add.u64  %rd7, %rd1, %rd7;

    // src byte address = src + tid * 4
    shl.b64  %rd8, %rd5, 2;
    add.u64  %rd8, %rd0, %rd8;

    ld.global.f32 %f1, [%rd8];
    st.global.f32 [%rd7], %f1;

done:
    ret;
}
"#;

const BLOCK_SIZE: u32 = 256;
const KERNEL_NAME: &str = "kv_scatter_write";

/// Scatter-write kernel loader and launcher.
pub struct KvScatterWrite {
    cu_module:   sys::CUmodule,
    cu_function: sys::CUfunction,
}

unsafe impl Send for KvScatterWrite {}
unsafe impl Sync for KvScatterWrite {}

impl KvScatterWrite {
    /// Load the PTX module and resolve the kernel function.
    ///
    /// Must be called after a CUDA device context is active (i.e. after any
    /// `CutileMatMul::new()` or similar CUDA initialisation).
    pub fn new() -> Result<Self> {
        let mut ptx_nul: Vec<u8> = PTX_SOURCE.bytes().collect();
        ptx_nul.push(0u8); // cuModuleLoadData requires null-terminated PTX

        let mut cu_module = std::ptr::null_mut::<sys::CUmod_st>();
        let rc = unsafe {
            sys::cuModuleLoadData(
                &mut cu_module,
                ptx_nul.as_ptr() as *const std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleLoadData failed: {rc}"
            )));
        }

        let name = std::ffi::CString::new(KERNEL_NAME).unwrap();
        let mut cu_function = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function, cu_module, name.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME}) failed: {rc}"
            )));
        }

        Ok(Self { cu_module, cu_function })
    }

    /// Copy `kv_dim` f32 elements from `src_ptr` into
    /// `cache_base[*write_off_ptr * kv_dim .. (*write_off_ptr + 1) * kv_dim]`.
    ///
    /// All pointers are device addresses.  The call is asynchronous on `cu_stream`.
    pub fn copy_with_device_offset(
        &self,
        src_ptr:       sys::CUdeviceptr,
        cache_base:    sys::CUdeviceptr,
        write_off_ptr: sys::CUdeviceptr,
        kv_dim:        u32,
        cu_stream:     sys::CUstream,
    ) -> Result<()> {
        let grid = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_src   = src_ptr;
        let mut p_cache = cache_base;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;

        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut p_src   as *mut _ as *mut std::ffi::c_void,
            &mut p_cache as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function,
                grid, 1, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_with_device_offset cuLaunchKernel failed: {rc}"
            )));
        }
        Ok(())
    }
}

impl Drop for KvScatterWrite {
    fn drop(&mut self) {
        if !self.cu_module.is_null() {
            let _ = unsafe { sys::cuModuleUnload(self.cu_module) };
        }
    }
}

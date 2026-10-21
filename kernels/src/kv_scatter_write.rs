/// Scatter-write kernel for the KV cache: copies `kv_dim` f32 elements from
/// a source tensor into the KV cache at the row given by a device-side offset.
///
/// The device pointer `write_off_ptr` is captured once in the CUDA graph and
/// never changes.  The *value* at that address is updated H->D each step (4
/// bytes).  Because the kernel reads the offset at execution time there are no
/// `cuGraphExecMemcpyNodeSetParams` calls -- the graph is topology-stable.
///
/// A batch variant (`kv_scatter_write_batch`) handles M simultaneous sequences
/// using `blockIdx.y` as the sequence index; it reads per-sequence source and
/// cache-base pointers from device-resident pointer arrays.
use crate::{KernelError, Result};
use cuda_core::sys;

// PTX module containing both the single-sequence and M-sequence scatter-write
// kernels.  Both entries share the same .version / .target / .address_size header.
//
// Single-sequence kernel (kv_scatter_write):
//   Grid:  ceil(kv_dim / BLOCK_SIZE) blocks.
//   Block: BLOCK_SIZE threads.
//   Params: src (u64), cache_base (u64), write_off_ptr (u64 -> *const i32), kv_dim (u32).
//
// Batch kernel (kv_scatter_write_batch):
//   Grid:  (ceil(kv_dim / BLOCK_SIZE), M, 1).
//   Block: BLOCK_SIZE threads.
//   Params: src_ptrs (*const u64 [M]), cache_bases (*const u64 [M]),
//           write_off_ptr (u64 -> *const i32 -- single offset shared by all seqs),
//           kv_dim (u32), m (u32).
//
// PTX ISA 8.7 / sm_120 (Blackwell consumer, RTX 5090).
const PTX_SOURCE: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// Single-sequence entry
.visible .entry kv_scatter_write(
    .param .u64 param_src,
    .param .u64 param_cache_base,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim
)
{
    // %r0 = kv_dim  %r1 = ctaid.x  %r2 = ntid.x  %r3 = global tid  %r4 = write_off
    // %rd0 = src  %rd1 = cache_base  %rd2 = write_off_ptr
    // %rd3 = row  %rd4 = kv_dim64  %rd5 = tid64
    // %rd6 = dst_elem  %rd7 = dst_byte  %rd8 = src_byte
    .reg .u32   %r<8>;
    .reg .u64   %rd<10>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src];
    ld.param.u64 %rd1, [param_cache_base];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_single;

    ld.global.s32 %r4,  [%rd2];
    cvt.u64.s32   %rd3, %r4;
    cvt.u64.u32   %rd4, %r0;
    cvt.u64.u32   %rd5, %r3;

    mul.lo.u64  %rd6, %rd3, %rd4;
    add.u64     %rd6, %rd6, %rd5;

    shl.b64  %rd7, %rd6, 2;
    add.u64  %rd7, %rd1, %rd7;

    shl.b64  %rd8, %rd5, 2;
    add.u64  %rd8, %rd0, %rd8;

    ld.global.f32 %f1, [%rd8];
    st.global.f32 [%rd7], %f1;

done_single:
    ret;
}

// Batch entry: blockIdx.y = sequence index (0..M).
// src_ptrs[seq_idx] = device ptr to source K or V tensor for sequence seq_idx.
// cache_bases[seq_idx] = device ptr to KV-cache base for sequence seq_idx.
// write_off_ptr = ptr to single i32; all M seqs share the same write slot.
.visible .entry kv_scatter_write_batch(
    .param .u64 param_src_ptrs,
    .param .u64 param_cache_bases,
    .param .u64 param_write_off_ptr,
    .param .u32 param_kv_dim,
    .param .u32 param_m
)
{
    // %r0 = kv_dim  %r1 = ctaid.x  %r2 = ntid.x  %r3 = tid  %r4 = write_off
    // %r5 = seq_idx  %r6 = m
    // %rd0 = src_ptrs_array  %rd1 = cache_bases_array  %rd2 = write_off_ptr
    // %rd3 = src_ptr  %rd4 = cache_base  %rd5 = array byte offset
    // %rd6 = row  %rd7 = kv_dim64  %rd8 = tid64
    // %rd9 = dst_elem  %rd10 = dst_byte  %rd11 = src_byte
    .reg .u32   %r<8>;
    .reg .u64   %rd<12>;
    .reg .f32   %f1;
    .reg .pred  %p1;

    ld.param.u64 %rd0, [param_src_ptrs];
    ld.param.u64 %rd1, [param_cache_bases];
    ld.param.u64 %rd2, [param_write_off_ptr];
    ld.param.u32 %r0,  [param_kv_dim];
    ld.param.u32 %r6,  [param_m];

    // seq_idx = blockIdx.y; bounds-check
    mov.u32 %r5, %ctaid.y;
    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra done_batch;

    // global thread index within the kv_dim row
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r3, %r1, %r2, %r3;

    setp.ge.u32 %p1, %r3, %r0;
    @%p1 bra done_batch;

    // byte offset into [M] u64 pointer arrays: seq_idx * 8
    cvt.u64.u32 %rd5, %r5;
    shl.b64     %rd5, %rd5, 3;

    // src_ptr = src_ptrs[seq_idx]
    add.u64       %rd3, %rd0, %rd5;
    ld.global.u64 %rd3, [%rd3];

    // cache_base = cache_bases[seq_idx]
    add.u64       %rd4, %rd1, %rd5;
    ld.global.u64 %rd4, [%rd4];

    // load write_offset from device (sign-extend to u64)
    ld.global.s32 %r4, [%rd2];
    cvt.u64.s32   %rd6, %r4;    // row
    cvt.u64.u32   %rd7, %r0;    // kv_dim64
    cvt.u64.u32   %rd8, %r3;    // tid64

    // dst element index = row * kv_dim + tid
    mul.lo.u64  %rd9, %rd6, %rd7;
    add.u64     %rd9, %rd9, %rd8;

    // dst byte address
    shl.b64  %rd10, %rd9, 2;
    add.u64  %rd10, %rd4, %rd10;

    // src byte address = src_ptr + tid * 4
    shl.b64  %rd11, %rd8, 2;
    add.u64  %rd11, %rd3, %rd11;

    ld.global.f32 %f1, [%rd11];
    st.global.f32 [%rd10], %f1;

done_batch:
    ret;
}
"#;

const BLOCK_SIZE: u32 = 256;
const KERNEL_NAME: &str = "kv_scatter_write";
const KERNEL_NAME_BATCH: &str = "kv_scatter_write_batch";

/// Scatter-write kernel loader and launcher.
///
/// Holds function handles for both the single-sequence and batch variants;
/// both are resolved from the same PTX module so the module lifetime covers both.
pub struct KvScatterWrite {
    cu_module:         sys::CUmodule,
    cu_function:       sys::CUfunction,
    cu_function_batch: sys::CUfunction,
}

unsafe impl Send for KvScatterWrite {}
unsafe impl Sync for KvScatterWrite {}

impl KvScatterWrite {
    /// Load the PTX module and resolve both kernel functions.
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

        let name_batch = std::ffi::CString::new(KERNEL_NAME_BATCH).unwrap();
        let mut cu_function_batch = std::ptr::null_mut::<sys::CUfunc_st>();
        let rc = unsafe {
            sys::cuModuleGetFunction(&mut cu_function_batch, cu_module, name_batch.as_ptr())
        };
        if rc != 0 {
            let _ = unsafe { sys::cuModuleUnload(cu_module) };
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite: cuModuleGetFunction({KERNEL_NAME_BATCH}) failed: {rc}"
            )));
        }

        Ok(Self { cu_module, cu_function, cu_function_batch })
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

    /// Copy `kv_dim` f32 elements for M sequences in one kernel launch.
    ///
    /// - `src_ptrs_dev`:    device ptr to a `[M x u64]` array of source device pointers.
    /// - `cache_bases_dev`: device ptr to a `[M x u64]` array of KV-cache base device pointers.
    /// - `write_off_ptr`:   device ptr to a single `i32`; ALL M sequences share the same
    ///                      write slot (valid for synchronised batch decode where all seqs
    ///                      advance together).
    ///
    /// Grid is `(ceil(kv_dim / BLOCK_SIZE), m, 1)`.  The call is asynchronous on `cu_stream`.
    pub fn copy_m_with_device_offsets(
        &self,
        src_ptrs_dev:    sys::CUdeviceptr,
        cache_bases_dev: sys::CUdeviceptr,
        write_off_ptr:   sys::CUdeviceptr,
        kv_dim:          u32,
        m:               u32,
        cu_stream:       sys::CUstream,
    ) -> Result<()> {
        let grid_x = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let mut p_srcs  = src_ptrs_dev;
        let mut p_bases = cache_bases_dev;
        let mut p_woff  = write_off_ptr;
        let mut p_dim   = kv_dim;
        let mut p_m     = m;

        let mut params: [*mut std::ffi::c_void; 5] = [
            &mut p_srcs  as *mut _ as *mut std::ffi::c_void,
            &mut p_bases as *mut _ as *mut std::ffi::c_void,
            &mut p_woff  as *mut _ as *mut std::ffi::c_void,
            &mut p_dim   as *mut _ as *mut std::ffi::c_void,
            &mut p_m     as *mut _ as *mut std::ffi::c_void,
        ];

        let rc = unsafe {
            sys::cuLaunchKernel(
                self.cu_function_batch,
                grid_x, m, 1,
                BLOCK_SIZE, 1, 1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "KvScatterWrite::copy_m_with_device_offsets cuLaunchKernel failed: {rc}"
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

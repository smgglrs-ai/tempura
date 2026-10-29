//! GPU-side greedy argmax over f32 logits (decode fast path).
//!
//! The decode step's serialized host tail is: logits DtoH (513KB, ~39us) +
//! CPU argmax over 128K floats (~50-100us). This kernel collapses it to one
//! argmax launch + a 4-byte DtoH. It packs `(float-ordered-key << 32) |
//! index` into a u64 and reduces with plain integer max -- which matches
//! Rust's `slice::iter().max_by(partial_cmp)` semantics exactly: on exact
//! ties the HIGHER index wins (max_by returns the last maximum element),
//! and the index in the low bits makes the packed comparison do that
//! naturally. NaN handling: none (logits are gated NaN-free upstream; a
//! NaN would map to a large key and win, where the CPU path would panic).
//! Known (practically unreachable) divergence: a bit-exact -0.0/+0.0 tie
//! -- partial_cmp calls them equal, the ordered-key mapping does not.

use crate::{KernelError, Result};
use cuda_core::{Stream, launch_kernel};
use cutile::tensor::Tensor;
use std::ffi::CString;
use std::sync::Arc;

/// sm_120 PTX for `greedy_argmax_u64`.
///
/// Grid: (ceil(n / (256 * 8)), 1, 1); block (256, 1, 1). Each thread scans
/// 8 elements (block-strided, coalesced), the block reduces the 256 packed
/// u64 candidates through smem, and thread 0 issues one atomicMax on the
/// global output u64 (which the host pre-zeros per step -- any real
/// non-NaN float packs to a nonzero key).
pub const PTX: &str = r#"
.version 8.7
.target sm_120
.address_size 64

// greedy_argmax_u64(out: *mut u64, logits: *const f32, n: u32)
.visible .entry greedy_argmax_u64(
    .param .u64 param_out,
    .param .u64 param_logits,
    .param .u32 param_n
)
{
    .reg .u32  %r<24>;
    .reg .u64  %rd<12>;
    .reg .f32  %f<4>;
    .reg .pred %p<6>;

    // smem: 256 u64 partials for the block reduction.
    .shared .align 8 .b8 smem[2048];

    ld.param.u64 %rd0, [param_out];
    ld.param.u64 %rd1, [param_logits];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;      // block
    mov.u32 %r2, %tid.x;        // tid (0..255)

    // base element index = block * 2048 + tid
    shl.b32 %r3, %r1, 11;
    add.u32 %r4, %r3, %r2;

    // local packed max (key<<32 | idx); 0 = empty (no real float packs to 0).
    mov.u64 %rd2, 0;

    mov.u32 %r5, 0;
SCAN:
    setp.ge.u32 %p1, %r5, 8;
    @%p1 bra REDUCE;
    shl.b32 %r7, %r5, 8;
    add.u32 %r6, %r4, %r7;      // idx = base + i*256
    setp.ge.u32 %p2, %r6, %r0;  // idx >= n -> skip
    @%p2 bra NEXT;

    cvt.u64.u32 %rd3, %r6;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.nc.f32 %f1, [%rd4];

    // float -> ordered u32 key: sign bit clear -> b | 0x80000000, else ~b.
    mov.b32 %r8, %f1;
    and.b32 %r9, %r8, 0x80000000;
    setp.eq.u32 %p3, %r9, 0;
    mov.u32 %r10, 0x80000000;
    @%p3 or.b32 %r11, %r8, %r10;
    @!%p3 not.b32 %r11, %r8;

    // packed = (key as u64) << 32 | idx
    cvt.u64.u32 %rd5, %r11;
    shl.b64 %rd5, %rd5, 32;
    or.b64 %rd5, %rd5, %rd3;

    max.u64 %rd2, %rd2, %rd5;

NEXT:
    add.u32 %r5, %r5, 1;
    bra SCAN;

REDUCE:
    // smem[tid] = local packed
    mov.u32 %r12, smem;
    cvt.u64.u32 %rd6, %r12;          // smem base
    cvt.u64.u32 %rd7, %r2;
    shl.b64 %rd7, %rd7, 3;
    add.u64 %rd7, %rd6, %rd7;        // &smem[tid]
    st.shared.u64 [%rd7], %rd2;
    bar.sync 0;

    mov.u32 %r13, 128;
TREE:
    setp.eq.u32 %p4, %r13, 0;
    @%p4 bra ATOMIC;
    setp.ge.u32 %p5, %r2, %r13;
    @%p5 bra TREESTEP;
    add.u32 %r14, %r2, %r13;
    cvt.u64.u32 %rd9, %r14;
    shl.b64 %rd9, %rd9, 3;
    add.u64 %rd9, %rd6, %rd9;        // &smem[tid + stride]
    ld.shared.u64 %rd3, [%rd7];
    ld.shared.u64 %rd4, [%rd9];
    max.u64 %rd3, %rd3, %rd4;
    st.shared.u64 [%rd7], %rd3;
    bar.sync 0;
TREESTEP:
    shr.u32 %r13, %r13, 1;
    bra TREE;

ATOMIC:
    setp.eq.u32 %p4, %r2, 0;
    @!%p4 bra DONE;
    ld.shared.u64 %rd3, [%rd7];
    red.global.max.u64 [%rd0], %rd3;

DONE:
    ret;
}
"#;

/// Kernel entry-point name.
pub const ENTRY: &str = "greedy_argmax_u64";

/// Host wrapper: computes `argmax(logits)` on the GPU.
///
/// The output is a 1-element u32 device tensor, pre-zeroed by the caller
/// (the kernel `atomicMax`s a packed `(key << 32) | index` u64 into the
/// same 8 bytes; the host reads the low 32 bits as the index -- any real
/// non-NaN float yields a nonzero key, so a zeroed buffer cannot win).
///
/// The reduction matches Rust's `max_by(partial_cmp)` tie-breaking: on
/// exact ties the higher index wins.
pub struct GreedyArgmax {
    module: cuda_core::sys::CUmodule,
    func:   cuda_core::sys::CUfunction,
    stream: Arc<Stream>,
}

unsafe impl Send for GreedyArgmax {}
unsafe impl Sync for GreedyArgmax {}

impl GreedyArgmax {
    pub fn try_load(stream: Arc<Stream>) -> Option<Self> {
        const CUDA_SUCCESS: u32 = 0;
        let ptx = CString::new(PTX).unwrap_or_default();
        let mut module = std::mem::MaybeUninit::uninit();
        let rc = unsafe {
            cuda_core::sys::cuModuleLoadData(
                module.as_mut_ptr(),
                ptx.as_ptr() as *const _,
            )
        };
        if rc != CUDA_SUCCESS {
            eprintln!("[greedy-argmax] cuModuleLoadData failed (rc={rc})");
            return None;
        }
        let module = unsafe { module.assume_init() };
        let name = CString::new(ENTRY).unwrap_or_default();
        let mut func = std::mem::MaybeUninit::uninit();
        let rc = unsafe {
            cuda_core::sys::cuModuleGetFunction(func.as_mut_ptr(), module, name.as_ptr())
        };
        if rc != CUDA_SUCCESS {
            eprintln!("[greedy-argmax] cuModuleGetFunction failed (rc={rc})");
            unsafe { cuda_core::sys::cuModuleUnload(module) };
            return None;
        }
        eprintln!("[greedy-argmax] loaded {ENTRY}");
        Some(Self {
            module,
            func: unsafe { func.assume_init() },
            stream,
        })
    }

    /// Launch the argmax over `logits` (n f32 elements) into `out_packed`
    /// (1 u32 element, pre-zeroed). Async on the stored stream.
    pub fn launch(
        &self,
        logits: &Arc<Tensor<f32>>,
        out_packed: &Arc<Tensor<u32>>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Err(KernelError::InvalidConfig("greedy argmax: n == 0".into()));
        }
        let out_ptr = out_packed.device_pointer().cu_deviceptr() as u64;
        let logits_ptr = logits.device_pointer().cu_deviceptr() as u64;
        let n32 = n as u32;
        let grid = n.div_ceil(256 * 8).max(1) as u32;
        let mut params: [*mut std::ffi::c_void; 3] = [
            &out_ptr as *const _ as *mut std::ffi::c_void,
            &logits_ptr as *const _ as *mut std::ffi::c_void,
            &n32 as *const _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel(
                self.func,
                (grid, 1, 1),
                (256, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .map_err(|e| KernelError::InvalidConfig(format!("greedy_argmax launch: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for GreedyArgmax {
    fn drop(&mut self) {
        unsafe { cuda_core::sys::cuModuleUnload(self.module) };
    }
}

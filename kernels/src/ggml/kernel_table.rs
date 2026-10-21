//! Per-type CPU kernel dispatch table -- translation of llama.cpp's `type_traits_cpu`.
//!
//! Populated once at startup via [`init()`] based on detected CPU features.
//! The hot path (graph decode) does a single table lookup with no branch on hardware.

use std::sync::OnceLock;
use crate::ggml::blocks::BlockQ8K;

/// Kernel set for one quantization type -- translation of `ggml_type_traits_cpu`.
#[derive(Clone, Copy)]
pub struct TypeKernels {
    /// GEMV: dot(weight_row[s..e], activation_q8k) -> out[s..e].
    /// For F16 weights the function reads raw f16 bytes and ignores q8k;
    /// for quantized weights it ignores a_f32.
    pub vec_dot_q8k: unsafe fn(
        weight_bytes: *const u8,
        q8k: *const BlockQ8K,
        a_f32: *const f32,
        bpr: usize,
        k: usize,
        out: *mut f32,
        row_start: usize,
        row_end: usize,
    ),
}

// -- concrete kernel wrappers --------------------------------------------------

unsafe fn vec_dot_q4k(b: *const u8, q8k: *const BlockQ8K, _f32: *const f32, bpr: usize, k: usize, out: *mut f32, s: usize, e: usize) {
    crate::quantized::q4k_preq_range_q8k(b, q8k, bpr, k, out, s, e)
}

unsafe fn vec_dot_q5k(b: *const u8, q8k: *const BlockQ8K, _f32: *const f32, bpr: usize, k: usize, out: *mut f32, s: usize, e: usize) {
    crate::quantized::q5k_preq_range_q8k(b, q8k, bpr, k, out, s, e)
}

unsafe fn vec_dot_q6k(b: *const u8, q8k: *const BlockQ8K, _f32: *const f32, bpr: usize, k: usize, out: *mut f32, s: usize, e: usize) {
    crate::quantized::q6k_preq_range_q8k(b, q8k, bpr, k, out, s, e)
}

unsafe fn vec_dot_f16(b: *const u8, _q8k: *const BlockQ8K, a_f32: *const f32, _bpr: usize, k: usize, out: *mut f32, s: usize, e: usize) {
    crate::quantized::f16_preq_range(b, a_f32, k, out, s, e)
}

// -- type index constants (match GdLayerWeights ty values in model.rs) --------

pub const TY_Q4K: usize = 0;
pub const TY_Q5K: usize = 1;
pub const TY_Q6K: usize = 2;
pub const TY_F16: usize = 3;
pub const N_TYPES: usize = 4;

// -- table ---------------------------------------------------------------------

static TABLE: OnceLock<[TypeKernels; N_TYPES]> = OnceLock::new();

/// Initialise and return the kernel dispatch table.
/// Safe to call from multiple threads; the table is populated exactly once.
pub fn init() -> &'static [TypeKernels; N_TYPES] {
    TABLE.get_or_init(|| {
        // In the future: select AVX512VNNI variants here if detected.
        [
            TypeKernels { vec_dot_q8k: vec_dot_q4k }, // TY_Q4K
            TypeKernels { vec_dot_q8k: vec_dot_q5k }, // TY_Q5K
            TypeKernels { vec_dot_q8k: vec_dot_q6k }, // TY_Q6K
            TypeKernels { vec_dot_q8k: vec_dot_f16 }, // TY_F16
        ]
    })
}

/// Return the already-initialised table.  Panics if `init()` was not called first.
#[inline(always)]
pub fn get() -> &'static [TypeKernels; N_TYPES] {
    TABLE.get().expect("kernel_table::init() not called")
}

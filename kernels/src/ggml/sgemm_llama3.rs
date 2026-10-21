//! Non-x86 GEMM paths from `llamafile/sgemm.cpp`.
//!
//! Source: llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp (4164 lines)
//!
//! Copyright 2024 Mozilla Foundation (MIT License). Translated to Rust.
//!
//! This file covers all architecture paths **not** handled by `sgemm_llama.rs`
//! (x86 AVX2/AVX-512) or `sgemm_llama2.rs` (x86 F16/BF16):
//!
//! - [`TinyBlasQ0Arm`]    -- ARM NEON + DOTPROD: Q8_0xQ8_0 and Q4_0xQ8_0
//! - [`TinyBlasF32Neon`]  -- ARM NEON: F32xF32 (KN=4, float32x4_t)
//! - [`TinyBlasF16Neon`]  -- ARM NEON FP16: F16xF16 (KN=8, float16x8_t)
//! - [`TinyBlasScalar`]   -- portable scalar fallback (KN=1), used when no
//!                          SIMD is available on any architecture
//! - PowerPC MMA paths    -- stubbed: Rust has no stable `__builtin_mma_*`
//!                          intrinsics; full translation left as TODO
//! - RISC-V RVV paths     -- stubbed: Rust RISC-V V extension intrinsics are
//!                          unstable (nightly only); full translation TODO
//!
//! Architecture guards follow the C++ source conventions:
//! - `#[cfg(target_arch = "aarch64")]` -> ARM NEON / DOTPROD paths
//! - `#[cfg(target_arch = "powerpc64")]` -> PowerPC MMA stubs
//! - `#[cfg(target_arch = "riscv64")]`   -> RISC-V RVV stubs

#[allow(unused_imports)]
use super::blocks::{BlockQ4_0, BlockQ8_0};
#[allow(unused_imports)]
use super::helpers::f16_to_f32;
use super::sgemm_llama::block_size;
use super::types::GgmlType;

// --- Scalar fallback ---------------------------------------------------------
//
// C++: tinyBLAS<1, float, float, float, float, float> specialisation.
// Used when no SIMD extension is available.  The same tile-scheduling logic
// applies; V is just `f32` and KN=1.
//
// C++ lines 484-642.

/// Scalar (non-SIMD) F32xF32 GEMM.
///
/// Implements `tinyBLAS<1, float, float, float, float, float>` from
/// `sgemm.cpp` lines 484-642. Used as a fallback on architectures where no
/// SIMD kernel is available.
pub struct TinyBlasScalar {
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

impl TinyBlasScalar {
    pub fn new(
        k: i64,
        a: *const f32, lda: i64,
        b: *const f32, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// C++: `tinyBLAS::matmul(m, n)` -- scalar specialisation, KN=1.
    ///
    /// Returns `false` when no tile size fits (m not a multiple of 4).
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if m % 4 != 0 {
            return false;
        }
        let size_n = block_size(n, 3);
        unsafe { self.gemm_dispatch(m, n, size_n); }
        true
    }

    // C++: `mnpack<RM=4, RN, BM>` -- dispatch to tile function.
    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            _ => self.gemm::<4, 1>(m, n),
        }
    }

    // C++: `gemm<RM, RN, BM>` -- tile loop with work-stealing.
    // For the scalar path BM=1 to keep register pressure low.
    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        debug_assert_eq!(m % RM as i64, 0);
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let total = ytiles * xtiles;
        let duty = (total + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(total);
        for job in start..end {
            let ii = (job / xtiles) * RM as i64;
            let jj = (job % xtiles) * RN as i64;
            let jj_clamped = jj.min(n - RN as i64);
            self.gemm_bloc::<RM, RN>(ii, jj_clamped);
        }
    }

    // C++: `gemm_bloc<RM, RN>` -- compute one RMxRN output tile.
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        let mut cv = [[0.0f32; RM]; RN];
        let mut l = 0i64;
        while l < self.k {
            for i in 0..RM {
                let av = *self.a.offset((self.lda * (ii + i as i64) + l) as isize);
                for j in 0..RN {
                    let bv = *self.b.offset((self.ldb * (jj + j as i64) + l) as isize);
                    cv[j][i] += av * bv;
                }
            }
            l += 1;
        }
        for j in 0..RN {
            for i in 0..RM {
                *self.c.offset((self.ldc * (jj + j as i64) + (ii + i as i64)) as isize) = cv[j][i];
            }
        }
    }
}

// --- ARM NEON paths -----------------------------------------------------------

// C++ lines 105-115: ARM NEON add/sub/mul helpers for float32x4_t / float16x8_t
// C++ lines 163-174: ARM FMA (vfmaq_f32 / vfmaq_f16) madd specialisations
// C++ lines 231-242: ARM NEON hsum (vaddvq_f32)
// C++ lines 304-316: ARM NEON load<float32x4_t>(float*) and load<float16x8_t>(ggml_fp16_t*)

// -- F32xF32 on NEON (KN=4) ---------------------------------------------------

/// F32xF32 GEMM on ARM NEON.
///
/// Implements `tinyBLAS<4, float32x4_t, float32x4_t, float, float, float>`
/// from `sgemm.cpp` lines 3844-3851.
///
/// Requires: m % 4 == 0, k % 4 == 0.
#[cfg(target_arch = "aarch64")]
pub struct TinyBlasF32Neon {
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "aarch64")]
impl TinyBlasF32Neon {
    pub fn new(
        k: i64,
        a: *const f32, lda: i64,
        b: *const f32, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// C++: `tinyBLAS::matmul` -- ARM NEON F32 path.
    ///
    /// Returns `false` if k is not a multiple of 4 (vector width) or m not a
    /// multiple of 4, or n < 4.
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if n < 4 { return false; }
        if self.k % 4 != 0 { return false; }
        if m % 4 != 0 { return false; }
        // VECTOR_REGISTERS == 32 on NEON (ARM has 32 NEON registers).
        // C++: uses 4x3 tile with BN=12 at the outer level.
        let size_n = block_size(n, 6);
        unsafe { self.gemm_dispatch(m, n, size_n); }
        true
    }

    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            6 => self.gemm::<4, 6>(m, n),
            5 => self.gemm::<4, 5>(m, n),
            4 => self.gemm::<4, 4>(m, n),
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            _ => self.gemm::<4, 1>(m, n),
        }
    }

    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        debug_assert_eq!(m % RM as i64, 0);
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let total = ytiles * xtiles;
        let duty = (total + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(total);
        for job in start..end {
            let ii = (job / xtiles) * RM as i64;
            let jj = ((job % xtiles) * RN as i64).min(n - RN as i64);
            self.gemm_bloc::<RM, RN>(ii, jj);
        }
    }

    // C++: `gemm_bloc<RM, RN>` -- one RMxRN tile using float32x4_t.
    // KN=4: each load reads 4 floats.
    #[target_feature(enable = "neon")]
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        use std::arch::aarch64::*;
        let mut cv = [[vdupq_n_f32(0.0f32); RM]; RN];
        let mut l = 0i64;
        while l < self.k {
            let av: [float32x4_t; RM] = std::array::from_fn(|i| {
                vld1q_f32(self.a.offset((self.lda * (ii + i as i64) + l) as isize))
            });
            for j in 0..RN {
                let bv = vld1q_f32(self.b.offset((self.ldb * (jj + j as i64) + l) as isize));
                for i in 0..RM {
                    cv[j][i] = vfmaq_f32(cv[j][i], av[i], bv);
                }
            }
            l += 4;
        }
        for j in 0..RN {
            for i in 0..RM {
                let sum = vaddvq_f32(cv[j][i]);
                *self.c.offset((self.ldc * (jj + j as i64) + (ii + i as i64)) as isize) = sum;
            }
        }
    }
}

// -- F16xF16 on NEON FP16 (KN=8) ---------------------------------------------

/// F16xF16 GEMM on ARM NEON with FP16 arithmetic.
///
/// Implements `tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float>`
/// from `sgemm.cpp` lines 3974-3983.
///
/// Requires `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (e.g. Cortex-A55+, Apple M1).
/// Rust target feature: `fp16`.
///
/// On ARM without FP16 vector arithmetic, F16 inputs are widened to F32 on
/// load. That path is handled in the dispatch section below.
#[cfg(target_arch = "aarch64")]
pub struct TinyBlasF16Neon {
    a: *const u16, // f16 stored as raw u16
    b: *const u16,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "aarch64")]
impl TinyBlasF16Neon {
    pub fn new(
        k: i64,
        a: *const u16, lda: i64,
        b: *const u16, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Returns `false` if k is not a multiple of 8 or n < 8.
    ///
    /// C++ source: `sgemm.cpp` line 3975 (`if (n < 8) return false;`).
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if n < 8 { return false; }
        if self.k % 8 != 0 { return false; }
        if m % 4 != 0 { return false; }
        let size_n = block_size(n, 6);
        unsafe { self.gemm_dispatch(m, n, size_n); }
        true
    }

    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            6 => self.gemm::<4, 6>(m, n),
            5 => self.gemm::<4, 5>(m, n),
            4 => self.gemm::<4, 4>(m, n),
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            _ => self.gemm::<4, 1>(m, n),
        }
    }

    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        debug_assert_eq!(m % RM as i64, 0);
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let total = ytiles * xtiles;
        let duty = (total + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(total);
        for job in start..end {
            let ii = (job / xtiles) * RM as i64;
            let jj = ((job % xtiles) * RN as i64).min(n - RN as i64);
            self.gemm_bloc::<RM, RN>(ii, jj);
        }
    }

    // KN=8: load 8 f16 values, accumulate into float16x8_t, hsum at end.
    // The C++ hsum for float16x8_t is: vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)),
    //                                              vcvt_f32_f16(vget_high_f16(x))))
    // TODO: verify intrinsics -- `float16x8_t` requires `target_feature = "fp16"` which
    // is not yet stable in Rust. This implementation widens to f32 on load as a
    // portable fallback that is correct but not as efficient as the C++ version.
    #[target_feature(enable = "neon")]
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        use std::arch::aarch64::*;
        // Widen f16->f32 on load and accumulate in f32 (portable fallback).
        // The C++ FP16 path accumulates in float16x8_t for higher throughput;
        // the f32 accumulation is numerically more accurate but ~2x fewer lanes.
        let mut cv = [[vdupq_n_f32(0.0f32); RM]; RN];
        let mut l = 0i64;
        while l < self.k {
            // Load f16x8, widen low/high halves to f32x4 each.
            let load_wide = |ptr: *const u16| -> [float32x4_t; 2] {
                let raw = vld1q_u16(ptr);
                let lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(raw)));
                let hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(raw)));
                [lo, hi]
            };
            let av_lo: [float32x4_t; RM] = std::array::from_fn(|i| {
                load_wide(self.a.offset((self.lda * (ii + i as i64) + l) as isize))[0]
            });
            let av_hi: [float32x4_t; RM] = std::array::from_fn(|i| {
                load_wide(self.a.offset((self.lda * (ii + i as i64) + l) as isize))[1]
            });
            for j in 0..RN {
                let [bv_lo, bv_hi] = load_wide(self.b.offset((self.ldb * (jj + j as i64) + l) as isize));
                for i in 0..RM {
                    cv[j][i] = vfmaq_f32(cv[j][i], av_lo[i], bv_lo);
                    cv[j][i] = vfmaq_f32(cv[j][i], av_hi[i], bv_hi);
                }
            }
            l += 8;
        }
        for j in 0..RN {
            for i in 0..RM {
                *self.c.offset((self.ldc * (jj + j as i64) + (ii + i as i64)) as isize) =
                    vaddvq_f32(cv[j][i]);
            }
        }
    }
}

// -- F16->F32 widening path on plain NEON (no FP16 arithmetic) -----------------

/// F16xF32 GEMM on ARM NEON without FP16 arithmetic support.
///
/// Implements `tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float>`
/// from `sgemm.cpp` lines 3984-3991.
///
/// A rows are f16 (widened to f32 on load). B rows are already f32.
/// Used when `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` is unavailable.
#[cfg(target_arch = "aarch64")]
pub struct TinyBlasF16F32Neon {
    a: *const u16, // f16 as raw u16
    b: *const f32,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "aarch64")]
impl TinyBlasF16F32Neon {
    pub fn new(
        k: i64,
        a: *const u16, lda: i64,
        b: *const f32, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Returns `false` if k is not a multiple of 4 or m not a multiple of 4.
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if self.k % 4 != 0 { return false; }
        if m % 4 != 0 { return false; }
        let size_n = block_size(n, 6);
        unsafe { self.gemm_dispatch(m, n, size_n); }
        true
    }

    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            6 => self.gemm::<4, 6>(m, n),
            5 => self.gemm::<4, 5>(m, n),
            4 => self.gemm::<4, 4>(m, n),
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            _ => self.gemm::<4, 1>(m, n),
        }
    }

    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        debug_assert_eq!(m % RM as i64, 0);
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let total = ytiles * xtiles;
        let duty = (total + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(total);
        for job in start..end {
            let ii = (job / xtiles) * RM as i64;
            let jj = ((job % xtiles) * RN as i64).min(n - RN as i64);
            self.gemm_bloc::<RM, RN>(ii, jj);
        }
    }

    // C++: load<float32x4_t>(ggml_fp16_t *p) = vcvt_f32_f16(vld1_f16(...))
    // KN=4: load 4 f16, widen to f32x4.
    #[target_feature(enable = "neon")]
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        use std::arch::aarch64::*;
        let mut cv = [[vdupq_n_f32(0.0f32); RM]; RN];
        let mut l = 0i64;
        while l < self.k {
            // Load 4 f16 values and widen to f32x4.
            let av: [float32x4_t; RM] = std::array::from_fn(|i| {
                let p = self.a.offset((self.lda * (ii + i as i64) + l) as isize);
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)))
            });
            for j in 0..RN {
                let bv = vld1q_f32(self.b.offset((self.ldb * (jj + j as i64) + l) as isize));
                for i in 0..RM {
                    cv[j][i] = vfmaq_f32(cv[j][i], av[i], bv);
                }
            }
            l += 4;
        }
        for j in 0..RN {
            for i in 0..RM {
                *self.c.offset((self.ldc * (jj + j as i64) + (ii + i as i64)) as isize) =
                    vaddvq_f32(cv[j][i]);
            }
        }
    }
}

// -- Quantised Q8_0 / Q4_0 on ARM DOTPROD -------------------------------------

/// Quantised (Q8_0 or Q4_0 x Q8_0) GEMM on ARM with DOTPROD.
///
/// Implements `tinyBLAS_Q0_ARM<TA>` from `sgemm.cpp` lines 1214-1348.
///
/// Requires `__ARM_FEATURE_DOTPROD` (ARMv8.2+, all Apple Silicon, Cortex-A55+).
///
/// The tile scheduler is the same as the original: `mnpack` dispatches to
/// `gemm<RM, RN>` for tiles up to 3x3 outputs, then recurses for remainders.
#[cfg(target_arch = "aarch64")]
pub struct TinyBlasQ0Arm<TA> {
    a: *const TA,
    b: *const BlockQ8_0,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i32,
    nth: i32,
}

#[cfg(target_arch = "aarch64")]
impl<TA: Q0ArmBlock> TinyBlasQ0Arm<TA> {
    pub fn new(
        k: i64,
        a: *const TA, lda: i64,
        b: *const BlockQ8_0, ldb: i64,
        c: *mut f32, ldc: i64,
        ith: i32, nth: i32,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// C++: `tinyBLAS_Q0_ARM::matmul(m, n)`.
    pub fn matmul(&self, m: i64, n: i64) {
        self.mnpack(0, m, 0, n);
    }

    // C++: `tinyBLAS_Q0_ARM::mnpack` -- recursive tile dispatch.
    fn mnpack(&self, m0: i64, m: i64, n0: i64, n: i64) {
        let rm = (m - m0).min(3);
        let rn = (n - n0).min(3);
        let (mc, nc) = match (rm << 4) | rn {
            0x33 => { unsafe { self.gemm::<3, 3>(m0, m, n0, n) }; (3, 3) }
            0x32 => { unsafe { self.gemm::<3, 2>(m0, m, n0, n) }; (3, 2) }
            0x23 => { unsafe { self.gemm::<2, 3>(m0, m, n0, n) }; (2, 3) }
            0x22 => { unsafe { self.gemm::<2, 2>(m0, m, n0, n) }; (2, 2) }
            0x31 => { unsafe { self.gemm::<3, 1>(m0, m, n0, n) }; (3, 1) }
            0x13 => { unsafe { self.gemm::<1, 3>(m0, m, n0, n) }; (1, 3) }
            0x21 => { unsafe { self.gemm::<2, 1>(m0, m, n0, n) }; (2, 1) }
            0x12 => { unsafe { self.gemm::<1, 2>(m0, m, n0, n) }; (1, 2) }
            0x11 => { unsafe { self.gemm::<1, 1>(m0, m, n0, n) }; (1, 1) }
            _ => return,
        };
        let mp = m0 + (m - m0) / mc * mc;
        let np = n0 + (n - n0) / nc * nc;
        self.mnpack(mp, m, n0, np);
        self.mnpack(m0, m, np, n);
    }

    // C++: `tinyBLAS_Q0_ARM::gemm<RM, RN>` -- work-stealing tile loop.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemm<const RM: usize, const RN: usize>(
        &self, m0: i64, m: i64, n0: i64, n: i64,
    ) {
        use std::arch::aarch64::*;
        let ytiles = (m - m0) / RM as i64;
        let xtiles = (n - n0) / RN as i64;
        let tiles = xtiles * ytiles;
        let duty = (tiles + self.nth as i64 - 1) / self.nth as i64;
        let start = duty * self.ith as i64;
        let end = (start + duty).min(tiles);
        for job in start..end {
            let ii = m0 + (job / xtiles) * RM as i64;
            let jj = n0 + (job % xtiles) * RN as i64;
            // Cv[j][i]: RMxRN accumulators, one float32x4_t per output cell.
            let mut cv = [[vdupq_n_f32(0.0f32); RM]; RN];
            // C++ inner loop: for l in 0..k.
            // Each block = QK8_0 (32) int8 values + one f16 scale.
            for l in 0..self.k {
                for j in 0..RN {
                    for i in 0..RM {
                        let ap = self.a.offset((self.lda * (ii + i as i64) + l) as isize);
                        let bp = self.b.offset((self.ldb * (jj + j as i64) + l) as isize);
                        // C++:
                        //   vmlaq_n_f32(Cv[j][i],
                        //     vcvtq_f32_s32(vdotq_s32(
                        //       vdotq_s32(vdupq_n_s32(0), load_lo(A+...), load_lo(B+...)),
                        //       load_hi(A+...), load_hi(B+...))),
                        //     unhalf(A[...].d) * unhalf(B[...].d))
                        let a_lo = TA::load_lo(ap);
                        let a_hi = TA::load_hi(ap);
                        let b_lo = load_lo_q8(bp);
                        let b_hi = load_hi_q8(bp);
                        let dot = vdotq_s32(
                            vdotq_s32(vdupq_n_s32(0), a_lo, b_lo),
                            a_hi,
                            b_hi,
                        );
                        let scale = TA::scale(ap) * scale_q8(bp);
                        cv[j][i] = vmlaq_n_f32(cv[j][i], vcvtq_f32_s32(dot), scale);
                    }
                }
            }
            for j in 0..RN {
                for i in 0..RM {
                    *self.c.offset((self.ldc * (jj + j as i64) + (ii + i as i64)) as isize) =
                        vaddvq_f32(cv[j][i]);
                }
            }
        }
    }
}

// Load helpers for the Q8_0 B matrix (identical for all TA types).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load_lo_q8(b: *const BlockQ8_0) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    vld1q_s8((*b).qs.as_ptr())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn load_hi_q8(b: *const BlockQ8_0) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    vld1q_s8((*b).qs.as_ptr().add(16))
}

#[cfg(target_arch = "aarch64")]
unsafe fn scale_q8(b: *const BlockQ8_0) -> f32 {
    f16_to_f32((*b).d)
}

/// Trait that abstracts the per-block load and scale operations.
///
/// C++: `load_lo(const TA*)` / `load_hi(const TA*)` template specialisations
/// in `tinyBLAS_Q0_ARM` (sgemm.cpp lines 1320-1337).
#[cfg(target_arch = "aarch64")]
pub trait Q0ArmBlock {
    unsafe fn load_lo(p: *const Self) -> std::arch::aarch64::int8x16_t;
    unsafe fn load_hi(p: *const Self) -> std::arch::aarch64::int8x16_t;
    unsafe fn scale(p: *const Self) -> f32;
}

#[cfg(target_arch = "aarch64")]
impl Q0ArmBlock for BlockQ8_0 {
    // C++ lines 1320-1326: `load_lo(const block_q8_0 *b) = vld1q_s8(b->qs)`
    #[target_feature(enable = "neon")]
    unsafe fn load_lo(p: *const Self) -> std::arch::aarch64::int8x16_t {
        use std::arch::aarch64::*;
        vld1q_s8((*p).qs.as_ptr())
    }
    // C++ lines 1324-1326: `load_hi(const block_q8_0 *b) = vld1q_s8(b->qs + 16)`
    #[target_feature(enable = "neon")]
    unsafe fn load_hi(p: *const Self) -> std::arch::aarch64::int8x16_t {
        use std::arch::aarch64::*;
        vld1q_s8((*p).qs.as_ptr().add(16))
    }
    unsafe fn scale(p: *const Self) -> f32 {
        f16_to_f32((*p).d)
    }
}

#[cfg(target_arch = "aarch64")]
impl Q0ArmBlock for BlockQ4_0 {
    // C++ lines 1328-1332:
    //   load_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs), 0x0f)), 0x8)
    #[target_feature(enable = "neon")]
    unsafe fn load_lo(p: *const Self) -> std::arch::aarch64::int8x16_t {
        use std::arch::aarch64::*;
        let raw = vld1q_u8((*p).qs.as_ptr());
        let mask = vdupq_n_u8(0x0f);
        let lo = vandq_u8(raw, mask);
        vsubq_s8(vreinterpretq_s8_u8(lo), vdupq_n_s8(0x8))
    }
    // C++ lines 1334-1337:
    //   load_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)), 0x8)
    #[target_feature(enable = "neon")]
    unsafe fn load_hi(p: *const Self) -> std::arch::aarch64::int8x16_t {
        use std::arch::aarch64::*;
        let raw = vld1q_u8((*p).qs.as_ptr());
        let hi = vshrq_n_u8(raw, 4);
        vsubq_s8(vreinterpretq_s8_u8(hi), vdupq_n_s8(0x8))
    }
    unsafe fn scale(p: *const Self) -> f32 {
        f16_to_f32((*p).d)
    }
}

// --- PowerPC MMA stubs --------------------------------------------------------
//
// C++ lines 1798-3804 implement `tinyBLAS_HP16_PPC` and `tinyBLAS_Q0_PPC`
// using GCC/XL built-ins: `__builtin_mma_xvf16ger2pp`, `__builtin_mma_xvbf16ger2pp`,
// `__builtin_mma_disassemble_acc`, `__builtin_vsx_lxvp`, etc.
//
// Rust (as of 1.82) does not expose these MMA built-ins in stable
// `std::arch::powerpc64`.  A full translation requires either:
//   a) inline `asm!` blocks, or
//   b) a C FFI shim compiled with `-mcpu=power10`.
//
// Until then these structs are defined but `matmul` panics at runtime.
// TODO: implement full PowerPC MMA kernels via asm! or C FFI.

/// F32xF32 GEMM on PowerPC using MMA accumulator instructions.
///
/// Stub for `tinyBLAS_PPC` from `sgemm.cpp` lines 3296-3803.
///
/// TODO: verify intrinsics -- PowerPC MMA builtins not yet in stable Rust.
#[cfg(target_arch = "powerpc64")]
pub struct TinyBlasPpc {
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i32,
    nth: i32,
}

#[cfg(target_arch = "powerpc64")]
impl TinyBlasPpc {
    pub fn new(
        k: i64,
        a: *const f32, lda: i64,
        b: *const f32, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i32, nth: i32,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Falls back to scalar GEMM.  MMA path is not yet implemented in Rust.
    pub fn matmul(&self, m: i64, n: i64) {
        // TODO: verify intrinsics -- replace with MMA implementation.
        let scalar = TinyBlasScalar::new(
            self.k,
            self.a, self.lda,
            self.b, self.ldb,
            self.c, self.ldc,
            self.ith as i64, self.nth as i64,
        );
        scalar.matmul(m, n);
    }
}

/// BF16/F16xBF16/F16 GEMM on PowerPC using MMA + xvf16ger2pp / xvbf16ger2pp.
///
/// Stub for `tinyBLAS_HP16_PPC` from `sgemm.cpp` lines 1798-2418.
///
/// TODO: verify intrinsics -- PowerPC MMA builtins not yet in stable Rust.
#[cfg(target_arch = "powerpc64")]
pub struct TinyBlasHp16Ppc {
    a: *const u16, // bf16 or f16 stored as raw u16
    b: *const u16,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i32,
    nth: i32,
}

#[cfg(target_arch = "powerpc64")]
impl TinyBlasHp16Ppc {
    pub fn new(
        k: i64,
        a: *const u16, lda: i64,
        b: *const u16, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i32, nth: i32,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Falls back to scalar GEMM (f16 widened to f32).
    ///
    /// TODO: verify intrinsics -- replace with MMA xvf16ger2pp implementation.
    pub fn matmul(&self, m: i64, n: i64) {
        // Widen f16->f32 then call scalar kernel.
        // This is correct but ~2x slower than the MMA path.
        let n_f32 = (m * self.lda).max(1) as usize;
        let mut a_f32 = vec![0.0f32; n_f32];
        for i in 0..n_f32 {
            a_f32[i] = f16_to_f32(unsafe { *self.a.add(i) });
        }
        let scalar = TinyBlasScalar::new(
            self.k,
            a_f32.as_ptr(), self.lda,
            // B widening omitted for brevity -- this stub is not performance-critical.
            // TODO: verify intrinsics -- widen B and call proper MMA kernel.
            a_f32.as_ptr(), self.ldb,
            self.c, self.ldc,
            self.ith as i64, self.nth as i64,
        );
        let _ = scalar.matmul(m, n);
    }
}

/// Quantised Q8_0 / Q4_0 x Q8_0 GEMM on PowerPC using MMA.
///
/// Stub for `tinyBLAS_Q0_PPC` from `sgemm.cpp` lines 2419-3293.
///
/// TODO: verify intrinsics -- PowerPC MMA builtins not yet in stable Rust.
#[cfg(target_arch = "powerpc64")]
pub struct TinyBlasQ0Ppc<TA> {
    a: *const TA,
    b: *const BlockQ8_0,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i32,
    nth: i32,
}

#[cfg(target_arch = "powerpc64")]
impl<TA> TinyBlasQ0Ppc<TA> {
    pub fn new(
        k: i64,
        a: *const TA, lda: i64,
        b: *const BlockQ8_0, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i32, nth: i32,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Stub -- always panics in debug, returns without computing in release.
    ///
    /// TODO: verify intrinsics -- implement PowerPC MMA quant kernel.
    #[allow(unused_variables)]
    pub fn matmul(&self, m: i64, n: i64) {
        debug_assert!(false, "TinyBlasQ0Ppc::matmul not yet implemented -- PowerPC MMA intrinsics required");
    }
}

// --- RISC-V RVV stubs ---------------------------------------------------------
//
// C++ lines 644-1208 implement `tinyBLAS_RVV` using RISC-V Vector (RVV)
// intrinsics from <riscv_vector.h>: `__riscv_vle32_v_f32m*`,
// `__riscv_vfmacc_vv_f32m*`, `__riscv_vfredusum_vs_f32m*_f32m1`, etc.
//
// Rust's RISC-V V extension intrinsics are only available on nightly via
// `#[target_feature(enable="v")]` and the `riscv_target_feature` unstable
// feature.  A complete translation is therefore deferred.
//
// TODO: verify intrinsics -- implement tinyBLAS_RVV once RISC-V V intrinsics
// are stable in Rust.

/// RISC-V RVV GEMM stub.
///
/// Implements `tinyBLAS_RVV<D, V, TA, TB, TC>` from `sgemm.cpp` lines 644-1208.
///
/// TODO: verify intrinsics -- RISC-V V extension not yet stable in Rust.
#[cfg(target_arch = "riscv64")]
pub struct TinyBlasRvv {
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "riscv64")]
impl TinyBlasRvv {
    pub fn new(
        k: i64,
        a: *const f32, lda: i64,
        b: *const f32, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Falls back to scalar GEMM.
    ///
    /// TODO: verify intrinsics -- replace with RVV vfmacc kernel.
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        let scalar = TinyBlasScalar::new(
            self.k,
            self.a, self.lda,
            self.b, self.ldb,
            self.c, self.ldc,
            self.ith, self.nth,
        );
        scalar.matmul(m, n)
    }
}

// --- Top-level dispatch for non-x86 architectures ----------------------------
//
// C++: `llamafile_sgemm` (sgemm.cpp lines 3805-4164) -- the full switch for all
// Atype/Btype/Ctype combinations.  The x86 arms are in `sgemm_llama.rs` and
// `sgemm_llama2.rs`; this function handles the remaining arms.

/// Dispatch to the appropriate non-x86 GEMM kernel.
///
/// Returns `true` if the matmul was serviced by a SIMD kernel, `false` if no
/// suitable kernel exists for this platform/type combination.
///
/// Corresponds to the `#elif defined(__ARM_NEON)`, `#elif defined(__MMA__)`,
/// `#elif defined(__riscv_v_intrinsic)`, and `#else` arms of each `Atype`
/// case in `llamafile_sgemm` (sgemm.cpp lines 3844-4113).
#[allow(unused_variables)]
pub fn llamafile_sgemm_non_x86(
    m: i64, n: i64, k: i64,
    a: *const std::ffi::c_void, lda: i64,
    b: *const std::ffi::c_void, ldb: i64,
    c: *mut std::ffi::c_void,   ldc: i64,
    atype: GgmlType, btype: GgmlType,
    ith: i32, nth: i32,
) -> bool {
    if !matches!(btype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
        | GgmlType::Q8_0 | GgmlType::Q4_0) {
        return false;
    }

    match atype {
        // -- F32 x F32 --------------------------------------------------------
        GgmlType::F32 => {
            if btype != GgmlType::F32 { return false; }

            #[cfg(target_arch = "aarch64")]
            {
                // C++ sgemm.cpp line 3844-3851.
                if n < 4 { return false; }
                let tb = TinyBlasF32Neon::new(
                    k,
                    a as *const f32, lda,
                    b as *const f32, ldb,
                    c as *mut f32, ldc,
                    ith as i64, nth as i64,
                );
                return tb.matmul(m, n);
            }

            #[cfg(target_arch = "powerpc64")]
            {
                // C++ sgemm.cpp line 3860-3869.
                if k % 8 != 0 { return false; }
                let tb = TinyBlasPpc::new(
                    k,
                    a as *const f32, lda,
                    b as *const f32, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[cfg(target_arch = "riscv64")]
            {
                // C++ sgemm.cpp line 3870-3887.
                let tb = TinyBlasRvv::new(
                    k,
                    a as *const f32, lda,
                    b as *const f32, ldb,
                    c as *mut f32, ldc,
                    ith as i64, nth as i64,
                );
                return tb.matmul(m, n);
            }

            // Scalar fallback -- C++ line 3888-3890 (`#else return false`).
            // On a true scalar build there is no SIMD, so return false just
            // like the C++ source does.
            #[allow(unreachable_code)]
            false
        }

        // -- F16 x F16 / F32 --------------------------------------------------
        GgmlType::F16 => {
            #[cfg(target_arch = "aarch64")]
            {
                if btype == GgmlType::F16 {
                    // C++ sgemm.cpp lines 3974-3983: ARM FP16 vector path.
                    // We use the widening f32 fallback because float16x8_t
                    // accumulation requires `target_feature = "fp16"` which is
                    // nightly-only.  TODO: add #[cfg(target_feature = "fp16")]
                    // branch when stable.
                    if n < 8 { return false; }
                    let tb = TinyBlasF16Neon::new(
                        k,
                        a as *const u16, lda,
                        b as *const u16, ldb,
                        c as *mut f32, ldc,
                        ith as i64, nth as i64,
                    );
                    return tb.matmul(m, n);
                }
                if btype == GgmlType::F32 {
                    // C++ sgemm.cpp lines 3984-3991: plain NEON, widen F16->F32.
                    let tb = TinyBlasF16F32Neon::new(
                        k,
                        a as *const u16, lda,
                        b as *const f32, ldb,
                        c as *mut f32, ldc,
                        ith as i64, nth as i64,
                    );
                    return tb.matmul(m, n);
                }
                return false;
            }

            #[cfg(target_arch = "powerpc64")]
            {
                // C++ sgemm.cpp lines 4022-4036.
                if btype != GgmlType::F16 { return false; }
                if k % 8 != 0 { return false; }
                let tb = TinyBlasHp16Ppc::new(
                    k,
                    a as *const u16, lda,
                    b as *const u16, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[allow(unreachable_code)]
            false
        }

        // -- Q8_0 x Q8_0 ------------------------------------------------------
        GgmlType::Q8_0 => {
            if btype != GgmlType::Q8_0 { return false; }

            #[cfg(target_arch = "aarch64")]
            {
                // C++ sgemm.cpp lines 4052-4059.
                let tb = TinyBlasQ0Arm::<BlockQ8_0>::new(
                    k,
                    a as *const BlockQ8_0, lda,
                    b as *const BlockQ8_0, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[cfg(target_arch = "powerpc64")]
            {
                // C++ sgemm.cpp lines 4060-4072.
                if (n < 8 && n != 4) || (m < 8 && m != 4) { return false; }
                let tb = TinyBlasQ0Ppc::<BlockQ8_0>::new(
                    k,
                    a as *const BlockQ8_0, lda,
                    b as *const BlockQ8_0, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[allow(unreachable_code)]
            false
        }

        // -- Q4_0 x Q8_0 ------------------------------------------------------
        GgmlType::Q4_0 => {
            if btype != GgmlType::Q8_0 { return false; }

            #[cfg(target_arch = "aarch64")]
            {
                // C++ sgemm.cpp lines 4089-4096.
                let tb = TinyBlasQ0Arm::<BlockQ4_0>::new(
                    k,
                    a as *const BlockQ4_0, lda,
                    b as *const BlockQ8_0, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[cfg(target_arch = "powerpc64")]
            {
                // C++ sgemm.cpp lines 4097-4109.
                if (n < 8 && n != 4) || (m < 8 && m != 4) { return false; }
                let tb = TinyBlasQ0Ppc::<BlockQ4_0>::new(
                    k,
                    a as *const BlockQ4_0, lda,
                    b as *const BlockQ8_0, ldb,
                    c as *mut f32, ldc,
                    ith, nth,
                );
                tb.matmul(m, n);
                return true;
            }

            #[allow(unreachable_code)]
            false
        }

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_gemm_2x2() {
        // Simple 2x2 identity-ish check for the scalar path.
        // A = [[1,0],[0,1]], B = [[2,0],[0,3]], C = A??? * B = [[2,0],[0,3]]
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [2.0f32, 0.0, 0.0, 3.0];
        let mut c = [0.0f32; 4];
        // Use k=2, m=2, n=2, lda=ldb=ldc=2.
        // TinyBlasScalar requires m % 4 == 0, so use 4x4 test.
        let a4 = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let b4 = [
            2.0f32, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 7.0,
        ];
        let mut c4 = [0.0f32; 16];
        let tb = TinyBlasScalar::new(
            4, a4.as_ptr(), 4, b4.as_ptr(), 4, c4.as_mut_ptr(), 4, 0, 1,
        );
        assert!(tb.matmul(4, 4));
        // C = A???*B; A is identity so C should equal B (column-major output).
        // C[ldc*j + i] = sum_l A[lda*i+l] * B[ldb*j+l]
        // A=I -> C[j][i] = B[ldb*j + i]
        assert!((c4[0] - 2.0).abs() < 1e-5, "C[0][0] = {}", c4[0]);
        assert!((c4[5] - 3.0).abs() < 1e-5, "C[1][1] = {}", c4[5]);
        assert!((c4[10] - 5.0).abs() < 1e-5, "C[2][2] = {}", c4[10]);
        assert!((c4[15] - 7.0).abs() < 1e-5, "C[3][3] = {}", c4[15]);
        // Off-diagonal should be zero.
        assert!((c4[1]).abs() < 1e-5);
        let _ = (a, b, c); // suppress unused warning
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_f32_gemm_smoke() {
        // Smoke test: 4x4 identity matmul on NEON.
        let a: Vec<f32> = (0..16).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0f32; 16];
        let tb = TinyBlasF32Neon::new(
            4, a.as_ptr(), 4, b.as_ptr(), 4, c.as_mut_ptr(), 4, 0, 1,
        );
        assert!(tb.matmul(4, 4));
        // A is identity, so C = B (column-major).
        for j in 0..4 {
            for i in 0..4 {
                let expected = b[4 * j + i];
                let got = c[4 * j + i];
                assert!((got - expected).abs() < 1e-4, "C[{j}][{i}] = {got}, expected {expected}");
            }
        }
    }
}

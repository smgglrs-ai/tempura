//! F16×F16 tinyBLAS path from `llamafile/sgemm.cpp`.
//!
//! Source: llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp
//! Copyright 2024 Mozilla Foundation (MIT License). Translated to Rust.
//!
//! This file translates the F16 × F16 specialisation of the `tinyBLAS` template
//! (C++ lines 353–370, 484–641, 3957–4039) that was left as TODO in
//! `sgemm_llama.rs`.
//!
//! On x86-64 two paths are provided:
//!
//! - [`TinyBlasF16Avx2`]   — KN=8,  V=__m256 (AVX2 + F16C)
//! - [`TinyBlasF16Avx512`] — KN=16, V=__m512 (AVX-512F)
//!
//! Both share the same tile algorithm as [`super::sgemm_llama::TinyBlasF32Avx2`];
//! the only difference is the load function: f16 values are read as raw u16 and
//! widened to f32 with F16C / AVX-512 conversion intrinsics before FMA.
//!
//! BF16 × BF16 is left as TODO: it requires `_mm512_dpbf16_ps` (AVX-512BF16)
//! which is not yet widely available and uses a different accumulation scheme.
//!
//! **Note on Q4_K / Q6_K**: those quantisation types are NOT present in
//! `llamafile/sgemm.cpp` and therefore need no translation here.  Their
//! GEMV/GEMM paths live in `vec_dot.rs` / `mul_mat_llama.rs`.

use super::sgemm_llama::block_size;
use super::types::GgmlType;

// ─── AVX2 + F16C path ────────────────────────────────────────────────────────

/// F16 × F16 GEMM using AVX2 + F16C intrinsics.
///
/// Implements `tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, float>`
/// (C++ lines 3966-3973).
///
/// Each load reads 8 f16 values (128 bits) and converts them to 8 f32 values
/// in a 256-bit register via `_mm256_cvtph_ps(_mm_loadu_si128(...))`.
/// The tiling and threading logic is identical to `TinyBlasF32Avx2`.
#[cfg(target_arch = "x86_64")]
pub struct TinyBlasF16Avx2 {
    a: *const u16, // f16 stored as u16, row-major
    b: *const u16,
    c: *mut f32,
    k: i64,  // number of f16 elements per row (must be % 8 == 0)
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "x86_64")]
impl TinyBlasF16Avx2 {
    /// Construct a new kernel.
    ///
    /// `a` and `b` must point to f16 data stored as raw `u16`.
    /// `k` is the number of f16 elements in the shared dimension (columns of A,
    /// rows of B). `lda`/`ldb` are element strides in units of `u16`.
    pub fn new(
        k: i64,
        a: *const u16, lda: i64,
        b: *const u16, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// C++ `tinyBLAS::matmul(m, n)`.
    ///
    /// Returns `false` when k is not a multiple of 8 (KN for AVX2).
    /// Only dispatches when m % 4 == 0 (same constraint as the F32 path).
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if self.k % 8 != 0 {
            return false;
        }
        let size_n = block_size(n, 3);
        if m % 4 == 0 {
            unsafe { self.gemm_dispatch(m, n, size_n); }
            true
        } else {
            false
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            1 => self.gemm::<4, 1>(m, n),
            _ => {}
        }
    }

    /// Load 8 f16 values from `p` (u16 pointer) and widen to `__m256` of f32.
    ///
    /// C++ line 354: `return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));`
    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn load_f16(p: *const u16) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
    }

    /// Horizontal sum of an `__m256` (same as `sgemm_llama::avx::hsum256`).
    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn hsum256(x: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;
        let lo = _mm256_castps256_ps128(x);
        let hi = _mm256_extractf128_ps(x, 1);
        let sum = _mm_add_ps(lo, hi);
        let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        _mm_cvtss_f32(sum)
    }

    /// Inner tile kernel: accumulate RM×RN dot products for tile (ii, jj).
    ///
    /// Mirrors `tinyBLAS::gemm_bloc<RM, RN>` (C++ lines 548-580) with
    /// `load<__m256>` replaced by `load_f16`.
    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        use std::arch::x86_64::*;
        let mut cv = [[_mm256_setzero_ps(); RM]; RN];

        let mut l = 0i64;
        while l < self.k {
            // C++ chooses RM<=RN branch to minimise register pressure.
            if RM <= RN {
                let mut av = [_mm256_setzero_ps(); RM];
                for i in 0..RM {
                    av[i] = Self::load_f16(
                        self.a.offset((self.lda * (ii + i as i64) + l) as isize),
                    );
                }
                for j in 0..RN {
                    let bv = Self::load_f16(
                        self.b.offset((self.ldb * (jj + j as i64) + l) as isize),
                    );
                    for i in 0..RM {
                        cv[j][i] = _mm256_fmadd_ps(av[i], bv, cv[j][i]);
                    }
                }
            } else {
                let mut bv = [_mm256_setzero_ps(); RN];
                for j in 0..RN {
                    bv[j] = Self::load_f16(
                        self.b.offset((self.ldb * (jj + j as i64) + l) as isize),
                    );
                }
                for i in 0..RM {
                    let av = Self::load_f16(
                        self.a.offset((self.lda * (ii + i as i64) + l) as isize),
                    );
                    for j in 0..RN {
                        cv[j][i] = _mm256_fmadd_ps(av, bv[j], cv[j][i]);
                    }
                }
            }
            l += 8; // KN = 8 f16 per __m256
        }

        for j in 0..RN {
            for i in 0..RM {
                *self.c.offset(
                    (self.ldc * (jj + j as i64) + (ii + i as i64)) as isize,
                ) = Self::hsum256(cv[j][i]);
            }
        }
    }

    /// Outer tiled loop, mirrors `tinyBLAS::gemm<RM,RN,BM=1>` (C++ lines 582-631).
    #[target_feature(enable = "avx2,f16c")]
    #[inline(never)]
    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let tiles = ytiles * xtiles;
        let duty = (tiles + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(tiles);

        for job in start..end {
            let ii = (job % ytiles) * RM as i64;
            let jj = (job / ytiles) * RN as i64;

            if jj + RN as i64 <= n {
                self.gemm_bloc::<RM, RN>(ii, jj);
            } else {
                match n - jj {
                    2 if RN >= 2 => self.gemm_bloc::<RM, 2>(ii, jj),
                    1             => self.gemm_bloc::<RM, 1>(ii, jj),
                    _             => {}
                }
            }
        }
    }
}

// ─── AVX-512F path ────────────────────────────────────────────────────────────

/// F16 × F16 GEMM using AVX-512F intrinsics.
///
/// Implements `tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, float>`
/// (C++ lines 3958-3965).
///
/// Each load reads 16 f16 values (256 bits) and converts to 16 f32 values in a
/// 512-bit register via `_mm512_cvtph_ps(_mm256_loadu_si256(...))`.
/// On AVX-512 (32 registers) the max tile is 4×6 vs 4×3 for AVX2.
#[cfg(target_arch = "x86_64")]
pub struct TinyBlasF16Avx512 {
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    k: i64,
    lda: i64,
    ldb: i64,
    ldc: i64,
    ith: i64,
    nth: i64,
}

#[cfg(target_arch = "x86_64")]
impl TinyBlasF16Avx512 {
    pub fn new(
        k: i64,
        a: *const u16, lda: i64,
        b: *const u16, ldb: i64,
        c: *mut f32,   ldc: i64,
        ith: i64, nth: i64,
    ) -> Self {
        Self { a, b, c, k, lda, ldb, ldc, ith, nth }
    }

    /// Returns `false` when k is not a multiple of 16 (KN for AVX-512).
    pub fn matmul(&self, m: i64, n: i64) -> bool {
        if self.k % 16 != 0 {
            return false;
        }
        let size_n = block_size(n, 6);
        if m % 4 == 0 {
            unsafe { self.gemm_dispatch(m, n, size_n); }
            true
        } else {
            false
        }
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f,f16c")]
    unsafe fn gemm_dispatch(&self, m: i64, n: i64, size_n: i64) {
        match size_n {
            6 => self.gemm::<4, 6>(m, n),
            5 => self.gemm::<4, 5>(m, n),
            4 => self.gemm::<4, 4>(m, n),
            3 => self.gemm::<4, 3>(m, n),
            2 => self.gemm::<4, 2>(m, n),
            1 => self.gemm::<4, 1>(m, n),
            _ => {}
        }
    }

    #[cfg(not(target_feature = "avx512f"))]
    unsafe fn gemm_dispatch(&self, _m: i64, _n: i64, _size_n: i64) {}

    /// Load 16 f16 values from `p` and widen to `__m512` of f32.
    ///
    /// C++ line 363: `return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));`
    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f,f16c")]
    #[inline]
    unsafe fn load_f16(p: *const u16) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i))
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f,f16c")]
    #[inline]
    unsafe fn hsum512(x: std::arch::x86_64::__m512) -> f32 {
        use std::arch::x86_64::*;
        _mm512_reduce_add_ps(x)
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f,f16c")]
    #[inline]
    unsafe fn gemm_bloc<const RM: usize, const RN: usize>(&self, ii: i64, jj: i64) {
        use std::arch::x86_64::*;
        let mut cv = [[_mm512_setzero_ps(); RM]; RN];

        let mut l = 0i64;
        while l < self.k {
            if RM <= RN {
                let mut av = [_mm512_setzero_ps(); RM];
                for i in 0..RM {
                    av[i] = Self::load_f16(
                        self.a.offset((self.lda * (ii + i as i64) + l) as isize),
                    );
                }
                for j in 0..RN {
                    let bv = Self::load_f16(
                        self.b.offset((self.ldb * (jj + j as i64) + l) as isize),
                    );
                    for i in 0..RM {
                        cv[j][i] = _mm512_fmadd_ps(av[i], bv, cv[j][i]);
                    }
                }
            } else {
                let mut bv = [_mm512_setzero_ps(); RN];
                for j in 0..RN {
                    bv[j] = Self::load_f16(
                        self.b.offset((self.ldb * (jj + j as i64) + l) as isize),
                    );
                }
                for i in 0..RM {
                    let av = Self::load_f16(
                        self.a.offset((self.lda * (ii + i as i64) + l) as isize),
                    );
                    for j in 0..RN {
                        cv[j][i] = _mm512_fmadd_ps(av, bv[j], cv[j][i]);
                    }
                }
            }
            l += 16; // KN = 16 f16 per __m512
        }

        for j in 0..RN {
            for i in 0..RM {
                *self.c.offset(
                    (self.ldc * (jj + j as i64) + (ii + i as i64)) as isize,
                ) = Self::hsum512(cv[j][i]);
            }
        }
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f,f16c")]
    #[inline(never)]
    unsafe fn gemm<const RM: usize, const RN: usize>(&self, m: i64, n: i64) {
        let ytiles = m / RM as i64;
        let xtiles = (n + RN as i64 - 1) / RN as i64;
        let tiles = ytiles * xtiles;
        let duty = (tiles + self.nth - 1) / self.nth;
        let start = duty * self.ith;
        let end = (start + duty).min(tiles);

        for job in start..end {
            let ii = (job % ytiles) * RM as i64;
            let jj = (job / ytiles) * RN as i64;

            if jj + RN as i64 <= n {
                self.gemm_bloc::<RM, RN>(ii, jj);
            } else {
                match n - jj {
                    5 if RN >= 5 => self.gemm_bloc::<RM, 5>(ii, jj),
                    4 if RN >= 4 => self.gemm_bloc::<RM, 4>(ii, jj),
                    3 if RN >= 3 => self.gemm_bloc::<RM, 3>(ii, jj),
                    2 if RN >= 2 => self.gemm_bloc::<RM, 2>(ii, jj),
                    1             => self.gemm_bloc::<RM, 1>(ii, jj),
                    _             => {}
                }
            }
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Dispatch F16 × F16 GEMM, called from `sgemm_llama::llamafile_sgemm`.
///
/// Selects AVX-512 when available (KN=16), otherwise falls back to AVX2+F16C
/// (KN=8). Returns `false` if neither is available or `k` is not aligned.
///
/// C++ lines 3957-4039.
///
/// # Safety
///
/// `a` and `b` must point to f16 data with `lda*m` and `ldb*n` u16 elements
/// respectively. `c` must point to `ldc*n` writable f32 elements.
pub unsafe fn llamafile_sgemm_f16(
    m: i64,
    n: i64,
    k: i64,
    a: *const u8,
    lda: i64,
    b: *const u8,
    ldb: i64,
    c: *mut f32,
    ldc: i64,
    ith: i64,
    nth: i64,
    b_type: GgmlType,
) -> bool {
    if b_type != GgmlType::F16 {
        return false;
    }

    let a_u16 = a as *const u16;
    let b_u16 = b as *const u16;

    #[cfg(target_arch = "x86_64")]
    {
        // Try AVX-512 first (wider registers, KN=16).
        // Fall through to AVX2 if k is not a multiple of 16.
        #[cfg(target_feature = "avx512f")]
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("f16c") {
            let tb = TinyBlasF16Avx512::new(
                k, a_u16, lda, b_u16, ldb, c, ldc, ith, nth,
            );
            let ok = tb.matmul(m, n);
            if ok { return true; }
            // k not multiple of 16 — try AVX2 (KN=8) below.
        }

        // AVX2 + F16C path (KN=8).
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            let tb = TinyBlasF16Avx2::new(
                k, a_u16, lda, b_u16, ldb, c, ldc, ith, nth,
            );
            return tb.matmul(m, n);
        }
    }

    #[allow(unused_variables)]
    let _ = (m, n, k, a_u16, b_u16, c, ldc, ith, nth);
    false
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_f16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = (bits >> 31) as u16;
        let exp  = ((bits >> 23) & 0xff) as i32;
        let mant = (bits & 0x7fffff) as u16;
        if exp == 0 { return sign << 15; }
        let e16 = exp - 127 + 15;
        if e16 <= 0 { return sign << 15; }
        if e16 >= 31 { return (sign << 15) | (31 << 10); }
        (sign << 15) | ((e16 as u16) << 10) | (mant >> 13)
    }

    // Fill a row-major f16 matrix (stored as u16) where every element = value.
    fn fill_f16(n: usize, value: f32) -> Vec<u16> {
        vec![f32_to_f16(value); n]
    }

    // 4×4 F16×F16: A is 4 rows × 8 cols of 1.0, B is 4 rows × 8 cols of 1.0.
    // Each dot product sums 8 pairs of (1.0 × 1.0) = 8.0.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f16_f16_4x4() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 8i64;
        let a = fill_f16((4 * k) as usize, 1.0);
        let b = fill_f16((4 * k) as usize, 1.0);
        let mut c = vec![0.0f32; 4 * 4];

        let tb = TinyBlasF16Avx2::new(
            k,
            a.as_ptr(), k,
            b.as_ptr(), k,
            c.as_mut_ptr(), 4,
            0, 1,
        );
        let ok = tb.matmul(4, 4);
        assert!(ok, "F16 4×4 matmul should succeed");

        for j in 0..4 {
            for i in 0..4 {
                let v = c[j * 4 + i];
                assert!(
                    (v - 8.0).abs() < 0.05,
                    "C[{j},{i}] = {v}, expected ≈ 8.0"
                );
            }
        }
    }

    // Verify that k not divisible by 8 returns false.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f16_k_not_aligned() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 7i64; // not a multiple of 8
        let a = fill_f16((4 * k) as usize, 1.0);
        let b = fill_f16((4 * k) as usize, 1.0);
        let mut c = vec![0.0f32; 4 * 4];

        let tb = TinyBlasF16Avx2::new(
            k, a.as_ptr(), k, b.as_ptr(), k, c.as_mut_ptr(), 4, 0, 1,
        );
        assert!(!tb.matmul(4, 4), "should return false for k not % 8");
    }

    // 4×2 tile with n=2 (exercises SIZE_N=2 dispatch).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f16_f16_4x2() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 8i64;
        let a = fill_f16((4 * k) as usize, 1.0);
        let b = fill_f16((2 * k) as usize, 1.0);
        let mut c = vec![0.0f32; 4 * 2];

        let tb = TinyBlasF16Avx2::new(
            k,
            a.as_ptr(), k,
            b.as_ptr(), k,
            c.as_mut_ptr(), 4,
            0, 1,
        );
        let ok = tb.matmul(4, 2);
        assert!(ok);

        for v in &c {
            assert!((*v - 8.0).abs() < 0.05, "expected ≈ 8.0, got {v}");
        }
    }

    // llamafile_sgemm_f16 integration test.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_llamafile_sgemm_f16_integration() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 8i64;
        let a = fill_f16((4 * k) as usize, 1.0);
        let b = fill_f16((4 * k) as usize, 1.0);
        let mut c = vec![0.0f32; 4 * 4];

        let handled = unsafe {
            llamafile_sgemm_f16(
                4, 4, k,
                a.as_ptr() as *const u8, k,
                b.as_ptr() as *const u8, k,
                c.as_mut_ptr(), 4,
                0, 1,
                GgmlType::F16,
            )
        };
        assert!(handled);

        for v in &c {
            assert!((*v - 8.0).abs() < 0.05, "expected ≈ 8.0, got {v}");
        }
    }
}

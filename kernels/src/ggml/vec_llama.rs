//! Rust translation of llama.cpp ggml/src/ggml-cpu/vec.h and vec.cpp.
//! Vectorised element-wise CPU operations for LLM inference.
//!
//! Scalar paths are always present.  AVX2+FMA and AVX-512 paths are gated on
//! `#[cfg(target_arch = "x86_64")]` + `#[target_feature(enable = "...")]`.
//! Public functions do runtime feature detection via `is_x86_feature_detected!`.

use half::{bf16, f16};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// erff is in the C standard library; Rust std does not expose it.
#[inline]
fn erf_f32(x: f32) -> f32 {
    extern "C" {
        fn erff(x: f32) -> f32;
    }
    unsafe { erff(x) }
}

// ── Constants from vec.h ────────────────────────────────────────────────────
const GELU_COEF_A: f32 = 0.044715;
const GELU_QUICK_COEF: f32 = -1.702;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const SQRT_2_INV: f32 = 0.7071067811865475;

// ── SIMD helper: horizontal sum of 8×f32 AVX register ──────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f32_8(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_movehdup_ps(sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

// ── Fast vectorised expf (ARM limited / llama.cpp polynomial) ───────────────
//
// Maximum error ≈ 1.45358 ulp + 0.5 ulp.
// Flushes to ∞ above 88.38 and to 0 below −103.97.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn expf_avx2(x: __m256) -> __m256 {
    // 0x1.8p23f = 12582912.0  (magic rounding constant)
    let r = _mm256_set1_ps(12582912.0_f32);
    // multiply by log2(e) and add r to round to integer in the float biased encoding
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950216_f32), r);
    let n = _mm256_sub_ps(z, r);
    // b = x - n*ln2_hi - n*ln2_lo
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(1.4286270e-6_f32),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931457519_f32), x),
    );
    // e = 2^n via bit manipulation of the float exponent field
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0_f32)),
    ));
    // check whether |n| > 126 (overflow / underflow)
    let c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0_f32), n),
        _mm256_set1_ps(126.0_f32),
        _CMP_GT_OQ,
    ));
    let u = _mm256_mul_ps(b, b);
    // polynomial: j = ((c4*b + c3)*u + (c2*b + c1))*u + c0*b
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(0.00824762_f32),
                b,
                _mm256_set1_ps(0.04189994_f32),
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(0.16668272_f32),
                b,
                _mm256_set1_ps(0.49996537_f32),
            ),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(0.99999702_f32), b),
    );
    // fast path: no over/underflow
    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    // slow path: handle over/underflow
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000_u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000_u32 as i32)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0_f32), n),
        _mm256_set1_ps(192.0_f32),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

// ── AVX-512 fast expf ───────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
pub unsafe fn expf_avx512(x: __m512) -> __m512 {
    let r = _mm512_set1_ps(12582912.0_f32);
    let z = _mm512_fmadd_ps(x, _mm512_set1_ps(1.4426950216_f32), r);
    let n = _mm512_sub_ps(z, r);
    let b = _mm512_fnmadd_ps(
        n,
        _mm512_set1_ps(1.4286270e-6_f32),
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0.6931457519_f32), x),
    );
    let d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192.0_f32), _CMP_GT_OQ);
    let u = _mm512_mul_ps(b, b);
    let j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_set1_ps(0.00824762_f32),
                b,
                _mm512_set1_ps(0.04189994_f32),
            ),
            u,
            _mm512_fmadd_ps(
                _mm512_set1_ps(0.16668272_f32),
                b,
                _mm512_set1_ps(0.49996537_f32),
            ),
        ),
        u,
        _mm512_fmadd_ps(_mm512_set1_ps(0.99999702_f32), b, _mm512_set1_ps(1.0_f32)),
    );
    let res = _mm512_scalef_ps(j, n);
    if _mm512_kortestz(d, d) != 0 {
        return res;
    }
    let zero = _mm512_setzero_ps();
    let alt = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ),
        _mm512_set1_ps(f32::INFINITY),
        zero,
    );
    _mm512_mask_blend_ps(d, res, alt)
}

// ── AVX2 vectorised SiLU: x / (1 + exp(-x)) ────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn silu_avx2(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0_f32);
    let zero = _mm256_setzero_ps();
    let neg_x = _mm256_sub_ps(zero, x);
    let exp_neg_x = expf_avx2(neg_x);
    let denom = _mm256_add_ps(one, exp_neg_x);
    _mm256_div_ps(x, denom)
}

// ── AVX-512 vectorised SiLU ─────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
pub unsafe fn silu_avx512(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0_f32);
    let zero = _mm512_setzero_ps();
    let neg_x = _mm512_sub_ps(zero, x);
    let exp_neg_x = expf_avx512(neg_x);
    let denom = _mm512_add_ps(one, exp_neg_x);
    _mm512_div_ps(x, denom)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Set / copy
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
pub fn vec_set_i8(x: &mut [i8], v: i8) {
    x.fill(v);
}

#[inline]
pub fn vec_set_i16(x: &mut [i16], v: i16) {
    x.fill(v);
}

#[inline]
pub fn vec_set_i32(x: &mut [i32], v: i32) {
    x.fill(v);
}

#[inline]
pub fn vec_cpy_i32(y: &mut [i32], x: &[i32]) {
    y.copy_from_slice(x);
}

#[inline]
pub fn vec_set_f32(x: &mut [f32], v: f32) {
    x.fill(v);
}

#[inline]
pub fn vec_cpy_f32(y: &mut [f32], x: &[f32]) {
    y.copy_from_slice(x);
}

#[inline]
pub fn vec_set_f16(x: &mut [f16], v: f16) {
    x.fill(v);
}

#[inline]
pub fn vec_set_bf16(x: &mut [bf16], v: bf16) {
    x.fill(v);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Element-wise arithmetic — f32
// ═══════════════════════════════════════════════════════════════════════════

/// z = x + y  (AVX2: 8-wide; scalar tail)
pub fn vec_add_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    let n = z.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            while i + 7 < n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let vy = _mm256_loadu_ps(y.as_ptr().add(i));
                _mm256_storeu_ps(z.as_mut_ptr().add(i), _mm256_add_ps(vx, vy));
                i += 8;
            }
        }
    }
    while i < n {
        z[i] = x[i] + y[i];
        i += 1;
    }
}

/// z = x + y  (f16, scalar)
#[inline]
pub fn vec_add_f16(z: &mut [f16], x: &[f16], y: &[f16]) {
    for i in 0..z.len() {
        z[i] = f16::from_f32(x[i].to_f32() + y[i].to_f32());
    }
}

/// z = x + v  (scalar broadcast)
#[inline]
pub fn vec_add1_f32(z: &mut [f32], x: &[f32], v: f32) {
    for i in 0..z.len() {
        z[i] = x[i] + v;
    }
}

/// y += x
#[inline]
pub fn vec_acc_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] += x[i];
    }
}

/// y += v
#[inline]
pub fn vec_acc1_f32(y: &mut [f32], v: f32) {
    for yi in y.iter_mut() {
        *yi += v;
    }
}

/// z = x - y  (f32, scalar)
#[inline]
pub fn vec_sub_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    for i in 0..z.len() {
        z[i] = x[i] - y[i];
    }
}

/// z = x - y  (f16, scalar)
#[inline]
pub fn vec_sub_f16(z: &mut [f16], x: &[f16], y: &[f16]) {
    for i in 0..z.len() {
        z[i] = f16::from_f32(x[i].to_f32() - y[i].to_f32());
    }
}

/// y = -x  (f32)
#[inline]
pub fn vec_neg_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = -x[i];
    }
}

/// y = -x  (f16)
#[inline]
pub fn vec_neg_f16(y: &mut [f16], x: &[f16]) {
    for i in 0..y.len() {
        y[i] = f16::from_f32(-x[i].to_f32());
    }
}

/// z = x * y  (f32)
#[inline]
pub fn vec_mul_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    for i in 0..z.len() {
        z[i] = x[i] * y[i];
    }
}

/// z = x * y  (f16)
#[inline]
pub fn vec_mul_f16(z: &mut [f16], x: &[f16], y: &[f16]) {
    for i in 0..z.len() {
        z[i] = f16::from_f32(x[i].to_f32() * y[i].to_f32());
    }
}

/// z = x / y  (f32)
#[inline]
pub fn vec_div_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
    for i in 0..z.len() {
        z[i] = x[i] / y[i];
    }
}

/// z = x / y  (f16)
#[inline]
pub fn vec_div_f16(z: &mut [f16], x: &[f16], y: &[f16]) {
    for i in 0..z.len() {
        z[i] = f16::from_f32(x[i].to_f32() / y[i].to_f32());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Multiply-add variants
// ═══════════════════════════════════════════════════════════════════════════

/// y += x * v  (f32, AVX2 unrolled 8×4=32 per step)
pub fn vec_mad_f32(y: &mut [f32], x: &[f32], v: f32) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let np = n & !31;
        unsafe {
            let vv = _mm256_set1_ps(v);
            while i < np {
                let ay0 = _mm256_loadu_ps(y.as_ptr().add(i));
                let ay1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
                let ay2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
                let ay3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i),
                    _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vv, ay0),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 8),
                    _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vv, ay1),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 16),
                    _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 16)), vv, ay2),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 24),
                    _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 24)), vv, ay3),
                );
                i += 32;
            }
        }
    }
    while i < n {
        y[i] += x[i] * v;
        i += 1;
    }
}

/// y = x * s + b  (f32, scalar; called ggml_vec_mad1_f32 in llama.cpp)
pub fn vec_mad1_f32(y: &mut [f32], x: &[f32], s: f32, b: f32) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let np = n & !31;
        unsafe {
            let vs = _mm256_set1_ps(s);
            let vb = _mm256_set1_ps(b);
            while i < np {
                let ay0 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vs, vb);
                let ay1 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vs, vb);
                let ay2 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 16)), vs, vb);
                let ay3 = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i + 24)), vs, vb);
                _mm256_storeu_ps(y.as_mut_ptr().add(i), ay0);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), ay1);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 16), ay2);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 24), ay3);
                i += 32;
            }
        }
    }
    while i < n {
        y[i] = x[i] * s + b;
        i += 1;
    }
}

/// Unrolled MAD: y += x[k] * v[k]  for k in 0..UNROLL (=32).
///
/// `xs` — row stride of x in bytes; `vs` — row stride of v in bytes.
/// Mirrors `ggml_vec_mad_f32_unroll`.
pub fn vec_mad_f32_unroll<const UNROLL: usize>(
    y: &mut [f32],
    x_rows: [&[f32]; UNROLL],
    v_scalars: [f32; UNROLL],
) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let np = n & !31;
        unsafe {
            let mut vv = [_mm256_setzero_ps(); UNROLL];
            for k in 0..UNROLL {
                vv[k] = _mm256_set1_ps(v_scalars[k]);
            }
            while i < np {
                let mut ay0 = _mm256_loadu_ps(y.as_ptr().add(i));
                let mut ay1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
                let mut ay2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
                let mut ay3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
                for k in 0..UNROLL {
                    ay0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_rows[k].as_ptr().add(i)), vv[k], ay0);
                    ay1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_rows[k].as_ptr().add(i + 8)), vv[k], ay1);
                    ay2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_rows[k].as_ptr().add(i + 16)), vv[k], ay2);
                    ay3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_rows[k].as_ptr().add(i + 24)), vv[k], ay3);
                }
                _mm256_storeu_ps(y.as_mut_ptr().add(i), ay0);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), ay1);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 16), ay2);
                _mm256_storeu_ps(y.as_mut_ptr().add(i + 24), ay3);
                i += 32;
            }
        }
    }
    while i < n {
        for k in 0..UNROLL {
            y[i] += x_rows[k][i] * v_scalars[k];
        }
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Scale in-place
// ═══════════════════════════════════════════════════════════════════════════

/// y *= v  (f32, AVX2)
pub fn vec_scale_f32(y: &mut [f32], v: f32) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        let np = n & !31;
        unsafe {
            let vv = _mm256_set1_ps(v);
            while i < np {
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i),
                    _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr().add(i)), vv),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 8),
                    _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr().add(i + 8)), vv),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 16),
                    _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr().add(i + 16)), vv),
                );
                _mm256_storeu_ps(
                    y.as_mut_ptr().add(i + 24),
                    _mm256_mul_ps(_mm256_loadu_ps(y.as_ptr().add(i + 24)), vv),
                );
                i += 32;
            }
        }
    }
    while i < n {
        y[i] *= v;
        i += 1;
    }
}

/// y *= v  (f16, scalar)
#[inline]
pub fn vec_scale_f16(y: &mut [f16], v: f32) {
    for yi in y.iter_mut() {
        *yi = f16::from_f32(yi.to_f32() * v);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dot products
// ═══════════════════════════════════════════════════════════════════════════

/// Dot product of two f32 slices (AVX2 4×8-wide accumulation).
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let mut i = 0;
    let mut sumf = 0.0_f32;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let np = n & !31;
        unsafe {
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();
            while i < np {
                s0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                    s0,
                );
                s1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 8)),
                    _mm256_loadu_ps(y.as_ptr().add(i + 8)),
                    s1,
                );
                s2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 16)),
                    _mm256_loadu_ps(y.as_ptr().add(i + 16)),
                    s2,
                );
                s3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 24)),
                    _mm256_loadu_ps(y.as_ptr().add(i + 24)),
                    s3,
                );
                i += 32;
            }
            s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            sumf = hsum_f32_8(s0);
        }
    }

    while i < n {
        sumf += x[i] * y[i];
        i += 1;
    }
    sumf
}

/// Dot product of two f16 slices (scalar via f32 widening).
pub fn vec_dot_f16(x: &[f16], y: &[f16]) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let mut sumf = 0.0_f64;
    for i in 0..n {
        sumf += (x[i].to_f32() * y[i].to_f32()) as f64;
    }
    sumf as f32
}

/// Dot product of two bf16 slices (AVX2: expand to f32 then multiply).
pub fn vec_dot_bf16(x: &[bf16], y: &[bf16]) -> f32 {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let mut i = 0;
    let mut sumf = 0.0_f64;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // Expand bf16 → f32 by shifting the 16-bit word into the upper 16 bits of a 32-bit float.
        let np = n & !31;
        unsafe {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();
            while i < np {
                // load 8 bf16 values and expand to f32
                let load_bf16_x8 = |ptr: *const bf16| -> __m256 {
                    let raw = _mm_loadu_si128(ptr as *const __m128i);
                    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16))
                };
                c0 = _mm256_add_ps(_mm256_mul_ps(load_bf16_x8(x.as_ptr().add(i)), load_bf16_x8(y.as_ptr().add(i))), c0);
                c1 = _mm256_add_ps(_mm256_mul_ps(load_bf16_x8(x.as_ptr().add(i + 8)), load_bf16_x8(y.as_ptr().add(i + 8))), c1);
                c2 = _mm256_add_ps(_mm256_mul_ps(load_bf16_x8(x.as_ptr().add(i + 16)), load_bf16_x8(y.as_ptr().add(i + 16))), c2);
                c3 = _mm256_add_ps(_mm256_mul_ps(load_bf16_x8(x.as_ptr().add(i + 24)), load_bf16_x8(y.as_ptr().add(i + 24))), c3);
                i += 32;
            }
            c0 = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
            sumf += hsum_f32_8(c0) as f64;
        }
    }

    while i < n {
        sumf += (x[i].to_f32() * y[i].to_f32()) as f64;
        i += 1;
    }
    sumf as f32
}

/// Compute 2 dot products simultaneously (unrolled; mirrors ggml_vec_dot_f16_unroll).
pub fn vec_dot_f16_unroll(x0: &[f16], x1: &[f16], y: &[f16]) -> [f32; 2] {
    let n = y.len();
    let mut s = [0.0_f64; 2];
    for i in 0..n {
        let yi = y[i].to_f32();
        s[0] += (x0[i].to_f32() * yi) as f64;
        s[1] += (x1[i].to_f32() * yi) as f64;
    }
    [s[0] as f32, s[1] as f32]
}

// ═══════════════════════════════════════════════════════════════════════════
//  Norm
// ═══════════════════════════════════════════════════════════════════════════

/// L2 norm: √(x·x)
#[inline]
pub fn vec_norm_f32(x: &[f32]) -> f32 {
    vec_dot_f32(x, x).sqrt()
}

/// 1 / L2 norm
#[inline]
pub fn vec_norm_inv_f32(x: &[f32]) -> f32 {
    1.0 / vec_norm_f32(x)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Elementwise unary — scalar
// ═══════════════════════════════════════════════════════════════════════════

#[inline] pub fn vec_sqr_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i]*x[i]; } }
#[inline] pub fn vec_sqrt_f32(y: &mut [f32], x: &[f32]) { for i in 0..y.len() { y[i] = x[i].sqrt(); } }
#[inline] pub fn vec_log_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i].ln(); } }
#[inline] pub fn vec_sin_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i].sin(); } }
#[inline] pub fn vec_cos_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i].cos(); } }
#[inline] pub fn vec_abs_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i].abs(); } }
#[inline] pub fn vec_exp_f32(y: &mut [f32], x: &[f32])  { for i in 0..y.len() { y[i] = x[i].exp(); } }
#[inline] pub fn vec_tanh_f32(y: &mut [f32], x: &[f32]) { for i in 0..y.len() { y[i] = x[i].tanh(); } }

#[inline]
pub fn vec_sgn_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = if x[i] > 0.0 { 1.0 } else if x[i] < 0.0 { -1.0 } else { 0.0 };
    }
}

#[inline]
pub fn vec_step_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = if x[i] > 0.0 { 1.0 } else { 0.0 };
    }
}

/// ELU: x if x > 0, else exp(x) - 1
#[inline]
pub fn vec_elu_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = if x[i] > 0.0 { x[i] } else { x[i].exp_m1() };
    }
}

/// ReLU: max(0, x)
#[inline]
pub fn vec_relu_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = if x[i] > 0.0 { x[i] } else { 0.0 };
    }
}

/// Leaky ReLU: max(0,x) + ns*min(0,x)
#[inline]
pub fn vec_leaky_relu_f32(y: &mut [f32], x: &[f32], ns: f32) {
    for i in 0..y.len() {
        let pos = if x[i] > 0.0 { x[i] } else { 0.0 };
        let neg = if x[i] < 0.0 { x[i] } else { 0.0 };
        y[i] = pos + ns * neg;
    }
}

/// Sigmoid: 1 / (1 + exp(-x))
#[inline]
pub fn vec_sigmoid_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = 1.0 / (1.0 + (-x[i]).exp());
    }
}

/// Hard swish: x * clamp((x+3)/6, 0, 1)
#[inline]
pub fn vec_hardswish_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = x[i] * ((x[i] + 3.0) / 6.0).clamp(0.0, 1.0);
    }
}

/// Hard sigmoid: clamp((x+3)/6, 0, 1)
#[inline]
pub fn vec_hardsigmoid_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = ((x[i] + 3.0) / 6.0).clamp(0.0, 1.0);
    }
}

// f16 variants (scalar, dequant→compute→quant)
#[inline] pub fn vec_sqr_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(v*v); } }
#[inline] pub fn vec_sqrt_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().sqrt()); } }
#[inline] pub fn vec_log_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().ln()); } }
#[inline] pub fn vec_sin_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().sin()); } }
#[inline] pub fn vec_cos_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().cos()); } }
#[inline] pub fn vec_abs_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().abs()); } }
#[inline] pub fn vec_exp_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().exp()); } }
#[inline] pub fn vec_tanh_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { y[i] = f16::from_f32(x[i].to_f32().tanh()); } }
#[inline] pub fn vec_elu_f16(y: &mut [f16], x: &[f16])  { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(if v > 0.0 { v } else { v.exp_m1() }); } }
#[inline] pub fn vec_relu_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(if v > 0.0 { v } else { 0.0 }); } }
#[inline] pub fn vec_sigmoid_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(1.0 / (1.0 + (-v).exp())); } }
#[inline] pub fn vec_hardswish_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(v * ((v + 3.0) / 6.0).clamp(0.0, 1.0)); } }
#[inline] pub fn vec_hardsigmoid_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { y[i] = f16::from_f32(((x[i].to_f32() + 3.0) / 6.0).clamp(0.0, 1.0)); } }
#[inline] pub fn vec_sgn_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32(if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }); } }
#[inline] pub fn vec_step_f16(y: &mut [f16], x: &[f16]) { for i in 0..y.len() { y[i] = f16::from_f32(if x[i].to_f32() > 0.0 { 1.0 } else { 0.0 }); } }
#[inline] pub fn vec_leaky_relu_f16(y: &mut [f16], x: &[f16], ns: f32) { for i in 0..y.len() { let v = x[i].to_f32(); y[i] = f16::from_f32((if v > 0.0 { v } else { 0.0 }) + ns * (if v < 0.0 { v } else { 0.0 })); } }

// ═══════════════════════════════════════════════════════════════════════════
//  GELU
// ═══════════════════════════════════════════════════════════════════════════

/// GELU scalar: 0.5 * x * (1 + tanh(√(2/π) * x * (1 + GELU_COEF_A * x²)))
/// Uses the f16 lookup table — zero tanh/exp at inference time.
#[inline]
pub fn gelu_f32(x: f32) -> f32 {
    super::helpers::gelu_f32_lut(x)
}

/// GELU vector (direct formula; no table).
pub fn vec_gelu_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        let xi = x[i];
        if xi <= -10.0 {
            y[i] = 0.0;
        } else if xi >= 10.0 {
            y[i] = xi;
        } else {
            y[i] = gelu_f32(xi);
        }
    }
}

/// GELU with erf: 0.5 * x * (1 + erf(x / √2))
pub fn vec_gelu_erf_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        let xi = x[i];
        y[i] = 0.5 * xi * (1.0 + erf_f32(xi * SQRT_2_INV));
    }
}

/// Quick GELU scalar: x * sigmoid(1.702 * x)
/// Uses the f16 lookup table — zero exp at inference time.
#[inline]
pub fn gelu_quick_f32(x: f32) -> f32 {
    super::helpers::gelu_quick_f32_lut(x)
}

/// Quick GELU vector.
pub fn vec_gelu_quick_f32(y: &mut [f32], x: &[f32]) {
    for i in 0..y.len() {
        y[i] = gelu_quick_f32(x[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SiLU  x / (1 + exp(-x))
// ═══════════════════════════════════════════════════════════════════════════

/// SiLU scalar
#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SiLU vector (AVX-512 → AVX2 → scalar)
pub fn vec_silu_f32(y: &mut [f32], x: &[f32]) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            unsafe {
                while i + 15 < n {
                    _mm512_storeu_ps(
                        y.as_mut_ptr().add(i),
                        silu_avx512(_mm512_loadu_ps(x.as_ptr().add(i))),
                    );
                    i += 16;
                }
            }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                while i + 7 < n {
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        silu_avx2(_mm256_loadu_ps(x.as_ptr().add(i))),
                    );
                    i += 8;
                }
            }
        }
    }
    while i < n {
        y[i] = silu_f32(x[i]);
        i += 1;
    }
}

/// SwiGLU: y = silu(x) * g  (fused for cache efficiency)
pub fn vec_swiglu_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
    let n = y.len();
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            unsafe {
                while i + 15 < n {
                    _mm512_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm512_mul_ps(
                            silu_avx512(_mm512_loadu_ps(x.as_ptr().add(i))),
                            _mm512_loadu_ps(g.as_ptr().add(i)),
                        ),
                    );
                    i += 16;
                }
            }
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                while i + 7 < n {
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(
                            silu_avx2(_mm256_loadu_ps(x.as_ptr().add(i))),
                            _mm256_loadu_ps(g.as_ptr().add(i)),
                        ),
                    );
                    i += 8;
                }
            }
        }
    }
    while i < n {
        y[i] = silu_f32(x[i]) * g[i];
        i += 1;
    }
}

// SiLU backward: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[inline]
pub fn silu_backward_f32(x: f32, dy: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    dy * s * (1.0 + x * (1.0 - s))
}

pub fn vec_silu_backward_f32(dx: &mut [f32], x: &[f32], dy: &[f32]) {
    for i in 0..dx.len() {
        dx[i] = silu_backward_f32(x[i], dy[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  GLU gate variants
// ═══════════════════════════════════════════════════════════════════════════

/// ReGLU: y = relu(x) * g
#[inline]
pub fn vec_reglu_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
    for i in 0..y.len() {
        y[i] = if x[i] > 0.0 { x[i] * g[i] } else { 0.0 };
    }
}

/// GeGLU: y = gelu(x) * g
pub fn vec_geglu_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
    for i in 0..y.len() {
        let xi = x[i];
        let gv = if xi <= -10.0 {
            0.0
        } else if xi >= 10.0 {
            xi
        } else {
            gelu_f32(xi)
        };
        y[i] = gv * g[i];
    }
}

/// GeGLU with erf
pub fn vec_geglu_erf_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
    for i in 0..y.len() {
        let xi = x[i];
        y[i] = 0.5 * xi * (1.0 + erf_f32(xi * SQRT_2_INV)) * g[i];
    }
}

/// Quick GeGLU
pub fn vec_geglu_quick_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
    for i in 0..y.len() {
        y[i] = gelu_quick_f32(x[i]) * g[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Sum of all elements (f64 accumulator for precision)
pub fn vec_sum_f32(x: &[f32]) -> f32 {
    let mut s = 0.0_f64;
    for &v in x {
        s += v as f64;
    }
    s as f32
}

/// Prefix-sum (in-place cumulative sum)
pub fn vec_cumsum_f32(y: &mut [f32], x: &[f32]) {
    let mut acc = 0.0_f32;
    for i in 0..y.len() {
        acc += x[i];
        y[i] = acc;
    }
}

/// Sum into f64 accumulator (ggml_vec_sum_f32_ggf)
pub fn vec_sum_f32_f64(x: &[f32]) -> f64 {
    let mut s = 0.0_f64;
    for &v in x {
        s += v as f64;
    }
    s
}

/// Sum of f16 values into f32
pub fn vec_sum_f16(x: &[f16]) -> f32 {
    let mut s = 0.0_f32;
    for &v in x {
        s += v.to_f32();
    }
    s
}

/// Sum of bf16 values into f32
pub fn vec_sum_bf16(x: &[bf16]) -> f32 {
    let mut s = 0.0_f32;
    for &v in x {
        s += v.to_f32();
    }
    s
}

/// Maximum element
pub fn vec_max_f32(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Index of maximum element
pub fn vec_argmax_f32(x: &[f32]) -> usize {
    let mut max = f32::NEG_INFINITY;
    let mut idx = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > max {
            max = v;
            idx = i;
        }
    }
    idx
}

// ═══════════════════════════════════════════════════════════════════════════
//  Softmax utilities
// ═══════════════════════════════════════════════════════════════════════════

/// y[i] = exp(x[i] - max); returns Σ y[i].
/// AVX2+FMA path uses the fast polynomial exp approximation.
pub fn vec_soft_max_f32(y: &mut [f32], x: &[f32], max: f32) -> f64 {
    let n = y.len();
    let mut i = 0;
    let mut sum = 0.0_f64;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            let vmax = _mm256_set1_ps(max);
            while i + 7 < n {
                let val = expf_avx2(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vmax));
                _mm256_storeu_ps(y.as_mut_ptr().add(i), val);
                // horizontal sum of 8 lanes
                let hi = _mm256_extractf128_ps(val, 1);
                let lo = _mm256_castps256_ps128(val);
                let s128 = _mm_add_ps(lo, hi);
                let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
                let s32 = _mm_add_ss(s64, _mm_movehdup_ps(s64));
                sum += _mm_cvtss_f32(s32) as f64;
                i += 8;
            }
        }
    }

    while i < n {
        let v = (x[i] - max).exp();
        y[i] = v;
        sum += v as f64;
        i += 1;
    }
    sum
}

/// Log-softmax: y[i] = x[i] - max; returns log(Σ exp(y[i])).
pub fn vec_log_soft_max_f32(y: &mut [f32], x: &[f32], max: f32) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..y.len() {
        let v = x[i] - max;
        y[i] = v;
        sum += v.exp() as f64;
    }
    sum.ln()
}

/// Centred variance: y[i] = x[i] - mean; returns Σ(y[i]²) / n.
/// AVX2+FMA path.
pub fn vec_cvar_f32(y: &mut [f32], x: &[f32], mean: f32) -> f64 {
    let n = y.len();
    let mut i = 0;
    let mut sum = 0.0_f64;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            let vmean = _mm256_set1_ps(mean);
            while i + 7 < n {
                let val = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vmean);
                _mm256_storeu_ps(y.as_mut_ptr().add(i), val);
                let sq = _mm256_mul_ps(val, val);
                sum += hsum_f32_8(sq) as f64;
                i += 8;
            }
        }
    }

    while i < n {
        let v = x[i] - mean;
        y[i] = v;
        sum += (v * v) as f64;
        i += 1;
    }
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════
//  f16 SiLU (scalar)
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
pub fn vec_silu_f16(y: &mut [f16], x: &[f16]) {
    for i in 0..y.len() {
        let v = x[i].to_f32();
        y[i] = f16::from_f32(v / (1.0 + (-v).exp()));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn naive_dot(x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn make_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect()
    }

    // ── dot product ─────────────────────────────────────────────────────────

    #[test]
    fn test_dot_f32_small() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0];
        let got = vec_dot_f32(&x, &y);
        assert_relative_eq!(got, 20.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dot_f32_large() {
        let n = 256;
        let x = make_f32(n);
        let y = make_f32(n);
        let expected = naive_dot(&x, &y);
        let got = vec_dot_f32(&x, &y);
        assert_relative_eq!(got, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_dot_f32_non_multiple_of_32() {
        let n = 37;
        let x = make_f32(n);
        let y = make_f32(n);
        let expected = naive_dot(&x, &y);
        let got = vec_dot_f32(&x, &y);
        assert_relative_eq!(got, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_dot_f16_against_scalar() {
        let n = 64;
        let x_f32 = make_f32(n);
        let y_f32 = make_f32(n);
        let x: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let y: Vec<f16> = y_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let naive: f32 = x.iter().zip(y.iter()).map(|(a, b)| a.to_f32() * b.to_f32()).sum();
        let got = vec_dot_f16(&x, &y);
        assert_relative_eq!(got, naive, epsilon = 1e-2);
    }

    #[test]
    fn test_dot_bf16_against_scalar() {
        let n = 64;
        let x_f32 = make_f32(n);
        let y_f32 = make_f32(n);
        let x: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let y: Vec<bf16> = y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let naive: f32 = x.iter().zip(y.iter()).map(|(a, b)| a.to_f32() * b.to_f32()).sum();
        let got = vec_dot_bf16(&x, &y);
        assert_relative_eq!(got, naive, epsilon = 1e-1);
    }

    // ── add ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_add_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y: Vec<f32> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut z = vec![0.0_f32; 9];
        vec_add_f32(&mut z, &x, &y);
        for v in &z {
            assert_relative_eq!(*v, 10.0, epsilon = 1e-6);
        }
    }

    // ── mad ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mad_f32() {
        let n = 35;
        let x = make_f32(n);
        let mut y = vec![0.0_f32; n];
        vec_mad_f32(&mut y, &x, 2.0);
        for i in 0..n {
            assert_relative_eq!(y[i], x[i] * 2.0, epsilon = 1e-5);
        }
    }

    // ── scale ───────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_f32() {
        let n = 33;
        let mut y = make_f32(n);
        let orig = y.clone();
        vec_scale_f32(&mut y, 3.0);
        for i in 0..n {
            assert_relative_eq!(y[i], orig[i] * 3.0, epsilon = 1e-5);
        }
    }

    // ── silu ────────────────────────────────────────────────────────────────

    #[test]
    fn test_silu_f32_against_scalar() {
        let n = 17;
        let x = make_f32(n);
        let expected: Vec<f32> = x.iter().map(|&v| silu_f32(v)).collect();
        let mut y = vec![0.0_f32; n];
        vec_silu_f32(&mut y, &x);
        for i in 0..n {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_silu_scalar() {
        assert_relative_eq!(silu_f32(0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(silu_f32(1.0), 0.7310586_f32, epsilon = 1e-5);
        assert_relative_eq!(silu_f32(-1.0), -0.2689414_f32, epsilon = 1e-5);
    }

    // ── gelu ────────────────────────────────────────────────────────────────

    #[test]
    fn test_gelu_scalar() {
        assert_relative_eq!(gelu_f32(0.0), 0.0, epsilon = 1e-6);
        // gelu(1.0) ≈ 0.8413
        assert_relative_eq!(gelu_f32(1.0), 0.8413_f32, epsilon = 1e-3);
    }

    #[test]
    fn test_gelu_f32_clamps() {
        let mut y = vec![0.0_f32; 3];
        let x = vec![-11.0, 11.0, 0.5];
        vec_gelu_f32(&mut y, &x);
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(y[1], 11.0, epsilon = 1e-6);
        assert_relative_eq!(y[2], gelu_f32(0.5), epsilon = 1e-5);
    }

    // ── swiglu ──────────────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_f32_against_scalar() {
        let n = 17;
        let x = make_f32(n);
        let g = make_f32(n);
        let expected: Vec<f32> = x.iter().zip(g.iter()).map(|(&xi, &gi)| silu_f32(xi) * gi).collect();
        let mut y = vec![0.0_f32; n];
        vec_swiglu_f32(&mut y, &x, &g);
        for i in 0..n {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-5);
        }
    }

    // ── softmax ─────────────────────────────────────────────────────────────

    #[test]
    fn test_soft_max_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let max = 3.0_f32;
        let mut y = vec![0.0_f32; 3];
        let sum = vec_soft_max_f32(&mut y, &x, max);
        // y = [exp(-2), exp(-1), exp(0)] = [0.1353, 0.3679, 1.0]
        assert_relative_eq!(y[0], (-2.0_f32).exp(), epsilon = 1e-5);
        assert_relative_eq!(y[1], (-1.0_f32).exp(), epsilon = 1e-5);
        assert_relative_eq!(y[2], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sum as f32, y[0] + y[1] + y[2], epsilon = 1e-4);
    }

    #[test]
    fn test_soft_max_avx2_matches_scalar() {
        let n = 17;
        let x = make_f32(n);
        let max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut y_vec = vec![0.0_f32; n];
        let mut y_scalar = vec![0.0_f32; n];

        let sum_vec = vec_soft_max_f32(&mut y_vec, &x, max);
        for i in 0..n {
            let v = (x[i] - max).exp();
            y_scalar[i] = v;
        }
        let sum_scalar: f64 = y_scalar.iter().map(|&v| v as f64).sum();

        assert_relative_eq!(sum_vec as f32, sum_scalar as f32, epsilon = 1e-3);
        for i in 0..n {
            assert_relative_eq!(y_vec[i], y_scalar[i], epsilon = 1e-4);
        }
    }

    // ── cvar ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cvar_f32() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0_f32;
        let mut y = vec![0.0_f32; 5];
        let var = vec_cvar_f32(&mut y, &x, mean);
        // y = [-2, -1, 0, 1, 2]; variance = (4+1+0+1+4)/5 = 2.0
        assert_relative_eq!(var, 2.0_f64, epsilon = 1e-5);
        assert_relative_eq!(y[0], -2.0, epsilon = 1e-6);
        assert_relative_eq!(y[4], 2.0, epsilon = 1e-6);
    }

    // ── norm ────────────────────────────────────────────────────────────────

    #[test]
    fn test_norm_f32() {
        let x = vec![3.0_f32, 4.0];
        assert_relative_eq!(vec_norm_f32(&x), 5.0, epsilon = 1e-5);
    }

    // ── reduction ───────────────────────────────────────────────────────────

    #[test]
    fn test_sum_f32() {
        let x: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        assert_relative_eq!(vec_sum_f32(&x), 55.0, epsilon = 1e-4);
    }

    #[test]
    fn test_max_f32() {
        let x = vec![1.0_f32, 5.0, 3.0, 2.0];
        assert_eq!(vec_max_f32(&x), 5.0);
    }

    #[test]
    fn test_argmax_f32() {
        let x = vec![1.0_f32, 5.0, 3.0, 2.0];
        assert_eq!(vec_argmax_f32(&x), 1);
    }

    #[test]
    fn test_cumsum_f32() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0_f32; 4];
        vec_cumsum_f32(&mut y, &x);
        assert_eq!(y, vec![1.0, 3.0, 6.0, 10.0]);
    }

    // ── activations ─────────────────────────────────────────────────────────

    #[test]
    fn test_relu_f32() {
        let x = vec![-1.0_f32, 0.0, 1.0, 2.0];
        let mut y = vec![0.0_f32; 4];
        vec_relu_f32(&mut y, &x);
        assert_eq!(y, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_f32() {
        let x = vec![0.0_f32];
        let mut y = vec![0.0_f32; 1];
        vec_sigmoid_f32(&mut y, &x);
        assert_relative_eq!(y[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_log_soft_max_f32() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let max = 3.0_f32;
        let mut y = vec![0.0_f32; 3];
        let log_sum = vec_log_soft_max_f32(&mut y, &x, max);
        // y = [x-max] = [-2, -1, 0]
        // sum = exp(-2) + exp(-1) + 1
        let expected_sum = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        assert_relative_eq!(log_sum, expected_sum.ln(), epsilon = 1e-5);
    }
}

//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c
//! Plain AVX paths (#elif defined(__AVX__)) -- 128-bit split operations.

#![allow(unused_imports, dead_code, unused_variables, unused_mut)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;
use super::iq_tables::*;
use super::dequantize::ue4m3_to_f32_pub;

const QK_K: usize = 256;
const QK4_NL: usize = 32;

// keven_signs_q2xs[1024]: recast as u64[128], indexed by 7-bit sign pattern.
// Copied verbatim from quants.c lines 2624-2656.
static KEVEN_SIGNS_Q2XS: [i8; 1024] = [
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
];

// e8m0_to_f32_half: E8M0 byte -> f32, then halved (GGML_CPU_E8M0_TO_FP32_HALF)
#[inline(always)]
fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 { 0x00200000u32 << x } else { (x as u32 - 1) << 23 };
    f32::from_bits(bits)
}

// -- AVX helpers ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

// AVX version: 128-bit split (lines 196-206 of quants.c)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmpl = _mm_loadu_si128(rsi as *const __m128i);
    let tmph = _mm_srli_epi16(tmpl, 4);
    let low_mask = _mm_set1_epi8(0xF);
    mm256_set_m128i(_mm_and_si128(low_mask, tmph), _mm_and_si128(low_mask, tmpl))
}

// AVX version: two 128-bit shuffles (lines 180-193 of quants.c)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_bits_32(x: *const u8) -> __m256i {
    let mut x32: u32 = 0;
    std::ptr::copy_nonoverlapping(x, &mut x32 as *mut u32 as *mut u8, 4);
    let v = _mm_set1_epi32(x32 as i32);
    let shuf_l = _mm_set_epi64x(0x0101010101010101u64 as i64, 0x0000000000000000u64 as i64);
    let shuf_h = _mm_set_epi64x(0x0303030303030303u64 as i64, 0x0202020202020202u64 as i64);
    let bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfeu64 as i64);
    let mut bl = _mm_shuffle_epi8(v, shuf_l);
    let mut bh = _mm_shuffle_epi8(v, shuf_h);
    bl = _mm_or_si128(bl, bit_mask);
    bh = _mm_or_si128(bh, bit_mask);
    bl = _mm_cmpeq_epi8(bl, _mm_set1_epi64x(-1i64));
    bh = _mm_cmpeq_epi8(bh, _mm_set1_epi64x(-1i64));
    mm256_set_m128i(bh, bl)
}

// SSSE3 sign+maddubs on 128-bit (lines 173-177 of quants.c)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_add_epi8_sse(x: __m128i, y: __m128i) -> __m128i {
    _mm_maddubs_epi16(_mm_sign_epi8(x, x), _mm_sign_epi8(y, x))
}

// AVX sum_i16_pairs_float: takes (hi, lo) 128-bit halves (lines 208-215)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sum_i16_pairs_float(xh: __m128i, xl: __m128i) -> __m256 {
    let ones = _mm_set1_epi16(1);
    _mm256_cvtepi32_ps(mm256_set_m128i(_mm_madd_epi16(ones, xh), _mm_madd_epi16(ones, xl)))
}

// AVX mul_sum_us8_pairs_float: split 256->2x128 (lines 217-226)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let axl = _mm256_castsi256_si128(ax);
    let axh = _mm256_extractf128_si256(ax, 1);
    let syl = _mm256_castsi256_si128(sy);
    let syh = _mm256_extractf128_si256(sy, 1);
    sum_i16_pairs_float(_mm_maddubs_epi16(axh, syh), _mm_maddubs_epi16(axl, syl))
}

// AVX mul_sum_i8_pairs_float: split 256->2x128, sign trick (lines 229-244)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let xl = _mm256_castsi256_si128(x);
    let xh = _mm256_extractf128_si256(x, 1);
    let yl = _mm256_castsi256_si128(y);
    let yh = _mm256_extractf128_si256(y, 1);
    let axl = _mm_sign_epi8(xl, xl);
    let axh = _mm_sign_epi8(xh, xh);
    sum_i16_pairs_float(
        _mm_maddubs_epi16(axh, _mm_sign_epi8(yh, xh)),
        _mm_maddubs_epi16(axl, _mm_sign_epi8(yl, xl)),
    )
}

// mul_sum_i8_quad_float: 4 pairs of 128-bit -> __m256 (lines 247-262)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_i8_quad_float(
    x10: __m128i, x11: __m128i, x20: __m128i, x21: __m128i,
    y10: __m128i, y11: __m128i, y20: __m128i, y21: __m128i,
) -> __m256 {
    let one = _mm_set1_epi16(1);
    let p0 = _mm_madd_epi16(mul_add_epi8_sse(x10, y10), one);
    let p1 = _mm_madd_epi16(mul_add_epi8_sse(x11, y11), one);
    let p2 = _mm_madd_epi16(mul_add_epi8_sse(x20, y20), one);
    let p3 = _mm_madd_epi16(mul_add_epi8_sse(x21, y21), one);
    _mm256_cvtepi32_ps(mm256_set_m128i(_mm_add_epi32(p2, p3), _mm_add_epi32(p0, p1)))
}

// quad_fp16_delta_float: (x0,y0)->lo, (x1,y1)->hi (lines 265-269)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn quad_fp16_delta_float(x0: u16, y0: u16, x1: u16, y1: u16) -> __m256 {
    _mm256_set_m128(
        _mm_set1_ps(f16_to_f32(x1) * f16_to_f32(y1)),
        _mm_set1_ps(f16_to_f32(x0) * f16_to_f32(y0)),
    )
}

// quad_mx_delta_float: E8M0 x, f16 y (lines 271-273)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn quad_mx_delta_float(x0: u8, y0: u16, x1: u8, y1: u16) -> __m256 {
    _mm256_set_m128(
        _mm_set1_ps(e8m0_to_f32_half(x1) * f16_to_f32(y1)),
        _mm_set1_ps(e8m0_to_f32_half(x0) * f16_to_f32(y0)),
    )
}

// get_scale_shuffle: K_SHUFFLE table, 128-byte, 16-byte chunks (lines 540-552)
static K_SHUFFLE: [u8; 128] = [
     0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
     2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,
     4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,
     6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,
     8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
];

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    _mm_loadu_si128(K_SHUFFLE.as_ptr().add(i * 16) as *const __m128i)
}

// signs64: cast KEVEN_SIGNS_Q2XS as u64 array
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn signs64(idx: usize) -> u64 {
    *(KEVEN_SIGNS_Q2XS.as_ptr().add(idx * 8) as *const u64)
}

// -- ggml_vec_dot_q1_0_q8_0 -- AVX section, line 607 --------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q1_0_q8_0_avx(
    n: usize,
    x: &[BlockQ1_0],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / 128; // QK1_0 = 128
    let ones8 = _mm_set1_epi8(1);
    let ones16 = _mm_set1_epi16(1);
    let zero = _mm_setzero_si128();
    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d0 = f16_to_f32(x[ib].d);
        let y_base = ib * 4;
        // K=0
        let bit_mask = bytes_from_bits_32(x[ib].qs.as_ptr());
        let bm0 = _mm256_castsi256_si128(bit_mask);
        let bm1 = _mm256_extractf128_si256(bit_mask, 1);
        let qy0 = _mm_loadu_si128(y[y_base].qs.as_ptr() as *const __m128i);
        let qy1 = _mm_loadu_si128(y[y_base].qs.as_ptr().add(16) as *const __m128i);
        let sm0 = _mm_cmpeq_epi8(bm0, zero);
        let sm1 = _mm_cmpeq_epi8(bm1, zero);
        let sy0 = _mm_sub_epi8(_mm_xor_si128(qy0, sm0), sm0);
        let sy1 = _mm_sub_epi8(_mm_xor_si128(qy1, sm1), sm1);
        let s0 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, sy0), ones16);
        let s1 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, sy1), ones16);
        let q = _mm256_cvtepi32_ps(mm256_set_m128i(s1, s0));
        let mut acc_block = _mm256_mul_ps(_mm256_set1_ps(f16_to_f32(y[y_base].d)), q);
        // K=1..3
        for k in 1..4usize {
            let bit_mask = bytes_from_bits_32(x[ib].qs.as_ptr().add(k * 4));
            let bm0 = _mm256_castsi256_si128(bit_mask);
            let bm1 = _mm256_extractf128_si256(bit_mask, 1);
            let qy0 = _mm_loadu_si128(y[y_base + k].qs.as_ptr() as *const __m128i);
            let qy1 = _mm_loadu_si128(y[y_base + k].qs.as_ptr().add(16) as *const __m128i);
            let sm0 = _mm_cmpeq_epi8(bm0, zero);
            let sm1 = _mm_cmpeq_epi8(bm1, zero);
            let sy0 = _mm_sub_epi8(_mm_xor_si128(qy0, sm0), sm0);
            let sy1 = _mm_sub_epi8(_mm_xor_si128(qy1, sm1), sm1);
            let s0 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, sy0), ones16);
            let s1 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, sy1), ones16);
            let q = _mm256_cvtepi32_ps(mm256_set_m128i(s1, s0));
            acc_block = _mm256_add_ps(acc_block, _mm256_mul_ps(_mm256_set1_ps(f16_to_f32(y[y_base + k].d)), q));
        }
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d0), acc_block));
    }
    hsum_float_8(acc)
}

// -- ggml_vec_dot_q4_0_q8_0 -- AVX section, line 742 --------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q4_0_q8_0_avx(
    n: usize,
    x: &[BlockQ4_0],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / 32;
    let mut ib = 0usize;
    let mut sumf = 0.0f32;
    let mut accum = _mm256_setzero_ps();

    while ib + 1 < nb {
        let q4bits_1 = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let q4bits_2 = _mm_loadu_si128(x[ib + 1].qs.as_ptr() as *const __m128i);
        let q8b_1_0 = _mm_loadu_si128(y[ib].qs.as_ptr() as *const __m128i);
        let q8b_1_1 = _mm_loadu_si128(y[ib].qs.as_ptr().add(16) as *const __m128i);
        let q8b_2_0 = _mm_loadu_si128(y[ib + 1].qs.as_ptr() as *const __m128i);
        let q8b_2_1 = _mm_loadu_si128(y[ib + 1].qs.as_ptr().add(16) as *const __m128i);
        let m15 = _mm_set1_epi8(15);
        let q4b_1_0 = _mm_sub_epi8(_mm_and_si128(m15, q4bits_1), _mm_set1_epi8(8));
        let q4b_1_1 = _mm_sub_epi8(_mm_and_si128(m15, _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        let q4b_2_0 = _mm_sub_epi8(_mm_and_si128(m15, q4bits_2), _mm_set1_epi8(8));
        let q4b_2_1 = _mm_sub_epi8(_mm_and_si128(m15, _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));
        let p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
        let p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
        let p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
        let p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
        let p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
        let p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
        let p = sum_i16_pairs_float(p_2, p_1);
        let deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        ib += 2;
    }
    sumf = hsum_float_8(accum);
    // scalar tail
    while ib < nb {
        let mut s0 = 0i32; let mut s1 = 0i32;
        for j in 0..16 {
            let v0 = (x[ib].qs[j] & 0x0F) as i32 - 8;
            let v1 = (x[ib].qs[j] >> 4) as i32 - 8;
            s0 += v0 * y[ib].qs[j] as i32;
            s1 += v1 * y[ib].qs[j + 16] as i32;
        }
        sumf += (s0 + s1) as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
        ib += 1;
    }
    sumf
}

// -- ggml_vec_dot_mxfp4_q8_0 -- AVX section, line 965 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_mxfp4_q8_0_avx(
    n: usize,
    x: &[BlockMxfp4],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / 32; // QK_MXFP4 = 32
    let mut ib = 0usize;
    let mut sumf = 0.0f32;
    let values128 = _mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mut accum = _mm256_setzero_ps();

    while ib + 1 < nb {
        let q4bits_1 = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let q4bits_2 = _mm_loadu_si128(x[ib + 1].qs.as_ptr() as *const __m128i);
        let q8b_1_0 = _mm_loadu_si128(y[ib].qs.as_ptr() as *const __m128i);
        let q8b_1_1 = _mm_loadu_si128(y[ib].qs.as_ptr().add(16) as *const __m128i);
        let q8b_2_0 = _mm_loadu_si128(y[ib + 1].qs.as_ptr() as *const __m128i);
        let q8b_2_1 = _mm_loadu_si128(y[ib + 1].qs.as_ptr().add(16) as *const __m128i);
        let q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        let q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        let q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        let q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
        let p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        let deltas = quad_mx_delta_float(x[ib].e, y[ib].d, x[ib + 1].e, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        ib += 2;
    }
    sumf = hsum_float_8(accum);
    while ib < nb {
        let d = e8m0_to_f32_half(x[ib].e) * f16_to_f32(y[ib].d);
        let mut s1 = 0i32; let mut s2 = 0i32;
        for j in 0..16 {
            s1 += y[ib].qs[j] as i32 * KVALUES_MXFP4[(x[ib].qs[j] & 0xf) as usize] as i32;
            s2 += y[ib].qs[j + 16] as i32 * KVALUES_MXFP4[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (s1 + s2) as f32;
        ib += 1;
    }
    sumf
}

// -- ggml_vec_dot_nvfp4_q8_0 -- AVX section, line 1065 ------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_nvfp4_q8_0_avx(
    n: usize,
    x: &[BlockNvfp4],
    y: &[BlockQ8_0],
) -> f32 {
    // QK_NVFP4 = 64, each nvfp4 block pairs with 2 q8_0 blocks
    let nb = n / 64;
    let mut ib = 0usize;
    let mut sumf = 0.0f32;
    let values128 = _mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mut accum = _mm256_setzero_ps();

    while ib < nb {
        let q4bits_01 = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let q4bits_23 = _mm_loadu_si128(x[ib].qs.as_ptr().add(16) as *const __m128i);
        let q8_0 = _mm_loadu_si128(y[2*ib].qs.as_ptr() as *const __m128i);
        let q8_1 = _mm_loadu_si128(y[2*ib].qs.as_ptr().add(16) as *const __m128i);
        let q8_2 = _mm_loadu_si128(y[2*ib+1].qs.as_ptr() as *const __m128i);
        let q8_3 = _mm_loadu_si128(y[2*ib+1].qs.as_ptr().add(16) as *const __m128i);
        let q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        let q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        let q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        let q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));
        let q4_0 = _mm_unpacklo_epi64(q4_01_lo, q4_01_hi);
        let q4_1 = _mm_unpackhi_epi64(q4_01_lo, q4_01_hi);
        let q4_2 = _mm_unpacklo_epi64(q4_23_lo, q4_23_hi);
        let q4_3 = _mm_unpackhi_epi64(q4_23_lo, q4_23_hi);
        let p0 = _mm_cvtepi32_ps(mul_sum_i8_pairs_128(q4_0, q8_0));
        let p1 = _mm_cvtepi32_ps(mul_sum_i8_pairs_128(q4_1, q8_1));
        let p2 = _mm_cvtepi32_ps(mul_sum_i8_pairs_128(q4_2, q8_2));
        let p3 = _mm_cvtepi32_ps(mul_sum_i8_pairs_128(q4_3, q8_3));
        let dy0 = f16_to_f32(y[2*ib].d);
        let dy1 = f16_to_f32(y[2*ib+1].d);
        let s0 = ue4m3_to_f32_pub(x[ib].d[0]) * dy0;
        let s1 = ue4m3_to_f32_pub(x[ib].d[1]) * dy0;
        let s2 = ue4m3_to_f32_pub(x[ib].d[2]) * dy1;
        let s3 = ue4m3_to_f32_pub(x[ib].d[3]) * dy1;
        let p01 = _mm256_set_m128(p1, p0);
        let p23 = _mm256_set_m128(p3, p2);
        let sc01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        let sc23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));
        accum = _mm256_add_ps(accum, _mm256_mul_ps(p01, sc01));
        accum = _mm256_add_ps(accum, _mm256_mul_ps(p23, sc23));
        ib += 1;
    }
    sumf = hsum_float_8(accum);
    sumf
}

// Helper: SSE mul_sum_i8_pairs producing __m128i (for nvfp4)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_i8_pairs_128(x: __m128i, y: __m128i) -> __m128i {
    let ax = _mm_sign_epi8(x, x);
    let sy = _mm_sign_epi8(y, x);
    let dot = _mm_maddubs_epi16(ax, sy);
    _mm_madd_epi16(_mm_set1_epi16(1), dot)
}

// -- ggml_vec_dot_q5_0_q8_0 -- AVX section, line 1182 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q5_0_q8_0_avx(
    n: usize,
    x: &[BlockQ5_0],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / 32;
    let mut acc = _mm256_setzero_ps();
    let mask = _mm_set1_epi8(-16i8); // 0xF0

    for ib in 0..nb {
        let d = _mm256_set1_ps(f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));
        let bx_0 = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let bxhi = bytes_from_bits_32(x[ib].qh.as_ptr());
        let bxhil = _mm_andnot_si128(_mm256_castsi256_si128(bxhi), mask);
        let bxhih = _mm_andnot_si128(_mm256_extractf128_si256(bxhi, 1), mask);
        let bxl = _mm_or_si128(_mm256_castsi256_si128(bx_0), bxhil);
        let bxh = _mm_or_si128(_mm256_extractf128_si256(bx_0, 1), bxhih);
        let bx_0 = mm256_set_m128i(bxh, bxl);
        let by_0 = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float(bx_0, by_0);
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }
    hsum_float_8(acc)
}

// -- ggml_vec_dot_q5_1_q8_1 -- AVX section, line 1265 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q5_1_q8_1_avx(
    n: usize,
    x: &[BlockQ5_1],
    y: &[BlockQ8_1],
) -> f32 {
    let nb = n / 32;
    let mut acc = _mm256_setzero_ps();
    let mask = _mm_set1_epi8(0x10);
    let mut summs = 0.0f32;

    for ib in 0..nb {
        let dx = _mm256_set1_ps(f16_to_f32(x[ib].d));
        summs += f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);
        let bx_0 = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let bxhi = bytes_from_bits_32(x[ib].qh.as_ptr());
        let bxhil = _mm_and_si128(_mm256_castsi256_si128(bxhi), mask);
        let bxhih = _mm_and_si128(_mm256_extractf128_si256(bxhi, 1), mask);
        let bxl = _mm_or_si128(_mm256_castsi256_si128(bx_0), bxhil);
        let bxh = _mm_or_si128(_mm256_extractf128_si256(bx_0, 1), bxhih);
        let bx_0 = mm256_set_m128i(bxh, bxl);
        let dy = _mm256_set1_ps(f16_to_f32(y[ib].d));
        let by_0 = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);
        let q = mul_sum_us8_pairs_float(bx_0, by_0);
        acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
    }
    hsum_float_8(acc) + summs
}

// -- ggml_vec_dot_q8_0_q8_0 -- AVX section, line 1343 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q8_0_q8_0_avx(
    n: usize,
    x: &[BlockQ8_0],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / 32;
    let mut ib = 0usize;
    let mut sumf = 0.0f32;
    let mut accum = _mm256_setzero_ps();

    while ib + 1 < nb {
        let qx_1_0 = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let qx_1_1 = _mm_loadu_si128(x[ib].qs.as_ptr().add(16) as *const __m128i);
        let qx_2_0 = _mm_loadu_si128(x[ib+1].qs.as_ptr() as *const __m128i);
        let qx_2_1 = _mm_loadu_si128(x[ib+1].qs.as_ptr().add(16) as *const __m128i);
        let qy_1_0 = _mm_loadu_si128(y[ib].qs.as_ptr() as *const __m128i);
        let qy_1_1 = _mm_loadu_si128(y[ib].qs.as_ptr().add(16) as *const __m128i);
        let qy_2_0 = _mm_loadu_si128(y[ib+1].qs.as_ptr() as *const __m128i);
        let qy_2_1 = _mm_loadu_si128(y[ib+1].qs.as_ptr().add(16) as *const __m128i);
        let p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        let deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib+1].d, y[ib+1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        ib += 2;
    }
    sumf = hsum_float_8(accum);
    while ib < nb {
        let mut s = 0i32;
        for j in 0..32 { s += x[ib].qs[j] as i32 * y[ib].qs[j] as i32; }
        sumf += s as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
        ib += 1;
    }
    sumf
}

// -- ggml_vec_dot_q2_K_q8_K -- AVX section, line 1652 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q2_k_q8_k_avx(
    n: usize,
    x: &[BlockQ2K],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let m3 = _mm_set1_epi8(0x3);
    let m4 = _mm_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let dall = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);
        let mut q2 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();

        let mas = _mm_loadu_si128(x[i].scales.as_ptr() as *const __m128i);
        let scales16 = _mm_and_si128(mas, m4);
        let mins16 = _mm_and_si128(_mm_srli_epi16(mas, 4), m4);
        let mins_0 = _mm_cvtepi8_epi16(mins16);
        let mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));
        let bsums_ptr = y[i].bsums.as_ptr();
        let summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128(bsums_ptr as *const __m128i));
        let summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128(bsums_ptr.add(8) as *const __m128i));
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(mm256_set_m128i(summs_1, summs_0))), acc);

        let scales_0 = _mm_cvtepi8_epi16(scales16);
        let scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        let scales = [scales_0, scales_1];
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();

        for j in 0..(QK_K / 128) {
            let q8_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_4 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_5 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_6 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_7 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let mut q2bits = _mm_loadu_si128(q2 as *const __m128i); q2 = q2.add(16);
            let q2_0 = _mm_and_si128(q2bits, m3);
            let q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            let q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            let q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128(q2 as *const __m128i); q2 = q2.add(16);
            let q2_1 = _mm_and_si128(q2bits, m3);
            let q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            let q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            let q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            let m2 = _mm_set1_epi8(0x2);
            let mut p0 = _mm_maddubs_epi16(q2_0, q8_0);
            let mut p1 = _mm_maddubs_epi16(q2_1, q8_1);
            let mut p2 = _mm_maddubs_epi16(q2_2, q8_2);
            let mut p3 = _mm_maddubs_epi16(q2_3, q8_3);
            let mut p4 = _mm_maddubs_epi16(q2_4, q8_4);
            let mut p5 = _mm_maddubs_epi16(q2_5, q8_5);
            let mut p6 = _mm_maddubs_epi16(q2_6, q8_6);
            let mut p7 = _mm_maddubs_epi16(q2_7, q8_7);
            let mut sh = _mm_set1_epi16(0x0100i16);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p0); sh = _mm_add_epi16(sh, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p1); sh = _mm_add_epi16(sh, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p2); sh = _mm_add_epi16(sh, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p3); sh = _mm_add_epi16(sh, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p4); sh = _mm_add_epi16(sh, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p5); sh = _mm_add_epi16(sh, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p6); sh = _mm_add_epi16(sh, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p7);
            p0 = _mm_add_epi32(p0, p1); p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5); p6 = _mm_add_epi32(p6, p7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }
        let sumi = mm256_set_m128i(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dall), _mm256_cvtepi32_ps(sumi)), acc);
    }
    hsum_float_8(acc)
}

// -- ggml_vec_dot_q3_K_q8_K -- AVX section, line 1886 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q3_k_q8_k_avx(
    n: usize,
    x: &[BlockQ3K],
    y: &[BlockQ8K],
) -> f32 {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let nb = n / QK_K;
    let m3 = _mm_set1_epi8(3);
    let mone = _mm_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);
    let m2 = _mm_set1_epi8(2);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let mut q3 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let mut aux = [0u32; 3];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), aux.as_mut_ptr() as *mut u8, 12);
        let mut scales128 = _mm_set_epi32(
            (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
            (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
            ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
            ((aux[0] & KMASK2) | (((aux[2] >> 0) & KMASK1) << 4)) as i32,
        );
        scales128 = _mm_sub_epi8(scales128, m32);
        let scales_0 = _mm_cvtepi8_epi16(scales128);
        let scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        let scales = [scales_0, scales_1];
        let hbits_0 = _mm_loadu_si128(x[i].hmask.as_ptr() as *const __m128i);
        let hbits_1 = _mm_loadu_si128(x[i].hmask.as_ptr().add(16) as *const __m128i);
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();

        for j in 0..(QK_K / 128) {
            let bit = (j * 4) as i32;
            let q3bits_0 = _mm_loadu_si128(q3 as *const __m128i); q3 = q3.add(16);
            let q3bits_1 = _mm_loadu_si128(q3 as *const __m128i); q3 = q3.add(16);
            let mk = |h: __m128i, b: i32| {
                let bv = _mm_cvtsi32_si128(b);
                _mm_slli_epi16(_mm_srl_epi16(_mm_andnot_si128(h, _mm_sll_epi16(mone, bv)), bv), 2)
            };
            let q3h_0 = mk(hbits_0, bit);   let q3h_1 = mk(hbits_1, bit);
            let q3h_2 = mk(hbits_0, bit+1); let q3h_3 = mk(hbits_1, bit+1);
            let q3h_4 = mk(hbits_0, bit+2); let q3h_5 = mk(hbits_1, bit+2);
            let q3h_6 = mk(hbits_0, bit+3); let q3h_7 = mk(hbits_1, bit+3);
            let q3l_0 = _mm_and_si128(q3bits_0, m3);
            let q3l_1 = _mm_and_si128(q3bits_1, m3);
            let q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            let q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            let q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            let q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            let q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            let q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            let q8_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_4 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_5 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_6 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_7 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let mut p16_0 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_0, q8_0), _mm_maddubs_epi16(q3h_0, q8_0));
            let mut p16_1 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_1, q8_1), _mm_maddubs_epi16(q3h_1, q8_1));
            let mut p16_2 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_2, q8_2), _mm_maddubs_epi16(q3h_2, q8_2));
            let mut p16_3 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_3, q8_3), _mm_maddubs_epi16(q3h_3, q8_3));
            let mut p16_4 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_4, q8_4), _mm_maddubs_epi16(q3h_4, q8_4));
            let mut p16_5 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_5, q8_5), _mm_maddubs_epi16(q3h_5, q8_5));
            let mut p16_6 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_6, q8_6), _mm_maddubs_epi16(q3h_6, q8_6));
            let mut p16_7 = _mm_sub_epi16(_mm_maddubs_epi16(q3l_7, q8_7), _mm_maddubs_epi16(q3h_7, q8_7));
            let mut sh = _mm_set1_epi16(0x0100i16);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_0); sh = _mm_add_epi16(sh, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_1); sh = _mm_add_epi16(sh, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_2); sh = _mm_add_epi16(sh, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_3); sh = _mm_add_epi16(sh, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_4); sh = _mm_add_epi16(sh, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_5); sh = _mm_add_epi16(sh, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_6); sh = _mm_add_epi16(sh, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], sh), p16_7);
            p16_0 = _mm_add_epi32(p16_0, p16_1); p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5); p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));
        }
        let sumi = mm256_set_m128i(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    hsum_float_8(acc)
}

// -- ggml_vec_dot_q4_K_q8_K -- AVX section, line 2122 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q4_k_q8_k_avx(
    n: usize,
    x: &[BlockQ4K],
    y: &[BlockQ8K],
) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let nb = n / QK_K;
    let m4 = _mm_set1_epi8(0xF);
    let m2 = _mm_set1_epi8(0x2);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        let utmps = _mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32);
        let scales = _mm_cvtepu8_epi16(utmps);
        let mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
        let q8sums_0 = _mm_loadu_si128(y[i].bsums.as_ptr() as *const __m128i);
        let q8sums_1 = _mm_loadu_si128(y[i].bsums.as_ptr().add(8) as *const __m128i);
        let q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        let prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);
        let mut q4 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();
        let mut shuffle = _mm_set1_epi16(0x0100i16);
        for _j in 0..(QK_K / 64) {
            let scale_l = _mm_shuffle_epi8(scales, shuffle); shuffle = _mm_add_epi16(shuffle, m2);
            let scale_h = _mm_shuffle_epi8(scales, shuffle); shuffle = _mm_add_epi16(shuffle, m2);
            let mut q4bits = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4l_0 = _mm_and_si128(q4bits, m4);
            let q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4l_1 = _mm_and_si128(q4bits, m4);
            let q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            let q8l_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let mut p16l = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_0, q8l_0));
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            let q8l_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            p16l = _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_1, q8l_1));
            sumi_1 = _mm_add_epi32(sumi_1, p16l);
            let q8h_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let mut p16h = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_0, q8h_0));
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            let q8h_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            p16h = _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_1, q8h_1));
            sumi_1 = _mm_add_epi32(sumi_1, p16h);
        }
        let sumi = mm256_set_m128i(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

// -- ggml_vec_dot_q5_K_q8_K -- AVX section, line 2314 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q5_k_q8_k_avx(
    n: usize,
    x: &[BlockQ5K],
    y: &[BlockQ8K],
) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let nb = n / QK_K;
    let m4 = _mm_set1_epi8(0xF);
    let mzero = _mm_setzero_si128();
    let mone = _mm_set1_epi8(1);
    let m2 = _mm_set1_epi8(2);
    let mut acc = _mm256_setzero_ps();
    let mut summs = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        let utmps = _mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32);
        let scales = _mm_cvtepu8_epi16(utmps);
        let mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
        let q8sums_0 = _mm_loadu_si128(y[i].bsums.as_ptr() as *const __m128i);
        let q8sums_1 = _mm_loadu_si128(y[i].bsums.as_ptr().add(8) as *const __m128i);
        let q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        let prod = _mm_madd_epi16(mins, q8s);
        let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0) as f32;
        let hbits_0 = _mm_loadu_si128(x[i].qh.as_ptr() as *const __m128i);
        let hbits_1 = _mm_loadu_si128(x[i].qh.as_ptr().add(16) as *const __m128i);
        let mut hmask = mone;
        let mut q5 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();
        let mut bit = 0i32;
        let mut shuffle = _mm_set1_epi16(0x0100i16);
        for _j in 0..(QK_K / 64) {
            let scale_0 = _mm_shuffle_epi8(scales, shuffle); shuffle = _mm_add_epi16(shuffle, m2);
            let scale_1 = _mm_shuffle_epi8(scales, shuffle); shuffle = _mm_add_epi16(shuffle, m2);
            let q5bits_0 = _mm_loadu_si128(q5 as *const __m128i); q5 = q5.add(16);
            let q5bits_1 = _mm_loadu_si128(q5 as *const __m128i); q5 = q5.add(16);
            let q5l_0 = _mm_and_si128(q5bits_0, m4);
            let q5l_1 = _mm_and_si128(q5bits_1, m4);
            let bv = _mm_cvtsi32_si128(bit);
            let q5h_0 = _mm_slli_epi16(_mm_srl_epi16(_mm_and_si128(hbits_0, hmask), bv), 4);
            let q5h_1 = _mm_slli_epi16(_mm_srl_epi16(_mm_and_si128(hbits_1, hmask), bv), 4);
            let q5_0 = _mm_add_epi8(q5l_0, q5h_0);
            let q5_1 = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1); bit += 1;
            let q8_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5_0, q8_0)));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5_1, q8_1)));
            let q5l_2 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            let q5l_3 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            let bv2 = _mm_cvtsi32_si128(bit);
            let q5h_2 = _mm_slli_epi16(_mm_srl_epi16(_mm_and_si128(hbits_0, hmask), bv2), 4);
            let q5h_3 = _mm_slli_epi16(_mm_srl_epi16(_mm_and_si128(hbits_1, hmask), bv2), 4);
            let q5_2 = _mm_add_epi8(q5l_2, q5h_2);
            let q5_3 = _mm_add_epi8(q5l_3, q5h_3);
            hmask = _mm_slli_epi16(hmask, 1); bit += 1;
            let q8_2 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5_2, q8_2)));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5_3, q8_3)));
        }
        let sumi = mm256_set_m128i(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    hsum_float_8(acc) + summs
}

// -- ggml_vec_dot_q6_K_q8_K -- AVX section, line 2510 -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_q6_k_q8_k_avx(
    n: usize,
    x: &[BlockQ6K],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let m3 = _mm_set1_epi8(3);
    let m15 = _mm_set1_epi8(15);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let mut q4 = x[i].ql.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let q8sums_0 = _mm_loadu_si128(y[i].bsums.as_ptr() as *const __m128i);
        let q8sums_1 = _mm_loadu_si128(y[i].bsums.as_ptr().add(8) as *const __m128i);
        let scales = _mm_loadu_si128(x[i].scales.as_ptr() as *const __m128i);
        let scales_16_0 = _mm_cvtepi8_epi16(scales);
        let scales_16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales, 8));
        let q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        let q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();
        let mut is = 0usize;

        for _j in 0..(QK_K / 128) {
            let q4bh_0 = _mm_loadu_si128(qh as *const __m128i); qh = qh.add(16);
            let q4bh_1 = _mm_loadu_si128(qh as *const __m128i); qh = qh.add(16);
            let q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bh_0, m3), 4);
            let q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bh_1, m3), 4);
            let q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bh_0, _mm_set1_epi8(12)), 2);
            let q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bh_1, _mm_set1_epi8(12)), 2);
            let q4h_4 = _mm_and_si128(q4bh_0, _mm_set1_epi8(48));
            let q4h_5 = _mm_and_si128(q4bh_1, _mm_set1_epi8(48));
            let q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bh_0, _mm_set1_epi8(-64i8)), 2);
            let q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bh_1, _mm_set1_epi8(-64i8)), 2);
            let q4b1_0 = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4b1_1 = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4b2_0 = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4b2_1 = _mm_loadu_si128(q4 as *const __m128i); q4 = q4.add(16);
            let q4_0 = _mm_or_si128(_mm_and_si128(q4b1_0, m15), q4h_0);
            let q4_1 = _mm_or_si128(_mm_and_si128(q4b1_1, m15), q4h_1);
            let q4_2 = _mm_or_si128(_mm_and_si128(q4b2_0, m15), q4h_2);
            let q4_3 = _mm_or_si128(_mm_and_si128(q4b2_1, m15), q4h_3);
            let q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4b1_0, 4), m15), q4h_4);
            let q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4b1_1, 4), m15), q4h_5);
            let q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4b2_0, 4), m15), q4h_6);
            let q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4b2_1, 4), m15), q4h_7);
            let q8_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_4 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_5 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_6 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_7 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let sc0 = get_scale_shuffle(is+0); let sc1 = get_scale_shuffle(is+1);
            let sc2 = get_scale_shuffle(is+2); let sc3 = get_scale_shuffle(is+3);
            is += 4;
            let p0 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, sc0)), _mm_maddubs_epi16(q4_0, q8_0));
            let p1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, sc0), 8)), _mm_maddubs_epi16(q4_1, q8_1));
            let p2 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, sc1)), _mm_maddubs_epi16(q4_2, q8_2));
            let p3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, sc1), 8)), _mm_maddubs_epi16(q4_3, q8_3));
            let p4 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, sc2)), _mm_maddubs_epi16(q4_4, q8_4));
            let p5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, sc2), 8)), _mm_maddubs_epi16(q4_5, q8_5));
            let p6 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, sc3)), _mm_maddubs_epi16(q4_6, q8_6));
            let p7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, sc3), 8)), _mm_maddubs_epi16(q4_7, q8_7));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p1, p3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p4, p6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p5, p7));
        }
        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        let sumi = mm256_set_m128i(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }
    hsum_float_8(acc)
}

// -- ggml_vec_dot_iq2_xxs_q8_K -- AVX section, line 2715 ----------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq2_xxs_q8_k_avx(
    n: usize,
    x: &[BlockIQ2XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let mut accumf = _mm256_setzero_ps();
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q2 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();
        let mut aux32 = [0u32; 4];
        for ib32 in (0..(QK_K/32)).step_by(2) {
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            std::ptr::copy_nonoverlapping(q2.add(ib32 * 2) as *const u8, aux32.as_mut_ptr() as *mut u8, 16);
            let aux8 = aux32.as_ptr() as *const u8;
            let q2_1_0 = _mm_set_epi64x(IQ2XXS_GRID[*aux8.add(1) as usize] as i64, IQ2XXS_GRID[*aux8 as usize] as i64);
            let q2_1_1 = _mm_set_epi64x(IQ2XXS_GRID[*aux8.add(3) as usize] as i64, IQ2XXS_GRID[*aux8.add(2) as usize] as i64);
            let q2_2_0 = _mm_set_epi64x(IQ2XXS_GRID[*aux8.add(9) as usize] as i64, IQ2XXS_GRID[*aux8.add(8) as usize] as i64);
            let q2_2_1 = _mm_set_epi64x(IQ2XXS_GRID[*aux8.add(11) as usize] as i64, IQ2XXS_GRID[*aux8.add(10) as usize] as i64);
            let s2_1_0 = _mm_set_epi64x(signs64((aux32[1] >> 7) as usize & 127) as i64, signs64((aux32[1] >> 0) as usize & 127) as i64);
            let s2_1_1 = _mm_set_epi64x(signs64((aux32[1] >> 21) as usize & 127) as i64, signs64((aux32[1] >> 14) as usize & 127) as i64);
            let s2_2_0 = _mm_set_epi64x(signs64((aux32[3] >> 7) as usize & 127) as i64, signs64((aux32[3] >> 0) as usize & 127) as i64);
            let s2_2_1 = _mm_set_epi64x(signs64((aux32[3] >> 21) as usize & 127) as i64, signs64((aux32[3] >> 14) as usize & 127) as i64);
            let q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            let q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            let q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            let q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            let dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            let dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            let ls1 = (aux32[1] >> 28) as i16;
            let ls2 = (aux32[3] >> 28) as i16;
            let ls1v = _mm_set1_epi16(2 * ls1 + 1);
            let ls2v = _mm_set1_epi16(2 * ls2 + 1);
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, ls1v));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, ls1v));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, ls2v));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, ls2v));
        }
        let sumi = mm256_set_m128i(
            _mm_add_epi32(sumi1_1, sumi2_1),
            _mm_add_epi32(sumi1_0, sumi2_0),
        );
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), accumf);
    }
    0.125 * hsum_float_8(accumf)
}

// -- ggml_vec_dot_iq2_xs_q8_K -- AVX section, line 2909 -----------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq2_xs_q8_k_avx(
    n: usize,
    x: &[BlockIQ2XS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let mone = _mm_set1_epi8(1);
    let block_sign_shuffle_1: [u8; 16] = [0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2];
    let block_sign_shuffle_2: [u8; 16] = [4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6];
    let bss1_0 = _mm_loadu_si128(block_sign_shuffle_1.as_ptr() as *const __m128i);
    // The shuffle masks in C are 32 bytes split in two 128-bit halves
    // We use indices 0..15 for first half, 16..31 for second half
    let bss1_lo: [u8;16] = [0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2];
    let bss1_hi: [u8;16] = [4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6];
    let bss2_lo: [u8;16] = [8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10];
    let bss2_hi: [u8;16] = [12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14];
    let bss1_0_lo = _mm_loadu_si128(bss1_lo.as_ptr() as *const __m128i);
    let bss1_0_hi = _mm_loadu_si128(bss1_hi.as_ptr() as *const __m128i);
    let bss2_0_lo = _mm_loadu_si128(bss2_lo.as_ptr() as *const __m128i);
    let bss2_0_hi = _mm_loadu_si128(bss2_hi.as_ptr() as *const __m128i);
    let bit_sel_bytes: [u8; 16] = [1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128];
    let bsm0 = _mm_loadu_si128(bit_sel_bytes.as_ptr() as *const __m128i);
    let bsm1 = bsm0;
    let k_bit_helper: [u8; 16] = [0,128,128,0,128,0,0,128,128,0,0,128,0,128,128,0];
    let bh0 = _mm_loadu_si128(k_bit_helper.as_ptr() as *const __m128i);
    let bh1 = bh0;
    let m511 = _mm_set1_epi16(511);
    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);
    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut q2 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let mut aux64: u64 = 0;
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), &mut aux64 as *mut u64 as *mut u8, 8);
        let stmp = _mm_set1_epi64x(aux64 as i64);
        let stmp2 = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        let scales = _mm_add_epi8(_mm_slli_epi16(stmp2, 1), m1);
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();

        for ib32 in (0..(QK_K/32)).step_by(4) {
            let q2d0 = _mm_loadu_si128(q2 as *const __m128i);
            let q2d1 = _mm_loadu_si128(q2.add(8) as *const __m128i);
            q2 = q2.add(16);
            // gindices: packed into a __m256i for indexing
            let gi0 = _mm_and_si128(q2d0, m511);
            let gi1 = _mm_and_si128(q2d1, m511);
            // We need 16 u16 gindices total
            let gidx: [u16; 16] = {
                let mut arr = [0u16; 16];
                std::ptr::copy_nonoverlapping(&gi0 as *const __m128i as *const u16, arr.as_mut_ptr(), 8);
                std::ptr::copy_nonoverlapping(&gi1 as *const __m128i as *const u16, arr.as_mut_ptr().add(8), 8);
                arr
            };
            let psb0 = _mm_srli_epi16(q2d0, 9);
            let psb1 = _mm_srli_epi16(q2d1, 9);
            let psu0 = _mm_srli_epi16(q2d0, 13);
            let psu1 = _mm_srli_epi16(q2d1, 13);
            let psfc0 = _mm_xor_si128(psb0, psu0);
            let psfc1 = _mm_xor_si128(psb1, psu1);
            let ob0 = _mm_shuffle_epi8(bh0, psfc0);
            let ob1 = _mm_shuffle_epi8(bh1, psfc1);
            let fsb0 = _mm_or_si128(psb0, ob0);
            let fsb1 = _mm_or_si128(psb1, ob1);
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_3_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_4_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_4_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q2_1_0 = _mm_set_epi64x(IQ2XS_GRID[gidx[1] as usize] as i64, IQ2XS_GRID[gidx[0] as usize] as i64);
            let q2_1_1 = _mm_set_epi64x(IQ2XS_GRID[gidx[3] as usize] as i64, IQ2XS_GRID[gidx[2] as usize] as i64);
            let q2_2_0 = _mm_set_epi64x(IQ2XS_GRID[gidx[5] as usize] as i64, IQ2XS_GRID[gidx[4] as usize] as i64);
            let q2_2_1 = _mm_set_epi64x(IQ2XS_GRID[gidx[7] as usize] as i64, IQ2XS_GRID[gidx[6] as usize] as i64);
            let q2_3_0 = _mm_set_epi64x(IQ2XS_GRID[gidx[9] as usize] as i64, IQ2XS_GRID[gidx[8] as usize] as i64);
            let q2_3_1 = _mm_set_epi64x(IQ2XS_GRID[gidx[11] as usize] as i64, IQ2XS_GRID[gidx[10] as usize] as i64);
            let q2_4_0 = _mm_set_epi64x(IQ2XS_GRID[gidx[13] as usize] as i64, IQ2XS_GRID[gidx[12] as usize] as i64);
            let q2_4_1 = _mm_set_epi64x(IQ2XS_GRID[gidx[15] as usize] as i64, IQ2XS_GRID[gidx[14] as usize] as i64);
            let mut s0 = _mm_shuffle_epi8(fsb0, bss1_0_lo);
            let mut s1 = _mm_shuffle_epi8(fsb0, bss1_0_hi);
            s0 = _mm_cmpeq_epi8(_mm_and_si128(s0, bsm0), bsm0);
            s1 = _mm_cmpeq_epi8(_mm_and_si128(s1, bsm1), bsm1);
            let q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(s0, mone));
            let q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(s1, mone));
            s0 = _mm_shuffle_epi8(fsb0, bss2_0_lo);
            s1 = _mm_shuffle_epi8(fsb0, bss2_0_hi);
            s0 = _mm_cmpeq_epi8(_mm_and_si128(s0, bsm0), bsm0);
            s1 = _mm_cmpeq_epi8(_mm_and_si128(s1, bsm1), bsm1);
            let q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(s0, mone));
            let q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(s1, mone));
            s0 = _mm_shuffle_epi8(fsb1, bss1_0_lo);
            s1 = _mm_shuffle_epi8(fsb1, bss1_0_hi);
            s0 = _mm_cmpeq_epi8(_mm_and_si128(s0, bsm0), bsm0);
            s1 = _mm_cmpeq_epi8(_mm_and_si128(s1, bsm1), bsm1);
            let q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(s0, mone));
            let q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(s1, mone));
            s0 = _mm_shuffle_epi8(fsb1, bss2_0_lo);
            s1 = _mm_shuffle_epi8(fsb1, bss2_0_hi);
            s0 = _mm_cmpeq_epi8(_mm_and_si128(s0, bsm0), bsm0);
            s1 = _mm_cmpeq_epi8(_mm_and_si128(s1, bsm1), bsm1);
            let q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(s0, mone));
            let q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(s1, mone));
            let dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            let dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            let dot3_0 = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            let dot3_1 = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            let dot4_0 = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            let dot4_1 = _mm_maddubs_epi16(q2_4_1, q8s_4_1);
            let sc1_0 = _mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            let sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)), 8));
            let sc2_0 = _mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            let sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)), 8));
            let sc3_0 = _mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            let sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)), 8));
            let sc4_0 = _mm_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));
            let sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)), 8));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(
            _mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);
    }
    0.125 * hsum_float_8(accumf)
}

// -- ggml_vec_dot_iq2_s_q8_K -- AVX section, line 3160 ------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq2_s_q8_k_avx(
    n: usize,
    x: &[BlockIQ2S],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let k_mask1: [u8;16] = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1];
    let k_mask2: [u8;16] = [1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128];
    let k_mask1_0 = _mm_loadu_si128(k_mask1.as_ptr() as *const __m128i);
    let k_mask1_1 = _mm_loadu_si128([2u8,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3].as_ptr() as *const __m128i);
    let k_mask2_0 = _mm_loadu_si128(k_mask2.as_ptr() as *const __m128i);
    let k_mask2_1 = k_mask2_0;
    let m4 = _mm_set1_epi8(0xf);
    let m1 = _mm_set1_epi8(1);
    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let signs_ptr = x[i].qs.as_ptr().add(QK_K/8) as *const u16;
        let mut q8 = y[i].qs.as_ptr();
        let mut aux64: u64 = 0;
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), &mut aux64 as *mut u64 as *mut u8, 8);
        let stmp = _mm_set_epi64x((aux64 >> 4) as i64, aux64 as i64);
        let scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(stmp, m4), 1), m1);
        let scales16_0 = _mm_cvtepi8_epi16(scales8);
        let scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales8, 8));
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();

        // K_SHUFFLE_K4 for scales lookup
        static K_SHUFFLE_K4: [u8; 256] = [
             0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
             2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
             4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
             6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
             8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
            10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
            12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
            14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
        ];
        let get_k4_shuffle = |i: usize| -> __m128i {
            _mm_loadu_si128(K_SHUFFLE_K4.as_ptr().add(i * 32) as *const __m128i)
        };

        let mut signs = signs_ptr;
        for ib32 in (0..(QK_K/32)).step_by(2) {
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let qh0 = *qh as usize; let qh1 = *qh.add(1) as usize;
            let q2_1_0 = _mm_set_epi64x(IQ2S_GRID[*qs.add(1) as usize | ((qh0 << 6) & 0x300)] as i64, IQ2S_GRID[*qs as usize | ((qh0 << 8) & 0x300)] as i64);
            let q2_1_1 = _mm_set_epi64x(IQ2S_GRID[*qs.add(3) as usize | ((qh0 << 2) & 0x300)] as i64, IQ2S_GRID[*qs.add(2) as usize | ((qh0 << 4) & 0x300)] as i64);
            let q2_2_0 = _mm_set_epi64x(IQ2S_GRID[*qs.add(5) as usize | ((qh1 << 6) & 0x300)] as i64, IQ2S_GRID[*qs.add(4) as usize | ((qh1 << 8) & 0x300)] as i64);
            let q2_2_1 = _mm_set_epi64x(IQ2S_GRID[*qs.add(7) as usize | ((qh1 << 2) & 0x300)] as i64, IQ2S_GRID[*qs.add(6) as usize | ((qh1 << 4) & 0x300)] as i64);
            qs = qs.add(8); qh = qh.add(2);
            let s0 = *signs as u32 | ((*signs.add(1) as u32) << 16);
            let s2 = *signs.add(2) as u32 | ((*signs.add(3) as u32) << 16);
            signs = signs.add(4);
            let mut a0 = _mm_set1_epi32(s0 as i32);
            let mut a1 = a0;
            a0 = _mm_and_si128(_mm_shuffle_epi8(a0, k_mask1_0), k_mask2_0);
            a1 = _mm_and_si128(_mm_shuffle_epi8(a1, k_mask1_1), k_mask2_1);
            let s2_1_0 = _mm_cmpeq_epi8(a0, k_mask2_0);
            let s2_1_1 = _mm_cmpeq_epi8(a1, k_mask2_1);
            let q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            let q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);
            let mut b0 = _mm_set1_epi32(s2 as i32);
            let mut b1 = b0;
            b0 = _mm_and_si128(_mm_shuffle_epi8(b0, k_mask1_0), k_mask2_0);
            b1 = _mm_and_si128(_mm_shuffle_epi8(b1, k_mask1_1), k_mask2_1);
            let s2_2_0 = _mm_cmpeq_epi8(b0, k_mask2_0);
            let s2_2_1 = _mm_cmpeq_epi8(b1, k_mask2_1);
            let q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            let q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);
            let dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            let dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            // scales: use get_k4_shuffle but split lo/hi
            let sh0 = get_k4_shuffle(ib32+0);
            let sh1 = _mm_loadu_si128(K_SHUFFLE_K4.as_ptr().add((ib32+0)*32 + 16) as *const __m128i);
            let sh2 = get_k4_shuffle(ib32+1);
            let sh3 = _mm_loadu_si128(K_SHUFFLE_K4.as_ptr().add((ib32+1)*32 + 16) as *const __m128i);
            let p1_0 = _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, sh0));
            let p1_1 = _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, sh1));
            let p2_0 = _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, sh2));
            let p2_1 = _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, sh3));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(
            _mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);
    }
    0.125 * hsum_float_8(accumf)
}

// -- ggml_vec_dot_iq3_xxs_q8_K -- AVX section, line 3319 ----------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq3_xxs_q8_k_avx(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let gas = x[i].qs.as_ptr().add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();
        let mut aux32 = [0u32; 2];

        for ib32 in (0..(QK_K/32)).step_by(2) {
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q2_1_0 = _mm_set_epi32(IQ3XXS_GRID[*q3.add(3) as usize] as i32, IQ3XXS_GRID[*q3.add(2) as usize] as i32, IQ3XXS_GRID[*q3.add(1) as usize] as i32, IQ3XXS_GRID[*q3 as usize] as i32);
            let q2_1_1 = _mm_set_epi32(IQ3XXS_GRID[*q3.add(7) as usize] as i32, IQ3XXS_GRID[*q3.add(6) as usize] as i32, IQ3XXS_GRID[*q3.add(5) as usize] as i32, IQ3XXS_GRID[*q3.add(4) as usize] as i32);
            q3 = q3.add(8);
            let q2_2_0 = _mm_set_epi32(IQ3XXS_GRID[*q3.add(3) as usize] as i32, IQ3XXS_GRID[*q3.add(2) as usize] as i32, IQ3XXS_GRID[*q3.add(1) as usize] as i32, IQ3XXS_GRID[*q3 as usize] as i32);
            let q2_2_1 = _mm_set_epi32(IQ3XXS_GRID[*q3.add(7) as usize] as i32, IQ3XXS_GRID[*q3.add(6) as usize] as i32, IQ3XXS_GRID[*q3.add(5) as usize] as i32, IQ3XXS_GRID[*q3.add(4) as usize] as i32);
            q3 = q3.add(8);
            std::ptr::copy_nonoverlapping(gas.add(ib32 * 4), aux32.as_mut_ptr() as *mut u8, 8);
            let s2_1_0 = _mm_set_epi64x(signs64((aux32[0] >> 7) as usize & 127) as i64, signs64((aux32[0] >> 0) as usize & 127) as i64);
            let s2_1_1 = _mm_set_epi64x(signs64((aux32[0] >> 21) as usize & 127) as i64, signs64((aux32[0] >> 14) as usize & 127) as i64);
            let s2_2_0 = _mm_set_epi64x(signs64((aux32[1] >> 7) as usize & 127) as i64, signs64((aux32[1] >> 0) as usize & 127) as i64);
            let s2_2_1 = _mm_set_epi64x(signs64((aux32[1] >> 21) as usize & 127) as i64, signs64((aux32[1] >> 14) as usize & 127) as i64);
            let q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            let q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            let q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            let q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            let dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            let dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            let ls1 = (aux32[0] >> 28) as i16;
            let ls2 = (aux32[1] >> 28) as i16;
            let ls1v = _mm_set1_epi16(2 * ls1 + 1);
            let ls2v = _mm_set1_epi16(2 * ls2 + 1);
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, ls1v));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, ls1v));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, ls2v));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, ls2v));
        }
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(
            _mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);
    }
    0.25 * hsum_float_8(accumf)
}

// -- ggml_vec_dot_iq3_s_q8_K -- AVX section, line 3480 ------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq3_s_q8_k_avx(
    n: usize,
    x: &[BlockIQ3S],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let k_mask1: [u8;16] = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1];
    let k_mask2: [u8;16] = [1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128];
    let mask1_0 = _mm_loadu_si128(k_mask1.as_ptr() as *const __m128i);
    let mask1_1 = _mm_loadu_si128([2u8,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3].as_ptr() as *const __m128i);
    let mask2_0 = _mm_loadu_si128(k_mask2.as_ptr() as *const __m128i);
    let mask2_1 = mask2_0;
    let idx_mul_0 = _mm_set_epi32(32, 64, 128, 256);
    let idx_mul_1 = _mm_set_epi32(2, 4, 8, 16);
    let idx_mask = _mm_set1_epi32(256);
    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let signs_ptr = x[i].signs.as_ptr() as *const u16;
        let mut q8 = y[i].qs.as_ptr();
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();
        let mut signs = signs_ptr;

        for ib32 in (0..(QK_K/32)).step_by(2) {
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let qs_tmp = _mm_loadu_si128(qs as *const __m128i);
            let idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
            let idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8));
            qs = qs.add(16);
            let qh0 = *qh as i32; let qh1 = *qh.add(1) as i32; qh = qh.add(2);
            let mut iv0 = _mm_set1_epi32(qh0);
            let mut iv1 = _mm_set1_epi32(qh0);
            let mut iv2 = _mm_set1_epi32(qh1);
            let mut iv3 = _mm_set1_epi32(qh1);
            iv0 = _mm_and_si128(_mm_mullo_epi32(iv0, idx_mul_0), idx_mask);
            iv1 = _mm_and_si128(_mm_mullo_epi32(iv1, idx_mul_1), idx_mask);
            iv2 = _mm_and_si128(_mm_mullo_epi32(iv2, idx_mul_0), idx_mask);
            iv3 = _mm_and_si128(_mm_mullo_epi32(iv3, idx_mul_1), idx_mask);
            iv0 = _mm_or_si128(iv0, _mm_cvtepi16_epi32(idx_l_0));
            iv1 = _mm_or_si128(iv1, _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
            iv2 = _mm_or_si128(iv2, _mm_cvtepi16_epi32(idx_l_1));
            iv3 = _mm_or_si128(iv3, _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));
            let idx: [u32; 16] = {
                let mut arr = [0u32; 16];
                std::ptr::copy_nonoverlapping(&iv0 as *const _ as *const u32, arr.as_mut_ptr(), 4);
                std::ptr::copy_nonoverlapping(&iv1 as *const _ as *const u32, arr.as_mut_ptr().add(4), 4);
                std::ptr::copy_nonoverlapping(&iv2 as *const _ as *const u32, arr.as_mut_ptr().add(8), 4);
                std::ptr::copy_nonoverlapping(&iv3 as *const _ as *const u32, arr.as_mut_ptr().add(12), 4);
                arr
            };
            let q2_1_0 = _mm_set_epi32(IQ3S_GRID[idx[3] as usize] as i32, IQ3S_GRID[idx[2] as usize] as i32, IQ3S_GRID[idx[1] as usize] as i32, IQ3S_GRID[idx[0] as usize] as i32);
            let q2_1_1 = _mm_set_epi32(IQ3S_GRID[idx[7] as usize] as i32, IQ3S_GRID[idx[6] as usize] as i32, IQ3S_GRID[idx[5] as usize] as i32, IQ3S_GRID[idx[4] as usize] as i32);
            let q2_2_0 = _mm_set_epi32(IQ3S_GRID[idx[11] as usize] as i32, IQ3S_GRID[idx[10] as usize] as i32, IQ3S_GRID[idx[9] as usize] as i32, IQ3S_GRID[idx[8] as usize] as i32);
            let q2_2_1 = _mm_set_epi32(IQ3S_GRID[idx[15] as usize] as i32, IQ3S_GRID[idx[14] as usize] as i32, IQ3S_GRID[idx[13] as usize] as i32, IQ3S_GRID[idx[12] as usize] as i32);
            let s0 = *signs as u32 | ((*signs.add(1) as u32) << 16);
            let s2 = *signs.add(2) as u32 | ((*signs.add(3) as u32) << 16);
            signs = signs.add(4);
            let mut a0 = _mm_set1_epi32(s0 as i32); let mut a1 = a0;
            a0 = _mm_and_si128(_mm_shuffle_epi8(a0, mask1_0), mask2_0);
            a1 = _mm_and_si128(_mm_shuffle_epi8(a1, mask1_1), mask2_1);
            let s2_1_0 = _mm_cmpeq_epi8(a0, mask2_0);
            let s2_1_1 = _mm_cmpeq_epi8(a1, mask2_1);
            let q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            let q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);
            let mut b0 = _mm_set1_epi32(s2 as i32); let mut b1 = b0;
            b0 = _mm_and_si128(_mm_shuffle_epi8(b0, mask1_0), mask2_0);
            b1 = _mm_and_si128(_mm_shuffle_epi8(b1, mask1_1), mask2_1);
            let s2_2_0 = _mm_cmpeq_epi8(b0, mask2_0);
            let s2_2_1 = _mm_cmpeq_epi8(b1, mask2_1);
            let q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            let q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);
            let dot1_0 = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot1_1 = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_0 = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            let dot2_1 = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            let ls1 = (x[i].scales[ib32/2] & 0xf) as i16;
            let ls2 = (x[i].scales[ib32/2] >> 4) as i16;
            let ls1v = _mm_set1_epi16(2 * ls1 + 1);
            let ls2v = _mm_set1_epi16(2 * ls2 + 1);
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, ls1v));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, ls1v));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, ls2v));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, ls2v));
        }
        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(
            _mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);
    }
    hsum_float_8(accumf)
}


// -- ggml_vec_dot_iq1_s_q8_K -- AVX section, line 3657 ------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq1_s_q8_k_avx(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut accum = _mm256_setzero_ps();
    let mut accum1 = 0.0f32;
    for i in 0..nb {
        let mut q8 = y[i].qs.as_ptr();
        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr() as *const usize;
        let mut sumi_0 = _mm_setzero_si128();
        let mut sumi_1 = _mm_setzero_si128();
        let mut sumi_s = 0i32;
        for ib in (0..(QK_K/32)).step_by(2) {
            let qh0 = *qh.add(ib) as usize & 0xffff;
            let qh1 = *qh.add(ib+1) as usize & 0xffff;
            let q1b_1_0 = _mm_set_epi64x(IQ1S_GRID[*qs.add(1) as usize | ((qh0 << 5) & 0x700)] as i64, IQ1S_GRID[*qs as usize | ((qh0 << 8) & 0x700)] as i64);
            let q1b_1_1 = _mm_set_epi64x(IQ1S_GRID[*qs.add(3) as usize | ((qh0 >> 1) & 0x700)] as i64, IQ1S_GRID[*qs.add(2) as usize | ((qh0 << 2) & 0x700)] as i64);
            let q1b_2_0 = _mm_set_epi64x(IQ1S_GRID[*qs.add(5) as usize | ((qh1 << 5) & 0x700)] as i64, IQ1S_GRID[*qs.add(4) as usize | ((qh1 << 8) & 0x700)] as i64);
            let q1b_2_1 = _mm_set_epi64x(IQ1S_GRID[*qs.add(7) as usize | ((qh1 >> 1) & 0x700)] as i64, IQ1S_GRID[*qs.add(6) as usize | ((qh1 << 2) & 0x700)] as i64);
            qs = qs.add(8);
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let d1_0 = mul_add_epi8_sse(q1b_1_0, q8_1_0);
            let d1_1 = mul_add_epi8_sse(q1b_1_1, q8_1_1);
            let d2_0 = mul_add_epi8_sse(q1b_2_0, q8_2_0);
            let d2_1 = mul_add_epi8_sse(q1b_2_1, q8_2_1);
            let ls1 = (2 * ((qh0 >> 12) & 7) + 1) as i16;
            let ls2 = (2 * ((qh1 >> 12) & 7) + 1) as i16;
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(_mm_madd_epi16(d1_0, _mm_set1_epi16(ls1)), _mm_madd_epi16(d2_0, _mm_set1_epi16(ls2))));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(_mm_madd_epi16(d1_1, _mm_set1_epi16(ls1)), _mm_madd_epi16(d2_1, _mm_set1_epi16(ls2))));
            sumi_s += (y[i].bsums[2*ib] + y[i].bsums[2*ib+1]) as i32 * (if qh0 & 0x8000 != 0 { -1 } else { 1 }) * ls1 as i32
                    + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) as i32 * (if qh1 & 0x8000 != 0 { -1 } else { 1 }) * ls2 as i32;
        }
        let d = y[i].d * f16_to_f32(x[i].d);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(sumi_1, sumi_0))), accum);
        accum1 += d * sumi_s as f32;
    }
    hsum_float_8(accum) + IQ1S_DELTA * accum1
}

// -- ggml_vec_dot_iq1_m_q8_K -- AVX section, line 3825 ------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq1_m_q8_k_avx(
    n: usize,
    x: &[BlockIQ1M],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let mask = _mm_set1_epi16(0x7);
    let mone = _mm_set1_epi16(1);
    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    for i in 0..nb {
        let mut q8 = y[i].qs.as_ptr();
        let mut qs = x[i].qs.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let sc = x[i].scales.as_ptr() as *const u16;
        let sc0 = *sc; let sc1 = *sc.add(1); let sc2 = *sc.add(2); let sc3 = *sc.add(3);
        let scale_u16 = ((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)) as u16;
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();
        for ib in (0..(QK_K/32)).step_by(2) {
            let qh0 = *qh as usize; let qh1 = *qh.add(1) as usize;
            let qh2 = *qh.add(2) as usize; let qh3 = *qh.add(3) as usize; qh = qh.add(4);
            let q1b_1_0 = _mm_set_epi64x(IQ1S_GRID[*qs.add(1) as usize | ((qh0 << 4) & 0x700)] as i64, IQ1S_GRID[*qs as usize | ((qh0 << 8) & 0x700)] as i64);
            let q1b_1_1 = _mm_set_epi64x(IQ1S_GRID[*qs.add(3) as usize | ((qh1 << 4) & 0x700)] as i64, IQ1S_GRID[*qs.add(2) as usize | ((qh1 << 8) & 0x700)] as i64);
            let q1b_2_0 = _mm_set_epi64x(IQ1S_GRID[*qs.add(5) as usize | ((qh2 << 4) & 0x700)] as i64, IQ1S_GRID[*qs.add(4) as usize | ((qh2 << 8) & 0x700)] as i64);
            let q1b_2_1 = _mm_set_epi64x(IQ1S_GRID[*qs.add(7) as usize | ((qh3 << 4) & 0x700)] as i64, IQ1S_GRID[*qs.add(6) as usize | ((qh3 << 8) & 0x700)] as i64);
            qs = qs.add(8);
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let dot1_0 = mul_add_epi8_sse(q1b_1_0, q8_1_0);
            let dot1_1 = mul_add_epi8_sse(q1b_1_1, q8_1_1);
            let dot2_0 = mul_add_epi8_sse(q1b_2_0, q8_2_0);
            let dot2_1 = mul_add_epi8_sse(q1b_2_1, q8_2_1);
            let delta1_0 = _mm_set_epi64x(if qh0 & 0x80 != 0 { -1i64 } else { 0x0101010101010101i64 }, if qh0 & 0x08 != 0 { -1i64 } else { 0x0101010101010101i64 });
            let delta1_1 = _mm_set_epi64x(if qh1 & 0x80 != 0 { -1i64 } else { 0x0101010101010101i64 }, if qh1 & 0x08 != 0 { -1i64 } else { 0x0101010101010101i64 });
            let delta2_0 = _mm_set_epi64x(if qh2 & 0x80 != 0 { -1i64 } else { 0x0101010101010101i64 }, if qh2 & 0x08 != 0 { -1i64 } else { 0x0101010101010101i64 });
            let delta2_1 = _mm_set_epi64x(if qh3 & 0x80 != 0 { -1i64 } else { 0x0101010101010101i64 }, if qh3 & 0x08 != 0 { -1i64 } else { 0x0101010101010101i64 });
            let dot3_0 = mul_add_epi8_sse(delta1_0, q8_1_0);
            let dot3_1 = mul_add_epi8_sse(delta1_1, q8_1_1);
            let dot4_0 = mul_add_epi8_sse(delta2_0, q8_2_0);
            let dot4_1 = mul_add_epi8_sse(delta2_1, q8_2_1);
            let sc_ib = *sc.add(ib / 2);
            let mut sc1_0 = _mm_set1_epi16((sc_ib >> 0) as i16);
            let mut sc1_1 = _mm_set1_epi16((sc_ib >> 3) as i16);
            let mut sc2_0 = _mm_set1_epi16((sc_ib >> 6) as i16);
            let mut sc2_1 = _mm_set1_epi16((sc_ib >> 9) as i16);
            sc1_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(sc1_0, mask), 1), mone);
            sc1_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(sc1_1, mask), 1), mone);
            sc2_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(sc2_0, mask), 1), mone);
            sc2_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(sc2_1, mask), 1), mone);
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(_mm_madd_epi16(dot1_0, sc1_0), _mm_madd_epi16(dot2_0, sc2_0)));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(_mm_madd_epi16(dot1_1, sc1_1), _mm_madd_epi16(dot2_1, sc2_1)));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_add_epi32(_mm_madd_epi16(dot3_0, sc1_0), _mm_madd_epi16(dot4_0, sc2_0)));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_add_epi32(_mm_madd_epi16(dot3_1, sc1_1), _mm_madd_epi16(dot4_1, sc2_1)));
        }
        let d = y[i].d * f16_to_f32(scale_u16);
        accum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(sumi1_1, sumi1_0))), accum1);
        accum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mm256_set_m128i(sumi2_1, sumi2_0))), accum2);
    }
    hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2)
}

// -- ggml_vec_dot_iq4_nl_q8_0 -- AVX section, line 3966 -----------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq4_nl_q8_0_avx(
    n: usize,
    x: &[BlockIQ4NL],
    y: &[BlockQ8_0],
) -> f32 {
    let nb = n / QK4_NL;
    let mut ib = 0usize;
    let mut sumf = 0.0f32;
    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mut accum = _mm256_setzero_ps();
    while ib + 1 < nb {
        let q4bits_1 = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let q4bits_2 = _mm_loadu_si128(x[ib + 1].qs.as_ptr() as *const __m128i);
        let q8_1_0 = _mm_loadu_si128(y[ib].qs.as_ptr() as *const __m128i);
        let q8_1_1 = _mm_loadu_si128(y[ib].qs.as_ptr().add(16) as *const __m128i);
        let q8_2_0 = _mm_loadu_si128(y[ib + 1].qs.as_ptr() as *const __m128i);
        let q8_2_1 = _mm_loadu_si128(y[ib + 1].qs.as_ptr().add(16) as *const __m128i);
        let q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        let q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        let q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        let q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
        let p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8_1_0, q8_1_1, q8_2_0, q8_2_1);
        let deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
        ib += 2;
    }
    sumf = hsum_float_8(accum);
    while ib < nb {
        let d = f16_to_f32(y[ib].d) * f16_to_f32(x[ib].d);
        let mut s1 = 0i32; let mut s2 = 0i32;
        for j in 0..16 {
            s1 += y[ib].qs[j] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] & 0xf) as usize] as i32;
            s2 += y[ib].qs[j + 16] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (s1 + s2) as f32;
        ib += 1;
    }
    sumf
}

// -- ggml_vec_dot_iq4_xs_q8_K -- AVX section, line 4054 -----------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma,ssse3,sse4.1")]
pub unsafe fn ggml_vec_dot_iq4_xs_q8_k_avx(
    n: usize,
    x: &[BlockIQ4XS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mut accum = _mm256_setzero_ps();
    for ibl in 0..nb {
        let mut qs = x[ibl].qs.as_ptr();
        let mut q8 = y[ibl].qs.as_ptr();
        let mut sh = x[ibl].scales_h;
        let mut sumi1_0 = _mm_setzero_si128();
        let mut sumi1_1 = _mm_setzero_si128();
        let mut sumi2_0 = _mm_setzero_si128();
        let mut sumi2_1 = _mm_setzero_si128();
        for ib in (0..(QK_K/32)).step_by(2) {
            let q4bits_1 = _mm_loadu_si128(qs as *const __m128i); qs = qs.add(16);
            let q4bits_2 = _mm_loadu_si128(qs as *const __m128i); qs = qs.add(16);
            let q8_1_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_1_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_0 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q8_2_1 = _mm_loadu_si128(q8 as *const __m128i); q8 = q8.add(16);
            let q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            let q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            let q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            let q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            let p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8_1_0);
            let p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8_1_1);
            let p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8_2_0);
            let p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8_2_1);
            let ls1 = (((x[ibl].scales_l[ib/2] & 0xf) as i16) | (((sh << 4) as i16) & 0x30)) - 32;
            let ls2 = (((x[ibl].scales_l[ib/2] >> 4) as i16) | (((sh << 2) as i16) & 0x30)) - 32;
            sh >>= 4;
            sumi1_0 = _mm_add_epi32(_mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1)), sumi1_0);
            sumi1_1 = _mm_add_epi32(_mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1)), sumi1_1);
            sumi2_0 = _mm_add_epi32(_mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2)), sumi2_0);
            sumi2_1 = _mm_add_epi32(_mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2)), sumi2_1);
        }
        let sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
        let sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(f16_to_f32(x[ibl].d) * y[ibl].d),
            _mm256_cvtepi32_ps(mm256_set_m128i(sumi12_1, sumi12_0))), accum);
    }
    hsum_float_8(accum)
}

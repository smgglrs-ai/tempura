//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c
//! AVX-512 VNNI paths only -- functions that use mul_sum_us8_pairs_float via dpbusd.
//! The `#if defined(__AVX512VNNI__) && defined(__AVX512VL__)` variant of the helper
//! replaces maddubs+madd with a single _mm256_dpbusd_epi32, benefiting q4_1 and q5_1.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;

// -- Local helpers (same as in quants_llama.rs, private to this module) --------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_bits_32(x: *const u8) -> __m256i {
    let mut x32: u32 = 0;
    std::ptr::copy_nonoverlapping(x, &mut x32 as *mut u32 as *mut u8, 4);
    let shuf_mask = _mm256_set_epi64x(
        0x0303030303030303u64 as i64,
        0x0202020202020202u64 as i64,
        0x0101010101010101u64 as i64,
        0x0000000000000000u64 as i64,
    );
    let mut bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32 as i32), shuf_mask);
    let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeu64 as i64);
    bytes = _mm256_or_si256(bytes, bit_mask);
    _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1i64))
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

// -- VNNI-accelerated mul_sum_us8_pairs_float ----------------------------------
// C: #if defined(__AVX512VNNI__) && defined(__AVX512VL__)
//    const __m256i zero = _mm256_setzero_si256();
//    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
//    return _mm256_cvtepi32_ps(summed_pairs);
//
// ax: unsigned bytes (u8), sy: signed bytes (i8)
// result: horizontal sum of products, 8 f32 lanes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl")]
unsafe fn mul_sum_us8_pairs_float_vnni(ax: __m256i, sy: __m256i) -> __m256 {
    let zero = _mm256_setzero_si256();
    let summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    _mm256_cvtepi32_ps(summed_pairs)
}

// -- ggml_vec_dot_q4_1_q8_1 -- AVX-512 VNNI path -------------------------------
// Same body as the AVX2 translation in quants_llama.rs but uses VNNI helper.
// Source: line 859-916 of quants.c, shared #if defined(__AVX2__) || defined(__AVX__)
// with d0d1 accumulation replaced by add+mul for the shared path, but fmadd is OK
// when compiling with AVX2+AVX512VNNI.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl,fma")]
pub unsafe fn ggml_vec_dot_q4_1_q8_1_vnni(
    n: usize,
    x: &[BlockQ4_1],
    y: &[BlockQ8_1],
) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();
    let mut summs: f32 = 0.0;

    for ib in 0..nb {
        let d0 = f16_to_f32(x[ib].d);
        let d1 = f16_to_f32(y[ib].d);

        summs += f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);

        let d0d1 = _mm256_mul_ps(_mm256_set1_ps(d0), _mm256_set1_ps(d1));

        let qx = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let xy = mul_sum_us8_pairs_float_vnni(qx, qy);

        acc = _mm256_fmadd_ps(d0d1, xy, acc);
    }

    hsum_float_8(acc) + summs
}

// -- ggml_vec_dot_q5_1_q8_1 -- AVX-512 VNNI path -------------------------------
// Source: line 1222-1305 of quants.c, AVX2/AVX shared body.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vnni,avx512vl,fma")]
pub unsafe fn ggml_vec_dot_q5_1_q8_1_vnni(
    n: usize,
    x: &[BlockQ5_1],
    y: &[BlockQ8_1],
) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();
    let mut summs: f32 = 0.0;

    for ib in 0..nb {
        let dx = _mm256_set1_ps(f16_to_f32(x[ib].d));

        summs += f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);

        let mut qx = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let mut bxhi = bytes_from_bits_32(x[ib].qh.as_ptr());
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);

        let dy = _mm256_set1_ps(f16_to_f32(y[ib].d));
        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let q = mul_sum_us8_pairs_float_vnni(qx, qy);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    hsum_float_8(acc) + summs
}

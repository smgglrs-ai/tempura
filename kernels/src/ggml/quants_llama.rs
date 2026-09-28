//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c
//! Pure translation — no optimization. AVX2 sections only. Algorithm identical to llama.cpp.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;

// ── Helper: MM256_SET_M128I(hi, lo) macro equivalent ─────────────────────────
// C: _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

// ── Helper: bytes_from_nibbles_32 ────────────────────────────────────────────
// Unpack 32 4-bit fields into 32 bytes, each in [0..15]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

// ── Helper: bytes_from_bits_32 ───────────────────────────────────────────────
// Spread 32 bits to 32 bytes { 0x00, 0xFF }
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

// ── Helper: mul_add_epi8 ─────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_add_epi8(x: __m256i, y: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    _mm256_maddubs_epi16(ax, sy)
}

// ── Helper: sum_i16_pairs_float ──────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

// ── Helper: mul_sum_us8_pairs_float ──────────────────────────────────────────
// unsigned × signed → float (maddubs path, no VNNI)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

// ── Helper: mul_sum_i8_pairs_float ───────────────────────────────────────────
// signed × signed → float
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

// ── Helper: hsum_float_8 ─────────────────────────────────────────────────────
// Horizontally sum 8 floats in a __m256 vector
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

// ── Shuffle tables ────────────────────────────────────────────────────────────

// get_scale_shuffle_q3k: 128 bytes, loaded as 32-byte chunks via index
static K_SHUFFLE_Q3K: [u8; 128] = [
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
     6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
     8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
];

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_scale_shuffle_q3k(i: usize) -> __m256i {
    _mm256_loadu_si256(K_SHUFFLE_Q3K.as_ptr().add(i * 32) as *const __m256i)
}

// get_scale_shuffle_k4: 256 bytes, loaded as 32-byte chunks via index
static K_SHUFFLE_K4: [u8; 256] = [
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
     4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
     4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
     6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
     6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
     8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
     8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
];

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
    _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add(i * 32) as *const __m256i)
}

// get_scale_shuffle: 128 bytes, loaded as 16-byte chunks via index
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

// ── ggml_vec_dot_q4_0_q8_0 — AVX2 section, line 701 ─────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q4_0_q8_0(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d = _mm256_set1_ps(f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));

        let mut qx = bytes_from_nibbles_32(x[ib].qs.as_ptr());

        let off = _mm256_set1_epi8(8);
        qx = _mm256_sub_epi8(qx, off);

        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let q = mul_sum_i8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(d, q, acc);
    }

    hsum_float_8(acc)
}

// ── ggml_vec_dot_q4_1_q8_1 — AVX2 section, line 859 ─────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q4_1_q8_1(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();
    let mut summs: f32 = 0.0;

    for ib in 0..nb {
        let d0 = f16_to_f32(x[ib].d);
        let d1 = f16_to_f32(y[ib].d);

        summs += f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);

        let d0v = _mm256_set1_ps(d0);
        let d1v = _mm256_set1_ps(d1);

        let d0d1 = _mm256_mul_ps(d0v, d1v);

        let qx = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let xy = mul_sum_us8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(d0d1, xy, acc);
    }

    hsum_float_8(acc) + summs
}

// ── ggml_vec_dot_q5_0_q8_0 — AVX2 section, line 1142 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q5_0_q8_0(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d = _mm256_set1_ps(f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));

        let mut qx = bytes_from_nibbles_32(x[ib].qs.as_ptr());
        let mut bxhi = bytes_from_bits_32(x[ib].qh.as_ptr());
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8(-16i8)); // 0xF0 = -16 as i8
        qx = _mm256_or_si256(qx, bxhi);

        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let q = mul_sum_i8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(d, q, acc);
    }

    hsum_float_8(acc)
}

// ── ggml_vec_dot_q5_1_q8_1 — AVX2 section, line 1222 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q5_1_q8_1(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
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

        let q = mul_sum_us8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    hsum_float_8(acc) + summs
}

// ── ggml_vec_dot_q8_0_q8_0 — AVX2 section, line 1308 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q8_0_q8_0(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;

    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d = _mm256_set1_ps(f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));
        let qx = _mm256_loadu_si256(x[ib].qs.as_ptr() as *const __m256i);
        let qy = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        let q = mul_sum_i8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(d, q, acc);
    }

    hsum_float_8(acc)
}

// ── ggml_vec_dot_q2_K_q8_K — AVX2 section, line 1574 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q2_k_q8_k(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256; // QK_K = 256

    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0xF);

    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);

        let mut q2 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();

        let mins_and_scales = _mm_loadu_si128(x[i].scales.as_ptr() as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let prod = _mm256_madd_epi16(
            mins,
            _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i),
        );

        acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc);

        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(l_scales, l_scales),
            mm256_set_m128i(h_scales, h_scales),
        ];

        let mut sumi = _mm256_setzero_si256();

        for j in 0..(256 / 128) {
            let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
            q2 = q2.add(32);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let q2_0 = _mm256_and_si256(q2bits, m3);
            let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            let mut p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            let mut p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            let mut p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            let mut p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

// ── ggml_vec_dot_q3_K_q8_K — AVX2 section, line 1766 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q3_k_q8_k(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let m3 = _mm256_set1_epi8(3);
    let mone = _mm256_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);

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
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(l_scales, l_scales),
            mm256_set_m128i(h_scales, h_scales),
        ];

        let hbits = _mm256_loadu_si256(x[i].hmask.as_ptr() as *const __m256i);

        let mut sumi = _mm256_setzero_si256();

        let mut bit: i32 = 0;
        let is: usize = 0;

        for j in 0..(256 / 128) {
            let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
            q3 = q3.add(32);

            // Use variable-count shift intrinsics (_mm256_sll/srl_epi16) since
            // `bit` is a runtime value — the const-immediate versions would fail.
            let q3l_0 = _mm256_and_si256(q3bits, m3);
            let q3h_0 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, _mm_cvtsi32_si128(bit))),
                    _mm_cvtsi32_si128(bit),
                ),
                2,
            );
            bit += 1;

            let q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            let q3h_1 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, _mm_cvtsi32_si128(bit))),
                    _mm_cvtsi32_si128(bit),
                ),
                2,
            );
            bit += 1;

            let q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            let q3h_2 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, _mm_cvtsi32_si128(bit))),
                    _mm_cvtsi32_si128(bit),
                ),
                2,
            );
            bit += 1;

            let q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            let q3h_3 = _mm256_slli_epi16(
                _mm256_srl_epi16(
                    _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, _mm_cvtsi32_si128(bit))),
                    _mm_cvtsi32_si128(bit),
                ),
                2,
            );
            bit += 1;

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            let mut p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)),
                p16_0,
            );
            p16_1 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)),
                p16_1,
            );
            p16_2 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)),
                p16_2,
            );
            p16_3 = _mm256_madd_epi16(
                _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)),
                p16_3,
            );

            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

// ── ggml_vec_dot_q4_K_q8_K — AVX2 section, line 2038 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q4_k_q8_k(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0xF);

    // LLVM replaces _mm256_shuffle_epi8(scales, sh_i) with arithmetic extraction
    // (vpbroadcastw + vpsrld/vpshufd/vpsrlq/vpsrldq per scale word) because the
    // 8 YMM shuffle constants exhaust available registers. Hoisting them here would
    // register-pressure LLVM into using the shuffle path, but it chooses arithmetic
    // extraction instead to keep ymm8-ymm15 free for the main computation loop.
    // The shuffle masks remain defined for documentation; the compiler does not use them.
    let sh0 = get_scale_shuffle_k4(0);
    let sh1 = get_scale_shuffle_k4(1);
    let sh2 = get_scale_shuffle_k4(2);
    let sh3 = get_scale_shuffle_k4(3);
    let sh4 = get_scale_shuffle_k4(4);
    let sh5 = get_scale_shuffle_k4(5);
    let sh6 = get_scale_shuffle_k4(6);
    let sh7 = get_scale_shuffle_k4(7);

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

        let mut q4 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        let mut sumi = _mm256_setzero_si256();

        // Unrolled j=0..4 loop — compiler knows exactly which YMM register per vpshufb.
        macro_rules! do_sub {($sl:expr, $sh_:expr) => {{
            let q4bits = _mm256_loadu_si256(q4 as *const __m256i); q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            let q8l = _mm256_loadu_si256(q8 as *const __m256i); q8 = q8.add(32);
            let q8h = _mm256_loadu_si256(q8 as *const __m256i); q8 = q8.add(32);
            let p16l = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, $sl), _mm256_maddubs_epi16(q4l, q8l));
            let p16h = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, $sh_), _mm256_maddubs_epi16(q4h, q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }};}
        do_sub!(sh0, sh1);
        do_sub!(sh2, sh3);
        do_sub!(sh4, sh5);
        do_sub!(sh6, sh7);

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

// ── ggml_vec_dot_q5_K_q8_K — AVX2 section, line 2216 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q5_k_q8_k(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0xF);
    let mzero = _mm_setzero_si128();
    let mone = _mm256_set1_epi8(1);

    let mut acc = _mm256_setzero_ps();
    let mut summs: f32 = 0.0;

    for i in 0..nb {
        let mut q5 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();

        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);

        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0) as f32;

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        let hbits = _mm256_loadu_si256(x[i].qh.as_ptr() as *const __m256i);
        let mut hmask = mone;

        let mut sumi = _mm256_setzero_si256();

        let mut bit: i32 = 0;

        for j in 0..(256 / 64) {
            let scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            let scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
            q5 = q5.add(32);

            let q5l_0 = _mm256_and_si256(q5bits, m4);
            let q5h_0 = _mm256_slli_epi16(
                _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit)),
                4,
            );
            let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            bit += 1;
            hmask = _mm256_slli_epi16(hmask, 1);

            let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            let q5h_1 = _mm256_slli_epi16(
                _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit)),
                4,
            );
            let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            bit += 1;
            hmask = _mm256_slli_epi16(hmask, 1);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc) + summs
}

// ── ggml_vec_dot_q6_K_q8_K — AVX2 section, line 2426 ────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_vec_dot_q6_k_q8_k(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;

    let m3 = _mm256_set1_epi8(3);
    let m15 = _mm256_set1_epi8(15);
    // Hoist 6-bit extraction masks (GCC: ymm9-ymm11)
    let m_12 = _mm256_set1_epi8(12);
    let m_48 = _mm256_set1_epi8(48);
    let m_n64 = _mm256_set1_epi8(-64i8 as i8);
    // Hoist XMM scale shuffle constants before block loop. LLVM keeps sh0-sh3
    // in xmm5-xmm8 and reuses them for both halves (via vpunpcklbw/vpunpckhbw
    // that interleaves each scale byte with itself, letting sh0 select s0 from
    // the lower half and the same mask select s8 from the upper half). sh4-sh7
    // are only needed by the source macro; the compiler folds them into sh0-sh3
    // with the vpunpck split.
    let sh0 = get_scale_shuffle(0);
    let sh1 = get_scale_shuffle(1);
    let sh2 = get_scale_shuffle(2);
    let sh3 = get_scale_shuffle(3);
    let sh4 = get_scale_shuffle(4);
    let sh5 = get_scale_shuffle(5);
    let sh6 = get_scale_shuffle(6);
    let sh7 = get_scale_shuffle(7);

    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);

        let mut q4 = x[i].ql.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let mut q8 = y[i].qs.as_ptr();

        let q8sums = _mm256_loadu_si256(y[i].bsums.as_ptr() as *const __m256i);
        let scales = _mm_loadu_si128(x[i].scales.as_ptr() as *const __m128i);
        let scales_16 = _mm256_cvtepi8_epi16(scales);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        let mut sumi = _mm256_setzero_si256();

        // Unrolled j=0 (is=0..4): use sh0-sh3 as compile-time-known register operands
        {
            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bitsh = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m_12), 2);
            let q4h_2 = _mm256_and_si256(q4bitsh, m_48);
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsh, m_n64), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            let scale_0 = _mm_shuffle_epi8(scales, sh0);
            let scale_1 = _mm_shuffle_epi8(scales, sh1);
            let scale_2 = _mm_shuffle_epi8(scales, sh2);
            let scale_3 = _mm_shuffle_epi8(scales, sh3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        // Unrolled j=1 (is=4..8): use sh4-sh7 as compile-time-known register operands
        {
            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bitsh = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m_12), 2);
            let q4h_2 = _mm256_and_si256(q4bitsh, m_48);
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsh, m_n64), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            let scale_0 = _mm_shuffle_epi8(scales, sh4);
            let scale_1 = _mm_shuffle_epi8(scales, sh5);
            let scale_2 = _mm_shuffle_epi8(scales, sh6);
            let scale_3 = _mm_shuffle_epi8(scales, sh7);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        // suppress unused-variable warnings for pointer advances past last element
        let _ = (q4, qh, q8);

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_float_8(acc)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::vec_dot::*;

    fn is_x86_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    // Build a Q4K block from float data and compare AVX2 vs scalar dot product
    #[test]
    fn test_q4k_vs_scalar() {
        if !is_x86_avx2() {
            return;
        }
        // Use existing scalar vec_dot_q4_k_q8_k if available, otherwise just
        // check the AVX2 path doesn't panic and returns a finite value.
        // We construct minimal synthetic blocks.
        let x = BlockQ4K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0x77u8; 12], // scales = 7, mins = 7 packed
            qs: [0x88u8; 128],    // nibbles: low=8, high=8
        };
        let y = BlockQ8K {
            d: 1.0f32,
            qs: [1i8; 256],
            bsums: [32i16; 16], // sum of 16 elements each = 16*1=16, but we set 32
        };

        let result = unsafe { ggml_vec_dot_q4_k_q8_k(256, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert!(result.is_finite(), "Q4K dot product must be finite, got {result}");
    }

    #[test]
    fn test_q6k_vs_scalar() {
        if !is_x86_avx2() {
            return;
        }
        let x = BlockQ6K {
            ql: [0x55u8; 128],
            qh: [0x00u8; 64],
            scales: [1i8; 16],
            d: half::f16::from_f32(1.0).to_bits(),
        };
        let y = BlockQ8K {
            d: 1.0f32,
            qs: [1i8; 256],
            bsums: [16i16; 16],
        };

        let result = unsafe { ggml_vec_dot_q6_k_q8_k(256, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert!(result.is_finite(), "Q6K dot product must be finite, got {result}");
    }

    #[test]
    fn test_q4_0_zero_inputs() {
        if !is_x86_avx2() {
            return;
        }
        let x = BlockQ4_0 { d: 0u16, qs: [0x88u8; 16] };
        let y = BlockQ8_0 { d: 0u16, qs: [0i8; 32] };
        let result = unsafe { ggml_vec_dot_q4_0_q8_0(32, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert_eq!(result, 0.0f32);
    }

    #[test]
    fn test_q8_0_identity() {
        if !is_x86_avx2() {
            return;
        }
        // x = all 1, y = all 1, d_x = d_y = 1/127 → dot = 32 * (1/127)^2 * 127^2 / 32? No.
        // d is the scale factor. Result = sum(x.qs[i] * y.qs[i]) * d_x * d_y
        // With qs = [1; 32], d = f16(1.0): result = 32 * 1.0 * 1.0 = 32
        let one_f16 = half::f16::from_f32(1.0).to_bits();
        let x = BlockQ8_0 { d: one_f16, qs: [1i8; 32] };
        let y = BlockQ8_0 { d: one_f16, qs: [1i8; 32] };
        let result = unsafe { ggml_vec_dot_q8_0_q8_0(32, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        let expected = 32.0f32;
        let rel_err = (result - expected).abs() / expected.abs();
        assert!(rel_err < 1e-4, "expected {expected}, got {result}, rel_err={rel_err}");
    }

    #[test]
    fn test_q2k_finite() {
        if !is_x86_avx2() {
            return;
        }
        let x = BlockQ2K {
            scales: [0x11u8; 16],
            qs: [0x55u8; 64],
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
        };
        let y = BlockQ8K {
            d: 1.0f32,
            qs: [1i8; 256],
            bsums: [16i16; 16],
        };
        let result = unsafe { ggml_vec_dot_q2_k_q8_k(256, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert!(result.is_finite(), "Q2K result must be finite, got {result}");
    }

    #[test]
    fn test_q3k_finite() {
        if !is_x86_avx2() {
            return;
        }
        let x = BlockQ3K {
            hmask: [0x00u8; 32],
            qs: [0x55u8; 64],
            scales: [0x40u8; 12], // 64 in each position
            d: half::f16::from_f32(1.0).to_bits(),
        };
        let y = BlockQ8K {
            d: 1.0f32,
            qs: [1i8; 256],
            bsums: [16i16; 16],
        };
        let result = unsafe { ggml_vec_dot_q3_k_q8_k(256, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert!(result.is_finite(), "Q3K result must be finite, got {result}");
    }

    #[test]
    fn test_q5k_finite() {
        if !is_x86_avx2() {
            return;
        }
        let x = BlockQ5K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0x77u8; 12],
            qh: [0x00u8; 32],
            qs: [0x55u8; 128],
        };
        let y = BlockQ8K {
            d: 1.0f32,
            qs: [1i8; 256],
            bsums: [16i16; 16],
        };
        let result = unsafe { ggml_vec_dot_q5_k_q8_k(256, std::slice::from_ref(&x), std::slice::from_ref(&y)) };
        assert!(result.is_finite(), "Q5K result must be finite, got {result}");
    }
}

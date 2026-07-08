//! TL2 (Ternary LUT 2-bit) kernel — direct translation of bitnet.cpp's codegen_tl2.py output.
//!
//! Architecture:
//!   1. `per_tensor_quant`: quantize activations to i8 (scale = 127/max_abs)
//!   2. `three_lut_ctor`: build 16-entry LUT for triplets of activations, transpose+pack
//!   3. `two_lut_ctor`: same for pairs
//!   4. `three_tbl_impl`: tiled GEMM: shuffle-based LUT lookup + sign correction, i16 accum
//!   5. `two_tbl_impl`: same without sign bits
//!   6. `preprocessor`: quantize + build LUTs
//!   7. `qgemm_lut`: dispatch to tiled GEMM

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── AVX2 helpers (translated from bitnet.cpp) ──────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mm256_merge_epi32(v0: __m256i, v1: __m256i) -> (__m256i, __m256i) {
    let va = _mm256_permute4x64_epi64(v0, 0b11_01_10_00);
    let vb = _mm256_permute4x64_epi64(v1, 0b11_01_10_00);
    (_mm256_unpacklo_epi32(va, vb), _mm256_unpackhi_epi32(va, vb))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mm256_merge_epi64(v0: __m256i, v1: __m256i) -> (__m256i, __m256i) {
    let va = _mm256_permute4x64_epi64(v0, 0b11_01_10_00);
    let vb = _mm256_permute4x64_epi64(v1, 0b11_01_10_00);
    (_mm256_unpacklo_epi64(va, vb), _mm256_unpackhi_epi64(va, vb))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mm256_merge_si128(v0: __m256i, v1: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256(v0, v1, 0x20),
        _mm256_permute2x128_si256(v0, v1, 0x31),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8_8(v: &mut [__m256i; 8]) {
    let (w0, w1) = mm256_merge_epi32(v[0], v[1]);
    let (w2, w3) = mm256_merge_epi32(v[2], v[3]);
    let (w4, w5) = mm256_merge_epi32(v[4], v[5]);
    let (w6, w7) = mm256_merge_epi32(v[6], v[7]);
    let (x0, x1) = mm256_merge_epi64(w0, w2);
    let (x2, x3) = mm256_merge_epi64(w1, w3);
    let (x4, x5) = mm256_merge_epi64(w4, w6);
    let (x6, x7) = mm256_merge_epi64(w5, w7);
    let (r0, r1) = mm256_merge_si128(x0, x4);
    let (r2, r3) = mm256_merge_si128(x1, x5);
    let (r4, r5) = mm256_merge_si128(x2, x6);
    let (r6, r7) = mm256_merge_si128(x3, x7);
    v[0] = r0; v[1] = r1; v[2] = r2; v[3] = r3;
    v[4] = r4; v[5] = r5; v[6] = r6; v[7] = r7;
}

// ── per_tensor_quant ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn per_tensor_quant(k: usize, b: &[f32]) -> f32 {
    let mut max_vec = _mm256_setzero_ps();
    let sign = _mm256_set1_ps(-0.0);
    for i in 0..k / 8 {
        let v = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        max_vec = _mm256_max_ps(max_vec, _mm256_andnot_ps(sign, v));
    }
    let hi = _mm256_extractf128_ps(max_vec, 1);
    let lo = _mm256_castps256_ps128(max_vec);
    let max1 = _mm_max_ps(hi, lo);
    let max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    let max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    127.0 / _mm_cvtss_f32(max1)
}

// ── three_lut_ctor ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn three_lut_ctor(qlut: &mut [i8], b: &[f32], scale: f32, act_k: usize) {
    let vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
    let scale_vec = _mm256_set1_ps(scale);
    let shuffle_mask = _mm256_set_epi8(
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
    );
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    for k in 0..act_k / 24 {
        let b0 = _mm256_i32gather_ps(b.as_ptr().add(k * 24), vec_bi, 1);
        let b1 = _mm256_i32gather_ps(b.as_ptr().add(k * 24 + 1), vec_bi, 1);
        let b2 = _mm256_i32gather_ps(b.as_ptr().add(k * 24 + 2), vec_bi, 1);

        let b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b0, scale_vec), ROUND));
        let b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b1, scale_vec), ROUND));
        let b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b2, scale_vec), ROUND));

        let zero = _mm256_setzero_si256();
        let mut lut = [zero; 16];
        lut[15] = zero;
        lut[14] = zero;
        lut[13] = _mm256_add_epi32(_mm256_add_epi32(b0i, b1i), b2i);
        lut[12] = _mm256_add_epi32(b0i, b1i);
        lut[11] = _mm256_sub_epi32(_mm256_add_epi32(b0i, b1i), b2i);
        lut[10] = _mm256_add_epi32(b0i, b2i);
        lut[9] = b0i;
        lut[8] = _mm256_sub_epi32(b0i, b2i);
        lut[7] = _mm256_add_epi32(_mm256_sub_epi32(b0i, b1i), b2i);
        lut[6] = _mm256_sub_epi32(b0i, b1i);
        lut[5] = _mm256_sub_epi32(_mm256_sub_epi32(b0i, b1i), b2i);
        lut[4] = _mm256_add_epi32(b1i, b2i);
        lut[3] = b1i;
        lut[2] = _mm256_sub_epi32(b1i, b2i);
        lut[1] = b2i;
        lut[0] = zero;

        let mut ix = lut;

        let mut lo = [ix[0], ix[1], ix[2], ix[3], ix[4], ix[5], ix[6], ix[7]];
        let mut hi = [ix[8], ix[9], ix[10], ix[11], ix[12], ix[13], ix[14], ix[15]];
        transpose_8_8(&mut lo);
        transpose_8_8(&mut hi);
        ix[0] = lo[0]; ix[1] = lo[1]; ix[2] = lo[2]; ix[3] = lo[3];
        ix[4] = lo[4]; ix[5] = lo[5]; ix[6] = lo[6]; ix[7] = lo[7];
        ix[8] = hi[0]; ix[9] = hi[1]; ix[10] = hi[2]; ix[11] = hi[3];
        ix[12] = hi[4]; ix[13] = hi[5]; ix[14] = hi[6]; ix[15] = hi[7];

        for g in 0..8 {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], 0b11_01_10_00);
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], 0b11_01_10_00);
        }

        for g in 0..8 {
            _mm256_storeu_si256(
                qlut.as_mut_ptr().add(k * 256 + g * 32) as *mut __m256i,
                ix[g],
            );
        }
    }
}

// ── two_lut_ctor ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn two_lut_ctor(qlut: &mut [i8], b: &[f32], scale: f32, act_k: usize) {
    let vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    let scale_vec = _mm256_set1_ps(scale);
    let shuffle_mask = _mm256_set_epi8(
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
        0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
    );
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    for k in 0..act_k / 16 {
        let b0f = _mm256_i32gather_ps(b.as_ptr().add(k * 16), vec_bi, 1);
        let b1f = _mm256_i32gather_ps(b.as_ptr().add(k * 16 + 1), vec_bi, 1);

        let b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b0f, scale_vec), ROUND));
        let b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b1f, scale_vec), ROUND));

        let zero = _mm256_setzero_si256();
        let mut lut = [zero; 16];
        lut[15] = zero; lut[14] = zero; lut[13] = zero;
        lut[12] = zero; lut[11] = zero; lut[10] = zero; lut[9] = zero;
        lut[8] = _mm256_add_epi32(b0, b1);
        lut[7] = b0;
        lut[6] = _mm256_sub_epi32(b0, b1);
        lut[5] = b1;
        lut[4] = zero;
        lut[3] = _mm256_sub_epi32(zero, b1);
        lut[2] = _mm256_add_epi32(_mm256_sub_epi32(zero, b0), b1);
        lut[1] = _mm256_sub_epi32(zero, b0);
        lut[0] = _mm256_sub_epi32(_mm256_sub_epi32(zero, b0), b1);

        let mut ix = lut;

        let mut lo = [ix[0], ix[1], ix[2], ix[3], ix[4], ix[5], ix[6], ix[7]];
        let mut hi = [ix[8], ix[9], ix[10], ix[11], ix[12], ix[13], ix[14], ix[15]];
        transpose_8_8(&mut lo);
        transpose_8_8(&mut hi);
        ix[0] = lo[0]; ix[1] = lo[1]; ix[2] = lo[2]; ix[3] = lo[3];
        ix[4] = lo[4]; ix[5] = lo[5]; ix[6] = lo[6]; ix[7] = lo[7];
        ix[8] = hi[0]; ix[9] = hi[1]; ix[10] = hi[2]; ix[11] = hi[3];
        ix[12] = hi[4]; ix[13] = hi[5]; ix[14] = hi[6]; ix[15] = hi[7];

        for g in 0..8 {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], 0b11_01_10_00);
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], 0b11_01_10_00);
        }

        for g in 0..8 {
            _mm256_storeu_si256(
                qlut.as_mut_ptr().add(k * 256 + g * 32) as *mut __m256i,
                ix[g],
            );
        }
    }
}

// ── three_tbl_impl ─────────────────────────────────────────────────────────
//
// Tiled GEMM: processes BM output rows × BK/3 reduction elements per call.
// For each 32-row tile within BM:
//   - Load weight bytes (packed 4-bit indices) and sign bits
//   - For each batch element:
//     - For each group of 8 elements (KK/8 iterations):
//       - 4 sub-iterations processing 4 weight bytes each
//       - Each sub-iter: extract top/bot nibble, shuffle LUT, apply sign, accumulate i16
//     - Widen i16→i32, store to output

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn three_tbl_impl(
    c: &mut [i32],
    lut: &[i8],
    a: &[u8],
    sign: &[u8],
    bm: usize,
    bk: usize,
    k3: usize,
    batch_size: usize,
) {
    let vec_mask = _mm256_set1_epi8(0x0f);
    let kk = bk / 3;

    for i in (0..bm).step_by(32) {
        let mut vec_as = vec![_mm256_setzero_si256(); kk / 2];
        let mut vec_signs = vec![_mm256_setzero_si256(); kk / 8];

        for ai in 0..kk / 2 {
            vec_as[ai] = _mm256_loadu_si256(
                a.as_ptr().add(i * kk / 2 + ai * 32) as *const __m256i,
            );
        }
        for s in 0..kk / 8 {
            vec_signs[s] = _mm256_loadu_si256(
                sign.as_ptr().add(i * kk / 8 + s * 32) as *const __m256i,
            );
        }

        for bs in 0..batch_size {
            let mut vec_c0 = _mm256_setzero_si256();
            let mut vec_c1 = _mm256_setzero_si256();

            macro_rules! three_tbl_sub {
                ($kg:expr, $j:literal, $shl0:literal, $shl1:literal, $shl2:literal, $shl3:literal) => {{
                    let vec_a = vec_as[$kg * 4 + $j];
                    let lut_off = $kg * 32 * 8 + $j * 64 + k3 / 3 * 32 * bs;
                    let vec_sign = vec_signs[$kg];

                    let k1 = _mm_loadu_si128(lut.as_ptr().add(lut_off) as *const __m128i);
                    let k2 = _mm_loadu_si128(lut.as_ptr().add(lut_off + 16) as *const __m128i);
                    let k3v = _mm_loadu_si128(lut.as_ptr().add(lut_off + 32) as *const __m128i);
                    let k4 = _mm_loadu_si128(lut.as_ptr().add(lut_off + 48) as *const __m128i);

                    let slh = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, $shl0), 15);
                    let sll = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, $shl1), 15);
                    let v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                    let vtf = _mm256_shuffle_epi8(_mm256_set_m128i(k1, k1), v_top);
                    let vts = _mm256_shuffle_epi8(_mm256_set_m128i(k2, k2), v_top);

                    let srh = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, $shl2), 15);
                    let srl = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, $shl3), 15);
                    let v_bot = _mm256_and_si256(vec_a, vec_mask);
                    let vbf = _mm256_shuffle_epi8(_mm256_set_m128i(k3v, k3v), v_bot);
                    let vbs = _mm256_shuffle_epi8(_mm256_set_m128i(k4, k4), v_bot);

                    let vtlo = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vtf, vts), sll), sll);
                    let vthi = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vtf, vts), slh), slh);
                    let vblo = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vbf, vbs), srl), srl);
                    let vbhi = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vbf, vbs), srh), srh);

                    vec_c0 = _mm256_add_epi16(vec_c0, vthi);
                    vec_c0 = _mm256_add_epi16(vec_c0, vbhi);
                    vec_c1 = _mm256_add_epi16(vec_c1, vtlo);
                    vec_c1 = _mm256_add_epi16(vec_c1, vblo);
                }};
            }

            for kg in 0..kk / 8 {
                three_tbl_sub!(kg, 0, 0, 1, 2, 3);
                three_tbl_sub!(kg, 1, 4, 5, 6, 7);
                three_tbl_sub!(kg, 2, 8, 9, 10, 11);
                three_tbl_sub!(kg, 3, 12, 13, 14, 15);
            }

            // Widen i16→i32 and accumulate into output
            let off = i + bm * bs;
            let mut gc0 = _mm256_loadu_si256(c.as_ptr().add(off) as *const __m256i);
            let mut gc1 = _mm256_loadu_si256(c.as_ptr().add(off + 8) as *const __m256i);
            let mut gc2 = _mm256_loadu_si256(c.as_ptr().add(off + 16) as *const __m256i);
            let mut gc3 = _mm256_loadu_si256(c.as_ptr().add(off + 24) as *const __m256i);

            gc0 = _mm256_add_epi32(gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
            gc1 = _mm256_add_epi32(gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
            gc2 = _mm256_add_epi32(gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
            gc3 = _mm256_add_epi32(gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

            _mm256_storeu_si256(c.as_mut_ptr().add(off) as *mut __m256i, gc0);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 8) as *mut __m256i, gc1);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 16) as *mut __m256i, gc2);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 24) as *mut __m256i, gc3);
        }
    }
}

// ── two_tbl_impl ───────────────────────────────────────────────────────────

const BK2: usize = 32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn two_tbl_impl(
    c: &mut [i32],
    lut: &[i8],
    a: &[u8],
    bm: usize,
    k2: usize,
    batch_size: usize,
) {
    let vec_mask = _mm256_set1_epi8(0x0f);
    let kk = BK2 / 2;

    for i in (0..bm).step_by(32) {
        let mut vec_as = vec![_mm256_setzero_si256(); kk / 2];
        for ai in 0..kk / 2 {
            vec_as[ai] = _mm256_loadu_si256(
                a.as_ptr().add(i * kk / 2 + ai * 32) as *const __m256i,
            );
        }

        for bs in 0..batch_size {
            let mut vec_c0 = _mm256_setzero_si256();
            let mut vec_c1 = _mm256_setzero_si256();

            for kg in 0..kk / 8 {
                for j in 0..4usize {
                    let vec_a = vec_as[kg * 4 + j];
                    let lut_off = kg * 32 * 8 + j * 64 + k2 / 2 * 32 * bs;

                    let k1 = _mm_loadu_si128(lut.as_ptr().add(lut_off) as *const __m128i);
                    let k2_v = _mm_loadu_si128(lut.as_ptr().add(lut_off + 16) as *const __m128i);
                    let k3 = _mm_loadu_si128(lut.as_ptr().add(lut_off + 32) as *const __m128i);
                    let k4 = _mm_loadu_si128(lut.as_ptr().add(lut_off + 48) as *const __m128i);

                    let v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                    let v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(k1, k1), v_top);
                    let v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(k2_v, k2_v), v_top);

                    let v_bot = _mm256_and_si256(vec_a, vec_mask);
                    let v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(k3, k3), v_bot);
                    let v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(k4, k4), v_bot);

                    // No sign correction needed for two_tbl
                    let v_top_lo = _mm256_unpackhi_epi8(v_top_fir, v_top_sec);
                    let v_top_hi = _mm256_unpacklo_epi8(v_top_fir, v_top_sec);
                    let v_bot_lo = _mm256_unpackhi_epi8(v_bot_fir, v_bot_sec);
                    let v_bot_hi = _mm256_unpacklo_epi8(v_bot_fir, v_bot_sec);

                    vec_c0 = _mm256_add_epi16(vec_c0, v_top_hi);
                    vec_c0 = _mm256_add_epi16(vec_c0, v_bot_hi);
                    vec_c1 = _mm256_add_epi16(vec_c1, v_top_lo);
                    vec_c1 = _mm256_add_epi16(vec_c1, v_bot_lo);
                }
            }

            let off = i + bm * bs;
            let mut gc0 = _mm256_loadu_si256(c.as_ptr().add(off) as *const __m256i);
            let mut gc1 = _mm256_loadu_si256(c.as_ptr().add(off + 8) as *const __m256i);
            let mut gc2 = _mm256_loadu_si256(c.as_ptr().add(off + 16) as *const __m256i);
            let mut gc3 = _mm256_loadu_si256(c.as_ptr().add(off + 24) as *const __m256i);

            gc0 = _mm256_add_epi32(gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
            gc1 = _mm256_add_epi32(gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
            gc2 = _mm256_add_epi32(gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
            gc3 = _mm256_add_epi32(gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

            _mm256_storeu_si256(c.as_mut_ptr().add(off) as *mut __m256i, gc0);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 8) as *mut __m256i, gc1);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 16) as *mut __m256i, gc2);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 24) as *mut __m256i, gc3);
        }
    }
}

// ── Top-level GEMM dispatch ────────────────────────────────────────────────

/// TL2 tile configuration for a specific (m, k) weight shape.
pub struct Tl2Config {
    pub m: usize,
    pub k: usize,
    pub bm: usize,
    pub bk: usize,
    pub three_k: usize,
    pub two_k: usize,
}

impl Tl2Config {
    pub fn new(m: usize, k: usize, bm: usize, bk: usize) -> Self {
        let bk_num = k / bk;
        let three_k = bk_num * bk;
        let two_k = k - three_k;
        Self { m, k, bm, bk, three_k, two_k }
    }

    pub fn for_dims(m: usize, k: usize) -> Option<Self> {
        // bitnet.cpp preset dimension configs (BBK=96 for all x86 TL2)
        // BM must evenly divide m; BK=96 is universal
        let bk = 96usize;
        // Use BM from kernel_config.ini presets, or compute: largest power-of-2
        // factor of m that is <= 320 and >= 32
        let bm = match (m, k) {
            // bitnet_b1_58-3B presets
            (3200, 8640) => 160,
            (3200, 3200) => 320,
            (8640, 3200) => 320,
            // BitNet-b1.58-2B-4T (h=2560, inter=6912, kv_dim=640)
            (640, 2560) => 128,
            (2560, 2560) => 128,
            (2560, 6912) => 128,
            (6912, 2560) => 128,
            _ => {
                // Auto-select: largest factor of m that is 32..=320
                let mut best = 0;
                for candidate in [320, 256, 160, 128, 64, 32] {
                    if m % candidate == 0 { best = candidate; break; }
                }
                if best == 0 { return None; }
                best
            }
        };
        if m % bm != 0 { return None; }
        Some(Self::new(m, k, bm, bk))
    }
}

/// Run a full TL2 GEMV: preprocess activations + tiled GEMM.
/// `a_data` and `sign_data` are sub-regions of the TL2 weight tensor.
/// Thread 0 only — TL2 handles its own tiling internally.
#[cfg(target_arch = "x86_64")]
pub fn tl2_forward(
    activations: &[f32],
    weight_data: &[u8],
    w_scale: f32,
    output: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };

    let n_tile_num = m / cfg.bm;
    let three_k = cfg.three_k;

    // Sign data starts after the three-part weight indices
    let sign_offset = three_k * m / 6;
    if sign_offset > weight_data.len() { return false; }

    let mut lut_scale = 0.0f32;
    let three_lut_size = three_k / 3 * 32;
    let two_lut_size = if cfg.two_k > 0 { cfg.two_k / 2 * 32 } else { 0 };
    let mut three_qlut = vec![0i8; three_lut_size];
    let mut two_qlut = vec![0i8; two_lut_size.max(1)];

    unsafe {
        preprocessor(&cfg, activations, &mut lut_scale, &mut three_qlut, &mut two_qlut, 1);
    }

    for tile in 0..n_tile_num {
        let w_tile_size = three_k * m / 6 / n_tile_num;
        let sign_tile_size = three_k * m / 24 / n_tile_num;
        let c_tile_size = m / n_tile_num;

        let w_off = tile * w_tile_size;
        let sign_off = sign_offset + tile * sign_tile_size;
        let out_off = tile * c_tile_size;

        let tile_out = &mut output[out_off..out_off + cfg.bm];
        unsafe {
            qgemm_lut(
                &cfg,
                &weight_data[w_off..],
                &weight_data[sign_off..],
                &three_qlut,
                w_scale,
                lut_scale,
                tile_out,
                1,
            );
        }
    }
    true
}

/// Preprocess activations: per-tensor quantize + build three/two LUTs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn preprocessor(
    cfg: &Tl2Config,
    b: &[f32],
    lut_scale: &mut f32,
    three_qlut: &mut [i8],
    two_qlut: &mut [i8],
    batch_size: usize,
) {
    for bs in 0..batch_size {
        let b_off = bs * (cfg.three_k + cfg.two_k);
        *lut_scale = per_tensor_quant(cfg.two_k + cfg.three_k, &b[b_off..]);
        three_lut_ctor(
            &mut three_qlut[bs * cfg.three_k / 3 * 32..],
            &b[b_off..],
            *lut_scale,
            cfg.three_k,
        );
        if cfg.two_k > 0 {
            two_lut_ctor(
                &mut two_qlut[bs * cfg.two_k / 2 * 32..],
                &b[b_off + cfg.three_k..],
                *lut_scale,
                cfg.two_k,
            );
        }
    }
}

/// Run tiled GEMM: three_tbl_impl over BK blocks, then two_tbl_impl for remainder.
/// `a` = repacked weights, `sign` = sign bits, `lut` = activation LUT.
/// Output written to `c` as i32, caller rescales by `1/lut_scale * w_scale`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn qgemm_lut(
    cfg: &Tl2Config,
    a: &[u8],
    sign: &[u8],
    lut: &[i8],
    w_scale: f32,
    lut_scale: f32,
    output: &mut [f32],
    batch_size: usize,
) {
    let mut cbits = vec![0i32; batch_size * cfg.bm];

    // Three-valued part
    let n_three_blocks = cfg.three_k / cfg.bk;
    for k_outer in 0..n_three_blocks {
        let lut_off = k_outer * cfg.bk / 3 * 32;
        let a_off = k_outer * cfg.bk / 3 / 2 * cfg.bm;
        let sign_off = k_outer * cfg.bk / 3 / 8 * cfg.bm;
        three_tbl_impl(
            &mut cbits,
            &lut[lut_off..],
            &a[a_off..],
            &sign[sign_off..],
            cfg.bm,
            cfg.bk,
            cfg.three_k,
            batch_size,
        );
    }

    // Two-valued remainder
    if cfg.two_k > 0 {
        let two_k_blocks = cfg.two_k / BK2;
        for k_outer in 0..two_k_blocks {
            let lut_off = k_outer * BK2 / 2 * 32;
            let a_off = k_outer * BK2 / 2 / 2 * cfg.bm;
            two_tbl_impl(
                &mut cbits,
                &lut[lut_off..],
                &a[a_off..],
                cfg.bm,
                cfg.two_k,
                batch_size,
            );
        }
    }

    // Rescale and write output
    for bs in 0..batch_size {
        for i in 0..cfg.bm {
            output[i] = cbits[i + bs * cfg.bm] as f32 / lut_scale * w_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_tensor_quant() {
        let a = vec![1.0f32, -2.0, 0.5, 3.0, -1.5, 0.0, 2.5, -3.0];
        let scale = unsafe { per_tensor_quant(8, &a) };
        // max_abs = 3.0, scale = 127/3 ≈ 42.33
        assert!((scale - 42.333).abs() < 0.1, "scale={scale}");
    }

    #[test]
    fn test_three_lut_ctor_runs() {
        // 24 activations → 1 LUT block (24/24=1)
        let b: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let scale = 127.0 / 1.2; // max_abs = 1.2
        let mut qlut = vec![0i8; 256];
        unsafe { three_lut_ctor(&mut qlut, &b, scale, 24) };
        // Just verify it doesn't crash and produces non-zero output
        assert!(qlut.iter().any(|&v| v != 0), "LUT should have non-zero entries");
    }

    #[test]
    fn test_two_lut_ctor_runs() {
        let b: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let scale = 127.0 / 0.8;
        let mut qlut = vec![0i8; 256];
        unsafe { two_lut_ctor(&mut qlut, &b, scale, 16) };
        assert!(qlut.iter().any(|&v| v != 0), "LUT should have non-zero entries");
    }
}

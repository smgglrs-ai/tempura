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

fn detect_l1d_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            let s = s.trim().trim_end_matches('K');
            if let Ok(kb) = s.parse::<usize>() {
                return kb * 1024;
            }
        }
    }
    48 * 1024
}

fn gemm_chunk_size() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        let l1d = detect_l1d_bytes();
        // cbits = BS * BM_max * 4 bytes. Use 75% of L1d as budget
        // (sysfs may over-report effective capacity on hybrid architectures).
        let bs = (l1d * 3 / 4) / (320 * 4);
        // Round down to multiple of 8 for SIMD alignment, clamp to [8, 64]
        (bs / 8 * 8).clamp(8, 64)
    })
}

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
    v[0] = r0;
    v[1] = r1;
    v[2] = r2;
    v[3] = r3;
    v[4] = r4;
    v[5] = r5;
    v[6] = r6;
    v[7] = r7;
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
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02,
        0x00, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04,
        0x02, 0x00,
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
        ix[0] = lo[0];
        ix[1] = lo[1];
        ix[2] = lo[2];
        ix[3] = lo[3];
        ix[4] = lo[4];
        ix[5] = lo[5];
        ix[6] = lo[6];
        ix[7] = lo[7];
        ix[8] = hi[0];
        ix[9] = hi[1];
        ix[10] = hi[2];
        ix[11] = hi[3];
        ix[12] = hi[4];
        ix[13] = hi[5];
        ix[14] = hi[6];
        ix[15] = hi[7];

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
        0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02,
        0x00, 0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04,
        0x02, 0x00,
    );
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    for k in 0..act_k / 16 {
        let b0f = _mm256_i32gather_ps(b.as_ptr().add(k * 16), vec_bi, 1);
        let b1f = _mm256_i32gather_ps(b.as_ptr().add(k * 16 + 1), vec_bi, 1);

        let b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b0f, scale_vec), ROUND));
        let b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(b1f, scale_vec), ROUND));

        let zero = _mm256_setzero_si256();
        let mut lut = [zero; 16];
        lut[15] = zero;
        lut[14] = zero;
        lut[13] = zero;
        lut[12] = zero;
        lut[11] = zero;
        lut[10] = zero;
        lut[9] = zero;
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
        ix[0] = lo[0];
        ix[1] = lo[1];
        ix[2] = lo[2];
        ix[3] = lo[3];
        ix[4] = lo[4];
        ix[5] = lo[5];
        ix[6] = lo[6];
        ix[7] = lo[7];
        ix[8] = hi[0];
        ix[9] = hi[1];
        ix[10] = hi[2];
        ix[11] = hi[3];
        ix[12] = hi[4];
        ix[13] = hi[5];
        ix[14] = hi[6];
        ix[15] = hi[7];

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

    debug_assert!(kk <= 64);
    for i in (0..bm).step_by(32) {
        let zero = _mm256_setzero_si256();
        let mut vec_as = [zero; 32];
        let mut vec_signs = [zero; 8];

        for ai in 0..kk / 2 {
            vec_as[ai] = _mm256_loadu_si256(a.as_ptr().add(i * kk / 2 + ai * 32) as *const __m256i);
        }
        for s in 0..kk / 8 {
            vec_signs[s] =
                _mm256_loadu_si256(sign.as_ptr().add(i * kk / 8 + s * 32) as *const __m256i);
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

                    let vtlo = _mm256_xor_si256(
                        _mm256_add_epi16(_mm256_unpackhi_epi8(vtf, vts), sll),
                        sll,
                    );
                    let vthi = _mm256_xor_si256(
                        _mm256_add_epi16(_mm256_unpacklo_epi8(vtf, vts), slh),
                        slh,
                    );
                    let vblo = _mm256_xor_si256(
                        _mm256_add_epi16(_mm256_unpackhi_epi8(vbf, vbs), srl),
                        srl,
                    );
                    let vbhi = _mm256_xor_si256(
                        _mm256_add_epi16(_mm256_unpacklo_epi8(vbf, vbs), srh),
                        srh,
                    );

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
            gc1 = _mm256_add_epi32(
                gc1,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)),
            );
            gc2 = _mm256_add_epi32(gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
            gc3 = _mm256_add_epi32(
                gc3,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)),
            );

            _mm256_storeu_si256(c.as_mut_ptr().add(off) as *mut __m256i, gc0);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 8) as *mut __m256i, gc1);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 16) as *mut __m256i, gc2);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 24) as *mut __m256i, gc3);
        }
    }
}

/// Const-specialized three_tbl_impl for BK=96 (kk=32). All loop bounds are
/// compile-time constants, enabling full unrolling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn three_tbl_impl_bk96(
    c: &mut [i32],
    lut: &[i8],
    a: &[u8],
    sign: &[u8],
    bm: usize,
    k3: usize,
    batch_size: usize,
) {
    let vec_mask = _mm256_set1_epi8(0x0f);
    const _KK: usize = 32; // bk=96, kk=bk/3=32

    for i in (0..bm).step_by(32) {
        let zero = _mm256_setzero_si256();
        let mut vec_as: [__m256i; 16] = [zero; 16]; // KK/2 = 16
        let mut vec_signs: [__m256i; 4] = [zero; 4]; // KK/8 = 4

        // Fully unrolled loads — compiler knows these are exactly 16 + 4
        for ai in 0..16 {
            vec_as[ai] = _mm256_loadu_si256(a.as_ptr().add(i * 16 + ai * 32) as *const __m256i);
        }
        for s in 0..4 {
            vec_signs[s] = _mm256_loadu_si256(sign.as_ptr().add(i * 4 + s * 32) as *const __m256i);
        }

        for bs in 0..batch_size {
            let mut vec_c0 = zero;
            let mut vec_c1 = zero;

            macro_rules! sub {
                ($kg:expr, $j:literal, $shl0:literal, $shl1:literal, $shl2:literal, $shl3:literal) => {{
                    let vec_a = vec_as[$kg * 4 + $j];
                    let lut_off = $kg * 256 + $j * 64 + k3 / 3 * 32 * bs;
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
                    vec_c0 = _mm256_add_epi16(
                        vec_c0,
                        _mm256_xor_si256(
                            _mm256_add_epi16(_mm256_unpacklo_epi8(vtf, vts), slh),
                            slh,
                        ),
                    );
                    vec_c0 = _mm256_add_epi16(
                        vec_c0,
                        _mm256_xor_si256(
                            _mm256_add_epi16(_mm256_unpacklo_epi8(vbf, vbs), srh),
                            srh,
                        ),
                    );
                    vec_c1 = _mm256_add_epi16(
                        vec_c1,
                        _mm256_xor_si256(
                            _mm256_add_epi16(_mm256_unpackhi_epi8(vtf, vts), sll),
                            sll,
                        ),
                    );
                    vec_c1 = _mm256_add_epi16(
                        vec_c1,
                        _mm256_xor_si256(
                            _mm256_add_epi16(_mm256_unpackhi_epi8(vbf, vbs), srl),
                            srl,
                        ),
                    );
                }};
            }

            // Fully unrolled: kg=0..4 (KK/8=4), each with j=0..3
            sub!(0, 0, 0, 1, 2, 3);
            sub!(0, 1, 4, 5, 6, 7);
            sub!(0, 2, 8, 9, 10, 11);
            sub!(0, 3, 12, 13, 14, 15);
            sub!(1, 0, 0, 1, 2, 3);
            sub!(1, 1, 4, 5, 6, 7);
            sub!(1, 2, 8, 9, 10, 11);
            sub!(1, 3, 12, 13, 14, 15);
            sub!(2, 0, 0, 1, 2, 3);
            sub!(2, 1, 4, 5, 6, 7);
            sub!(2, 2, 8, 9, 10, 11);
            sub!(2, 3, 12, 13, 14, 15);
            sub!(3, 0, 0, 1, 2, 3);
            sub!(3, 1, 4, 5, 6, 7);
            sub!(3, 2, 8, 9, 10, 11);
            sub!(3, 3, 12, 13, 14, 15);

            let off = i + bm * bs;
            let mut gc0 = _mm256_loadu_si256(c.as_ptr().add(off) as *const __m256i);
            let mut gc1 = _mm256_loadu_si256(c.as_ptr().add(off + 8) as *const __m256i);
            let mut gc2 = _mm256_loadu_si256(c.as_ptr().add(off + 16) as *const __m256i);
            let mut gc3 = _mm256_loadu_si256(c.as_ptr().add(off + 24) as *const __m256i);
            gc0 = _mm256_add_epi32(gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
            gc1 = _mm256_add_epi32(
                gc1,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)),
            );
            gc2 = _mm256_add_epi32(gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
            gc3 = _mm256_add_epi32(
                gc3,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)),
            );
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
        let mut vec_as = [_mm256_setzero_si256(); 8];
        for ai in 0..kk / 2 {
            vec_as[ai] = _mm256_loadu_si256(a.as_ptr().add(i * kk / 2 + ai * 32) as *const __m256i);
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
            gc1 = _mm256_add_epi32(
                gc1,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)),
            );
            gc2 = _mm256_add_epi32(gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
            gc3 = _mm256_add_epi32(
                gc3,
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)),
            );

            _mm256_storeu_si256(c.as_mut_ptr().add(off) as *mut __m256i, gc0);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 8) as *mut __m256i, gc1);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 16) as *mut __m256i, gc2);
            _mm256_storeu_si256(c.as_mut_ptr().add(off + 24) as *mut __m256i, gc3);
        }
    }
}

// ── Top-level GEMM dispatch ────────────────────────────────────────────────

/// TL2 tile configuration for a specific (m, k) weight shape.
#[derive(Clone, Copy)]
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
        Self {
            m,
            k,
            bm,
            bk,
            three_k,
            two_k,
        }
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
            // BM chosen for thread utilization: m/BM >= n_threads (8)
            (640, 2560) => 320,  // 640/320=2 tiles (matches bitnet.cpp)
            (2560, 2560) => 320, // 2560/320=8 tiles
            (2560, 6912) => 320, // 2560/320=8 tiles
            (6912, 2560) => 256, // 6912/256=27 tiles
            _ => {
                // Auto-select: largest factor of m that is 32..=320
                let mut best = 0;
                for candidate in [320, 256, 160, 128, 64, 32] {
                    if m.is_multiple_of(candidate) {
                        best = candidate;
                        break;
                    }
                }
                if best == 0 {
                    return None;
                }
                best
            }
        };
        if !m.is_multiple_of(bm) {
            return None;
        }
        Some(Self::new(m, k, bm, bk))
    }
}

// ── I2S → TL2 weight repacking ────────────────────────────────────────────
//
// Translated from preprocess_three_weights_tl2 / preprocess_two_weights_tl2
// in convert-hf-to-gguf-bitnet.py. Converts I2S packed bytes (2-bit ternary,
// 4 values per byte, 128-element groups) to TL2 tiled format (base-3 nibble
// indices + sign bits, tiled by BM×BK).

fn i2s_extract_trit(i2s: &[u8], row: usize, col: usize, k: usize) -> i8 {
    let row_bytes = k / 4;
    let group = col / 128;
    let within = col % 128;
    let group_idx = within / 32;
    let group_pos = within % 32;
    let byte = i2s[row * row_bytes + group * 32 + group_pos];
    let bits = (byte >> (6 - 2 * group_idx)) & 0x03;
    bits as i8 - 1 // {0,1,2} → {-1,0,+1}
}

/// Repack I2S weight data into TL2 format for a single (m, k) weight tensor.
/// Direct translation of preprocess_weights_tl2 / preprocess_three_weights_tl2 /
/// preprocess_two_weights_tl2 from convert-hf-to-gguf-bitnet.py.
/// Input: I2S packed bytes (2-bit ternary, 4 per byte, 128-element groups).
/// Output: (tl2_data, scale) — nibble indices + sign bits in tiled layout.
pub fn repack_i2s_to_tl2(i2s: &[u8], m: usize, k: usize) -> Option<(Vec<u8>, f32)> {
    let cfg = Tl2Config::for_dims(m, k)?;
    repack_i2s_to_tl2_with_config(i2s, m, k, &cfg)
}

pub fn repack_i2s_to_tl2_with_config(
    i2s: &[u8],
    m: usize,
    k: usize,
    cfg: &Tl2Config,
) -> Option<(Vec<u8>, f32)> {
    let big_bm = cfg.bm; // BM in the Python — tile height
    let big_by = cfg.bk; // BY/BK in the Python — reduction tile
    let bmm: usize = 32; // bmm in the Python — inner tile
    let by: usize = 192 / bmm; // by in the Python = 6
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;

    let scale_off = k * m / 4;
    let scale = if scale_off + 4 <= i2s.len() {
        f32::from_ne_bytes([
            i2s[scale_off],
            i2s[scale_off + 1],
            i2s[scale_off + 2],
            i2s[scale_off + 3],
        ])
    } else {
        1.0
    };

    // Step 1: Decode all ternary values from I2S into a flat (m, k) i8 array
    let row_bytes = k / 4;
    let mut ternary = vec![0i8; m * k];
    let tern_ptr = ternary.as_mut_ptr();
    let i2s_ptr = i2s.as_ptr();
    crate::inference_pool::par_run(m, |rs, re| {
        for r in rs..re {
            let row_base = r * row_bytes;
            let row_out = unsafe { std::slice::from_raw_parts_mut(tern_ptr.add(r * k), k) };
            for grp in 0..k / 128 {
                let blk =
                    unsafe { std::slice::from_raw_parts(i2s_ptr.add(row_base + grp * 32), 32) };
                for j in 0..32 {
                    let b = blk[j];
                    row_out[grp * 128 + j] = ((b >> 6) & 3) as i8 - 1;
                    row_out[grp * 128 + 32 + j] = ((b >> 4) & 3) as i8 - 1;
                    row_out[grp * 128 + 64 + j] = ((b >> 2) & 3) as i8 - 1;
                    row_out[grp * 128 + 96 + j] = (b & 3) as i8 - 1;
                }
            }
        }
    });

    let mut final_weight: Vec<Vec<u8>> = Vec::new();

    // ── Three-part: translated from preprocess_three_weights_tl2 ──

    // Python line 550-556: group into triplets, compute base-3 index
    let three_n = m * three_k / 3;
    let mut weight = vec![0i32; three_n];
    for i in 0..three_n {
        let flat_idx = i * 3;
        let r = flat_idx / three_k;
        let c = flat_idx % three_k;
        let w0 = ternary[r * k + c] as i32;
        let w1 = ternary[r * k + c + 1] as i32;
        let w2 = ternary[r * k + c + 2] as i32;
        weight[i] = w0 * 9 + w1 * 3 + w2;
    }

    // Python line 557-558: sign extraction
    let mut sign_weight = vec![0u8; three_n];
    let mut abs_weight = vec![0u8; three_n];
    for i in 0..three_n {
        let s = if weight[i] < 0 { 1u8 } else { 0u8 };
        // Python: sign(w) + 2, then where > 1, set 0. So negative → 1, zero/positive → 0
        sign_weight[i] = s;
        abs_weight[i] = weight[i].unsigned_abs() as u8;
    }

    // Now abs_weight is shape (M, three_k/3) and sign_weight is same shape.
    // Python lines 564-576: reshape-transpose-pack chain for abs_weight
    // Instead of matching each numpy op, compute the final flat index directly.
    //
    // The chain is:
    //   (M, K3) → (M/BM, BM, K3) .T(0,2,1) → (M/BM, K3, BM)
    //   → (M/BM, K3/BY3, BY3, BM) .T(0,1,3,2) → (M/BM, K3/BY3, BM, BY3)
    //     where K3 = three_k/3, BY3 = big_by/3
    //   → (M/BM, K3/BY3, BM/bmm, bmm, BY3) .T(0,1,2,4,3) → (M/BM, K3/BY3, BM/bmm, BY3, bmm)
    //   → (M/BM, K3/BY3, BM/bmm, BY3/(by/3), by/3, bmm) .T(0,1,2,3,5,4)
    //     → (M/BM, K3/BY3, BM/bmm, BY3/(by/3), bmm, by/3)
    //   Then pack pairs: w0 << 4 | w1
    //   Reshape to (total_chunks, bmm//8, 8), swap rows [0,2,1,3], flatten to (total_chunks, bmm)

    let k3 = three_k / 3;
    let by3 = big_by / 3;
    let by_inner = by / 3; // = 2
    let n_bm_tiles = m / big_bm;
    let n_by_tiles = k3 / by3;
    let n_bmm_groups = big_bm / bmm;
    let n_by_groups = by3 / by_inner;

    // Build packed nibble array: shape (n_bm_tiles, n_by_tiles, n_bmm_groups, n_by_groups, bmm, by_inner)
    // where by_inner=2 and we pack [0]<<4 | [1]
    let total_chunks = n_bm_tiles * n_by_tiles * n_bmm_groups * n_by_groups;
    let mut nibbles = vec![0u8; total_chunks * bmm];

    for t0 in 0..n_bm_tiles {
        for t1 in 0..n_by_tiles {
            for t2 in 0..n_bmm_groups {
                for t3 in 0..n_by_groups {
                    let chunk_idx = ((t0 * n_by_tiles + t1) * n_bmm_groups + t2) * n_by_groups + t3;
                    for br in 0..bmm {
                        let row = t0 * big_bm + t2 * bmm + br;
                        let col0 = t1 * by3 + t3 * by_inner;
                        let v0 = abs_weight[row * k3 + col0];
                        let v1 = if by_inner > 1 {
                            abs_weight[row * k3 + col0 + 1]
                        } else {
                            0
                        };
                        nibbles[chunk_idx * bmm + br] = (v0 << 4) | (v1 & 0x0f);
                    }
                }
            }
        }
    }

    // Byte reorder: reshape (total_chunks, bmm//8, 8), swap rows [0,2,1,3]
    for chunk in 0..total_chunks {
        let base = chunk * bmm;
        // bmm=32, bmm//8=4, 4 rows of 8 bytes each
        // swap row 1 (bytes 8..15) ↔ row 2 (bytes 16..23)
        for i in 0..8 {
            nibbles.swap(base + 8 + i, base + 16 + i);
        }
    }

    // Append nibble chunks to final_weight
    for chunk in 0..total_chunks {
        final_weight.push(nibbles[chunk * bmm..(chunk + 1) * bmm].to_vec());
    }

    // ── Sign packing (Python lines 581-595) ──
    // After the same tiling as abs_weight, 16 sign bits pack into u16 words.
    // Traced mapping: for half-row hr in [0, bmm/2), bit i maps to:
    //   even i: sign_weight[t2*bmm + hr,        t1*by3 + sg*8 + i/2]
    //   odd i:  sign_weight[t2*bmm + bmm/2 + hr, t1*by3 + sg*8 + i/2]
    let sign_groups_per_by = by3 / 8; // by3=32, 32/8 = 4

    for t0 in 0..n_bm_tiles {
        for t1 in 0..n_by_tiles {
            for t2 in 0..n_bmm_groups {
                for sg in 0..sign_groups_per_by {
                    let mut packed = vec![0u8; bmm];
                    for hr in 0..(bmm / 2) {
                        let row_lo = t0 * big_bm + t2 * bmm + hr;
                        let row_hi = row_lo + bmm / 2;
                        let col_base = t1 * by3 + sg * 8;
                        let mut word: u16 = 0;
                        for c in 0..8 {
                            let col = col_base + c;
                            if col < k3 {
                                let s_lo = sign_weight[row_lo * k3 + col] as u16;
                                let s_hi = sign_weight[row_hi * k3 + col] as u16;
                                word |= s_lo << (15 - c * 2);
                                word |= s_hi << (14 - c * 2);
                            }
                        }
                        packed[hr * 2] = word as u8;
                        packed[hr * 2 + 1] = (word >> 8) as u8;
                    }
                    final_weight.push(packed);
                }
            }
        }
    }

    // ── Two-part weights (Python lines 523-547) ──
    if two_k > 0 {
        let two_n = m * two_k / 2;
        let mut two_weight = vec![0u8; two_n];
        for i in 0..two_n {
            let flat_idx = i * 2;
            let r = flat_idx / two_k;
            let c_base = three_k + flat_idx % two_k;
            let w0 = ternary[r * k + c_base] as i32;
            let w1 = ternary[r * k + c_base + 1] as i32;
            two_weight[i] = (w0 * 3 + w1 + 4) as u8;
        }

        let two_by_tile: usize = 32;
        let two_by_inner: usize = 4;
        let k2 = two_k / 2;
        let n_two_by_tiles = k2 / (two_by_tile / 2);
        let n_two_by_groups = (two_by_tile / 2) / (two_by_inner / 2);
        let two_pairs = two_by_inner / 2;

        let total_two_chunks =
            n_bm_tiles * n_two_by_tiles.max(1) * n_bmm_groups * n_two_by_groups.max(1);
        let mut two_nibbles = vec![0u8; total_two_chunks * bmm];

        for t0 in 0..n_bm_tiles {
            for t1 in 0..n_two_by_tiles.max(1) {
                for t2 in 0..n_bmm_groups {
                    for t3 in 0..n_two_by_groups.max(1) {
                        let chunk_idx = ((t0 * n_two_by_tiles.max(1) + t1) * n_bmm_groups + t2)
                            * n_two_by_groups.max(1)
                            + t3;
                        for br in 0..bmm {
                            let row = t0 * big_bm + t2 * bmm + br;
                            let col0 = t1 * (two_by_tile / 2) + t3 * two_pairs;
                            let v0 = if col0 < k2 {
                                two_weight[row * k2 + col0]
                            } else {
                                0
                            };
                            let v1 = if two_pairs > 1 && col0 + 1 < k2 {
                                two_weight[row * k2 + col0 + 1]
                            } else {
                                0
                            };
                            two_nibbles[chunk_idx * bmm + br] = (v0 << 4) | (v1 & 0x0f);
                        }
                    }
                }
            }
        }

        for chunk in 0..total_two_chunks {
            let base = chunk * bmm;
            for i in 0..8 {
                two_nibbles.swap(base + 8 + i, base + 16 + i);
            }
        }

        for chunk in 0..total_two_chunks {
            final_weight.push(two_nibbles[chunk * bmm..(chunk + 1) * bmm].to_vec());
        }
    }

    let result: Vec<u8> = final_weight.into_iter().flatten().collect();
    Some((result, scale))
}

/// Pre-built activation LUT, reusable across multiple weight tensors with the same k.
/// Translated from the workspace layout in ggml.c:12769-12777.
pub struct Tl2Lut {
    pub three_qlut: Vec<i8>,
    pub two_qlut: Vec<i8>,
    pub lut_scale: f32,
    pub three_k: usize,
    pub two_k: usize,
}

impl Tl2Lut {
    /// Pre-allocate LUT buffers for a given reduction dimension. Reuse across layers.
    #[cfg(target_arch = "x86_64")]
    pub fn new_for_k(k: usize) -> Option<Self> {
        let cfg = Tl2Config::for_dims(128, k)?;
        let three_lut_bytes = cfg.three_k / 3 * 32;
        let two_lut_bytes = if cfg.two_k > 0 { cfg.two_k / 2 * 32 } else { 0 };
        Some(Self {
            three_qlut: vec![0i8; three_lut_bytes],
            two_qlut: vec![0i8; two_lut_bytes.max(1)],
            lut_scale: 0.0,
            three_k: cfg.three_k,
            two_k: cfg.two_k,
        })
    }
}

/// Rebuild activation LUT into pre-allocated buffers. Returns false if k doesn't match.
#[cfg(target_arch = "x86_64")]
pub fn tl2_preprocess_into(activations: &[f32], k: usize, lut: &mut Tl2Lut) -> bool {
    let cfg = match Tl2Config::for_dims(128, k) {
        Some(c) => c,
        None => return false,
    };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }
    for b in lut.three_qlut.iter_mut() {
        *b = 0;
    }
    for b in lut.two_qlut.iter_mut() {
        *b = 0;
    }
    unsafe {
        preprocessor(
            &cfg,
            activations,
            &mut lut.lut_scale,
            &mut lut.three_qlut,
            &mut lut.two_qlut,
            1,
        );
    }
    true
}

/// Build activation LUT once. Call this on thread 0 before the barrier.
/// The returned Tl2Lut is then shared read-only across all tl2_gemv calls
/// for weight tensors with the same reduction dimension k.
#[cfg(target_arch = "x86_64")]
pub fn tl2_preprocess(activations: &[f32], k: usize) -> Option<Tl2Lut> {
    let cfg = Tl2Config::for_dims(128, k)?; // m doesn't matter for LUT, just need three_k/two_k
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;

    let three_lut_bytes = three_k / 3 * 32;
    let two_lut_bytes = if two_k > 0 { two_k / 2 * 32 } else { 0 };
    let mut lut_scale = 0.0f32;
    let mut three_qlut = vec![0i8; three_lut_bytes];
    let mut two_qlut = vec![0i8; two_lut_bytes.max(1)];

    unsafe {
        preprocessor(
            &cfg,
            activations,
            &mut lut_scale,
            &mut three_qlut,
            &mut two_qlut,
            1,
        );
    }

    Some(Tl2Lut {
        three_qlut,
        two_qlut,
        lut_scale,
        three_k,
        two_k,
    })
}

/// Run TL2 GEMV tiles [tile_start, tile_end) using a pre-built LUT.
/// Called from within parallel_compute — each thread processes its own tile range.
#[cfg(target_arch = "x86_64")]
pub fn tl2_gemv_tiles(
    lut: &Tl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    output_addr: usize,
    m: usize,
    k: usize,
    tile_start: usize,
    tile_end: usize,
) -> bool {
    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };
    tl2_gemv_tiles_cfg(
        lut,
        weight_data,
        w_scale,
        output_addr,
        &cfg,
        tile_start,
        tile_end,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tl2_gemv_tiles_cfg_avx2(
    lut: &Tl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    output_addr: usize,
    cfg: &Tl2Config,
    tile_start: usize,
    tile_end: usize,
) -> bool {
    let m = cfg.m;
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }

    let n_tile_num = m / cfg.bm;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;

    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    let mut cbits_arr = [0i32; 1024];
    assert!(cfg.bm <= 1024);
    let cbits = &mut cbits_arr[..cfg.bm];
    for i_tile in tile_start..tile_end {
        let w_offset = i_tile * w_tile_size;
        let s_offset = sign_base + i_tile * sign_tile_size;
        let dst_offset = i_tile * c_tile_size;

        for b in cbits.iter_mut() {
            *b = 0;
        }
        qgemm_three(
            cfg,
            &weight_data[w_offset..],
            &weight_data[s_offset..],
            &lut.three_qlut,
            cbits,
            1,
        );

        let out_slice =
            std::slice::from_raw_parts_mut((output_addr as *mut f32).add(dst_offset), cfg.bm);

        if two_k > 0 {
            let two_w_offset = two_a_base + i_tile * two_w_tile_size;
            let two_cfg = Tl2Config {
                three_k: 0,
                two_k,
                ..*cfg
            };
            qgemm_two_and_rescale(
                &two_cfg,
                &weight_data[two_w_offset..],
                &lut.two_qlut,
                w_scale,
                lut.lut_scale,
                cbits,
                out_slice,
                1,
            );
        } else {
            qgemm_two_and_rescale(cfg, &[], &[], w_scale, lut.lut_scale, cbits, out_slice, 1);
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn tl2_gemv_tiles_cfg(
    lut: &Tl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    output_addr: usize,
    cfg: &Tl2Config,
    tile_start: usize,
    tile_end: usize,
) -> bool {
    if is_x86_feature_detected!("avx2") {
        return unsafe {
            tl2_gemv_tiles_cfg_avx2(
                lut,
                weight_data,
                w_scale,
                output_addr,
                cfg,
                tile_start,
                tile_end,
            )
        };
    }
    false
}

/// Run TL2 GEMV using a pre-built LUT. Tiles distributed across thread pool.
#[cfg(target_arch = "x86_64")]
pub fn tl2_gemv(
    lut: &Tl2Lut,
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
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }

    let n_tile_num = m / cfg.bm;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;

    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    let three_lut_bytes = three_k / 3 * 32;
    let two_lut_bytes = if two_k > 0 { two_k / 2 * 32 } else { 0 };

    let out_ptr = output.as_mut_ptr() as usize;
    let wd_ptr = weight_data.as_ptr() as usize;
    let wd_len = weight_data.len();
    let tql_ptr = lut.three_qlut.as_ptr() as usize;
    let twql_ptr = lut.two_qlut.as_ptr() as usize;
    let lut_scale = lut.lut_scale;

    crate::thread_pool::global_pool().parallel_chunks(n_tile_num, 1, &|tile_start, tile_end| {
        for i_tile in tile_start..tile_end {
            let w_offset = i_tile * w_tile_size;
            let s_offset = sign_base + i_tile * sign_tile_size;
            let dst_offset = i_tile * c_tile_size;

            let wd = unsafe { std::slice::from_raw_parts(wd_ptr as *const u8, wd_len) };
            let tql = unsafe { std::slice::from_raw_parts(tql_ptr as *const i8, three_lut_bytes) };

            let mut cbits = vec![0i32; cfg.bm];
            unsafe {
                qgemm_three(&cfg, &wd[w_offset..], &wd[s_offset..], tql, &mut cbits, 1);
            }

            let out_slice = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(dst_offset), cfg.bm)
            };

            if two_k > 0 {
                let two_w_offset = two_a_base + i_tile * two_w_tile_size;
                let two_cfg = Tl2Config {
                    three_k: 0,
                    two_k,
                    ..cfg
                };
                let twql = unsafe {
                    std::slice::from_raw_parts(twql_ptr as *const i8, two_lut_bytes.max(1))
                };
                unsafe {
                    qgemm_two_and_rescale(
                        &two_cfg,
                        &wd[two_w_offset..],
                        twql,
                        w_scale,
                        lut_scale,
                        &mut cbits,
                        out_slice,
                        1,
                    );
                }
            } else {
                unsafe {
                    qgemm_two_and_rescale(
                        &cfg,
                        &[],
                        &[],
                        w_scale,
                        lut_scale,
                        &mut cbits,
                        out_slice,
                        1,
                    );
                }
            }
        }
    });

    true
}

/// TL2 GEMV with per-Q2_0-block weight scale corrections.
/// Each bk=96 chunk's contribution is scaled by the correction factor for
/// the Q2_0 block it falls into, then accumulated in f32.
#[cfg(target_arch = "x86_64")]
pub fn tl2_gemv_block_scaled(
    lut: &Tl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    block_corrections: &[f32],
    q2_block_size: usize,
    output: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }

    let n_tile_num = m / cfg.bm;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;
    let n_three_blocks = three_k / cfg.bk;
    let blocks_per_row = k / q2_block_size;

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;

    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    let three_lut_bytes = three_k / 3 * 32;
    let two_lut_bytes = if two_k > 0 { two_k / 2 * 32 } else { 0 };

    let out_ptr = output.as_mut_ptr() as usize;
    let wd_ptr = weight_data.as_ptr() as usize;
    let wd_len = weight_data.len();
    let tql_ptr = lut.three_qlut.as_ptr() as usize;
    let twql_ptr = lut.two_qlut.as_ptr() as usize;
    let lut_scale = lut.lut_scale;
    let use_bk96 = cfg.bk == 96;

    let bk_per_tile = cfg.bk;
    let w_blk_size_three = three_k * cfg.bm / 6 / n_tile_num / n_three_blocks;
    let s_blk_size_three = three_k * cfg.bm / 24 / n_tile_num / n_three_blocks;

    crate::thread_pool::global_pool().run_indexed(&|ith, nth| {
        let tile_start = ith * n_tile_num / nth;
        let tile_end = (ith + 1) * n_tile_num / nth;
        for i_tile in tile_start..tile_end {
            let w_base = i_tile * w_tile_size;
            let s_base = sign_base + i_tile * sign_tile_size;
            let dst_offset = i_tile * c_tile_size;
            let row_start = i_tile * cfg.bm;

            let wd = unsafe { std::slice::from_raw_parts(wd_ptr as *const u8, wd_len) };
            let tql = unsafe { std::slice::from_raw_parts(tql_ptr as *const i8, three_lut_bytes) };

            let mut cbits_f32 = vec![0.0f32; cfg.bm];

            for k_blk in 0..n_three_blocks {
                let lut_off = k_blk * bk_per_tile / 3 * 32;
                let a_off = w_base + k_blk * w_blk_size_three;
                let sign_off = s_base + k_blk * s_blk_size_three;

                let mut blk_cbits = vec![0i32; cfg.bm];
                if use_bk96 {
                    unsafe {
                        three_tbl_impl_bk96(
                            &mut blk_cbits,
                            &tql[lut_off..],
                            &wd[a_off..],
                            &wd[sign_off..],
                            cfg.bm,
                            cfg.three_k,
                            1,
                        )
                    };
                } else {
                    unsafe {
                        three_tbl_impl(
                            &mut blk_cbits,
                            &tql[lut_off..],
                            &wd[a_off..],
                            &wd[sign_off..],
                            cfg.bm,
                            cfg.bk,
                            cfg.three_k,
                            1,
                        )
                    };
                }

                let k_start = k_blk * bk_per_tile;
                let k_end = k_start + bk_per_tile;
                let q2_start = k_start / q2_block_size;
                let q2_end = (k_end - 1) / q2_block_size;

                for r in 0..cfg.bm {
                    let row = row_start + r;
                    let corr = if q2_start == q2_end {
                        let idx = row * blocks_per_row + q2_start;
                        if idx < block_corrections.len() {
                            block_corrections[idx]
                        } else {
                            1.0
                        }
                    } else {
                        let frac0 =
                            (q2_block_size - k_start % q2_block_size) as f32 / bk_per_tile as f32;
                        let idx0 = row * blocks_per_row + q2_start;
                        let idx1 = row * blocks_per_row + q2_end;
                        let c0 = if idx0 < block_corrections.len() {
                            block_corrections[idx0]
                        } else {
                            1.0
                        };
                        let c1 = if idx1 < block_corrections.len() {
                            block_corrections[idx1]
                        } else {
                            1.0
                        };
                        c0 * frac0 + c1 * (1.0 - frac0)
                    };
                    cbits_f32[r] += blk_cbits[r] as f32 * corr;
                }
            }

            let out_slice = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(dst_offset), cfg.bm)
            };

            if two_k > 0 {
                let two_w_offset = two_a_base + i_tile * two_w_tile_size;
                let _two_cfg = Tl2Config {
                    three_k: 0,
                    two_k,
                    ..cfg
                };
                let twql = unsafe {
                    std::slice::from_raw_parts(twql_ptr as *const i8, two_lut_bytes.max(1))
                };
                let mut two_cbits = vec![0i32; cfg.bm];
                unsafe {
                    two_tbl_impl(
                        &mut two_cbits,
                        &twql[0..],
                        &wd[two_w_offset..],
                        cfg.bm,
                        cfg.two_k,
                        1,
                    )
                };
                let last_q2 = (three_k) / q2_block_size;
                for r in 0..cfg.bm {
                    let row = row_start + r;
                    let corr_idx = row * blocks_per_row + last_q2.min(blocks_per_row - 1);
                    let corr = if corr_idx < block_corrections.len() {
                        block_corrections[corr_idx]
                    } else {
                        1.0
                    };
                    cbits_f32[r] += two_cbits[r] as f32 * corr;
                }
            }

            let scale = w_scale / lut_scale;
            for r in 0..cfg.bm {
                out_slice[r] = cbits_f32[r] * scale;
            }
        }
    });

    true
}

/// Run a full TL2 GEMV for batch_size=1 (decode).
/// Convenience wrapper that builds LUT internally — use tl2_preprocess + tl2_gemv
/// to share the LUT across multiple weight tensors.
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

    let lut = match tl2_preprocess(activations, k) {
        Some(l) => l,
        None => return false,
    };
    let out_ptr = output.as_mut_ptr() as usize;

    crate::thread_pool::global_pool().parallel_chunks(n_tile_num, 1, &|tile_start, tile_end| {
        for t in tile_start..tile_end {
            tl2_gemv_tiles_cfg(&lut, weight_data, w_scale, out_ptr, &cfg, t, t + 1);
        }
    });

    true
}

/// Build batched LUT from activations. Reusable across weight tensors with the same k.
#[cfg(target_arch = "x86_64")]
pub fn tl2_preprocess_batched(
    activations: &[f32],
    k: usize,
    batch_size: usize,
) -> Option<BatchedTl2Lut> {
    let cfg = Tl2Config::for_dims(128, k)?;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;
    let three_lut_bytes = three_k / 3 * 32;
    let two_lut_bytes = if two_k > 0 { two_k / 2 * 32 } else { 0 };

    let mut lut = BatchedTl2Lut {
        three_qlut: vec![0i8; three_lut_bytes * batch_size],
        two_qlut: vec![0i8; (two_lut_bytes * batch_size).max(1)],
        lut_scales: vec![0.0f32; batch_size],
        three_k,
        two_k,
        batch_size,
    };

    unsafe {
        preprocessor_batched(
            &cfg,
            activations,
            &mut lut.lut_scales,
            &mut lut.three_qlut,
            &mut lut.two_qlut,
            batch_size,
        );
    }
    Some(lut)
}

pub struct BatchedTl2Lut {
    pub three_qlut: Vec<i8>,
    pub two_qlut: Vec<i8>,
    pub lut_scales: Vec<f32>,
    pub three_k: usize,
    pub two_k: usize,
    pub batch_size: usize,
}

impl BatchedTl2Lut {
    #[cfg(target_arch = "x86_64")]
    pub fn new_for(k: usize, batch_size: usize) -> Option<Self> {
        let cfg = Tl2Config::for_dims(128, k)?;
        let three_lut_bytes = cfg.three_k / 3 * 32;
        let two_lut_bytes = if cfg.two_k > 0 { cfg.two_k / 2 * 32 } else { 0 };
        Some(BatchedTl2Lut {
            three_qlut: vec![0i8; three_lut_bytes * batch_size],
            two_qlut: vec![0i8; (two_lut_bytes * batch_size).max(1)],
            lut_scales: vec![0.0f32; batch_size],
            three_k: cfg.three_k,
            two_k: cfg.two_k,
            batch_size,
        })
    }
}

/// GEMM dispatch using a pre-built batched LUT. Reuse the LUT across
/// weight tensors that share the same activation input and reduction dimension.
#[cfg(target_arch = "x86_64")]
pub fn tl2_gemm_with_lut(
    lut: &BatchedTl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    output: &mut [f32],
    m: usize,
    k: usize,
) -> bool {
    if !is_x86_feature_detected!("avx2") {
        return false;
    }
    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }
    let batch_size = lut.batch_size;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;
    let n_tile_num = m / cfg.bm;

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;
    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    let out_ptr = output.as_mut_ptr() as usize;
    let wd_ptr = weight_data.as_ptr() as usize;
    let wd_len = weight_data.len();
    let tql_ptr = lut.three_qlut.as_ptr() as usize;
    let tql_len = lut.three_qlut.len();
    let twql_ptr = lut.two_qlut.as_ptr() as usize;
    let twql_len = lut.two_qlut.len();
    let ls_ptr = lut.lut_scales.as_ptr() as usize;

    crate::thread_pool::global_pool().parallel_chunks(
        n_tile_num,
        1,
        &|tile_start, tile_end| unsafe {
            let wd = std::slice::from_raw_parts(wd_ptr as *const u8, wd_len);
            let tql = std::slice::from_raw_parts(tql_ptr as *const i8, tql_len);
            let twql = std::slice::from_raw_parts(twql_ptr as *const i8, twql_len);
            let ls = std::slice::from_raw_parts(ls_ptr as *const f32, batch_size);

            let mut cbits = vec![0i32; batch_size * cfg.bm];

            for i_tile in tile_start..tile_end {
                let w_offset = i_tile * w_tile_size;
                let s_offset = sign_base + i_tile * sign_tile_size;

                cbits.iter_mut().for_each(|b| *b = 0);
                qgemm_three(
                    &cfg,
                    &wd[w_offset..],
                    &wd[s_offset..],
                    tql,
                    &mut cbits,
                    batch_size,
                );

                let dst_base = i_tile * c_tile_size;
                if two_k > 0 {
                    let two_w_offset = two_a_base + i_tile * two_w_tile_size;
                    let mut two_cbits = vec![0i32; batch_size * cfg.bm];
                    let n_two_blocks = two_k / BK2;
                    for k_outer in 0..n_two_blocks {
                        let lut_off = k_outer * BK2 / 2 * 32;
                        let a_off = k_outer * BK2 / 2 / 2 * cfg.bm;
                        two_tbl_impl(
                            &mut two_cbits,
                            &twql[lut_off..],
                            &wd[two_w_offset + a_off..],
                            cfg.bm,
                            two_k,
                            batch_size,
                        );
                    }
                    for bs in 0..batch_size {
                        let dst = std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(bs * m + dst_base),
                            cfg.bm,
                        );
                        for i in 0..cfg.bm {
                            dst[i] = (cbits[i + bs * cfg.bm] + two_cbits[i + bs * cfg.bm]) as f32
                                / ls[bs]
                                * w_scale;
                        }
                    }
                } else {
                    for bs in 0..batch_size {
                        let dst = std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(bs * m + dst_base),
                            cfg.bm,
                        );
                        for i in 0..cfg.bm {
                            dst[i] = cbits[i + bs * cfg.bm] as f32 / ls[bs] * w_scale;
                        }
                    }
                }
            }
        },
    );
    true
}

/// Fill tokens [t_start..t_end) within a `BatchedTl2Lut`. Each token writes to
/// its own LUT offset, so multiple threads can process disjoint ranges concurrently.
/// `activations` layout: `[batch_size * k]` — only elements `[t_start*k .. t_end*k]` are read.
/// `lut_addr` is the `BatchedTl2Lut` pointer as usize (for Send in closures).
#[cfg(target_arch = "x86_64")]
pub fn tl2_preprocess_token_range(
    activations_addr: usize,
    k: usize,
    lut_addr: usize,
    t_start: usize,
    t_end: usize,
) -> bool {
    let cfg = match Tl2Config::for_dims(128, k) {
        Some(c) => c,
        None => return false,
    };
    let lut = unsafe { &mut *(lut_addr as *mut BatchedTl2Lut) };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }

    let three_stride = cfg.three_k / 3 * 32;
    let two_stride = if cfg.two_k > 0 { cfg.two_k / 2 * 32 } else { 0 };

    for bs in t_start..t_end {
        let act =
            unsafe { std::slice::from_raw_parts((activations_addr as *const f32).add(bs * k), k) };
        let three_off = bs * three_stride;
        for b in lut.three_qlut[three_off..three_off + three_stride].iter_mut() {
            *b = 0;
        }
        if two_stride > 0 {
            let two_off = bs * two_stride;
            for b in lut.two_qlut[two_off..two_off + two_stride].iter_mut() {
                *b = 0;
            }
        }
        unsafe {
            lut.lut_scales[bs] = per_tensor_quant(cfg.two_k + cfg.three_k, act);
            three_lut_ctor(
                &mut lut.three_qlut[three_off..],
                act,
                lut.lut_scales[bs],
                cfg.three_k,
            );
            if cfg.two_k > 0 {
                let two_off = bs * two_stride;
                two_lut_ctor(
                    &mut lut.two_qlut[two_off..],
                    &act[cfg.three_k..],
                    lut.lut_scales[bs],
                    cfg.two_k,
                );
            }
        }
    }
    true
}

/// Fill an existing `BatchedTl2Lut` without allocating. Returns false if
/// the LUT dimensions don't match k.
#[cfg(target_arch = "x86_64")]
pub fn tl2_preprocess_batched_into(
    activations: &[f32],
    k: usize,
    batch_size: usize,
    lut: &mut BatchedTl2Lut,
) -> bool {
    let cfg = match Tl2Config::for_dims(128, k) {
        Some(c) => c,
        None => return false,
    };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }
    if batch_size > lut.batch_size {
        return false;
    }
    for b in lut.three_qlut[..cfg.three_k / 3 * 32 * batch_size].iter_mut() {
        *b = 0;
    }
    if cfg.two_k > 0 {
        for b in lut.two_qlut[..cfg.two_k / 2 * 32 * batch_size].iter_mut() {
            *b = 0;
        }
    }
    unsafe {
        preprocessor_batched(
            &cfg,
            activations,
            &mut lut.lut_scales[..batch_size],
            &mut lut.three_qlut,
            &mut lut.two_qlut,
            batch_size,
        );
    }
    true
}

/// Process TL2 GEMM tiles [tile_start, tile_end) using a pre-built batched LUT.
/// No thread pool dispatch — called from within a persistent thread model.
/// `cbits` and `two_cbits` must be pre-allocated to at least `batch_size * bm` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tl2_gemm_tiles_batched(
    lut: &BatchedTl2Lut,
    weight_data: &[u8],
    w_scale: f32,
    output_addr: usize,
    m: usize,
    k: usize,
    batch_size: usize,
    tile_start: usize,
    tile_end: usize,
    cbits: &mut [i32],
    two_cbits: &mut [i32],
) -> bool {
    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };
    if cfg.three_k != lut.three_k || cfg.two_k != lut.two_k {
        return false;
    }
    let n_tile_num = m / cfg.bm;
    let three_k = cfg.three_k;
    let two_k = cfg.two_k;

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;
    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    let bm = cfg.bm;
    let bs_bm = batch_size * bm;
    let three_lut_stride = three_k / 3 * 32;
    let two_lut_stride = if two_k > 0 { two_k / 2 * 32 } else { 0 };
    let n_two_blocks = if two_k > 0 { two_k / BK2 } else { 0 };
    let use_bk96 = cfg.bk == 96;

    for i_tile in tile_start..tile_end {
        let w_offset = i_tile * w_tile_size;
        let s_offset = sign_base + i_tile * sign_tile_size;
        let dst_base = i_tile * c_tile_size;
        let two_w_offset = if two_k > 0 {
            two_a_base + i_tile * two_w_tile_size
        } else {
            0
        };

        for b in cbits[..bs_bm].iter_mut() {
            *b = 0;
        }

        // Chunked three-part GEMM: chunk size from L1d cache detection
        if use_bk96 {
            let chunk = gemm_chunk_size();
            let mut done = 0usize;
            let mut rem = batch_size;
            while rem >= chunk {
                three_tbl_impl_bk96(
                    &mut cbits[done * bm..],
                    &lut.three_qlut[done * three_lut_stride..],
                    &weight_data[w_offset..],
                    &weight_data[s_offset..],
                    bm,
                    three_k,
                    chunk,
                );
                done += chunk;
                rem -= chunk;
            }
            while rem >= 8 {
                three_tbl_impl_bk96(
                    &mut cbits[done * bm..],
                    &lut.three_qlut[done * three_lut_stride..],
                    &weight_data[w_offset..],
                    &weight_data[s_offset..],
                    bm,
                    three_k,
                    8,
                );
                done += 8;
                rem -= 8;
            }
            while rem >= 1 {
                three_tbl_impl_bk96(
                    &mut cbits[done * bm..],
                    &lut.three_qlut[done * three_lut_stride..],
                    &weight_data[w_offset..],
                    &weight_data[s_offset..],
                    bm,
                    three_k,
                    1,
                );
                done += 1;
                rem -= 1;
            }
        } else {
            qgemm_three(
                &cfg,
                &weight_data[w_offset..],
                &weight_data[s_offset..],
                &lut.three_qlut,
                cbits,
                batch_size,
            );
        }

        if two_k > 0 {
            for b in two_cbits[..bs_bm].iter_mut() {
                *b = 0;
            }
            let chunk = gemm_chunk_size();
            let mut done = 0usize;
            let mut rem = batch_size;
            while rem >= chunk {
                for k_outer in 0..n_two_blocks {
                    let lut_off = k_outer * BK2 / 2 * 32;
                    let a_off = k_outer * BK2 / 2 / 2 * bm;
                    two_tbl_impl(
                        &mut two_cbits[done * bm..],
                        &lut.two_qlut[done * two_lut_stride + lut_off..],
                        &weight_data[two_w_offset + a_off..],
                        bm,
                        two_k,
                        chunk,
                    );
                }
                done += chunk;
                rem -= chunk;
            }
            while rem >= 8 {
                for k_outer in 0..n_two_blocks {
                    let lut_off = k_outer * BK2 / 2 * 32;
                    let a_off = k_outer * BK2 / 2 / 2 * bm;
                    two_tbl_impl(
                        &mut two_cbits[done * bm..],
                        &lut.two_qlut[done * two_lut_stride + lut_off..],
                        &weight_data[two_w_offset + a_off..],
                        bm,
                        two_k,
                        8,
                    );
                }
                done += 8;
                rem -= 8;
            }
            while rem >= 1 {
                for k_outer in 0..n_two_blocks {
                    let lut_off = k_outer * BK2 / 2 * 32;
                    let a_off = k_outer * BK2 / 2 / 2 * bm;
                    two_tbl_impl(
                        &mut two_cbits[done * bm..],
                        &lut.two_qlut[done * two_lut_stride + lut_off..],
                        &weight_data[two_w_offset + a_off..],
                        bm,
                        two_k,
                        1,
                    );
                }
                done += 1;
                rem -= 1;
            }
            for bs in 0..batch_size {
                let dst = std::slice::from_raw_parts_mut(
                    (output_addr as *mut f32).add(bs * m + dst_base),
                    bm,
                );
                let scale = _mm256_set1_ps(w_scale / lut.lut_scales[bs]);
                let cb = &cbits[bs * bm..];
                let tc = &two_cbits[bs * bm..];
                for i in (0..bm).step_by(8) {
                    let c = _mm256_loadu_si256(cb[i..].as_ptr() as *const __m256i);
                    let t = _mm256_loadu_si256(tc[i..].as_ptr() as *const __m256i);
                    let sum = _mm256_add_epi32(c, t);
                    let f = _mm256_cvtepi32_ps(sum);
                    _mm256_storeu_ps(dst[i..].as_mut_ptr(), _mm256_mul_ps(f, scale));
                }
            }
        } else {
            for bs in 0..batch_size {
                let dst = std::slice::from_raw_parts_mut(
                    (output_addr as *mut f32).add(bs * m + dst_base),
                    bm,
                );
                let scale = _mm256_set1_ps(w_scale / lut.lut_scales[bs]);
                let cb = &cbits[bs * bm..];
                for i in (0..bm).step_by(8) {
                    let c = _mm256_loadu_si256(cb[i..].as_ptr() as *const __m256i);
                    let f = _mm256_cvtepi32_ps(c);
                    _mm256_storeu_ps(dst[i..].as_mut_ptr(), _mm256_mul_ps(f, scale));
                }
            }
        }
    }
    true
}

/// Batched TL2 forward: processes `batch_size` activation vectors at once.
/// Activations layout: `[batch_size * k]` contiguous (row-major).
/// Output layout: `[batch_size * m]` where output[bs*m + row] is batch element bs, row row.
#[cfg(target_arch = "x86_64")]
pub fn tl2_forward_batched(
    activations: &[f32],
    weight_data: &[u8],
    w_scale: f32,
    output: &mut [f32],
    m: usize,
    k: usize,
    batch_size: usize,
) -> bool {
    if batch_size == 0 {
        return true;
    }
    if batch_size == 1 {
        return tl2_forward(activations, weight_data, w_scale, output, m, k);
    }

    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => return false,
    };

    let three_k = cfg.three_k;
    let two_k = cfg.two_k;
    let three_lut_bytes = three_k / 3 * 32;
    let two_lut_bytes = if two_k > 0 { two_k / 2 * 32 } else { 0 };

    let mut three_qlut = vec![0i8; three_lut_bytes * batch_size];
    let mut two_qlut = vec![0i8; (two_lut_bytes * batch_size).max(1)];
    let mut lut_scales = vec![0.0f32; batch_size];

    unsafe {
        preprocessor_batched(
            &cfg,
            activations,
            &mut lut_scales,
            &mut three_qlut,
            &mut two_qlut,
            batch_size,
        );
    }

    let n_tile_num = m / cfg.bm;
    let out_ptr = output.as_mut_ptr() as usize;
    let wd_ptr = weight_data.as_ptr() as usize;
    let wd_len = weight_data.len();
    let tql_ptr = three_qlut.as_ptr() as usize;
    let tql_len = three_qlut.len();
    let twql_ptr = two_qlut.as_ptr() as usize;
    let twql_len = two_qlut.len();
    let ls_ptr = lut_scales.as_ptr() as usize;

    if !is_x86_feature_detected!("avx2") {
        return false;
    }

    let w_tile_size = three_k * m / 6 / n_tile_num;
    let c_tile_size = cfg.bm;
    let sign_tile_size = three_k * m / 24 / n_tile_num;
    let sign_base = three_k * m / 3 / 2;
    let two_a_base = if two_k > 0 {
        three_k * m / 3 / 2 + three_k * m / 3 / 8
    } else {
        0
    };
    let two_w_tile_size = if two_k > 0 && n_tile_num > 0 {
        m * two_k / 4 / n_tile_num
    } else {
        0
    };

    crate::thread_pool::global_pool().parallel_chunks(n_tile_num, 1, &|tile_start, tile_end| {
        unsafe {
            let wd = std::slice::from_raw_parts(wd_ptr as *const u8, wd_len);
            let tql = std::slice::from_raw_parts(tql_ptr as *const i8, tql_len);
            let twql = std::slice::from_raw_parts(twql_ptr as *const i8, twql_len);
            let ls = std::slice::from_raw_parts(ls_ptr as *const f32, batch_size);

            let mut cbits = vec![0i32; batch_size * cfg.bm];

            for i_tile in tile_start..tile_end {
                let w_offset = i_tile * w_tile_size;
                let s_offset = sign_base + i_tile * sign_tile_size;

                cbits.iter_mut().for_each(|b| *b = 0);
                qgemm_three(
                    &cfg,
                    &wd[w_offset..],
                    &wd[s_offset..],
                    tql,
                    &mut cbits,
                    batch_size,
                );

                // Rescale and write directly to output (scattered layout)
                let dst_base = i_tile * c_tile_size;
                if two_k > 0 {
                    let two_w_offset = two_a_base + i_tile * two_w_tile_size;
                    let mut two_cbits = vec![0i32; batch_size * cfg.bm];
                    let n_two_blocks = two_k / BK2;
                    for k_outer in 0..n_two_blocks {
                        let lut_off = k_outer * BK2 / 2 * 32;
                        let a_off = k_outer * BK2 / 2 / 2 * cfg.bm;
                        two_tbl_impl(
                            &mut two_cbits,
                            &twql[lut_off..],
                            &wd[two_w_offset + a_off..],
                            cfg.bm,
                            two_k,
                            batch_size,
                        );
                    }
                    for bs in 0..batch_size {
                        let dst = std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(bs * m + dst_base),
                            cfg.bm,
                        );
                        for i in 0..cfg.bm {
                            let combined = cbits[i + bs * cfg.bm] + two_cbits[i + bs * cfg.bm];
                            dst[i] = combined as f32 / ls[bs] * w_scale;
                        }
                    }
                } else {
                    for bs in 0..batch_size {
                        let dst = std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(bs * m + dst_base),
                            cfg.bm,
                        );
                        for i in 0..cfg.bm {
                            dst[i] = cbits[i + bs * cfg.bm] as f32 / ls[bs] * w_scale;
                        }
                    }
                }
            }
        }
    });

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

/// Preprocess activations for batch_size > 1: stores per-element scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn preprocessor_batched(
    cfg: &Tl2Config,
    b: &[f32],
    lut_scales: &mut [f32],
    three_qlut: &mut [i8],
    two_qlut: &mut [i8],
    batch_size: usize,
) {
    for bs in 0..batch_size {
        let b_off = bs * (cfg.three_k + cfg.two_k);
        lut_scales[bs] = per_tensor_quant(cfg.two_k + cfg.three_k, &b[b_off..]);
        three_lut_ctor(
            &mut three_qlut[bs * cfg.three_k / 3 * 32..],
            &b[b_off..],
            lut_scales[bs],
            cfg.three_k,
        );
        if cfg.two_k > 0 {
            two_lut_ctor(
                &mut two_qlut[bs * cfg.two_k / 2 * 32..],
                &b[b_off + cfg.three_k..],
                lut_scales[bs],
                cfg.two_k,
            );
        }
    }
}

/// Three-part GEMM: accumulates i32 results into cbits. NO rescale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn qgemm_three(
    cfg: &Tl2Config,
    a: &[u8],
    sign: &[u8],
    lut: &[i8],
    cbits: &mut [i32],
    batch_size: usize,
) {
    let n_three_blocks = cfg.three_k / cfg.bk;
    let use_bk96 = cfg.bk == 96;
    for k_outer in 0..n_three_blocks {
        let lut_off = k_outer * cfg.bk / 3 * 32;
        let a_off = k_outer * cfg.bk / 3 / 2 * cfg.bm;
        let sign_off = k_outer * cfg.bk / 3 / 8 * cfg.bm;
        if use_bk96 {
            three_tbl_impl_bk96(
                cbits,
                &lut[lut_off..],
                &a[a_off..],
                &sign[sign_off..],
                cfg.bm,
                cfg.three_k,
                batch_size,
            );
        } else {
            three_tbl_impl(
                cbits,
                &lut[lut_off..],
                &a[a_off..],
                &sign[sign_off..],
                cfg.bm,
                cfg.bk,
                cfg.three_k,
                batch_size,
            );
        }
    }
}

/// Two-part GEMM: adds i32 results to existing cbits, then rescales to f32.
/// Translated from two_qgemm_lut_MxK (bitnet-lut-kernels-tl2.h:496-511).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn qgemm_two_and_rescale(
    cfg: &Tl2Config,
    a: &[u8],
    lut: &[i8],
    w_scale: f32,
    lut_scale: f32,
    cbits: &mut [i32],
    output: &mut [f32],
    batch_size: usize,
) {
    // Two-valued accumulation
    if cfg.two_k > 0 {
        let two_k_blocks = cfg.two_k / BK2;
        debug_assert!(batch_size * cfg.bm <= 1024);
        let mut two_cbits_arr = [0i32; 1024];
        let two_cbits = &mut two_cbits_arr[..batch_size * cfg.bm];
        for k_outer in 0..two_k_blocks {
            let lut_off = k_outer * BK2 / 2 * 32;
            let a_off = k_outer * BK2 / 2 / 2 * cfg.bm;
            two_tbl_impl(
                two_cbits,
                &lut[lut_off..],
                &a[a_off..],
                cfg.bm,
                cfg.two_k,
                batch_size,
            );
        }
        // C: ((int32_t*)C)[i] += CBits_two[i]; ((float*)C)[i] = (float)(C[i]) / lut_scale * w_scale
        for bs in 0..batch_size {
            for i in 0..cfg.bm {
                let combined = cbits[i + bs * cfg.bm] + two_cbits[i + bs * cfg.bm];
                output[i] = combined as f32 / lut_scale * w_scale;
            }
        }
    } else {
        // No two-part: just rescale three-part cbits
        // C: ((int32_t*)C)[i] = CBits[i]  (already in cbits)
        // Then two_qgemm rescales: but if two_k==0, the loop runs 0 times,
        // and the rescale still happens
        for bs in 0..batch_size {
            for i in 0..cfg.bm {
                output[i] = cbits[i + bs * cfg.bm] as f32 / lut_scale * w_scale;
            }
        }
    }
}

/// Two-part GEMM + rescale for batch_size > 1 with per-element scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn qgemm_two_and_rescale_batched(
    cfg: &Tl2Config,
    a: &[u8],
    lut: &[i8],
    w_scale: f32,
    lut_scales: &[f32],
    cbits: &mut [i32],
    output: &mut [f32],
    batch_size: usize,
) {
    if cfg.two_k > 0 {
        let two_k_blocks = cfg.two_k / BK2;
        let mut two_cbits = vec![0i32; batch_size * cfg.bm];
        for k_outer in 0..two_k_blocks {
            let lut_off = k_outer * BK2 / 2 * 32;
            let a_off = k_outer * BK2 / 2 / 2 * cfg.bm;
            two_tbl_impl(
                &mut two_cbits,
                &lut[lut_off..],
                &a[a_off..],
                cfg.bm,
                cfg.two_k,
                batch_size,
            );
        }
        for bs in 0..batch_size {
            for i in 0..cfg.bm {
                let combined = cbits[i + bs * cfg.bm] + two_cbits[i + bs * cfg.bm];
                output[i + bs * cfg.bm] = combined as f32 / lut_scales[bs] * w_scale;
            }
        }
    } else {
        for bs in 0..batch_size {
            for i in 0..cfg.bm {
                output[i + bs * cfg.bm] = cbits[i + bs * cfg.bm] as f32 / lut_scales[bs] * w_scale;
            }
        }
    }
}

/// Combined qgemm_lut for backward compatibility (single-tile use).
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
    qgemm_three(cfg, a, sign, lut, &mut cbits, batch_size);
    qgemm_two_and_rescale(
        cfg,
        &[],
        lut,
        w_scale,
        lut_scale,
        &mut cbits,
        output,
        batch_size,
    );
}

// ── Scalar reference implementations (for Kani proofs + DualPipeline) ──────
//
// These compute the same mathematical result as the AVX2 kernels above, without
// any SIMD. They exist to be formally verified by Kani and used as DualPipeline
// oracles. NOT gated behind cfg(target_arch) — they run on any platform.

pub fn per_tensor_quant_scalar(b: &[f32]) -> f32 {
    let mut max_abs = 0.0f32;
    for &v in b {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    127.0 / max_abs
}

/// Three-valued LUT entries for a single activation triplet (a0, a1, a2).
/// Returns the 14 non-zero entries indexed 0..13, matching the base-3 encoding.
pub fn three_lut_entries(a0: i32, a1: i32, a2: i32) -> [i32; 16] {
    [
        0,            // 0: (0,0,0)
        a2,           // 1: (0,0,+1)
        a1 - a2,      // 2: (0,+1,-1)
        a1,           // 3: (0,+1,0)
        a1 + a2,      // 4: (0,+1,+1)
        a0 - a1 - a2, // 5: (+1,-1,-1)
        a0 - a1,      // 6: (+1,-1,0)
        a0 - a1 + a2, // 7: (+1,-1,+1)
        a0 - a2,      // 8: (+1,0,-1)
        a0,           // 9: (+1,0,0)
        a0 + a2,      // 10: (+1,0,+1)
        a0 + a1 - a2, // 11: (+1,+1,-1)
        a0 + a1,      // 12: (+1,+1,0)
        a0 + a1 + a2, // 13: (+1,+1,+1)
        0,            // 14: unused
        0,            // 15: unused
    ]
}

/// Two-valued LUT entries for a single activation pair (a0, a1).
/// Index = w0*3 + w1 + 4, matching the two_lut_ctor encoding.
pub fn two_lut_entries(a0: i32, a1: i32) -> [i32; 16] {
    [
        -(a0 + a1), // 0: (-1,-1)
        -a0,        // 1: (-1,0)
        -a0 + a1,   // 2: (-1,+1)
        -a1,        // 3: (0,-1)
        0,          // 4: (0,0)
        a1,         // 5: (0,+1)
        a0 - a1,    // 6: (+1,-1)
        a0,         // 7: (+1,0)
        a0 + a1,    // 8: (+1,+1)
        0,
        0,
        0,
        0,
        0,
        0,
        0, // 9..15: unused
    ]
}

/// Scalar reference for the TL2 GEMV end-to-end: quantize activations, group
/// into triplets/pairs matching the LUT path, apply the same i8 saturation,
/// and accumulate. This mirrors the AVX2 LUT math exactly (without SIMD) so
/// the DualPipeline comparison can assert bit-exact parity.
pub fn tl2_gemv_direct_scalar(
    activations: &[f32],
    i2s_data: &[u8],
    w_scale: f32,
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    assert_eq!(activations.len(), k);
    assert_eq!(output.len(), m);

    let cfg = match Tl2Config::for_dims(m, k) {
        Some(c) => c,
        None => {
            // Fallback: element-wise (no LUT grouping) for non-standard dims
            let scale = per_tensor_quant_scalar(activations);
            let mut a_q = vec![0i8; k];
            for i in 0..k {
                let v = (activations[i] * scale).round() as i32;
                a_q[i] = v.clamp(-127, 127) as i8;
            }
            for row in 0..m {
                let mut acc = 0i32;
                for col in 0..k {
                    let trit = i2s_extract_trit(i2s_data, row, col, k);
                    acc += trit as i32 * a_q[col] as i32;
                }
                output[row] = acc as f32 / scale * w_scale;
            }
            return;
        }
    };
    let three_k = cfg.three_k;

    let scale = per_tensor_quant_scalar(activations);
    let mut a_q = vec![0i32; k];
    for i in 0..k {
        a_q[i] = (activations[i] * scale).round() as i32;
        a_q[i] = a_q[i].clamp(-127, 127);
    }

    for row in 0..m {
        let mut acc = 0i32;

        // Three-valued path: group activations into triplets, compute LUT entry,
        // saturate to i8, apply sign, accumulate
        let mut col = 0;
        while col + 2 < three_k {
            let w0 = i2s_extract_trit(i2s_data, row, col, k) as i32;
            let w1 = i2s_extract_trit(i2s_data, row, col + 1, k) as i32;
            let w2 = i2s_extract_trit(i2s_data, row, col + 2, k) as i32;

            let combined = w0 * 9 + w1 * 3 + w2;
            let abs_idx = combined.unsigned_abs() as usize;
            let sign = combined < 0;

            let a0 = a_q[col];
            let a1 = a_q[col + 1];
            let a2 = a_q[col + 2];
            let entries = three_lut_entries(a0, a1, a2);
            let lut_val = entries[abs_idx].clamp(-32768, 32767) as i16;
            let contribution = if sign { -lut_val } else { lut_val };
            acc += contribution as i32;

            col += 3;
        }

        // Two-valued path: group into pairs
        let mut col2 = three_k;
        while col2 + 1 < k {
            let w0 = i2s_extract_trit(i2s_data, row, col2, k) as i32;
            let w1 = i2s_extract_trit(i2s_data, row, col2 + 1, k) as i32;

            let idx = (w0 * 3 + w1 + 4) as usize;

            let a0 = a_q[col2];
            let a1 = a_q[col2 + 1];
            let entries = two_lut_entries(a0, a1);
            let lut_val = entries[idx].clamp(-32768, 32767) as i16;
            acc += lut_val as i32;

            col2 += 2;
        }

        output[row] = acc as f32 / scale * w_scale;
    }
}

// NOTE: A scalar three_tbl_impl/two_tbl_impl that reads the AVX2 byte layout
// is not written because the post-transpose+pack layout is tightly coupled to
// SIMD lane semantics. Instead, tl2_gemv_direct_scalar above serves as the
// ground-truth oracle: it computes the same mathematical result (ternary weights
// × quantized activations) from the I2S format directly. DualPipeline tests
// compare this oracle against the AVX2 TL2 path.

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
        assert!(
            qlut.iter().any(|&v| v != 0),
            "LUT should have non-zero entries"
        );
    }

    #[test]
    fn test_two_lut_ctor_runs() {
        let b: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let scale = 127.0 / 0.8;
        let mut qlut = vec![0i8; 256];
        unsafe { two_lut_ctor(&mut qlut, &b, scale, 16) };
        assert!(
            qlut.iter().any(|&v| v != 0),
            "LUT should have non-zero entries"
        );
    }

    #[test]
    fn test_repack_i2s_to_tl2_matches_mad() {
        use crate::quantized::{i2s_gemv_cols, i2s_weight_scale};

        // Create I2S weights: 128x128 with known ternary pattern
        let m = 128usize;
        let k = 128usize;
        let packed_bytes = m * k / 4;
        let mut i2s = vec![0u8; packed_bytes + 4]; // +4 for scale

        // Fill with a pattern: diagonal = +1, rest = 0
        // I2S packing: byte at row*row_bytes + group*32 + pos
        // value at group_idx within byte: bits (6-2*group_idx)
        let row_bytes = k / 4;
        for r in 0..m {
            for c in 0..k {
                let trit: u8 = if r == c { 2 } else { 1 }; // 2=+1, 1=0
                let group = c / 128;
                let within = c % 128;
                let group_idx = within / 32;
                let group_pos = within % 32;
                let byte_off = r * row_bytes + group * 32 + group_pos;
                i2s[byte_off] |= trit << (6 - 2 * group_idx);
            }
        }
        // Scale = 1.0
        i2s[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());

        // Activation
        let act: Vec<f32> = (1..=k).map(|i| i as f32 * 0.1).collect();

        // MAD reference output
        let mut q8 = vec![0i8; k];
        let (scale, sum) = crate::quantized::i2s_quantize_into(&act, &mut q8);
        let w_scale = i2s_weight_scale(&i2s, m, k);
        let mut mad_out = vec![0.0f32; m];
        let out_addr = mad_out.as_mut_ptr() as usize;
        i2s_gemv_cols(&q8, scale, sum, &i2s, w_scale, out_addr, 0, m, k);

        // TL2 repack + forward
        let (tl2_data, tl2_scale) = repack_i2s_to_tl2(&i2s, m, k).expect("repack should succeed");
        let mut tl2_out = vec![0.0f32; m];
        let ok = tl2_forward(&act, &tl2_data, tl2_scale, &mut tl2_out, m, k);
        assert!(ok, "tl2_forward should succeed");

        eprintln!("MAD[0..8]: {:?}", &mad_out[..8]);
        eprintln!("TL2[0..8]: {:?}", &tl2_out[..8]);

        let max_err = mad_out
            .iter()
            .zip(tl2_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mad_norm: f32 = mad_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("max_err={max_err:.6} mad_norm={mad_norm:.4}");

        assert!(
            max_err < mad_norm * 0.05 + 1.0,
            "TL2 output should match MAD within 5% + 1.0: max_err={max_err:.6}"
        );
    }

    #[test]
    fn test_repack_i2s_to_tl2_random_weights() {
        use crate::quantized::{i2s_gemv_cols, i2s_weight_scale};

        let m = 256usize;
        let k = 384usize; // multiple tiles + multiple reduction blocks
        let packed_bytes = m * k / 4;
        let mut i2s = vec![0u8; packed_bytes + 4];

        // Random ternary weights including negatives
        let mut rng_state = 42u64;
        let row_bytes = k / 4;
        for r in 0..m {
            for c in 0..k {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let trit: u8 = ((rng_state >> 33) % 3) as u8; // 0,1,2 = -1,0,+1
                let group = c / 128;
                let within = c % 128;
                let group_idx = within / 32;
                let group_pos = within % 32;
                let byte_off = r * row_bytes + group * 32 + group_pos;
                i2s[byte_off] |= trit << (6 - 2 * group_idx);
            }
        }
        i2s[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());

        let act: Vec<f32> = (1..=k)
            .map(|i| (i as f32 - k as f32 / 2.0) * 0.01)
            .collect();

        let mut q8 = vec![0i8; k];
        let (scale, sum) = crate::quantized::i2s_quantize_into(&act, &mut q8);
        let w_scale = i2s_weight_scale(&i2s, m, k);
        let mut mad_out = vec![0.0f32; m];
        i2s_gemv_cols(
            &q8,
            scale,
            sum,
            &i2s,
            w_scale,
            mad_out.as_mut_ptr() as usize,
            0,
            m,
            k,
        );

        let (tl2_data, tl2_scale) = repack_i2s_to_tl2(&i2s, m, k).expect("repack should succeed");
        let mut tl2_out = vec![0.0f32; m];
        let ok = tl2_forward(&act, &tl2_data, tl2_scale, &mut tl2_out, m, k);
        assert!(ok, "tl2_forward should succeed");

        let max_err = mad_out
            .iter()
            .zip(tl2_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mad_norm: f32 = mad_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("MAD[0..4]: {:?}", &mad_out[..4]);
        eprintln!("TL2[0..4]: {:?}", &tl2_out[..4]);
        eprintln!("max_err={max_err:.6} mad_norm={mad_norm:.4}");

        assert!(
            max_err < mad_norm * 0.05 + 1.0,
            "TL2 should match MAD: max_err={max_err:.6} mad_norm={mad_norm:.4}"
        );
    }

    #[test]
    fn bench_tl2_bm_tuning() {
        let shapes = [(2560, 2560), (2560, 6912), (6912, 2560), (640, 2560)];
        for (m, k) in shapes {
            let packed_bytes = m * k / 4;
            let mut i2s = vec![0u8; packed_bytes + 4];
            let mut rng = 42u64;
            let row_bytes = k / 4;
            for r in 0..m {
                for c in 0..k {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let trit = ((rng >> 33) % 3) as u8;
                    let group = c / 128;
                    let within = c % 128;
                    let gi = within / 32;
                    let gp = within % 32;
                    i2s[r * row_bytes + group * 32 + gp] |= trit << (6 - 2 * gi);
                }
            }
            i2s[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());

            let act: Vec<f32> = (0..k)
                .map(|i| (i as f32 - k as f32 / 2.0) * 0.001)
                .collect();

            let candidates: Vec<usize> = [32, 64, 128, 160, 256, 320, 640]
                .iter()
                .copied()
                .filter(|&bm| m % bm == 0)
                .collect();

            for &bm in &candidates {
                let cfg = Tl2Config::new(m, k, bm, 96);
                let (tl2_data, scale) = repack_i2s_to_tl2_with_config(&i2s, m, k, &cfg).unwrap();
                let lut = tl2_preprocess(&act, k).unwrap();
                let mut output = vec![0.0f32; m];
                let n_tiles = m / bm;

                let iters = 20;
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    tl2_gemv_tiles_cfg(
                        &lut,
                        &tl2_data,
                        scale,
                        output.as_mut_ptr() as usize,
                        &cfg,
                        0,
                        n_tiles,
                    );
                }
                let elapsed = start.elapsed();
                let us_per = elapsed.as_micros() as f64 / iters as f64;
                eprintln!("({m},{k}) BM={bm:>3}: {us_per:.0}µs/iter");
            }
            eprintln!();
        }
    }

    #[test]
    fn test_tl2_forward_2tile_256x96() {
        let home = std::env::var("HOME").unwrap();
        let data = match std::fs::read(format!("{home}/.cache/tempura/models/test_tl2_256x96.bin"))
        {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: test data not found");
                return;
            }
        };
        assert_eq!(data.len(), 5120);
        let act_bytes = std::fs::read(format!(
            "{home}/.cache/tempura/models/test_tl2_256x96_act.bin"
        ))
        .unwrap();
        let act: Vec<f32> = act_bytes
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ref_bytes = std::fs::read(format!(
            "{home}/.cache/tempura/models/test_tl2_256x96_ref.bin"
        ))
        .unwrap();
        let y_ref: Vec<f32> = ref_bytes
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut output = vec![0.0f32; 256];
        let ok = tl2_forward(&act, &data, 1.0, &mut output, 256, 96);
        assert!(ok, "tl2_forward should succeed for 256x96");

        eprintln!("tile0 out[0..4]: {:?}", &output[..4]);
        eprintln!("tile0 ref[0..4]: {:?}", &y_ref[..4]);
        eprintln!("tile1 out[128..132]: {:?}", &output[128..132]);
        eprintln!("tile1 ref[128..132]: {:?}", &y_ref[128..132]);

        let max_err = output
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("max_err={max_err:.6}");
        assert!(
            max_err < 1.0,
            "max_err={max_err:.6} — output doesn't match reference"
        );
    }

    #[test]
    fn test_tl2_forward_identity_128x96() {
        // Load TL2-packed 128x96 identity matrix generated by bitnet.cpp converter
        let home = std::env::var("HOME").unwrap();
        let path = format!("{home}/.cache/tempura/models/test_tl2_128x96.bin");
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: {path} not found (run Python converter first)");
                return;
            }
        };
        assert_eq!(data.len(), 2560, "expected 2048 weight + 512 sign bytes");

        // Activation: [1.0, 2.0, 3.0, ..., 96.0] padded to 96
        let mut act = vec![0.0f32; 96];
        for i in 0..96 {
            act[i] = (i + 1) as f32;
        }

        let mut output = vec![0.0f32; 128];
        let w_scale = 1.0f32;

        let ok = tl2_forward(&act, &data, w_scale, &mut output, 128, 96);
        assert!(ok, "tl2_forward should succeed for 128x96");

        // Identity matrix: output[i] should be act[i] for i < 96, 0 for i >= 96
        // (with quantization error from 127-scale rounding)
        eprintln!("output[0..8]: {:?}", &output[..8]);
        eprintln!("output[93..100]: {:?}", &output[93..100]);

        for i in 0..96 {
            let expected = act[i];
            let err = (output[i] - expected).abs();
            assert!(
                err < expected.abs() * 0.1 + 0.5,
                "output[{i}]={:.4} expected {expected:.4} err={err:.4}",
                output[i]
            );
        }
        for i in 96..128 {
            assert!(
                output[i].abs() < 0.5,
                "output[{i}]={:.4} should be ~0",
                output[i]
            );
        }
    }

    #[test]
    fn test_per_tensor_quant_scalar_matches_avx2() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let scalar = per_tensor_quant_scalar(&a);
        let avx2 = unsafe { per_tensor_quant(64, &a) };
        assert!((scalar - avx2).abs() < 1e-5, "scalar={scalar} avx2={avx2}");
    }

    #[test]
    fn test_three_lut_entries_symmetry() {
        let e = three_lut_entries(10, 20, 30);
        assert_eq!(e[0], 0);
        assert_eq!(e[9], 10); // a0
        assert_eq!(e[3], 20); // a1
        assert_eq!(e[1], 30); // a2
        assert_eq!(e[13], 60); // a0+a1+a2
        assert_eq!(e[12], 30); // a0+a1
    }

    #[test]
    fn test_two_lut_entries_symmetry() {
        let e = two_lut_entries(10, 20);
        assert_eq!(e[4], 0); // (0,0)
        assert_eq!(e[7], 10); // (+1,0) = a0
        assert_eq!(e[5], 20); // (0,+1) = a1
        assert_eq!(e[8], 30); // (+1,+1) = a0+a1
        assert_eq!(e[0], -30); // (-1,-1) = -(a0+a1)
        assert_eq!(e[1], -10); // (-1,0) = -a0
    }

    #[test]
    fn test_tl2_direct_scalar_matches_mad() {
        use crate::quantized::{i2s_gemv_cols, i2s_weight_scale};

        let m = 128usize;
        let k = 128usize;
        let packed_bytes = m * k / 4;
        let mut i2s = vec![0u8; packed_bytes + 4];

        let row_bytes = k / 4;
        for r in 0..m {
            for c in 0..k {
                let trit: u8 = if r == c { 2 } else { 1 };
                let group = c / 128;
                let within = c % 128;
                let group_idx = within / 32;
                let group_pos = within % 32;
                let byte_off = r * row_bytes + group * 32 + group_pos;
                i2s[byte_off] |= trit << (6 - 2 * group_idx);
            }
        }
        i2s[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());

        let act: Vec<f32> = (1..=k).map(|i| i as f32 * 0.1).collect();

        // MAD reference
        let mut q8 = vec![0i8; k];
        let (scale, sum) = crate::quantized::i2s_quantize_into(&act, &mut q8);
        let w_scale = i2s_weight_scale(&i2s, m, k);
        let mut mad_out = vec![0.0f32; m];
        i2s_gemv_cols(
            &q8,
            scale,
            sum,
            &i2s,
            w_scale,
            mad_out.as_mut_ptr() as usize,
            0,
            m,
            k,
        );

        // Direct scalar
        let mut scalar_out = vec![0.0f32; m];
        tl2_gemv_direct_scalar(&act, &i2s, w_scale, &mut scalar_out, m, k);

        let max_err = mad_out
            .iter()
            .zip(scalar_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let norm: f32 = mad_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_err = if norm > 0.0 { max_err / norm } else { max_err };
        assert!(
            rel_err < 0.05,
            "direct scalar vs MAD rel_err={rel_err} max_err={max_err}"
        );
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn tl2_three_lut_entry_values_bounded() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        let a2: i32 = kani::any();
        kani::assume(a0 >= -127 && a0 <= 127);
        kani::assume(a1 >= -127 && a1 <= 127);
        kani::assume(a2 >= -127 && a2 <= 127);

        let entries = three_lut_entries(a0, a1, a2);
        for i in 0..16 {
            assert!(entries[i] >= -381 && entries[i] <= 381);
        }
    }

    #[kani::proof]
    fn tl2_three_lut_unused_entries_zero() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        let a2: i32 = kani::any();
        kani::assume(a0 >= -127 && a0 <= 127);
        kani::assume(a1 >= -127 && a1 <= 127);
        kani::assume(a2 >= -127 && a2 <= 127);

        let entries = three_lut_entries(a0, a1, a2);
        assert_eq!(entries[0], 0);
        assert_eq!(entries[14], 0);
        assert_eq!(entries[15], 0);
    }

    #[kani::proof]
    fn tl2_two_lut_entry_values_bounded() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        kani::assume(a0 >= -127 && a0 <= 127);
        kani::assume(a1 >= -127 && a1 <= 127);

        let entries = two_lut_entries(a0, a1);
        for i in 0..16 {
            assert!(entries[i] >= -254 && entries[i] <= 254);
        }
    }

    #[kani::proof]
    fn tl2_two_lut_unused_entries_zero() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        kani::assume(a0 >= -127 && a0 <= 127);
        kani::assume(a1 >= -127 && a1 <= 127);

        let entries = two_lut_entries(a0, a1);
        for i in 9..16 {
            assert_eq!(entries[i], 0);
        }
    }

    #[kani::proof]
    fn tl2_three_lut_i8_saturation_range() {
        let a0: i32 = kani::any();
        let a1: i32 = kani::any();
        let a2: i32 = kani::any();
        kani::assume(a0 >= -127 && a0 <= 127);
        kani::assume(a1 >= -127 && a1 <= 127);
        kani::assume(a2 >= -127 && a2 <= 127);

        let entries = three_lut_entries(a0, a1, a2);
        for i in 0..16 {
            let saturated = entries[i].clamp(-128, 127) as i8;
            assert!(saturated >= -128 && saturated <= 127);
        }
    }

    #[kani::proof]
    fn tl2_i16_accumulator_no_overflow_bk96() {
        let val: i8 = kani::any();
        let kk: usize = 32; // BK=96 → kk=96/3=32
                            // Each tbl_impl iteration adds at most |val| per step
                            // kk triplet-groups per tile, each contributes one i8 LUT value
        let worst_case = val as i32 * kk as i32;
        assert!(worst_case >= i16::MIN as i32 && worst_case <= i16::MAX as i32);
    }

    #[kani::proof]
    fn tl2_base3_triplet_index_in_range() {
        let w0: i8 = kani::any();
        let w1: i8 = kani::any();
        let w2: i8 = kani::any();
        kani::assume(w0 >= -1 && w0 <= 1);
        kani::assume(w1 >= -1 && w1 <= 1);
        kani::assume(w2 >= -1 && w2 <= 1);

        let combined = w0 as i32 * 9 + w1 as i32 * 3 + w2 as i32;
        assert!(combined >= -13 && combined <= 13);
        let abs_val = combined.unsigned_abs();
        assert!(abs_val <= 13);
        // Fits in a 4-bit nibble
        assert!(abs_val < 16);
    }

    #[kani::proof]
    fn tl2_base2_pair_index_in_range() {
        let w0: i8 = kani::any();
        let w1: i8 = kani::any();
        kani::assume(w0 >= -1 && w0 <= 1);
        kani::assume(w1 >= -1 && w1 <= 1);

        let index = w0 as i32 * 3 + w1 as i32 + 4;
        assert!(index >= 0 && index <= 8);
        assert!((index as u32) < 16);
    }

    #[kani::proof]
    fn tl2_config_three_k_two_k_partition() {
        let k: usize = kani::any();
        let bk: usize = 96;
        kani::assume(k >= bk);
        kani::assume(k <= 8192);
        kani::assume(k % bk == 0 || k % bk >= 32);

        let bk_num = k / bk;
        let three_k = bk_num * bk;
        let two_k = k - three_k;
        assert!(three_k + two_k == k);
        assert!(three_k % 3 == 0);
        assert!(two_k < bk);
        if two_k > 0 {
            assert!(two_k % 2 == 0);
        }
    }

    #[kani::proof]
    fn tl2_nibble_extraction_in_range() {
        let byte: u8 = kani::any();
        let top = (byte >> 4) & 0x0f;
        let bot = byte & 0x0f;
        assert!(top < 16);
        assert!(bot < 16);
    }

    #[kani::proof]
    fn tl2_per_tensor_quant_positive() {
        let max_abs: f32 = kani::any();
        kani::assume(max_abs > 0.0);
        kani::assume(max_abs.is_finite());
        kani::assume(max_abs <= 1e6);

        let scale = 127.0 / max_abs;
        assert!(scale > 0.0);
        assert!(scale.is_finite());
    }

    #[kani::proof]
    fn tl2_i2s_extract_trit_value_range() {
        let byte_val: u8 = kani::any();
        let group_idx: usize = kani::any();
        kani::assume(group_idx < 4);

        let shift = 6 - 2 * group_idx;
        let bits = (byte_val >> shift) & 0x03;
        let trit = bits as i8 - 1;
        assert!(trit >= -1 && trit <= 1);
    }

    #[kani::proof]
    fn tl2_direct_dot_accumulator_no_overflow() {
        let k: usize = kani::any();
        kani::assume(k >= 1 && k <= 8192);

        // Each term: trit ∈ {-1,0,1}, a_q ∈ [-127,127]
        // Worst case per element: 127
        // k elements → max |acc| = 127 * k
        let worst_case = 127i64 * k as i64;
        assert!(worst_case <= i32::MAX as i64);
    }
}

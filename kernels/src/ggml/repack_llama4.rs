/// Direct translation of ggml_gemm_q2_K_8x8_q8_K (arch/x86/repack.cpp lines 3526-6407).
/// AVX2 GEMM: BlockQ2Kx8 weights × BlockQ8Kx4 activations → float output.
///
/// Translated path: the `for (; y < nr/4; y++)` AVX2-only loop
/// (source lines 5733-6400), which handles one BlockQ8Kx4 group (4 rows)
/// per outer iteration. Identical in structure to ggml_gemm_q4_k_8x8_q8_k
/// but with 2-bit quant extraction and 4-bit packed scale+min encoding.
///
/// - vx: packed weights as BlockQ2Kx8 (nc/8 × nb)
/// - vy: activations as BlockQ8Kx4 (nr/4 × nb)
/// - s:  output matrix, row-major, nr rows × nc cols floats
/// - bs: output row stride in elements
/// - n:  token dimension (multiple of QK_K = 256)

const QK_K: usize = 256;

use super::blocks::{BlockQ2Kx8, BlockQ8Kx4};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemm_q2_k_8x8_q8_k(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    use std::arch::x86_64::*;

    let nb = n / QK_K;
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let m3b         = _mm256_set1_epi8(3);
    let m4b_sse     = _mm_set1_epi8(0xF_u8 as i8);
    // scalesmask1: picks even-indexed bytes (0,2,4,...,14), each repeated twice
    let scalesmask1_sse = _mm_set_epi8(14,14,12,12,10,10,8,8,6,6,4,4,2,2,0,0);
    // scalesmask2: picks odd-indexed bytes (1,3,5,...,15), each repeated twice
    let scalesmask2_sse = _mm_set_epi8(15,15,13,13,11,11,9,9,7,7,5,5,3,3,1,1);

    let b_ptr_start = vx as *const BlockQ2Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    for y in 0..nr / 4 {
        let a_ptr = a_ptr_start.add(y * nb);

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);

            let mut acc_rows     = [_mm256_setzero_ps(); 4];
            let mut acc_min_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb {
                let bp = &*b_ptr.add(b);
                let ap = &*a_ptr.add(b);

                // Load fp16 super-block scale and dmin for 8 weight columns
                let col_scale_f32 = _mm256_cvtph_ps(_mm_loadu_si128(bp.d.as_ptr() as *const __m128i));
                let col_dmin_f32  = _mm256_cvtph_ps(_mm_loadu_si128(bp.dmin.as_ptr() as *const __m128i));

                // Two sub-block iterations per super-block (QK_K/128 = 2)
                for sb in 0..QK_K / 128 {
                    let qs_base  = bp.qs.as_ptr().add(sb * 256);
                    let sc_base  = bp.scales.as_ptr().add(sb * 64);
                    let lhs_base = (ap.qs.as_ptr() as *const u8).add(512 * sb);
                    let bs_base  = ap.bsums.as_ptr().add(32 * sb);

                    // ── Load raw RHS (weight) bytes ───────────────────────────────────────
                    let rhs_raw_mat_0123_0 = _mm256_loadu_si256(qs_base.add(  0) as *const __m256i);
                    let rhs_raw_mat_4567_0 = _mm256_loadu_si256(qs_base.add( 32) as *const __m256i);
                    let rhs_raw_mat_0123_1 = _mm256_loadu_si256(qs_base.add( 64) as *const __m256i);
                    let rhs_raw_mat_4567_1 = _mm256_loadu_si256(qs_base.add( 96) as *const __m256i);
                    let rhs_raw_mat_0123_2 = _mm256_loadu_si256(qs_base.add(128) as *const __m256i);
                    let rhs_raw_mat_4567_2 = _mm256_loadu_si256(qs_base.add(160) as *const __m256i);
                    let rhs_raw_mat_0123_3 = _mm256_loadu_si256(qs_base.add(192) as *const __m256i);
                    let rhs_raw_mat_4567_3 = _mm256_loadu_si256(qs_base.add(224) as *const __m256i);

                    // ── Rearrange to B0B1B4B5 / B2B3B6B7 interleaving ────────────────────
                    let rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, required_order), 240);
                    let rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, required_order), rhs_raw_mat_4567_0, 240);
                    let rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, required_order), 240);
                    let rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, required_order), rhs_raw_mat_4567_1, 240);
                    let rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, required_order), 240);
                    let rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, required_order), rhs_raw_mat_4567_2, 240);
                    let rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, required_order), 240);
                    let rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, required_order), rhs_raw_mat_4567_3, 240);

                    // ── Extract 2-bit sub-blocks (8 sub-blocks × 2 halves each) ──────────
                    // Sub-block 0 (bits 1:0 of raw_*_0 and raw_*_1)
                    let rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m3b);
                    let rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m3b);
                    let rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m3b);
                    let rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m3b);
                    // Sub-block 1 (bits 1:0 of raw_*_2 and raw_*_3)
                    let rhs_mat_0145_10 = _mm256_and_si256(rhs_raw_mat_0145_2, m3b);
                    let rhs_mat_2367_10 = _mm256_and_si256(rhs_raw_mat_2367_2, m3b);
                    let rhs_mat_0145_11 = _mm256_and_si256(rhs_raw_mat_0145_3, m3b);
                    let rhs_mat_2367_11 = _mm256_and_si256(rhs_raw_mat_2367_3, m3b);
                    // Sub-block 2 (bits 3:2 of raw_*_0 and raw_*_1)
                    let rhs_mat_0145_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 2), m3b);
                    let rhs_mat_2367_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 2), m3b);
                    let rhs_mat_0145_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 2), m3b);
                    let rhs_mat_2367_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 2), m3b);
                    // Sub-block 3 (bits 3:2 of raw_*_2 and raw_*_3)
                    let rhs_mat_0145_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 2), m3b);
                    let rhs_mat_2367_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 2), m3b);
                    let rhs_mat_0145_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 2), m3b);
                    let rhs_mat_2367_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 2), m3b);
                    // Sub-block 4 (bits 5:4 of raw_*_0 and raw_*_1)
                    let rhs_mat_0145_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m3b);
                    let rhs_mat_2367_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m3b);
                    let rhs_mat_0145_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m3b);
                    let rhs_mat_2367_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m3b);
                    // Sub-block 5 (bits 5:4 of raw_*_2 and raw_*_3)
                    let rhs_mat_0145_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m3b);
                    let rhs_mat_2367_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m3b);
                    let rhs_mat_0145_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m3b);
                    let rhs_mat_2367_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m3b);
                    // Sub-block 6 (bits 7:6 of raw_*_0 and raw_*_1)
                    let rhs_mat_0145_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 6), m3b);
                    let rhs_mat_2367_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 6), m3b);
                    let rhs_mat_0145_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 6), m3b);
                    let rhs_mat_2367_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 6), m3b);
                    // Sub-block 7 (bits 7:6 of raw_*_2 and raw_*_3)
                    let rhs_mat_0145_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 6), m3b);
                    let rhs_mat_2367_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 6), m3b);
                    let rhs_mat_0145_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 6), m3b);
                    let rhs_mat_2367_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 6), m3b);

                    // ── RHS shuffle pattern 1 (imm8=136 = 0b10_00_10_00) ─────────────────
                    let rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136);
                    let rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136);
                    let rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136);
                    let rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136);
                    let rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136);
                    let rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136);
                    let rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136);
                    let rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136);
                    let rhs_mat_0145_20_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_20, 136);
                    let rhs_mat_2367_20_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_20, 136);
                    let rhs_mat_0145_21_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_21, 136);
                    let rhs_mat_2367_21_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_21, 136);
                    let rhs_mat_0145_30_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_30, 136);
                    let rhs_mat_2367_30_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_30, 136);
                    let rhs_mat_0145_31_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_31, 136);
                    let rhs_mat_2367_31_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_31, 136);
                    let rhs_mat_0145_40_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_40, 136);
                    let rhs_mat_2367_40_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_40, 136);
                    let rhs_mat_0145_41_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_41, 136);
                    let rhs_mat_2367_41_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_41, 136);
                    let rhs_mat_0145_50_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_50, 136);
                    let rhs_mat_2367_50_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_50, 136);
                    let rhs_mat_0145_51_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_51, 136);
                    let rhs_mat_2367_51_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_51, 136);
                    let rhs_mat_0145_60_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_60, 136);
                    let rhs_mat_2367_60_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_60, 136);
                    let rhs_mat_0145_61_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_61, 136);
                    let rhs_mat_2367_61_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_61, 136);
                    let rhs_mat_0145_70_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_70, 136);
                    let rhs_mat_2367_70_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_70, 136);
                    let rhs_mat_0145_71_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_71, 136);
                    let rhs_mat_2367_71_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_71, 136);

                    // ── RHS shuffle pattern 2 (imm8=221 = 0b11_01_11_01) ─────────────────
                    let rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221);
                    let rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221);
                    let rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221);
                    let rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221);
                    let rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221);
                    let rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221);
                    let rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221);
                    let rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221);
                    let rhs_mat_0145_20_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_20, 221);
                    let rhs_mat_2367_20_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_20, 221);
                    let rhs_mat_0145_21_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_21, 221);
                    let rhs_mat_2367_21_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_21, 221);
                    let rhs_mat_0145_30_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_30, 221);
                    let rhs_mat_2367_30_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_30, 221);
                    let rhs_mat_0145_31_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_31, 221);
                    let rhs_mat_2367_31_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_31, 221);
                    let rhs_mat_0145_40_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_40, 221);
                    let rhs_mat_2367_40_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_40, 221);
                    let rhs_mat_0145_41_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_41, 221);
                    let rhs_mat_2367_41_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_41, 221);
                    let rhs_mat_0145_50_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_50, 221);
                    let rhs_mat_2367_50_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_50, 221);
                    let rhs_mat_0145_51_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_51, 221);
                    let rhs_mat_2367_51_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_51, 221);
                    let rhs_mat_0145_60_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_60, 221);
                    let rhs_mat_2367_60_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_60, 221);
                    let rhs_mat_0145_61_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_61, 221);
                    let rhs_mat_2367_61_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_61, 221);
                    let rhs_mat_0145_70_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_70, 221);
                    let rhs_mat_2367_70_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_70, 221);
                    let rhs_mat_0145_71_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_71, 221);
                    let rhs_mat_2367_71_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_71, 221);

                    // ── Load packed scales+mins (4-bit each) ──────────────────────────────
                    // Layout: s00 m00 s01 m01 s10 m10 ... for 8 sub-block pairs
                    let mins_and_scales_01 = _mm_loadu_si128(sc_base.add( 0) as *const __m128i);
                    let mins_and_scales_23 = _mm_loadu_si128(sc_base.add(16) as *const __m128i);
                    let mins_and_scales_45 = _mm_loadu_si128(sc_base.add(32) as *const __m128i);
                    let mins_and_scales_67 = _mm_loadu_si128(sc_base.add(48) as *const __m128i);

                    // Lower nibbles → scales; upper nibbles → mins
                    let scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    let scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    let scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    let scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    let mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    let mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    let mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    let mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    // Expand each 4-bit scale to 16-bit (for madd_epi16)
                    let scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask1_sse));
                    let scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask2_sse));
                    let scales_2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask1_sse));
                    let scales_3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask2_sse));
                    let scales_4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask1_sse));
                    let scales_5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask2_sse));
                    let scales_6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask1_sse));
                    let scales_7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask2_sse));

                    // Shuffle scales into 0145 and 2367 column formats (imm8=68=0b01_00_01_00 and 238=0b11_10_11_10)
                    let scale_0145_0 = _mm256_shuffle_epi32(scales_0,  68);
                    let scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);
                    let scale_0145_1 = _mm256_shuffle_epi32(scales_1,  68);
                    let scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);
                    let scale_0145_2 = _mm256_shuffle_epi32(scales_2,  68);
                    let scale_2367_2 = _mm256_shuffle_epi32(scales_2, 238);
                    let scale_0145_3 = _mm256_shuffle_epi32(scales_3,  68);
                    let scale_2367_3 = _mm256_shuffle_epi32(scales_3, 238);
                    let scale_0145_4 = _mm256_shuffle_epi32(scales_4,  68);
                    let scale_2367_4 = _mm256_shuffle_epi32(scales_4, 238);
                    let scale_0145_5 = _mm256_shuffle_epi32(scales_5,  68);
                    let scale_2367_5 = _mm256_shuffle_epi32(scales_5, 238);
                    let scale_0145_6 = _mm256_shuffle_epi32(scales_6,  68);
                    let scale_2367_6 = _mm256_shuffle_epi32(scales_6, 238);
                    let scale_0145_7 = _mm256_shuffle_epi32(scales_7,  68);
                    let scale_2367_7 = _mm256_shuffle_epi32(scales_7, 238);

                    // ── Load LHS (activation) bytes ───────────────────────────────────────
                    // 8 pairs × 32 bytes each = 256 bytes per half super-block
                    let lhs_mat_0123_00 = _mm256_loadu_si256(lhs_base.add(  0) as *const __m256i);
                    let lhs_mat_01_00   = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00,  0);
                    let lhs_mat_23_00   = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    let lhs_mat_0123_01 = _mm256_loadu_si256(lhs_base.add( 32) as *const __m256i);
                    let lhs_mat_01_01   = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01,  0);
                    let lhs_mat_23_01   = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    let lhs_mat_0123_10 = _mm256_loadu_si256(lhs_base.add( 64) as *const __m256i);
                    let lhs_mat_01_10   = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10,  0);
                    let lhs_mat_23_10   = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    let lhs_mat_0123_11 = _mm256_loadu_si256(lhs_base.add( 96) as *const __m256i);
                    let lhs_mat_01_11   = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11,  0);
                    let lhs_mat_23_11   = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    let lhs_mat_0123_20 = _mm256_loadu_si256(lhs_base.add(128) as *const __m256i);
                    let lhs_mat_01_20   = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20,  0);
                    let lhs_mat_23_20   = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 17);
                    let lhs_mat_0123_21 = _mm256_loadu_si256(lhs_base.add(160) as *const __m256i);
                    let lhs_mat_01_21   = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21,  0);
                    let lhs_mat_23_21   = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 17);
                    let lhs_mat_0123_30 = _mm256_loadu_si256(lhs_base.add(192) as *const __m256i);
                    let lhs_mat_01_30   = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30,  0);
                    let lhs_mat_23_30   = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 17);
                    let lhs_mat_0123_31 = _mm256_loadu_si256(lhs_base.add(224) as *const __m256i);
                    let lhs_mat_01_31   = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31,  0);
                    let lhs_mat_23_31   = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 17);
                    let lhs_mat_0123_40 = _mm256_loadu_si256(lhs_base.add(256) as *const __m256i);
                    let lhs_mat_01_40   = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40,  0);
                    let lhs_mat_23_40   = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 17);
                    let lhs_mat_0123_41 = _mm256_loadu_si256(lhs_base.add(288) as *const __m256i);
                    let lhs_mat_01_41   = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41,  0);
                    let lhs_mat_23_41   = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 17);
                    let lhs_mat_0123_50 = _mm256_loadu_si256(lhs_base.add(320) as *const __m256i);
                    let lhs_mat_01_50   = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50,  0);
                    let lhs_mat_23_50   = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 17);
                    let lhs_mat_0123_51 = _mm256_loadu_si256(lhs_base.add(352) as *const __m256i);
                    let lhs_mat_01_51   = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51,  0);
                    let lhs_mat_23_51   = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 17);
                    let lhs_mat_0123_60 = _mm256_loadu_si256(lhs_base.add(384) as *const __m256i);
                    let lhs_mat_01_60   = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60,  0);
                    let lhs_mat_23_60   = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 17);
                    let lhs_mat_0123_61 = _mm256_loadu_si256(lhs_base.add(416) as *const __m256i);
                    let lhs_mat_01_61   = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61,  0);
                    let lhs_mat_23_61   = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 17);
                    let lhs_mat_0123_70 = _mm256_loadu_si256(lhs_base.add(448) as *const __m256i);
                    let lhs_mat_01_70   = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70,  0);
                    let lhs_mat_23_70   = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 17);
                    let lhs_mat_0123_71 = _mm256_loadu_si256(lhs_base.add(480) as *const __m256i);
                    let lhs_mat_01_71   = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71,  0);
                    let lhs_mat_23_71   = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 17);

                    // ── Load bsums (i16; 4 groups of 8) ──────────────────────────────────
                    let lhs_raw_bsums_01_0123 = _mm_loadu_si128(bs_base.add( 0) as *const __m128i);
                    let lhs_raw_bsums_23_0123 = _mm_loadu_si128(bs_base.add( 8) as *const __m128i);
                    let lhs_raw_bsums_01_4567 = _mm_loadu_si128(bs_base.add(16) as *const __m128i);
                    let lhs_raw_bsums_23_4567 = _mm_loadu_si128(bs_base.add(24) as *const __m128i);

                    // ── LHS shuffle pattern 1 (imm8=160 = 0b10_10_00_00) ─────────────────
                    let lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160);
                    let lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160);
                    let lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);
                    let lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);
                    let lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160);
                    let lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160);
                    let lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);
                    let lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);
                    let lhs_mat_01_20_sp1 = _mm256_shuffle_epi32(lhs_mat_01_20, 160);
                    let lhs_mat_23_20_sp1 = _mm256_shuffle_epi32(lhs_mat_23_20, 160);
                    let lhs_mat_01_21_sp1 = _mm256_shuffle_epi32(lhs_mat_01_21, 160);
                    let lhs_mat_23_21_sp1 = _mm256_shuffle_epi32(lhs_mat_23_21, 160);
                    let lhs_mat_01_30_sp1 = _mm256_shuffle_epi32(lhs_mat_01_30, 160);
                    let lhs_mat_23_30_sp1 = _mm256_shuffle_epi32(lhs_mat_23_30, 160);
                    let lhs_mat_01_31_sp1 = _mm256_shuffle_epi32(lhs_mat_01_31, 160);
                    let lhs_mat_23_31_sp1 = _mm256_shuffle_epi32(lhs_mat_23_31, 160);
                    let lhs_mat_01_40_sp1 = _mm256_shuffle_epi32(lhs_mat_01_40, 160);
                    let lhs_mat_23_40_sp1 = _mm256_shuffle_epi32(lhs_mat_23_40, 160);
                    let lhs_mat_01_41_sp1 = _mm256_shuffle_epi32(lhs_mat_01_41, 160);
                    let lhs_mat_23_41_sp1 = _mm256_shuffle_epi32(lhs_mat_23_41, 160);
                    let lhs_mat_01_50_sp1 = _mm256_shuffle_epi32(lhs_mat_01_50, 160);
                    let lhs_mat_23_50_sp1 = _mm256_shuffle_epi32(lhs_mat_23_50, 160);
                    let lhs_mat_01_51_sp1 = _mm256_shuffle_epi32(lhs_mat_01_51, 160);
                    let lhs_mat_23_51_sp1 = _mm256_shuffle_epi32(lhs_mat_23_51, 160);
                    let lhs_mat_01_60_sp1 = _mm256_shuffle_epi32(lhs_mat_01_60, 160);
                    let lhs_mat_23_60_sp1 = _mm256_shuffle_epi32(lhs_mat_23_60, 160);
                    let lhs_mat_01_61_sp1 = _mm256_shuffle_epi32(lhs_mat_01_61, 160);
                    let lhs_mat_23_61_sp1 = _mm256_shuffle_epi32(lhs_mat_23_61, 160);
                    let lhs_mat_01_70_sp1 = _mm256_shuffle_epi32(lhs_mat_01_70, 160);
                    let lhs_mat_23_70_sp1 = _mm256_shuffle_epi32(lhs_mat_23_70, 160);
                    let lhs_mat_01_71_sp1 = _mm256_shuffle_epi32(lhs_mat_01_71, 160);
                    let lhs_mat_23_71_sp1 = _mm256_shuffle_epi32(lhs_mat_23_71, 160);

                    // ── LHS shuffle pattern 2 (imm8=245 = 0b11_11_01_01) ─────────────────
                    let lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245);
                    let lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245);
                    let lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245);
                    let lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245);
                    let lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245);
                    let lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245);
                    let lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245);
                    let lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245);
                    let lhs_mat_01_20_sp2 = _mm256_shuffle_epi32(lhs_mat_01_20, 245);
                    let lhs_mat_23_20_sp2 = _mm256_shuffle_epi32(lhs_mat_23_20, 245);
                    let lhs_mat_01_21_sp2 = _mm256_shuffle_epi32(lhs_mat_01_21, 245);
                    let lhs_mat_23_21_sp2 = _mm256_shuffle_epi32(lhs_mat_23_21, 245);
                    let lhs_mat_01_30_sp2 = _mm256_shuffle_epi32(lhs_mat_01_30, 245);
                    let lhs_mat_23_30_sp2 = _mm256_shuffle_epi32(lhs_mat_23_30, 245);
                    let lhs_mat_01_31_sp2 = _mm256_shuffle_epi32(lhs_mat_01_31, 245);
                    let lhs_mat_23_31_sp2 = _mm256_shuffle_epi32(lhs_mat_23_31, 245);
                    let lhs_mat_01_40_sp2 = _mm256_shuffle_epi32(lhs_mat_01_40, 245);
                    let lhs_mat_23_40_sp2 = _mm256_shuffle_epi32(lhs_mat_23_40, 245);
                    let lhs_mat_01_41_sp2 = _mm256_shuffle_epi32(lhs_mat_01_41, 245);
                    let lhs_mat_23_41_sp2 = _mm256_shuffle_epi32(lhs_mat_23_41, 245);
                    let lhs_mat_01_50_sp2 = _mm256_shuffle_epi32(lhs_mat_01_50, 245);
                    let lhs_mat_23_50_sp2 = _mm256_shuffle_epi32(lhs_mat_23_50, 245);
                    let lhs_mat_01_51_sp2 = _mm256_shuffle_epi32(lhs_mat_01_51, 245);
                    let lhs_mat_23_51_sp2 = _mm256_shuffle_epi32(lhs_mat_23_51, 245);
                    let lhs_mat_01_60_sp2 = _mm256_shuffle_epi32(lhs_mat_01_60, 245);
                    let lhs_mat_23_60_sp2 = _mm256_shuffle_epi32(lhs_mat_23_60, 245);
                    let lhs_mat_01_61_sp2 = _mm256_shuffle_epi32(lhs_mat_01_61, 245);
                    let lhs_mat_23_61_sp2 = _mm256_shuffle_epi32(lhs_mat_23_61, 245);
                    let lhs_mat_01_70_sp2 = _mm256_shuffle_epi32(lhs_mat_01_70, 245);
                    let lhs_mat_23_70_sp2 = _mm256_shuffle_epi32(lhs_mat_23_70, 245);
                    let lhs_mat_01_71_sp2 = _mm256_shuffle_epi32(lhs_mat_01_71, 245);
                    let lhs_mat_23_71_sp2 = _mm256_shuffle_epi32(lhs_mat_23_71, 245);

                    // ── Dot products via maddubs (unsigned × signed → i16, summed pairs) ─
                    // Each iacc_mat_XY_K_spP accumulates 2-bit quant × int8 for sub-block K
                    // X=0 → LHS rows 0,1; X=1 → LHS rows 2,3
                    // Y=0 → RHS cols 0,1,4,5; Y=1 → RHS cols 2,3,6,7
                    let iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1));
                    let iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1));
                    let iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1));
                    let iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1));
                    let iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1));
                    let iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1));
                    let iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1));
                    let iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1));
                    let iacc_mat_00_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_01_20_sp1), _mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_01_21_sp1));
                    let iacc_mat_01_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_01_20_sp1), _mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_01_21_sp1));
                    let iacc_mat_10_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_23_20_sp1), _mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_23_21_sp1));
                    let iacc_mat_11_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_23_20_sp1), _mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_23_21_sp1));
                    let iacc_mat_00_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_01_30_sp1), _mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_01_31_sp1));
                    let iacc_mat_01_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_01_30_sp1), _mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_01_31_sp1));
                    let iacc_mat_10_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_23_30_sp1), _mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_23_31_sp1));
                    let iacc_mat_11_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_23_30_sp1), _mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_23_31_sp1));
                    let iacc_mat_00_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_01_40_sp1), _mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_01_41_sp1));
                    let iacc_mat_01_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_01_40_sp1), _mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_01_41_sp1));
                    let iacc_mat_10_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_23_40_sp1), _mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_23_41_sp1));
                    let iacc_mat_11_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_23_40_sp1), _mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_23_41_sp1));
                    let iacc_mat_00_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_01_50_sp1), _mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_01_51_sp1));
                    let iacc_mat_01_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_01_50_sp1), _mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_01_51_sp1));
                    let iacc_mat_10_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_23_50_sp1), _mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_23_51_sp1));
                    let iacc_mat_11_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_23_50_sp1), _mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_23_51_sp1));
                    let iacc_mat_00_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_01_60_sp1), _mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_01_61_sp1));
                    let iacc_mat_01_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_01_60_sp1), _mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_01_61_sp1));
                    let iacc_mat_10_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_23_60_sp1), _mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_23_61_sp1));
                    let iacc_mat_11_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_23_60_sp1), _mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_23_61_sp1));
                    let iacc_mat_00_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_01_70_sp1), _mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_01_71_sp1));
                    let iacc_mat_01_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_01_70_sp1), _mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_01_71_sp1));
                    let iacc_mat_10_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_23_70_sp1), _mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_23_71_sp1));
                    let iacc_mat_11_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_23_70_sp1), _mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_23_71_sp1));

                    let iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2));
                    let iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2));
                    let iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2));
                    let iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2));
                    let iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2));
                    let iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2));
                    let iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2));
                    let iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2));
                    let iacc_mat_00_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_01_20_sp2), _mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_01_21_sp2));
                    let iacc_mat_01_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_01_20_sp2), _mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_01_21_sp2));
                    let iacc_mat_10_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_23_20_sp2), _mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_23_21_sp2));
                    let iacc_mat_11_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_23_20_sp2), _mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_23_21_sp2));
                    let iacc_mat_00_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_01_30_sp2), _mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_01_31_sp2));
                    let iacc_mat_01_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_01_30_sp2), _mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_01_31_sp2));
                    let iacc_mat_10_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_23_30_sp2), _mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_23_31_sp2));
                    let iacc_mat_11_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_23_30_sp2), _mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_23_31_sp2));
                    let iacc_mat_00_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_01_40_sp2), _mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_01_41_sp2));
                    let iacc_mat_01_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_01_40_sp2), _mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_01_41_sp2));
                    let iacc_mat_10_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_23_40_sp2), _mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_23_41_sp2));
                    let iacc_mat_11_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_23_40_sp2), _mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_23_41_sp2));
                    let iacc_mat_00_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_01_50_sp2), _mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_01_51_sp2));
                    let iacc_mat_01_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_01_50_sp2), _mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_01_51_sp2));
                    let iacc_mat_10_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_23_50_sp2), _mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_23_51_sp2));
                    let iacc_mat_11_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_23_50_sp2), _mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_23_51_sp2));
                    let iacc_mat_00_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_01_60_sp2), _mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_01_61_sp2));
                    let iacc_mat_01_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_01_60_sp2), _mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_01_61_sp2));
                    let iacc_mat_10_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_23_60_sp2), _mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_23_61_sp2));
                    let iacc_mat_11_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_23_60_sp2), _mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_23_61_sp2));
                    let iacc_mat_00_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_01_70_sp2), _mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_01_71_sp2));
                    let iacc_mat_01_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_01_70_sp2), _mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_01_71_sp2));
                    let iacc_mat_10_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_23_70_sp2), _mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_23_71_sp2));
                    let iacc_mat_11_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_23_70_sp2), _mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_23_71_sp2));

                    // ── Combine sp1 and sp2 ───────────────────────────────────────────────
                    let mut iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    let mut iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    let mut iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    let mut iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                    let mut iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    let mut iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    let mut iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    let mut iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);
                    let mut iacc_mat_00_2 = _mm256_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                    let mut iacc_mat_01_2 = _mm256_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                    let mut iacc_mat_10_2 = _mm256_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                    let mut iacc_mat_11_2 = _mm256_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);
                    let mut iacc_mat_00_3 = _mm256_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                    let mut iacc_mat_01_3 = _mm256_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                    let mut iacc_mat_10_3 = _mm256_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                    let mut iacc_mat_11_3 = _mm256_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);
                    let mut iacc_mat_00_4 = _mm256_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                    let mut iacc_mat_01_4 = _mm256_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                    let mut iacc_mat_10_4 = _mm256_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                    let mut iacc_mat_11_4 = _mm256_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);
                    let mut iacc_mat_00_5 = _mm256_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                    let mut iacc_mat_01_5 = _mm256_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                    let mut iacc_mat_10_5 = _mm256_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                    let mut iacc_mat_11_5 = _mm256_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);
                    let mut iacc_mat_00_6 = _mm256_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                    let mut iacc_mat_01_6 = _mm256_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                    let mut iacc_mat_10_6 = _mm256_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                    let mut iacc_mat_11_6 = _mm256_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);
                    let mut iacc_mat_00_7 = _mm256_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                    let mut iacc_mat_01_7 = _mm256_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                    let mut iacc_mat_10_7 = _mm256_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                    let mut iacc_mat_11_7 = _mm256_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                    // ── Apply per-sub-block scales via madd_epi16 ─────────────────────────
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);
                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);
                    iacc_mat_00_2 = _mm256_madd_epi16(iacc_mat_00_2, scale_0145_2);
                    iacc_mat_01_2 = _mm256_madd_epi16(iacc_mat_01_2, scale_2367_2);
                    iacc_mat_10_2 = _mm256_madd_epi16(iacc_mat_10_2, scale_0145_2);
                    iacc_mat_11_2 = _mm256_madd_epi16(iacc_mat_11_2, scale_2367_2);
                    iacc_mat_00_3 = _mm256_madd_epi16(iacc_mat_00_3, scale_0145_3);
                    iacc_mat_01_3 = _mm256_madd_epi16(iacc_mat_01_3, scale_2367_3);
                    iacc_mat_10_3 = _mm256_madd_epi16(iacc_mat_10_3, scale_0145_3);
                    iacc_mat_11_3 = _mm256_madd_epi16(iacc_mat_11_3, scale_2367_3);
                    iacc_mat_00_4 = _mm256_madd_epi16(iacc_mat_00_4, scale_0145_4);
                    iacc_mat_01_4 = _mm256_madd_epi16(iacc_mat_01_4, scale_2367_4);
                    iacc_mat_10_4 = _mm256_madd_epi16(iacc_mat_10_4, scale_0145_4);
                    iacc_mat_11_4 = _mm256_madd_epi16(iacc_mat_11_4, scale_2367_4);
                    iacc_mat_00_5 = _mm256_madd_epi16(iacc_mat_00_5, scale_0145_5);
                    iacc_mat_01_5 = _mm256_madd_epi16(iacc_mat_01_5, scale_2367_5);
                    iacc_mat_10_5 = _mm256_madd_epi16(iacc_mat_10_5, scale_0145_5);
                    iacc_mat_11_5 = _mm256_madd_epi16(iacc_mat_11_5, scale_2367_5);
                    iacc_mat_00_6 = _mm256_madd_epi16(iacc_mat_00_6, scale_0145_6);
                    iacc_mat_01_6 = _mm256_madd_epi16(iacc_mat_01_6, scale_2367_6);
                    iacc_mat_10_6 = _mm256_madd_epi16(iacc_mat_10_6, scale_0145_6);
                    iacc_mat_11_6 = _mm256_madd_epi16(iacc_mat_11_6, scale_2367_6);
                    iacc_mat_00_7 = _mm256_madd_epi16(iacc_mat_00_7, scale_0145_7);
                    iacc_mat_01_7 = _mm256_madd_epi16(iacc_mat_01_7, scale_2367_7);
                    iacc_mat_10_7 = _mm256_madd_epi16(iacc_mat_10_7, scale_0145_7);
                    iacc_mat_11_7 = _mm256_madd_epi16(iacc_mat_11_7, scale_2367_7);

                    // ── Sum all 8 sub-block partial sums ──────────────────────────────────
                    let iacc_mat_00 = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm256_add_epi32(iacc_mat_00_2, iacc_mat_00_3)),
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm256_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                    let iacc_mat_01 = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm256_add_epi32(iacc_mat_01_2, iacc_mat_01_3)),
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm256_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                    let iacc_mat_10 = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm256_add_epi32(iacc_mat_10_2, iacc_mat_10_3)),
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm256_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                    let iacc_mat_11 = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm256_add_epi32(iacc_mat_11_2, iacc_mat_11_3)),
                        _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm256_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                    // ── Straighten interleaved 00/01/10/11 into 4 output row vectors ──────
                    let iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    let iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    let iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    let iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Broadcast d[0..4] from BlockQ8Kx4 into a 256-bit vector
                    let row_scale_f32_sse = _mm_loadu_ps(ap.d.as_ptr());
                    let row_scale_f32 = {
                        let v = _mm256_castps128_ps256(row_scale_f32_sse);
                        _mm256_insertf128_ps(v, row_scale_f32_sse, 1)
                    };

                    // ── Accumulate weighted dot products ──────────────────────────────────
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    // ── Min correction via bsums ──────────────────────────────────────────
                    // Broadcast each raw bsum 128-bit load into 256-bit
                    let lhs_bsums_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                    let lhs_bsums_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                    let lhs_bsums_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                    let lhs_bsums_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);

                    // Pair each bsum with its sub-block min and accumulate
                    let iacc_row_min_0_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123,   0), mins_01);
                    let iacc_row_min_1_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 170), mins_01);
                    let iacc_row_min_2_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123,   0), mins_01);
                    let iacc_row_min_3_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 170), mins_01);
                    let iacc_row_min_0_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123,  85), mins_23);
                    let iacc_row_min_1_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 255), mins_23);
                    let iacc_row_min_2_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123,  85), mins_23);
                    let iacc_row_min_3_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 255), mins_23);
                    let iacc_row_min_0_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567,   0), mins_45);
                    let iacc_row_min_1_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 170), mins_45);
                    let iacc_row_min_2_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567,   0), mins_45);
                    let iacc_row_min_3_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 170), mins_45);
                    let iacc_row_min_0_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567,  85), mins_67);
                    let iacc_row_min_1_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 255), mins_67);
                    let iacc_row_min_2_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567,  85), mins_67);
                    let iacc_row_min_3_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 255), mins_67);

                    let iacc_row_min_0 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm256_add_epi32(iacc_row_min_0_45, iacc_row_min_0_67));
                    let iacc_row_min_1 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm256_add_epi32(iacc_row_min_1_45, iacc_row_min_1_67));
                    let iacc_row_min_2 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm256_add_epi32(iacc_row_min_2_45, iacc_row_min_2_67));
                    let iacc_row_min_3 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm256_add_epi32(iacc_row_min_3_45, iacc_row_min_3_67));

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                } // end sb loop
            } // end b loop

            // ── Store 4 output rows × 8 columns ──────────────────────────────────
            for i in 0..4 {
                _mm256_storeu_ps(
                    s.add((y * 4 + i) * bs + x * 8),
                    _mm256_sub_ps(acc_rows[i], acc_min_rows[i]),
                );
            }
        } // end x loop
    } // end y loop
}

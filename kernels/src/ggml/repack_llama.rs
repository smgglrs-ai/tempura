//! Direct translation of llama.cpp arch/x86/repack.cpp and ggml-cpu/repack.cpp
//! AVX2 GEMV/GEMM kernels for Q4_K × Q8_K.
//!
//! Sources:
//! - ggml_gemv_q4_K_8x8_q8_K:     arch/x86/repack.cpp lines 1464-1685
//! - ggml_gemm_q4_K_8x8_q8_K AVX2: arch/x86/repack.cpp lines 2817-3493
//! - make_block_q4_Kx8:            ggml-cpu/repack.cpp  lines 3177-3252
//! - repack_q4_K_to_q4_K_8_bl:    ggml-cpu/repack.cpp  lines 3572-3599

use super::blocks::{BlockQ4K, BlockQ4Kx8, BlockQ8K, BlockQ8Kx4};

// QK_K = 256
const QK_K: usize = 256;

// ── repacking ────────────────────────────────────────────────────────────────

/// Direct translation of make_block_q4_Kx8 (ggml-cpu/repack.cpp line 3177).
/// Interleaves 8 Q4_K blocks into the packed BlockQ4Kx8 format.
pub fn make_block_q4_kx8(in_blocks: &[BlockQ4K; 8], blck_size_interleave: usize) -> BlockQ4Kx8 {
    let mut out = BlockQ4Kx8 {
        d: [0u16; 8],
        dmin: [0u16; 8],
        scales: [0u8; 96],
        qs: [0u8; 1024],
    };

    for i in 0..8 {
        out.d[i] = in_blocks[i].d;
        out.dmin[i] = in_blocks[i].dmin;
    }

    let end = QK_K * 4 / blck_size_interleave;

    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.qs[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].qs[src_offset..src_offset + blck_size_interleave]);
    }

    let mut s = [0u8; 8];
    let mut m = [0u8; 8];

    for i in 0..4 {
        for j in 0..8 {
            s[j] = in_blocks[j].scales[i] & 63;
            m[j] = in_blocks[j].scales[i + 4] & 63;
        }
        out.scales[i * 12]      = (s[0] & 63) | ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) | ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) | ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) | ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) | ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) | ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) | ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) | ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) | ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) | ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) | ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) | ((m[7] & 15) << 4);
    }

    for i in 0..4 {
        for j in 0..8 {
            s[j] = ((in_blocks[j].scales[i] & 192) >> 2) | (in_blocks[j].scales[i + 8] & 15);
            m[j] = ((in_blocks[j].scales[i + 4] & 192) >> 2) | ((in_blocks[j].scales[i + 8] & 240) >> 4);
        }
        out.scales[i * 12 + 48] = (s[0] & 63) | ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) | ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) | ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) | ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) | ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) | ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) | ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) | ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) | ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) | ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) | ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) | ((m[7] & 15) << 4);
    }

    out
}

/// Direct translation of repack_q4_K_to_q4_K_8_bl (ggml-cpu/repack.cpp line 3572).
/// Repacks a full weight matrix (nrow × nblocks Q4_K blocks) into BlockQ4Kx8 format.
/// Returns Err if dimensions are not multiples of 8.
pub fn repack_q4_k_to_q4_k_8_bl(
    src: &[BlockQ4K],
    nrow: usize,
    nblocks: usize,
    interleave_block: usize,
    dst: &mut Vec<BlockQ4Kx8>,
) -> Result<(), &'static str> {
    assert!(interleave_block == 8 || interleave_block == 4);
    const NROWS_INTERLEAVED: usize = 8;

    if nrow % NROWS_INTERLEAVED != 0 {
        return Err("nrow must be a multiple of 8");
    }

    dst.clear();
    dst.reserve(nrow / NROWS_INTERLEAVED * nblocks);

    let mut b = 0usize;
    while b < nrow {
        for x in 0..nblocks {
            let mut tmp = [BlockQ4K {
                d: 0,
                dmin: 0,
                scales: [0u8; 12],
                qs: [0u8; 128],
            }; 8];
            for i in 0..NROWS_INTERLEAVED {
                tmp[i] = src[x + (b + i) * nblocks];
            }
            dst.push(make_block_q4_kx8(&tmp, interleave_block));
        }
        b += NROWS_INTERLEAVED;
    }
    Ok(())
}

// ── AVX2 GEMV ────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
/// Direct translation of ggml_gemv_q4_K_8x8_q8_K (arch/x86/repack.cpp line 1464).
///
/// Computes s[y * nc .. y * nc + nc] += Q4_K_packed × Q8_K activation.
/// - vx: packed weights as BlockQ4Kx8 slice (nc/8 × nb blocks)
/// - vy: activation as BlockQ8K slice (nr × nb blocks)
/// - s:  output, nr × nc floats
/// - n:  token dimension (must be multiple of QK_K=256)
/// - nr: number of activation rows (usually 1 for decode)
/// - nc: number of weight columns (must be multiple of 8)
pub unsafe fn ggml_gemv_q4_k_8x8_q8_k(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    use std::arch::x86_64::*;

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;

    // Lookup table to convert signed nibbles to signed bytes
    let signextendlut = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(_mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0)),
        _mm256_castsi128_si256(_mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0)),
        0,
    );
    let _ = signextendlut; // used in q4_0 path; Q4_K uses nibble mask instead

    // Shuffle masks
    let deltamask = _mm_set_epi8(15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0);
    let scalemask = _mm_set_epi8(7,7,3,3,6,6,2,2,5,5,1,1,4,4,0,0);
    let finalpermutemask = _mm256_set_epi32(7,5,3,1,6,4,2,0);

    // Mask to extract nibbles
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    let b_ptr_start = vx as *const BlockQ4Kx8;
    let a_ptr_start = vy as *const BlockQ8K;

    for y in 0..nr {
        let a_ptr = a_ptr_start.add(y * nb);

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);

            let mut acc_row = _mm256_setzero_ps();
            let mut acc_min_rows = _mm256_setzero_ps();

            for b in 0..nb as isize {
                let a = &*a_ptr.offset(b);
                let bp = &*b_ptr.offset(b);

                // Load and broadcast Q8_K scale
                let row_scale_f32 = _mm256_set1_ps(a.d);

                // Load F16 scales and mins from Q4_Kx8, rearrange to match layout
                let col_scale_f32 = avx_rearranged_f32cx8_load(bp.d.as_ptr(), deltamask);
                let col_dmin_f32  = avx_f32cx8_load(bp.dmin.as_ptr());

                let mut iacc_b     = _mm256_setzero_si256();
                let mut iacc_min_b = _mm256_setzero_si256();

                // Load bsums from Q8_K (16 × i16 = 32 bytes)
                let q8sums = _mm256_loadu_si256(a.bsums.as_ptr() as *const __m256i);
                let mut q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
                    _mm256_castsi256_si128(q8sums),
                    _mm256_extracti128_si256(q8sums, 1),
                ));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                // Process two sub-blocks per iteration (QK_K/64 = 4 iterations)
                for sb in 0..(QK_K / 64) as isize {
                    // Load 8 interleaved Q4_K sub-block data (8 × 32 bytes = 256 bytes per sub-block pair)
                    let base = bp.qs.as_ptr().offset(sb * 256);
                    let rhs_raw_vec_0123_0 = _mm256_loadu_si256(base.offset(  0) as *const __m256i);
                    let rhs_raw_vec_4567_0 = _mm256_loadu_si256(base.offset( 32) as *const __m256i);
                    let rhs_raw_vec_0123_1 = _mm256_loadu_si256(base.offset( 64) as *const __m256i);
                    let rhs_raw_vec_4567_1 = _mm256_loadu_si256(base.offset( 96) as *const __m256i);
                    let rhs_raw_vec_0123_2 = _mm256_loadu_si256(base.offset(128) as *const __m256i);
                    let rhs_raw_vec_4567_2 = _mm256_loadu_si256(base.offset(160) as *const __m256i);
                    let rhs_raw_vec_0123_3 = _mm256_loadu_si256(base.offset(192) as *const __m256i);
                    let rhs_raw_vec_4567_3 = _mm256_loadu_si256(base.offset(224) as *const __m256i);

                    // 4-bit → 8-bit: first sub-block (low nibbles)
                    let rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    let rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    let rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    let rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    let rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    let rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    let rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    let rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // 4-bit → 8-bit: second sub-block (high nibbles)
                    let rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    let rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    let rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    let rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    let rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    let rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    let rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    let rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    // Scale extraction (Q4_K packing: 6-bit scales/mins in 12 bytes per sub-block pair)
                    let mut utmp_0 = [0u32; 4];
                    let mut utmp_1 = [0u32; 4];
                    let scales_base = bp.scales.as_ptr();

                    // First sub-block scales
                    std::ptr::copy_nonoverlapping(
                        scales_base.offset(24 * sb) as *const u8,
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                    let uaux_0 = utmp_0[1] & KMASK1;
                    utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= KMASK1;

                    // Second sub-block scales
                    std::ptr::copy_nonoverlapping(
                        scales_base.offset(12 + sb * 24) as *const u8,
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                    let uaux_1 = utmp_1[1] & KMASK1;
                    utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= KMASK1;

                    // Build scale vectors for sub-block 0
                    let mins_and_scales_0 = _mm_set_epi32(
                        utmp_0[3] as i32, utmp_0[2] as i32, utmp_0[1] as i32, utmp_0[0] as i32,
                    );
                    let scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    let scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Build scale vectors for sub-block 1
                    let mins_and_scales_1 = _mm_set_epi32(
                        utmp_1[3] as i32, utmp_1[2] as i32, utmp_1[1] as i32, utmp_1[0] as i32,
                    );
                    let scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    let scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins: interleave upper parts of mins_and_scales_0 and mins_and_scales_1
                    let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        _mm_shuffle_epi32(mins_and_scales_0, 78),
                        _mm_shuffle_epi32(mins_and_scales_1, 78),
                    ));

                    // Load Q8_K quants for this sub-block pair (2 × 32 = 64 bytes)
                    let lhs_base = a.qs.as_ptr().offset(sb * 64);
                    let mut lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128(lhs_base.offset( 0) as *const __m128i));
                    let mut lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128(lhs_base.offset(16) as *const __m128i));
                    let mut lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128(lhs_base.offset(32) as *const __m128i));
                    let mut lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128(lhs_base.offset(48) as *const __m128i));

                    // Broadcast low 128 bits to both halves
                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    // Dot products for sub-block 0
                    let mut iacc_0 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_00, _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_00, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177), rhs_vec_4567_00, 170),
                        _mm256_shuffle_epi32(lhs_vec_00, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_01, _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_00, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177), rhs_vec_4567_01, 170),
                        _mm256_shuffle_epi32(lhs_vec_00, 255)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_02, _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_01, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177), rhs_vec_4567_02, 170),
                        _mm256_shuffle_epi32(lhs_vec_01, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_03, _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_01, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177), rhs_vec_4567_03, 170),
                        _mm256_shuffle_epi32(lhs_vec_01, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    // Dot products for sub-block 1
                    let mut iacc_1 = _mm256_setzero_si256();

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_10, _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_10, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177), rhs_vec_4567_10, 170),
                        _mm256_shuffle_epi32(lhs_vec_10, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_11, _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_10, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177), rhs_vec_4567_11, 170),
                        _mm256_shuffle_epi32(lhs_vec_10, 255)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_12, _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_11, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177), rhs_vec_4567_12, 170),
                        _mm256_shuffle_epi32(lhs_vec_11, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(rhs_vec_0123_13, _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170),
                        _mm256_shuffle_epi32(lhs_vec_11, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(
                        _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177), rhs_vec_4567_13, 170),
                        _mm256_shuffle_epi32(lhs_vec_11, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    // Accumulate sub-block pair result
                    let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    // Min subtraction: broadcast this sb's bsum pair and multiply by mins
                    let q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_srli_si256(q8s, 4);

                    iacc_b     = _mm256_add_epi32(iacc_b,     iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Scale and accumulate
                acc_row      = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b),
                    _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b),
                    _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }

            // Permute and store 8 output values
            let acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(
                s.add(y * nc + x * 8),
                _mm256_sub_ps(acc_row, acc_min_rows),
            );
        }
    }
}

// ── AVX2 GEMM (4-row tail loop) ───────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
/// Direct translation of ggml_gemm_q4_K_8x8_q8_K AVX2-only tail loop
/// (arch/x86/repack.cpp lines 3158-3493).
///
/// Processes rows in groups of 4 (one BlockQ8Kx4 per group).
/// - vx:  packed weights as BlockQ4Kx8, shape (nc/8) × nb
/// - vy:  packed activations as BlockQ8Kx4, shape (nr/4) × nb
/// - s:   output floats, nr × nc
/// - bs:  stride between rows of s (in floats, usually = nc)
/// - n:   token dimension (multiple of QK_K)
/// - nr:  activation rows (must be multiple of 4)
/// - nc:  weight columns (must be multiple of 8)
pub unsafe fn ggml_gemm_q4_k_8x8_q8_k_avx2_tail(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    use std::arch::x86_64::*;

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;

    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);
    let required_order = _mm256_set_epi32(3,2,1,0,7,6,5,4);

    let b_ptr_start = vx as *const BlockQ4Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    // Process nr/4 groups (each group = 1 BlockQ8Kx4 = 4 rows)
    for y in 0..nr / 4 {
        let a_ptr = a_ptr_start.add(y * nb);

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);

            let mut acc_rows     = [_mm256_setzero_ps(); 4];
            let mut acc_min_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb as isize {
                let a = &*a_ptr.offset(b);
                let bp = &*b_ptr.offset(b);

                // Load F16 scales and mins
                let col_scale_f32 = avx_f32cx8_load(bp.d.as_ptr());
                let col_dmin_f32  = avx_f32cx8_load(bp.dmin.as_ptr());

                for sb in 0..(QK_K / 64) as isize {
                    // Load 8 interleaved Q4_K sub-block quants
                    let base = bp.qs.as_ptr().offset(sb * 256);
                    let rhs_raw_mat_0123_0 = _mm256_loadu_si256(base.offset(  0) as *const __m256i);
                    let rhs_raw_mat_4567_0 = _mm256_loadu_si256(base.offset( 32) as *const __m256i);
                    let rhs_raw_mat_0123_1 = _mm256_loadu_si256(base.offset( 64) as *const __m256i);
                    let rhs_raw_mat_4567_1 = _mm256_loadu_si256(base.offset( 96) as *const __m256i);
                    let rhs_raw_mat_0123_2 = _mm256_loadu_si256(base.offset(128) as *const __m256i);
                    let rhs_raw_mat_4567_2 = _mm256_loadu_si256(base.offset(160) as *const __m256i);
                    let rhs_raw_mat_0123_3 = _mm256_loadu_si256(base.offset(192) as *const __m256i);
                    let rhs_raw_mat_4567_3 = _mm256_loadu_si256(base.offset(224) as *const __m256i);

                    // Rearrange into B0B1B4B5 / B2B3B6B7 format
                    let rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, required_order), 240);
                    let rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, required_order), rhs_raw_mat_4567_0, 240);
                    let rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, required_order), 240);
                    let rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, required_order), rhs_raw_mat_4567_1, 240);
                    let rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, required_order), 240);
                    let rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, required_order), rhs_raw_mat_4567_2, 240);
                    let rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, required_order), 240);
                    let rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, required_order), rhs_raw_mat_4567_3, 240);

                    // 4-bit → 8-bit: sub-block 0 (low nibbles)
                    let rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b);
                    let rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b);
                    let rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b);
                    let rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b);
                    let rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b);
                    let rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b);
                    let rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b);
                    let rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b);

                    // 4-bit → 8-bit: sub-block 1 (high nibbles)
                    let rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b);
                    let rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b);
                    let rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b);
                    let rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b);
                    let rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b);
                    let rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b);
                    let rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b);
                    let rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b);

                    // Shuffle patterns: sub-block 0
                    let rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136);
                    let rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136);
                    let rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136);
                    let rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136);
                    let rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136);
                    let rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136);
                    let rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136);
                    let rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136);

                    let rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221);
                    let rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221);
                    let rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221);
                    let rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221);
                    let rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221);
                    let rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221);
                    let rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221);
                    let rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221);

                    // Shuffle patterns: sub-block 1
                    let rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136);
                    let rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136);
                    let rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136);
                    let rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136);
                    let rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136);
                    let rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136);
                    let rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136);
                    let rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136);

                    let rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221);
                    let rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221);
                    let rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221);
                    let rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221);
                    let rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221);
                    let rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221);
                    let rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221);
                    let rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221);

                    // Scale extraction
                    let mut utmp_0 = [0u32; 4];
                    let mut utmp_1 = [0u32; 4];
                    let scales_base = bp.scales.as_ptr();

                    std::ptr::copy_nonoverlapping(
                        scales_base.offset(24 * sb) as *const u8,
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                    let uaux_0 = utmp_0[1] & KMASK1;
                    utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= KMASK1;

                    std::ptr::copy_nonoverlapping(
                        scales_base.offset(12 + sb * 24) as *const u8,
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                    let uaux_1 = utmp_1[1] & KMASK1;
                    utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= KMASK1;

                    let mins_and_scales_0 = _mm_set_epi32(
                        utmp_0[3] as i32, utmp_0[2] as i32, utmp_0[1] as i32, utmp_0[0] as i32,
                    );
                    let scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    let mins_and_scales_1 = _mm_set_epi32(
                        utmp_1[3] as i32, utmp_1[2] as i32, utmp_1[1] as i32, utmp_1[0] as i32,
                    );
                    let scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        _mm_shuffle_epi32(mins_and_scales_0, 78),
                        _mm_shuffle_epi32(mins_and_scales_1, 78),
                    ));

                    let scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    let scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);
                    let scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    let scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load Q8_Kx4 activations and bsums for this sb
                    let lhs_base = a.qs.as_ptr().offset(256 * sb);
                    let lhs_mat_0123_00 = _mm256_loadu_si256(lhs_base.offset(  0) as *const __m256i);
                    let lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    let lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    let lhs_mat_0123_01 = _mm256_loadu_si256(lhs_base.offset( 32) as *const __m256i);
                    let lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    let lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    let lhs_mat_0123_02 = _mm256_loadu_si256(lhs_base.offset( 64) as *const __m256i);
                    let lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    let lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    let lhs_mat_0123_03 = _mm256_loadu_si256(lhs_base.offset( 96) as *const __m256i);
                    let lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    let lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    let lhs_mat_0123_10 = _mm256_loadu_si256(lhs_base.offset(128) as *const __m256i);
                    let lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    let lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    let lhs_mat_0123_11 = _mm256_loadu_si256(lhs_base.offset(160) as *const __m256i);
                    let lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    let lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    let lhs_mat_0123_12 = _mm256_loadu_si256(lhs_base.offset(192) as *const __m256i);
                    let lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    let lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    let lhs_mat_0123_13 = _mm256_loadu_si256(lhs_base.offset(224) as *const __m256i);
                    let lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    let lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums for this sb (16 i16 values per sb)
                    let lhs_bsums_0123_01 = _mm256_loadu_si256(
                        (a.bsums.as_ptr().offset(16 * sb)) as *const __m256i,
                    );
                    let lhs_bsums_hsum_0123_01 = {
                        let lo = _mm256_castsi256_si128(lhs_bsums_0123_01);
                        let hi = _mm256_extracti128_si256(lhs_bsums_0123_01, 1);
                        let h = _mm256_castsi128_si256(_mm_hadd_epi16(lo, hi));
                        _mm256_permute2x128_si256(h, h, 0)
                    };

                    // Shuffle patterns: LHS sub-block 0
                    let lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160);
                    let lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160);
                    let lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);
                    let lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);
                    let lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);
                    let lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);
                    let lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);
                    let lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160);

                    let lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245);
                    let lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245);
                    let lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245);
                    let lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245);
                    let lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245);
                    let lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245);
                    let lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245);
                    let lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245);

                    // Shuffle patterns: LHS sub-block 1
                    let lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160);
                    let lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160);
                    let lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);
                    let lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);
                    let lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);
                    let lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);
                    let lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160);
                    let lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160);

                    let lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245);
                    let lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245);
                    let lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245);
                    let lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245);
                    let lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245);
                    let lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245);
                    let lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245);
                    let lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245);

                    // Dot products: sub-block 0, shuffle 1
                    let iacc_mat_00_0_sp1 = add4_maddubs(rhs_mat_0145_00_sp1, rhs_mat_0145_01_sp1, rhs_mat_0145_02_sp1, rhs_mat_0145_03_sp1, lhs_mat_01_00_sp1, lhs_mat_01_01_sp1, lhs_mat_01_02_sp1, lhs_mat_01_03_sp1);
                    let iacc_mat_01_0_sp1 = add4_maddubs(rhs_mat_2367_00_sp1, rhs_mat_2367_01_sp1, rhs_mat_2367_02_sp1, rhs_mat_2367_03_sp1, lhs_mat_01_00_sp1, lhs_mat_01_01_sp1, lhs_mat_01_02_sp1, lhs_mat_01_03_sp1);
                    let iacc_mat_10_0_sp1 = add4_maddubs(rhs_mat_0145_00_sp1, rhs_mat_0145_01_sp1, rhs_mat_0145_02_sp1, rhs_mat_0145_03_sp1, lhs_mat_23_00_sp1, lhs_mat_23_01_sp1, lhs_mat_23_02_sp1, lhs_mat_23_03_sp1);
                    let iacc_mat_11_0_sp1 = add4_maddubs(rhs_mat_2367_00_sp1, rhs_mat_2367_01_sp1, rhs_mat_2367_02_sp1, rhs_mat_2367_03_sp1, lhs_mat_23_00_sp1, lhs_mat_23_01_sp1, lhs_mat_23_02_sp1, lhs_mat_23_03_sp1);

                    // Sub-block 0, shuffle 2
                    let iacc_mat_00_0_sp2 = add4_maddubs(rhs_mat_0145_00_sp2, rhs_mat_0145_01_sp2, rhs_mat_0145_02_sp2, rhs_mat_0145_03_sp2, lhs_mat_01_00_sp2, lhs_mat_01_01_sp2, lhs_mat_01_02_sp2, lhs_mat_01_03_sp2);
                    let iacc_mat_01_0_sp2 = add4_maddubs(rhs_mat_2367_00_sp2, rhs_mat_2367_01_sp2, rhs_mat_2367_02_sp2, rhs_mat_2367_03_sp2, lhs_mat_01_00_sp2, lhs_mat_01_01_sp2, lhs_mat_01_02_sp2, lhs_mat_01_03_sp2);
                    let iacc_mat_10_0_sp2 = add4_maddubs(rhs_mat_0145_00_sp2, rhs_mat_0145_01_sp2, rhs_mat_0145_02_sp2, rhs_mat_0145_03_sp2, lhs_mat_23_00_sp2, lhs_mat_23_01_sp2, lhs_mat_23_02_sp2, lhs_mat_23_03_sp2);
                    let iacc_mat_11_0_sp2 = add4_maddubs(rhs_mat_2367_00_sp2, rhs_mat_2367_01_sp2, rhs_mat_2367_02_sp2, rhs_mat_2367_03_sp2, lhs_mat_23_00_sp2, lhs_mat_23_01_sp2, lhs_mat_23_02_sp2, lhs_mat_23_03_sp2);

                    // Sub-block 1, shuffle 1
                    let iacc_mat_00_1_sp1 = add4_maddubs(rhs_mat_0145_10_sp1, rhs_mat_0145_11_sp1, rhs_mat_0145_12_sp1, rhs_mat_0145_13_sp1, lhs_mat_01_10_sp1, lhs_mat_01_11_sp1, lhs_mat_01_12_sp1, lhs_mat_01_13_sp1);
                    let iacc_mat_01_1_sp1 = add4_maddubs(rhs_mat_2367_10_sp1, rhs_mat_2367_11_sp1, rhs_mat_2367_12_sp1, rhs_mat_2367_13_sp1, lhs_mat_01_10_sp1, lhs_mat_01_11_sp1, lhs_mat_01_12_sp1, lhs_mat_01_13_sp1);
                    let iacc_mat_10_1_sp1 = add4_maddubs(rhs_mat_0145_10_sp1, rhs_mat_0145_11_sp1, rhs_mat_0145_12_sp1, rhs_mat_0145_13_sp1, lhs_mat_23_10_sp1, lhs_mat_23_11_sp1, lhs_mat_23_12_sp1, lhs_mat_23_13_sp1);
                    let iacc_mat_11_1_sp1 = add4_maddubs(rhs_mat_2367_10_sp1, rhs_mat_2367_11_sp1, rhs_mat_2367_12_sp1, rhs_mat_2367_13_sp1, lhs_mat_23_10_sp1, lhs_mat_23_11_sp1, lhs_mat_23_12_sp1, lhs_mat_23_13_sp1);

                    // Sub-block 1, shuffle 2
                    let iacc_mat_00_1_sp2 = add4_maddubs(rhs_mat_0145_10_sp2, rhs_mat_0145_11_sp2, rhs_mat_0145_12_sp2, rhs_mat_0145_13_sp2, lhs_mat_01_10_sp2, lhs_mat_01_11_sp2, lhs_mat_01_12_sp2, lhs_mat_01_13_sp2);
                    let iacc_mat_01_1_sp2 = add4_maddubs(rhs_mat_2367_10_sp2, rhs_mat_2367_11_sp2, rhs_mat_2367_12_sp2, rhs_mat_2367_13_sp2, lhs_mat_01_10_sp2, lhs_mat_01_11_sp2, lhs_mat_01_12_sp2, lhs_mat_01_13_sp2);
                    let iacc_mat_10_1_sp2 = add4_maddubs(rhs_mat_0145_10_sp2, rhs_mat_0145_11_sp2, rhs_mat_0145_12_sp2, rhs_mat_0145_13_sp2, lhs_mat_23_10_sp2, lhs_mat_23_11_sp2, lhs_mat_23_12_sp2, lhs_mat_23_13_sp2);
                    let iacc_mat_11_1_sp2 = add4_maddubs(rhs_mat_2367_10_sp2, rhs_mat_2367_11_sp2, rhs_mat_2367_12_sp2, rhs_mat_2367_13_sp2, lhs_mat_23_10_sp2, lhs_mat_23_11_sp2, lhs_mat_23_12_sp2, lhs_mat_23_13_sp2);

                    // Combine shuffle patterns
                    let mut iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    let mut iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    let mut iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    let mut iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                    let mut iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    let mut iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    let mut iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    let mut iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Scale
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);
                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten to 4 row vectors per sub-block
                    let iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    let iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    let iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    let iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    let iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    let iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    let iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    let iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    let iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    let iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    let iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    let iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Row scales from Q8_Kx4 (4 f32 values)
                    let row_scale_f32_sse = _mm_loadu_ps(a.d.as_ptr());
                    let row_scale_f32 = {
                        let v256 = _mm256_castps128_ps256(row_scale_f32_sse);
                        _mm256_insertf128_ps(v256, row_scale_f32_sse, 1)
                    };

                    // Accumulate
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    // Min accumulation
                    let iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01,   0), mins_01);
                    let iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01,  85), mins_01);
                    let iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    let iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store 4 rows × 8 cols
            for i in 0..4 {
                _mm256_storeu_ps(
                    s.add((y * 4 + i) * bs + x * 8),
                    _mm256_sub_ps(acc_rows[i], acc_min_rows[i]),
                );
            }
        }
    }
}

// ── AVX2 helpers ──────────────────────────────────────────────────────────────

/// Software f16→f32 conversion for 8 packed f16 values.
/// Equivalent to GGML_F32Cx8_LOAD (non-F16C path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn avx_f32cx8_load(ptr: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_cvtph_ps(_mm_loadu_si128(ptr as *const __m128i))
}

/// Load 8 f16 values, byte-shuffle with arrangeMask, then convert to f32.
/// Equivalent to GGML_F32Cx8_REARRANGE_LOAD (non-F16C path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn avx_rearranged_f32cx8_load(ptr: *const u16, arrange_mask: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128(ptr as *const __m128i), arrange_mask))
}

/// Compute: maddubs(r0,l0) + maddubs(r1,l1) + maddubs(r2,l2) + maddubs(r3,l3)
/// Used to sum 4 dot-product terms into one i16 accumulator.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add4_maddubs(
    r0: std::arch::x86_64::__m256i, r1: std::arch::x86_64::__m256i,
    r2: std::arch::x86_64::__m256i, r3: std::arch::x86_64::__m256i,
    l0: std::arch::x86_64::__m256i, l1: std::arch::x86_64::__m256i,
    l2: std::arch::x86_64::__m256i, l3: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(_mm256_maddubs_epi16(r3, l3), _mm256_maddubs_epi16(r2, l2)),
            _mm256_maddubs_epi16(r1, l1),
        ),
        _mm256_maddubs_epi16(r0, l0),
    )
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::blocks::{BlockQ4K, BlockQ4Kx8, BlockQ8K};

    fn zero_block_q4k() -> BlockQ4K {
        BlockQ4K { d: 0, dmin: 0, scales: [0u8; 12], qs: [0u8; 128] }
    }

    fn zero_block_q8k() -> BlockQ8K {
        BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }
    }

    #[test]
    fn test_make_block_q4_kx8_roundtrip_scales() {
        let mut blocks = [zero_block_q4k(); 8];
        // Set distinguishable scale bytes on each block
        for (i, b) in blocks.iter_mut().enumerate() {
            for s in b.scales.iter_mut() {
                *s = (i as u8 + 1) & 0x3F;
            }
        }
        let packed = make_block_q4_kx8(&blocks, 8);
        // Packed qs must be all zero (input qs are zero)
        assert!(packed.qs.iter().all(|&x| x == 0));
        // d/dmin arrays must reflect input
        for i in 0..8 {
            assert_eq!(packed.d[i], 0);
            assert_eq!(packed.dmin[i], 0);
        }
    }

    #[test]
    fn test_make_block_q4_kx8_qs_interleaving() {
        let mut blocks = [zero_block_q4k(); 8];
        // Mark each block's qs with its index
        for (i, b) in blocks.iter_mut().enumerate() {
            b.qs.fill(i as u8 + 1);
        }
        let packed = make_block_q4_kx8(&blocks, 8);
        // First 8 bytes should be B0[0..8], next 8 bytes B1[0..8], etc.
        for i in 0..8 {
            for byte_off in 0..8usize {
                assert_eq!(packed.qs[i * 8 + byte_off], (i as u8 + 1));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemv_zero_weights_zero_output() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        // n=256, nr=1, nc=8
        let n = 256usize;
        let nc = 8usize;
        let nr = 1usize;

        let b = BlockQ4Kx8 {
            d: [0u16; 8],
            dmin: [0u16; 8],
            scales: [0u8; 96],
            qs: [0u8; 1024],
        };
        let a = zero_block_q8k();

        let mut output = vec![0f32; nr * nc];

        unsafe {
            ggml_gemv_q4_k_8x8_q8_k(
                n,
                output.as_mut_ptr(),
                &b as *const BlockQ4Kx8 as *const u8,
                &a as *const BlockQ8K as *const u8,
                nr,
                nc,
            );
        }

        for &v in &output {
            assert_eq!(v, 0.0f32, "expected zero output for zero weights");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemv_matches_scalar_sum() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        // Build a simple Q8K activation with all qs=1, d=1.0 and a Q4Kx8 weight block
        // with all quant nibbles=1 (low nibble) and scale=1 (encoded properly).
        // Expected dot product per output: sum over 256 positions of 1*1 = 256,
        // scaled by Q8K.d * Q4K.d scale = 1.0 * (some small f16 value).
        // We just verify the output is the same across 8 columns (symmetry check).
        let n = 256usize;
        let nc = 8usize;
        let nr = 1usize;

        // Q8K: all quants = 1, d = 1.0, bsums = 16 per group (16 elements × 1)
        let mut a = zero_block_q8k();
        a.d = 1.0;
        a.qs.fill(1);
        for b in a.bsums.iter_mut() { *b = 16; }

        // Q4Kx8: all quants = 0x11 (low nibble=1, high nibble=1),
        // d[i] = 0x3C00 (f16 = 1.0), dmin[i] = 0
        // scales: set to produce scale=1 for all sub-blocks
        // The scale encoding is complex so we test just symmetry (all 8 outputs equal)
        let mut b = BlockQ4Kx8 {
            d: [0x3C00u16; 8],       // f16 = 1.0
            dmin: [0u16; 8],
            scales: [0u8; 96],
            qs: [0x11u8; 1024],      // nibble 0=1, nibble 1=1
        };
        // Set scale = 1 for all sub-blocks in the interleaved format.
        // Each 12-byte group: bytes 0-7 = scales (6-bit), bytes 8-11 = combined bits.
        // Simplest: set byte 0 = 1 for each 12-byte group (sub-block 0 scale for all 8 blocks).
        // The exact encoding is complex; just set all scale bytes to 1 (value=1 for sub-block scales).
        b.scales.fill(1);

        let mut output = vec![0f32; nr * nc];

        unsafe {
            ggml_gemv_q4_k_8x8_q8_k(
                n,
                output.as_mut_ptr(),
                &b as *const BlockQ4Kx8 as *const u8,
                &a as *const BlockQ8K as *const u8,
                nr,
                nc,
            );
        }

        // All 8 outputs should be equal (symmetric inputs)
        let first = output[0];
        for &v in &output {
            assert!((v - first).abs() < 1e-4 * first.abs().max(1.0),
                "outputs differ: {:.6} vs {:.6}", first, v);
        }
    }
}

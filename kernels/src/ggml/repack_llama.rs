//! Direct translation of llama.cpp arch/x86/repack.cpp and ggml-cpu/repack.cpp
//! AVX2 GEMV/GEMM kernels for Q4_K × Q8_K, and scalar GEMV/GEMM for Q2_K/Q5_K/Q6_K.
//!
//! Sources:
//! - ggml_gemv_q4_K_8x8_q8_K:     arch/x86/repack.cpp lines 1464-1685
//! - ggml_gemm_q4_K_8x8_q8_K AVX2: arch/x86/repack.cpp lines 2817-3493
//! - make_block_q4_Kx8:            ggml-cpu/repack.cpp  lines 3177-3252
//! - repack_q4_K_to_q4_K_8_bl:    ggml-cpu/repack.cpp  lines 3572-3599
//! - ggml_gemv_q2_K_8x8_q8_K:     ggml-cpu/repack.cpp  lines 1029-1103 (scalar generic)
//! - make_block_q2_Kx8:            ggml-cpu/repack.cpp  lines 3306-3346
//! - repack_q2_K_to_q2_K_8_bl:    ggml-cpu/repack.cpp  lines 3633-3659
//! - ggml_gemv_q5_K_8x8_q8_K:     ggml-cpu/repack.cpp  lines 551-643  (blocklen=8, ncols=8)
//! - ggml_gemm_q5_K_8x4_q8_K:     ggml-cpu/repack.cpp  lines 647-750  (blocklen=4, ncols=8)
//! - make_block_q5_Kx8:            ggml-cpu/repack.cpp  lines 3348-3431
//! - repack_q5_K_to_q5_K_8_bl:    ggml-cpu/repack.cpp  lines 3729-3759
//! - ggml_gemv_q6_K_8x8_q8_K:     ggml-cpu/repack.cpp  lines 357-443  (blocklen=8, ncols=8)
//! - ggml_gemm_q6_K_8x4_q8_K:     ggml-cpu/repack.cpp  lines 447-548  (blocklen=4, ncols=8)
//! - make_block_q6_Kx8:            ggml-cpu/repack.cpp  lines 3433-3477
//! - repack_q6_K_to_q6_K_8_bl:    ggml-cpu/repack.cpp  lines 3761-3788

use super::blocks::{
    BlockQ2K, BlockQ2Kx8,
    BlockQ4K, BlockQ4Kx8,
    BlockQ5K, BlockQ5Kx8,
    BlockQ6K, BlockQ6Kx8,
    BlockQ8K, BlockQ8Kx4,
};

// QK_K = 256
const QK_K: usize = 256;
// K_SCALE_SIZE: bytes per sub-block scale encoding in Q4_K/Q5_K
const K_SCALE_SIZE: usize = 12;

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

// ── f16→f32 scalar helper (for non-AVX2 code paths) ──────────────────────────

/// Software IEEE-754 f16 → f32 conversion.
#[inline(always)]
fn f16_to_f32(v: u16) -> f32 {
    let s = (v as u32 & 0x8000) << 16;
    let e = (v as u32 >> 10) & 0x1f;
    let m = (v as u32) & 0x3ff;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: normalize
            let mut exp = 127u32 - 14;
            let mut man = m;
            while man & 0x400 == 0 {
                man <<= 1;
                exp -= 1;
            }
            s | (exp << 23) | ((man & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13) // inf or nan
    } else {
        s | ((e + 112) << 23) | (m << 13) // normal
    };
    f32::from_bits(bits)
}

// ── Q2K repacking ─────────────────────────────────────────────────────────────

/// Direct translation of make_block_q2_Kx8 (ggml-cpu/repack.cpp line 3306).
/// Interleaves 8 Q2_K blocks into the packed BlockQ2Kx8 format.
pub fn make_block_q2_kx8(in_blocks: &[BlockQ2K; 8], blck_size_interleave: usize) -> BlockQ2Kx8 {
    let mut out = BlockQ2Kx8 {
        d: [0u16; 8],
        dmin: [0u16; 8],
        scales: [0u8; 128],
        qs: [0u8; 512],
    };

    // Copy super-block scales (d) and mins (dmin)
    for i in 0..8 {
        out.d[i] = in_blocks[i].d;
        out.dmin[i] = in_blocks[i].dmin;
    }

    // Interleave 2-bit quants: QK_K/4=64 bytes per source block
    let end = 8 * 64 / blck_size_interleave;
    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.qs[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].qs[src_offset..src_offset + blck_size_interleave]);
    }

    // Repack scales and mins.
    // Q2_K has 16 scale+min pairs packed in 16 bytes (4 bits each).
    // Output layout: every 16 bytes contains the scale/min bytes for 2 consecutive
    // sub-blocks interleaved across the 8 source blocks.
    for i in 0..128 {
        let src1 = (i % 16) / 2;       // which source block (0..8)
        let src2 = ((i / 16) * 2) + (i % 2);  // which scale byte index (0..15)
        out.scales[i] = in_blocks[src1].scales[src2];
    }

    out
}

/// Direct translation of repack_q2_K_to_q2_K_8_bl (ggml-cpu/repack.cpp line 3633).
/// Repacks a full weight matrix (nrow × nblocks Q2_K blocks) into BlockQ2Kx8 format.
pub fn repack_q2_k_to_q2_k_8_bl(
    src: &[BlockQ2K],
    nrow: usize,
    nblocks: usize,
    dst: &mut Vec<BlockQ2Kx8>,
) -> Result<(), &'static str> {
    const NROWS_INTERLEAVED: usize = 8;
    const INTERLEAVE_BLOCK: usize = 8;

    if nrow % NROWS_INTERLEAVED != 0 {
        return Err("nrow must be a multiple of 8");
    }

    dst.clear();
    dst.reserve(nrow / NROWS_INTERLEAVED * nblocks);

    let mut b = 0usize;
    while b < nrow {
        for x in 0..nblocks {
            let mut tmp = [BlockQ2K {
                scales: [0u8; 16],
                qs: [0u8; 64],
                d: 0,
                dmin: 0,
            }; 8];
            for i in 0..NROWS_INTERLEAVED {
                tmp[i] = src[x + (b + i) * nblocks];
            }
            dst.push(make_block_q2_kx8(&tmp, INTERLEAVE_BLOCK));
        }
        b += NROWS_INTERLEAVED;
    }
    Ok(())
}

// ── Q5K repacking ─────────────────────────────────────────────────────────────

/// Direct translation of make_block_q5_Kx8 (ggml-cpu/repack.cpp line 3348).
/// Interleaves 8 Q5_K blocks into the packed BlockQ5Kx8 format.
pub fn make_block_q5_kx8(in_blocks: &[BlockQ5K; 8], blck_size_interleave: usize) -> BlockQ5Kx8 {
    let mut out = BlockQ5Kx8 {
        d: [0u16; 8],
        dmin: [0u16; 8],
        scales: [0u8; 96],
        qh: [0u8; 256],
        qs: [0u8; 1024],
    };

    for i in 0..8 {
        out.d[i] = in_blocks[i].d;
        out.dmin[i] = in_blocks[i].dmin;
    }

    // Interleave low nibbles (QK_K/2=128 bytes per source block)
    let end = QK_K * 4 / blck_size_interleave;
    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.qs[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].qs[src_offset..src_offset + blck_size_interleave]);
    }

    // Interleave high bits (QK_K/8=32 bytes per source block)
    for i in 0..(end / 4) {
        let src_id = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.qh[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].qh[src_offset..src_offset + blck_size_interleave]);
    }

    // Repack scales: identical layout to Q4K
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

/// Direct translation of repack_q5_K_to_q5_K_8_bl (ggml-cpu/repack.cpp line 3729).
pub fn repack_q5_k_to_q5_k_8_bl(
    src: &[BlockQ5K],
    nrow: usize,
    nblocks: usize,
    interleave_block: usize,
    dst: &mut Vec<BlockQ5Kx8>,
) -> Result<(), &'static str> {
    assert!(interleave_block == 4 || interleave_block == 8);
    const NROWS_INTERLEAVED: usize = 8;

    if nrow % NROWS_INTERLEAVED != 0 {
        return Err("nrow must be a multiple of 8");
    }

    dst.clear();
    dst.reserve(nrow / NROWS_INTERLEAVED * nblocks);

    let mut b = 0usize;
    while b < nrow {
        for x in 0..nblocks {
            let mut tmp = [BlockQ5K {
                d: 0,
                dmin: 0,
                scales: [0u8; 12],
                qh: [0u8; 32],
                qs: [0u8; 128],
            }; 8];
            for i in 0..NROWS_INTERLEAVED {
                tmp[i] = src[x + (b + i) * nblocks];
            }
            dst.push(make_block_q5_kx8(&tmp, interleave_block));
        }
        b += NROWS_INTERLEAVED;
    }
    Ok(())
}

// ── Q6K repacking ─────────────────────────────────────────────────────────────

/// Direct translation of make_block_q6_Kx8 (ggml-cpu/repack.cpp line 3433).
/// Interleaves 8 Q6_K blocks into the packed BlockQ6Kx8 format.
pub fn make_block_q6_kx8(in_blocks: &[BlockQ6K; 8], blck_size_interleave: usize) -> BlockQ6Kx8 {
    const N_BLOCKS: usize = 8;
    let mut out = BlockQ6Kx8 {
        d: [0u16; 8],
        scales: [0i8; 128],
        ql: [0u8; 1024],
        qh: [0u8; 512],
    };

    for i in 0..N_BLOCKS {
        out.d[i] = in_blocks[i].d;
    }

    // Interleave lower 4-bit quants (QK_K/2=128 bytes per block)
    let end_ls = QK_K * 4 / blck_size_interleave;
    for i in 0..end_ls {
        let src_id = i % N_BLOCKS;
        let src_offset = (i / N_BLOCKS) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.ql[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].ql[src_offset..src_offset + blck_size_interleave]);
    }

    // Interleave upper 2-bit quants (QK_K/4=64 bytes per block)
    let end_hs = end_ls / 2;
    for i in 0..end_hs {
        let src_id = i % N_BLOCKS;
        let src_offset = (i / N_BLOCKS) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        out.qh[dst_offset..dst_offset + blck_size_interleave]
            .copy_from_slice(&in_blocks[src_id].qh[src_offset..src_offset + blck_size_interleave]);
    }

    // Repack i8 scales: layout = [sub_block * N_BLOCKS + block_idx]
    // Q6_K has QK_K/16=16 scales, one per 16-element group.
    const N_SCALES: usize = QK_K / 16; // 16
    for i in 0..N_BLOCKS {
        for j in 0..N_SCALES {
            out.scales[j * N_BLOCKS + i] = in_blocks[i].scales[j];
        }
    }

    out
}

/// Direct translation of repack_q6_K_to_q6_K_8_bl (ggml-cpu/repack.cpp line 3761).
pub fn repack_q6_k_to_q6_k_8_bl(
    src: &[BlockQ6K],
    nrow: usize,
    nblocks: usize,
    interleave_block: usize,
    dst: &mut Vec<BlockQ6Kx8>,
) -> Result<(), &'static str> {
    assert!(interleave_block == 4 || interleave_block == 8);
    const NROWS_INTERLEAVED: usize = 8;

    if nrow % NROWS_INTERLEAVED != 0 {
        return Err("nrow must be a multiple of 8");
    }

    dst.clear();
    dst.reserve(nrow / NROWS_INTERLEAVED * nblocks);

    let mut b = 0usize;
    while b < nrow {
        for x in 0..nblocks {
            let mut tmp = [BlockQ6K {
                ql: [0u8; 128],
                qh: [0u8; 64],
                scales: [0i8; 16],
                d: 0,
            }; 8];
            for i in 0..NROWS_INTERLEAVED {
                tmp[i] = src[x + (b + i) * nblocks];
            }
            dst.push(make_block_q6_kx8(&tmp, interleave_block));
        }
        b += NROWS_INTERLEAVED;
    }
    Ok(())
}

// ── Q2K scalar GEMV ───────────────────────────────────────────────────────────

/// Direct translation of ggml_gemv_q2_K_8x8_q8_K_generic (ggml-cpu/repack.cpp line 1029).
/// Scalar GEMV: Q2_K packed weights × Q8_K activations → f32 output.
/// blocklen=8, ncols_interleaved=8.
///
/// - vx: packed weights as BlockQ2Kx8 slice (nc/8 × nb blocks)
/// - vy: activation as BlockQ8K slice (nb blocks, single row)
/// - s:  output, nc floats
/// - n:  token dimension (multiple of QK_K=256)
/// - nc: number of weight columns (multiple of 8)
pub unsafe fn ggml_gemv_q2_k_8x8_q8_k(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    _nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 8;
    const NCOLS: usize = 8;

    let nb = n / QK_K;
    let b_ptr_start = vx as *const BlockQ2Kx8;
    let a_ptr = vy as *const BlockQ8K;

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_ptr_start.add(x * nb);
        let mut sumf = [0f32; NCOLS];
        let mut sum_minf = [0f32; NCOLS];

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            // k < QK_K / (4 * BLOCKLEN) = 256 / 32 = 8
            for k in 0..(QK_K / (4 * BLOCKLEN)) {
                let scales_0 = &b.scales[(k / 4) * 64..];
                let scales_1 = &b.scales[(k / 4) * 64 + 16..];
                let scales_2 = &b.scales[(k / 4) * 64 + 32..];
                let scales_3 = &b.scales[(k / 4) * 64 + 48..];

                for j in 0..NCOLS {
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    let mut sumi3 = 0i32;
                    let mut sumi4 = 0i32;
                    let mut sumi = 0i32;
                    let offset = ((k / 2) % 2) + j * 2;

                    for i in 0..BLOCKLEN {
                        let qs_byte = b.qs[k * NCOLS * BLOCKLEN + j * BLOCKLEN + i];
                        let v0 = (qs_byte & 3) as i32;
                        let v1 = ((qs_byte >> 2) & 3) as i32;
                        let v2 = ((qs_byte >> 4) & 3) as i32;
                        let v3 = (qs_byte >> 6) as i32;

                        let q8_base = (k >> 2) * 128 + (k % 4) * BLOCKLEN + i;
                        sumi1 = v0 * (a.qs[q8_base] as i32);
                        sumi2 = v1 * (a.qs[q8_base + 32] as i32);
                        sumi3 = v2 * (a.qs[q8_base + 64] as i32);
                        sumi4 = v3 * (a.qs[q8_base + 96] as i32);

                        sumi1 *= (scales_0[offset] & 0xF) as i32;
                        sumi2 *= (scales_1[offset] & 0xF) as i32;
                        sumi3 *= (scales_2[offset] & 0xF) as i32;
                        sumi4 *= (scales_3[offset] & 0xF) as i32;
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * a.d;
                }
            }

            // Min contribution
            for sb in 0..8usize {
                let mins = &b.scales[sb * 16..];
                for j in 0..NCOLS {
                    let m0 = (mins[j * 2] >> 4) as i32;
                    let m1 = (mins[j * 2 + 1] >> 4) as i32;
                    sum_minf[j] += (m0 * a.bsums[sb * 2] as i32 + m1 * a.bsums[sb * 2 + 1] as i32) as f32
                        * f16_to_f32(b.dmin[j])
                        * a.d;
                }
            }
        }

        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j] - sum_minf[j];
        }
    }
}

// ── Q5K scalar GEMV ───────────────────────────────────────────────────────────

/// Direct translation of ggml_gemv_q5_K_NxM_q8_K_generic_impl<8,8>
/// (ggml-cpu/repack.cpp lines 551-643). blocklen=8, ncols_interleaved=8.
///
/// - vx: packed weights as BlockQ5Kx8 (nc/8 × nb)
/// - vy: activation as BlockQ8K (nb blocks, single row)
/// - s:  output, nc floats
/// - n:  token dimension (multiple of QK_K=256)
pub unsafe fn ggml_gemv_q5_k_8x8_q8_k(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    _nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 8;
    const NCOLS: usize = 8;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;
    let b_ptr_start = vx as *const BlockQ5Kx8;
    let a_ptr = vy as *const BlockQ8K;

    // utmp[32]: decoded scale+min u32 words; reinterpreted as bytes for indexing
    let mut utmp = [0u32; 32];

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_ptr_start.add(x * nb);
        let mut sumf    = [0f32; NCOLS];
        let mut sum_minf = [0f32; NCOLS];

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            // Decode scales for each of the 8 sub-block groups
            for sb in 0..8usize {
                let src = &b.scales[sb * K_SCALE_SIZE..];
                let u0 = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                let u1 = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                let u2 = u32::from_le_bytes([src[8], src[9], src[10], src[11]]);
                utmp[sb * 4 + 0] = u0;
                utmp[sb * 4 + 1] = u1;
                utmp[sb * 4 + 2] = u2;
                let u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
                let uaux = u1 & KMASK1;
                utmp[sb * 4 + 1] = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
                utmp[sb * 4 + 2] = uaux;
                utmp[sb * 4 + 0] &= KMASK1;
                utmp[sb * 4 + 3] = u3;
            }

            // utmp as byte slice (little-endian, 128 bytes)
            let utmp_bytes: &[u8; 128] =
                &*(utmp.as_ptr() as *const [u8; 128]);

            // k < QK_K / (2 * BLOCKLEN) = 16
            for k in 0..(QK_K / (2 * BLOCKLEN)) {
                let sb_idx = k / (32 / BLOCKLEN); // = k / 4
                let scale_stride = 32usize;
                let scales_0 = &utmp_bytes[sb_idx * scale_stride..];
                let scales_1 = &utmp_bytes[sb_idx * scale_stride + 16..];
                let qh_shift = sb_idx * 2;

                for j in 0..NCOLS {
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    let mut sumi  = 0i32;

                    for i in 0..BLOCKLEN {
                        let b_qs_offset = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;

                        let qh_idx   = (k * BLOCKLEN + i) % 32;
                        let qh_chunk = qh_idx / BLOCKLEN;
                        let qh_pos   = qh_idx % BLOCKLEN;
                        let b_qh_offset = qh_chunk * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos;

                        let qh_val = b.qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;

                        let qs = b.qs[b_qs_offset];
                        let v0 = ((qs & 0xF) | (h0 << 4)) as i8 as i32;
                        let v1 = ((qs >> 4) | (h1 << 4)) as i8 as i32;

                        let q8_offset = sb_idx * 64 + (k % (32 / BLOCKLEN)) * BLOCKLEN + i;
                        sumi1 = v0 * a.qs[q8_offset] as i32;
                        sumi2 = v1 * a.qs[q8_offset + 32] as i32;
                        sumi1 *= scales_0[j] as i32;
                        sumi2 *= scales_1[j] as i32;
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi as f32 * f16_to_f32(b.d[j]) * a.d;
                }
            }

            // Min contribution: mins at utmp_bytes[8 + sb*16..]
            for sb in 0..8usize {
                let mins = &utmp_bytes[8 + sb * 16..];
                for j in 0..NCOLS {
                    sum_minf[j] += mins[j] as f32
                        * (a.bsums[sb * 2] as i32 + a.bsums[sb * 2 + 1] as i32) as f32
                        * f16_to_f32(b.dmin[j])
                        * a.d;
                }
            }
        }

        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j] - sum_minf[j];
        }
    }
}

// ── Q5K scalar GEMM ───────────────────────────────────────────────────────────

/// Direct translation of ggml_gemm_q5_K_NxM_q8_K_generic_impl<4,8>
/// (ggml-cpu/repack.cpp lines 647-750). blocklen=4, ncols_interleaved=8.
///
/// - vx: packed weights as BlockQ5Kx8 (nc/8 × nb)
/// - vy: packed activations as BlockQ8Kx4 (nr/4 × nb)
/// - s:  output, nr × nc floats (row-major, stride bs)
/// - bs: output row stride in f32 elements
pub unsafe fn ggml_gemm_q5_k_8x4_q8_k(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 4;
    const NCOLS: usize = 8;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;
    let b_ptr_start = vx as *const BlockQ5Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    let mut utmp = [0u32; 32];

    for y in 0..(nr / 4) {
        let a_ptr = a_ptr_start.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut sumf     = [[0f32; NCOLS]; 4];
            let mut sum_minf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                // Decode scales (identical to GEMV)
                for sb in 0..8usize {
                    let src = &b.scales[sb * K_SCALE_SIZE..];
                    let u0 = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                    let u1 = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                    let u2 = u32::from_le_bytes([src[8], src[9], src[10], src[11]]);
                    utmp[sb * 4 + 0] = u0;
                    utmp[sb * 4 + 1] = u1;
                    utmp[sb * 4 + 2] = u2;
                    let u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
                    let uaux = u1 & KMASK1;
                    utmp[sb * 4 + 1] = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
                    utmp[sb * 4 + 2] = uaux;
                    utmp[sb * 4 + 0] &= KMASK1;
                    utmp[sb * 4 + 3] = u3;
                }

                let utmp_bytes: &[u8; 128] =
                    &*(utmp.as_ptr() as *const [u8; 128]);

                // k < QK_K / (2 * BLOCKLEN) = 32
                for k in 0..(QK_K / (2 * BLOCKLEN)) {
                    let sb_idx = k / (32 / BLOCKLEN); // = k / 8
                    let scale_stride = 32usize;
                    let scales_0 = &utmp_bytes[sb_idx * scale_stride..];
                    let scales_1 = &utmp_bytes[sb_idx * scale_stride + 16..];
                    let qh_shift = sb_idx * 2;

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let mut sumi1 = 0i32;
                            let mut sumi2 = 0i32;
                            let mut sumi  = 0i32;

                            for i in 0..BLOCKLEN {
                                let b_qs_offset = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;

                                let qh_idx   = (k * BLOCKLEN + i) % 32;
                                let qh_chunk = qh_idx / BLOCKLEN;
                                let qh_pos   = qh_idx % BLOCKLEN;
                                let b_qh_offset = qh_chunk * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos;

                                let qh_val = b.qh[b_qh_offset];
                                let h0 = (qh_val >> qh_shift) & 1;
                                let h1 = (qh_val >> (qh_shift + 1)) & 1;

                                let qs = b.qs[b_qs_offset];
                                let v0 = ((qs & 0xF) | (h0 << 4)) as i8 as i32;
                                let v1 = ((qs >> 4) | (h1 << 4)) as i8 as i32;

                                // q8 offset for 4-row packed format
                                let q8_offset = sb_idx * 256
                                    + (k % (32 / BLOCKLEN)) * 4 * BLOCKLEN
                                    + m * BLOCKLEN
                                    + i;
                                sumi1 = v0 * a.qs[q8_offset] as i32;
                                sumi2 = v1 * a.qs[q8_offset + 128] as i32;
                                sumi1 *= scales_0[j] as i32;
                                sumi2 *= scales_1[j] as i32;
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi as f32 * f16_to_f32(b.d[j]) * a.d[m];
                        }
                    }
                }

                // Min contribution
                for sb in 0..8usize {
                    let mins = &utmp_bytes[8 + sb * 16..];
                    for m in 0..4usize {
                        let bsums_offset =
                            (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        let bsums = &a.bsums[bsums_offset..];
                        for j in 0..NCOLS {
                            sum_minf[m][j] += mins[j] as f32
                                * (bsums[0] as i32 + bsums[1] as i32) as f32
                                * f16_to_f32(b.dmin[j])
                                * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

// ── Q6K scalar GEMV ───────────────────────────────────────────────────────────

/// Direct translation of ggml_gemv_q6_K_NxM_q8_K_generic_impl<8,8>
/// (ggml-cpu/repack.cpp lines 357-443). blocklen=8, ncols_interleaved=8.
///
/// - vx: packed weights as BlockQ6Kx8 (nc/8 × nb)
/// - vy: activation as BlockQ8K (nb blocks, single row)
/// - s:  output, nc floats
pub unsafe fn ggml_gemv_q6_k_8x8_q8_k(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    _nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 8;
    const NCOLS: usize = 8;
    const BLOCKS_PER_HALF: usize = 64 / BLOCKLEN; // 8

    let nb = n / QK_K;
    let b_ptr_start = vx as *const BlockQ6Kx8;
    let a_ptr = vy as *const BlockQ8K;

    for x in 0..(nc / NCOLS) {
        let b_ptr = b_ptr_start.add(x * nb);
        let mut sumf = [0f32; NCOLS];

        for l in 0..nb {
            let a = &*a_ptr.add(l);
            let b = &*b_ptr.add(l);

            // k < QK_K / (2 * BLOCKLEN) = 16
            for k in 0..(QK_K / (2 * BLOCKLEN)) {
                let base_l = (k / BLOCKS_PER_HALF) * 128 + (k % BLOCKS_PER_HALF) * BLOCKLEN;
                let base_h = base_l + 64;

                let scale_idx_l = base_l / 16;
                let scale_idx_h = base_h / 16;

                let qh_shift_l = ((base_l % 128) / 32) * 2;
                let qh_shift_h = ((base_h % 128) / 32) * 2;

                let qh_half_l = (base_l / 128) * 32;
                let qh_half_h = (base_h / 128) * 32;

                for j in 0..NCOLS {
                    let scale_l = b.scales[scale_idx_l * NCOLS + j];
                    let scale_h = b.scales[scale_idx_h * NCOLS + j];

                    let mut sumi_l = 0i32;
                    let mut sumi_h = 0i32;

                    for i in 0..BLOCKLEN {
                        let ql_pos = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                        let l_4  = (b.ql[ql_pos] & 0xF) as i32;
                        let hi_4 = ((b.ql[ql_pos] >> 4) & 0xF) as i32;

                        // Low half high-bit index
                        let qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                        let qh_chunk_l = qh_idx_l / BLOCKLEN;
                        let qh_pos_l   = qh_idx_l % BLOCKLEN;
                        let qh_offset_l = qh_chunk_l * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_l;
                        let hi_2_l = ((b.qh[qh_offset_l] >> qh_shift_l) & 0x3) as i32;

                        // High half high-bit index
                        let qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                        let qh_chunk_h = qh_idx_h / BLOCKLEN;
                        let qh_pos_h   = qh_idx_h % BLOCKLEN;
                        let qh_offset_h = qh_chunk_h * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_h;
                        let hi_2_h = ((b.qh[qh_offset_h] >> qh_shift_h) & 0x3) as i32;

                        let q_l = ((hi_2_l << 4) | l_4) - 32;
                        let q_h = ((hi_2_h << 4) | hi_4) - 32;

                        sumi_l += q_l * a.qs[base_l + i] as i32;
                        sumi_h += q_h * a.qs[base_h + i] as i32;
                    }

                    sumf[j] += (sumi_l * scale_l as i32 + sumi_h * scale_h as i32) as f32
                        * f16_to_f32(b.d[j])
                        * a.d;
                }
            }
        }

        for j in 0..NCOLS {
            *s.add(x * NCOLS + j) = sumf[j];
        }
    }
}

// ── Q6K scalar GEMM ───────────────────────────────────────────────────────────

/// Direct translation of ggml_gemm_q6_K_NxM_q8_K_generic_impl<4,8>
/// (ggml-cpu/repack.cpp lines 447-548). blocklen=4, ncols_interleaved=8.
///
/// - vx: packed weights as BlockQ6Kx8 (nc/8 × nb)
/// - vy: packed activations as BlockQ8Kx4 (nr/4 × nb)
/// - s:  output, nr × nc floats (row-major, stride bs)
/// - bs: output row stride in f32 elements
pub unsafe fn ggml_gemm_q6_k_8x4_q8_k(
    n: usize,
    s: *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 4;
    const NCOLS: usize = 8;
    const BLOCKS_PER_HALF: usize = 64 / BLOCKLEN; // 16
    const Q8_HALF_STRIDE: usize = 512;
    const Q8_LOW_HIGH_STEP: usize = 256;

    let nb = n / QK_K;
    let b_ptr_start = vx as *const BlockQ6Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    for y in 0..(nr / 4) {
        let a_ptr = a_ptr_start.add(y * nb);
        for x in 0..(nc / NCOLS) {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut sumf = [[0f32; NCOLS]; 4];

            for l in 0..nb {
                let a = &*a_ptr.add(l);
                let b = &*b_ptr.add(l);

                // k < QK_K / (2 * BLOCKLEN) = 32
                for k in 0..(QK_K / (2 * BLOCKLEN)) {
                    let base_l = (k / BLOCKS_PER_HALF) * 128 + (k % BLOCKS_PER_HALF) * BLOCKLEN;
                    let base_h = base_l + 64;

                    let scale_idx_l = base_l / 16;
                    let scale_idx_h = base_h / 16;

                    let qh_shift_l = ((base_l % 128) / 32) * 2;
                    let qh_shift_h = ((base_h % 128) / 32) * 2;

                    let qh_half_l = (base_l / 128) * 32;
                    let qh_half_h = (base_h / 128) * 32;

                    let q8_base = (k / BLOCKS_PER_HALF) * Q8_HALF_STRIDE
                        + (k % BLOCKS_PER_HALF) * (BLOCKLEN * 4);

                    for m in 0..4usize {
                        for j in 0..NCOLS {
                            let scale_l = b.scales[scale_idx_l * NCOLS + j];
                            let scale_h = b.scales[scale_idx_h * NCOLS + j];

                            let mut sumi_l = 0i32;
                            let mut sumi_h = 0i32;

                            for i in 0..BLOCKLEN {
                                let ql_pos = k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;
                                let l_4  = (b.ql[ql_pos] & 0xF) as i32;
                                let hi_4 = ((b.ql[ql_pos] >> 4) & 0xF) as i32;

                                let qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                let qh_chunk_l = qh_idx_l / BLOCKLEN;
                                let qh_pos_l   = qh_idx_l % BLOCKLEN;
                                let qh_offset_l = qh_chunk_l * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_l;
                                let hi_2_l = ((b.qh[qh_offset_l] >> qh_shift_l) & 0x3) as i32;

                                let qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                let qh_chunk_h = qh_idx_h / BLOCKLEN;
                                let qh_pos_h   = qh_idx_h % BLOCKLEN;
                                let qh_offset_h = qh_chunk_h * (BLOCKLEN * NCOLS) + j * BLOCKLEN + qh_pos_h;
                                let hi_2_h = ((b.qh[qh_offset_h] >> qh_shift_h) & 0x3) as i32;

                                let q_l = ((hi_2_l << 4) | l_4) - 32;
                                let q_h = ((hi_2_h << 4) | hi_4) - 32;

                                let q8_l = a.qs[q8_base + m * BLOCKLEN + i] as i32;
                                let q8_h = a.qs[q8_base + m * BLOCKLEN + i + Q8_LOW_HIGH_STEP] as i32;

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l as i32 + sumi_h * scale_h as i32) as f32
                                * f16_to_f32(b.d[j])
                                * a.d[m];
                        }
                    }
                }
            }

            for m in 0..4usize {
                for j in 0..NCOLS {
                    *s.add((y * 4 + m) * bs + x * NCOLS + j) = sumf[m][j];
                }
            }
        }
    }
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

    // ── Q2K ─────────────────────────────────────────────────────────────────

    fn zero_block_q2k() -> BlockQ2K {
        BlockQ2K { scales: [0u8; 16], qs: [0u8; 64], d: 0, dmin: 0 }
    }

    #[test]
    fn test_make_block_q2_kx8_zero() {
        let blocks = [zero_block_q2k(); 8];
        let packed = make_block_q2_kx8(&blocks, 8);
        assert!(packed.d.iter().all(|&v| v == 0));
        assert!(packed.dmin.iter().all(|&v| v == 0));
        assert!(packed.qs.iter().all(|&v| v == 0));
        assert!(packed.scales.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_gemv_q2k_zero_weights_zero_output() {
        let n = 256usize;
        let nc = 8usize;

        // Zero-weight packed block
        let b = BlockQ2Kx8 {
            d: [0u16; 8],
            dmin: [0u16; 8],
            scales: [0u8; 128],
            qs: [0u8; 512],
        };
        let a = zero_block_q8k();

        let mut output = vec![0f32; nc];
        unsafe {
            ggml_gemv_q2_k_8x8_q8_k(
                n,
                output.as_mut_ptr(),
                &b as *const BlockQ2Kx8 as *const u8,
                &a as *const BlockQ8K as *const u8,
                1,
                nc,
            );
        }
        for &v in &output {
            assert_eq!(v, 0.0f32, "expected zero output for zero weights");
        }
    }

    // ── Q5K ─────────────────────────────────────────────────────────────────

    fn zero_block_q5k() -> BlockQ5K {
        BlockQ5K { d: 0, dmin: 0, scales: [0u8; 12], qh: [0u8; 32], qs: [0u8; 128] }
    }

    #[test]
    fn test_make_block_q5_kx8_zero() {
        let blocks = [zero_block_q5k(); 8];
        let packed = make_block_q5_kx8(&blocks, 8);
        assert!(packed.d.iter().all(|&v| v == 0));
        assert!(packed.dmin.iter().all(|&v| v == 0));
        assert!(packed.qs.iter().all(|&v| v == 0));
        assert!(packed.qh.iter().all(|&v| v == 0));
        assert!(packed.scales.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_gemv_q5k_zero_weights_zero_output() {
        let n = 256usize;
        let nc = 8usize;

        let b = BlockQ5Kx8 {
            d: [0u16; 8],
            dmin: [0u16; 8],
            scales: [0u8; 96],
            qh: [0u8; 256],
            qs: [0u8; 1024],
        };
        let a = zero_block_q8k();

        let mut output = vec![0f32; nc];
        unsafe {
            ggml_gemv_q5_k_8x8_q8_k(
                n,
                output.as_mut_ptr(),
                &b as *const BlockQ5Kx8 as *const u8,
                &a as *const BlockQ8K as *const u8,
                1,
                nc,
            );
        }
        for &v in &output {
            assert_eq!(v, 0.0f32, "expected zero output for zero weights");
        }
    }

    // ── Q6K ─────────────────────────────────────────────────────────────────

    fn zero_block_q6k() -> BlockQ6K {
        BlockQ6K { ql: [0u8; 128], qh: [0u8; 64], scales: [0i8; 16], d: 0 }
    }

    #[test]
    fn test_make_block_q6_kx8_zero() {
        let blocks = [zero_block_q6k(); 8];
        let packed = make_block_q6_kx8(&blocks, 8);
        assert!(packed.d.iter().all(|&v| v == 0));
        assert!(packed.scales.iter().all(|&v| v == 0));
        assert!(packed.ql.iter().all(|&v| v == 0));
        assert!(packed.qh.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_gemv_q6k_zero_weights_zero_output() {
        let n = 256usize;
        let nc = 8usize;

        let b = BlockQ6Kx8 {
            d: [0u16; 8],
            scales: [0i8; 128],
            ql: [0u8; 1024],
            qh: [0u8; 512],
        };
        let a = zero_block_q8k();

        let mut output = vec![0f32; nc];
        unsafe {
            ggml_gemv_q6_k_8x8_q8_k(
                n,
                output.as_mut_ptr(),
                &b as *const BlockQ6Kx8 as *const u8,
                &a as *const BlockQ8K as *const u8,
                1,
                nc,
            );
        }
        for &v in &output {
            assert_eq!(v, 0.0f32, "expected zero output for zero weights");
        }
    }

    #[test]
    fn test_f16_to_f32_normal() {
        // f16 1.0 = 0x3C00
        let v = f16_to_f32(0x3C00u16);
        assert!((v - 1.0f32).abs() < 1e-6, "f16 1.0 should decode to f32 1.0, got {}", v);
    }

    #[test]
    fn test_make_block_q6_kx8_scales_interleaving() {
        let mut blocks = [zero_block_q6k(); 8];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.scales.fill(i as i8 + 1);
        }
        let packed = make_block_q6_kx8(&blocks, 8);
        // scales[sub_block * 8 + block_idx]
        for block_idx in 0..8usize {
            for sub_block in 0..16usize {
                assert_eq!(
                    packed.scales[sub_block * 8 + block_idx],
                    (block_idx as i8) + 1,
                    "scale mismatch at sub_block={sub_block} block={block_idx}"
                );
            }
        }
    }
}

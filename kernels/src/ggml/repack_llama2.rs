//! Direct translation of llama.cpp ggml-cpu/repack.cpp and arch/x86/repack.cpp.
//! Q4_0 × Q8_0 GEMV/GEMM, Q8 quantization, and repacking routines.
//!
//! Sources:
//! - ggml_quantize_mat_q8_0_4x8:   arch/x86/repack.cpp  lines 178-288
//! - ggml_quantize_mat_q8_K_4x8:   arch/x86/repack.cpp  lines 290-511
//! - gemv_q4_b32_8x8_q8_0 (AVX2):  arch/x86/repack.cpp  lines 521-636
//! - ggml_gemv_q4_0_8x8_q8_0:      arch/x86/repack.cpp  lines 1448-1461
//! - ggml_gemm_q4_K_8x8_q8_K AVX2: arch/x86/repack.cpp  lines 2042-end
//! - ggml_gemm_q4_K_8x4_q8_K:      ggml-cpu/repack.cpp  lines 1949-2030
//! - make_block_q4_0x8:             ggml-cpu/repack.cpp  lines 3128-3150
//! - make_block_q8_0x4:             ggml-cpu/repack.cpp  lines 3022-3037
//! - repack_q4_0_to_q4_0_8_bl:      ggml-cpu/repack.cpp  lines 3790-3817

use super::blocks::{BlockQ4_0, BlockQ4Kx8, BlockQ8_0, BlockQ8Kx4};

const QK4_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;

// ── New interleaved block structs ─────────────────────────────────────────────

/// 8 Q4_0 blocks interleaved for 8-wide SIMD GEMV/GEMM (block<4,8> from repack.h).
/// d[8]: f16 scales; qs[128]: 4-bit nibbles XOR'd with 0x88 per 8-byte chunk, interleaved.
/// Total: 8*2 + 128 = 144 bytes.
///
/// Layout: B0[0:8], B1[0:8], ..., B7[0:8], B0[8:16], B1[8:16], ..., B7[8:16].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0x8 {
    pub d:  [u16; 8],
    pub qs: [i8; 128],
}

/// 4 Q8_0 blocks interleaved for 4×8 GEMM (block<8,4> from repack.h).
/// d[4]: f16 scales; qs[128]: 8-bit quants, 4-row interleaved in chunks.
/// Total: 4*2 + 128 = 136 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0x4 {
    pub d:  [u16; 4],
    pub qs: [i8; 128],
}

// ── Repacking ─────────────────────────────────────────────────────────────────

/// Direct translation of make_block_q4_0x8 (ggml-cpu/repack.cpp line 3128).
/// Interleaves 8 Q4_0 blocks into BlockQ4_0x8 format.
/// blck_size_interleave must be 8 (required for x86 SIMD kernels).
pub fn make_block_q4_0x8(in_blocks: &[BlockQ4_0; 8], blck_size_interleave: usize) -> BlockQ4_0x8 {
    assert_eq!(blck_size_interleave, 8, "blck_size_interleave must be 8");
    let mut out = BlockQ4_0x8 {
        d:  [0u16; 8],
        qs: [0i8; 128],
    };
    for i in 0..8 {
        out.d[i] = in_blocks[i].d;
    }
    let xor_mask: u64 = 0x8888888888888888;
    let end = QK4_0 * 4 / blck_size_interleave; // 32*4/8 = 16
    for i in 0..end {
        let src_id     = i % 8;
        let src_offset = (i / 8) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        let mut elems: u64 = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                (&mut elems) as *mut u64 as *mut u8,
                8,
            );
        }
        elems ^= xor_mask;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&elems) as *const u64 as *const u8,
                out.qs.as_mut_ptr().add(dst_offset) as *mut u8,
                8,
            );
        }
    }
    out
}

/// Direct translation of make_block_q8_0x4 (ggml-cpu/repack.cpp line 3022).
/// Interleaves 4 Q8_0 blocks into BlockQ8_0x4 format.
pub fn make_block_q8_0x4(in_blocks: &[BlockQ8_0; 4], blck_size_interleave: usize) -> BlockQ8_0x4 {
    let mut out = BlockQ8_0x4 {
        d:  [0u16; 4],
        qs: [0i8; 128],
    };
    for i in 0..4 {
        out.d[i] = in_blocks[i].d;
    }
    let end = QK8_0 * 4 / blck_size_interleave;
    for i in 0..end {
        let src_id     = i % 4;
        let src_offset = (i / 4) * blck_size_interleave;
        let dst_offset = i * blck_size_interleave;
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_blocks[src_id].qs.as_ptr().add(src_offset),
                out.qs.as_mut_ptr().add(dst_offset),
                blck_size_interleave,
            );
        }
    }
    out
}

/// Direct translation of repack_q4_0_to_q4_0_8_bl (ggml-cpu/repack.cpp line 3790).
/// Repacks a full weight matrix (nrow × nblocks Q4_0 blocks) into BlockQ4_0x8 format.
/// nrow must be a multiple of 8; interleave_block must be 8.
pub fn repack_q4_0_to_q4_0_8_bl(
    src: &[BlockQ4_0],
    nrow: usize,
    nblocks: usize,
    interleave_block: usize,
    dst: &mut Vec<BlockQ4_0x8>,
) -> Result<(), &'static str> {
    assert_eq!(interleave_block, 8);
    const NROWS_INTERLEAVED: usize = 8;

    if nrow % NROWS_INTERLEAVED != 0 {
        return Err("nrow must be a multiple of 8");
    }

    dst.clear();
    dst.reserve(nrow / NROWS_INTERLEAVED * nblocks);

    let mut row_base = 0usize;
    while row_base < nrow {
        for x in 0..nblocks {
            let mut tmp = [BlockQ4_0 { d: 0, qs: [0u8; 16] }; 8];
            for i in 0..NROWS_INTERLEAVED {
                tmp[i] = src[x + (row_base + i) * nblocks];
            }
            dst.push(make_block_q4_0x8(&tmp, interleave_block));
        }
        row_base += NROWS_INTERLEAVED;
    }
    Ok(())
}

// ── AVX2 quantization ─────────────────────────────────────────────────────────

/// Direct translation of ggml_quantize_mat_q8_0_4x8 (arch/x86/repack.cpp line 178).
/// Quantizes 4 rows of fp32 into BlockQ8_0x4 format.
/// - x:   pointer to 4 rows of k floats each (row stride = k)
/// - vy:  output BlockQ8_0x4 array (nb = k/32 blocks)
/// - k:   number of elements per row (must be multiple of 32)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn ggml_quantize_mat_q8_0_4x8(x: *const f32, vy: *mut u8, k: i64) {
    use std::arch::x86_64::*;
    debug_assert_eq!(k % QK8_0 as i64, 0);
    let nb = (k as usize) / QK8_0;
    let y = vy as *mut BlockQ8_0x4;

    let mut srcv  = [[_mm256_setzero_ps(); 4]; 4];
    let mut idvec = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        for row in 0..4usize {
            let base = x.add(row * k as usize + i * QK8_0);
            let v0 = _mm256_loadu_ps(base);
            let v1 = _mm256_loadu_ps(base.add(8));
            let v2 = _mm256_loadu_ps(base.add(16));
            let v3 = _mm256_loadu_ps(base.add(24));

            let sign_bit = _mm256_set1_ps(-0.0f32);
            let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

            let max4 = _mm_max_ps(
                _mm256_extractf128_ps(max_abs, 1),
                _mm256_castps256_ps128(max_abs),
            );
            let max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            let max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            let max_scalar = _mm_cvtss_f32(max4);

            let d  = max_scalar / 127.0f32;
            let id = if max_scalar != 0.0 { 127.0f32 / max_scalar } else { 0.0f32 };

            // Convert d to f16 and store
            let d_f16 = _mm_cvtps_ph(_mm_set_ss(d), 0);
            (*y.add(i)).d[row] = _mm_extract_epi16(d_f16, 0) as u16;

            srcv[row][0] = v0;
            srcv[row][1] = v1;
            srcv[row][2] = v2;
            srcv[row][3] = v3;
            idvec[row] = _mm256_set1_ps(id);
        }

        // Quantize and interleave 4 chunks of 8 floats across 4 rows
        for j in 0..4usize {
            let v0 = _mm256_round_ps(
                _mm256_mul_ps(srcv[0][j], idvec[0]), 0);
            let v1 = _mm256_round_ps(
                _mm256_mul_ps(srcv[1][j], idvec[1]), 0);
            let v2 = _mm256_round_ps(
                _mm256_mul_ps(srcv[2][j], idvec[2]), 0);
            let v3 = _mm256_round_ps(
                _mm256_mul_ps(srcv[3][j], idvec[3]), 0);

            let i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);

            let i0 = _mm256_packs_epi32(i0, i1);
            let i2 = _mm256_packs_epi32(i2, i3);
            let i0 = _mm256_packs_epi16(i0, i2);

            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let i0 = _mm256_permutevar8x32_epi32(i0, perm);

            _mm256_storeu_si256(
                ((*y.add(i)).qs.as_mut_ptr().add(32 * j)) as *mut __m256i,
                i0,
            );
        }
    }
}

/// Direct translation of ggml_quantize_mat_q8_K_4x8 (arch/x86/repack.cpp line 290).
/// Quantizes 4 rows of fp32 into BlockQ8Kx4 format (Q8_K super-blocks of 256).
/// - x:   pointer to 4 rows of k floats each (row stride = k)
/// - vy:  output BlockQ8Kx4 array (nb = k/256 blocks)
/// - k:   number of elements per row (must be multiple of 256)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn ggml_quantize_mat_q8_k_4x8(x: *const f32, vy: *mut u8, k: i64) {
    use std::arch::x86_64::*;
    debug_assert_eq!(k % QK_K as i64, 0);
    let nb = (k as usize) / QK_K;
    let y = vy as *mut BlockQ8Kx4;

    // 8 sub-blocks of 32 floats per row per super-block → 32 AVX8 vectors per row
    let mut srcv: [[__m256; 32]; 4] = [[_mm256_setzero_ps(); 32]; 4];
    let mut iscale_vec = [_mm256_setzero_ps(); 4];

    // Shuffle masks for bsums computation (replicate into both 128-bit lanes)
    let shuffle_mask_sb2 = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,0,1,4,5,6,7,8,9,8,9,12,13,14,15)),
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,0,1,4,5,6,7,8,9,8,9,12,13,14,15)),
        0,
    );
    let shuffle_mask_sb3 = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,2,3,0,1,6,7,8,9,10,11,8,9,14,15)),
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,2,3,0,1,6,7,8,9,10,11,8,9,14,15)),
        0,
    );
    let shuffle_mask_sb4 = _mm256_permute2f128_si256(
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,2,3,4,5,0,1,8,9,10,11,12,13,8,9)),
        _mm256_castsi128_si256(_mm_setr_epi8(0,1,2,3,4,5,0,1,8,9,10,11,12,13,8,9)),
        0,
    );

    for i in 0..nb {
        for row in 0..4usize {
            let sign_bit = _mm256_set1_ps(-0.0f32);
            let base0 = x.add(row * k as usize + i * QK_K);

            // Load first 32 floats and initialize max tracking
            let v0 = _mm256_loadu_ps(base0);
            let v1 = _mm256_loadu_ps(base0.add(8));
            let v2 = _mm256_loadu_ps(base0.add(16));
            let v3 = _mm256_loadu_ps(base0.add(24));

            let abs0 = _mm256_andnot_ps(sign_bit, v0);
            let abs1 = _mm256_andnot_ps(sign_bit, v1);
            let abs2 = _mm256_andnot_ps(sign_bit, v2);
            let abs3 = _mm256_andnot_ps(sign_bit, v3);

            let mut max_abs = _mm256_max_ps(_mm256_max_ps(abs0, abs1), _mm256_max_ps(abs2, abs3));

            let mask0 = _mm256_cmp_ps(max_abs, v0, _CMP_EQ_OQ);
            let mask1 = _mm256_cmp_ps(max_abs, v1, _CMP_EQ_OQ);
            let mask2 = _mm256_cmp_ps(max_abs, v2, _CMP_EQ_OQ);
            let mask3 = _mm256_cmp_ps(max_abs, v3, _CMP_EQ_OQ);
            let mut mask_abs = _mm256_or_ps(
                _mm256_or_ps(mask0, mask1),
                _mm256_or_ps(mask2, mask3),
            );

            srcv[row][0] = v0;
            srcv[row][1] = v1;
            srcv[row][2] = v2;
            srcv[row][3] = v3;

            for sb in 1..8usize {
                let temp_abs = max_abs;
                let base = base0.add(sb * 32);

                let v0 = _mm256_loadu_ps(base);
                let v1 = _mm256_loadu_ps(base.add(8));
                let v2 = _mm256_loadu_ps(base.add(16));
                let v3 = _mm256_loadu_ps(base.add(24));

                let abs0 = _mm256_andnot_ps(sign_bit, v0);
                let abs1 = _mm256_andnot_ps(sign_bit, v1);
                let abs2 = _mm256_andnot_ps(sign_bit, v2);
                let abs3 = _mm256_andnot_ps(sign_bit, v3);

                max_abs = _mm256_max_ps(max_abs, abs0);
                max_abs = _mm256_max_ps(max_abs, abs1);
                max_abs = _mm256_max_ps(max_abs, abs2);
                max_abs = _mm256_max_ps(max_abs, abs3);

                let mask_prev = _mm256_cmp_ps(temp_abs, max_abs, _CMP_EQ_OQ);
                mask_abs = _mm256_and_ps(mask_abs, mask_prev);

                let mask0 = _mm256_cmp_ps(max_abs, v0, _CMP_EQ_OQ);
                let mask1 = _mm256_cmp_ps(max_abs, v1, _CMP_EQ_OQ);
                let mask2 = _mm256_cmp_ps(max_abs, v2, _CMP_EQ_OQ);
                let mask3 = _mm256_cmp_ps(max_abs, v3, _CMP_EQ_OQ);
                let mask_curr = _mm256_or_ps(
                    _mm256_or_ps(mask0, mask1),
                    _mm256_or_ps(mask2, mask3),
                );
                mask_abs = _mm256_or_ps(mask_abs, mask_curr);

                srcv[row][sb * 4    ] = v0;
                srcv[row][sb * 4 + 1] = v1;
                srcv[row][sb * 4 + 2] = v2;
                srcv[row][sb * 4 + 3] = v3;
            }

            let max4 = _mm_max_ps(
                _mm256_extractf128_ps(max_abs, 1),
                _mm256_castps256_ps128(max_abs),
            );
            let max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            let max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            let max_scalar = _mm_cvtss_f32(max4);

            let max_scalar_vec = _mm256_set1_ps(max_scalar);
            let mask_next = _mm256_cmp_ps(max_scalar_vec, max_abs, _CMP_EQ_OQ);
            let final_mask = _mm256_and_ps(mask_abs, mask_next);
            let mask = _mm256_movemask_ps(final_mask);

            let mut iscale = if max_scalar != 0.0 { 127.0f32 / max_scalar } else { 0.0f32 };
            if mask != 0 {
                iscale = if max_scalar != 0.0 { -127.0f32 / max_scalar } else { 0.0f32 };
            }

            (*y.add(i)).d[row] = if max_scalar != 0.0 { 1.0 / iscale } else { 0.0 };
            iscale_vec[row] = _mm256_set1_ps(iscale);
        }

        // Quantize all 32 chunks and interleave across 4 rows
        let mut quants_interleaved = [_mm256_setzero_si256(); 32];
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        for j in 0..32usize {
            let v0 = _mm256_round_ps(_mm256_mul_ps(srcv[0][j], iscale_vec[0]), 0);
            let v1 = _mm256_round_ps(_mm256_mul_ps(srcv[1][j], iscale_vec[1]), 0);
            let v2 = _mm256_round_ps(_mm256_mul_ps(srcv[2][j], iscale_vec[2]), 0);
            let v3 = _mm256_round_ps(_mm256_mul_ps(srcv[3][j], iscale_vec[3]), 0);

            let i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);

            let i0 = _mm256_packs_epi32(i0, i1);
            let i2 = _mm256_packs_epi32(i2, i3);
            let i0 = _mm256_packs_epi16(i0, i2);
            let i0 = _mm256_permutevar8x32_epi32(i0, perm);

            _mm256_storeu_si256(
                ((*y.add(i)).qs.as_mut_ptr().add(32 * j)) as *mut __m256i,
                i0,
            );
            quants_interleaved[j] = i0;
        }

        // Compute bsums: 16 partial sums of 16 elements each, interleaved across 4 rows
        let one = _mm256_set1_epi8(1);

        for kk in 0..4usize {
            let mut q0 = quants_interleaved[kk * 8    ];
            let mut q1 = quants_interleaved[kk * 8 + 1];
            let mut q2 = quants_interleaved[kk * 8 + 2];
            let mut q3 = quants_interleaved[kk * 8 + 3];
            let mut q4 = quants_interleaved[kk * 8 + 4];
            let mut q5 = quants_interleaved[kk * 8 + 5];
            let mut q6 = quants_interleaved[kk * 8 + 6];
            let mut q7 = quants_interleaved[kk * 8 + 7];

            // First half: interleave 2 values per sub-block pair
            let sb2_h1  = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
            let mut sb_h1 = _mm256_blend_epi16(q0, sb2_h1, 34);
            let sb3_h1  = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
            sb_h1 = _mm256_blend_epi16(sb_h1, sb3_h1, 68);
            let sb4_h1  = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
            sb_h1 = _mm256_blend_epi16(sb_h1, sb4_h1, 136);

            let mut bsums_r1 = _mm256_maddubs_epi16(one, sb_h1);

            for _ in 0..3 {
                q0 = _mm256_srli_epi64(q0, 16);
                q2 = _mm256_srli_epi64(q2, 16);
                q4 = _mm256_srli_epi64(q4, 16);
                q6 = _mm256_srli_epi64(q6, 16);

                let sb2_h1 = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
                let mut sb_h1 = _mm256_blend_epi16(q0, sb2_h1, 34);
                let sb3_h1 = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
                sb_h1 = _mm256_blend_epi16(sb_h1, sb3_h1, 68);
                let sb4_h1 = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
                sb_h1 = _mm256_blend_epi16(sb_h1, sb4_h1, 136);

                bsums_r1 = _mm256_add_epi16(bsums_r1, _mm256_maddubs_epi16(one, sb_h1));
            }

            // Second half
            let sb2_h2  = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
            let mut sb_h2 = _mm256_blend_epi16(q1, sb2_h2, 34);
            let sb3_h2  = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
            sb_h2 = _mm256_blend_epi16(sb_h2, sb3_h2, 68);
            let sb4_h2  = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
            sb_h2 = _mm256_blend_epi16(sb_h2, sb4_h2, 136);

            let mut bsums_r2 = _mm256_maddubs_epi16(one, sb_h2);

            for _ in 0..3 {
                q1 = _mm256_srli_epi64(q1, 16);
                q3 = _mm256_srli_epi64(q3, 16);
                q5 = _mm256_srli_epi64(q5, 16);
                q7 = _mm256_srli_epi64(q7, 16);

                let sb2_h2 = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
                let mut sb_h2 = _mm256_blend_epi16(q1, sb2_h2, 34);
                let sb3_h2 = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
                sb_h2 = _mm256_blend_epi16(sb_h2, sb3_h2, 68);
                let sb4_h2 = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
                sb_h2 = _mm256_blend_epi16(sb_h2, sb4_h2, 136);

                bsums_r2 = _mm256_add_epi16(bsums_r2, _mm256_maddubs_epi16(one, sb_h2));
            }

            let bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
            _mm256_storeu_si256(
                ((*y.add(i)).bsums.as_mut_ptr().add(16 * kk)) as *mut __m256i,
                bsums_r,
            );
        }
    }
}

// ── AVX2 GEMV for Q4_0 × Q8_0 ────────────────────────────────────────────────

/// Inner AVX2 LUT GEMV kernel for Q4 (32-element blocks) × Q8_0.
/// Direct translation of gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>
/// (arch/x86/repack.cpp lines 521-636).
///
/// - s[y * nr + x * 8 .. +8] += Q4_0x8_block[x] · Q8_0_block[y]  (for all x, y)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemv_q4_b32_8x8_q8_0_lut_avx(
    n:             usize,
    s:             *mut f32,
    vx:            *const u8,
    vy:            *const u8,
    nr:            usize,
    nc:            usize,
    signextendlut: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;

    let nb = n / QK8_0;
    let final_permute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    let b_ptr_start = vx as *const BlockQ4_0x8;
    let a_ptr_start = vy as *const BlockQ8_0;

    for y in 0..nr {
        let a_ptr = a_ptr_start.add(y * nb);
        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_row = _mm256_setzero_ps();

            for b in 0..nb as isize {
                // Load 4 × 32-byte chunks covering 8 interleaved blocks (128 bytes total)
                let rhs_raw_0 = _mm256_loadu_si256((*b_ptr.offset(b)).qs.as_ptr() as *const __m256i);
                let rhs_raw_1 = _mm256_loadu_si256((*b_ptr.offset(b)).qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_2 = _mm256_loadu_si256((*b_ptr.offset(b)).qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_3 = _mm256_loadu_si256((*b_ptr.offset(b)).qs.as_ptr().add(96) as *const __m256i);

                // 4-bit → 8-bit with sign extension via LUT: low nibbles (elements 0-7,16-23)
                let rhs_vec_0123_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0, m4b));
                let rhs_vec_4567_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_1, m4b));
                let rhs_vec_0123_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2, m4b));
                let rhs_vec_4567_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_3, m4b));
                // High nibbles (elements 8-15, 24-31)
                let rhs_vec_0123_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0, 4), m4b));
                let rhs_vec_4567_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_1, 4), m4b));
                let rhs_vec_0123_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2, 4), m4b));
                let rhs_vec_4567_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_3, 4), m4b));

                // Load column scales for 8 blocks
                let changemask = _mm_set_epi8(15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0);
                let col_scale_f32 = _mm256_cvtph_ps(
                    _mm_shuffle_epi8(_mm_loadu_si128((*b_ptr.offset(b)).d.as_ptr() as *const __m128i), changemask));

                // Load and broadcast Q8_0 row scale (f16 → f32)
                let row_d_f16 = (*a_ptr.offset(b)).d;
                let row_d_f32 = _mm_cvtss_f32(
                    _mm_cvtph_ps(_mm_insert_epi16(_mm_setzero_si128(), row_d_f16 as i32, 0))
                );
                let row_scale_f32 = _mm256_set1_ps(row_d_f32);

                // Load Q8_0 LHS activation (32 bytes) and broadcast each 16-byte half
                let lhs_vec_0 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128((*a_ptr.offset(b)).qs.as_ptr() as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128((*a_ptr.offset(b)).qs.as_ptr() as *const __m128i)),
                    0,
                );
                let lhs_vec_1 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128((*a_ptr.offset(b)).qs.as_ptr().add(16) as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128((*a_ptr.offset(b)).qs.as_ptr().add(16) as *const __m128i)),
                    0,
                );

                let mut iacc = _mm256_setzero_si256();

                // Dot product: B0/B4 interleaved with A0
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(rhs_vec_0123_0, _mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170),
                    _mm256_shuffle_epi32(lhs_vec_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177), rhs_vec_4567_0, 170),
                    _mm256_shuffle_epi32(lhs_vec_0, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(rhs_vec_0123_1, _mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170),
                    _mm256_shuffle_epi32(lhs_vec_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177), rhs_vec_4567_1, 170),
                    _mm256_shuffle_epi32(lhs_vec_0, 255));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(rhs_vec_0123_2, _mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170),
                    _mm256_shuffle_epi32(lhs_vec_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177), rhs_vec_4567_2, 170),
                    _mm256_shuffle_epi32(lhs_vec_1, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(rhs_vec_0123_3, _mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170),
                    _mm256_shuffle_epi32(lhs_vec_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc,
                    _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177), rhs_vec_4567_3, 170),
                    _mm256_shuffle_epi32(lhs_vec_1, 255));

                acc_row = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc),
                    _mm256_mul_ps(col_scale_f32, row_scale_f32),
                    acc_row,
                );
            }

            // Permute to correct column order and store 8 output floats
            let acc_row = _mm256_permutevar8x32_ps(acc_row, final_permute);
            _mm256_storeu_ps(s.add(y * nr + x * 8), acc_row);
        }
    }
}

/// mul_sum_i8_pairs_acc_int32x8: acc += sign(x)*|x| · sign(x)*y via maddubs.
/// Direct translation of mul_sum_i8_pairs_acc_int32x8 (arch/x86/repack.cpp line 165).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_sum_i8_pairs_acc_int32x8(
    acc: std::arch::x86_64::__m256i,
    x:   std::arch::x86_64::__m256i,
    y:   std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    let ones = _mm256_set1_epi16(1);
    _mm256_add_epi32(acc, _mm256_madd_epi16(ones, dot))
}

/// Direct translation of ggml_gemv_q4_0_8x8_q8_0 (arch/x86/repack.cpp line 1448).
/// AVX2 GEMV: BlockQ4_0x8 weights × Q8_0 activations → float output.
///
/// Computes s[y * nr + x * 8 .. +8] for each activation row y and weight tile x.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemv_q4_0_8x8_q8_0(
    n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let signextendlut = {
        let half = _mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8,7,6,5,4,3,2,1,0);
        _mm256_permute2f128_si256(_mm256_castsi128_si256(half), _mm256_castsi128_si256(half), 0)
    };
    gemv_q4_b32_8x8_q8_0_lut_avx(n, s, vx, vy, nr, nc, signextendlut);
}

// ── AVX2 GEMM for Q4_K × Q8_K (8×8, 4 rows at a time) ───────────────────────

/// Direct translation of ggml_gemm_q4_K_8x8_q8_K AVX2 path
/// (arch/x86/repack.cpp lines 2042-end).
///
/// Computes s[(y*4+m)*bs + x*8 .. +8] for m in 0..4, using:
/// - vx: packed weights as BlockQ4Kx8, shape (nc/8) × nb blocks
/// - vy: packed activations as BlockQ8Kx4, shape (nr/4) × nb blocks
/// - s:  output floats, nr × nc
/// - bs: row stride in floats (typically = nc)
/// - n:  token dimension (multiple of QK_K = 256)
/// - nr: activation rows (multiple of 4)
/// - nc: weight columns (multiple of 8)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemm_q4_k_8x8_q8_k(
    n:  usize,
    s:  *mut f32,
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
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

    let b_ptr_start = vx as *const BlockQ4Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    for y in 0..nr / 4 {
        let a_ptr = a_ptr_start.add(y * nb);
        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_rows     = [_mm256_setzero_ps(); 4];
            let mut acc_min_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb as isize {
                let a  = &*a_ptr.offset(b);
                let bp = &*b_ptr.offset(b);

                let col_scale_f32 = _mm256_cvtph_ps(_mm_loadu_si128(bp.d.as_ptr() as *const __m128i));
                let col_dmin_f32  = _mm256_cvtph_ps(_mm_loadu_si128(bp.dmin.as_ptr() as *const __m128i));

                for sb in 0..(QK_K / 64) as isize {
                    let base = bp.qs.as_ptr().offset(sb * 256);

                    let rhs_raw_mat_0123_0 = _mm256_loadu_si256(base.add(  0) as *const __m256i);
                    let rhs_raw_mat_4567_0 = _mm256_loadu_si256(base.add( 32) as *const __m256i);
                    let rhs_raw_mat_0123_1 = _mm256_loadu_si256(base.add( 64) as *const __m256i);
                    let rhs_raw_mat_4567_1 = _mm256_loadu_si256(base.add( 96) as *const __m256i);
                    let rhs_raw_mat_0123_2 = _mm256_loadu_si256(base.add(128) as *const __m256i);
                    let rhs_raw_mat_4567_2 = _mm256_loadu_si256(base.add(160) as *const __m256i);
                    let rhs_raw_mat_0123_3 = _mm256_loadu_si256(base.add(192) as *const __m256i);
                    let rhs_raw_mat_4567_3 = _mm256_loadu_si256(base.add(224) as *const __m256i);

                    let rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, required_order), 240);
                    let rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, required_order), rhs_raw_mat_4567_0, 240);
                    let rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, required_order), 240);
                    let rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, required_order), rhs_raw_mat_4567_1, 240);
                    let rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, required_order), 240);
                    let rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, required_order), rhs_raw_mat_4567_2, 240);
                    let rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, required_order), 240);
                    let rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, required_order), rhs_raw_mat_4567_3, 240);

                    let rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b);
                    let rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b);
                    let rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b);
                    let rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b);
                    let rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b);
                    let rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b);
                    let rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b);
                    let rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b);

                    let rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b);
                    let rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b);
                    let rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b);
                    let rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b);
                    let rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b);
                    let rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b);
                    let rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b);
                    let rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b);

                    let rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136);
                    let rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136);
                    let rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136);
                    let rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136);
                    let rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136);
                    let rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136);
                    let rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136);
                    let rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136);
                    let rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136);
                    let rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136);
                    let rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136);
                    let rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136);
                    let rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136);
                    let rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136);
                    let rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136);
                    let rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136);

                    let rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221);
                    let rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221);
                    let rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221);
                    let rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221);
                    let rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221);
                    let rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221);
                    let rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221);
                    let rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221);
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
                        scales_base.offset(24 * sb),
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                    let uaux_0 = utmp_0[1] & KMASK1;
                    utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= KMASK1;

                    std::ptr::copy_nonoverlapping(
                        scales_base.offset(12 + sb * 24),
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                    let uaux_1 = utmp_1[1] & KMASK1;
                    utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= KMASK1;

                    let mins_and_scales_0 = _mm_set_epi32(
                        utmp_0[3] as i32, utmp_0[2] as i32, utmp_0[1] as i32, utmp_0[0] as i32);
                    let scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));
                    let mins_and_scales_1 = _mm_set_epi32(
                        utmp_1[3] as i32, utmp_1[2] as i32, utmp_1[1] as i32, utmp_1[0] as i32);
                    let scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));
                    let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        _mm_shuffle_epi32(mins_and_scales_0, 78),
                        _mm_shuffle_epi32(mins_and_scales_1, 78),
                    ));

                    let scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    let scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);
                    let scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    let scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    let lhs_base = a.qs.as_ptr().offset(256 * sb);
                    let lhs_mat_0123_00 = _mm256_loadu_si256(lhs_base.add(  0) as *const __m256i);
                    let lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    let lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    let lhs_mat_0123_01 = _mm256_loadu_si256(lhs_base.add( 32) as *const __m256i);
                    let lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    let lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    let lhs_mat_0123_02 = _mm256_loadu_si256(lhs_base.add( 64) as *const __m256i);
                    let lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    let lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    let lhs_mat_0123_03 = _mm256_loadu_si256(lhs_base.add( 96) as *const __m256i);
                    let lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    let lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    let lhs_mat_0123_10 = _mm256_loadu_si256(lhs_base.add(128) as *const __m256i);
                    let lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    let lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    let lhs_mat_0123_11 = _mm256_loadu_si256(lhs_base.add(160) as *const __m256i);
                    let lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    let lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    let lhs_mat_0123_12 = _mm256_loadu_si256(lhs_base.add(192) as *const __m256i);
                    let lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    let lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    let lhs_mat_0123_13 = _mm256_loadu_si256(lhs_base.add(224) as *const __m256i);
                    let lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    let lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    let lhs_bsums_0123 = _mm256_loadu_si256(
                        (a.bsums.as_ptr().offset(16 * sb)) as *const __m256i);
                    let lhs_bsums_hsum = {
                        let lo = _mm256_castsi256_si128(lhs_bsums_0123);
                        let hi = _mm256_extracti128_si256(lhs_bsums_0123, 1);
                        let h = _mm256_castsi128_si256(_mm_hadd_epi16(lo, hi));
                        _mm256_permute2x128_si256(h, h, 0)
                    };

                    let lhs_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160);
                    let lhs_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160);
                    let lhs_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);
                    let lhs_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);
                    let lhs_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);
                    let lhs_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);
                    let lhs_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);
                    let lhs_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160);
                    let lhs_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160);
                    let lhs_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160);
                    let lhs_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);
                    let lhs_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);
                    let lhs_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);
                    let lhs_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);
                    let lhs_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160);
                    let lhs_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160);

                    let lhs_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245);
                    let lhs_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245);
                    let lhs_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245);
                    let lhs_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245);
                    let lhs_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245);
                    let lhs_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245);
                    let lhs_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245);
                    let lhs_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245);
                    let lhs_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245);
                    let lhs_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245);
                    let lhs_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245);
                    let lhs_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245);
                    let lhs_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245);
                    let lhs_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245);
                    let lhs_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245);
                    let lhs_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245);

                    let iacc_mat_00_0_sp1 = add4_maddubs(rhs_mat_0145_00_sp1, rhs_mat_0145_01_sp1, rhs_mat_0145_02_sp1, rhs_mat_0145_03_sp1, lhs_01_00_sp1, lhs_01_01_sp1, lhs_01_02_sp1, lhs_01_03_sp1);
                    let iacc_mat_01_0_sp1 = add4_maddubs(rhs_mat_2367_00_sp1, rhs_mat_2367_01_sp1, rhs_mat_2367_02_sp1, rhs_mat_2367_03_sp1, lhs_01_00_sp1, lhs_01_01_sp1, lhs_01_02_sp1, lhs_01_03_sp1);
                    let iacc_mat_10_0_sp1 = add4_maddubs(rhs_mat_0145_00_sp1, rhs_mat_0145_01_sp1, rhs_mat_0145_02_sp1, rhs_mat_0145_03_sp1, lhs_23_00_sp1, lhs_23_01_sp1, lhs_23_02_sp1, lhs_23_03_sp1);
                    let iacc_mat_11_0_sp1 = add4_maddubs(rhs_mat_2367_00_sp1, rhs_mat_2367_01_sp1, rhs_mat_2367_02_sp1, rhs_mat_2367_03_sp1, lhs_23_00_sp1, lhs_23_01_sp1, lhs_23_02_sp1, lhs_23_03_sp1);
                    let iacc_mat_00_0_sp2 = add4_maddubs(rhs_mat_0145_00_sp2, rhs_mat_0145_01_sp2, rhs_mat_0145_02_sp2, rhs_mat_0145_03_sp2, lhs_01_00_sp2, lhs_01_01_sp2, lhs_01_02_sp2, lhs_01_03_sp2);
                    let iacc_mat_01_0_sp2 = add4_maddubs(rhs_mat_2367_00_sp2, rhs_mat_2367_01_sp2, rhs_mat_2367_02_sp2, rhs_mat_2367_03_sp2, lhs_01_00_sp2, lhs_01_01_sp2, lhs_01_02_sp2, lhs_01_03_sp2);
                    let iacc_mat_10_0_sp2 = add4_maddubs(rhs_mat_0145_00_sp2, rhs_mat_0145_01_sp2, rhs_mat_0145_02_sp2, rhs_mat_0145_03_sp2, lhs_23_00_sp2, lhs_23_01_sp2, lhs_23_02_sp2, lhs_23_03_sp2);
                    let iacc_mat_11_0_sp2 = add4_maddubs(rhs_mat_2367_00_sp2, rhs_mat_2367_01_sp2, rhs_mat_2367_02_sp2, rhs_mat_2367_03_sp2, lhs_23_00_sp2, lhs_23_01_sp2, lhs_23_02_sp2, lhs_23_03_sp2);

                    let iacc_mat_00_1_sp1 = add4_maddubs(rhs_mat_0145_10_sp1, rhs_mat_0145_11_sp1, rhs_mat_0145_12_sp1, rhs_mat_0145_13_sp1, lhs_01_10_sp1, lhs_01_11_sp1, lhs_01_12_sp1, lhs_01_13_sp1);
                    let iacc_mat_01_1_sp1 = add4_maddubs(rhs_mat_2367_10_sp1, rhs_mat_2367_11_sp1, rhs_mat_2367_12_sp1, rhs_mat_2367_13_sp1, lhs_01_10_sp1, lhs_01_11_sp1, lhs_01_12_sp1, lhs_01_13_sp1);
                    let iacc_mat_10_1_sp1 = add4_maddubs(rhs_mat_0145_10_sp1, rhs_mat_0145_11_sp1, rhs_mat_0145_12_sp1, rhs_mat_0145_13_sp1, lhs_23_10_sp1, lhs_23_11_sp1, lhs_23_12_sp1, lhs_23_13_sp1);
                    let iacc_mat_11_1_sp1 = add4_maddubs(rhs_mat_2367_10_sp1, rhs_mat_2367_11_sp1, rhs_mat_2367_12_sp1, rhs_mat_2367_13_sp1, lhs_23_10_sp1, lhs_23_11_sp1, lhs_23_12_sp1, lhs_23_13_sp1);
                    let iacc_mat_00_1_sp2 = add4_maddubs(rhs_mat_0145_10_sp2, rhs_mat_0145_11_sp2, rhs_mat_0145_12_sp2, rhs_mat_0145_13_sp2, lhs_01_10_sp2, lhs_01_11_sp2, lhs_01_12_sp2, lhs_01_13_sp2);
                    let iacc_mat_01_1_sp2 = add4_maddubs(rhs_mat_2367_10_sp2, rhs_mat_2367_11_sp2, rhs_mat_2367_12_sp2, rhs_mat_2367_13_sp2, lhs_01_10_sp2, lhs_01_11_sp2, lhs_01_12_sp2, lhs_01_13_sp2);
                    let iacc_mat_10_1_sp2 = add4_maddubs(rhs_mat_0145_10_sp2, rhs_mat_0145_11_sp2, rhs_mat_0145_12_sp2, rhs_mat_0145_13_sp2, lhs_23_10_sp2, lhs_23_11_sp2, lhs_23_12_sp2, lhs_23_13_sp2);
                    let iacc_mat_11_1_sp2 = add4_maddubs(rhs_mat_2367_10_sp2, rhs_mat_2367_11_sp2, rhs_mat_2367_12_sp2, rhs_mat_2367_13_sp2, lhs_23_10_sp2, lhs_23_11_sp2, lhs_23_12_sp2, lhs_23_13_sp2);

                    let mut iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    let mut iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    let mut iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    let mut iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                    let mut iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    let mut iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    let mut iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    let mut iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);
                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

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

                    let row_scale_f32_sse = _mm_loadu_ps(a.d.as_ptr());
                    let row_scale_f32 = {
                        let v256 = _mm256_castps128_ps256(row_scale_f32_sse);
                        _mm256_insertf128_ps(v256, row_scale_f32_sse, 1)
                    };

                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    let iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum,   0), mins_01);
                    let iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum,  85), mins_01);
                    let iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum, 170), mins_01);
                    let iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,   0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            for i in 0..4 {
                _mm256_storeu_ps(
                    s.add((y * 4 + i) * bs + x * 8),
                    _mm256_sub_ps(acc_rows[i], acc_min_rows[i]),
                );
            }
        }
    }
}

/// Helper: add4_maddubs (same as in repack_llama.rs, local copy to avoid cross-module dep).
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

// ── Scalar GEMM for Q4_K×8 × Q8_Kx4 with blocklen=4 ─────────────────────────

/// Direct translation of ggml_gemm_q4_K_8x4_q8_K_generic (ggml-cpu/repack.cpp line 1949).
/// Scalar reference implementation: Q4_Kx8 weights × Q8_Kx4 activations, blocklen=4.
pub fn ggml_gemm_q4_k_8x4_q8_k_generic(
    n:  usize,
    s:  *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
) {
    const BLOCKLEN: usize = 4;
    const NCOLS_INTERLEAVED: usize = 8;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    debug_assert_eq!(nr % 4, 0);
    debug_assert_eq!(nc % NCOLS_INTERLEAVED, 0);

    let b_ptr_start = vx as *const BlockQ4Kx8;
    let a_ptr_start = vy as *const BlockQ8Kx4;

    // f16 → f32 via bit manipulation (soft conversion)
    let f16_to_f32 = |bits: u16| -> f32 {
        let sign: u32   = ((bits as u32) >> 15) << 31;
        let exp: u32    = (((bits as u32) >> 10) & 0x1F).wrapping_sub(15).wrapping_add(127) << 23;
        let mant: u32   = ((bits as u32) & 0x3FF) << 13;
        if ((bits >> 10) & 0x1F) == 0 { return 0.0; }  // denorm/zero
        f32::from_bits(sign | exp | mant)
    };

    for y in 0..nr / 4 {
        let a_ptr = unsafe { a_ptr_start.add(y * nb) };
        for x in 0..nc / NCOLS_INTERLEAVED {
            let b_ptr = unsafe { b_ptr_start.add(x * nb) };
            let mut sumf     = [[0.0f32; 8]; 4];
            let mut sum_minf = [[0.0f32; 8]; 4];

            for l in 0..nb {
                let a = unsafe { &*a_ptr.add(l) };
                let b = unsafe { &*b_ptr.add(l) };

                // Decode 8 sub-block scale groups
                let mut utmp = [0u32; 32];
                for sb in 0..8usize {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            b.scales.as_ptr().add(sb * 12),
                            utmp.as_mut_ptr().add(sb * 4) as *mut u8,
                            12,
                        );
                    }
                    let base = sb * 4;
                    utmp[base + 3] = ((utmp[base + 2] >> 4) & KMASK2) |
                                     (((utmp[base + 1] >> 6) & KMASK3) << 4);
                    let uaux = utmp[base + 1] & KMASK1;
                    utmp[base + 1] = (utmp[base + 2] & KMASK2) |
                                     (((utmp[base + 0] >> 6) & KMASK3) << 4);
                    utmp[base + 2] = uaux;
                    utmp[base + 0] &= KMASK1;
                }

                // For each (k, m, j) compute dot products
                let nk = QK_K / (2 * BLOCKLEN); // 256/(2*4) = 32
                for k in 0..nk {
                    let scales_0_base = (k / 8) * 32;
                    let scales_1_base = (k / 8) * 32 + 16;
                    let scales_0 = unsafe {
                        std::slice::from_raw_parts(
                            (utmp.as_ptr() as *const u8).add(scales_0_base), 16)
                    };
                    let scales_1 = unsafe {
                        std::slice::from_raw_parts(
                            (utmp.as_ptr() as *const u8).add(scales_1_base), 16)
                    };

                    for m in 0..4usize {
                        for j in 0..NCOLS_INTERLEAVED {
                            let mut sumi1 = 0i32;
                            let mut sumi2 = 0i32;
                            for ii in 0..BLOCKLEN {
                                let qs_idx = k * NCOLS_INTERLEAVED * BLOCKLEN + j * BLOCKLEN + ii;
                                let byte = b.qs[qs_idx];
                                let v0 = (byte & 0xF) as i8 as i32;
                                let v1 = (byte >> 4)   as i8 as i32;
                                // a layout: (k/8)*256 + (k%8)*4*BLOCKLEN + m*BLOCKLEN + ii
                                let qs0_idx = (k / 8) * 256 + (k % 8) * 4 * BLOCKLEN + m * BLOCKLEN + ii;
                                let qs1_idx = qs0_idx + 128;
                                let a0 = a.qs[qs0_idx] as i32;
                                let a1 = a.qs[qs1_idx] as i32;
                                sumi1 += v0 * a0 * scales_0[j] as i32;
                                sumi2 += v1 * a1 * scales_1[j] as i32;
                            }
                            sumf[m][j] += (sumi1 + sumi2) as f32
                                * f16_to_f32(b.d[j as u16 as usize])
                                * a.d[m];
                        }
                    }
                }

                // Min subtraction
                for sb in 0..8usize {
                    let mins_base = 8 + sb * 16;
                    let mins = unsafe {
                        std::slice::from_raw_parts(
                            (utmp.as_ptr() as *const u8).add(mins_base), 8)
                    };
                    for m in 0..4usize {
                        // bsums index: sb * 8 + m * 4 - (sb % 2) * 6
                        let bsums_idx = (sb * 8 + m * 4).wrapping_sub((sb % 2) * 6);
                        let bsums = &a.bsums[bsums_idx..bsums_idx + 2];
                        let bsum = (bsums[0] + bsums[1]) as f32;
                        for j in 0..NCOLS_INTERLEAVED {
                            sum_minf[m][j] += mins[j] as f32
                                * bsum
                                * f16_to_f32(b.dmin[j])
                                * a.d[m];
                        }
                    }
                }
            }

            // Store results
            for m in 0..4usize {
                for j in 0..NCOLS_INTERLEAVED {
                    unsafe {
                        let out = s.add((y * 4 + m) * bs + x * NCOLS_INTERLEAVED + j);
                        *out = sumf[m][j] - sum_minf[m][j];
                    }
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::blocks::BlockQ4_0;

    fn zero_block_q4_0() -> BlockQ4_0 {
        BlockQ4_0 { d: 0, qs: [0u8; 16] }
    }

    fn zero_block_q8_0() -> BlockQ8_0 {
        BlockQ8_0 { d: 0, qs: [0i8; 32] }
    }

    #[test]
    fn test_block_q4_0x8_size() {
        assert_eq!(std::mem::size_of::<BlockQ4_0x8>(), 8 * 2 + 128);
    }

    #[test]
    fn test_block_q8_0x4_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_0x4>(), 4 * 2 + 128);
    }

    #[test]
    fn test_make_block_q4_0x8_scales() {
        let mut blocks = [zero_block_q4_0(); 8];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i as u16 + 1) * 0x100; // distinct non-zero f16 values
        }
        let packed = make_block_q4_0x8(&blocks, 8);
        for i in 0..8 {
            assert_eq!(packed.d[i], (i as u16 + 1) * 0x100);
        }
    }

    #[test]
    fn test_make_block_q4_0x8_interleaving() {
        let mut blocks = [zero_block_q4_0(); 8];
        for (i, b) in blocks.iter_mut().enumerate() {
            // Fill qs with value (i+1) repeated (without XOR for checking, raw)
            // XOR mask 0x88 per byte; to get predictable output, use 0
            b.qs.fill(0u8);
        }
        let packed = make_block_q4_0x8(&blocks, 8);
        // All qs should be 0 XOR 0x88 = 0x88 = -120 in i8
        let expected = 0x88u8 as i8;
        for &byte in packed.qs.iter() {
            assert_eq!(byte, expected);
        }
    }

    #[test]
    fn test_make_block_q8_0x4_scales() {
        let mut blocks = [zero_block_q8_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.d = (i as u16 + 1) * 0x100;
        }
        let packed = make_block_q8_0x4(&blocks, 8);
        for i in 0..4 {
            assert_eq!(packed.d[i], (i as u16 + 1) * 0x100);
        }
    }

    #[test]
    fn test_make_block_q8_0x4_interleaving() {
        let mut blocks = [zero_block_q8_0(); 4];
        for (i, b) in blocks.iter_mut().enumerate() {
            b.qs.fill((i as i8 + 1));
        }
        let packed = make_block_q8_0x4(&blocks, 8);
        // Expected: B0[0:8], B1[0:8], B2[0:8], B3[0:8], B0[8:16], ...
        for i in 0..16usize {
            let src_id = i % 4;
            for byte_off in 0..8usize {
                assert_eq!(packed.qs[i * 8 + byte_off], (src_id as i8 + 1));
            }
        }
    }

    #[test]
    fn test_repack_q4_0_to_q4_0_8_bl_roundtrip() {
        let nrow = 8;
        let nblocks = 2;
        let mut src = vec![zero_block_q4_0(); nrow * nblocks];
        for (i, b) in src.iter_mut().enumerate() {
            b.d = i as u16 + 1;
        }
        let mut dst = Vec::new();
        repack_q4_0_to_q4_0_8_bl(&src, nrow, nblocks, 8, &mut dst).unwrap();
        assert_eq!(dst.len(), 2); // (8/8) * 2 = 2 packed blocks
        // Each packed block's d[0..8] should come from the 8 rows of that column
        for (i, pb) in dst.iter().enumerate() {
            for row in 0..8usize {
                let expected_d = src[i + row * nblocks].d;
                assert_eq!(pb.d[row], expected_d);
            }
        }
    }

    #[test]
    fn test_repack_q4_0_requires_multiple_of_8() {
        let src = vec![zero_block_q4_0(); 3]; // 3 rows × 1 block
        let mut dst = Vec::new();
        let result = repack_q4_0_to_q4_0_8_bl(&src, 3, 1, 8, &mut dst);
        assert!(result.is_err());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_q8_0_4x8_zero_input() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 32i64;
        let x = vec![0.0f32; 4 * k as usize];
        let mut vy = vec![0u8; std::mem::size_of::<BlockQ8_0x4>()];
        unsafe {
            ggml_quantize_mat_q8_0_4x8(x.as_ptr(), vy.as_mut_ptr(), k);
        }
        let block = unsafe { &*(vy.as_ptr() as *const BlockQ8_0x4) };
        assert!(block.qs.iter().all(|&q| q == 0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_quantize_q8_0_4x8_uniform_input() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let k = 32i64;
        let val = 1.0f32;
        let x = vec![val; 4 * k as usize];
        let mut vy = vec![0u8; std::mem::size_of::<BlockQ8_0x4>()];
        unsafe {
            ggml_quantize_mat_q8_0_4x8(x.as_ptr(), vy.as_mut_ptr(), k);
        }
        let block = unsafe { &*(vy.as_ptr() as *const BlockQ8_0x4) };
        // All quants should be 127 (max positive)
        for &q in block.qs.iter() {
            assert_eq!(q, 127i8, "expected 127, got {}", q);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemv_q4_0_zero_output() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        let n = 32usize;
        let nr = 1;
        let nc = 8;
        let weights = BlockQ4_0x8 { d: [0u16; 8], qs: [0i8; 128] };
        let acts = BlockQ8_0 { d: 0, qs: [0i8; 32] };
        let mut out = vec![0.0f32; nr * nc];
        unsafe {
            ggml_gemv_q4_0_8x8_q8_0(
                n,
                out.as_mut_ptr(),
                &weights as *const BlockQ4_0x8 as *const u8,
                &acts   as *const BlockQ8_0    as *const u8,
                nr,
                nc,
            );
        }
        for &v in &out {
            assert_eq!(v, 0.0f32);
        }
    }

    #[test]
    fn test_gemm_q4_k_8x4_scalar_zero() {
        let n = QK_K;
        let nr = 4;
        let nc = 8;
        let mut s = vec![0.0f32; nr * nc];
        let vx = vec![0u8; std::mem::size_of::<BlockQ4Kx8>()];
        let vy = vec![0u8; std::mem::size_of::<BlockQ8Kx4>()];
        ggml_gemm_q4_k_8x4_q8_k_generic(
            n,
            s.as_mut_ptr(),
            nc,
            vx.as_ptr(),
            vy.as_ptr(),
            nr,
            nc,
        );
        for &v in &s {
            assert_eq!(v, 0.0f32);
        }
    }
}

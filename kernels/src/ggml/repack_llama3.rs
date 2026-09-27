//! Direct translation of llama.cpp arch/x86/repack.cpp — IQ4_NL / MXFP4 GEMV and
//! Q4_0 / IQ4_NL / MXFP4 8×8 GEMM (AVX2 path).
//!
//! Sources:
//! - ggml_gemv_iq4_nl_8x8_q8_0:   arch/x86/repack.cpp  lines 1687-1698
//! - ggml_gemv_mxfp4_8x8_q8_0:    arch/x86/repack.cpp  lines 1700-1711
//! - gemm_q4_b32_8x8_q8_0 (AVX2): arch/x86/repack.cpp  lines 1100-1444
//! - ggml_gemm_q4_0_8x8_q8_0:     arch/x86/repack.cpp  lines 2026-2040
//! - ggml_gemm_iq4_nl_8x8_q8_0:   arch/x86/repack.cpp  lines 3496-3509
//! - ggml_gemm_mxfp4_8x8_q8_0:    arch/x86/repack.cpp  lines 3511-3524

use super::repack_llama2::BlockQ8_0x4;

const QK8_0: usize = 32;

// IQ4_NL 4-bit codebook (= kvalues_iq4nl in ggml-common.h)
const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// MXFP4 / FP4 codebook (= kvalues_fp4 / kvalues_mxfp4 in ggml-common.h)
const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// ── New interleaved block structs ─────────────────────────────────────────────

/// 8 IQ4_NL blocks interleaved — same memory layout as BlockQ4_0x8.
/// d[8]: f16 scales; qs[128]: 4-bit nibbles (iq4_nl index, not raw quant).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockIQ4NLx8 {
    pub d:  [u16; 8],
    pub qs: [i8; 128],
}

/// 8 MXFP4 blocks interleaved.
/// e[8]: E8M0 exponent bytes; qs[128]: 4-bit nibbles.
/// Total: 8 + 128 = 136 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockMxfp4x8 {
    pub e:  [u8; 8],
    pub qs: [i8; 128],
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// GGML_E8M0_TO_FP32_HALF: convert E8M0 byte exponent → f32 with ×0.5.
#[inline(always)]
fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 { 0x00200000u32 << x } else { (x as u32 - 1) << 23 };
    f32::from_bits(bits)
}

/// acc += sign(x)·|x| ⊗ sign(x)·y  (maddubs + madd trick, AVX2).
/// Direct translation of mul_sum_i8_pairs_acc_int32x8 (arch/x86/repack.cpp line 165).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_sum_i8_pairs(
    acc: std::arch::x86_64::__m256i,
    x:   std::arch::x86_64::__m256i,
    y:   std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let ax   = _mm256_sign_epi8(x, x);
    let sy   = _mm256_sign_epi8(y, x);
    let dot  = _mm256_maddubs_epi16(ax, sy);
    let ones = _mm256_set1_epi16(1);
    _mm256_add_epi32(acc, _mm256_madd_epi16(ones, dot))
}

// ── GEMV helpers ──────────────────────────────────────────────────────────────

/// AVX2 GEMV for q4-style blocks with 8 f16 deltas.
/// Used for Q4_0 and IQ4_NL (identical layout, different signextendlut).
/// Direct translation of gemv_q4_b32_8x8_q8_0_lut_avx (arch/x86/repack.cpp line 522)
/// for block_q4_0x8 / block_iq4_nlx8.
///
/// vx must point to (nc/8 × nb) of block_q4_0x8 / block_iq4_nlx8.
/// vy must point to (nr × nb) of block_q8_0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemv_q4_d_lut(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
    signextendlut: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    use super::repack_llama2::BlockQ4_0x8;
    use super::blocks::BlockQ8_0;

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
                let bp = &*b_ptr.offset(b);
                let ap = &*a_ptr.offset(b);

                let rhs_raw_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_2 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_3 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                let rhs_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0, m4b));
                let rhs_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_1, m4b));
                let rhs_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2, m4b));
                let rhs_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_3, m4b));
                let rhs_4 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0, 4), m4b));
                let rhs_5 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_1, 4), m4b));
                let rhs_6 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2, 4), m4b));
                let rhs_7 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_3, 4), m4b));

                // column scales: 8 f16 deltas → 8 f32
                let col_scale_f32 = _mm256_cvtph_ps(_mm_loadu_si128(bp.d.as_ptr() as *const __m128i));

                // row scale
                let row_d_f32 = f16_to_f32(ap.d);
                let row_scale = _mm256_set1_ps(row_d_f32);

                // activations: 32 i8 values → broadcast each 8-byte half
                let lhs_v0 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr() as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr() as *const __m128i)),
                    0,
                );
                let lhs_v1 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr().add(16) as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr().add(16) as *const __m128i)),
                    0,
                );

                let mut iacc = _mm256_setzero_si256();
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_0, _mm256_shuffle_epi32(rhs_1, 177), 170), _mm256_shuffle_epi32(lhs_v0, 0));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0, 177), rhs_1, 170), _mm256_shuffle_epi32(lhs_v0, 85));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_2, _mm256_shuffle_epi32(rhs_3, 177), 170), _mm256_shuffle_epi32(lhs_v0, 170));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_2, 177), rhs_3, 170), _mm256_shuffle_epi32(lhs_v0, 255));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_4, _mm256_shuffle_epi32(rhs_5, 177), 170), _mm256_shuffle_epi32(lhs_v1, 0));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_4, 177), rhs_5, 170), _mm256_shuffle_epi32(lhs_v1, 85));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_6, _mm256_shuffle_epi32(rhs_7, 177), 170), _mm256_shuffle_epi32(lhs_v1, 170));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_6, 177), rhs_7, 170), _mm256_shuffle_epi32(lhs_v1, 255));

                acc_row = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc),
                    _mm256_mul_ps(col_scale_f32, row_scale),
                    acc_row,
                );
            }

            let acc_row = _mm256_permutevar8x32_ps(acc_row, final_permute);
            _mm256_storeu_ps(s.add(y * nr + x * 8), acc_row);
        }
    }
}

/// AVX2 GEMV for MXFP4 blocks (e8m0 exponent scales instead of f16 deltas).
/// Direct translation of gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>
/// (arch/x86/repack.cpp lines 522-636, mxfp4 specialization).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemv_mxfp4_lut(
    n: usize,
    s: *mut f32,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
    signextendlut: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    use super::blocks::BlockQ8_0;

    let nb = n / QK8_0;
    let final_permute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    let b_ptr_start = vx as *const BlockMxfp4x8;
    let a_ptr_start = vy as *const BlockQ8_0;

    for y in 0..nr {
        let a_ptr = a_ptr_start.add(y * nb);
        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_row = _mm256_setzero_ps();

            for b in 0..nb as isize {
                let bp = &*b_ptr.offset(b);
                let ap = &*a_ptr.offset(b);

                let rhs_raw_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_2 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_3 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                let rhs_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0, m4b));
                let rhs_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_1, m4b));
                let rhs_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2, m4b));
                let rhs_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_3, m4b));
                let rhs_4 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0, 4), m4b));
                let rhs_5 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_1, 4), m4b));
                let rhs_6 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2, 4), m4b));
                let rhs_7 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_3, 4), m4b));

                // column scales: 8 E8M0 exponent bytes → 8 f32
                let col_scale_f32 = _mm256_set_ps(
                    e8m0_to_f32_half(bp.e[7]),
                    e8m0_to_f32_half(bp.e[6]),
                    e8m0_to_f32_half(bp.e[5]),
                    e8m0_to_f32_half(bp.e[4]),
                    e8m0_to_f32_half(bp.e[3]),
                    e8m0_to_f32_half(bp.e[2]),
                    e8m0_to_f32_half(bp.e[1]),
                    e8m0_to_f32_half(bp.e[0]),
                );

                let row_d_f32 = f16_to_f32(ap.d);
                let row_scale = _mm256_set1_ps(row_d_f32);

                let lhs_v0 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr() as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr() as *const __m128i)),
                    0,
                );
                let lhs_v1 = _mm256_permute2f128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr().add(16) as *const __m128i)),
                    _mm256_castsi128_si256(_mm_loadu_si128(ap.qs.as_ptr().add(16) as *const __m128i)),
                    0,
                );

                let mut iacc = _mm256_setzero_si256();
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_0, _mm256_shuffle_epi32(rhs_1, 177), 170), _mm256_shuffle_epi32(lhs_v0, 0));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0, 177), rhs_1, 170), _mm256_shuffle_epi32(lhs_v0, 85));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_2, _mm256_shuffle_epi32(rhs_3, 177), 170), _mm256_shuffle_epi32(lhs_v0, 170));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_2, 177), rhs_3, 170), _mm256_shuffle_epi32(lhs_v0, 255));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_4, _mm256_shuffle_epi32(rhs_5, 177), 170), _mm256_shuffle_epi32(lhs_v1, 0));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_4, 177), rhs_5, 170), _mm256_shuffle_epi32(lhs_v1, 85));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(rhs_6, _mm256_shuffle_epi32(rhs_7, 177), 170), _mm256_shuffle_epi32(lhs_v1, 170));
                iacc = mul_sum_i8_pairs(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_6, 177), rhs_7, 170), _mm256_shuffle_epi32(lhs_v1, 255));

                acc_row = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc),
                    _mm256_mul_ps(col_scale_f32, row_scale),
                    acc_row,
                );
            }

            let acc_row = _mm256_permutevar8x32_ps(acc_row, final_permute);
            _mm256_storeu_ps(s.add(y * nr + x * 8), acc_row);
        }
    }
}

// ── GEMM AVX2 core ────────────────────────────────────────────────────────────

/// Load 4 Q8_0x4 row scales (d[4] f16) and repeat: [d0,d1,d0,d1,d2,d3,d2,d3] → 8 f32.
/// Direct translation of GGML_F32Cx8_REPEAT_LOAD (arch/x86/repack.cpp line 35).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn repeat_load_row_scales(d: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let load_mask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(-1_i32), 3);
    let raw = _mm_maskload_epi32(d as *const i32, load_mask);
    let repeated = _mm_shuffle_epi32(raw, 68); // 0b01_00_01_00: [0,1,0,1]
    _mm256_cvtph_ps(repeated)
}

/// Core AVX2 GEMM for blocks with f16 deltas in `d[8]` (q4_0x8 / iq4_nlx8 layout).
/// Direct translation of gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8> AVX2 path
/// (arch/x86/repack.cpp lines 1097-1444).
///
/// - vx: packed weights as (nc/8 × nb) of block with d[8]+qs[128]
/// - vy: packed activations as (nr/4 × nb) of BlockQ8_0x4
/// - s:  output floats, row-major, row stride = bs floats
/// - n, nr, nc: must satisfy n%32==0, nr%4==0, nc%8==0
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemm_q4_b32_avx2_d(
    n:  usize,
    s:  *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
    signextendlut: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    use super::repack_llama2::BlockQ4_0x8;

    let nb = n / QK8_0;
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    let b_ptr_start = vx as *const BlockQ4_0x8;
    let a_ptr_start = vy as *const BlockQ8_0x4;

    let anr = nr - nr % 16;
    let mut y: usize = 0;

    // ── Big loop: 16 output rows at a time (4 groups × 4 rows) ──────────────
    while y < anr / 4 {
        let a_ptrs = [
            a_ptr_start.add(y       * nb),
            a_ptr_start.add((y + 1) * nb),
            a_ptr_start.add((y + 2) * nb),
            a_ptr_start.add((y + 3) * nb),
        ];

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_rows = [_mm256_setzero_ps(); 16];

            for b in 0..nb as isize {
                let bp = &*b_ptr.offset(b);

                // Load 4×32 bytes of RHS nibbles
                let rhs_raw_0123_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_4567_0 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_0123_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_4567_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                // Rearrange to B0B1B4B5 / B2B3B6B7 interleave
                let rhs_raw_0145_0 = _mm256_blend_epi32(rhs_raw_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_4567_0, required_order), 240);
                let rhs_raw_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_0, required_order), rhs_raw_4567_0, 240);
                let rhs_raw_0145_1 = _mm256_blend_epi32(rhs_raw_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_4567_1, required_order), 240);
                let rhs_raw_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_1, required_order), rhs_raw_4567_1, 240);

                // Sign-extend nibbles via LUT (4 groups: lo/hi × first/second half)
                let rhs_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_0, m4b));
                let rhs_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_0, m4b));
                let rhs_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_1, m4b));
                let rhs_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_1, m4b));
                let rhs_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_0, 4), m4b));
                let rhs_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_0, 4), m4b));
                let rhs_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_1, 4), m4b));
                let rhs_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_1, 4), m4b));

                // Shuffle patterns for dot product
                let r0145_0_sp1 = _mm256_shuffle_epi32(rhs_0145_0, 136);
                let r2367_0_sp1 = _mm256_shuffle_epi32(rhs_2367_0, 136);
                let r0145_1_sp1 = _mm256_shuffle_epi32(rhs_0145_1, 136);
                let r2367_1_sp1 = _mm256_shuffle_epi32(rhs_2367_1, 136);
                let r0145_2_sp1 = _mm256_shuffle_epi32(rhs_0145_2, 136);
                let r2367_2_sp1 = _mm256_shuffle_epi32(rhs_2367_2, 136);
                let r0145_3_sp1 = _mm256_shuffle_epi32(rhs_0145_3, 136);
                let r2367_3_sp1 = _mm256_shuffle_epi32(rhs_2367_3, 136);

                let r0145_0_sp2 = _mm256_shuffle_epi32(rhs_0145_0, 221);
                let r2367_0_sp2 = _mm256_shuffle_epi32(rhs_2367_0, 221);
                let r0145_1_sp2 = _mm256_shuffle_epi32(rhs_0145_1, 221);
                let r2367_1_sp2 = _mm256_shuffle_epi32(rhs_2367_1, 221);
                let r0145_2_sp2 = _mm256_shuffle_epi32(rhs_0145_2, 221);
                let r2367_2_sp2 = _mm256_shuffle_epi32(rhs_2367_2, 221);
                let r0145_3_sp2 = _mm256_shuffle_epi32(rhs_0145_3, 221);
                let r2367_3_sp2 = _mm256_shuffle_epi32(rhs_2367_3, 221);

                // Column scale: 8 f16 deltas → 8 f32
                let col_scale = _mm256_cvtph_ps(_mm_loadu_si128(bp.d.as_ptr() as *const __m128i));

                // Process 4 row groups
                for rp in 0..4_usize {
                    let ap = &*a_ptrs[rp].offset(b);

                    // Load LHS (4 rows of Q8_0x4, each 32 i8)
                    let lhs_raw_0 = _mm256_loadu_si256(ap.qs.as_ptr() as *const __m256i);
                    let lhs_01_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 0);
                    let lhs_23_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 17);
                    let lhs_raw_1 = _mm256_loadu_si256(ap.qs.as_ptr().add(32) as *const __m256i);
                    let lhs_01_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 0);
                    let lhs_23_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 17);
                    let lhs_raw_2 = _mm256_loadu_si256(ap.qs.as_ptr().add(64) as *const __m256i);
                    let lhs_01_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 0);
                    let lhs_23_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 17);
                    let lhs_raw_3 = _mm256_loadu_si256(ap.qs.as_ptr().add(96) as *const __m256i);
                    let lhs_01_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 0);
                    let lhs_23_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 17);

                    // LHS shuffle patterns
                    let l01_0_sp1 = _mm256_shuffle_epi32(lhs_01_0, 160);
                    let l23_0_sp1 = _mm256_shuffle_epi32(lhs_23_0, 160);
                    let l01_1_sp1 = _mm256_shuffle_epi32(lhs_01_1, 160);
                    let l23_1_sp1 = _mm256_shuffle_epi32(lhs_23_1, 160);
                    let l01_2_sp1 = _mm256_shuffle_epi32(lhs_01_2, 160);
                    let l23_2_sp1 = _mm256_shuffle_epi32(lhs_23_2, 160);
                    let l01_3_sp1 = _mm256_shuffle_epi32(lhs_01_3, 160);
                    let l23_3_sp1 = _mm256_shuffle_epi32(lhs_23_3, 160);

                    let l01_0_sp2 = _mm256_shuffle_epi32(lhs_01_0, 245);
                    let l23_0_sp2 = _mm256_shuffle_epi32(lhs_23_0, 245);
                    let l01_1_sp2 = _mm256_shuffle_epi32(lhs_01_1, 245);
                    let l23_1_sp2 = _mm256_shuffle_epi32(lhs_23_1, 245);
                    let l01_2_sp2 = _mm256_shuffle_epi32(lhs_01_2, 245);
                    let l23_2_sp2 = _mm256_shuffle_epi32(lhs_23_2, 245);
                    let l01_3_sp2 = _mm256_shuffle_epi32(lhs_01_3, 245);
                    let l23_3_sp2 = _mm256_shuffle_epi32(lhs_23_3, 245);

                    // Dot products: 4 sub-accumulators (sp1)
                    let zero = _mm256_setzero_si256();
                    let i00s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r0145_3_sp1), l01_2_sp1, r0145_2_sp1), l01_1_sp1, r0145_1_sp1), l01_0_sp1, r0145_0_sp1);
                    let i01s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r2367_3_sp1), l01_2_sp1, r2367_2_sp1), l01_1_sp1, r2367_1_sp1), l01_0_sp1, r2367_0_sp1);
                    let i10s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r0145_3_sp1), l23_2_sp1, r0145_2_sp1), l23_1_sp1, r0145_1_sp1), l23_0_sp1, r0145_0_sp1);
                    let i11s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r2367_3_sp1), l23_2_sp1, r2367_2_sp1), l23_1_sp1, r2367_1_sp1), l23_0_sp1, r2367_0_sp1);

                    let i00s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r0145_3_sp2), l01_2_sp2, r0145_2_sp2), l01_1_sp2, r0145_1_sp2), l01_0_sp2, r0145_0_sp2);
                    let i01s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r2367_3_sp2), l01_2_sp2, r2367_2_sp2), l01_1_sp2, r2367_1_sp2), l01_0_sp2, r2367_0_sp2);
                    let i10s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r0145_3_sp2), l23_2_sp2, r0145_2_sp2), l23_1_sp2, r0145_1_sp2), l23_0_sp2, r0145_0_sp2);
                    let i11s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r2367_3_sp2), l23_2_sp2, r2367_2_sp2), l23_1_sp2, r2367_1_sp2), l23_0_sp2, r2367_0_sp2);

                    let i00 = _mm256_add_epi32(i00s1, i00s2);
                    let i01 = _mm256_add_epi32(i01s1, i01s2);
                    let i10 = _mm256_add_epi32(i10s1, i10s2);
                    let i11 = _mm256_add_epi32(i11s1, i11s2);

                    // Rearrange into 4 row vectors
                    let row_0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204);
                    let row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204);
                    let row_2 = _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204);
                    let row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204);

                    // Row scales: d[4] f16, repeated as [d0,d1,d0,d1,d2,d3,d2,d3]
                    let row_scale = repeat_load_row_scales(ap.d.as_ptr());

                    acc_rows[rp * 4    ] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_0), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,   0)), acc_rows[rp * 4    ]);
                    acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_1), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,  85)), acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_2), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_3), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 255)), acc_rows[rp * 4 + 3]);
                }
            }

            for i in 0..16 {
                _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc_rows[i]);
            }
        }
        y += 4;
    }

    // ── Small loop: 4 output rows at a time (1 group) ────────────────────────
    while y < nr / 4 {
        let a_ptr = a_ptr_start.add(y * nb);

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb as isize {
                let bp = &*b_ptr.offset(b);
                let ap = &*a_ptr.offset(b);

                let rhs_raw_0123_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_4567_0 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_0123_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_4567_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                let rhs_raw_0145_0 = _mm256_blend_epi32(rhs_raw_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_4567_0, required_order), 240);
                let rhs_raw_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_0, required_order), rhs_raw_4567_0, 240);
                let rhs_raw_0145_1 = _mm256_blend_epi32(rhs_raw_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_4567_1, required_order), 240);
                let rhs_raw_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_1, required_order), rhs_raw_4567_1, 240);

                let rhs_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_0, m4b));
                let rhs_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_0, m4b));
                let rhs_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_1, m4b));
                let rhs_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_1, m4b));
                let rhs_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_0, 4), m4b));
                let rhs_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_0, 4), m4b));
                let rhs_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_1, 4), m4b));
                let rhs_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_1, 4), m4b));

                let r0145_0_sp1 = _mm256_shuffle_epi32(rhs_0145_0, 136);
                let r2367_0_sp1 = _mm256_shuffle_epi32(rhs_2367_0, 136);
                let r0145_1_sp1 = _mm256_shuffle_epi32(rhs_0145_1, 136);
                let r2367_1_sp1 = _mm256_shuffle_epi32(rhs_2367_1, 136);
                let r0145_2_sp1 = _mm256_shuffle_epi32(rhs_0145_2, 136);
                let r2367_2_sp1 = _mm256_shuffle_epi32(rhs_2367_2, 136);
                let r0145_3_sp1 = _mm256_shuffle_epi32(rhs_0145_3, 136);
                let r2367_3_sp1 = _mm256_shuffle_epi32(rhs_2367_3, 136);

                let r0145_0_sp2 = _mm256_shuffle_epi32(rhs_0145_0, 221);
                let r2367_0_sp2 = _mm256_shuffle_epi32(rhs_2367_0, 221);
                let r0145_1_sp2 = _mm256_shuffle_epi32(rhs_0145_1, 221);
                let r2367_1_sp2 = _mm256_shuffle_epi32(rhs_2367_1, 221);
                let r0145_2_sp2 = _mm256_shuffle_epi32(rhs_0145_2, 221);
                let r2367_2_sp2 = _mm256_shuffle_epi32(rhs_2367_2, 221);
                let r0145_3_sp2 = _mm256_shuffle_epi32(rhs_0145_3, 221);
                let r2367_3_sp2 = _mm256_shuffle_epi32(rhs_2367_3, 221);

                let col_scale = _mm256_cvtph_ps(_mm_loadu_si128(bp.d.as_ptr() as *const __m128i));

                // Single row group
                let lhs_raw_0 = _mm256_loadu_si256(ap.qs.as_ptr() as *const __m256i);
                let lhs_01_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 0);
                let lhs_23_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 17);
                let lhs_raw_1 = _mm256_loadu_si256(ap.qs.as_ptr().add(32) as *const __m256i);
                let lhs_01_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 0);
                let lhs_23_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 17);
                let lhs_raw_2 = _mm256_loadu_si256(ap.qs.as_ptr().add(64) as *const __m256i);
                let lhs_01_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 0);
                let lhs_23_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 17);
                let lhs_raw_3 = _mm256_loadu_si256(ap.qs.as_ptr().add(96) as *const __m256i);
                let lhs_01_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 0);
                let lhs_23_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 17);

                let l01_0_sp1 = _mm256_shuffle_epi32(lhs_01_0, 160);
                let l23_0_sp1 = _mm256_shuffle_epi32(lhs_23_0, 160);
                let l01_1_sp1 = _mm256_shuffle_epi32(lhs_01_1, 160);
                let l23_1_sp1 = _mm256_shuffle_epi32(lhs_23_1, 160);
                let l01_2_sp1 = _mm256_shuffle_epi32(lhs_01_2, 160);
                let l23_2_sp1 = _mm256_shuffle_epi32(lhs_23_2, 160);
                let l01_3_sp1 = _mm256_shuffle_epi32(lhs_01_3, 160);
                let l23_3_sp1 = _mm256_shuffle_epi32(lhs_23_3, 160);

                let l01_0_sp2 = _mm256_shuffle_epi32(lhs_01_0, 245);
                let l23_0_sp2 = _mm256_shuffle_epi32(lhs_23_0, 245);
                let l01_1_sp2 = _mm256_shuffle_epi32(lhs_01_1, 245);
                let l23_1_sp2 = _mm256_shuffle_epi32(lhs_23_1, 245);
                let l01_2_sp2 = _mm256_shuffle_epi32(lhs_01_2, 245);
                let l23_2_sp2 = _mm256_shuffle_epi32(lhs_23_2, 245);
                let l01_3_sp2 = _mm256_shuffle_epi32(lhs_01_3, 245);
                let l23_3_sp2 = _mm256_shuffle_epi32(lhs_23_3, 245);

                let zero = _mm256_setzero_si256();
                let i00s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r0145_3_sp1), l01_2_sp1, r0145_2_sp1), l01_1_sp1, r0145_1_sp1), l01_0_sp1, r0145_0_sp1);
                let i01s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r2367_3_sp1), l01_2_sp1, r2367_2_sp1), l01_1_sp1, r2367_1_sp1), l01_0_sp1, r2367_0_sp1);
                let i10s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r0145_3_sp1), l23_2_sp1, r0145_2_sp1), l23_1_sp1, r0145_1_sp1), l23_0_sp1, r0145_0_sp1);
                let i11s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r2367_3_sp1), l23_2_sp1, r2367_2_sp1), l23_1_sp1, r2367_1_sp1), l23_0_sp1, r2367_0_sp1);

                let i00s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r0145_3_sp2), l01_2_sp2, r0145_2_sp2), l01_1_sp2, r0145_1_sp2), l01_0_sp2, r0145_0_sp2);
                let i01s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r2367_3_sp2), l01_2_sp2, r2367_2_sp2), l01_1_sp2, r2367_1_sp2), l01_0_sp2, r2367_0_sp2);
                let i10s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r0145_3_sp2), l23_2_sp2, r0145_2_sp2), l23_1_sp2, r0145_1_sp2), l23_0_sp2, r0145_0_sp2);
                let i11s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r2367_3_sp2), l23_2_sp2, r2367_2_sp2), l23_1_sp2, r2367_1_sp2), l23_0_sp2, r2367_0_sp2);

                let i00 = _mm256_add_epi32(i00s1, i00s2);
                let i01 = _mm256_add_epi32(i01s1, i01s2);
                let i10 = _mm256_add_epi32(i10s1, i10s2);
                let i11 = _mm256_add_epi32(i11s1, i11s2);

                let row_0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204);
                let row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204);
                let row_2 = _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204);
                let row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204);

                let row_scale = repeat_load_row_scales(ap.d.as_ptr());

                acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_0), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,   0)), acc_rows[0]);
                acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_1), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,  85)), acc_rows[1]);
                acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_2), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 170)), acc_rows[2]);
                acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_3), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 255)), acc_rows[3]);
            }

            for i in 0..4 {
                _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc_rows[i]);
            }
        }
        y += 1;
    }
}

/// Core AVX2 GEMM for MXFP4 blocks (e8m0 exponent scales, e[8]+qs[128]).
/// Direct translation of gemm_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8> AVX2 path
/// (arch/x86/repack.cpp lines 1097-1444, mxfp4 specialization).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn gemm_q4_b32_avx2_mxfp4(
    n:  usize,
    s:  *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: usize,
    nc: usize,
    signextendlut: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;

    let nb = n / QK8_0;
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    let b_ptr_start = vx as *const BlockMxfp4x8;
    let a_ptr_start = vy as *const BlockQ8_0x4;

    let anr = nr - nr % 16;
    let mut y: usize = 0;

    // ── Big loop: 16 output rows at a time ───────────────────────────────────
    while y < anr / 4 {
        let a_ptrs = [
            a_ptr_start.add(y       * nb),
            a_ptr_start.add((y + 1) * nb),
            a_ptr_start.add((y + 2) * nb),
            a_ptr_start.add((y + 3) * nb),
        ];

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_rows = [_mm256_setzero_ps(); 16];

            for b in 0..nb as isize {
                let bp = &*b_ptr.offset(b);

                let rhs_raw_0123_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_4567_0 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_0123_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_4567_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                let rhs_raw_0145_0 = _mm256_blend_epi32(rhs_raw_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_4567_0, required_order), 240);
                let rhs_raw_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_0, required_order), rhs_raw_4567_0, 240);
                let rhs_raw_0145_1 = _mm256_blend_epi32(rhs_raw_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_4567_1, required_order), 240);
                let rhs_raw_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_1, required_order), rhs_raw_4567_1, 240);

                let rhs_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_0, m4b));
                let rhs_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_0, m4b));
                let rhs_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_1, m4b));
                let rhs_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_1, m4b));
                let rhs_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_0, 4), m4b));
                let rhs_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_0, 4), m4b));
                let rhs_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_1, 4), m4b));
                let rhs_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_1, 4), m4b));

                let r0145_0_sp1 = _mm256_shuffle_epi32(rhs_0145_0, 136);
                let r2367_0_sp1 = _mm256_shuffle_epi32(rhs_2367_0, 136);
                let r0145_1_sp1 = _mm256_shuffle_epi32(rhs_0145_1, 136);
                let r2367_1_sp1 = _mm256_shuffle_epi32(rhs_2367_1, 136);
                let r0145_2_sp1 = _mm256_shuffle_epi32(rhs_0145_2, 136);
                let r2367_2_sp1 = _mm256_shuffle_epi32(rhs_2367_2, 136);
                let r0145_3_sp1 = _mm256_shuffle_epi32(rhs_0145_3, 136);
                let r2367_3_sp1 = _mm256_shuffle_epi32(rhs_2367_3, 136);

                let r0145_0_sp2 = _mm256_shuffle_epi32(rhs_0145_0, 221);
                let r2367_0_sp2 = _mm256_shuffle_epi32(rhs_2367_0, 221);
                let r0145_1_sp2 = _mm256_shuffle_epi32(rhs_0145_1, 221);
                let r2367_1_sp2 = _mm256_shuffle_epi32(rhs_2367_1, 221);
                let r0145_2_sp2 = _mm256_shuffle_epi32(rhs_0145_2, 221);
                let r2367_2_sp2 = _mm256_shuffle_epi32(rhs_2367_2, 221);
                let r0145_3_sp2 = _mm256_shuffle_epi32(rhs_0145_3, 221);
                let r2367_3_sp2 = _mm256_shuffle_epi32(rhs_2367_3, 221);

                // Column scales from E8M0 exponents
                let col_scale = _mm256_set_ps(
                    e8m0_to_f32_half(bp.e[7]), e8m0_to_f32_half(bp.e[6]),
                    e8m0_to_f32_half(bp.e[5]), e8m0_to_f32_half(bp.e[4]),
                    e8m0_to_f32_half(bp.e[3]), e8m0_to_f32_half(bp.e[2]),
                    e8m0_to_f32_half(bp.e[1]), e8m0_to_f32_half(bp.e[0]),
                );

                for rp in 0..4_usize {
                    let ap = &*a_ptrs[rp].offset(b);

                    let lhs_raw_0 = _mm256_loadu_si256(ap.qs.as_ptr() as *const __m256i);
                    let lhs_01_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 0);
                    let lhs_23_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 17);
                    let lhs_raw_1 = _mm256_loadu_si256(ap.qs.as_ptr().add(32) as *const __m256i);
                    let lhs_01_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 0);
                    let lhs_23_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 17);
                    let lhs_raw_2 = _mm256_loadu_si256(ap.qs.as_ptr().add(64) as *const __m256i);
                    let lhs_01_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 0);
                    let lhs_23_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 17);
                    let lhs_raw_3 = _mm256_loadu_si256(ap.qs.as_ptr().add(96) as *const __m256i);
                    let lhs_01_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 0);
                    let lhs_23_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 17);

                    let l01_0_sp1 = _mm256_shuffle_epi32(lhs_01_0, 160);
                    let l23_0_sp1 = _mm256_shuffle_epi32(lhs_23_0, 160);
                    let l01_1_sp1 = _mm256_shuffle_epi32(lhs_01_1, 160);
                    let l23_1_sp1 = _mm256_shuffle_epi32(lhs_23_1, 160);
                    let l01_2_sp1 = _mm256_shuffle_epi32(lhs_01_2, 160);
                    let l23_2_sp1 = _mm256_shuffle_epi32(lhs_23_2, 160);
                    let l01_3_sp1 = _mm256_shuffle_epi32(lhs_01_3, 160);
                    let l23_3_sp1 = _mm256_shuffle_epi32(lhs_23_3, 160);

                    let l01_0_sp2 = _mm256_shuffle_epi32(lhs_01_0, 245);
                    let l23_0_sp2 = _mm256_shuffle_epi32(lhs_23_0, 245);
                    let l01_1_sp2 = _mm256_shuffle_epi32(lhs_01_1, 245);
                    let l23_1_sp2 = _mm256_shuffle_epi32(lhs_23_1, 245);
                    let l01_2_sp2 = _mm256_shuffle_epi32(lhs_01_2, 245);
                    let l23_2_sp2 = _mm256_shuffle_epi32(lhs_23_2, 245);
                    let l01_3_sp2 = _mm256_shuffle_epi32(lhs_01_3, 245);
                    let l23_3_sp2 = _mm256_shuffle_epi32(lhs_23_3, 245);

                    let zero = _mm256_setzero_si256();
                    let i00s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r0145_3_sp1), l01_2_sp1, r0145_2_sp1), l01_1_sp1, r0145_1_sp1), l01_0_sp1, r0145_0_sp1);
                    let i01s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r2367_3_sp1), l01_2_sp1, r2367_2_sp1), l01_1_sp1, r2367_1_sp1), l01_0_sp1, r2367_0_sp1);
                    let i10s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r0145_3_sp1), l23_2_sp1, r0145_2_sp1), l23_1_sp1, r0145_1_sp1), l23_0_sp1, r0145_0_sp1);
                    let i11s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r2367_3_sp1), l23_2_sp1, r2367_2_sp1), l23_1_sp1, r2367_1_sp1), l23_0_sp1, r2367_0_sp1);

                    let i00s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r0145_3_sp2), l01_2_sp2, r0145_2_sp2), l01_1_sp2, r0145_1_sp2), l01_0_sp2, r0145_0_sp2);
                    let i01s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r2367_3_sp2), l01_2_sp2, r2367_2_sp2), l01_1_sp2, r2367_1_sp2), l01_0_sp2, r2367_0_sp2);
                    let i10s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r0145_3_sp2), l23_2_sp2, r0145_2_sp2), l23_1_sp2, r0145_1_sp2), l23_0_sp2, r0145_0_sp2);
                    let i11s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r2367_3_sp2), l23_2_sp2, r2367_2_sp2), l23_1_sp2, r2367_1_sp2), l23_0_sp2, r2367_0_sp2);

                    let i00 = _mm256_add_epi32(i00s1, i00s2);
                    let i01 = _mm256_add_epi32(i01s1, i01s2);
                    let i10 = _mm256_add_epi32(i10s1, i10s2);
                    let i11 = _mm256_add_epi32(i11s1, i11s2);

                    let row_0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204);
                    let row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204);
                    let row_2 = _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204);
                    let row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204);

                    let row_scale = repeat_load_row_scales(ap.d.as_ptr());

                    acc_rows[rp * 4    ] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_0), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,   0)), acc_rows[rp * 4    ]);
                    acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_1), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,  85)), acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_2), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_3), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 255)), acc_rows[rp * 4 + 3]);
                }
            }

            for i in 0..16 {
                _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc_rows[i]);
            }
        }
        y += 4;
    }

    // ── Small loop: 4 output rows at a time ──────────────────────────────────
    while y < nr / 4 {
        let a_ptr = a_ptr_start.add(y * nb);

        for x in 0..nc / 8 {
            let b_ptr = b_ptr_start.add(x * nb);
            let mut acc_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb as isize {
                let bp = &*b_ptr.offset(b);
                let ap = &*a_ptr.offset(b);

                let rhs_raw_0123_0 = _mm256_loadu_si256(bp.qs.as_ptr() as *const __m256i);
                let rhs_raw_4567_0 = _mm256_loadu_si256(bp.qs.as_ptr().add(32) as *const __m256i);
                let rhs_raw_0123_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(64) as *const __m256i);
                let rhs_raw_4567_1 = _mm256_loadu_si256(bp.qs.as_ptr().add(96) as *const __m256i);

                let rhs_raw_0145_0 = _mm256_blend_epi32(rhs_raw_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_4567_0, required_order), 240);
                let rhs_raw_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_0, required_order), rhs_raw_4567_0, 240);
                let rhs_raw_0145_1 = _mm256_blend_epi32(rhs_raw_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_4567_1, required_order), 240);
                let rhs_raw_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_0123_1, required_order), rhs_raw_4567_1, 240);

                let rhs_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_0, m4b));
                let rhs_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_0, m4b));
                let rhs_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_0145_1, m4b));
                let rhs_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_2367_1, m4b));
                let rhs_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_0, 4), m4b));
                let rhs_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_0, 4), m4b));
                let rhs_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_0145_1, 4), m4b));
                let rhs_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_2367_1, 4), m4b));

                let r0145_0_sp1 = _mm256_shuffle_epi32(rhs_0145_0, 136);
                let r2367_0_sp1 = _mm256_shuffle_epi32(rhs_2367_0, 136);
                let r0145_1_sp1 = _mm256_shuffle_epi32(rhs_0145_1, 136);
                let r2367_1_sp1 = _mm256_shuffle_epi32(rhs_2367_1, 136);
                let r0145_2_sp1 = _mm256_shuffle_epi32(rhs_0145_2, 136);
                let r2367_2_sp1 = _mm256_shuffle_epi32(rhs_2367_2, 136);
                let r0145_3_sp1 = _mm256_shuffle_epi32(rhs_0145_3, 136);
                let r2367_3_sp1 = _mm256_shuffle_epi32(rhs_2367_3, 136);

                let r0145_0_sp2 = _mm256_shuffle_epi32(rhs_0145_0, 221);
                let r2367_0_sp2 = _mm256_shuffle_epi32(rhs_2367_0, 221);
                let r0145_1_sp2 = _mm256_shuffle_epi32(rhs_0145_1, 221);
                let r2367_1_sp2 = _mm256_shuffle_epi32(rhs_2367_1, 221);
                let r0145_2_sp2 = _mm256_shuffle_epi32(rhs_0145_2, 221);
                let r2367_2_sp2 = _mm256_shuffle_epi32(rhs_2367_2, 221);
                let r0145_3_sp2 = _mm256_shuffle_epi32(rhs_0145_3, 221);
                let r2367_3_sp2 = _mm256_shuffle_epi32(rhs_2367_3, 221);

                let col_scale = _mm256_set_ps(
                    e8m0_to_f32_half(bp.e[7]), e8m0_to_f32_half(bp.e[6]),
                    e8m0_to_f32_half(bp.e[5]), e8m0_to_f32_half(bp.e[4]),
                    e8m0_to_f32_half(bp.e[3]), e8m0_to_f32_half(bp.e[2]),
                    e8m0_to_f32_half(bp.e[1]), e8m0_to_f32_half(bp.e[0]),
                );

                let lhs_raw_0 = _mm256_loadu_si256(ap.qs.as_ptr() as *const __m256i);
                let lhs_01_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 0);
                let lhs_23_0 = _mm256_permute2f128_si256(lhs_raw_0, lhs_raw_0, 17);
                let lhs_raw_1 = _mm256_loadu_si256(ap.qs.as_ptr().add(32) as *const __m256i);
                let lhs_01_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 0);
                let lhs_23_1 = _mm256_permute2f128_si256(lhs_raw_1, lhs_raw_1, 17);
                let lhs_raw_2 = _mm256_loadu_si256(ap.qs.as_ptr().add(64) as *const __m256i);
                let lhs_01_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 0);
                let lhs_23_2 = _mm256_permute2f128_si256(lhs_raw_2, lhs_raw_2, 17);
                let lhs_raw_3 = _mm256_loadu_si256(ap.qs.as_ptr().add(96) as *const __m256i);
                let lhs_01_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 0);
                let lhs_23_3 = _mm256_permute2f128_si256(lhs_raw_3, lhs_raw_3, 17);

                let l01_0_sp1 = _mm256_shuffle_epi32(lhs_01_0, 160);
                let l23_0_sp1 = _mm256_shuffle_epi32(lhs_23_0, 160);
                let l01_1_sp1 = _mm256_shuffle_epi32(lhs_01_1, 160);
                let l23_1_sp1 = _mm256_shuffle_epi32(lhs_23_1, 160);
                let l01_2_sp1 = _mm256_shuffle_epi32(lhs_01_2, 160);
                let l23_2_sp1 = _mm256_shuffle_epi32(lhs_23_2, 160);
                let l01_3_sp1 = _mm256_shuffle_epi32(lhs_01_3, 160);
                let l23_3_sp1 = _mm256_shuffle_epi32(lhs_23_3, 160);

                let l01_0_sp2 = _mm256_shuffle_epi32(lhs_01_0, 245);
                let l23_0_sp2 = _mm256_shuffle_epi32(lhs_23_0, 245);
                let l01_1_sp2 = _mm256_shuffle_epi32(lhs_01_1, 245);
                let l23_1_sp2 = _mm256_shuffle_epi32(lhs_23_1, 245);
                let l01_2_sp2 = _mm256_shuffle_epi32(lhs_01_2, 245);
                let l23_2_sp2 = _mm256_shuffle_epi32(lhs_23_2, 245);
                let l01_3_sp2 = _mm256_shuffle_epi32(lhs_01_3, 245);
                let l23_3_sp2 = _mm256_shuffle_epi32(lhs_23_3, 245);

                let zero = _mm256_setzero_si256();
                let i00s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r0145_3_sp1), l01_2_sp1, r0145_2_sp1), l01_1_sp1, r0145_1_sp1), l01_0_sp1, r0145_0_sp1);
                let i01s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp1, r2367_3_sp1), l01_2_sp1, r2367_2_sp1), l01_1_sp1, r2367_1_sp1), l01_0_sp1, r2367_0_sp1);
                let i10s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r0145_3_sp1), l23_2_sp1, r0145_2_sp1), l23_1_sp1, r0145_1_sp1), l23_0_sp1, r0145_0_sp1);
                let i11s1 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp1, r2367_3_sp1), l23_2_sp1, r2367_2_sp1), l23_1_sp1, r2367_1_sp1), l23_0_sp1, r2367_0_sp1);

                let i00s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r0145_3_sp2), l01_2_sp2, r0145_2_sp2), l01_1_sp2, r0145_1_sp2), l01_0_sp2, r0145_0_sp2);
                let i01s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l01_3_sp2, r2367_3_sp2), l01_2_sp2, r2367_2_sp2), l01_1_sp2, r2367_1_sp2), l01_0_sp2, r2367_0_sp2);
                let i10s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r0145_3_sp2), l23_2_sp2, r0145_2_sp2), l23_1_sp2, r0145_1_sp2), l23_0_sp2, r0145_0_sp2);
                let i11s2 = mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(mul_sum_i8_pairs(zero, l23_3_sp2, r2367_3_sp2), l23_2_sp2, r2367_2_sp2), l23_1_sp2, r2367_1_sp2), l23_0_sp2, r2367_0_sp2);

                let i00 = _mm256_add_epi32(i00s1, i00s2);
                let i01 = _mm256_add_epi32(i01s1, i01s2);
                let i10 = _mm256_add_epi32(i10s1, i10s2);
                let i11 = _mm256_add_epi32(i11s1, i11s2);

                let row_0 = _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204);
                let row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204);
                let row_2 = _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204);
                let row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204);

                let row_scale = repeat_load_row_scales(ap.d.as_ptr());

                acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_0), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,   0)), acc_rows[0]);
                acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_1), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale,  85)), acc_rows[1]);
                acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_2), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 170)), acc_rows[2]);
                acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(row_3), _mm256_mul_ps(col_scale, _mm256_shuffle_ps(row_scale, row_scale, 255)), acc_rows[3]);
            }

            for i in 0..4 {
                _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc_rows[i]);
            }
        }
        y += 1;
    }
}

// ── f16 → f32 conversion helper ──────────────────────────────────────────────

#[inline(always)]
fn f16_to_f32(v: u16) -> f32 {
    // IEEE 754 half → single via bitfield expansion.
    let sign = ((v as u32) >> 15) & 1;
    let exp  = ((v as u32) >> 10) & 0x1f;
    let mant =  (v as u32)        & 0x3ff;
    let bits = if exp == 0 {
        // subnormal → flush to zero for our use case
        if mant == 0 { sign << 31 } else { (sign << 31) | ((mant as u32) << 13) }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ── Public GEMV functions ─────────────────────────────────────────────────────

/// Direct translation of ggml_gemv_iq4_nl_8x8_q8_0 (arch/x86/repack.cpp line 1687).
/// AVX2 GEMV: BlockIQ4NLx8 weights × Q8_0 activations → float output.
/// IQ4_NL layout is identical to BlockQ4_0x8 (d[8] f16 + qs[128]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemv_iq4_nl_8x8_q8_0(
    n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let lut128 = _mm_loadu_si128(IQ4_NL_VALUES.as_ptr() as *const __m128i);
    let signextendlut = _mm256_permute2f128_si256(_mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    gemv_q4_d_lut(n, s, vx, vy, nr, nc, signextendlut);
}

/// Direct translation of ggml_gemv_mxfp4_8x8_q8_0 (arch/x86/repack.cpp line 1700).
/// AVX2 GEMV: BlockMxfp4x8 weights × Q8_0 activations → float output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemv_mxfp4_8x8_q8_0(
    n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
    let signextendlut = _mm256_permute2f128_si256(_mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    gemv_mxfp4_lut(n, s, vx, vy, nr, nc, signextendlut);
}

// ── Public GEMM functions ─────────────────────────────────────────────────────

/// Direct translation of ggml_gemm_q4_0_8x8_q8_0 (arch/x86/repack.cpp line 2026).
/// AVX2 GEMM: BlockQ4_0x8 weights × BlockQ8_0x4 activations → float output.
///
/// vx: (nc/8 × nb) packed Q4_0 blocks;  vy: (nr/4 × nb) packed Q8_0x4 blocks.
/// s: output row-major, row stride bs.  Requires nr%4==0, nc%8==0, n%32==0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemm_q4_0_8x8_q8_0(
    n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let half = _mm_set_epi8(-1,-2,-3,-4,-5,-6,-7,-8, 7,6,5,4,3,2,1,0);
    let signextendlut = _mm256_permute2f128_si256(_mm256_castsi128_si256(half), _mm256_castsi128_si256(half), 0);
    gemm_q4_b32_avx2_d(n, s, bs, vx, vy, nr, nc, signextendlut);
}

/// Direct translation of ggml_gemm_iq4_nl_8x8_q8_0 (arch/x86/repack.cpp line 3496).
/// AVX2 GEMM: BlockIQ4NLx8 weights × BlockQ8_0x4 activations → float output.
/// Layout identical to Q4_0x8 (d[8] f16 + qs[128]); only the LUT differs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemm_iq4_nl_8x8_q8_0(
    n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let lut128 = _mm_loadu_si128(IQ4_NL_VALUES.as_ptr() as *const __m128i);
    let signextendlut = _mm256_permute2f128_si256(_mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    gemm_q4_b32_avx2_d(n, s, bs, vx, vy, nr, nc, signextendlut);
}

/// Direct translation of ggml_gemm_mxfp4_8x8_q8_0 (arch/x86/repack.cpp line 3511).
/// AVX2 GEMM: BlockMxfp4x8 weights × BlockQ8_0x4 activations → float output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn ggml_gemm_mxfp4_8x8_q8_0(
    n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize,
) {
    use std::arch::x86_64::*;
    let lut128 = _mm_loadu_si128(MXFP4_VALUES.as_ptr() as *const __m128i);
    let signextendlut = _mm256_permute2f128_si256(_mm256_castsi128_si256(lut128), _mm256_castsi128_si256(lut128), 0);
    gemm_q4_b32_avx2_mxfp4(n, s, bs, vx, vy, nr, nc, signextendlut);
}

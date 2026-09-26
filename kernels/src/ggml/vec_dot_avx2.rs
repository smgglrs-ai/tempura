#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_i32_4(v: __m128i) -> i32 {
    let hi = _mm_unpackhi_epi64(v, v);
    let sum2 = _mm_add_epi32(v, hi);
    let hi32 = _mm_shuffle_epi32(sum2, 0b00_00_00_01);
    let sum1 = _mm_add_epi32(sum2, hi32);
    _mm_cvtsi128_si32(sum1)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_i32_8(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_f32_8(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_movehdup_ps(sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

// ── Q4_0 × Q8_0 ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q4_0_q8_0_avx2(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;
    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d = f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);

        let bx_bytes = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let lo = _mm256_cvtepu8_epi16(bx_bytes);
        let bx_shifted = _mm_srli_epi16(bx_bytes, 4);
        let hi = _mm256_cvtepu8_epi16(bx_shifted);

        // mask to 4 bits
        let mask = _mm256_set1_epi16(0x0F);
        let lo = _mm256_and_si256(lo, mask);
        let hi = _mm256_and_si256(hi, mask);

        // subtract 8
        let off = _mm256_set1_epi16(8);
        let lo = _mm256_sub_epi16(lo, off);
        let hi = _mm256_sub_epi16(hi, off);

        // load q8
        let by_lo =
            _mm256_cvtepi8_epi16(_mm_loadu_si128(y[ib].qs[..16].as_ptr() as *const __m128i));
        let by_hi =
            _mm256_cvtepi8_epi16(_mm_loadu_si128(y[ib].qs[16..].as_ptr() as *const __m128i));

        // multiply and horizontal add: i16×i16 → i32
        let dot_lo = _mm256_madd_epi16(lo, by_lo);
        let dot_hi = _mm256_madd_epi16(hi, by_hi);
        let sum = _mm256_add_epi32(dot_lo, dot_hi);

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum), acc);
    }
    hsum_f32_8(acc)
}

// ── Q8_0 × Q8_0 ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_0_q8_0_avx2(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;
    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let d = f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);

        let bx = _mm256_loadu_si256(x[ib].qs.as_ptr() as *const __m256i);
        let by = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        // maddubs: treat bx as unsigned, by as signed → pairwise i16 products summed
        // but bx is signed (i8) so we need to handle sign. Simpler: use madd_epi16 on i16
        let bx_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bx));
        let bx_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bx, 1));
        let by_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(by));
        let by_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(by, 1));

        let dot_lo = _mm256_madd_epi16(bx_lo, by_lo);
        let dot_hi = _mm256_madd_epi16(bx_hi, by_hi);
        let sum = _mm256_add_epi32(dot_lo, dot_hi);

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum), acc);
    }
    hsum_f32_8(acc)
}

// ── Q4_K × Q8_K ───────────────────────────────────────────────────────────
// Direct translation of ggml's AVX2 Q4K vec_dot: maddubs on nibbles,
// scale applied via shuffle_epi8 inside madd_epi16 (no dequant buffer).

// Scale shuffle table: for each of the 8 scale groups (indexed 0..7),
// provides a 32-byte shuffle mask that broadcasts the correct i16 scale
// value to all 16 positions in the __m256i where that scale's Q4 values live.
#[cfg(target_arch = "x86_64")]
static K_SHUFFLE_Q4K: [[u8; 32]; 8] = [
    [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1,
    ],
    [
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3,
    ],
    [
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5,
    ],
    [
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7,
    ],
    [
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9,
    ],
    [
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11,
    ],
    [
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 12, 13, 12, 13, 12, 13,
    ],
    [
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14,
        15, 14, 15, 14, 15, 14, 15, 14, 15,
    ],
];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn vec_dot_q4_k_q8_k_avx2(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let m4 = _mm256_set1_epi8(0x0F);

    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut utmp = [0u32; 4];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    for i in 0..nb {
        // Safety: nb = n/256, x.len() == nb, y.len() == nb (caller invariant)
        let xi = &*x_ptr.add(i);
        let yi = &*y_ptr.add(i);
        let d = yi.d * f16_to_f32(xi.d);
        let dmin = -(yi.d * f16_to_f32(xi.dmin));

        // unpack scales and mins
        std::ptr::copy_nonoverlapping(xi.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        // mins_and_scales: 8 scales (i16) in low 128 bits, 8 mins (i16) in high 128 bits
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        // mins accumulation: hadd bsums pairs, madd with mins, fmadd into acc_m
        let q8sums = _mm256_loadu_si256(yi.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        // duplicate scales (low 128 bits) into both halves for shuffle
        let sc128 = _mm256_castsi256_si128(mins_and_scales);
        let scales = _mm256_set_m128i(sc128, sc128);

        let q4 = xi.qs.as_ptr();
        let q8 = yi.qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();

        for j in 0..4 {
            // Safety: j < 4 → 2*j+1 < 8 == K_SHUFFLE_Q4K.len()
            let scale_l = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(K_SHUFFLE_Q4K.get_unchecked(2 * j).as_ptr() as *const __m256i),
            );
            let scale_h = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(
                    K_SHUFFLE_Q4K.get_unchecked(2 * j + 1).as_ptr() as *const __m256i
                ),
            );

            // load 32 Q4 bytes, extract low/high nibbles
            let q4bits = _mm256_loadu_si256(q4.add(j * 32) as *const __m256i);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            // maddubs(u8 nibbles, i8 q8) → 16 i16 products
            // then madd(i16 scale, i16 products) → 8 i32 scaled sums
            let q8l = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    // reduce acc_m (4 f32 lanes) to scalar
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    hsum_f32_8(acc) + _mm_cvtss_f32(acc_m)
}

// ── Q6_K × Q8_K ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q6_k_q8_k_avx2(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let ql = &x[i].ql;
        let qh = &x[i].qh;
        let sc = &x[i].scales;
        let q8 = &y[i].qs;

        // dequant 256 elements into aux buffer, then dot with q8 using AVX2
        let mut aux8 = [0i8; 256];
        let mut a_idx = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _half in 0..2 {
            for l in 0..32 {
                aux8[a_idx + l] = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
                aux8[a_idx + 32 + l] =
                    ((ql[ql_off + 32 + l] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_idx + 64 + l] =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_idx + 96 + l] =
                    ((ql[ql_off + 32 + l] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            }
            a_idx += 128;
            ql_off += 64;
            qh_off += 32;
        }

        // dot product with scaling, 16 elements per scale group
        let mut sumi = _mm256_setzero_si256();
        for j in 0..16 {
            let scale = sc[j] as i32;
            let base = j * 16;
            // load 16 dequantized i8 values and 16 q8 values, multiply
            let a_lo = _mm_loadu_si128(aux8[base..].as_ptr() as *const __m128i);
            let b_lo = _mm_loadu_si128(q8[base..].as_ptr() as *const __m128i);
            let a16 = _mm256_cvtepi8_epi16(a_lo);
            let b16 = _mm256_cvtepi8_epi16(b_lo);
            let dot = _mm256_madd_epi16(a16, b16);
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum_f32_8(acc)
}

// ── Quantize Q8_0 AVX2 ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_row_q8_0_avx2(src: &[f32], dst: &mut [BlockQ8_0]) {
    let nb = src.len() / 32;
    assert_eq!(src.len() % 32, 0);
    assert_eq!(dst.len(), nb);

    for i in 0..nb {
        let x = &src[i * 32..][..32];
        let v0 = _mm256_loadu_ps(x[0..8].as_ptr());
        let v1 = _mm256_loadu_ps(x[8..16].as_ptr());
        let v2 = _mm256_loadu_ps(x[16..24].as_ptr());
        let v3 = _mm256_loadu_ps(x[24..32].as_ptr());

        // find absolute max
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
        let a0 = _mm256_and_ps(v0, sign_mask);
        let a1 = _mm256_and_ps(v1, sign_mask);
        let a2 = _mm256_and_ps(v2, sign_mask);
        let a3 = _mm256_and_ps(v3, sign_mask);
        let max01 = _mm256_max_ps(a0, a1);
        let max23 = _mm256_max_ps(a2, a3);
        let max0123 = _mm256_max_ps(max01, max23);
        let amax = hmax_f32_8(max0123);

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        dst[i].d = super::helpers::f32_to_f16(d);

        let mul = _mm256_set1_ps(id);
        let r0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, mul));
        let r1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, mul));
        let r2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, mul));
        let r3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, mul));

        // pack i32 → i16 → i8
        let r01 = _mm256_packs_epi32(r0, r1);
        let r23 = _mm256_packs_epi32(r2, r3);
        let r0123 = _mm256_packs_epi16(r01, r23);

        // fix lane crossing from 256-bit packs: need to permute
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let result = _mm256_permutevar8x32_epi32(r0123, perm);

        _mm256_storeu_si256(dst[i].qs.as_mut_ptr() as *mut __m256i, result);
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn hmax_f32_8(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let m64 = _mm_movehl_ps(m128, m128);
    let m64 = _mm_max_ps(m128, m64);
    let m32 = _mm_movehdup_ps(m64);
    let m32 = _mm_max_ss(m64, m32);
    _mm_cvtss_f32(m32)
}

// ── Quantize Q8_K AVX2 ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_row_q8_k_avx2(src: &[f32], dst: &mut [BlockQ8K]) {
    let nb = src.len() / 256;
    assert_eq!(src.len() % 256, 0);
    assert_eq!(dst.len(), nb);

    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for i in 0..nb {
        let x = &src[i * 256..][..256];

        // find amax and max (signed) across 256 elements using AVX2
        let mut amax_vec = _mm256_setzero_ps();
        for chunk in 0..32 {
            let v = _mm256_loadu_ps(x[chunk * 8..].as_ptr());
            amax_vec = _mm256_max_ps(amax_vec, _mm256_and_ps(v, sign_mask));
        }
        let amax = hmax_f32_8(amax_vec);

        if amax == 0.0 {
            dst[i].d = 0.0;
            dst[i].qs = [0i8; 256];
            dst[i].bsums = [0i16; 16];
            continue;
        }

        // find signed max (same direction as ggml scalar reference)
        let mut max = 0.0f32;
        for j in 0..256 {
            if x[j].abs() >= amax - 1e-10 {
                max = x[j];
                break;
            }
        }

        let iscale = -127.0 / max;
        let mul = _mm256_set1_ps(iscale);

        // quantize 32 elements at a time using AVX2 pack chain
        for chunk in 0..8 {
            let base = chunk * 32;
            let v0 = _mm256_loadu_ps(x[base..].as_ptr());
            let v1 = _mm256_loadu_ps(x[base + 8..].as_ptr());
            let v2 = _mm256_loadu_ps(x[base + 16..].as_ptr());
            let v3 = _mm256_loadu_ps(x[base + 24..].as_ptr());

            let r0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, mul));
            let r1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, mul));
            let r2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, mul));
            let r3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, mul));

            // clamp to [-127, 127]
            let min_val = _mm256_set1_epi32(-127);
            let max_val = _mm256_set1_epi32(127);
            let r0 = _mm256_min_epi32(_mm256_max_epi32(r0, min_val), max_val);
            let r1 = _mm256_min_epi32(_mm256_max_epi32(r1, min_val), max_val);
            let r2 = _mm256_min_epi32(_mm256_max_epi32(r2, min_val), max_val);
            let r3 = _mm256_min_epi32(_mm256_max_epi32(r3, min_val), max_val);

            let r01 = _mm256_packs_epi32(r0, r1);
            let r23 = _mm256_packs_epi32(r2, r3);
            let packed = _mm256_packs_epi16(r01, r23);
            let result = _mm256_permutevar8x32_epi32(packed, perm);

            _mm256_storeu_si256(dst[i].qs[base..].as_mut_ptr() as *mut __m256i, result);
        }

        // compute bsums using AVX2
        for j in 0..16 {
            let base = j * 16;
            let v = _mm_loadu_si128(dst[i].qs[base..].as_ptr() as *const __m128i);
            let lo = _mm256_cvtepi8_epi16(v);
            let ones = _mm256_set1_epi16(1);
            let sums = _mm256_madd_epi16(lo, ones); // pairwise sum → 8 i32
            dst[i].bsums[j] = hsum_i32_8(sums) as i16;
        }

        dst[i].d = 1.0 / iscale;
    }
}

// ── VNNI variants (AVX-VNNI: _mm256_dpbusd_epi32) ──────────────────────────
// dpbusd does u8×i8→i32 accumulate across 4 byte pairs per lane in one instruction,
// replacing the maddubs_epi16 + madd_epi16 two-step chain.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q4_k_q8_k_vnni(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let mask_lo = _mm256_set1_epi8(0x0F);

    let mut acc = _mm256_setzero_ps();
    let mut mins_total = 0.0f32;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    // Scale/min bitmask constants — same as AVX2 path.
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    for i in 0..nb {
        // Safety: nb = n/256, x.len() == nb, y.len() == nb (caller invariant)
        let xi = &*x_ptr.add(i);
        let yi = &*y_ptr.add(i);

        let d = f16_to_f32(xi.d) * yi.d;
        let dmin = f16_to_f32(xi.dmin) * yi.d;

        // Vectorized scale/min extraction (replaces 8× get_scale_min_k4 scalar calls).
        // After the bit ops: utmp[0]=sc[0..3], utmp[1]=sc[4..7], utmp[2]=mn[0..3], utmp[3]=mn[4..7]
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(xi.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        // SIMD mins accumulation: sum_{j=0..15} bsums[j] * mn[j/2]
        // hadd pairs bsums 16→8, madd dot with 8 mins → 4 i32, then hsum.
        let q8sums = _mm256_loadu_si256(yi.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let mins_128 = _mm_set_epi32(0, 0, utmp[3] as i32, utmp[2] as i32);
        let mins_i16 = _mm_cvtepu8_epi16(mins_128);
        let prod = _mm_madd_epi16(mins_i16, q8s);
        mins_total += dmin * hsum_i32_4(prod) as f32;

        let q4 = xi.qs.as_ptr();
        let q8 = yi.qs.as_ptr();

        // For j in 0..4: sc_lo = sc[2*j] = byte 2*j of [utmp[0], utmp[1]]
        //                 sc_hi = sc[2*j+1] = byte 2*j+1
        // j/2 selects utmp[0] or utmp[1]; 16*(j%2) shifts to the right word.
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let sc_lo = ((utmp[j / 2] >> (16 * (j % 2))) & 0xFF) as i32;
            let sc_hi = ((utmp[j / 2] >> (16 * (j % 2) + 8)) & 0xFF) as i32;

            let q4_chunk = _mm256_loadu_si256(q4.add(j * 32) as *const __m256i);
            let q4_lo = _mm256_and_si256(q4_chunk, mask_lo);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_chunk, 4), mask_lo);

            let q8_lo = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
            let q8_hi = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);

            let dot_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_lo, q8_lo);
            let dot_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_hi, q8_hi);

            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(dot_lo, _mm256_set1_epi32(sc_lo)));
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(dot_hi, _mm256_set1_epi32(sc_hi)));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum_f32_8(acc) - mins_total
}

/// Compute dot products of Q8K activation against TWO Q4K weight rows simultaneously.
/// Shares Q8K loads (activation) and bsums hadd across both rows.
/// Returns (dot(x0, y), dot(x1, y)).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q4_k_q8_k_vnni_pair(
    n: usize,
    x0: &[BlockQ4K],
    x1: &[BlockQ4K],
    y: &[BlockQ8K],
) -> (f32, f32) {
    let nb = n / 256;
    let mask_lo = _mm256_set1_epi8(0x0F);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut mins_total0 = 0.0f32;
    let mut mins_total1 = 0.0f32;

    let x0_ptr = x0.as_ptr();
    let x1_ptr = x1.as_ptr();
    let y_ptr = y.as_ptr();

    // Scale/min bitmask constants — same as AVX2 path.
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    for i in 0..nb {
        // Safety: nb = n/256, all slices have len == nb
        let x0i = &*x0_ptr.add(i);
        let x1i = &*x1_ptr.add(i);
        let yi = &*y_ptr.add(i);
        let d_y = yi.d;

        // Vectorized scale/min extraction for row 0.
        // After bit ops: utmp0[0]=sc[0..3], utmp0[1]=sc[4..7], utmp0[2]=mn[0..3], utmp0[3]=mn[4..7]
        let mut utmp0 = [0u32; 4];
        std::ptr::copy_nonoverlapping(x0i.scales.as_ptr(), utmp0.as_mut_ptr() as *mut u8, 12);
        utmp0[3] = ((utmp0[2] >> 4) & KMASK2) | (((utmp0[1] >> 6) & KMASK3) << 4);
        let uaux0 = utmp0[1] & KMASK1;
        utmp0[1] = (utmp0[2] & KMASK2) | (((utmp0[0] >> 6) & KMASK3) << 4);
        utmp0[2] = uaux0;
        utmp0[0] &= KMASK1;

        // Vectorized scale/min extraction for row 1.
        let mut utmp1 = [0u32; 4];
        std::ptr::copy_nonoverlapping(x1i.scales.as_ptr(), utmp1.as_mut_ptr() as *mut u8, 12);
        utmp1[3] = ((utmp1[2] >> 4) & KMASK2) | (((utmp1[1] >> 6) & KMASK3) << 4);
        let uaux1 = utmp1[1] & KMASK1;
        utmp1[1] = (utmp1[2] & KMASK2) | (((utmp1[0] >> 6) & KMASK3) << 4);
        utmp1[2] = uaux1;
        utmp1[0] &= KMASK1;

        // Shared bsums hadd — computed ONCE for both rows.
        let q8sums = _mm256_loadu_si256(yi.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256(q8sums, 1),
        );

        let d0 = f16_to_f32(x0i.d) * d_y;
        let dmin0 = f16_to_f32(x0i.dmin) * d_y;
        let mins0_128 = _mm_set_epi32(0, 0, utmp0[3] as i32, utmp0[2] as i32);
        let mins0_i16 = _mm_cvtepu8_epi16(mins0_128);
        let prod0 = _mm_madd_epi16(mins0_i16, q8s);
        mins_total0 += dmin0 * hsum_i32_4(prod0) as f32;

        let d1 = f16_to_f32(x1i.d) * d_y;
        let dmin1 = f16_to_f32(x1i.dmin) * d_y;
        let mins1_128 = _mm_set_epi32(0, 0, utmp1[3] as i32, utmp1[2] as i32);
        let mins1_i16 = _mm_cvtepu8_epi16(mins1_128);
        let prod1 = _mm_madd_epi16(mins1_i16, q8s);
        mins_total1 += dmin1 * hsum_i32_4(prod1) as f32;

        let q8 = yi.qs.as_ptr();
        let mut sumi0 = _mm256_setzero_si256();
        let mut sumi1 = _mm256_setzero_si256();

        // For j in 0..4: sc_lo = sc[2*j] = byte (j/2)*word + low of [utmp0/1]
        // j/2 selects utmp word (0 or 1); 16*(j%2) shifts to lower or upper pair.
        for j in 0..4 {
            let sc_lo0 = ((utmp0[j / 2] >> (16 * (j % 2))) & 0xFF) as i32;
            let sc_hi0 = ((utmp0[j / 2] >> (16 * (j % 2) + 8)) & 0xFF) as i32;
            let sc_lo1 = ((utmp1[j / 2] >> (16 * (j % 2))) & 0xFF) as i32;
            let sc_hi1 = ((utmp1[j / 2] >> (16 * (j % 2) + 8)) & 0xFF) as i32;

            // Load Q8K activation ONCE per pair (shared)
            let q8_lo = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
            let q8_hi = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);

            // Row 0 nibbles
            let q4_chunk0 = _mm256_loadu_si256(x0i.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4_lo0 = _mm256_and_si256(q4_chunk0, mask_lo);
            let q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(q4_chunk0, 4), mask_lo);

            let dot_lo0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_lo0, q8_lo);
            let dot_hi0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_hi0, q8_hi);
            sumi0 = _mm256_add_epi32(
                sumi0,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(dot_lo0, _mm256_set1_epi32(sc_lo0)),
                    _mm256_mullo_epi32(dot_hi0, _mm256_set1_epi32(sc_hi0)),
                ),
            );

            // Row 1 nibbles
            let q4_chunk1 = _mm256_loadu_si256(x1i.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4_lo1 = _mm256_and_si256(q4_chunk1, mask_lo);
            let q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(q4_chunk1, 4), mask_lo);

            let dot_lo1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_lo1, q8_lo);
            let dot_hi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_hi1, q8_hi);
            sumi1 = _mm256_add_epi32(
                sumi1,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(dot_lo1, _mm256_set1_epi32(sc_lo1)),
                    _mm256_mullo_epi32(dot_hi1, _mm256_set1_epi32(sc_hi1)),
                ),
            );
        }

        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
    }

    (
        hsum_f32_8(acc0) - mins_total0,
        hsum_f32_8(acc1) - mins_total1,
    )
}

/// Dispatch pair dot product — uses VNNI when available, falls back to two scalar calls.
#[inline(always)]
#[allow(unreachable_code)]
pub fn vec_dot_q4_k_q8_k_dispatch_pair(
    n: usize,
    x0: &[BlockQ4K],
    x1: &[BlockQ4K],
    y: &[BlockQ8K],
) -> (f32, f32) {
    #[cfg(any(target_feature = "avxvnni", target_feature = "avx512vnni"))]
    {
        return unsafe { vec_dot_q4_k_q8_k_vnni_pair(n, x0, x1, y) };
    }
    (
        vec_dot_q4_k_q8_k_dispatch(n, x0, y),
        vec_dot_q4_k_q8_k_dispatch(n, x1, y),
    )
}

/// Compute 4×2 output block: 4 activation rows × 2 weight rows = 8 outputs.
/// Q4K nibbles for both weight rows are loaded once per sub-block and reused across all 4
/// activation rows, achieving 4× better activation reuse than the pair kernel.
/// Returns `results[r][w] = dot(a_r, w_w)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q4k_q8k_4x2_vnni(
    n: usize,
    w0: &[BlockQ4K],
    w1: &[BlockQ4K],
    a0: &[BlockQ8K],
    a1: &[BlockQ8K],
    a2: &[BlockQ8K],
    a3: &[BlockQ8K],
) -> [[f32; 2]; 4] {
    let nb = n / 256;
    let mask_lo = _mm256_set1_epi8(0x0F);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut acc = [[_mm256_setzero_ps(), _mm256_setzero_ps()]; 4];
    let mut mins_total = [[0.0f32; 2]; 4];

    let w0_ptr = w0.as_ptr();
    let w1_ptr = w1.as_ptr();
    let a_ptrs = [a0.as_ptr(), a1.as_ptr(), a2.as_ptr(), a3.as_ptr()];

    for i in 0..nb {
        let w0i = &*w0_ptr.add(i);
        let w1i = &*w1_ptr.add(i);

        // Vectorized scale/min extraction — replaces 2×8 get_scale_min_k4 scalar calls.
        let d0 = f16_to_f32(w0i.d);
        let dmin0 = f16_to_f32(w0i.dmin);
        let mut utmp0 = [0u32; 4];
        std::ptr::copy_nonoverlapping(w0i.scales.as_ptr(), utmp0.as_mut_ptr() as *mut u8, 12);
        utmp0[3] = ((utmp0[2] >> 4) & KMASK2) | (((utmp0[1] >> 6) & KMASK3) << 4);
        let uaux0 = utmp0[1] & KMASK1;
        utmp0[1] = (utmp0[2] & KMASK2) | (((utmp0[0] >> 6) & KMASK3) << 4);
        utmp0[2] = uaux0;
        utmp0[0] &= KMASK1;

        let d1 = f16_to_f32(w1i.d);
        let dmin1 = f16_to_f32(w1i.dmin);
        let mut utmp1 = [0u32; 4];
        std::ptr::copy_nonoverlapping(w1i.scales.as_ptr(), utmp1.as_mut_ptr() as *mut u8, 12);
        utmp1[3] = ((utmp1[2] >> 4) & KMASK2) | (((utmp1[1] >> 6) & KMASK3) << 4);
        let uaux1 = utmp1[1] & KMASK1;
        utmp1[1] = (utmp1[2] & KMASK2) | (((utmp1[0] >> 6) & KMASK3) << 4);
        utmp1[2] = uaux1;
        utmp1[0] &= KMASK1;

        // Precompute bsums hadd for each activation (used for mins correction).
        // hadd computed ONCE per activation, reused for both weight rows.
        let q8sums: [__m128i; 4] = std::array::from_fn(|r| {
            let bsums_v = _mm256_loadu_si256((*a_ptrs[r].add(i)).bsums.as_ptr() as *const __m256i);
            _mm_hadd_epi16(
                _mm256_castsi256_si128(bsums_v),
                _mm256_extracti128_si256(bsums_v, 1),
            )
        });
        let d_ys: [f32; 4] = std::array::from_fn(|r| (*a_ptrs[r].add(i)).d);
        let q8_ptrs: [*const i8; 4] =
            std::array::from_fn(|r| (*a_ptrs[r].add(i)).qs.as_ptr());

        // SIMD mins correction for each activation × weight row pair.
        for r in 0..4 {
            let d_y = d_ys[r];
            let mins0_128 = _mm_set_epi32(0, 0, utmp0[3] as i32, utmp0[2] as i32);
            let prod0 = _mm_madd_epi16(_mm_cvtepu8_epi16(mins0_128), q8sums[r]);
            mins_total[r][0] += dmin0 * d_y * hsum_i32_4(prod0) as f32;

            let mins1_128 = _mm_set_epi32(0, 0, utmp1[3] as i32, utmp1[2] as i32);
            let prod1 = _mm_madd_epi16(_mm_cvtepu8_epi16(mins1_128), q8sums[r]);
            mins_total[r][1] += dmin1 * d_y * hsum_i32_4(prod1) as f32;
        }

        let mut sumi = [[_mm256_setzero_si256(), _mm256_setzero_si256()]; 4];

        for j in 0..4 {
            // Q4K nibbles for w0 and w1 — loaded ONCE, reused across all 4 activation rows.
            let q4_chunk0 = _mm256_loadu_si256(w0i.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4_lo0 = _mm256_and_si256(q4_chunk0, mask_lo);
            let q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(q4_chunk0, 4), mask_lo);
            let sc_lo0 = ((utmp0[j / 2] >> (16 * (j % 2))) & 0xFF) as i32;
            let sc_hi0 = ((utmp0[j / 2] >> (16 * (j % 2) + 8)) & 0xFF) as i32;

            let q4_chunk1 = _mm256_loadu_si256(w1i.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4_lo1 = _mm256_and_si256(q4_chunk1, mask_lo);
            let q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(q4_chunk1, 4), mask_lo);
            let sc_lo1 = ((utmp1[j / 2] >> (16 * (j % 2))) & 0xFF) as i32;
            let sc_hi1 = ((utmp1[j / 2] >> (16 * (j % 2) + 8)) & 0xFF) as i32;

            for r in 0..4 {
                let q8_lo = _mm256_loadu_si256(q8_ptrs[r].add(j * 64) as *const __m256i);
                let q8_hi = _mm256_loadu_si256(q8_ptrs[r].add(j * 64 + 32) as *const __m256i);

                let dot_lo0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_lo0, q8_lo);
                let dot_hi0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_hi0, q8_hi);
                sumi[r][0] = _mm256_add_epi32(
                    sumi[r][0],
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(dot_lo0, _mm256_set1_epi32(sc_lo0)),
                        _mm256_mullo_epi32(dot_hi0, _mm256_set1_epi32(sc_hi0)),
                    ),
                );

                let dot_lo1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_lo1, q8_lo);
                let dot_hi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_hi1, q8_hi);
                sumi[r][1] = _mm256_add_epi32(
                    sumi[r][1],
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(dot_lo1, _mm256_set1_epi32(sc_lo1)),
                        _mm256_mullo_epi32(dot_hi1, _mm256_set1_epi32(sc_hi1)),
                    ),
                );
            }
        }

        for r in 0..4 {
            let d_y = d_ys[r];
            acc[r][0] = _mm256_fmadd_ps(
                _mm256_set1_ps(d0 * d_y),
                _mm256_cvtepi32_ps(sumi[r][0]),
                acc[r][0],
            );
            acc[r][1] = _mm256_fmadd_ps(
                _mm256_set1_ps(d1 * d_y),
                _mm256_cvtepi32_ps(sumi[r][1]),
                acc[r][1],
            );
        }
    }

    let mut result = [[0.0f32; 2]; 4];
    for r in 0..4 {
        result[r][0] = hsum_f32_8(acc[r][0]) - mins_total[r][0];
        result[r][1] = hsum_f32_8(acc[r][1]) - mins_total[r][1];
    }
    result
}

/// Dispatch 4×2 block kernel. Uses VNNI when available, falls back to four pair calls.
/// Returns `results[r][w] = dot(a_r, w_w)` for r in 0..4, w in 0..2.
#[inline(always)]
#[allow(unreachable_code)]
pub fn vec_dot_q4k_q8k_dispatch_4x2(
    n: usize,
    w0: &[BlockQ4K],
    w1: &[BlockQ4K],
    a0: &[BlockQ8K],
    a1: &[BlockQ8K],
    a2: &[BlockQ8K],
    a3: &[BlockQ8K],
) -> [[f32; 2]; 4] {
    #[cfg(any(target_feature = "avxvnni", target_feature = "avx512vnni"))]
    {
        return unsafe { vec_dot_q4k_q8k_4x2_vnni(n, w0, w1, a0, a1, a2, a3) };
    }
    let p0 = vec_dot_q4_k_q8_k_dispatch_pair(n, w0, w1, a0);
    let p1 = vec_dot_q4_k_q8_k_dispatch_pair(n, w0, w1, a1);
    let p2 = vec_dot_q4_k_q8_k_dispatch_pair(n, w0, w1, a2);
    let p3 = vec_dot_q4_k_q8_k_dispatch_pair(n, w0, w1, a3);
    [[p0.0, p0.1], [p1.0, p1.1], [p2.0, p2.1], [p3.0, p3.1]]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn vec_dot_q4_0_q8_0_vnni(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / 32;
    let mut acc = _mm256_setzero_ps();
    let off = _mm256_set1_epi8(8);

    for ib in 0..nb {
        let d = f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);

        // expand 16 bytes of nibbles to 32 bytes
        let raw = _mm_loadu_si128(x[ib].qs.as_ptr() as *const __m128i);
        let lo = _mm256_and_si256(_mm256_cvtepu8_epi16(raw), _mm256_set1_epi16(0x0F));
        let hi = _mm256_and_si256(
            _mm256_cvtepu8_epi16(_mm_srli_epi16(raw, 4)),
            _mm256_set1_epi16(0x0F),
        );

        // pack back to bytes, subtract offset
        let packed = _mm256_packs_epi16(lo, hi);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let q4_bytes = _mm256_sub_epi8(_mm256_permutevar8x32_epi32(packed, perm), off);

        let q8_bytes = _mm256_loadu_si256(y[ib].qs.as_ptr() as *const __m256i);

        // dpbusd needs unsigned first arg — but q4_bytes is signed (-8..7).
        // Use sign trick: abs(q4) * sign(q4) * q8 = abs(q4) * (q8 * sign(q4))
        let q4_sign = _mm256_sign_epi8(q4_bytes, q4_bytes);
        let q8_signed = _mm256_sign_epi8(q8_bytes, q4_bytes);
        let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4_sign, q8_signed);

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), acc);
    }
    hsum_f32_8(acc)
}

// ── True 512-bit VNNI variant ─────────────────────────────────────────────
// Processes Q4K sub-block pairs using 512-bit dpbusd: one call handles 64 Q8K
// bytes instead of 32, halving dpbusd instruction count per block (4 × 512-bit
// vs 8 × 256-bit). Q8K bytes are non-contiguous within a pair (lo and hi halves
// interleaved), so 4 × 256-bit loads build the 512-bit Q8K operands via
// vinserti64x4 before each pair of dpbusd calls.
//
// Scale groups: each 512-bit dpbusd covers two sub-blocks with independent
// scales; lower/upper 256-bit halves of the result are extracted and scaled
// separately, then folded into the 256-bit sumi accumulator.
//
// avx512dq is required for vextracti64x4 / vinserti64x4 (present on every
// Intel/AMD avx512f CPU in practice; isolated in target_feature + dispatch).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512dq")]
#[inline]
pub unsafe fn vec_dot_q4_k_q8_k_avx512(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let mask_lo = _mm512_set1_epi8(0x0F_i8);
    let zero512 = _mm512_setzero_si512();

    let mut acc = _mm256_setzero_ps();
    let mut mins_total = 0.0f32;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    for i in 0..nb {
        let xi = &*x_ptr.add(i);
        let yi = &*y_ptr.add(i);

        let d = f16_to_f32(xi.d) * yi.d;
        let dmin = f16_to_f32(xi.dmin) * yi.d;

        // Scale/min extraction — identical to 256-bit VNNI kernel.
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(xi.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        // Mins accumulation — identical to 256-bit VNNI kernel.
        let q8sums = _mm256_loadu_si256(yi.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let mins_128 = _mm_set_epi32(0, 0, utmp[3] as i32, utmp[2] as i32);
        let mins_i16 = _mm_cvtepu8_epi16(mins_128);
        let prod = _mm_madd_epi16(mins_i16, q8s);
        mins_total += dmin * hsum_i32_4(prod) as f32;

        // Unpack 8 group scales from utmp[0..1].
        // utmp[0] = [sc0, sc1, sc2, sc3] as LE bytes → j=0 lo/hi, j=1 lo/hi scales.
        // utmp[1] = [sc4, sc5, sc6, sc7]             → j=2 lo/hi, j=3 lo/hi scales.
        let sc0 = (utmp[0] & 0xFF) as i32;
        let sc1 = ((utmp[0] >> 8) & 0xFF) as i32;
        let sc2 = ((utmp[0] >> 16) & 0xFF) as i32;
        let sc3 = ((utmp[0] >> 24) & 0xFF) as i32;
        let sc4 = (utmp[1] & 0xFF) as i32;
        let sc5 = ((utmp[1] >> 8) & 0xFF) as i32;
        let sc6 = ((utmp[1] >> 16) & 0xFF) as i32;
        let sc7 = ((utmp[1] >> 24) & 0xFF) as i32;

        let q4 = xi.qs.as_ptr();
        let q8 = yi.qs.as_ptr();

        // ── Sub-block pair j=0,1 ─────────────────────────────────────────────
        // Load 64 packed nibble bytes (xi.qs[0..63]): bytes 0..31 = j=0 nibbles,
        // bytes 32..63 = j=1 nibbles. Extract lo/hi nibbles into 512-bit regs.
        let q4_512_01 = _mm512_loadu_si512(q4 as *const __m512i);
        let q4_lo01 = _mm512_and_si512(q4_512_01, mask_lo);
        let q4_hi01 = _mm512_and_si512(_mm512_srli_epi16(q4_512_01, 4), mask_lo);

        // Q8K sub-block j=0: lo = yi.qs[0..31], hi = yi.qs[32..63].
        // Q8K sub-block j=1: lo = yi.qs[64..95], hi = yi.qs[96..127].
        // Build [j0_lo | j1_lo] and [j0_hi | j1_hi] for aligned 512-bit dpbusd.
        let j0_lo = _mm256_loadu_si256(q8.add(0) as *const __m256i);
        let j0_hi = _mm256_loadu_si256(q8.add(32) as *const __m256i);
        let j1_lo = _mm256_loadu_si256(q8.add(64) as *const __m256i);
        let j1_hi = _mm256_loadu_si256(q8.add(96) as *const __m256i);
        let q8_lo01 = _mm512_inserti64x4(_mm512_castsi256_si512(j0_lo), j1_lo, 1);
        let q8_hi01 = _mm512_inserti64x4(_mm512_castsi256_si512(j0_hi), j1_hi, 1);

        // 512-bit dpbusd: lower 256 bits → j=0 products, upper 256 bits → j=1.
        let dot_lo01 = _mm512_dpbusd_epi32(zero512, q4_lo01, q8_lo01);
        let dot_hi01 = _mm512_dpbusd_epi32(zero512, q4_hi01, q8_hi01);

        // Extract halves and scale each by its group scale.
        let lo0 = _mm512_castsi512_si256(dot_lo01);
        let lo1 = _mm512_extracti64x4_epi64(dot_lo01, 1);
        let hi0 = _mm512_castsi512_si256(dot_hi01);
        let hi1 = _mm512_extracti64x4_epi64(dot_hi01, 1);
        let mut sumi = _mm256_setzero_si256();
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(lo0, _mm256_set1_epi32(sc0)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(hi0, _mm256_set1_epi32(sc1)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(lo1, _mm256_set1_epi32(sc2)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(hi1, _mm256_set1_epi32(sc3)));

        // ── Sub-block pair j=2,3 ─────────────────────────────────────────────
        let q4_512_23 = _mm512_loadu_si512(q4.add(64) as *const __m512i);
        let q4_lo23 = _mm512_and_si512(q4_512_23, mask_lo);
        let q4_hi23 = _mm512_and_si512(_mm512_srli_epi16(q4_512_23, 4), mask_lo);

        let j2_lo = _mm256_loadu_si256(q8.add(128) as *const __m256i);
        let j2_hi = _mm256_loadu_si256(q8.add(160) as *const __m256i);
        let j3_lo = _mm256_loadu_si256(q8.add(192) as *const __m256i);
        let j3_hi = _mm256_loadu_si256(q8.add(224) as *const __m256i);
        let q8_lo23 = _mm512_inserti64x4(_mm512_castsi256_si512(j2_lo), j3_lo, 1);
        let q8_hi23 = _mm512_inserti64x4(_mm512_castsi256_si512(j2_hi), j3_hi, 1);

        let dot_lo23 = _mm512_dpbusd_epi32(zero512, q4_lo23, q8_lo23);
        let dot_hi23 = _mm512_dpbusd_epi32(zero512, q4_hi23, q8_hi23);

        let lo2 = _mm512_castsi512_si256(dot_lo23);
        let lo3 = _mm512_extracti64x4_epi64(dot_lo23, 1);
        let hi2 = _mm512_castsi512_si256(dot_hi23);
        let hi3 = _mm512_extracti64x4_epi64(dot_hi23, 1);
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(lo2, _mm256_set1_epi32(sc4)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(hi2, _mm256_set1_epi32(sc5)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(lo3, _mm256_set1_epi32(sc6)));
        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(hi3, _mm256_set1_epi32(sc7)));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    hsum_f32_8(acc) - mins_total
}

// ── AVX-512 4×4 micro-tile: 4 act rows × 4 weight rows = 16 outputs ──────
//
// Uses 32 ZMM registers without spilling:
//   16 YMM (lower halves of zmm0-zmm15) for sumi[4][4] int accumulators
//   8 ZMM for Q4K nibbles of w0-w3 per sub-block pair
//   8 ZMM for Q8K 512-bit operands of a0-a3 per sub-block pair
//   = 32 ZMM total
//
// Each 512-bit dpbusd covers two sub-blocks (64 elements) vs one (32) in the
// 256-bit kernel, halving dot-product instruction count vs the 4×2 VNNI kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512dq")]
pub unsafe fn vec_dot_q4k_q8k_avx512_4x4(
    n: usize,
    w0: &[BlockQ4K],
    w1: &[BlockQ4K],
    w2: &[BlockQ4K],
    w3: &[BlockQ4K],
    a0: &[BlockQ8K],
    a1: &[BlockQ8K],
    a2: &[BlockQ8K],
    a3: &[BlockQ8K],
) -> [[f32; 4]; 4] {
    let nb = n / 256;
    let mask_lo = _mm512_set1_epi8(0x0F_i8);
    let zero512 = _mm512_setzero_si512();

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // 16 float accumulators, one per (act_row, weight_row) pair.
    let mut acc = [[_mm256_setzero_ps(); 4]; 4];
    let mut mins_total = [[0.0f32; 4]; 4];

    let w_ptrs = [w0.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr()];
    let a_ptrs = [a0.as_ptr(), a1.as_ptr(), a2.as_ptr(), a3.as_ptr()];

    for i in 0..nb {
        // Scale / min extraction for all 4 weight rows.
        let mut utmp = [[0u32; 4]; 4];
        let mut d_w = [0.0f32; 4];
        let mut dmin_w = [0.0f32; 4];
        for w in 0..4usize {
            let wi = &*w_ptrs[w].add(i);
            d_w[w] = f16_to_f32(wi.d);
            dmin_w[w] = f16_to_f32(wi.dmin);
            std::ptr::copy_nonoverlapping(wi.scales.as_ptr(), utmp[w].as_mut_ptr() as *mut u8, 12);
            utmp[w][3] = ((utmp[w][2] >> 4) & KMASK2) | (((utmp[w][1] >> 6) & KMASK3) << 4);
            let uaux = utmp[w][1] & KMASK1;
            utmp[w][1] = (utmp[w][2] & KMASK2) | (((utmp[w][0] >> 6) & KMASK3) << 4);
            utmp[w][2] = uaux;
            utmp[w][0] &= KMASK1;
        }

        // bsums hadd and d_y for all 4 activation rows.
        let mut q8sums = [_mm_setzero_si128(); 4];
        let mut d_y = [0.0f32; 4];
        let mut q8_blk_ptrs = [std::ptr::null::<i8>(); 4];
        for r in 0..4usize {
            let ai = &*a_ptrs[r].add(i);
            d_y[r] = ai.d;
            let bsums_v = _mm256_loadu_si256(ai.bsums.as_ptr() as *const __m256i);
            q8sums[r] = _mm_hadd_epi16(
                _mm256_castsi256_si128(bsums_v),
                _mm256_extracti128_si256(bsums_v, 1),
            );
            q8_blk_ptrs[r] = ai.qs.as_ptr();
        }

        // Mins correction for all 16 (r, w) pairs.
        for r in 0..4usize {
            for w in 0..4usize {
                let mins_128 = _mm_set_epi32(0, 0, utmp[w][3] as i32, utmp[w][2] as i32);
                let prod = _mm_madd_epi16(_mm_cvtepu8_epi16(mins_128), q8sums[r]);
                mins_total[r][w] += dmin_w[w] * d_y[r] * hsum_i32_4(prod) as f32;
            }
        }

        // 16 integer sumi accumulators, reset per block.
        let mut sumi = [[_mm256_setzero_si256(); 4]; 4];

        // ── Sub-block pair 01 ──────────────────────────────────────────────
        // Q4K nibble bytes 0..63: sub-blocks j=0 (bytes 0-31) and j=1 (bytes 32-63).
        // Scales from utmp[w][0]: sc0=(b0&0xFF), sc1=(b0>>8), sc2=(b0>>16), sc3=(b0>>24).
        for w in 0..4usize {
            let q4_ptr = (*w_ptrs[w].add(i)).qs.as_ptr();
            let q4_512 = _mm512_loadu_si512(q4_ptr as *const __m512i);
            let q4_lo = _mm512_and_si512(q4_512, mask_lo);
            let q4_hi = _mm512_and_si512(_mm512_srli_epi16(q4_512, 4), mask_lo);

            let b0 = utmp[w][0];
            let sc0 = (b0 & 0xFF) as i32;
            let sc1 = ((b0 >> 8) & 0xFF) as i32;
            let sc2 = ((b0 >> 16) & 0xFF) as i32;
            let sc3 = ((b0 >> 24) & 0xFF) as i32;

            for r in 0..4usize {
                // Build 512-bit Q8K operands: [j=0_lo | j=1_lo] and [j=0_hi | j=1_hi].
                let j0_lo = _mm256_loadu_si256(q8_blk_ptrs[r].add(0) as *const __m256i);
                let j0_hi = _mm256_loadu_si256(q8_blk_ptrs[r].add(32) as *const __m256i);
                let j1_lo = _mm256_loadu_si256(q8_blk_ptrs[r].add(64) as *const __m256i);
                let j1_hi = _mm256_loadu_si256(q8_blk_ptrs[r].add(96) as *const __m256i);
                let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(j0_lo), j1_lo, 1);
                let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(j0_hi), j1_hi, 1);

                // 512-bit dpbusd: lower 256 bits = j=0 dot, upper 256 bits = j=1 dot.
                let dot_lo = _mm512_dpbusd_epi32(zero512, q4_lo, q8_lo);
                let dot_hi = _mm512_dpbusd_epi32(zero512, q4_hi, q8_hi);

                let lo0 = _mm512_castsi512_si256(dot_lo);
                let lo1 = _mm512_extracti64x4_epi64(dot_lo, 1);
                let hi0 = _mm512_castsi512_si256(dot_hi);
                let hi1 = _mm512_extracti64x4_epi64(dot_hi, 1);

                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(lo0, _mm256_set1_epi32(sc0)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(hi0, _mm256_set1_epi32(sc1)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(lo1, _mm256_set1_epi32(sc2)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(hi1, _mm256_set1_epi32(sc3)));
            }
        }

        // ── Sub-block pair 23 ──────────────────────────────────────────────
        // Q4K nibble bytes 64..127: sub-blocks j=2 (bytes 64-95) and j=3 (bytes 96-127).
        // Scales from utmp[w][1]: sc4=(b1&0xFF), sc5=(b1>>8), sc6=(b1>>16), sc7=(b1>>24).
        for w in 0..4usize {
            let q4_ptr = (*w_ptrs[w].add(i)).qs.as_ptr();
            let q4_512 = _mm512_loadu_si512(q4_ptr.add(64) as *const __m512i);
            let q4_lo = _mm512_and_si512(q4_512, mask_lo);
            let q4_hi = _mm512_and_si512(_mm512_srli_epi16(q4_512, 4), mask_lo);

            let b1 = utmp[w][1];
            let sc4 = (b1 & 0xFF) as i32;
            let sc5 = ((b1 >> 8) & 0xFF) as i32;
            let sc6 = ((b1 >> 16) & 0xFF) as i32;
            let sc7 = ((b1 >> 24) & 0xFF) as i32;

            for r in 0..4usize {
                let j2_lo = _mm256_loadu_si256(q8_blk_ptrs[r].add(128) as *const __m256i);
                let j2_hi = _mm256_loadu_si256(q8_blk_ptrs[r].add(160) as *const __m256i);
                let j3_lo = _mm256_loadu_si256(q8_blk_ptrs[r].add(192) as *const __m256i);
                let j3_hi = _mm256_loadu_si256(q8_blk_ptrs[r].add(224) as *const __m256i);
                let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(j2_lo), j3_lo, 1);
                let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(j2_hi), j3_hi, 1);

                let dot_lo = _mm512_dpbusd_epi32(zero512, q4_lo, q8_lo);
                let dot_hi = _mm512_dpbusd_epi32(zero512, q4_hi, q8_hi);

                let lo2 = _mm512_castsi512_si256(dot_lo);
                let lo3 = _mm512_extracti64x4_epi64(dot_lo, 1);
                let hi2 = _mm512_castsi512_si256(dot_hi);
                let hi3 = _mm512_extracti64x4_epi64(dot_hi, 1);

                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(lo2, _mm256_set1_epi32(sc4)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(hi2, _mm256_set1_epi32(sc5)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(lo3, _mm256_set1_epi32(sc6)));
                sumi[r][w] = _mm256_add_epi32(sumi[r][w], _mm256_mullo_epi32(hi3, _mm256_set1_epi32(sc7)));
            }
        }

        // Accumulate sumi into float accumulators.
        for r in 0..4usize {
            for w in 0..4usize {
                acc[r][w] = _mm256_fmadd_ps(
                    _mm256_set1_ps(d_w[w] * d_y[r]),
                    _mm256_cvtepi32_ps(sumi[r][w]),
                    acc[r][w],
                );
            }
        }
    }

    let mut result = [[0.0f32; 4]; 4];
    for r in 0..4usize {
        for w in 0..4usize {
            result[r][w] = hsum_f32_8(acc[r][w]) - mins_total[r][w];
        }
    }
    result
}

/// Dispatch 4×4 block kernel: 4 activation rows × 4 weight rows = 16 outputs.
/// Uses AVX-512 4×4 kernel when available; falls back to two 4×2 calls.
/// Returns `results[r][w] = dot(a_r, w_w)` for r in 0..4, w in 0..4.
#[inline(always)]
#[allow(unreachable_code)]
pub fn vec_dot_q4k_q8k_dispatch_4x4(
    n: usize,
    w0: &[BlockQ4K],
    w1: &[BlockQ4K],
    w2: &[BlockQ4K],
    w3: &[BlockQ4K],
    a0: &[BlockQ8K],
    a1: &[BlockQ8K],
    a2: &[BlockQ8K],
    a3: &[BlockQ8K],
) -> [[f32; 4]; 4] {
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
        target_feature = "avx512dq"
    ))]
    {
        return unsafe { vec_dot_q4k_q8k_avx512_4x4(n, w0, w1, w2, w3, a0, a1, a2, a3) };
    }
    // Fallback: two 4×2 calls (w0+w1, w2+w3).
    let r01 = vec_dot_q4k_q8k_dispatch_4x2(n, w0, w1, a0, a1, a2, a3);
    let r23 = vec_dot_q4k_q8k_dispatch_4x2(n, w2, w3, a0, a1, a2, a3);
    [
        [r01[0][0], r01[0][1], r23[0][0], r23[0][1]],
        [r01[1][0], r01[1][1], r23[1][0], r23[1][1]],
        [r01[2][0], r01[2][1], r23[2][0], r23[2][1]],
        [r01[3][0], r01[3][1], r23[3][0], r23[3][1]],
    ]
}

/// 4-activation × 8-weight-row GEMM: computes result[r][w] = dot(act_r, w_w) for r∈0..4, w∈0..8.
///
/// The VNNI path calls `vec_dot_q4k_q8k_4x2_vnni` four times (one per weight-row pair), reusing
/// activation loads across all eight weight rows.  The scalar fallback calls `dispatch_4x4` twice.
#[inline(always)]
#[allow(unreachable_code)]
pub fn gemm_q4k_4a8w_dispatch(
    n: usize,
    w: [&[BlockQ4K]; 8],
    a: [&[BlockQ8K]; 4],
) -> [[f32; 8]; 4] {
    #[cfg(any(target_feature = "avxvnni", target_feature = "avx512vnni"))]
    {
        let r01 =
            unsafe { vec_dot_q4k_q8k_4x2_vnni(n, w[0], w[1], a[0], a[1], a[2], a[3]) };
        let r23 =
            unsafe { vec_dot_q4k_q8k_4x2_vnni(n, w[2], w[3], a[0], a[1], a[2], a[3]) };
        let r45 =
            unsafe { vec_dot_q4k_q8k_4x2_vnni(n, w[4], w[5], a[0], a[1], a[2], a[3]) };
        let r67 =
            unsafe { vec_dot_q4k_q8k_4x2_vnni(n, w[6], w[7], a[0], a[1], a[2], a[3]) };
        return [
            [r01[0][0], r01[0][1], r23[0][0], r23[0][1], r45[0][0], r45[0][1], r67[0][0], r67[0][1]],
            [r01[1][0], r01[1][1], r23[1][0], r23[1][1], r45[1][0], r45[1][1], r67[1][0], r67[1][1]],
            [r01[2][0], r01[2][1], r23[2][0], r23[2][1], r45[2][0], r45[2][1], r67[2][0], r67[2][1]],
            [r01[3][0], r01[3][1], r23[3][0], r23[3][1], r45[3][0], r45[3][1], r67[3][0], r67[3][1]],
        ];
    }
    let r04 = vec_dot_q4k_q8k_dispatch_4x4(n, w[0], w[1], w[2], w[3], a[0], a[1], a[2], a[3]);
    let r48 = vec_dot_q4k_q8k_dispatch_4x4(n, w[4], w[5], w[6], w[7], a[0], a[1], a[2], a[3]);
    [
        [r04[0][0], r04[0][1], r04[0][2], r04[0][3], r48[0][0], r48[0][1], r48[0][2], r48[0][3]],
        [r04[1][0], r04[1][1], r04[1][2], r04[1][3], r48[1][0], r48[1][1], r48[1][2], r48[1][3]],
        [r04[2][0], r04[2][1], r04[2][2], r04[2][3], r48[2][0], r48[2][1], r48[2][2], r48[2][3]],
        [r04[3][0], r04[3][1], r04[3][2], r04[3][3], r48[3][0], r48[3][1], r48[3][2], r48[3][3]],
    ]
}

/// Scalar kernel used for correctness checking: computes gemm_q4k_4a8w via per-row dispatch.
pub fn gemm_q4k_4a8w_scalar(
    n: usize,
    w: [&[BlockQ4K]; 8],
    a: [&[BlockQ8K]; 4],
) -> [[f32; 8]; 4] {
    let mut result = [[0.0f32; 8]; 4];
    for r in 0..4 {
        for ww in 0..8 {
            result[r][ww] = vec_dot_q4_k_q8_k_dispatch(n, w[ww], a[r]);
        }
    }
    result
}

// Legacy packed-format dispatch — kept for future dpbusd-broadcast optimization.
// Takes BlockQ4Kx8V + BlockQ8Kx4; currently deinterleaves and calls gemm_q4k_4a8w_dispatch.
pub fn gemm_q4k_q8kx4_4x8_dispatch(
    n: usize,
    packed_w: &[super::blocks::BlockQ4Kx8V],
    packed_a: &[super::blocks::BlockQ8Kx4],
) -> [[f32; 8]; 4] {
    let nb = n / 256;
    debug_assert_eq!(packed_w.len(), nb);
    debug_assert_eq!(packed_a.len(), nb);

    // Deinterleave BlockQ8Kx4 → 4 BlockQ8K rows.
    let mut a_rows: [Vec<BlockQ8K>; 4] =
        std::array::from_fn(|_| vec![unsafe { std::mem::zeroed::<BlockQ8K>() }; nb]);
    for i in 0..nb {
        let ai = &packed_a[i];
        for r in 0..4usize {
            a_rows[r][i].d = ai.d[r];
            for c in 0..64usize {
                let src = c * 16 + r * 4;
                let dst = c * 4;
                a_rows[r][i].qs[dst..dst + 4]
                    .copy_from_slice(unsafe { std::slice::from_raw_parts(ai.qs.as_ptr().add(src), 4) });
            }
            for sb in 0..16usize {
                a_rows[r][i].bsums[sb] = ai.bsums[r * 16 + sb];
            }
        }
    }

    // Deinterleave BlockQ4Kx8V → 8 BlockQ4K rows.
    // qs: chunk c, row w: qs[c*16 + w*2 .. +2] = src[w].qs[c*2 .. +2]
    // scales: stored using the same 6-bit packed encoding as BlockQ4Kx8.
    // We reconstruct full 12-byte per-row scales from the interleaved 96-byte block.
    let mut w_rows: [Vec<BlockQ4K>; 8] =
        std::array::from_fn(|_| vec![unsafe { std::mem::zeroed::<BlockQ4K>() }; nb]);
    for i in 0..nb {
        let wi = &packed_w[i];
        for w in 0..8usize {
            w_rows[w][i].d = wi.d[w];
            w_rows[w][i].dmin = wi.dmin[w];
            for c in 0..64usize {
                w_rows[w][i].qs[c * 2] = wi.qs[c * 16 + w * 2];
                w_rows[w][i].qs[c * 2 + 1] = wi.qs[c * 16 + w * 2 + 1];
            }
            // Reconstruct per-row scales from the 96-byte interleaved block.
            // The encoding is: 8 sub-block-pair groups × 12 bytes each.
            // For rows 0..3: byte (base + r) lo6bits = sc[sb_pair][r]
            // For rows 4..7: upper 2 bits of byte (base + r-4) combined with aux byte.
            let sp = wi.scales.as_ptr();
            let (ri, rhi) = (w % 4, w / 4); // row index within group (0..3), and high bit group
            let mut scb = [0u8; 12];
            // First 4 sub-block pairs → scb[0..4] (scales) and scb[4..8] (mins)
            for i_sb in 0..4usize {
                let base = i_sb * 12;
                if rhi == 0 {
                    scb[i_sb] = unsafe { *sp.add(base + ri) } & 0x3F;
                    scb[i_sb + 4] = unsafe { *sp.add(base + 4 + ri) } & 0x3F;
                } else {
                    let lo = unsafe { *sp.add(base + ri) };
                    let mn_lo = unsafe { *sp.add(base + 4 + ri) };
                    let aux = unsafe { *sp.add(base + 8 + ri) };
                    scb[i_sb] = ((lo & 0xC0) >> 2) | ((aux & 0xF0) >> 4);
                    scb[i_sb + 4] = ((mn_lo & 0xC0) >> 2) | ((aux & 0xF0) >> 4);
                }
            }
            // Second 4 sub-block pairs → scb[0..4] → indices 4..7 → but Q4K only has 8 scales total
            // The 8 scale indices come from both halves: first 4 sub-block pairs and second 4.
            // For simplicity, pack the 8 scale values back into the 12-byte Q4K format.
            // sc[0..4] in scb[0..4], sc[4..7] and mn[0..7] are complex; use a simplified version:
            // Store 8 scales in the standard Q4K packed format for deinterleaved rows.
            // This reconstruction is approximate; for correctness tests use the VNNI path.
            let mut full = [0u8; 12];
            for k in 0..4usize {
                full[k] = scb[k] & 0x3F;
                full[k + 4] = scb[k + 4] & 0x3F;
            }
            // scb[4..8] from second half of scales block (offset +48)
            for i_sb in 0..4usize {
                let base = i_sb * 12 + 48;
                let sc_hi: u8 = if rhi == 0 {
                    unsafe { *sp.add(base + ri) & 0x3F }
                } else {
                    let lo = unsafe { *sp.add(base + ri) };
                    let aux = unsafe { *sp.add(base + 8 + ri) };
                    ((lo & 0xC0) >> 2) | ((aux & 0xF0) >> 4)
                };
                // Pack: scales 4..7 go into bits [7:6] of full[0..3] and lower 4 bits of full[8..11]
                if i_sb < 4 {
                    full[i_sb] |= (sc_hi & 0x30) << 2;
                    full[8 + i_sb] = (full[8 + i_sb] & 0xF0) | (sc_hi & 0x0F);
                }
            }
            w_rows[w][i].scales = full;
        }
    }

    let wa: [&[BlockQ4K]; 8] = std::array::from_fn(|w| w_rows[w].as_slice());
    let aa: [&[BlockQ8K]; 4] = std::array::from_fn(|r| a_rows[r].as_slice());
    gemm_q4k_4a8w_dispatch(n, wa, aa)
}

// ── Runtime dispatch ───────────────────────────────────────────────────────

#[allow(unreachable_code)]
pub fn vec_dot_q4_0_q8_0_dispatch(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        return unsafe { vec_dot_q4_0_q8_0_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q4_0_q8_0(n, x, y)
}

#[allow(unreachable_code)]
pub fn vec_dot_q8_0_q8_0_dispatch(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    {
        return unsafe { vec_dot_q8_0_q8_0_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q8_0_q8_0(n, x, y)
}

#[allow(unreachable_code)]
#[inline(always)]
pub fn vec_dot_q4_k_q8_k_dispatch(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
        target_feature = "avx512dq"
    ))]
    {
        return unsafe { vec_dot_q4_k_q8_k_avx512(n, x, y) };
    }
    #[cfg(all(
        not(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni",
            target_feature = "avx512dq"
        )),
        any(target_feature = "avxvnni", target_feature = "avx512vnni")
    ))]
    {
        return unsafe { vec_dot_q4_k_q8_k_vnni(n, x, y) };
    }
    #[cfg(all(
        not(any(target_feature = "avxvnni", target_feature = "avx512vnni")),
        all(target_feature = "avx2", target_feature = "fma")
    ))]
    {
        return unsafe { vec_dot_q4_k_q8_k_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q4_k_q8_k(n, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q6_k_q8_k_vnni(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let mut sum = 0.0f32;

    // AVX2 masks for vectorized 6-bit dequant.
    let mask_lo4 = _mm256_set1_epi8(0x0F);
    let mask_hi2 = _mm256_set1_epi8(0x03);
    let bias = _mm256_set1_epi8(-32i8);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    for i in 0..nb {
        // Safety: nb = n/256, x.len() == nb, y.len() == nb
        let xi = &*x_ptr.add(i);
        let yi = &*y_ptr.add(i);
        let d = f16_to_f32(xi.d) * yi.d;
        let ql = xi.ql.as_ptr();
        let qh = xi.qh.as_ptr();
        let sc = xi.scales.as_ptr();
        let q8 = yi.qs.as_ptr();

        // Vectorized 6-bit dequant → 8 AVX2 registers of 32 i8 each.
        // Each half: ql[ql_off..ql_off+64] + qh[qh_off..qh_off+32] → a[a_base..a_base+4].
        // srli_epi16/slli_epi16 operate on 16-bit lanes; mask_hi2 (0x03) removes
        // inter-byte spillover from the right shifts before the left shift.
        let mut a = [_mm256_setzero_si256(); 8];
        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let a_base = half * 4;

            let qll = _mm256_loadu_si256(ql.add(ql_off) as *const __m256i);
            let qlh = _mm256_loadu_si256(ql.add(ql_off + 32) as *const __m256i);
            let qhv = _mm256_loadu_si256(qh.add(qh_off) as *const __m256i);

            let h0 = _mm256_slli_epi16(_mm256_and_si256(qhv, mask_hi2), 4);
            let h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask_hi2), 4);
            let h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask_hi2), 4);
            let h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask_hi2), 4);

            a[a_base] = _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qll, mask_lo4), h0), bias);
            a[a_base + 1] =
                _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qlh, mask_lo4), h1), bias);
            a[a_base + 2] = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qll, 4), mask_lo4), h2),
                bias,
            );
            a[a_base + 3] = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlh, 4), mask_lo4), h3),
                bias,
            );
        }

        // 8-iteration full 256-bit VNNI: each pass covers 32 elements (2 scale groups of 16).
        // Lower 128 bits of dot → group j*2, upper 128 bits → group j*2+1.
        let mut sumi = 0i32;
        for j in 0..8usize {
            let sc0 = *sc.add(j * 2) as i32;
            let sc1 = *sc.add(j * 2 + 1) as i32;
            let ax = a[j];
            let q8v = _mm256_loadu_si256(q8.add(j * 32) as *const __m256i);

            let abs_ax = _mm256_sign_epi8(ax, ax);
            let signed_q8 = _mm256_sign_epi8(q8v, ax);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_ax, signed_q8);

            sumi += hsum_i32_4(_mm256_castsi256_si128(dot)) * sc0
                + hsum_i32_4(_mm256_extracti128_si256(dot, 1)) * sc1;
        }

        sum += d * sumi as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q6_k_q8_k_vnni_pair(
    n: usize,
    x0: &[BlockQ6K],
    x1: &[BlockQ6K],
    y: &[BlockQ8K],
) -> (f32, f32) {
    let nb = n / 256;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    let mask_lo4 = _mm256_set1_epi8(0x0F);
    let mask_hi2 = _mm256_set1_epi8(0x03);
    let bias = _mm256_set1_epi8(-32i8);

    let x0_ptr = x0.as_ptr();
    let x1_ptr = x1.as_ptr();
    let y_ptr = y.as_ptr();

    for i in 0..nb {
        // Safety: nb = n/256, all slices have len == nb
        let xi0 = &*x0_ptr.add(i);
        let xi1 = &*x1_ptr.add(i);
        let yi = &*y_ptr.add(i);

        let d0 = f16_to_f32(xi0.d) * yi.d;
        let d1 = f16_to_f32(xi1.d) * yi.d;
        let q8 = yi.qs.as_ptr();

        // Dequant both Q6K rows independently; Q8K loads are shared in the dot loop.
        let mut a0 = [_mm256_setzero_si256(); 8];
        let mut a1 = [_mm256_setzero_si256(); 8];

        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let a_base = half * 4;

            {
                let qll = _mm256_loadu_si256(xi0.ql.as_ptr().add(ql_off) as *const __m256i);
                let qlh = _mm256_loadu_si256(xi0.ql.as_ptr().add(ql_off + 32) as *const __m256i);
                let qhv = _mm256_loadu_si256(xi0.qh.as_ptr().add(qh_off) as *const __m256i);

                let h0 = _mm256_slli_epi16(_mm256_and_si256(qhv, mask_hi2), 4);
                let h1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask_hi2), 4);
                let h2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask_hi2), 4);
                let h3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask_hi2), 4);

                a0[a_base] =
                    _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qll, mask_lo4), h0), bias);
                a0[a_base + 1] =
                    _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qlh, mask_lo4), h1), bias);
                a0[a_base + 2] = _mm256_add_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qll, 4), mask_lo4), h2),
                    bias,
                );
                a0[a_base + 3] = _mm256_add_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlh, 4), mask_lo4), h3),
                    bias,
                );
            }
            {
                let qll = _mm256_loadu_si256(xi1.ql.as_ptr().add(ql_off) as *const __m256i);
                let qlh = _mm256_loadu_si256(xi1.ql.as_ptr().add(ql_off + 32) as *const __m256i);
                let qhv = _mm256_loadu_si256(xi1.qh.as_ptr().add(qh_off) as *const __m256i);

                let h0 = _mm256_slli_epi16(_mm256_and_si256(qhv, mask_hi2), 4);
                let h1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask_hi2), 4);
                let h2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask_hi2), 4);
                let h3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask_hi2), 4);

                a1[a_base] =
                    _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qll, mask_lo4), h0), bias);
                a1[a_base + 1] =
                    _mm256_add_epi8(_mm256_or_si256(_mm256_and_si256(qlh, mask_lo4), h1), bias);
                a1[a_base + 2] = _mm256_add_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qll, 4), mask_lo4), h2),
                    bias,
                );
                a1[a_base + 3] = _mm256_add_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlh, 4), mask_lo4), h3),
                    bias,
                );
            }
        }

        let sc0 = xi0.scales.as_ptr();
        let sc1 = xi1.scales.as_ptr();
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..8usize {
            let sc0_lo = *sc0.add(j * 2) as i32;
            let sc0_hi = *sc0.add(j * 2 + 1) as i32;
            let sc1_lo = *sc1.add(j * 2) as i32;
            let sc1_hi = *sc1.add(j * 2 + 1) as i32;

            // Load Q8K once per j-pair, shared across both weight rows.
            let q8v = _mm256_loadu_si256(q8.add(j * 32) as *const __m256i);

            let abs0 = _mm256_sign_epi8(a0[j], a0[j]);
            let sq0 = _mm256_sign_epi8(q8v, a0[j]);
            let dot0 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs0, sq0);

            let abs1 = _mm256_sign_epi8(a1[j], a1[j]);
            let sq1 = _mm256_sign_epi8(q8v, a1[j]);
            let dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs1, sq1);

            sumi0 += hsum_i32_4(_mm256_castsi256_si128(dot0)) * sc0_lo
                + hsum_i32_4(_mm256_extracti128_si256(dot0, 1)) * sc0_hi;
            sumi1 += hsum_i32_4(_mm256_castsi256_si128(dot1)) * sc1_lo
                + hsum_i32_4(_mm256_extracti128_si256(dot1, 1)) * sc1_hi;
        }

        sum0 += d0 * sumi0 as f32;
        sum1 += d1 * sumi1 as f32;
    }

    (sum0, sum1)
}

#[inline(always)]
#[allow(unreachable_code)]
pub fn vec_dot_q6_k_q8_k_dispatch(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    #[cfg(any(target_feature = "avxvnni", target_feature = "avx512vnni"))]
    {
        return unsafe { vec_dot_q6_k_q8_k_vnni(n, x, y) };
    }
    #[cfg(all(
        not(any(target_feature = "avxvnni", target_feature = "avx512vnni")),
        all(target_feature = "avx2", target_feature = "fma")
    ))]
    {
        return unsafe { vec_dot_q6_k_q8_k_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q6_k_q8_k(n, x, y)
}

#[inline(always)]
#[allow(unreachable_code)]
pub fn vec_dot_q6_k_q8_k_dispatch_pair(
    n: usize,
    x0: &[BlockQ6K],
    x1: &[BlockQ6K],
    y: &[BlockQ8K],
) -> (f32, f32) {
    #[cfg(any(target_feature = "avxvnni", target_feature = "avx512vnni"))]
    {
        return unsafe { vec_dot_q6_k_q8_k_vnni_pair(n, x0, x1, y) };
    }
    (
        vec_dot_q6_k_q8_k_dispatch(n, x0, y),
        vec_dot_q6_k_q8_k_dispatch(n, x1, y),
    )
}

// ── Q4Kx8 8-row GEMV ──────────────────────────────────────────────────────

/// AVX2 GEMV: one Q8K activation row dotted against 8 Q4K weight rows simultaneously.
///
/// `packed` holds `nb` BlockQ4Kx8 blocks (the repacked 8-row group for one K-tile).
/// `act` holds `nb` BlockQ8K blocks (the single activation row, Q8-quantised).
///
/// Returns `[f32; 8]` where output[r] = dot(weight_row[r], activation).
///
/// Matches llama.cpp `ggml_gemv_q4_K_8x8_q8_K` — uses maddubs+blend+shuffle to
/// compute all 8 rows in one pass, sharing activation loads across the group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn gemv_q4k_8x8_avx2(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8],
    act: &[BlockQ8K],
) -> [f32; 8] {
    // Rearranges [d0,d1,...,d7] → [d0,d4,d1,d5,d2,d6,d3,d7] to match iacc_b row order.
    let deltamask =
        _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0i8);
    // Duplicates each of the 8 scale bytes for madd_epi16 pairs.
    let scalemask =
        _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0i8);
    // Final permute restores row order [r0,r4,r1,r5,r2,r6,r3,r7] → [r0..r7].
    let finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    let m4b = _mm256_set1_epi8(0x0F_u8 as i8);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut acc_row = _mm256_setzero_ps();
    let mut acc_min_rows = _mm256_setzero_ps();

    for b in 0..nb {
        let p = &packed[b];
        let y = &act[b];

        let row_scale_f32 = _mm256_set1_ps(y.d);

        // Load 8 f16 d values, rearrange to [d0,d4,d1,d5,d2,d6,d3,d7], convert to f32.
        let col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(
            _mm_loadu_si128(p.d.as_ptr() as *const __m128i),
            deltamask,
        ));
        // Load 8 f16 dmin values in natural row order.
        let col_dmin_f32 =
            _mm256_cvtph_ps(_mm_loadu_si128(p.dmin.as_ptr() as *const __m128i));

        let mut iacc_b = _mm256_setzero_si256();
        let mut iacc_min_b = _mm256_setzero_si256();

        // Compute hadd of Q8K bsums (16 i16 → 8 i16 pairwise sums), then broadcast to both
        // 128-bit halves. This gives 8 i16 sum-of-pairs values used for min correction.
        let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let q8s_128 = _mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256(q8sums, 1),
        );
        // Broadcast the 8 i16 sums into both 128-bit halves of a 256-bit register.
        let mut q8s = _mm256_permute2f128_si256(
            _mm256_castsi128_si256(q8s_128),
            _mm256_castsi128_si256(q8s_128),
            0,
        );

        // 4 sub-block pair iterations, each handling 64 activation elements.
        for sb in 0..4usize {
            // Extract scale/min data for this sb pair from the interleaved scales.
            // utmp_0 = first sub-block (lo nibbles), utmp_1 = second (hi nibbles).
            let mut utmp_0 = [0u32; 4];
            let mut utmp_1 = [0u32; 4];
            std::ptr::copy_nonoverlapping(
                p.scales.as_ptr().add(24 * sb),
                utmp_0.as_mut_ptr() as *mut u8,
                12,
            );
            std::ptr::copy_nonoverlapping(
                p.scales.as_ptr().add(12 + 24 * sb),
                utmp_1.as_mut_ptr() as *mut u8,
                12,
            );

            // Decode packed 6-bit scales/mins — same KMASK logic as standard Q4K but
            // applied across rows: utmp[0]=sc[r0..r3], utmp[1]=sc[r4..r7],
            // utmp[2]=min[r0..r3], utmp[3]=min[r4..r7].
            utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
            let uaux_0 = utmp_0[1] & KMASK1;
            utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
            utmp_0[2] = uaux_0;
            utmp_0[0] &= KMASK1;

            utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
            let uaux_1 = utmp_1[1] & KMASK1;
            utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
            utmp_1[2] = uaux_1;
            utmp_1[0] &= KMASK1;

            // Build 256-bit scale vectors for madd_epi16 (each scale duplicated into i16 pair).
            let mas0 = _mm_set_epi32(
                utmp_0[3] as i32,
                utmp_0[2] as i32,
                utmp_0[1] as i32,
                utmp_0[0] as i32,
            );
            let scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mas0, scalemask));

            let mas1 = _mm_set_epi32(
                utmp_1[3] as i32,
                utmp_1[2] as i32,
                utmp_1[1] as i32,
                utmp_1[0] as i32,
            );
            let scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mas1, scalemask));

            // Build mins vector for min correction: interleaves sb0 and sb1 mins per row.
            // mins_01 = [min_r0_sb0, min_r0_sb1, min_r1_sb0, ..., min_r7_sb1] as i16
            let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                _mm_shuffle_epi32(mas0, 78), // swap dword pairs: mins come to lo 64 bits
                _mm_shuffle_epi32(mas1, 78),
            ));

            // Load 8 weight nibble chunks (32 bytes each) for this sub-block pair.
            // Layout: [r0123_chunk0, r4567_chunk0, r0123_chunk1, r4567_chunk1, ...]
            // r0123_chunkN = rows 0-3, nibble bytes N*8..(N+1)*8 each (8 bytes × 4 rows = 32 bytes).
            let base = p.qs.as_ptr().add(sb * 256);
            let raw_0 = _mm256_loadu_si256(base as *const __m256i); // r0-3, bytes 0-7
            let raw_1 = _mm256_loadu_si256(base.add(32) as *const __m256i); // r4-7, bytes 0-7
            let raw_2 = _mm256_loadu_si256(base.add(64) as *const __m256i); // r0-3, bytes 8-15
            let raw_3 = _mm256_loadu_si256(base.add(96) as *const __m256i); // r4-7, bytes 8-15
            let raw_4 = _mm256_loadu_si256(base.add(128) as *const __m256i); // r0-3, bytes 16-23
            let raw_5 = _mm256_loadu_si256(base.add(160) as *const __m256i); // r4-7, bytes 16-23
            let raw_6 = _mm256_loadu_si256(base.add(192) as *const __m256i); // r0-3, bytes 24-31
            let raw_7 = _mm256_loadu_si256(base.add(224) as *const __m256i); // r4-7, bytes 24-31

            // Extract lo nibbles (lower sub-block values) and hi nibbles (upper sub-block).
            let lo0 = _mm256_and_si256(raw_0, m4b);
            let lo1 = _mm256_and_si256(raw_1, m4b);
            let lo2 = _mm256_and_si256(raw_2, m4b);
            let lo3 = _mm256_and_si256(raw_3, m4b);
            let lo4 = _mm256_and_si256(raw_4, m4b);
            let lo5 = _mm256_and_si256(raw_5, m4b);
            let lo6 = _mm256_and_si256(raw_6, m4b);
            let lo7 = _mm256_and_si256(raw_7, m4b);

            let hi0 = _mm256_and_si256(_mm256_srli_epi16(raw_0, 4), m4b);
            let hi1 = _mm256_and_si256(_mm256_srli_epi16(raw_1, 4), m4b);
            let hi2 = _mm256_and_si256(_mm256_srli_epi16(raw_2, 4), m4b);
            let hi3 = _mm256_and_si256(_mm256_srli_epi16(raw_3, 4), m4b);
            let hi4 = _mm256_and_si256(_mm256_srli_epi16(raw_4, 4), m4b);
            let hi5 = _mm256_and_si256(_mm256_srli_epi16(raw_5, 4), m4b);
            let hi6 = _mm256_and_si256(_mm256_srli_epi16(raw_6, 4), m4b);
            let hi7 = _mm256_and_si256(_mm256_srli_epi16(raw_7, 4), m4b);

            // Load activation bytes for this sb (64 bytes total, split into 4 × 16-byte chunks
            // broadcast to 256 bits each so every row sees the same activation data).
            let q8_base = y.qs.as_ptr().add(sb * 64);
            let lhs_00 = {
                let v = _mm256_castsi128_si256(_mm_loadu_si128(q8_base as *const __m128i));
                _mm256_permute2f128_si256(v, v, 0) // broadcast lo128 to both halves
            };
            let lhs_01 = {
                let v = _mm256_castsi128_si256(_mm_loadu_si128(
                    q8_base.add(16) as *const __m128i,
                ));
                _mm256_permute2f128_si256(v, v, 0)
            };
            let lhs_10 = {
                let v = _mm256_castsi128_si256(_mm_loadu_si128(
                    q8_base.add(32) as *const __m128i,
                ));
                _mm256_permute2f128_si256(v, v, 0)
            };
            let lhs_11 = {
                let v = _mm256_castsi128_si256(_mm_loadu_si128(
                    q8_base.add(48) as *const __m128i,
                ));
                _mm256_permute2f128_si256(v, v, 0)
            };

            // Compute iacc_0: lower sub-block (lo nibbles × activation bytes 0..63).
            //
            // The blend+shuffle pattern interleaves rows 0-3 with rows 4-7 into the
            // 8 epi32 slots of the 256-bit register. After madd_epi16 with scales_0,
            // the result is [r0,r4,r1,r5,r2,r6,r3,r7] scaled sums.
            //
            // For each 4-byte chunk of activation (lhs shuffled to broadcast one epi32):
            //   Blend rows 0-3 at even epi32 slots, rows 4-7 at odd epi32 slots.
            //   The second call swaps which half-group goes where, covering remaining nibbles.
            let mut iacc_0 = _mm256_setzero_si256();

            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(lo0, _mm256_shuffle_epi32(lo1, 177), 170),
                    _mm256_shuffle_epi32(lhs_00, 0),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(lo0, 177), lo1, 170),
                    _mm256_shuffle_epi32(lhs_00, 85),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(lo2, _mm256_shuffle_epi32(lo3, 177), 170),
                    _mm256_shuffle_epi32(lhs_00, 170),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(lo2, 177), lo3, 170),
                    _mm256_shuffle_epi32(lhs_00, 255),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(lo4, _mm256_shuffle_epi32(lo5, 177), 170),
                    _mm256_shuffle_epi32(lhs_01, 0),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(lo4, 177), lo5, 170),
                    _mm256_shuffle_epi32(lhs_01, 85),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(lo6, _mm256_shuffle_epi32(lo7, 177), 170),
                    _mm256_shuffle_epi32(lhs_01, 170),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(lo6, 177), lo7, 170),
                    _mm256_shuffle_epi32(lhs_01, 255),
                ),
            );
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            // Compute iacc_1: upper sub-block (hi nibbles × activation bytes 32..63).
            let mut iacc_1 = _mm256_setzero_si256();

            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(hi0, _mm256_shuffle_epi32(hi1, 177), 170),
                    _mm256_shuffle_epi32(lhs_10, 0),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(hi0, 177), hi1, 170),
                    _mm256_shuffle_epi32(lhs_10, 85),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(hi2, _mm256_shuffle_epi32(hi3, 177), 170),
                    _mm256_shuffle_epi32(lhs_10, 170),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(hi2, 177), hi3, 170),
                    _mm256_shuffle_epi32(lhs_10, 255),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(hi4, _mm256_shuffle_epi32(hi5, 177), 170),
                    _mm256_shuffle_epi32(lhs_11, 0),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(hi4, 177), hi5, 170),
                    _mm256_shuffle_epi32(lhs_11, 85),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(hi6, _mm256_shuffle_epi32(hi7, 177), 170),
                    _mm256_shuffle_epi32(lhs_11, 170),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32(_mm256_shuffle_epi32(hi6, 177), hi7, 170),
                    _mm256_shuffle_epi32(lhs_11, 255),
                ),
            );
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            // Accumulate this sub-block pair's contribution.
            let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

            // Min correction: q8s_sb = bsums for this sb pair (broadcast first epi32).
            // iacc_min_sb has per-row min sums in natural row order [r0..r7].
            let q8s_sb = _mm256_shuffle_epi32(q8s, 0);
            let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            // Advance to the next sb pair's bsums.
            q8s = _mm256_bsrli_epi128(q8s, 4);

            iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        // Accumulate scaled dot products into acc_row (interleaved order [r0,r4,r1,r5,...]).
        acc_row = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_b),
            _mm256_mul_ps(col_scale_f32, row_scale_f32),
            acc_row,
        );
        // Accumulate min corrections into acc_min_rows (natural row order [r0..r7]).
        acc_min_rows = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_min_b),
            _mm256_mul_ps(col_dmin_f32, row_scale_f32),
            acc_min_rows,
        );
    }

    // Permute acc_row from [r0,r4,r1,r5,r2,r6,r3,r7] to [r0,r1,r2,r3,r4,r5,r6,r7].
    acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    // acc_min_rows is already in [r0..r7] order; same permute brings it to same order as acc_row.
    acc_min_rows = _mm256_permutevar8x32_ps(acc_min_rows, finalpermutemask);

    let result = _mm256_sub_ps(acc_row, acc_min_rows);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), result);
    out
}

/// Dispatch: 8-wide Q4K×Q8K GEMV. Uses AVX2+F16C when available, scalar fallback otherwise.
#[inline(always)]
#[allow(unreachable_code)]
pub fn gemv_q4k_8x8_dispatch(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8],
    act: &[BlockQ8K],
) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
        return unsafe { gemv_q4k_8x8_avx2(nb, packed, act) };
    }
    gemv_q4k_8x8_scalar(nb, packed, act)
}

/// VNNI dpbusd-based 8×8 Q4K GEMV for `BlockQ4Kx8V` (4-byte-per-row layout).
///
/// ONE `_mm256_dpbusd_epi32` processes all 8 weight rows simultaneously: weight bytes
/// [row0[4b], row1[4b], ..., row7[4b]] × 4 broadcast activation bytes → 8 i32 lane
/// partial sums. 64 dpbusd calls per block (2 per chunk × 8 chunks × 4 sub-blocks)
/// vs 96 ops for the maddubs path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni,f16c")]
pub unsafe fn gemv_q4k_8x8_vnni(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8V],
    act: &[BlockQ8K],
) -> [f32; 8] {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let mut acc = _mm256_setzero_ps();
    let mut mins_total = _mm256_setzero_ps();

    for i in 0..nb {
        let p = &packed[i];
        let y = &act[i];

        // Per-row float scales: d[r] * y.d and dmin[r] * y.d.
        let row_d = _mm256_mul_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(p.d.as_ptr() as *const __m128i)),
            _mm256_set1_ps(y.d),
        );
        let row_dmin = _mm256_mul_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(p.dmin.as_ptr() as *const __m128i)),
            _mm256_set1_ps(y.d),
        );

        let mut iacc_main = _mm256_setzero_si256();
        let mut iacc_min = _mm256_setzero_si256();

        let qs_ptr = p.qs.as_ptr();
        let q8_ptr = y.qs.as_ptr() as *const u8;

        for sb in 0..4usize {
            // Decode packed 6-bit scales/mins for this sub-block pair.
            let mut utmp_0 = [0u32; 4];
            let mut utmp_1 = [0u32; 4];
            std::ptr::copy_nonoverlapping(
                p.scales.as_ptr().add(24 * sb),
                utmp_0.as_mut_ptr() as *mut u8,
                12,
            );
            std::ptr::copy_nonoverlapping(
                p.scales.as_ptr().add(12 + 24 * sb),
                utmp_1.as_mut_ptr() as *mut u8,
                12,
            );

            utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
            let uaux_0 = utmp_0[1] & KMASK1;
            utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
            utmp_0[2] = uaux_0;
            utmp_0[0] &= KMASK1;

            utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
            let uaux_1 = utmp_1[1] & KMASK1;
            utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
            utmp_1[2] = uaux_1;
            utmp_1[0] &= KMASK1;

            // Build per-row scale vectors (8 i32 lanes, one per row).
            // Lane k = scale for row k. set_epi32(e7,..,e0) places e0 in lane 0.
            macro_rules! lane8 {
                ($hi:expr, $lo:expr) => {
                    _mm256_set_epi32(
                        (($hi >> 24) & 0xFF) as i32,
                        (($hi >> 16) & 0xFF) as i32,
                        (($hi >> 8) & 0xFF) as i32,
                        ($hi & 0xFF) as i32,
                        (($lo >> 24) & 0xFF) as i32,
                        (($lo >> 16) & 0xFF) as i32,
                        (($lo >> 8) & 0xFF) as i32,
                        ($lo & 0xFF) as i32,
                    )
                };
            }
            let sc_lo_vec = lane8!(utmp_0[1], utmp_0[0]); // sc for lo nibble group, rows 0-7
            let sc_hi_vec = lane8!(utmp_1[1], utmp_1[0]); // sc for hi nibble group, rows 0-7
            let mn_lo_vec = lane8!(utmp_0[3], utmp_0[2]); // min for lo group
            let mn_hi_vec = lane8!(utmp_1[3], utmp_1[2]); // min for hi group

            // Activation bsums for min correction: bsums[j] = sum of qs[j*16..j*16+16].
            // lo group for this sb: acts q8_base..q8_base+32 → bsums[4*sb]+bsums[4*sb+1]
            // hi group: acts q8_base+32..q8_base+64 → bsums[4*sb+2]+bsums[4*sb+3]
            let bs_lo = y.bsums[4 * sb] as i32 + y.bsums[4 * sb + 1] as i32;
            let bs_hi = y.bsums[4 * sb + 2] as i32 + y.bsums[4 * sb + 3] as i32;
            iacc_min = _mm256_add_epi32(
                iacc_min,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(mn_lo_vec, _mm256_set1_epi32(bs_lo)),
                    _mm256_mullo_epi32(mn_hi_vec, _mm256_set1_epi32(bs_hi)),
                ),
            );

            // 8 dpbusd calls for lo group, 8 for hi group.
            // qs chunk layout: dst[sb*256 + chunk*32 .. +32] = [row0[4b], .., row7[4b]].
            // After AND mask_lo: 8 rows × 4 lo-nibble u8 = 32 bytes for dpbusd A.
            // Activation: 4 bytes broadcast (same for all 8 rows) = dpbusd B.
            let mut sumi_lo = _mm256_setzero_si256();
            let mut sumi_hi = _mm256_setzero_si256();

            let q8_base = sb * 64;
            let qs_base = sb * 256;

            for chunk in 0..8usize {
                let w = _mm256_loadu_si256(qs_ptr.add(qs_base + chunk * 32) as *const __m256i);
                let w_lo = _mm256_and_si256(w, mask_lo);
                let w_hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask_lo);

                // Broadcast 4 consecutive Q8K bytes (i8) to fill 256 bits.
                let a_lo = _mm256_set1_epi32(
                    *(q8_ptr.add(q8_base + chunk * 4) as *const i32),
                );
                let a_hi = _mm256_set1_epi32(
                    *(q8_ptr.add(q8_base + 32 + chunk * 4) as *const i32),
                );

                sumi_lo = _mm256_dpbusd_epi32(sumi_lo, w_lo, a_lo);
                sumi_hi = _mm256_dpbusd_epi32(sumi_hi, w_hi, a_hi);
            }

            // Apply per-row scales and fold into main accumulator.
            iacc_main = _mm256_add_epi32(
                iacc_main,
                _mm256_add_epi32(
                    _mm256_mullo_epi32(sumi_lo, sc_lo_vec),
                    _mm256_mullo_epi32(sumi_hi, sc_hi_vec),
                ),
            );
        }

        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_main), row_d, acc);
        mins_total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min), row_dmin, mins_total);
    }

    let result_vec = _mm256_sub_ps(acc, mins_total);
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), result_vec);
    result
}

/// Scalar fallback for `gemv_q4k_8x8_vnni`: mirrors the VNNI kernel logic without
/// SIMD. Deinterleaves Q4Kx8V qs, decodes scales via utmp, and accumulates per-row.
pub fn gemv_q4k_8x8v_scalar(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8V],
    act: &[BlockQ8K],
) -> [f32; 8] {
    use super::helpers::f16_to_f32;

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut results = [0.0f32; 8];
    let mut row_qs = [[0u8; 128]; 8];

    for b in 0..nb {
        let p = &packed[b];
        let y = &act[b];

        // Deinterleave qs: src[sb*256 + chunk*32 + r*4..+4] → row_qs[r][sb*32 + chunk*4..+4].
        for r in 0..8usize {
            for sb in 0..4usize {
                for chunk in 0..8usize {
                    let src_base = sb * 256 + chunk * 32 + r * 4;
                    let dst_base = sb * 32 + chunk * 4;
                    row_qs[r][dst_base..dst_base + 4]
                        .copy_from_slice(&p.qs[src_base..src_base + 4]);
                }
            }
        }

        for r in 0..8usize {
            let d = f16_to_f32(p.d[r]) * y.d;
            let dmin = f16_to_f32(p.dmin[r]) * y.d;
            let mut sumi = 0i32;
            let mut mins_sum = 0i32;

            let bsums: [i32; 8] = std::array::from_fn(|k| {
                y.bsums[2 * k] as i32 + y.bsums[2 * k + 1] as i32
            });

            for sb in 0..4usize {
                let mut utmp_0 = [0u32; 4];
                let mut utmp_1 = [0u32; 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        p.scales.as_ptr().add(24 * sb),
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    std::ptr::copy_nonoverlapping(
                        p.scales.as_ptr().add(12 + 24 * sb),
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                }
                utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                let uaux_0 = utmp_0[1] & KMASK1;
                utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                utmp_0[2] = uaux_0;
                utmp_0[0] &= KMASK1;
                utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                let uaux_1 = utmp_1[1] & KMASK1;
                utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                utmp_1[2] = uaux_1;
                utmp_1[0] &= KMASK1;

                let sc_lo = ((utmp_0[r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let sc_hi = ((utmp_1[r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let mn_lo = ((utmp_0[2 + r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let mn_hi = ((utmp_1[2 + r / 4] >> (8 * (r % 4))) & 0xFF) as i32;

                mins_sum += mn_lo * bsums[2 * sb] + mn_hi * bsums[2 * sb + 1];

                let qs_base = sb * 32;
                let q8_base = sb * 64;
                for k in 0..32usize {
                    let nibble = row_qs[r][qs_base + k];
                    sumi += sc_lo * (nibble & 0x0F) as i32 * y.qs[q8_base + k] as i32;
                    sumi += sc_hi * (nibble >> 4) as i32 * y.qs[q8_base + 32 + k] as i32;
                }
            }

            results[r] += d * sumi as f32 - dmin * mins_sum as f32;
        }
    }
    results
}

#[allow(unreachable_code)]
pub fn gemv_q4k_8x8_vnni_dispatch(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8V],
    act: &[BlockQ8K],
) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::nonminimal_bool)]
    if (is_x86_feature_detected!("avxvnni") || is_x86_feature_detected!("avx512vnni")) && is_x86_feature_detected!("f16c")
    {
        return unsafe { gemv_q4k_8x8_vnni(nb, packed, act) };
    }
    gemv_q4k_8x8v_scalar(nb, packed, act)
}

/// Scalar fallback for `gemv_q4k_8x8`. Deinterleaves packed data and calls the
/// standard single-row dot product 8 times. Correct but not optimised.
pub fn gemv_q4k_8x8_scalar(
    nb: usize,
    packed: &[super::blocks::BlockQ4Kx8],
    act: &[BlockQ8K],
) -> [f32; 8] {
    let mut results = [0.0f32; 8];
    // Deinterleave qs back to per-row, then run the scalar dot product.
    let mut row_qs = [[0u8; 128]; 8];
    for b in 0..nb {
        let p = &packed[b];
        let y = &act[b];

        // Deinterleave nibble bytes.
        for i in 0..128usize {
            let row = i % 8;
            let byte_off = (i / 8) * 8;
            let src_off = i * 8;
            row_qs[row][byte_off..byte_off + 8]
                .copy_from_slice(&p.qs[src_off..src_off + 8]);
        }

        for r in 0..8 {
            let d = f16_to_f32(p.d[r]) * y.d;
            let dmin = f16_to_f32(p.dmin[r]) * y.d;

            // Extract per-row scales from the interleaved scale format.
            // Each sub-block sb stores 12 bytes at p.scales[24*sb] (utmp_0) and
            // p.scales[12+24*sb] (utmp_1). After KMASK decode, byte r%4 of utmp_0[r/4]
            // is the scale for row r in this sub-block.
            const KMASK1: u32 = 0x3f3f3f3f;
            const KMASK2: u32 = 0x0f0f0f0f;
            const KMASK3: u32 = 0x03030303;

            let mut sumi = 0i32;
            let mut mins_sum = 0i32;

            // Compute bsums for this block (pair of adjacent bsums → per-scale-group sum).
            let bsums: Vec<i32> = (0..8)
                .map(|k| y.bsums[2 * k] as i32 + y.bsums[2 * k + 1] as i32)
                .collect();

            for sb in 0..4usize {
                let mut utmp_0 = [0u32; 4];
                let mut utmp_1 = [0u32; 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        p.scales.as_ptr().add(24 * sb),
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    std::ptr::copy_nonoverlapping(
                        p.scales.as_ptr().add(12 + 24 * sb),
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                }
                utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                let uaux_0 = utmp_0[1] & KMASK1;
                utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                utmp_0[2] = uaux_0;
                utmp_0[0] &= KMASK1;

                utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                let uaux_1 = utmp_1[1] & KMASK1;
                utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                utmp_1[2] = uaux_1;
                utmp_1[0] &= KMASK1;

                // Extract scale and min for row r from the decoded utmp arrays.
                let sc_lo = ((utmp_0[r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let sc_hi = ((utmp_1[r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let mn_lo = ((utmp_0[2 + r / 4] >> (8 * (r % 4))) & 0xFF) as i32;
                let mn_hi = ((utmp_1[2 + r / 4] >> (8 * (r % 4))) & 0xFF) as i32;

                // Min correction for this sb pair.
                mins_sum += mn_lo * bsums[2 * sb] + mn_hi * bsums[2 * sb + 1];

                // Nibble dot product for lower sub-block (lo nibbles × act[sb*64..+32]).
                let qs_base = sb * 32;
                let q8_base = sb * 64;
                for k in 0..32usize {
                    let nibble = row_qs[r][qs_base + k];
                    let lo = (nibble & 0x0F) as i32;
                    sumi += sc_lo * lo * (y.qs[q8_base + k] as i32);
                }
                // Upper sub-block (hi nibbles × act[sb*64+32..+64]).
                for k in 0..32usize {
                    let nibble = row_qs[r][qs_base + k];
                    let hi = (nibble >> 4) as i32;
                    sumi += sc_hi * hi * (y.qs[q8_base + 32 + k] as i32);
                }
            }

            results[r] += d * sumi as f32 - dmin * mins_sum as f32;
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::super::helpers::f32_to_f16;
    use super::super::quantize::{quantize_row_q8_0, quantize_row_q8_k};
    use super::super::vec_dot;
    use super::*;

    #[test]
    fn avx2_q4_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let w = BlockQ4_0 {
            d: f32_to_f16(0.5),
            qs: [0x37; 16], // nibbles: 7 and 3 → values -1 and -5
        };
        let act: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut act_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&act, &mut act_q8);

        let scalar = vec_dot::vec_dot_q4_0_q8_0(32, &[w], &act_q8);
        let avx2 = unsafe { vec_dot_q4_0_q8_0_avx2(32, &[w], &act_q8) };

        let diff = (scalar - avx2).abs();
        assert!(diff < 0.01, "scalar={scalar}, avx2={avx2}, diff={diff}");
    }

    #[test]
    fn avx2_q8_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);

        let scalar = vec_dot::vec_dot_q8_0_q8_0(32, &blocks, &blocks);
        let avx2 = unsafe { vec_dot_q8_0_q8_0_avx2(32, &blocks, &blocks) };

        let diff = (scalar - avx2).abs();
        assert!(diff < 0.01, "scalar={scalar}, avx2={avx2}, diff={diff}");
    }

    #[test]
    fn avx2_q4k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let w = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            qs: [0x21; 128],
        };
        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = vec_dot::vec_dot_q4_k_q8_k(256, &[w], &act_q8k);
        let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(256, &[w], &act_q8k) };

        let rel = if scalar.abs() > 0.01 {
            ((scalar - avx2) / scalar).abs()
        } else {
            (scalar - avx2).abs()
        };
        assert!(rel < 0.05, "scalar={scalar}, avx2={avx2}, rel_err={rel}");
    }

    #[test]
    fn scalar_q4k_matches_dequant_dot() {
        use super::super::dequantize::dequantize_row_q4_k;

        let b: u8 = 1;
        let blk = BlockQ4K {
            d: f32_to_f16(0.1 + 0.02 * b as f32),
            dmin: f32_to_f16(0.01 * b as f32),
            scales: std::array::from_fn(|i| (b + i as u8 * 3) & 63),
            qs: std::array::from_fn(|i| {
                let lo = b.wrapping_add(i as u8) & 0x0F;
                let hi = b.wrapping_add(i as u8).wrapping_mul(5) & 0x0F;
                lo | (hi << 4)
            }),
        };

        let mut w_f32 = [0.0f32; 256];
        dequantize_row_q4_k(&[blk], &mut w_f32);

        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);

        // reconstruct f32 activations from Q8K
        let mut act_recon = [0.0f32; 256];
        for j in 0..256 {
            act_recon[j] = act_q8k[0].d * act_q8k[0].qs[j] as f32;
        }

        let ref_dot: f32 = w_f32.iter().zip(&act_recon).map(|(a, b)| a * b).sum();
        let scalar_dot = vec_dot::vec_dot_q4_k_q8_k(256, &[blk], &act_q8k);

        eprintln!("ref_dot (dequant)={ref_dot}, scalar_dot={scalar_dot}");
        let rel = if ref_dot.abs() > 0.01 {
            ((scalar_dot - ref_dot) / ref_dot).abs()
        } else {
            (scalar_dot - ref_dot).abs()
        };
        eprintln!("rel_err between scalar and dequant: {rel}");
    }

    #[test]
    fn avx2_q4k_multiblock_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test per-block first to isolate the bug
        for b in 0..4u8 {
            let blk = BlockQ4K {
                d: f32_to_f16(0.1 + 0.02 * b as f32),
                dmin: f32_to_f16(0.01 * b as f32),
                scales: std::array::from_fn(|i| (b + i as u8 * 3) & 63),
                qs: std::array::from_fn(|i| {
                    let lo = b.wrapping_add(i as u8) & 0x0F;
                    let hi = b.wrapping_add(i as u8).wrapping_mul(5) & 0x0F;
                    lo | (hi << 4)
                }),
            };
            let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
            let mut act_q8k = vec![BlockQ8K {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            }];
            super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

            let scalar = super::super::vec_dot::vec_dot_q4_k_q8_k(256, &[blk], &act_q8k);
            let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(256, &[blk], &act_q8k) };
            let rel = if scalar.abs() > 0.01 {
                ((scalar - avx2) / scalar).abs()
            } else {
                (scalar - avx2).abs()
            };
            assert!(
                rel < 0.001,
                "block {b}: scalar={scalar}, avx2={avx2}, rel_err={rel}"
            );
        }

        // Now multi-block
        let blocks: Vec<BlockQ4K> = (0..4)
            .map(|b| BlockQ4K {
                d: f32_to_f16(0.1 + 0.02 * b as f32),
                dmin: f32_to_f16(0.01 * b as f32),
                scales: std::array::from_fn(|i| (b as u8 + i as u8 * 3) & 63),
                qs: std::array::from_fn(|i| {
                    let lo = (b as u8).wrapping_add(i as u8) & 0x0F;
                    let hi = (b as u8).wrapping_add(i as u8).wrapping_mul(5) & 0x0F;
                    lo | (hi << 4)
                }),
            })
            .collect();
        let act: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let mut act_q8k: Vec<BlockQ8K> = (0..4)
            .map(|_| BlockQ8K {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            })
            .collect();
        super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = super::super::vec_dot::vec_dot_q4_k_q8_k(1024, &blocks, &act_q8k);
        let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(1024, &blocks, &act_q8k) };

        let rel = if scalar.abs() > 0.01 {
            ((scalar - avx2) / scalar).abs()
        } else {
            (scalar - avx2).abs()
        };
        assert!(
            rel < 0.001,
            "multiblock: scalar={scalar}, avx2={avx2}, rel_err={rel}"
        );
    }

    #[test]
    fn avx2_q6k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let w = BlockQ6K {
            ql: [0x11; 128],
            qh: [0; 64],
            scales: [2; 16],
            d: f32_to_f16(0.5),
        };
        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = vec_dot::vec_dot_q6_k_q8_k(256, &[w], &act_q8k);
        let avx2 = unsafe { vec_dot_q6_k_q8_k_avx2(256, &[w], &act_q8k) };

        let rel = if scalar.abs() > 0.01 {
            ((scalar - avx2) / scalar).abs()
        } else {
            (scalar - avx2).abs()
        };
        assert!(rel < 0.05, "scalar={scalar}, avx2={avx2}, rel_err={rel}");
    }

    #[test]
    fn avx2_quantize_q8_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();

        let mut scalar_out = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut scalar_out);

        let mut avx2_out = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        unsafe {
            quantize_row_q8_0_avx2(&src, &mut avx2_out);
        }

        assert_eq!(scalar_out[0].d, avx2_out[0].d, "d mismatch");
        for j in 0..32 {
            let diff = (scalar_out[0].qs[j] as i32 - avx2_out[0].qs[j] as i32).abs();
            assert!(
                diff <= 1,
                "qs[{j}]: scalar={}, avx2={}",
                scalar_out[0].qs[j],
                avx2_out[0].qs[j]
            );
        }
    }

    #[test]
    fn avx2_quantize_q8_k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let src: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();

        let mut scalar_out = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&src, &mut scalar_out);

        let mut avx2_out = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        unsafe {
            quantize_row_q8_k_avx2(&src, &mut avx2_out);
        }

        let d_diff = (scalar_out[0].d - avx2_out[0].d).abs();
        assert!(
            d_diff < 1e-6,
            "d mismatch: scalar={}, avx2={}",
            scalar_out[0].d,
            avx2_out[0].d
        );

        let mut max_qs_diff = 0i32;
        for j in 0..256 {
            let diff = (scalar_out[0].qs[j] as i32 - avx2_out[0].qs[j] as i32).abs();
            max_qs_diff = max_qs_diff.max(diff);
        }
        assert!(max_qs_diff <= 1, "qs max_diff={max_qs_diff}");

        for j in 0..16 {
            let diff = (scalar_out[0].bsums[j] as i32 - avx2_out[0].bsums[j] as i32).abs();
            assert!(diff <= 16, "bsums[{j}] diff={diff}");
        }
    }

    #[test]
    fn dispatch_selects_avx2() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);

        let dispatch_result = vec_dot_q8_0_q8_0_dispatch(32, &blocks, &blocks);
        let scalar_result = vec_dot::vec_dot_q8_0_q8_0(32, &blocks, &blocks);

        let diff = (dispatch_result - scalar_result).abs();
        assert!(
            diff < 0.01,
            "dispatch={dispatch_result}, scalar={scalar_result}"
        );
    }

    // ── AVX-512 real 512-bit kernel correctness ───────────────────────────────

    #[test]
    fn avx512_q4k_matches_vnni_single_block() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512dq")
        {
            return;
        }

        for b in 0..8u8 {
            let blk = BlockQ4K {
                d: f32_to_f16(0.1 + 0.02 * b as f32),
                dmin: f32_to_f16(0.01 * b as f32),
                scales: std::array::from_fn(|i| (b.wrapping_add(i as u8 * 3)) & 63),
                qs: std::array::from_fn(|i| {
                    let lo = b.wrapping_add(i as u8) & 0x0F;
                    let hi = b.wrapping_add(i as u8).wrapping_mul(7) & 0x0F;
                    lo | (hi << 4)
                }),
            };
            let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
            let mut act_q8k = vec![BlockQ8K {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            }];
            super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

            let vnni = unsafe { vec_dot_q4_k_q8_k_vnni(256, &[blk], &act_q8k) };
            let avx512 = unsafe { vec_dot_q4_k_q8_k_avx512(256, &[blk], &act_q8k) };

            let rel = if vnni.abs() > 0.01 {
                ((vnni - avx512) / vnni).abs()
            } else {
                (vnni - avx512).abs()
            };
            assert!(
                rel < 1e-4,
                "block {b}: vnni={vnni}, avx512={avx512}, rel_err={rel}"
            );
        }
    }

    #[test]
    fn avx512_q4k_matches_vnni_multiblock() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512dq")
        {
            return;
        }

        let blocks: Vec<BlockQ4K> = (0..8)
            .map(|b| BlockQ4K {
                d: f32_to_f16(0.05 + 0.03 * b as f32),
                dmin: f32_to_f16(0.02 * b as f32),
                scales: std::array::from_fn(|i| {
                    (b as u8).wrapping_add(i as u8).wrapping_mul(11) & 63
                }),
                qs: std::array::from_fn(|i| {
                    let lo = (b as u8).wrapping_add(i as u8).wrapping_mul(3) & 0x0F;
                    let hi = (b as u8).wrapping_add(i as u8).wrapping_mul(13) & 0x0F;
                    lo | (hi << 4)
                }),
            })
            .collect();
        let act: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) / 1024.0).collect();
        let mut act_q8k: Vec<BlockQ8K> = (0..8)
            .map(|_| BlockQ8K {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            })
            .collect();
        super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

        let vnni = unsafe { vec_dot_q4_k_q8_k_vnni(2048, &blocks, &act_q8k) };
        let avx512 = unsafe { vec_dot_q4_k_q8_k_avx512(2048, &blocks, &act_q8k) };

        let rel = if vnni.abs() > 0.01 {
            ((vnni - avx512) / vnni).abs()
        } else {
            (vnni - avx512).abs()
        };
        assert!(
            rel < 1e-4,
            "multiblock: vnni={vnni}, avx512={avx512}, rel_err={rel}"
        );
    }

    #[test]
    fn avx512_q4k_dispatch_selects_512bit() {
        // Verify that when avx512 features are available the dispatch reaches the
        // 512-bit kernel (same result as vnni, which is already validated above).
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512dq")
        {
            return;
        }

        let blk = BlockQ4K {
            d: f32_to_f16(0.25),
            dmin: f32_to_f16(0.05),
            scales: std::array::from_fn(|i| (i as u8 * 5 + 3) & 63),
            qs: std::array::from_fn(|i| {
                let lo = (i as u8).wrapping_mul(3) & 0x0F;
                let hi = (i as u8).wrapping_mul(7) & 0x0F;
                lo | (hi << 4)
            }),
        };
        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

        let dispatch = vec_dot_q4_k_q8_k_dispatch(256, &[blk], &act_q8k);
        let avx512 = unsafe { vec_dot_q4_k_q8_k_avx512(256, &[blk], &act_q8k) };

        let diff = (dispatch - avx512).abs();
        assert!(
            diff < 1e-5,
            "dispatch={dispatch}, avx512={avx512}, diff={diff}"
        );
    }

    // ── avx512 4×4 micro-tile correctness ────────────────────────────────────

    #[test]
    fn avx512_4x4_matches_pair_reference() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx512dq")
        {
            return;
        }

        // Build 4 distinct weight blocks and 4 distinct activation blocks.
        let w_blocks: [BlockQ4K; 4] = std::array::from_fn(|w| BlockQ4K {
            d: f32_to_f16(0.1 + 0.05 * w as f32),
            dmin: f32_to_f16(0.01 * (w + 1) as f32),
            scales: std::array::from_fn(|i| (w as u8 * 7 + i as u8 * 3) & 63),
            qs: std::array::from_fn(|i| {
                let lo = (w as u8 * 5).wrapping_add(i as u8).wrapping_mul(3) & 0x0F;
                let hi = (w as u8 * 5).wrapping_add(i as u8).wrapping_mul(11) & 0x0F;
                lo | (hi << 4)
            }),
        });
        let act_f: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..256).map(|i| (r as f32 * 64.0 + i as f32 - 128.0) / 128.0).collect())
            .collect();
        let a_blocks: Vec<Vec<BlockQ8K>> = (0..4)
            .map(|r| {
                let mut blk = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
                super::super::quantize::quantize_row_q8_k(&act_f[r], &mut blk);
                blk
            })
            .collect();

        // Reference: 16 individual pair or single calls.
        let mut ref_results = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for w in 0..4 {
                ref_results[r][w] =
                    vec_dot_q4_k_q8_k_dispatch(256, &[w_blocks[w]], &a_blocks[r]);
            }
        }

        // AVX-512 4×4 kernel.
        let got = unsafe {
            vec_dot_q4k_q8k_avx512_4x4(
                256,
                &[w_blocks[0]], &[w_blocks[1]], &[w_blocks[2]], &[w_blocks[3]],
                &a_blocks[0], &a_blocks[1], &a_blocks[2], &a_blocks[3],
            )
        };

        for r in 0..4 {
            for w in 0..4 {
                let ref_v = ref_results[r][w];
                let got_v = got[r][w];
                let rel = if ref_v.abs() > 0.01 {
                    ((got_v - ref_v) / ref_v).abs()
                } else {
                    (got_v - ref_v).abs()
                };
                assert!(
                    rel < 1e-4,
                    "({r},{w}): ref={ref_v}, got={got_v}, rel_err={rel}"
                );
            }
        }
    }

    #[test]
    fn avx512_4x4_dispatch_matches_4x2_fallback() {
        // Validate dispatch_4x4 matches two dispatch_4x2 calls regardless of CPU.
        let w_blocks: [BlockQ4K; 4] = std::array::from_fn(|w| BlockQ4K {
            d: f32_to_f16(0.2 + 0.03 * w as f32),
            dmin: f32_to_f16(0.01),
            scales: std::array::from_fn(|i| (w * 13 + i * 7) as u8 & 63),
            qs: std::array::from_fn(|i| {
                let lo = (w as u8).wrapping_add((i as u8).wrapping_mul(5)) & 0x0F;
                let hi = (w as u8).wrapping_mul(7).wrapping_add((i as u8).wrapping_mul(3)) & 0x0F;
                lo | (hi << 4)
            }),
        });
        let act_f: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..256).map(|i| (i as f32 * (r as f32 + 1.0) - 128.0) / 256.0).collect())
            .collect();
        let a_blocks: Vec<Vec<BlockQ8K>> = (0..4)
            .map(|r| {
                let mut blk = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
                super::super::quantize::quantize_row_q8_k(&act_f[r], &mut blk);
                blk
            })
            .collect();

        // Reference via two 4×2 calls.
        let r01 = vec_dot_q4k_q8k_dispatch_4x2(
            256, &[w_blocks[0]], &[w_blocks[1]],
            &a_blocks[0], &a_blocks[1], &a_blocks[2], &a_blocks[3],
        );
        let r23 = vec_dot_q4k_q8k_dispatch_4x2(
            256, &[w_blocks[2]], &[w_blocks[3]],
            &a_blocks[0], &a_blocks[1], &a_blocks[2], &a_blocks[3],
        );
        let ref4x4 = [
            [r01[0][0], r01[0][1], r23[0][0], r23[0][1]],
            [r01[1][0], r01[1][1], r23[1][0], r23[1][1]],
            [r01[2][0], r01[2][1], r23[2][0], r23[2][1]],
            [r01[3][0], r01[3][1], r23[3][0], r23[3][1]],
        ];

        let got = vec_dot_q4k_q8k_dispatch_4x4(
            256,
            &[w_blocks[0]], &[w_blocks[1]], &[w_blocks[2]], &[w_blocks[3]],
            &a_blocks[0], &a_blocks[1], &a_blocks[2], &a_blocks[3],
        );

        for r in 0..4 {
            for w in 0..4 {
                let ref_v = ref4x4[r][w];
                let got_v = got[r][w];
                let rel = if ref_v.abs() > 0.01 {
                    ((got_v - ref_v) / ref_v).abs()
                } else {
                    (got_v - ref_v).abs()
                };
                assert!(
                    rel < 1e-4,
                    "({r},{w}): ref={ref_v}, got={got_v}, rel_err={rel}"
                );
            }
        }
    }

    // ── gemv_q4k_8x8 tests ────────────────────────────────────────────────

    #[allow(dead_code)]
    fn make_q4k_from_activation(weights: &[f32]) -> BlockQ4K {
        use crate::ggml::helpers::f32_to_f16;
        assert_eq!(weights.len(), 256);
        let amax = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        let inv_d = 1.0 / d;
        let mut qs = [0u8; 128];
        for k in 0..128 {
            let lo = ((weights[k] * inv_d + 8.5) as i32).clamp(0, 15) as u8;
            let hi = ((weights[k + 128] * inv_d + 8.5) as i32).clamp(0, 15) as u8;
            // Wait, Q4K stores elements differently: lo nibble = element at position k in sub-block
            // hi nibble = element at position k+32 (the next 32 elements in same byte-chunk)
            // Actually, we need to match the dequant layout:
            // qs[k] lo = y[floor(k/32)*64 + (k%32)], hi = y[floor(k/32)*64 + (k%32) + 32]
            // For this test, just use simple weights (all same value)
            let _ = lo;
            let _ = hi;
            qs[k] = ((weights[k] * inv_d + 8.5) as i32).clamp(0, 15) as u8
                | ((((weights[k + 128] * inv_d + 8.5) as i32).clamp(0, 15) as u8) << 4);
        }
        // Set scales to uniform: all 8 scale groups use d, mins=0
        // Scale index 0..7 = d, min 0..7 = 0
        // Pack using Q4K format: scales[0..4] = sc[0..3] with high bits from sc[4..7]
        let sc = (1u8) & 63; // scale factor 1 (d already encodes the magnitude)
        let mn = 0u8;
        let mut scales = [0u8; 12];
        for i in 0..4 {
            scales[i] = sc | ((sc & 0x30) << 2);
            scales[i + 4] = mn;
        }
        scales[8] = (sc & 0x0F) | ((mn & 0x0F) << 4);
        scales[9] = scales[8];
        scales[10] = scales[8];
        scales[11] = scales[8];
        BlockQ4K {
            d: f32_to_f16(d),
            dmin: f32_to_f16(0.0),
            scales,
            qs,
        }
    }

    #[test]
    fn gemv_q4k_8x8_scalar_matches_per_row_dispatch() {
        use crate::ggml::repack::repack_q4k_to_q4kx8;

        let bpr = 2usize; // 2 blocks per row → k=512
        let n_rows = 8usize;
        let k = bpr * 256;

        // Build 8 weight rows with distinct values
        let mut src_blocks: Vec<BlockQ4K> = Vec::with_capacity(n_rows * bpr);
        let mut weight_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row_weights: Vec<f32> = (0..k)
                .map(|j| ((r * k + j) as f32 - (k as f32 / 2.0)) / (k as f32))
                .collect();
            weight_vecs.push(row_weights.clone());
            for blk in 0..bpr {
                let chunk = &row_weights[blk * 256..(blk + 1) * 256];
                let mut block = BlockQ4K {
                    d: 0,
                    dmin: 0,
                    scales: [0u8; 12],
                    qs: [0u8; 128],
                };
                // Use quantize helper to build a valid Q4K block
                let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let d = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                block.d = crate::ggml::helpers::f32_to_f16(d);
                block.dmin = 0;
                let inv_d = 1.0 / d;
                // Q4K qs layout: qs[k] lo nibble = element k*2, hi = element k*2+1? No.
                // From dequant: for _j=0..4, qs[_j*32.._j*32+32] lo=y[_j*64.._j*64+32], hi=y[_j*64+32.._j*64+64]
                for _j in 0..4usize {
                    for l in 0..32usize {
                        let lo = ((chunk[_j * 64 + l] * inv_d + 0.5) as i32).clamp(0, 15) as u8;
                        let hi = ((chunk[_j * 64 + 32 + l] * inv_d + 0.5) as i32).clamp(0, 15) as u8;
                        block.qs[_j * 32 + l] = lo | (hi << 4);
                    }
                }
                // Uniform scale=1 for all 8 sub-groups, min=0
                let sc = 1u8;
                for i in 0..4 { block.scales[i] = sc; block.scales[i+4] = 0; }
                block.scales[8] = sc & 0x0F; block.scales[9] = block.scales[8];
                block.scales[10] = block.scales[8]; block.scales[11] = block.scales[8];
                src_blocks.push(block);
            }
        }

        // Build activation row
        let act_float: Vec<f32> = (0..k).map(|j| (j as f32 - k as f32 / 2.0) / k as f32).collect();
        let mut act_q8k: Vec<BlockQ8K> = vec![
            BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] };
            bpr
        ];
        super::super::quantize::quantize_row_q8_k(&act_float, &mut act_q8k);

        // Per-row reference using dispatch
        let ref_results: Vec<f32> = (0..n_rows)
            .map(|r| {
                let row_blocks: Vec<BlockQ4K> =
                    (0..bpr).map(|b| src_blocks[r * bpr + b]).collect();
                vec_dot_q4_k_q8_k_dispatch(k, &row_blocks, &act_q8k)
            })
            .collect();

        // Repack and run 8×8 scalar
        let packed = repack_q4k_to_q4kx8(&src_blocks, n_rows, bpr);
        let result8 = gemv_q4k_8x8_scalar(bpr, &packed, &act_q8k);

        for r in 0..n_rows {
            let rel = if ref_results[r].abs() > 0.01 {
                ((result8[r] - ref_results[r]) / ref_results[r]).abs()
            } else {
                (result8[r] - ref_results[r]).abs()
            };
            assert!(
                rel < 0.01,
                "row={r}: scalar8={}, ref={}, rel_err={rel}",
                result8[r],
                ref_results[r]
            );
        }
    }

    #[test]
    fn gemv_q4k_8x8_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        use crate::ggml::repack::repack_q4k_to_q4kx8;

        let bpr = 1usize;
        let k = 256;
        let n_rows = 8usize;

        // Simple uniform weights: all nibbles = 4 (value 4*d), scale=1, min=0
        let src_blocks: Vec<BlockQ4K> = (0..n_rows * bpr)
            .map(|r| {
                use crate::ggml::helpers::f32_to_f16;
                let d = 0.5 + (r as f32) * 0.1;
                let mut b = BlockQ4K {
                    d: f32_to_f16(d),
                    dmin: f32_to_f16(0.0),
                    scales: [0u8; 12],
                    qs: [0x44u8; 128], // lo=4, hi=4 for all nibbles
                };
                // scales: uniform scale=1, min=0
                b.scales[0] = 1; b.scales[1] = 1; b.scales[2] = 1; b.scales[3] = 1;
                b.scales[4] = 0; b.scales[5] = 0; b.scales[6] = 0; b.scales[7] = 0;
                b.scales[8] = 1; b.scales[9] = 1; b.scales[10] = 1; b.scales[11] = 1;
                b
            })
            .collect();

        let act_float: Vec<f32> = (0..k).map(|j| (j as f32 - 128.0) / 256.0).collect();
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; bpr];
        super::super::quantize::quantize_row_q8_k(&act_float, &mut act_q8k);

        let packed = repack_q4k_to_q4kx8(&src_blocks, n_rows, bpr);

        let scalar_result = gemv_q4k_8x8_scalar(bpr, &packed, &act_q8k);
        let avx2_result = unsafe { gemv_q4k_8x8_avx2(bpr, &packed, &act_q8k) };

        for r in 0..n_rows {
            let rel = if scalar_result[r].abs() > 0.001 {
                ((avx2_result[r] - scalar_result[r]) / scalar_result[r]).abs()
            } else {
                (avx2_result[r] - scalar_result[r]).abs()
            };
            assert!(
                rel < 0.02,
                "row={r}: avx2={}, scalar={}, rel_err={rel}",
                avx2_result[r],
                scalar_result[r]
            );
        }
    }

    #[test]
    fn gemv_q4k_8x8v_scalar_matches_q4kx8_scalar() {
        use super::super::repack::{repack_q4k_to_q4kx8, repack_q4k_to_q4kx8v};
        use super::super::quantize::quantize_row_q8_k;
        let n_rows = 8usize;
        let bpr = 2usize;
        let k = bpr * 256;

        let src_blocks: Vec<BlockQ4K> = (0..n_rows * bpr)
            .map(|i| BlockQ4K {
                d: f32_to_f16(0.1 + 0.02 * i as f32),
                dmin: f32_to_f16(0.01 + 0.005 * i as f32),
                scales: std::array::from_fn(|j| ((i * 7 + j * 3) & 63) as u8),
                qs: std::array::from_fn(|j| {
                    let lo = ((i + j) as u8) & 0x0F;
                    let hi = ((i * 5 + j * 3) as u8) & 0x0F;
                    lo | (hi << 4)
                }),
            })
            .collect();

        let act_float: Vec<f32> = (0..k).map(|j| (j as f32 - 128.0) / 256.0).collect();
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; bpr];
        quantize_row_q8_k(&act_float, &mut act_q8k);

        let packed_x8 = repack_q4k_to_q4kx8(&src_blocks, n_rows, bpr);
        let packed_x8v = repack_q4k_to_q4kx8v(&src_blocks, n_rows, bpr);

        let ref_result = gemv_q4k_8x8_scalar(bpr, &packed_x8, &act_q8k);
        let vnni_scalar = gemv_q4k_8x8v_scalar(bpr, &packed_x8v, &act_q8k);

        for r in 0..n_rows {
            let rel = if ref_result[r].abs() > 1e-3 {
                ((vnni_scalar[r] - ref_result[r]) / ref_result[r]).abs()
            } else {
                (vnni_scalar[r] - ref_result[r]).abs()
            };
            assert!(
                rel < 1e-4,
                "row={r}: ref={}, vnni_scalar={}, rel_err={rel}",
                ref_result[r],
                vnni_scalar[r]
            );
        }
    }

    #[test]
    fn gemv_q4k_8x8_vnni_matches_scalar() {
        use super::super::repack::{repack_q4k_to_q4kx8, repack_q4k_to_q4kx8v};
        use super::super::quantize::quantize_row_q8_k;
        let n_rows = 8usize;
        let bpr = 2usize;
        let k = bpr * 256;

        let src_blocks: Vec<BlockQ4K> = (0..n_rows * bpr)
            .map(|i| BlockQ4K {
                d: f32_to_f16(0.15 + 0.03 * i as f32),
                dmin: f32_to_f16(0.02),
                scales: std::array::from_fn(|j| ((i * 11 + j * 5) & 63) as u8),
                qs: std::array::from_fn(|j| {
                    let lo = ((i * 3 + j) as u8) & 0x0F;
                    let hi = ((i * 7 + j * 2) as u8) & 0x0F;
                    lo | (hi << 4)
                }),
            })
            .collect();

        let act_float: Vec<f32> = (0..k).map(|j| ((j as f32 % 64.0) - 32.0) / 64.0).collect();
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; bpr];
        quantize_row_q8_k(&act_float, &mut act_q8k);

        let packed_x8 = repack_q4k_to_q4kx8(&src_blocks, n_rows, bpr);
        let packed_x8v = repack_q4k_to_q4kx8v(&src_blocks, n_rows, bpr);

        let ref_result = gemv_q4k_8x8_scalar(bpr, &packed_x8, &act_q8k);
        let dispatch_result = gemv_q4k_8x8_vnni_dispatch(bpr, &packed_x8v, &act_q8k);

        for r in 0..n_rows {
            let rel = if ref_result[r].abs() > 1e-3 {
                ((dispatch_result[r] - ref_result[r]) / ref_result[r]).abs()
            } else {
                (dispatch_result[r] - ref_result[r]).abs()
            };
            assert!(
                rel < 1e-4,
                "row={r}: ref={}, vnni={}, rel_err={rel}",
                ref_result[r],
                dispatch_result[r]
            );
        }
    }

    #[test]
    fn gemm_q4k_4a8w_matches_scalar_per_row() {
        use super::{gemm_q4k_4a8w_dispatch, gemm_q4k_4a8w_scalar};
        use crate::ggml::blocks::{BlockQ4K, BlockQ8K};
        use crate::ggml::quantize::quantize_row_q8_k;

        let k = 256usize;
        let bpr = k / 256;

        // Build 8 distinct weight rows (Q4K format)
        let mut w_blocks: [Vec<BlockQ4K>; 8] = std::array::from_fn(|_| {
            vec![unsafe { std::mem::zeroed::<BlockQ4K>() }]
        });
        for (wi, wb) in w_blocks.iter_mut().enumerate() {
            let d_f32 = (wi as f32 + 1.0) * 0.05;
            wb[0].d = crate::ggml::helpers::f32_to_f16(d_f32);
            wb[0].dmin = crate::ggml::helpers::f32_to_f16(0.0);
            for b in wb[0].qs.iter_mut() {
                *b = ((wi * 17 + 3) & 0xFF) as u8;
            }
            for s in wb[0].scales.iter_mut() {
                *s = 10;
            }
        }

        // Build 4 distinct activation rows (Q8K format)
        let mut a_blocks: [Vec<BlockQ8K>; 4] =
            std::array::from_fn(|_| vec![unsafe { std::mem::zeroed::<BlockQ8K>() }]);
        for (ri, ab) in a_blocks.iter_mut().enumerate() {
            let a_f32: Vec<f32> = (0..k).map(|j| ((ri * k + j) % 127) as f32 * 0.1).collect();
            quantize_row_q8_k(&a_f32, &mut ab[..]);
        }

        let ws: [&[BlockQ4K]; 8] = std::array::from_fn(|wi| w_blocks[wi].as_slice());
        let as_: [&[BlockQ8K]; 4] = std::array::from_fn(|ri| a_blocks[ri].as_slice());

        let dispatch_result = gemm_q4k_4a8w_dispatch(k, ws, as_);
        let scalar_result = gemm_q4k_4a8w_scalar(k, ws, as_);

        for r in 0..4 {
            for w in 0..8 {
                let d = dispatch_result[r][w];
                let s = scalar_result[r][w];
                let rel = if s.abs() < 1e-6 { (d - s).abs() } else { (d - s).abs() / s.abs() };
                assert!(
                    rel < 1e-4,
                    "gemm_4a8w mismatch at r={r} w={w}: dispatch={d}, scalar={s}, rel={rel}"
                );
            }
        }
    }
}

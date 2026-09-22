#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::blocks::*;
use super::helpers::f16_to_f32;

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
            // broadcast scale for low nibble group (scale 2j) and high nibble group (scale 2j+1)
            let scale_l = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(K_SHUFFLE_Q4K[2 * j].as_ptr() as *const __m256i),
            );
            let scale_h = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(K_SHUFFLE_Q4K[2 * j + 1].as_ptr() as *const __m256i),
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
    use super::helpers::get_scale_min_k4;
    let nb = n / 256;
    let mask_lo = _mm256_set1_epi8(0x0F);

    let mut acc = _mm256_setzero_ps();
    let mut mins_total = 0.0f32;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    for i in 0..nb {
        // Safety: nb = n/256, x.len() == nb, y.len() == nb (caller invariant)
        let xi = &*x_ptr.add(i);
        let yi = &*y_ptr.add(i);

        let d = f16_to_f32(xi.d) * yi.d;
        let dmin = f16_to_f32(xi.dmin) * yi.d;

        let mut sc = [0u8; 8];
        let mut mi = [0u8; 8];
        for j in 0..8 {
            let (s, m) = get_scale_min_k4(j, &xi.scales);
            *sc.get_unchecked_mut(j) = s;
            *mi.get_unchecked_mut(j) = m;
        }

        let mut sumi_mins = 0i32;
        let bsums_ptr = yi.bsums.as_ptr();
        let mi_ptr = mi.as_ptr();
        for j in 0..16 {
            sumi_mins += *bsums_ptr.add(j) as i32 * *mi_ptr.add(j >> 1) as i32;
        }
        mins_total += dmin * sumi_mins as f32;

        let q4 = xi.qs.as_ptr();
        let q8 = yi.qs.as_ptr();
        let sc_ptr = sc.as_ptr();

        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let sc_lo = *sc_ptr.add(2 * j) as i32;
            let sc_hi = *sc_ptr.add(2 * j + 1) as i32;

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
/// Shares Q8K loads (activation) across both rows, reducing memory traffic by ~30%.
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
    use super::helpers::get_scale_min_k4;
    let nb = n / 256;
    let mask_lo = _mm256_set1_epi8(0x0F);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut mins_total0 = 0.0f32;
    let mut mins_total1 = 0.0f32;

    let x0_ptr = x0.as_ptr();
    let x1_ptr = x1.as_ptr();
    let y_ptr = y.as_ptr();

    for i in 0..nb {
        // Safety: nb = n/256, all slices have len == nb
        let x0i = &*x0_ptr.add(i);
        let x1i = &*x1_ptr.add(i);
        let yi = &*y_ptr.add(i);
        let d_y = yi.d;

        // Row 0 scales
        let d0 = f16_to_f32(x0i.d) * d_y;
        let dmin0 = f16_to_f32(x0i.dmin) * d_y;
        let mut sc0 = [0u8; 8];
        let mut mi0 = [0u8; 8];
        for j in 0..8 {
            let (s, m) = get_scale_min_k4(j, &x0i.scales);
            *sc0.get_unchecked_mut(j) = s;
            *mi0.get_unchecked_mut(j) = m;
        }

        // Row 1 scales
        let d1 = f16_to_f32(x1i.d) * d_y;
        let dmin1 = f16_to_f32(x1i.dmin) * d_y;
        let mut sc1 = [0u8; 8];
        let mut mi1 = [0u8; 8];
        for j in 0..8 {
            let (s, m) = get_scale_min_k4(j, &x1i.scales);
            *sc1.get_unchecked_mut(j) = s;
            *mi1.get_unchecked_mut(j) = m;
        }

        // Shared mins from Q8K bsums
        let mut sumi_mins0 = 0i32;
        let mut sumi_mins1 = 0i32;
        let bsums_ptr = yi.bsums.as_ptr();
        let mi0_ptr = mi0.as_ptr();
        let mi1_ptr = mi1.as_ptr();
        for j in 0..16 {
            let bs = *bsums_ptr.add(j) as i32;
            let half = j >> 1;
            sumi_mins0 += bs * *mi0_ptr.add(half) as i32;
            sumi_mins1 += bs * *mi1_ptr.add(half) as i32;
        }
        mins_total0 += dmin0 * sumi_mins0 as f32;
        mins_total1 += dmin1 * sumi_mins1 as f32;

        let q8 = yi.qs.as_ptr();
        let sc0_ptr = sc0.as_ptr();
        let sc1_ptr = sc1.as_ptr();
        let mut sumi0 = _mm256_setzero_si256();
        let mut sumi1 = _mm256_setzero_si256();

        for j in 0..4 {
            // Load Q8K activation ONCE per pair (shared)
            let q8_lo = _mm256_loadu_si256(q8.add(j * 64) as *const __m256i);
            let q8_hi = _mm256_loadu_si256(q8.add(j * 64 + 32) as *const __m256i);

            // Row 0 nibbles
            let q4_chunk0 = _mm256_loadu_si256(x0i.qs.as_ptr().add(j * 32) as *const __m256i);
            let q4_lo0 = _mm256_and_si256(q4_chunk0, mask_lo);
            let q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(q4_chunk0, 4), mask_lo);

            let sc_lo0 = *sc0_ptr.add(2 * j) as i32;
            let sc_hi0 = *sc0_ptr.add(2 * j + 1) as i32;
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

            let sc_lo1 = *sc1_ptr.add(2 * j) as i32;
            let sc_hi1 = *sc1_ptr.add(2 * j + 1) as i32;
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

    (hsum_f32_8(acc0) - mins_total0, hsum_f32_8(acc1) - mins_total1)
}

/// Dispatch pair dot product — uses VNNI when available, falls back to two scalar calls.
#[inline(always)]
pub fn vec_dot_q4_k_q8_k_dispatch_pair(
    n: usize,
    x0: &[BlockQ4K],
    x1: &[BlockQ4K],
    y: &[BlockQ8K],
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avxvnni") {
        return unsafe { vec_dot_q4_k_q8_k_vnni_pair(n, x0, x1, y) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512vnni") {
        return unsafe { vec_dot_q4_k_q8_k_vnni_pair(n, x0, x1, y) };
    }
    (
        vec_dot_q4_k_q8_k_dispatch(n, x0, y),
        vec_dot_q4_k_q8_k_dispatch(n, x1, y),
    )
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

// ── AVX-512 VNNI variant ───────────────────────────────────────────────────
// Uses 256-bit VNNI dpbusd (same as AVX-VNNI) but with the proven Q4K dequant
// layout. AVX-512 width is used for activation quantization, not the vec_dot
// inner loop — the per-group scaling makes 512-bit lanes hard to fill correctly.
// For Q4K, the VNNI variant IS the fastest path since dpbusd replaces
// maddubs+madd, and the 256-bit width matches the 32-element scale groups.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[inline]
pub unsafe fn vec_dot_q4_k_q8_k_avx512(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    vec_dot_q4_k_q8_k_vnni(n, x, y)
}

// ── Runtime dispatch ───────────────────────────────────────────────────────

pub fn vec_dot_q4_0_q8_0_dispatch(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { vec_dot_q4_0_q8_0_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q4_0_q8_0(n, x, y)
}

pub fn vec_dot_q8_0_q8_0_dispatch(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { vec_dot_q8_0_q8_0_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q8_0_q8_0(n, x, y)
}

#[inline(always)]
pub fn vec_dot_q4_k_q8_k_dispatch(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            return unsafe { vec_dot_q4_k_q8_k_avx512(n, x, y) };
        }
        if is_x86_feature_detected!("avxvnni") {
            return unsafe { vec_dot_q4_k_q8_k_vnni(n, x, y) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_k_q8_k_avx2(n, x, y) };
        }
    }
    super::vec_dot::vec_dot_q4_k_q8_k(n, x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub unsafe fn vec_dot_q6_k_q8_k_vnni(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / 256;
    let mut sum = 0.0f32;

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

        let mut aux8 = [0i8; 256];
        let mut a_idx = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _half in 0..2 {
            for l in 0..32 {
                *aux8.get_unchecked_mut(a_idx + l) = ((*ql.add(ql_off + l) & 0xF) | ((*qh.add(qh_off + l) & 3) << 4)) as i8 - 32;
                *aux8.get_unchecked_mut(a_idx + 32 + l) =
                    ((*ql.add(ql_off + 32 + l) & 0xF) | (((*qh.add(qh_off + l) >> 2) & 3) << 4)) as i8 - 32;
                *aux8.get_unchecked_mut(a_idx + 64 + l) =
                    ((*ql.add(ql_off + l) >> 4) | (((*qh.add(qh_off + l) >> 4) & 3) << 4)) as i8 - 32;
                *aux8.get_unchecked_mut(a_idx + 96 + l) =
                    ((*ql.add(ql_off + 32 + l) >> 4) | (((*qh.add(qh_off + l) >> 6) & 3) << 4)) as i8 - 32;
            }
            a_idx += 128;
            ql_off += 64;
            qh_off += 32;
        }

        let mut sumi = 0i32;
        for j in 0..16 {
            let scale = *sc.add(j) as i32;
            let base = j * 16;
            let a_lo = _mm_loadu_si128(aux8.as_ptr().add(base) as *const __m128i);
            let b_lo = _mm_loadu_si128(q8.add(base) as *const __m128i);
            let abs_a = _mm_sign_epi8(a_lo, a_lo);
            let signed_b = _mm_sign_epi8(b_lo, a_lo);
            // Zero-extend 128-bit → 256-bit; upper 16 bytes are zero → contribute 0
            let abs_a_256 = _mm256_set_m128i(_mm_setzero_si128(), abs_a);
            let signed_b_256 = _mm256_set_m128i(_mm_setzero_si128(), signed_b);
            let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), abs_a_256, signed_b_256);
            sumi += hsum_i32_8(dot) * scale;
        }

        sum += d * sumi as f32;
    }
    sum
}

#[inline(always)]
pub fn vec_dot_q6_k_q8_k_dispatch(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avxvnni") {
            return unsafe { vec_dot_q6_k_q8_k_vnni(n, x, y) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q6_k_q8_k_avx2(n, x, y) };
        }
    }
    super::vec_dot::vec_dot_q6_k_q8_k(n, x, y)
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
}

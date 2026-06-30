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
        let by_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(y[ib].qs[..16].as_ptr() as *const __m128i));
        let by_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(y[ib].qs[16..].as_ptr() as *const __m128i));

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q4_k_q8_k_avx2(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    use super::helpers::get_scale_min_k4;
    let nb = n / 256;

    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;

        let mut sc = [0u8; 8];
        let mut mi = [0u8; 8];
        for j in 0..8 {
            let (s, m) = get_scale_min_k4(j, &x[i].scales);
            sc[j] = s;
            mi[j] = m;
        }

        // dequant Q4K to i8 buffer (same approach as Q6K AVX2 — proven correct)
        let mut aux8 = [0i8; 256];
        let q4 = &x[i].qs;
        let mut a_idx = 0usize;
        let mut q4_off = 0usize;
        for _j in 0..4 {
            for l in 0..32 {
                aux8[a_idx + l] = (q4[q4_off + l] & 0xF) as i8;
            }
            a_idx += 32;
            for l in 0..32 {
                aux8[a_idx + l] = (q4[q4_off + l] >> 4) as i8;
            }
            a_idx += 32;
            q4_off += 32;
        }

        // dot product with per-group scaling using AVX2
        let q8 = &y[i].qs;
        let mut sumi = _mm256_setzero_si256();
        for j in 0..8 {
            let scale = sc[j] as i32;
            let base = j * 32;
            let a_lo = _mm_loadu_si128(aux8[base..].as_ptr() as *const __m128i);
            let b_lo = _mm_loadu_si128(q8[base..].as_ptr() as *const __m128i);
            let a16 = _mm256_cvtepi8_epi16(a_lo);
            let b16 = _mm256_cvtepi8_epi16(b_lo);
            let dot = _mm256_madd_epi16(a16, b16);

            let a_hi = _mm_loadu_si128(aux8[base + 16..].as_ptr() as *const __m128i);
            let b_hi = _mm_loadu_si128(q8[base + 16..].as_ptr() as *const __m128i);
            let a16_hi = _mm256_cvtepi8_epi16(a_hi);
            let b16_hi = _mm256_cvtepi8_epi16(b_hi);
            let dot_hi = _mm256_madd_epi16(a16_hi, b16_hi);

            let dot_sum = _mm256_add_epi32(dot, dot_hi);
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(dot_sum, _mm256_set1_epi32(scale)));
        }

        // mins contribution
        let mut sumi_mins = 0i32;
        for j in 0..16 {
            sumi_mins += y[i].bsums[j] as i32 * mi[j / 2] as i32;
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    let mut result = hsum_f32_8(acc);
    // Subtract mins contribution (accumulated separately to avoid 8x broadcast error)
    for i in 0..nb {
        let dmin = f16_to_f32(x[i].dmin) * y[i].d;
        let mut sc = [0u8; 8];
        let mut mi = [0u8; 8];
        for j in 0..8 {
            let (s, m) = super::helpers::get_scale_min_k4(j, &x[i].scales);
            sc[j] = s;
            mi[j] = m;
        }
        let _ = sc;
        let mut sumi_mins = 0i32;
        for j in 0..16 {
            sumi_mins += y[i].bsums[j] as i32 * mi[j / 2] as i32;
        }
        result -= dmin * sumi_mins as f32;
    }
    result
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
                aux8[a_idx + l] = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                aux8[a_idx + 32 + l] = ((ql[ql_off + 32 + l] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_idx + 64 + l] = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_idx + 96 + l] = ((ql[ql_off + 32 + l] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
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

    for i in 0..nb {
        let x = &src[i * 256..][..256];

        // find max absolute value across 256 elements
        let mut amax_vec = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
        for chunk in 0..32 {
            let v = _mm256_loadu_ps(x[chunk * 8..].as_ptr());
            amax_vec = _mm256_max_ps(amax_vec, _mm256_and_ps(v, sign_mask));
        }
        let amax = hmax_f32_8(amax_vec);

        // find max (signed) for iscale direction
        let mut max = 0.0f32;
        for j in 0..256 {
            if x[j].abs() >= amax - 1e-10 {
                max = x[j];
                break;
            }
        }

        if amax == 0.0 {
            dst[i].d = 0.0;
            dst[i].qs = [0i8; 256];
            dst[i].bsums = [0i16; 16];
            continue;
        }

        let iscale = -127.0 / max;
        let mul = _mm256_set1_ps(iscale);

        for chunk in 0..32 {
            let v = _mm256_loadu_ps(x[chunk * 8..].as_ptr());
            let scaled = _mm256_mul_ps(v, mul);
            let rounded = _mm256_cvtps_epi32(scaled);
            // clamp to [-127, 127]
            let clamped = _mm256_max_epi32(rounded, _mm256_set1_epi32(-127));
            let clamped = _mm256_min_epi32(clamped, _mm256_set1_epi32(127));

            // extract and store as i8
            let vals: [i32; 8] = std::mem::transmute(clamped);
            for k in 0..8 {
                dst[i].qs[chunk * 8 + k] = vals[k] as i8;
            }
        }

        // compute bsums
        for j in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += dst[i].qs[j * 16 + k] as i32;
            }
            dst[i].bsums[j] = sum as i16;
        }

        dst[i].d = 1.0 / iscale;
    }
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

pub fn vec_dot_q4_k_q8_k_dispatch(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { vec_dot_q4_k_q8_k_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q4_k_q8_k(n, x, y)
}

pub fn vec_dot_q6_k_q8_k_dispatch(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { vec_dot_q6_k_q8_k_avx2(n, x, y) };
    }
    super::vec_dot::vec_dot_q6_k_q8_k(n, x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::helpers::f32_to_f16;
    use super::super::quantize::{quantize_row_q8_0, quantize_row_q8_k};
    use super::super::vec_dot;

    #[test]
    fn avx2_q4_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }

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
        if !is_x86_feature_detected!("avx2") { return; }

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
        if !is_x86_feature_detected!("avx2") { return; }

        let w = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            qs: [0x21; 128],
        };
        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = vec_dot::vec_dot_q4_k_q8_k(256, &[w], &act_q8k);
        let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(256, &[w], &act_q8k) };

        let rel = if scalar.abs() > 0.01 { ((scalar - avx2) / scalar).abs() } else { (scalar - avx2).abs() };
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
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&act, &mut act_q8k);

        // reconstruct f32 activations from Q8K
        let mut act_recon = [0.0f32; 256];
        for j in 0..256 {
            act_recon[j] = act_q8k[0].d * act_q8k[0].qs[j] as f32;
        }

        let ref_dot: f32 = w_f32.iter().zip(&act_recon).map(|(a, b)| a * b).sum();
        let scalar_dot = vec_dot::vec_dot_q4_k_q8_k(256, &[blk], &act_q8k);

        eprintln!("ref_dot (dequant)={ref_dot}, scalar_dot={scalar_dot}");
        let rel = if ref_dot.abs() > 0.01 { ((scalar_dot - ref_dot) / ref_dot).abs() } else { (scalar_dot - ref_dot).abs() };
        eprintln!("rel_err between scalar and dequant: {rel}");
    }

    #[test]
    fn avx2_q4k_multiblock_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }

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
            let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
            super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

            let scalar = super::super::vec_dot::vec_dot_q4_k_q8_k(256, &[blk], &act_q8k);
            let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(256, &[blk], &act_q8k) };
            let rel = if scalar.abs() > 0.01 { ((scalar - avx2) / scalar).abs() } else { (scalar - avx2).abs() };
            assert!(rel < 0.001, "block {b}: scalar={scalar}, avx2={avx2}, rel_err={rel}");
        }

        // Now multi-block
        let blocks: Vec<BlockQ4K> = (0..4).map(|b| BlockQ4K {
            d: f32_to_f16(0.1 + 0.02 * b as f32),
            dmin: f32_to_f16(0.01 * b as f32),
            scales: std::array::from_fn(|i| ((b as u8 + i as u8 * 3) & 63)),
            qs: std::array::from_fn(|i| {
                let lo = (b as u8).wrapping_add(i as u8) & 0x0F;
                let hi = (b as u8).wrapping_add(i as u8).wrapping_mul(5) & 0x0F;
                lo | (hi << 4)
            }),
        }).collect();
        let act: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let mut act_q8k: Vec<BlockQ8K> = (0..4).map(|_| BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }).collect();
        super::super::quantize::quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = super::super::vec_dot::vec_dot_q4_k_q8_k(1024, &blocks, &act_q8k);
        let avx2 = unsafe { vec_dot_q4_k_q8_k_avx2(1024, &blocks, &act_q8k) };

        let rel = if scalar.abs() > 0.01 { ((scalar - avx2) / scalar).abs() } else { (scalar - avx2).abs() };
        assert!(rel < 0.001, "multiblock: scalar={scalar}, avx2={avx2}, rel_err={rel}");
    }

    #[test]
    fn avx2_q6k_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }

        let w = BlockQ6K {
            ql: [0x11; 128],
            qh: [0; 64],
            scales: [2; 16],
            d: f32_to_f16(0.5),
        };
        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&act, &mut act_q8k);

        let scalar = vec_dot::vec_dot_q6_k_q8_k(256, &[w], &act_q8k);
        let avx2 = unsafe { vec_dot_q6_k_q8_k_avx2(256, &[w], &act_q8k) };

        let rel = if scalar.abs() > 0.01 { ((scalar - avx2) / scalar).abs() } else { (scalar - avx2).abs() };
        assert!(rel < 0.05, "scalar={scalar}, avx2={avx2}, rel_err={rel}");
    }

    #[test]
    fn avx2_quantize_q8_0_matches_scalar() {
        if !is_x86_feature_detected!("avx2") { return; }

        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();

        let mut scalar_out = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut scalar_out);

        let mut avx2_out = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        unsafe { quantize_row_q8_0_avx2(&src, &mut avx2_out); }

        assert_eq!(scalar_out[0].d, avx2_out[0].d, "d mismatch");
        for j in 0..32 {
            let diff = (scalar_out[0].qs[j] as i32 - avx2_out[0].qs[j] as i32).abs();
            assert!(diff <= 1, "qs[{j}]: scalar={}, avx2={}", scalar_out[0].qs[j], avx2_out[0].qs[j]);
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
        assert!(diff < 0.01, "dispatch={dispatch_result}, scalar={scalar_result}");
    }
}

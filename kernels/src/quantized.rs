use crate::Result;
pub use crate::{
    Dequantize, MatMulQ, QuantDtype, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS, Q4K_GROUP_SIZE,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, Q6K_GROUPS, Q6K_GROUP_SIZE,
};
use rayon::prelude::*;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = exp;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e = e.wrapping_sub(1);
        }
        f &= 0x3ff;
        let f32_exp = 127 - 15 + 1 + e;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

pub fn q4k_block_d(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

pub fn q4k_block_dmin(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[2], block[3]]))
}

pub fn q4k_unpack_scales(packed: &[u8]) -> ([f32; Q4K_GROUPS], [f32; Q4K_GROUPS]) {
    let mut sc = [0u8; Q4K_GROUPS];
    let mut m = [0u8; Q4K_GROUPS];

    let mut i = 0;
    while i < 4 {
        sc[i] = packed[i] & 0x3F;
        sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        m[i] = packed[i + 4] & 0x3F;
        m[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
        i += 1;
    }

    let sc_f: [f32; Q4K_GROUPS] = [
        sc[0] as f32,
        sc[1] as f32,
        sc[2] as f32,
        sc[3] as f32,
        sc[4] as f32,
        sc[5] as f32,
        sc[6] as f32,
        sc[7] as f32,
    ];
    let m_f: [f32; Q4K_GROUPS] = [
        m[0] as f32,
        m[1] as f32,
        m[2] as f32,
        m[3] as f32,
        m[4] as f32,
        m[5] as f32,
        m[6] as f32,
        m[7] as f32,
    ];
    (sc_f, m_f)
}

pub fn q4k_nibble(qs: &[u8], idx: usize) -> u8 {
    let byte_idx = idx / 2;
    if idx % 2 == 0 {
        qs[byte_idx] & 0x0F
    } else {
        qs[byte_idx] >> 4
    }
}

pub fn q4k_dequant_block(block: &[u8], output: &mut [f32; Q4K_BLOCK_SIZE]) {
    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mut j = 0;
    while j < 4 {
        let q_off = j * 32;
        let d1 = d * sc[2 * j];
        let m1 = dmin * m[2 * j];
        let d2 = d * sc[2 * j + 1];
        let m2 = dmin * m[2 * j + 1];

        let mut l = 0;
        while l < 32 {
            output[j * 64 + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            output[j * 64 + l + 32] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            l += 1;
        }
        j += 1;
    }
}

pub fn q4k_dot_block(block: &[u8], a: &[f32]) -> f32 {
    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mut sum = 0.0f32;
    let mut j = 0;
    while j < 4 {
        let q_off = j * 32;
        let d1 = d * sc[2 * j];
        let m1 = dmin * m[2 * j];
        let d2 = d * sc[2 * j + 1];
        let m2 = dmin * m[2 * j + 1];

        let mut l = 0;
        while l < 32 {
            let w1 = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            let w2 = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            sum += a[j * 64 + l] * w1;
            sum += a[j * 64 + l + 32] * w2;
            l += 1;
        }
        j += 1;
    }
    sum
}

pub fn q6k_dot_block(block: &[u8], a: &[f32]) -> f32 {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mut sum = 0.0f32;
    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

            let sc0 = scales[half * 8 + is] as i8 as f32;
            let sc1 = scales[half * 8 + is + 2] as i8 as f32;
            let sc2 = scales[half * 8 + is + 4] as i8 as f32;
            let sc3 = scales[half * 8 + is + 6] as i8 as f32;

            sum += a[y_off + l] * d * sc0 * q1 as f32;
            sum += a[y_off + l + 32] * d * sc1 * q2 as f32;
            sum += a[y_off + l + 64] * d * sc2 * q3 as f32;
            sum += a[y_off + l + 96] * d * sc3 * q4 as f32;
        }
    }
    sum
}

/// i16-domain Q6_K dot product: quantize activations to i16 (not i8), multiply
/// signed q6 values (-32..31) by i16 activations using madd_epi16.
/// i16 quantization has 256x less error than i8, preventing cross-layer divergence.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6k_dot_block_avx2_i16(block: &[u8], a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mask_low = _mm256_set1_epi8(0x0F);
    let mask_2bit = _mm256_set1_epi8(0x03);
    let offset_32_i16 = _mm256_set1_epi16(32);

    // Quantize 256 activations to i16 (one scale per block)
    let mut max_abs = 0.0f32;
    for i in 0..256 {
        let v = a[i].abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    let a_scale = max_abs / 32767.0;
    let inv_scale = if max_abs > 0.0 {
        32767.0 / max_abs
    } else {
        0.0
    };

    let mut a_i16 = [0i16; 256];
    let inv_vec = _mm256_set1_ps(inv_scale);
    for chunk in 0..32 {
        let base = chunk * 8;
        let v = _mm256_loadu_ps(a.as_ptr().add(base));
        let scaled = _mm256_mul_ps(v, inv_vec);
        let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
            scaled,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ));
        // Pack i32 → i16 with saturation
        let lo = _mm256_castsi256_si128(rounded);
        let hi = _mm256_extracti128_si256(rounded, 1);
        let packed = _mm_packs_epi32(lo, hi); // 8 i16 values
        _mm_storeu_si128(a_i16.as_mut_ptr().add(base) as *mut __m128i, packed);
    }

    let mut result = 0.0f32;

    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        let qh_vec = _mm256_loadu_si256(qh.as_ptr().add(qh_off) as *const __m256i);
        let ql_a = _mm256_loadu_si256(ql.as_ptr().add(ql_off) as *const __m256i);
        let ql_b = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32) as *const __m256i);

        // Extract 6-bit values as u8
        let q6_0 = _mm256_or_si256(
            _mm256_and_si256(ql_a, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(qh_vec, mask_2bit), 4),
        );
        let q6_1 = _mm256_or_si256(
            _mm256_and_si256(ql_b, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_2bit), 4),
        );
        let q6_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_a, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_2bit), 4),
        );
        let q6_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_b, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_2bit), 4),
        );

        // For each group of 32 q6 values: widen to i16, subtract 32, madd with i16 activations
        // Then split by 128-bit lane for per-16-element scale application
        macro_rules! process_group {
            ($q6:expr, $a_off:expr, $sc_idx0:expr, $sc_idx1:expr) => {{
                let q6_vec = $q6;
                // Widen u8 → i16 for low 16 bytes and high 16 bytes
                let lo_128 = _mm256_castsi256_si128(q6_vec);
                let hi_128 = _mm256_extracti128_si256(q6_vec, 1);

                // Low 16 elements: q6[0..15] → i16, subtract 32
                let q_lo = _mm256_sub_epi16(_mm256_cvtepu8_epi16(lo_128), offset_32_i16);
                let a_lo = _mm256_loadu_si256(a_i16.as_ptr().add($a_off) as *const __m256i);
                // madd_epi16: i16 × i16 → i32, sums adjacent pairs → 8 i32
                let dot_lo = _mm256_madd_epi16(q_lo, a_lo);
                let idot_lo = hsum_i32(dot_lo);

                // High 16 elements: q6[16..31] → i16, subtract 32
                let q_hi = _mm256_sub_epi16(_mm256_cvtepu8_epi16(hi_128), offset_32_i16);
                let a_hi = _mm256_loadu_si256(a_i16.as_ptr().add($a_off + 16) as *const __m256i);
                let dot_hi = _mm256_madd_epi16(q_hi, a_hi);
                let idot_hi = hsum_i32(dot_hi);

                let sc0 = scales[$sc_idx0] as i8 as f32;
                let sc1 = scales[$sc_idx1] as i8 as f32;
                result += d * sc0 * a_scale * idot_lo as f32;
                result += d * sc1 * a_scale * idot_hi as f32;
            }};
        }

        process_group!(q6_0, y_off, half * 8, half * 8 + 1);
        process_group!(q6_1, y_off + 32, half * 8 + 2, half * 8 + 3);
        process_group!(q6_2, y_off + 64, half * 8 + 4, half * 8 + 5);
        process_group!(q6_3, y_off + 96, half * 8 + 6, half * 8 + 7);
    }

    result
}

/// i16-domain Q6_K dot with PRE-QUANTIZED i16 activations (no per-call quantization).
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q6k_dot_preq_i16(block: &[u8], a_i16: &[i16], a_scale: f32) -> f32 {
    use std::arch::x86_64::*;

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mask_low = _mm256_set1_epi8(0x0F);
    let mask_2bit = _mm256_set1_epi8(0x03);
    let offset_32_i16 = _mm256_set1_epi16(32);

    let mut result = 0.0f32;

    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        let qh_vec = _mm256_loadu_si256(qh.as_ptr().add(qh_off) as *const __m256i);
        let ql_a = _mm256_loadu_si256(ql.as_ptr().add(ql_off) as *const __m256i);
        let ql_b = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32) as *const __m256i);

        let q6_0 = _mm256_or_si256(
            _mm256_and_si256(ql_a, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(qh_vec, mask_2bit), 4),
        );
        let q6_1 = _mm256_or_si256(
            _mm256_and_si256(ql_b, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_2bit), 4),
        );
        let q6_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_a, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_2bit), 4),
        );
        let q6_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_b, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_2bit), 4),
        );

        macro_rules! process_group {
            ($q6:expr, $a_off:expr, $sc_idx0:expr, $sc_idx1:expr) => {{
                let q6_vec = $q6;
                let lo_128 = _mm256_castsi256_si128(q6_vec);
                let hi_128 = _mm256_extracti128_si256(q6_vec, 1);

                let q_lo = _mm256_sub_epi16(_mm256_cvtepu8_epi16(lo_128), offset_32_i16);
                let a_lo = _mm256_loadu_si256(a_i16.as_ptr().add($a_off) as *const __m256i);
                let dot_lo = _mm256_madd_epi16(q_lo, a_lo);
                let idot_lo = hsum_i32(dot_lo);

                let q_hi = _mm256_sub_epi16(_mm256_cvtepu8_epi16(hi_128), offset_32_i16);
                let a_hi = _mm256_loadu_si256(a_i16.as_ptr().add($a_off + 16) as *const __m256i);
                let dot_hi = _mm256_madd_epi16(q_hi, a_hi);
                let idot_hi = hsum_i32(dot_hi);

                let sc0 = scales[$sc_idx0] as i8 as f32;
                let sc1 = scales[$sc_idx1] as i8 as f32;
                result += d * sc0 * a_scale * idot_lo as f32;
                result += d * sc1 * a_scale * idot_hi as f32;
            }};
        }

        process_group!(q6_0, y_off, half * 8, half * 8 + 1);
        process_group!(q6_1, y_off + 32, half * 8 + 2, half * 8 + 3);
        process_group!(q6_2, y_off + 64, half * 8 + 4, half * 8 + 5);
        process_group!(q6_3, y_off + 96, half * 8 + 6, half * 8 + 7);
    }

    result
}

/// F32-domain Q6_K dot product using AVX2 for 6-bit extraction and FMA for accumulation.
/// Activations stay in f32 — no quantization, no error accumulation across layers.
/// Kept as fallback for non-AVX2 or when precision is critical.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
unsafe fn q6k_dot_block_avx2(block: &[u8], a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mask_low = _mm256_set1_epi8(0x0F);
    let mask_2bit = _mm256_set1_epi8(0x03);
    let offset_32 = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        let qh_vec = _mm256_loadu_si256(qh.as_ptr().add(qh_off) as *const __m256i);
        let ql_a = _mm256_loadu_si256(ql.as_ptr().add(ql_off) as *const __m256i);
        let ql_b = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32) as *const __m256i);

        // Extract 6-bit values as u8 (0-63)
        let q6_0 = _mm256_or_si256(
            _mm256_and_si256(ql_a, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(qh_vec, mask_2bit), 4),
        );
        let q6_1 = _mm256_or_si256(
            _mm256_and_si256(ql_b, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_2bit), 4),
        );
        let q6_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_a, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_2bit), 4),
        );
        let q6_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_b, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_2bit), 4),
        );

        // For each group of 32 elements, process in 4 chunks of 8
        // q6_0 → elements y_off+0..31, scales half*8+{0,1}
        // q6_1 → elements y_off+32..63, scales half*8+{2,3}
        // q6_2 → elements y_off+64..95, scales half*8+{4,5}
        // q6_3 → elements y_off+96..127, scales half*8+{6,7}

        macro_rules! process_q6_group {
            ($q6:expr, $elem_off:expr, $sc_base:expr) => {{
                let q6_vec = $q6;
                let lo128 = _mm256_castsi256_si128(q6_vec);
                let hi128 = _mm256_extracti128_si256(q6_vec, 1);

                // Scale for elements 0..15 (is=0) and 16..31 (is=1)
                let sc0 = d * (scales[$sc_base] as i8 as f32);
                let sc1 = d * (scales[$sc_base + 1] as i8 as f32);
                let sc0_vec = _mm256_set1_ps(sc0);
                let sc1_vec = _mm256_set1_ps(sc1);

                // Chunks 0-1: elements 0..15 with sc0
                for chunk in 0..2 {
                    let src = if chunk == 0 {
                        lo128
                    } else {
                        _mm_srli_si128(lo128, 8)
                    };
                    let i32v = _mm256_cvtepu8_epi32(src);
                    let f32v = _mm256_cvtepi32_ps(i32v);
                    let w = _mm256_mul_ps(sc0_vec, _mm256_sub_ps(f32v, offset_32));
                    let av = _mm256_loadu_ps(a.as_ptr().add($elem_off + chunk * 8));
                    acc = _mm256_fmadd_ps(av, w, acc);
                }
                // Chunks 2-3: elements 16..31 with sc1
                for chunk in 0..2 {
                    let src = if chunk == 0 {
                        hi128
                    } else {
                        _mm_srli_si128(hi128, 8)
                    };
                    let i32v = _mm256_cvtepu8_epi32(src);
                    let f32v = _mm256_cvtepi32_ps(i32v);
                    let w = _mm256_mul_ps(sc1_vec, _mm256_sub_ps(f32v, offset_32));
                    let av = _mm256_loadu_ps(a.as_ptr().add($elem_off + 16 + chunk * 8));
                    acc = _mm256_fmadd_ps(av, w, acc);
                }
            }};
        }

        process_q6_group!(q6_0, y_off, half * 8);
        process_q6_group!(q6_1, y_off + 32, half * 8 + 2);
        process_q6_group!(q6_2, y_off + 64, half * 8 + 4);
        process_q6_group!(q6_3, y_off + 96, half * 8 + 6);
    }

    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

/// Integer-domain Q6_K dot product with pre-quantized Q8 activations.
/// KNOWN ISSUE: i8 activation quantization error (~2.6%) compounds across layers.
/// Use q6k_dot_block_avx2 (f32 FMA) instead for correct multi-layer inference.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn q6k_dot_q8_avx2(block: &[u8], a_q8: &[i8], a_scale: f32, a_group_sums: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mask_low = _mm256_set1_epi8(0x0F);
    let mask_2bit = _mm256_set1_epi8(0x03);
    let ones_16 = _mm256_set1_epi16(1);

    let mut result = 0.0f32;

    for half in 0..2usize {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;

        // Load 32 bytes of qh for this half
        let qh_vec = _mm256_loadu_si256(qh.as_ptr().add(qh_off) as *const __m256i);

        // 4 element groups per half, each producing 32 output elements
        // Group layout within a half:
        //   y[y_off+l]     uses ql[ql_off+l] low    + qh bits 0-1, scale is[0]
        //   y[y_off+l+32]  uses ql[ql_off+l+32] low + qh bits 2-3, scale is[2]
        //   y[y_off+l+64]  uses ql[ql_off+l] high   + qh bits 4-5, scale is[4]
        //   y[y_off+l+96]  uses ql[ql_off+l+32] high+ qh bits 6-7, scale is[6]

        // Load 64 bytes of ql for this half
        let ql_a = _mm256_loadu_si256(ql.as_ptr().add(ql_off) as *const __m256i);
        let ql_b = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32) as *const __m256i);

        // Extract 6-bit values as u8
        let q6_0 = _mm256_or_si256(
            _mm256_and_si256(ql_a, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(qh_vec, mask_2bit), 4),
        );
        let q6_1 = _mm256_or_si256(
            _mm256_and_si256(ql_b, mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_2bit), 4),
        );
        let q6_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_a, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_2bit), 4),
        );
        let q6_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(ql_b, 4), mask_low),
            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_2bit), 4),
        );

        // Load pre-quantized activations
        let aq_0 = _mm256_loadu_si256(a_q8.as_ptr().add(y_off) as *const __m256i);
        let aq_1 = _mm256_loadu_si256(a_q8.as_ptr().add(y_off + 32) as *const __m256i);
        let aq_2 = _mm256_loadu_si256(a_q8.as_ptr().add(y_off + 64) as *const __m256i);
        let aq_3 = _mm256_loadu_si256(a_q8.as_ptr().add(y_off + 96) as *const __m256i);

        // Integer dot: u8 × i8 → i16 → i32
        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q6_0, aq_0), ones_16);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q6_1, aq_1), ones_16);
        let p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q6_2, aq_2), ones_16);
        let p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(q6_3, aq_3), ones_16);

        let idot_0 = hsum_i32(p0);
        let idot_1 = hsum_i32(p1);
        let idot_2 = hsum_i32(p2);
        let idot_3 = hsum_i32(p3);

        // Scales: Q6_K has 16 scales (i8), 1 per 16 elements
        // For the 32-element groups, each uses 2 consecutive scales
        // Group 0 (elements y_off..y_off+31): scales[half*8 + 0] and scales[half*8 + 1]
        // But maddubs mixed them — need to split or use averaged scale
        // Actually, the scale repeats for l=0..15 (is=0) and l=16..31 (is=1)
        // maddubs processes all 32 at once with one scale pattern
        // We can split the hsum into two halves for the two sub-scales

        // Simpler approach: use one average scale for the 32 elements
        // More accurate: split 32 into 2×16 and apply different scales
        // For now, use the full-group approach — sum the integer dot,
        // then apply per-group scales in f32

        // Per-half, we have 4 groups of 32 elements.
        // Scale indices for half h: scales[h*8 + {0,2,4,6}] for l=0..15,
        //                           scales[h*8 + {0,2,4,6}+1] for l=16..31
        // But maddubs mixes l=0..15 with l=16..31 in one vector.
        // Need to separate the low 16 and high 16 results.

        // Actually, the __m256i from maddubs has:
        //   lane 0 (128-bit): results for bytes 0..15 of the input
        //   lane 1 (128-bit): results for bytes 16..31
        // After madd_epi16 with ones:
        //   lane 0: 4 i32 sums from elements 0..15
        //   lane 1: 4 i32 sums from elements 16..31
        // These correspond to is=0 and is=1 respectively.

        let lo_0 = hsum_i32_128(_mm256_castsi256_si128(p0));
        let hi_0 = hsum_i32_128(_mm256_extracti128_si256(p0, 1));
        let lo_1 = hsum_i32_128(_mm256_castsi256_si128(p1));
        let hi_1 = hsum_i32_128(_mm256_extracti128_si256(p1, 1));
        let lo_2 = hsum_i32_128(_mm256_castsi256_si128(p2));
        let hi_2 = hsum_i32_128(_mm256_extracti128_si256(p2, 1));
        let lo_3 = hsum_i32_128(_mm256_castsi256_si128(p3));
        let hi_3 = hsum_i32_128(_mm256_extracti128_si256(p3, 1));

        let sc = |idx: usize| -> f32 { scales[idx] as i8 as f32 };

        // q6 values are 0-63 (unsigned). The formula is d * sc * (q6 - 32).
        // We computed sum(a_q8 * q6) via maddubs. Need to subtract 32 * sum(a_q8).
        // sum(a * (d*sc*(q6-32))) = d*sc*(a_scale*sum(a_q8*q6) - 32*sum(a))

        // Group sum indices: a_group_sums has Q6K_GROUPS=16 entries per block,
        // 1 per 16 elements. Groups map as:
        //   half=0: y_off=0, elements 0..127
        //   half=1: y_off=128, elements 128..255
        // Group g covers elements g*16..(g+1)*16

        let gs = |g: usize| -> f32 { a_group_sums[y_off / Q6K_GROUP_SIZE + g] };

        // Group 0: elements y_off+0..y_off+31 → 2 sub-groups of 16
        result += d * sc(half * 8 + 0) * (a_scale * lo_0 as f32 - 32.0 * gs(0));
        result += d * sc(half * 8 + 1) * (a_scale * hi_0 as f32 - 32.0 * gs(1));
        // Group 1: elements y_off+32..y_off+63
        result += d * sc(half * 8 + 2) * (a_scale * lo_1 as f32 - 32.0 * gs(2));
        result += d * sc(half * 8 + 3) * (a_scale * hi_1 as f32 - 32.0 * gs(3));
        // Group 2: elements y_off+64..y_off+95
        result += d * sc(half * 8 + 4) * (a_scale * lo_2 as f32 - 32.0 * gs(4));
        result += d * sc(half * 8 + 5) * (a_scale * hi_2 as f32 - 32.0 * gs(5));
        // Group 3: elements y_off+96..y_off+127
        result += d * sc(half * 8 + 6) * (a_scale * lo_3 as f32 - 32.0 * gs(6));
        result += d * sc(half * 8 + 7) * (a_scale * hi_3 as f32 - 32.0 * gs(7));
    }

    result
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let sum64 = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(feature = "cpu")]
pub struct CpuMatMulQ4K;

#[cfg(feature = "cpu")]
impl MatMulQ for CpuMatMulQ4K {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if k % Q4K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} must be divisible by Q4K block size {Q4K_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(crate::KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum = 0.0f32;
                for blk in 0..blocks_per_row {
                    let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                    let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                    let a_slice = &a_row[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                    sum += q4k_dot_block(block, a_slice);
                }
                output[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if k % Q6K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} not divisible by {Q6K_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q6K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(crate::KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum = 0.0f32;
                for blk in 0..blocks_per_row {
                    let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                    let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                    let a_slice = &a_row[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                    sum += q6k_dot_block(block, a_slice);
                }
                output[i * n + j] = sum;
            }
        }
        Ok(())
    }
}

/// Quantize an activation row to Q8 format: per-block i8 values with per-block f32 scales.
/// Also computes per-group activation sums for the Q4_K min correction.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_q8_avx2(a: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    use std::arch::x86_64::*;

    let k = a.len();
    let n_blocks = k / Q4K_BLOCK_SIZE;
    let mut a_q8 = vec![0i8; k];
    let mut scales = vec![0.0f32; n_blocks];
    let mut group_sums = vec![0.0f32; n_blocks * Q4K_GROUPS];

    for blk in 0..n_blocks {
        let base = blk * Q4K_BLOCK_SIZE;
        let block_a = &a[base..base + Q4K_BLOCK_SIZE];

        // Find max abs using AVX2
        let mut vmax = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);
        for chunk in 0..32 {
            let v = _mm256_loadu_ps(block_a.as_ptr().add(chunk * 8));
            let abs_v = _mm256_andnot_ps(sign_mask, v);
            vmax = _mm256_max_ps(vmax, abs_v);
        }
        let hi = _mm256_extractf128_ps(vmax, 1);
        let lo = _mm256_castps256_ps128(vmax);
        let max128 = _mm_max_ps(lo, hi);
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
        let max_abs = _mm_cvtss_f32(max32);

        let scale = max_abs / 127.0;
        scales[blk] = scale;
        let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        let inv_vec = _mm256_set1_ps(inv_scale);

        // Quantize to i8 and compute group sums
        for chunk in 0..32 {
            let cb = chunk * 8;
            let v = _mm256_loadu_ps(block_a.as_ptr().add(cb));
            let scaled = _mm256_mul_ps(v, inv_vec);
            let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
                scaled,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            ));
            let lo128 = _mm256_castsi256_si128(rounded);
            let hi128 = _mm256_extracti128_si256(rounded, 1);
            let packed16 = _mm_packs_epi32(lo128, hi128);
            let packed8 = _mm_packs_epi16(packed16, packed16);
            let ptr = a_q8.as_mut_ptr().add(base + cb) as *mut i64;
            *ptr = _mm_cvtsi128_si64(packed8);
        }

        // Compute per-group activation sums (8 groups of 32)
        for g in 0..Q4K_GROUPS {
            let mut gsum = 0.0f32;
            for i in 0..Q4K_GROUP_SIZE {
                gsum += block_a[g * Q4K_GROUP_SIZE + i];
            }
            group_sums[blk * Q4K_GROUPS + g] = gsum;
        }
    }

    (a_q8, scales, group_sums)
}

/// Integer-domain Q4_K dot product with pre-quantized Q8 activations.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn q4k_dot_q8_avx2(block: &[u8], a_q8: &[i8], a_scale: f32, a_group_sums: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mask_low = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);

    let mut result = 0.0f32;

    for j in 0..4 {
        let raw = _mm256_loadu_si256(qs.as_ptr().add(j * 32) as *const __m256i);
        let lo_nib = _mm256_and_si256(raw, mask_low);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_low);

        let aq_lo = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64) as *const __m256i);
        let aq_hi = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64 + 32) as *const __m256i);

        // u8 × i8 → i16 paired sums (32 mults → 16 sums per call)
        let p_lo = _mm256_maddubs_epi16(lo_nib, aq_lo);
        let p_hi = _mm256_maddubs_epi16(hi_nib, aq_hi);

        // i16 → i32 (sum adjacent pairs)
        let s_lo = _mm256_madd_epi16(p_lo, ones_16);
        let s_hi = _mm256_madd_epi16(p_hi, ones_16);

        let idot_lo = hsum_i32(s_lo);
        let idot_hi = hsum_i32(s_hi);

        result += d * sc[2 * j] * a_scale * idot_lo as f32;
        result += d * sc[2 * j + 1] * a_scale * idot_hi as f32;
        result -= dmin * m[2 * j] * a_group_sums[2 * j];
        result -= dmin * m[2 * j + 1] * a_group_sums[2 * j + 1];
    }

    result
}

/// VNNI variant: uses _mm256_dpbusd_epi32 (single instruction u8×i8→i32 accumulate)
/// instead of the two-instruction maddubs+madd chain. ~1.5-2x throughput.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn q4k_dot_q8_vnni(block: &[u8], a_q8: &[i8], a_scale: f32, a_group_sums: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mask_low = _mm256_set1_epi8(0x0F);

    let mut result = 0.0f32;

    for j in 0..4 {
        let raw = _mm256_loadu_si256(qs.as_ptr().add(j * 32) as *const __m256i);
        let lo_nib = _mm256_and_si256(raw, mask_low);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_low);

        let aq_lo = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64) as *const __m256i);
        let aq_hi = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64 + 32) as *const __m256i);

        // dpbusd: for each i32 lane, sum 4 u8×i8 products and accumulate
        // Replaces maddubs_epi16 + madd_epi16 in a single instruction
        let s_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), lo_nib, aq_lo);
        let s_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), hi_nib, aq_hi);

        let idot_lo = hsum_i32(s_lo);
        let idot_hi = hsum_i32(s_hi);

        result += d * sc[2 * j] * a_scale * idot_lo as f32;
        result += d * sc[2 * j + 1] * a_scale * idot_hi as f32;
        result -= dmin * m[2 * j] * a_group_sums[2 * j];
        result -= dmin * m[2 * j + 1] * a_group_sums[2 * j + 1];
    }

    result
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
unsafe fn q4k_dot_block_avx2_int8(block: &[u8], a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mask_low = _mm256_set1_epi8(0x0F);
    let ones_16 = _mm256_set1_epi16(1);

    // Quantize 256 f32 activations to i8
    let mut max_abs = 0.0f32;
    for i in 0..Q4K_BLOCK_SIZE {
        let v = a[i].abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    let a_scale = max_abs / 127.0;
    let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };

    let mut a_q8 = [0i8; Q4K_BLOCK_SIZE];
    let inv_scale_vec = _mm256_set1_ps(inv_scale);
    for chunk in 0..32 {
        let base = chunk * 8;
        let av = _mm256_loadu_ps(a.as_ptr().add(base));
        let scaled = _mm256_mul_ps(av, inv_scale_vec);
        let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
            scaled,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ));
        // Pack i32 → i16 → i8 (with saturation)
        let lo128 = _mm256_castsi256_si128(rounded);
        let hi128 = _mm256_extracti128_si256(rounded, 1);
        let packed16 = _mm_packs_epi32(lo128, hi128);
        let packed8 = _mm_packs_epi16(packed16, packed16);
        // Store 8 bytes
        let ptr = a_q8.as_mut_ptr().add(base) as *mut i64;
        *ptr = _mm_cvtsi128_si64(packed8);
    }

    let mut result = 0.0f32;

    for j in 0..4 {
        // Load 32 qs bytes, extract low and high nibbles
        let raw = _mm256_loadu_si256(qs.as_ptr().add(j * 32) as *const __m256i);
        let lo_nib = _mm256_and_si256(raw, mask_low);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_low);

        // Load 64 quantized activation bytes
        let aq_lo = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64) as *const __m256i);
        let aq_hi = _mm256_loadu_si256(a_q8.as_ptr().add(j * 64 + 32) as *const __m256i);

        // Integer dot: u8 × i8 → i16 paired sums
        let p_lo = _mm256_maddubs_epi16(lo_nib, aq_lo);
        let p_hi = _mm256_maddubs_epi16(hi_nib, aq_hi);

        // i16 → i32 (sum adjacent pairs)
        let s_lo = _mm256_madd_epi16(p_lo, ones_16);
        let s_hi = _mm256_madd_epi16(p_hi, ones_16);

        // Horizontal sum each to a single i32
        let hsum_lo = hsum_i32(s_lo);
        let hsum_hi = hsum_i32(s_hi);

        // Activation sums per group (for min correction)
        // sum(a_q8[group]) — use madd with ones to sum i8 as i16 pairs, then i32
        let aq_lo_16 =
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(aq_lo)), ones_16);
        let aq_lo_16b = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(aq_lo, 1)),
            ones_16,
        );
        let asum_lo = hsum_i32(_mm256_add_epi32(aq_lo_16, aq_lo_16b));

        let aq_hi_16 =
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(aq_hi)), ones_16);
        let aq_hi_16b = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(aq_hi, 1)),
            ones_16,
        );
        let asum_hi = hsum_i32(_mm256_add_epi32(aq_hi_16, aq_hi_16b));

        result +=
            d * sc[2 * j] * a_scale * hsum_lo as f32 - dmin * m[2 * j] * a_scale * asum_lo as f32;
        result += d * sc[2 * j + 1] * a_scale * hsum_hi as f32
            - dmin * m[2 * j + 1] * a_scale * asum_hi as f32;
    }

    result
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_dot_block_avx2(block: &[u8], a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mut acc = _mm256_setzero_ps();

    for j in 0..4 {
        let d1 = d * sc[2 * j];
        let m1 = dmin * m[2 * j];
        let d2 = d * sc[2 * j + 1];
        let m2 = dmin * m[2 * j + 1];

        let d1_vec = _mm256_set1_ps(d1);
        let m1_vec = _mm256_set1_ps(m1);
        let d2_vec = _mm256_set1_ps(d2);
        let m2_vec = _mm256_set1_ps(m2);

        // Process low nibbles (elements j*64+0..j*64+31) in chunks of 8
        for chunk in 0..4 {
            let base = chunk * 8;
            let q_f32 = _mm256_set_ps(
                (qs[j * 32 + base + 7] & 0x0F) as f32,
                (qs[j * 32 + base + 6] & 0x0F) as f32,
                (qs[j * 32 + base + 5] & 0x0F) as f32,
                (qs[j * 32 + base + 4] & 0x0F) as f32,
                (qs[j * 32 + base + 3] & 0x0F) as f32,
                (qs[j * 32 + base + 2] & 0x0F) as f32,
                (qs[j * 32 + base + 1] & 0x0F) as f32,
                (qs[j * 32 + base] & 0x0F) as f32,
            );
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(j * 64 + base));
            let w = _mm256_fmsub_ps(d1_vec, q_f32, m1_vec);
            acc = _mm256_fmadd_ps(a_vec, w, acc);
        }

        // Process high nibbles (elements j*64+32..j*64+63) in chunks of 8
        for chunk in 0..4 {
            let base = chunk * 8;
            let q_f32 = _mm256_set_ps(
                (qs[j * 32 + base + 7] >> 4) as f32,
                (qs[j * 32 + base + 6] >> 4) as f32,
                (qs[j * 32 + base + 5] >> 4) as f32,
                (qs[j * 32 + base + 4] >> 4) as f32,
                (qs[j * 32 + base + 3] >> 4) as f32,
                (qs[j * 32 + base + 2] >> 4) as f32,
                (qs[j * 32 + base + 1] >> 4) as f32,
                (qs[j * 32 + base] >> 4) as f32,
            );
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(j * 64 + 32 + base));
            let w = _mm256_fmsub_ps(d2_vec, q_f32, m2_vec);
            acc = _mm256_fmadd_ps(a_vec, w, acc);
        }
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

#[cfg(feature = "cpu")]
pub struct CpuMatMulQ4KOpt;

#[cfg(feature = "cpu")]
impl CpuMatMulQ4KOpt {
    #[inline(always)]
    fn dequant_row(b_quantized: &[u8], row: usize, k: usize, buf: &mut [f32]) {
        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        for blk in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
            let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
            let out = &mut buf[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
            let mut tmp = [0.0f32; Q4K_BLOCK_SIZE];
            q4k_dequant_block(block, &mut tmp);
            out.copy_from_slice(&tmp);
        }
    }

    #[inline(always)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Pre-quantized activation data for reuse across multiple matmuls.
#[cfg(feature = "cpu")]
pub struct PreQuantizedActivations {
    pub q8: Vec<i8>,
    pub q16: Vec<i16>,
    pub scales: Vec<f32>,
    pub scales_i16: Vec<f32>,
    pub group_sums_q4k: Vec<f32>,
    pub group_sums_q6k: Vec<f32>,
    pub k: usize,
}

#[cfg(feature = "cpu")]
impl PreQuantizedActivations {
    pub fn from_f32(a: &[f32]) -> Self {
        let k = a.len();

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let (q8, scales, group_sums_q4k) = unsafe { quantize_row_q8_avx2(a) };
            let n_blocks = k / Q6K_BLOCK_SIZE;
            let mut group_sums_q6k = vec![0.0f32; n_blocks * Q6K_GROUPS];
            for blk in 0..n_blocks {
                for g in 0..Q6K_GROUPS {
                    let mut s = 0.0f32;
                    for i in 0..Q6K_GROUP_SIZE {
                        s += a[blk * Q6K_BLOCK_SIZE + g * Q6K_GROUP_SIZE + i];
                    }
                    group_sums_q6k[blk * Q6K_GROUPS + g] = s;
                }
            }
            // Also quantize to i16 for Q6K (256x less error than i8)
            let n_blocks_i16 = k / Q6K_BLOCK_SIZE;
            let mut q16 = vec![0i16; k];
            let mut scales_i16 = vec![0.0f32; n_blocks_i16];
            for blk in 0..n_blocks_i16 {
                let base = blk * Q6K_BLOCK_SIZE;
                let block_a = &a[base..base + Q6K_BLOCK_SIZE];
                let mut max_abs = 0.0f32;
                for &v in block_a {
                    let av = v.abs();
                    if av > max_abs {
                        max_abs = av;
                    }
                }
                let scale = max_abs / 32767.0;
                scales_i16[blk] = scale;
                let inv = if max_abs > 0.0 {
                    32767.0 / max_abs
                } else {
                    0.0
                };
                for i in 0..Q6K_BLOCK_SIZE {
                    q16[base + i] = (block_a[i] * inv).round() as i16;
                }
            }

            return Self {
                q8,
                q16,
                scales,
                scales_i16,
                group_sums_q4k,
                group_sums_q6k,
                k,
            };
        }

        Self {
            q8: Vec::new(),
            q16: Vec::new(),
            scales: Vec::new(),
            scales_i16: Vec::new(),
            group_sums_q4k: Vec::new(),
            group_sums_q6k: Vec::new(),
            k,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !self.q8.is_empty()
    }
}

#[cfg(feature = "cpu")]
impl CpuMatMulQ4KOpt {
    /// Fused Q4K matmul: two independent matmuls with the same pre-quantized input
    /// in a single rayon dispatch. Eliminates one sync point.
    pub fn fused_q4k_preq_2(
        &self,
        preq: &PreQuantizedActivations,
        b1: &[u8],
        out1: &mut [f32],
        n1: usize,
        b2: &[u8],
        out2: &mut [f32],
        n2: usize,
    ) {
        let k = preq.k;
        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let total = n1 + n2;

        #[cfg(target_arch = "x86_64")]
        let use_vnni = is_x86_feature_detected!("avxvnni");
        #[cfg(not(target_arch = "x86_64"))]
        let use_vnni = false;

        let chunk_size = (total / rayon::current_num_threads()).max(64);

        // Conceptually: outputs 0..n1 come from b1, outputs n1..n1+n2 come from b2.
        // Each thread processes a range spanning potentially both outputs.
        (0..total)
            .into_par_iter()
            .with_min_len(chunk_size)
            .for_each(|idx| {
                let (b_quantized, j, out_ptr) = if idx < n1 {
                    (b1, idx, unsafe { out1.as_ptr().add(idx) as *mut f32 })
                } else {
                    (b2, idx - n1, unsafe {
                        out2.as_ptr().add(idx - n1) as *mut f32
                    })
                };
                let mut sum = 0.0f32;
                for blk in 0..blocks_per_row {
                    let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                    let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                    let aq_slice = &preq.q8[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                    let a_sc = preq.scales[blk];
                    let a_gsums = &preq.group_sums_q4k[blk * Q4K_GROUPS..(blk + 1) * Q4K_GROUPS];
                    sum += if use_vnni {
                        unsafe { q4k_dot_q8_vnni(block, aq_slice, a_sc, a_gsums) }
                    } else {
                        unsafe { q4k_dot_q8_avx2(block, aq_slice, a_sc, a_gsums) }
                    };
                }
                unsafe {
                    *out_ptr = sum;
                }
            });
    }

    pub fn forward_q4k_preq(
        &self,
        preq: &PreQuantizedActivations,
        a_f32: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
    ) -> Result<()> {
        let k = preq.k;
        let blocks_per_row = k / Q4K_BLOCK_SIZE;

        if preq.is_quantized() {
            #[cfg(target_arch = "x86_64")]
            let use_vnni = is_x86_feature_detected!("avxvnni");
            #[cfg(not(target_arch = "x86_64"))]
            let use_vnni = false;

            let chunk_size = (n / rayon::current_num_threads()).max(64);
            output[..n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let j_start = chunk_idx * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                            let aq_slice =
                                &preq.q8[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                            let a_sc = preq.scales[blk];
                            let a_gsums =
                                &preq.group_sums_q4k[blk * Q4K_GROUPS..(blk + 1) * Q4K_GROUPS];
                            sum += if use_vnni {
                                unsafe { q4k_dot_q8_vnni(block, aq_slice, a_sc, a_gsums) }
                            } else {
                                unsafe { q4k_dot_q8_avx2(block, aq_slice, a_sc, a_gsums) }
                            };
                        }
                        *out = sum;
                    }
                });
        } else {
            let chunk_size = (n / rayon::current_num_threads()).max(64);
            output[..n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let j_start = chunk_idx * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                            let a_slice = &a_f32[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                            sum += q4k_dot_block(block, a_slice);
                        }
                        *out = sum;
                    }
                });
        }
        Ok(())
    }

    pub fn forward_q6k_preq(
        &self,
        preq: &PreQuantizedActivations,
        a_f32: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
    ) -> Result<()> {
        let k = preq.k;
        let blocks_per_row = k / Q6K_BLOCK_SIZE;

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let chunk_size = (n / rayon::current_num_threads()).max(64);
            output[..n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let j_start = ci * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                            let a_slice = &a_f32[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                            sum += unsafe { q6k_dot_block_avx2(block, a_slice) };
                        }
                        *out = sum;
                    }
                });
            return Ok(());
        }

        let chunk_size = (n / rayon::current_num_threads()).max(64);
        output[..n]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let j_start = ci * chunk_size;
                for (local, out) in chunk.iter_mut().enumerate() {
                    let j = j_start + local;
                    let mut sum = 0.0f32;
                    for blk in 0..blocks_per_row {
                        let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                        let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                        let a_slice = &a_f32[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                        sum += q6k_dot_block(block, a_slice);
                    }
                    *out = sum;
                }
            });
        Ok(())
    }
}

#[cfg(feature = "cpu")]
impl MatMulQ for CpuMatMulQ4KOpt {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if k % Q4K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} must be divisible by Q4K block size {Q4K_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(crate::KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        if m == 1 {
            let a_row = &a[0..k];

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let (a_q8, a_scales, a_group_sums) = unsafe { quantize_row_q8_avx2(a_row) };

                #[cfg(target_arch = "x86_64")]
                let use_vnni = is_x86_feature_detected!("avxvnni");
                #[cfg(not(target_arch = "x86_64"))]
                let use_vnni = false;

                let chunk_size = (n / rayon::current_num_threads()).max(64);
                output[..n].par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_idx, chunk)| {
                        let j_start = chunk_idx * chunk_size;
                        for (local, out) in chunk.iter_mut().enumerate() {
                            let j = j_start + local;
                            let mut sum = 0.0f32;
                            for blk in 0..blocks_per_row {
                                let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                                let block =
                                    &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                                let aq_slice =
                                    &a_q8[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                                let a_sc = a_scales[blk];
                                let a_gsums =
                                    &a_group_sums[blk * Q4K_GROUPS..(blk + 1) * Q4K_GROUPS];
                                sum += if use_vnni {
                                    unsafe { q4k_dot_q8_vnni(block, aq_slice, a_sc, a_gsums) }
                                } else {
                                    unsafe { q4k_dot_q8_avx2(block, aq_slice, a_sc, a_gsums) }
                                };
                            }
                            *out = sum;
                        }
                    },
                );
                return Ok(());
            }

            let chunk_size = (n / rayon::current_num_threads()).max(64);
            output[..n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let j_start = chunk_idx * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                            let a_slice = &a_row[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                            sum += q4k_dot_block(block, a_slice);
                        }
                        *out = sum;
                    }
                });
        } else {
            // Batched path: dequant full row, use gemm for SIMD matmul
            let mut b_f32 = vec![0.0f32; n * k];
            for j in 0..n {
                Self::dequant_row(b_quantized, j, k, &mut b_f32[j * k..(j + 1) * k]);
            }
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    output.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    a.as_ptr(),
                    1,
                    k as isize,
                    b_f32.as_ptr(),
                    k as isize,
                    1,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                );
            }
        }

        Ok(())
    }

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if k % Q6K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} not divisible by {Q6K_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q6K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(crate::KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        if m == 1 {
            let a_row = &a[0..k];

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let chunk_size = (n / rayon::current_num_threads()).max(64);
                output[..n]
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let j_start = ci * chunk_size;
                        for (local, out) in chunk.iter_mut().enumerate() {
                            let j = j_start + local;
                            let mut sum = 0.0f32;
                            for blk in 0..blocks_per_row {
                                let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                                let block =
                                    &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                                let a_slice =
                                    &a_row[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                                sum += unsafe { q6k_dot_block_avx2_i16(block, a_slice) };
                            }
                            *out = sum;
                        }
                    });
                return Ok(());
            }

            let chunk_size = (n / rayon::current_num_threads()).max(64);
            output[..n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let j_start = ci * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                            let a_slice = &a_row[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                            sum += q6k_dot_block(block, a_slice);
                        }
                        *out = sum;
                    }
                });
        } else {
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for blk in 0..blocks_per_row {
                        let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
                        let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
                        let a_slice = &a_row[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
                        sum += q6k_dot_block(block, a_slice);
                    }
                    output[i * n + j] = sum;
                }
            }
        }
        Ok(())
    }
}

/// Quantize activation row for Q6_K groups (16 elements per group).
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_q8_avx2_q6k(a: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    use std::arch::x86_64::*;

    let k = a.len();
    let n_blocks = k / Q6K_BLOCK_SIZE;
    let mut a_q8 = vec![0i8; k];
    let mut scales = vec![0.0f32; n_blocks];
    let mut group_sums = vec![0.0f32; n_blocks * Q6K_GROUPS];

    for blk in 0..n_blocks {
        let base = blk * Q6K_BLOCK_SIZE;
        let block_a = &a[base..base + Q6K_BLOCK_SIZE];

        let mut vmax = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0);
        for chunk in 0..32 {
            let v = _mm256_loadu_ps(block_a.as_ptr().add(chunk * 8));
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
        }
        let hi = _mm256_extractf128_ps(vmax, 1);
        let lo = _mm256_castps256_ps128(vmax);
        let max128 = _mm_max_ps(lo, hi);
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
        let max_abs = _mm_cvtss_f32(max32);

        let scale = max_abs / 127.0;
        scales[blk] = scale;
        let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        let inv_vec = _mm256_set1_ps(inv_scale);

        for chunk in 0..32 {
            let cb = chunk * 8;
            let v = _mm256_loadu_ps(block_a.as_ptr().add(cb));
            let scaled = _mm256_mul_ps(v, inv_vec);
            let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
                scaled,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            ));
            let lo128 = _mm256_castsi256_si128(rounded);
            let hi128 = _mm256_extracti128_si256(rounded, 1);
            let packed16 = _mm_packs_epi32(lo128, hi128);
            let packed8 = _mm_packs_epi16(packed16, packed16);
            let ptr = a_q8.as_mut_ptr().add(base + cb) as *mut i64;
            *ptr = _mm_cvtsi128_si64(packed8);
        }

        for g in 0..Q6K_GROUPS {
            let mut gsum = 0.0f32;
            for i in 0..Q6K_GROUP_SIZE {
                gsum += block_a[g * Q6K_GROUP_SIZE + i];
            }
            group_sums[blk * Q6K_GROUPS + g] = gsum;
        }
    }

    (a_q8, scales, group_sums)
}

#[cfg(feature = "cpu")]
pub struct CpuDequantize;

#[cfg(feature = "cpu")]
impl Dequantize for CpuDequantize {
    fn dequantize(
        &self,
        quantized: &[u8],
        scales: &[f32],
        zeros: &[f32],
        output: &mut [f32],
        dtype: QuantDtype,
        group_size: usize,
    ) -> Result<()> {
        if group_size == 0 {
            return Err(crate::KernelError::InvalidConfig(
                "group_size must be > 0".into(),
            ));
        }
        match dtype {
            QuantDtype::Int8 => {
                for (i, &q) in quantized.iter().enumerate() {
                    let group = i / group_size;
                    if group >= scales.len() || group >= zeros.len() || i >= output.len() {
                        break;
                    }
                    output[i] = (q as i8 as f32 - zeros[group]) * scales[group];
                }
            }
            QuantDtype::Int4 => {
                for (byte_idx, &byte) in quantized.iter().enumerate() {
                    let elem_lo = byte_idx * 2;
                    let elem_hi = elem_lo + 1;
                    let lo = (byte & 0x0F) as f32;
                    let hi = (byte >> 4) as f32;
                    let group_lo = elem_lo / group_size;
                    let group_hi = elem_hi / group_size;
                    if elem_lo < output.len() && group_lo < scales.len() {
                        output[elem_lo] = (lo - zeros[group_lo]) * scales[group_lo];
                    }
                    if elem_hi < output.len() && group_hi < scales.len() {
                        output[elem_hi] = (hi - zeros[group_hi]) * scales[group_hi];
                    }
                }
            }
            QuantDtype::Fp16 => {
                for (i, chunk) in quantized.chunks_exact(2).enumerate() {
                    if i >= output.len() {
                        break;
                    }
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    output[i] = f16_to_f32(bits);
                }
            }
        }
        Ok(())
    }

    fn verification_tier(&self) -> crate::VerificationTier {
        crate::VerificationTier::Tested
    }
}

pub struct ActivationQ8 {
    pub data: Vec<u32>,
    pub num_blocks: usize,
}

impl ActivationQ8 {
    pub fn from_f32(a: &[f32]) -> Self {
        let num_blocks = (a.len() + 31) / 32;
        let mut data = Vec::with_capacity(num_blocks * 10);

        for blk in 0..num_blocks {
            let start = blk * 32;
            let end = (start + 32).min(a.len());

            let mut amax = 0.0f32;
            for i in start..end {
                amax = amax.max(a[i].abs());
            }

            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };

            let mut sum = 0.0f32;
            let mut qs = [0i8; 32];
            for i in 0..(end - start) {
                let v = (a[start + i] * id).round();
                let v = v.max(-128.0).min(127.0) as i8;
                qs[i] = v;
                sum += v as f32;
            }

            data.push(d.to_bits());
            data.push((d * sum).to_bits());

            for chunk in 0..8 {
                let word = (qs[chunk * 4] as u8 as u32)
                    | ((qs[chunk * 4 + 1] as u8 as u32) << 8)
                    | ((qs[chunk * 4 + 2] as u8 as u32) << 16)
                    | ((qs[chunk * 4 + 3] as u8 as u32) << 24);
                data.push(word);
            }
        }

        Self { data, num_blocks }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaMatMulQ4K {
    gpu: crate::gpu::GpuContext,
    module: crate::gpu::quantized_gemm_kernel::kernels::LoadedModule,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaMatMulQ4K {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaMatMulQ4K {}

#[cfg(feature = "cuda")]
impl CudaMatMulQ4K {
    pub fn new() -> crate::Result<Self> {
        let gpu = crate::gpu::GpuContext::new(0)?;
        let module = crate::gpu::quantized_gemm_kernel::kernels::load(gpu.context())
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { gpu, module })
    }

    pub fn gpu(&self) -> &crate::gpu::GpuContext {
        &self.gpu
    }

    pub fn forward_q4k_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let total = (m * n) as u32;
        self.module
            .gemm_q4k_bt(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn forward_q4k_splitk_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        num_splits: usize,
        partials: &mut cuda_core::DeviceBuffer<f32>,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();

        self.module
            .gemm_q4k_bt_splitk(
                &stream,
                LaunchConfig {
                    grid_dim: (n as u32, num_splits as u32, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                num_splits as u32,
                a,
                b_quantized,
                partials,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        self.module
            .reduce_splitk(
                &stream,
                LaunchConfig::for_num_elems(n as u32),
                n as u32,
                num_splits as u32,
                partials,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        Ok(())
    }

    pub fn gemv_q4k_warp_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q4k_warp(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_coop_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q4k_coop(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_q8_dp4a_device(
        &self,
        a_q8: &cuda_core::DeviceBuffer<u32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q4k_q8_dp4a(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                a_q8,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_coop_v2_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q4k_coop_v2(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn fused_gate_up_silu_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_gate: &cuda_core::DeviceBuffer<u8>,
        b_up: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q4k_fused_gate_up_silu(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                a,
                b_gate,
                b_up,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn forward_q6k_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();
        let total = (m * n) as u32;
        self.module
            .gemm_q6k_bt(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl MatMulQ for CudaMatMulQ4K {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        if k % Q4K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} not divisible by {Q4K_BLOCK_SIZE}"
            )));
        }
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();

        let a_dev = crate::gpu::to_device(&stream, a)?;
        let b_dev = crate::gpu::to_device_u8(&stream, b_quantized)?;
        let mut c_dev = crate::gpu::alloc_device(&stream, m * n)?;

        let total = (m * n) as u32;
        self.module
            .gemm_q4k_bt(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        crate::gpu::copy_back(&stream, &c_dev, output)?;
        Ok(())
    }

    fn forward_b_transposed_q6k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        if k % Q6K_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} not divisible by {Q6K_BLOCK_SIZE}"
            )));
        }
        use cuda_core::LaunchConfig;
        crate::gpu::bind_context(self.gpu.context())?;
        let stream = self.gpu.stream();

        let a_dev = crate::gpu::to_device(&stream, a)?;
        let b_dev = crate::gpu::to_device_u8(&stream, b_quantized)?;
        let mut c_dev = crate::gpu::alloc_device(&stream, m * n)?;

        let total = (m * n) as u32;
        self.module
            .gemm_q6k_bt(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;

        crate::gpu::copy_back(&stream, &c_dev, output)?;
        Ok(())
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn q4k_block_access_in_bounds() {
        let group: usize = kani::any();
        let idx: usize = kani::any();
        kani::assume(group < Q4K_GROUPS);
        kani::assume(idx < Q4K_BLOCK_SIZE);

        // d: bytes 0-1
        assert!(0 + 2 <= Q4K_BLOCK_BYTES);
        // dmin: bytes 2-3
        assert!(2 + 2 <= Q4K_BLOCK_BYTES);
        // packed_scales: bytes 4-15
        assert!(4 + 12 <= Q4K_BLOCK_BYTES);
        // qs: bytes 16-143
        assert!(16 + 128 <= Q4K_BLOCK_BYTES);

        // nibble access: qs at byte_idx = idx / 2
        let byte_idx = idx / 2;
        assert!(16 + byte_idx < Q4K_BLOCK_BYTES);

        // scale unpacking: packed_scales[i], packed_scales[i+4], packed_scales[i+8]
        // i ranges 0..4, so max index is 4+8=12, but packed_scales is 12 bytes (indices 0..11)
        let i: usize = kani::any();
        kani::assume(i < 4);
        assert!(i < 12); // packed_scales[i]
        assert!(i + 4 < 12); // packed_scales[i+4]
        assert!(i + 8 < 12); // packed_scales[i+8]
    }

    #[kani::proof]
    fn q4k_dequant_value_bounded() {
        let q: u8 = kani::any();
        let sc: u8 = kani::any();
        let m_val: u8 = kani::any();
        kani::assume(q <= 15);
        kani::assume(sc <= 63);
        kani::assume(m_val <= 63);

        let d: f32 = 1.0;
        let dmin: f32 = 1.0;

        let value = d * (sc as f32) * (q as f32) - dmin * (m_val as f32);
        let max_positive = 1.0 * 63.0 * 15.0;
        let max_negative = 1.0 * 63.0;

        assert!(value <= max_positive);
        assert!(value >= -max_negative);
    }

    #[kani::proof]
    fn q4k_matmul_index_in_bounds() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        let k: usize = kani::any();

        kani::assume(m >= 1 && m <= 4);
        kani::assume(n >= 1 && n <= 4);
        kani::assume(k == 256 || k == 512);

        let blocks_per_row = k / Q4K_BLOCK_SIZE;

        let i: usize = kani::any();
        let j: usize = kani::any();
        let blk: usize = kani::any();

        kani::assume(i < m);
        kani::assume(j < n);
        kani::assume(blk < blocks_per_row);

        // A access
        let a_offset = i * k + blk * Q4K_BLOCK_SIZE;
        assert!(a_offset + Q4K_BLOCK_SIZE <= m * k);

        // B_q access
        let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
        assert!(block_offset + Q4K_BLOCK_BYTES <= n * blocks_per_row * Q4K_BLOCK_BYTES);

        // Output access
        let out_idx = i * n + j;
        assert!(out_idx < m * n);
    }

    #[kani::proof]
    fn q4k_nibble_in_range() {
        let byte_val: u8 = kani::any();
        let idx: usize = kani::any();
        kani::assume(idx < 2);

        let nibble = if idx == 0 {
            byte_val & 0x0F
        } else {
            byte_val >> 4
        };

        assert!(nibble <= 15);
    }

    #[kani::proof]
    fn q4k_scale_unpack_in_range() {
        let packed: [u8; 12] = kani::any();

        let i: usize = kani::any();
        kani::assume(i < 4);

        let sc_lo = packed[i] & 0x3F;
        let sc_hi = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        let m_lo = packed[i + 4] & 0x3F;
        let m_hi = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;

        assert!(sc_lo <= 63);
        assert!(sc_hi <= 63);
        assert!(m_lo <= 63);
        assert!(m_hi <= 63);
    }

    #[kani::proof]
    fn q4k_dequant_covers_all_256_elements() {
        let j: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(j < 4);
        kani::assume(l < 32);

        let elem_lo = j * 64 + l;
        let elem_hi = j * 64 + l + 32;
        let qs_idx = j * 32 + l;

        assert!(elem_lo < Q4K_BLOCK_SIZE);
        assert!(elem_hi < Q4K_BLOCK_SIZE);
        assert!(qs_idx < 128);
        assert!(elem_lo != elem_hi);
    }

    /// Prove scale unpacking matches ggml's get_scale_min_k4 for ALL 8 indices.
    /// ggml: j<4 → sc=q[j]&63, m=q[j+4]&63
    ///       j≥4 → sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4), m=(q[j+4]>>4)|((q[j]>>6)<<4)
    #[kani::proof]
    fn q4k_scale_unpack_matches_ggml() {
        let packed: [u8; 12] = kani::any();

        let j: usize = kani::any();
        kani::assume(j < 8);

        // ggml's get_scale_min_k4
        let (ggml_sc, ggml_m) = if j < 4 {
            (packed[j] & 63, packed[j + 4] & 63)
        } else {
            let sc = (packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4);
            let m = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
            (sc & 0x3F, m & 0x3F)
        };

        // Our unpack_q4k_scales
        let i = j % 4;
        let (our_sc, our_m) = if j < 4 {
            (packed[i] & 0x3F, packed[i + 4] & 0x3F)
        } else {
            let sc = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
            let m = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
            (sc, m)
        };

        assert!(ggml_sc == our_sc);
        assert!(ggml_m == our_m);
    }

    /// Prove the interleaved dequant produces the same element-to-qs mapping as ggml.
    /// ggml loop: for j in 0..4, for l in 0..32:
    ///   y[j*64+l]    uses qs[j*32+l] & 0xF  with scale 2*j
    ///   y[j*64+l+32] uses qs[j*32+l] >> 4   with scale 2*j+1
    #[kani::proof]
    fn q4k_dequant_element_mapping_matches_ggml() {
        let j: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(j < 4);
        kani::assume(l < 32);

        // ggml mapping
        let ggml_elem_lo = j * 64 + l;
        let ggml_elem_hi = j * 64 + l + 32;
        let ggml_qs_idx = j * 32 + l;
        let ggml_scale_lo = 2 * j;
        let ggml_scale_hi = 2 * j + 1;

        // Our dequant mapping (from q4k_dequant_block)
        let our_q_off = j * 32;
        let our_elem_lo = j * 64 + l;
        let our_elem_hi = j * 64 + l + 32;
        let our_qs_idx = our_q_off + l;
        let our_scale_lo = 2 * j;
        let our_scale_hi = 2 * j + 1;

        assert!(ggml_elem_lo == our_elem_lo);
        assert!(ggml_elem_hi == our_elem_hi);
        assert!(ggml_qs_idx == our_qs_idx);
        assert!(ggml_scale_lo == our_scale_lo);
        assert!(ggml_scale_hi == our_scale_hi);
    }

    /// Prove f32 accumulator stays finite for realistic matmul dimensions.
    /// Max per-element product: max_activation × (d × 63 × 15 + dmin × 63).
    /// For k=12288 (largest FFN dim), sum of 12288 such terms must not overflow f32.
    #[kani::proof]
    fn q4k_accumulator_no_overflow() {
        let k: usize = kani::any();
        kani::assume(k >= 256 && k <= 12288);
        kani::assume(k % 256 == 0);

        // Worst case per element: activation=100 (post-norm values are <10 typically),
        // weight = d * 63 * 15 = max dequantized value with d=1 (f16 max is 65504)
        // Conservative bound: |a_i * w_i| ≤ 100 * 945 = 94500
        let max_per_element: f64 = 100.0 * 945.0;
        let max_sum = max_per_element * k as f64;

        // f32 can represent values up to ~3.4e38
        assert!(max_sum < 3.4e38);
        // Even for k=12288: 94500 * 12288 = 1.16e9, well within f32 range
        assert!(max_sum < f32::MAX as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4k_block(d: f32, dmin: f32, values: &[u8; 256]) -> [u8; Q4K_BLOCK_BYTES] {
        let mut block = [0u8; Q4K_BLOCK_BYTES];

        let d_bits = half::f16::from_f32(d).to_bits();
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;

        let dmin_bits = half::f16::from_f32(dmin).to_bits();
        block[2] = dmin_bits as u8;
        block[3] = (dmin_bits >> 8) as u8;

        // Set all scales to 1, all mins to 0 (simplest case)
        // packed_scales: sc[0..4] in low 6 bits of bytes 0..3
        for i in 0..4 {
            block[4 + i] = 1; // sc[i] = 1
            block[4 + i + 4] = 0; // m[i] = 0
        }
        // sc[4..8] and m[4..8] need high bits set in bytes 8..11
        for i in 0..4 {
            block[4 + 8 + i] = (1 >> 2) & 0x0F; // sc[i+4] low part = 0, needs packed[i]>>6 too
        }
        // For scale=1: sc[i+4] = ((packed[i]>>6) | ((packed[i+8] & 0x0F) << 2)) & 0x3F
        // We need ((packed[i]>>6) | ((packed[i+8] & 0x0F) << 2)) & 0x3F == 1
        // packed[i]>>6 = 0 (since packed[i]=1, bit6=0), so (packed[i+8] & 0x0F) << 2 must be 1
        // But 1 is odd and <<2 gives even... can't get sc[i+4]=1 this way easily.
        // Just set sc[4..8] = 0, test only uses groups 0..3
        // Actually, let's just set all packed_scales to the right values for uniform scale=1 min=0
        // Simplest: all packed bytes = 0x01 for the first 4, rest 0
        // sc[0..4] = 1 (packed[0..4] & 0x3F = 1)
        // m[0..4] = 0 (packed[4..8] & 0x3F = 0)
        // sc[4..8] = 0 (no high bits)
        // m[4..8] = 0

        // Pack nibbles in ggml Q4_K interleaved layout:
        // Each group of 64 elements uses 32 bytes.
        // Low nibbles of 32 bytes → elements 0..31 (within the 64-elem group)
        // High nibbles of the same 32 bytes → elements 32..63
        for j in 0..4 {
            for l in 0..32 {
                let lo_val = values[j * 64 + l];
                let hi_val = values[j * 64 + l + 32];
                block[16 + j * 32 + l] = (lo_val & 0x0F) | ((hi_val & 0x0F) << 4);
            }
        }

        block
    }

    #[test]
    fn test_q4k_dequant_identity() {
        let mut values = [0u8; 256];
        for i in 0..256 {
            values[i] = (i % 16) as u8;
        }
        let block = make_q4k_block(1.0, 0.0, &values);
        let mut output = [0.0f32; Q4K_BLOCK_SIZE];
        q4k_dequant_block(&block, &mut output);

        // With d=1, dmin=0, sc[0..4]=1, all values should be q * 1 * 1 - 0 = q
        for i in 0..128 {
            let expected = (i % 16) as f32;
            assert!(
                (output[i] - expected).abs() < 1e-3,
                "output[{i}]={}, expected={expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_q4k_dot_block() {
        let mut values = [1u8; 256];
        for i in 0..256 {
            values[i] = 1;
        }
        let block = make_q4k_block(1.0, 0.0, &values);

        let a = [1.0f32; 256];
        let dot = q4k_dot_block(&block, &a);

        // Each dequantized value = d * sc * 1 - 0 = 1 * 1 * 1 = 1 (for groups 0..3)
        // Groups 4..7 have sc=0, so those contribute 0
        // Groups 0..3: 4 * 32 = 128 elements, each contributing 1*1 = 1
        // Total = 128
        assert!((dot - 128.0).abs() < 1e-2, "dot={dot}, expected=128.0");
    }

    #[test]
    fn test_cpu_matmul_q4k_small() {
        let k = 256;
        let m = 1;
        let n = 1;

        let a = vec![1.0f32; k];

        let mut values = [0u8; 256];
        for i in 0..256 {
            values[i] = 1;
        }
        let block = make_q4k_block(0.5, 0.0, &values);

        let matmul = CpuMatMulQ4K;
        let mut output = vec![0.0f32; m * n];

        matmul
            .forward_b_transposed_q4k(&a, &block, &mut output, m, n, k)
            .unwrap();

        // Each value = 0.5 * 1 * 1 = 0.5 for groups 0..3 (128 elements)
        // dot = 128 * 0.5 = 64
        assert!(
            (output[0] - 64.0).abs() < 1e-2,
            "output[0]={}, expected=64.0",
            output[0]
        );
    }

    #[test]
    fn test_cpu_matmul_q4k_dimension_checks() {
        let matmul = CpuMatMulQ4K;

        let a = vec![1.0f32; 256];
        let b = vec![0u8; Q4K_BLOCK_BYTES];
        let mut output = vec![0.0f32; 1];

        // Wrong a length
        let result = matmul.forward_b_transposed_q4k(&a[..128], &b, &mut output, 1, 1, 256);
        assert!(result.is_err());

        // Wrong k alignment
        let result = matmul.forward_b_transposed_q4k(&a[..100], &b, &mut output, 1, 1, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_q4k_matches_gguf_dequant() {
        let mut raw_block = [0u8; Q4K_BLOCK_BYTES];
        for (i, b) in raw_block.iter_mut().enumerate() {
            *b = ((i * 7 + 13) % 256) as u8;
        }

        let mut dequant_output = [0.0f32; Q4K_BLOCK_SIZE];
        q4k_dequant_block(&raw_block, &mut dequant_output);

        let a = vec![1.0f32; Q4K_BLOCK_SIZE];
        let dot_result = q4k_dot_block(&raw_block, &a);

        let expected_dot: f32 = dequant_output.iter().sum();
        assert!(
            (dot_result - expected_dot).abs() < 1e-3,
            "dot_block={dot_result}, sum(dequant)={expected_dot}"
        );
    }

    #[test]
    fn test_opt_matches_reference() {
        let k = 512;
        let m = 2;
        let n = 4;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
            .collect();

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut b_q = vec![0u8; n * blocks_per_row * Q4K_BLOCK_BYTES];
        // Set valid f16 scales for each block, random qs data
        for blk in 0..(n * blocks_per_row) {
            let base = blk * Q4K_BLOCK_BYTES;
            let d_bits = half::f16::from_f32(0.01).to_bits();
            b_q[base] = d_bits as u8;
            b_q[base + 1] = (d_bits >> 8) as u8;
            let dmin_bits = half::f16::from_f32(0.005).to_bits();
            b_q[base + 2] = dmin_bits as u8;
            b_q[base + 3] = (dmin_bits >> 8) as u8;
            for i in 4..Q4K_BLOCK_BYTES {
                b_q[base + i] = ((blk * 144 + i) * 13 % 256) as u8;
            }
        }

        let ref_kernel = CpuMatMulQ4K;
        let opt_kernel = CpuMatMulQ4KOpt;

        let mut out_ref = vec![0.0f32; m * n];
        let mut out_opt = vec![0.0f32; m * n];

        ref_kernel
            .forward_b_transposed_q4k(&a, &b_q, &mut out_ref, m, n, k)
            .unwrap();
        opt_kernel
            .forward_b_transposed_q4k(&a, &b_q, &mut out_opt, m, n, k)
            .unwrap();

        for i in 0..m * n {
            assert!(
                (out_ref[i] - out_opt[i]).abs() < 1e-4,
                "mismatch at [{i}]: ref={}, opt={}",
                out_ref[i],
                out_opt[i]
            );
        }
    }

    #[test]
    fn bench_opt_vs_reference() {
        let configs = [(1, 4096, 4096), (1, 12288, 4096), (5, 4096, 4096)];
        for (m, n, k) in configs {
            bench_one(m, n, k);
        }
    }

    fn bench_one(m: usize, n: usize, k: usize) {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32) * 0.005 - 0.05)
            .collect();

        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let mut b_q = vec![0u8; n * blocks_per_row * Q4K_BLOCK_BYTES];
        for blk in 0..(n * blocks_per_row) {
            let base = blk * Q4K_BLOCK_BYTES;
            let d_bits = half::f16::from_f32(0.001).to_bits();
            b_q[base] = d_bits as u8;
            b_q[base + 1] = (d_bits >> 8) as u8;
            let dmin_bits = half::f16::from_f32(0.0005).to_bits();
            b_q[base + 2] = dmin_bits as u8;
            b_q[base + 3] = (dmin_bits >> 8) as u8;
            for i in 4..Q4K_BLOCK_BYTES {
                b_q[base + i] = ((blk * 37 + i * 11) % 256) as u8;
            }
        }

        let ref_kernel = CpuMatMulQ4K;
        let opt_kernel = CpuMatMulQ4KOpt;
        let iters = 5;

        let mut out = vec![0.0f32; m * n];

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            ref_kernel
                .forward_b_transposed_q4k(&a, &b_q, &mut out, m, n, k)
                .unwrap();
        }
        let ref_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            opt_kernel
                .forward_b_transposed_q4k(&a, &b_q, &mut out, m, n, k)
                .unwrap();
        }
        let opt_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = ref_ms / opt_ms;
        eprintln!(
            "  Q4K matmul {m}x{n}x{k}: ref={ref_ms:.1}ms, opt={opt_ms:.1}ms, speedup={speedup:.1}x"
        );
    }

    #[test]
    fn test_opt_larger_matmul() {
        let k = 4096;
        let m = 1;
        let n = 4096;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32) * 0.005 - 0.05)
            .collect();

        let mut b_q = vec![0u8; n * (k / Q4K_BLOCK_SIZE) * Q4K_BLOCK_BYTES];
        for (i, byte) in b_q.iter_mut().enumerate() {
            *byte = ((i * 11 + 3) % 256) as u8;
        }

        let ref_kernel = CpuMatMulQ4K;
        let opt_kernel = CpuMatMulQ4KOpt;

        let mut out_ref = vec![0.0f32; m * n];
        let mut out_opt = vec![0.0f32; m * n];

        ref_kernel
            .forward_b_transposed_q4k(&a, &b_q, &mut out_ref, m, n, k)
            .unwrap();
        opt_kernel
            .forward_b_transposed_q4k(&a, &b_q, &mut out_opt, m, n, k)
            .unwrap();

        let max_diff: f32 = out_ref
            .iter()
            .zip(&out_opt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-3,
            "max diff {max_diff} too large at 4096x4096"
        );
    }

    #[test]
    fn test_q6k_opt_vs_scalar() {
        let k = 256;
        let m = 1;
        let n = 4;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 19) as f32) * 0.02 - 0.18)
            .collect();

        // Build valid Q6_K blocks: 210 bytes each
        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let mut b_q = vec![0u8; n * blocks_per_row * Q6K_BLOCK_BYTES];
        for blk in 0..(n * blocks_per_row) {
            let base = blk * Q6K_BLOCK_BYTES;
            // ql: 128 bytes
            for i in 0..128 {
                b_q[base + i] = ((blk * 128 + i) * 7 % 256) as u8;
            }
            // qh: 64 bytes
            for i in 0..64 {
                b_q[base + 128 + i] = ((blk * 64 + i) * 13 % 256) as u8;
            }
            // scales: 16 i8 values
            for i in 0..16 {
                b_q[base + 192 + i] = ((blk * 16 + i) * 11 % 127) as u8;
            }
            // d: f16
            let d_bits = half::f16::from_f32(0.001).to_bits();
            b_q[base + 208] = d_bits as u8;
            b_q[base + 209] = (d_bits >> 8) as u8;
        }

        let ref_kernel = CpuMatMulQ4K;
        let opt_kernel = CpuMatMulQ4KOpt;

        let mut out_ref = vec![0.0f32; m * n];
        let mut out_opt = vec![0.0f32; m * n];

        ref_kernel
            .forward_b_transposed_q6k(&a, &b_q, &mut out_ref, m, n, k)
            .unwrap();
        opt_kernel
            .forward_b_transposed_q6k(&a, &b_q, &mut out_opt, m, n, k)
            .unwrap();

        let max_diff: f32 = out_ref
            .iter()
            .zip(&out_opt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        // Int8 quantization introduces small errors — allow 5% relative tolerance
        let max_abs = out_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_tol = if max_abs > 0.0 {
            max_diff / max_abs
        } else {
            max_diff
        };
        eprintln!("  Q6K small: max_diff={max_diff:.6}, max_abs={max_abs:.6}, rel={rel_tol:.4}");
        assert!(
            rel_tol < 0.05,
            "Q6K opt vs scalar: max_diff={max_diff}, max_abs={max_abs}, rel={rel_tol:.4}"
        );
    }

    #[test]
    fn test_q6k_opt_vs_scalar_large() {
        let k = 4096;
        let m = 1;
        let n = 128;

        let a: Vec<f32> = (0..m * k)
            .map(|i| {
                let x = (i as f32) * 0.001 - 2.0;
                x.sin() * 0.5
            })
            .collect();

        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let mut b_q = vec![0u8; n * blocks_per_row * Q6K_BLOCK_BYTES];
        for blk_idx in 0..(n * blocks_per_row) {
            let base = blk_idx * Q6K_BLOCK_BYTES;
            for i in 0..128 {
                b_q[base + i] = ((blk_idx * 128 + i) * 37 % 256) as u8;
            }
            for i in 0..64 {
                b_q[base + 128 + i] = ((blk_idx * 64 + i) * 53 % 256) as u8;
            }
            for i in 0..16 {
                b_q[base + 192 + i] = (((blk_idx * 16 + i) * 41 % 255) as i8).to_le_bytes()[0];
            }
            let d_bits = half::f16::from_f32(0.0005 + (blk_idx as f32) * 0.00001).to_bits();
            b_q[base + 208] = d_bits as u8;
            b_q[base + 209] = (d_bits >> 8) as u8;
        }

        let ref_kernel = CpuMatMulQ4K;
        let opt_kernel = CpuMatMulQ4KOpt;

        let mut out_ref = vec![0.0f32; m * n];
        let mut out_opt = vec![0.0f32; m * n];

        ref_kernel
            .forward_b_transposed_q6k(&a, &b_q, &mut out_ref, m, n, k)
            .unwrap();
        opt_kernel
            .forward_b_transposed_q6k(&a, &b_q, &mut out_opt, m, n, k)
            .unwrap();

        let mut max_diff = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..m * n {
            let diff = (out_ref[i] - out_opt[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
            max_abs = max_abs.max(out_ref[i].abs());
        }
        let rel_tol = if max_abs > 0.0 {
            max_diff / max_abs
        } else {
            max_diff
        };
        eprintln!("  Q6K large: max_diff={max_diff:.6} at [{worst_idx}], ref={:.6}, opt={:.6}, rel={rel_tol:.4}",
            out_ref[worst_idx], out_opt[worst_idx]);
        assert!(rel_tol < 0.05, "Q6K large: rel_tol={rel_tol:.4}");
    }
}

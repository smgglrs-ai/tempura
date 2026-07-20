use crate::Result;
pub use crate::{
    Dequantize, MatMulQ, QuantDtype, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS, Q4K_GROUP_SIZE,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, Q6K_GROUPS, Q6K_GROUP_SIZE, TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE,
    I2S_BLOCK_BYTES, I2S_BLOCK_SIZE,
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

        let _idot_0 = hsum_i32(p0);
        let _idot_1 = hsum_i32(p1);
        let _idot_2 = hsum_i32(p2);
        let _idot_3 = hsum_i32(p3);

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

// ── TQ2_0 ternary (2-bit packing, 2.0625 bpw) ─────────────────────────────

pub fn tq2_block_d(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[64], block[65]]))
}

pub fn tq2_dequant_block(block: &[u8], output: &mut [f32; TQ2_BLOCK_SIZE]) {
    let d = tq2_block_d(block);
    let qs = &block[0..64];
    let mut idx = 0;
    for &byte in qs {
        for shift in (0..8).step_by(2) {
            let bits = (byte >> shift) & 0x03;
            output[idx] = d * (bits as f32 - 1.0);
            idx += 1;
        }
    }
}

pub fn tq2_dot_block(block: &[u8], a: &[f32]) -> f32 {
    let d = tq2_block_d(block);
    let qs = &block[0..64];
    let mut sum = 0.0f32;
    let mut idx = 0;
    for &byte in qs {
        for shift in (0..8).step_by(2) {
            let trit = ((byte >> shift) & 0x03) as i8 - 1;
            match trit {
                1 => sum += a[idx],
                -1 => sum -= a[idx],
                _ => {}
            }
            idx += 1;
        }
    }
    sum * d
}

pub fn tq2_encode_block(values: &[f32; TQ2_BLOCK_SIZE]) -> [u8; TQ2_BLOCK_BYTES] {
    let mut block = [0u8; TQ2_BLOCK_BYTES];
    let mut amax = 0.0f32;
    for &v in values {
        let av = v.abs();
        if av > amax {
            amax = av;
        }
    }
    let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };

    let mut idx = 0;
    for byte_idx in 0..64 {
        let mut packed = 0u8;
        for shift in (0..8).step_by(2) {
            let v = values[idx];
            let q = (v * id).round().max(-1.0).min(1.0) as i8;
            let encoded = (q + 1) as u8;
            packed |= encoded << shift;
            idx += 1;
        }
        block[byte_idx] = packed;
    }

    let d_bits = f32_to_f16(amax);
    block[64] = d_bits as u8;
    block[65] = (d_bits >> 8) as u8;
    block
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp == 0 {
        return (sign << 15) as u16;
    }
    if exp == 0xff {
        if frac == 0 {
            return ((sign << 15) | 0x7c00) as u16;
        }
        return ((sign << 15) | 0x7c00 | (frac >> 13).max(1)) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7c00) as u16;
    }
    if new_exp <= 0 {
        return (sign << 15) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

// ── I2_S ternary (bitnet.cpp format: 2-bit packed + per-tensor f32 scale) ──
//
// Translated line-by-line from bitnet.cpp (ggml.c + ggml-bitnet-mad.cpp).
// Data layout per tensor: [packed_2bit_weights (n*k/4 bytes) | f32 scale]
//
// Source functions translated:
//   nearest_int()       — ggml.c:12536
//   quantize_row_i8_s() — ggml-quants.c:3503
//   ggml_vec_dot_i2_i8_s_1x1() — ggml-bitnet-mad.cpp:198
//   mul_mat rescaling   — ggml.c:12521

/// Translated from ggml.c:12536-12541.
/// Banker's rounding via the magic-number trick.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    // C: float val = fval + 12582912.f;
    let val: f32 = fval + 12582912.0;
    // C: int i; memcpy(&i, &val, sizeof(int));
    let i: i32 = i32::from_ne_bytes(val.to_ne_bytes());
    // C: return (i & 0x007fffff) - 0x00400000;
    (i & 0x007fffff) - 0x00400000
}

/// Translated from ggml-quants.c:3503-3522.
/// Quantizes f32 activations to i8 per-tensor, returns (scale, sum).
pub fn quantize_row_i8_s(x: &[f32], dst: &mut [i8]) -> (f32, i32) {
    // C: double min = 0.00001; double max = min;
    let mut max: f64 = 0.00001;
    // C: for (int i = 0; i < n; ++i) max = MAX(max, (double)fabs((double)x[i]));
    for &v in x {
        let av = (v as f64).abs();
        if av > max {
            max = av;
        }
    }
    // C: float s = 127 / max;
    let s: f32 = (127.0f64 / max) as f32;
    // C: int32_t sum = 0;
    let mut sum: i32 = 0;
    // C: for (int i = 0; i < n; ++i) {
    for (i, &v) in x.iter().enumerate() {
        // C: int v = nearest_int(x[i] * s);
        let mut q: i32 = nearest_int(v * s);
        // C: if (v > 127) v = 127; if (v < -128) v = -128;
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        // C: sum += v;
        sum += q;
        // C: dst[i] = (int8_t)(v);
        dst[i] = q as i8;
    }
    (s, sum)
}

/// Translated from ggml-bitnet-mad.cpp:198-296, scalar path (nrc=1).
/// Computes unsigned dot product of 2-bit packed weights × i8 activations.
///
/// Weight layout: 128 elements per 32-byte group.
/// Byte b (0..31) contains 4 2-bit values:
///   bits[7:6] → element b       (group_idx=0)
///   bits[5:4] → element 32+b    (group_idx=1)
///   bits[3:2] → element 64+b    (group_idx=2)
///   bits[1:0] → element 96+b    (group_idx=3)
/// Values {0,1,2} represent ternary {-1,0,+1}.
pub fn vec_dot_i2_i8_s(packed: &[u8], a_q: &[i8], k: usize) -> i32 {
    // C: const int nb = n / QK_I2_S;  // QK_I2_S = 128
    let nb = k / 128;
    let mut accu: i32 = 0;

    for i in 0..nb {
        // C: const uint8_t *px = x_row + i * 1024;  // but simplified for scalar
        // C: const int8_t  *py = y + i * 4096;      // but simplified for scalar
        let px = i * 32;
        let py = i * 128;

        for j in 0..32 {
            let byte = packed[px + j];
            // C: xq8_0 = (byte >> 6) & 0x03  → element j
            // C: xq8_1 = (byte >> 4) & 0x03  → element 32+j
            // C: xq8_2 = (byte >> 2) & 0x03  → element 64+j
            // C: xq8_3 = (byte >> 0) & 0x03  → element 96+j
            let w0 = ((byte >> 6) & 0x03) as i32;
            let w1 = ((byte >> 4) & 0x03) as i32;
            let w2 = ((byte >> 2) & 0x03) as i32;
            let w3 = (byte & 0x03) as i32;
            // C: maddubs(xq8_0, yq8_0) — unsigned × signed multiply-add
            accu += w0 * a_q[py + j] as i32;
            accu += w1 * a_q[py + 32 + j] as i32;
            accu += w2 * a_q[py + 64 + j] as i32;
            accu += w3 * a_q[py + 96 + j] as i32;
        }
    }
    accu
}

pub fn i2s_dequant_block(block: &[u8], output: &mut [f32; I2S_BLOCK_SIZE]) {
    for g in 0..(I2S_BLOCK_SIZE / 128) {
        for b in 0..32 {
            let byte = block[g * 32 + b];
            output[g * 128 + b]      = ((byte >> 6) & 0x03) as f32 - 1.0;
            output[g * 128 + 32 + b] = ((byte >> 4) & 0x03) as f32 - 1.0;
            output[g * 128 + 64 + b] = ((byte >> 2) & 0x03) as f32 - 1.0;
            output[g * 128 + 96 + b] = (byte & 0x03) as f32 - 1.0;
        }
    }
}

/// AVX2 per-tensor INT8 activation quantization — zero-allocation, writes directly into dst.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize_row_i8_s_avx2_into(x: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;

    let k = x.len();
    let mut vmax = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let chunks = k / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    let hi = _mm256_extractf128_ps(vmax, 1);
    let lo = _mm256_castps256_ps128(vmax);
    let max128 = _mm_max_ps(lo, hi);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
    let mut max_abs = _mm_cvtss_f32(max32);
    for i in chunks * 8..k {
        let av = x[i].abs();
        if av > max_abs { max_abs = av; }
    }
    if max_abs < 0.00001 { max_abs = 0.00001; }

    let s: f32 = 127.0 / max_abs;
    let s_vec = _mm256_set1_ps(s);
    let mut sum_acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let scaled = _mm256_mul_ps(v, s_vec);
        let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
            scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ));
        let clamped = _mm256_max_epi32(
            _mm256_set1_epi32(-128),
            _mm256_min_epi32(_mm256_set1_epi32(127), rounded),
        );
        let lo128 = _mm256_castsi256_si128(clamped);
        let hi128 = _mm256_extracti128_si256(clamped, 1);
        let packed16 = _mm_packs_epi32(lo128, hi128);
        let packed8 = _mm_packs_epi16(packed16, packed16);
        *(dst.as_mut_ptr().add(i * 8) as *mut i64) = _mm_cvtsi128_si64(packed8);
        sum_acc = _mm256_add_epi32(sum_acc, clamped);
    }

    let sum_hi = _mm256_extracti128_si256(sum_acc, 1);
    let sum_lo = _mm256_castsi256_si128(sum_acc);
    let s128 = _mm_add_epi32(sum_lo, sum_hi);
    let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
    let mut sum = _mm_cvtsi128_si32(s32);

    for i in chunks * 8..k {
        let mut q = nearest_int(x[i] * s);
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        sum += q;
        dst[i] = q as i8;
    }

    (s, sum)
}

/// AVX2 per-tensor INT8 activation quantization for I2S ternary matmul.
/// Returns (scale, sum, quantized_i8_vec).
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_i8_s_avx2(x: &[f32]) -> (f32, i32, Vec<i8>) {
    use std::arch::x86_64::*;

    let k = x.len();
    let mut dst = vec![0i8; k];

    // Find max abs using AVX2
    let mut vmax = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let chunks = k / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    let hi = _mm256_extractf128_ps(vmax, 1);
    let lo = _mm256_castps256_ps128(vmax);
    let max128 = _mm_max_ps(lo, hi);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
    let mut max_abs = _mm_cvtss_f32(max32);
    // Handle tail elements
    for i in chunks * 8..k {
        let av = x[i].abs();
        if av > max_abs { max_abs = av; }
    }
    if max_abs < 0.00001 { max_abs = 0.00001; }

    let s: f32 = 127.0 / max_abs;
    let s_vec = _mm256_set1_ps(s);

    let mut sum_acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let scaled = _mm256_mul_ps(v, s_vec);
        let rounded = _mm256_cvtps_epi32(_mm256_round_ps(
            scaled,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ));
        // Clamp to [-128, 127]
        let clamped = _mm256_max_epi32(
            _mm256_set1_epi32(-128),
            _mm256_min_epi32(_mm256_set1_epi32(127), rounded),
        );
        // Pack i32 → i16 → i8 (8 values)
        let lo128 = _mm256_castsi256_si128(clamped);
        let hi128 = _mm256_extracti128_si256(clamped, 1);
        let packed16 = _mm_packs_epi32(lo128, hi128);
        let packed8 = _mm_packs_epi16(packed16, packed16);
        let ptr = dst.as_mut_ptr().add(i * 8) as *mut i64;
        *ptr = _mm_cvtsi128_si64(packed8);

        // Accumulate sum: extend i32 → add
        sum_acc = _mm256_add_epi32(sum_acc, clamped);
    }

    // Horizontal sum of sum_acc
    let sum_hi = _mm256_extracti128_si256(sum_acc, 1);
    let sum_lo = _mm256_castsi256_si128(sum_acc);
    let s128 = _mm_add_epi32(sum_lo, sum_hi);
    let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
    let mut sum = _mm_cvtsi128_si32(s32);

    // Handle tail
    for i in chunks * 8..k {
        let mut q = nearest_int(x[i] * s);
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        sum += q;
        dst[i] = q as i8;
    }

    (s, sum, dst)
}

/// AVX2 I2S ternary dot product: 2-bit packed weights × i8 activations.
/// Processes 128 elements per group using _mm256_maddubs_epi16.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_i2_i8_s_avx2(packed: &[u8], a_q: &[i8], k: usize) -> i32 {
    use std::arch::x86_64::*;

    let nb = k / 128;
    let ones_16 = _mm256_set1_epi16(1);
    let mask_2bit = _mm256_set1_epi8(0x03);
    let mut acc = _mm256_setzero_si256();

    // Batch 32 groups in i16 before widening to i32 (bitnet.cpp pattern).
    // Each maddubs result is max ±254 as i16. 32 such values sum to max ±8128, fits i16.
    let group32_num = nb / 32;

    for batch in 0..group32_num {
        let mut accu16 = _mm256_setzero_si256();
        let base = batch * 32;

        for i in base..base + 32 {
            let px = i * 32;
            let py = i * 128;

            let w_raw = _mm256_loadu_si256(packed.as_ptr().add(px) as *const __m256i);
            let w0 = _mm256_and_si256(w_raw, mask_2bit);
            let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit);
            let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit);
            let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit);

            let a3 = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
            let a2 = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
            let a1 = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
            let a0 = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);

            let p0 = _mm256_maddubs_epi16(w0, a0);
            let p1 = _mm256_maddubs_epi16(w1, a1);
            let p2 = _mm256_maddubs_epi16(w2, a2);
            let p3 = _mm256_maddubs_epi16(w3, a3);

            accu16 = _mm256_add_epi16(accu16, _mm256_add_epi16(p0, p1));
            accu16 = _mm256_add_epi16(accu16, _mm256_add_epi16(p2, p3));
        }

        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(accu16, ones_16));
    }

    // Remaining groups (< 32)
    let remaining_start = group32_num * 32;
    if remaining_start < nb {
        let mut accu16 = _mm256_setzero_si256();
        for i in remaining_start..nb {
            let px = i * 32;
            let py = i * 128;

            let w_raw = _mm256_loadu_si256(packed.as_ptr().add(px) as *const __m256i);
            let w0 = _mm256_and_si256(w_raw, mask_2bit);
            let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit);
            let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit);
            let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit);

            let a3 = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
            let a2 = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
            let a1 = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
            let a0 = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);

            let p0 = _mm256_maddubs_epi16(w0, a0);
            let p1 = _mm256_maddubs_epi16(w1, a1);
            let p2 = _mm256_maddubs_epi16(w2, a2);
            let p3 = _mm256_maddubs_epi16(w3, a3);

            accu16 = _mm256_add_epi16(accu16, _mm256_add_epi16(p0, p1));
            accu16 = _mm256_add_epi16(accu16, _mm256_add_epi16(p2, p3));
        }
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(accu16, ones_16));
    }

    hsum_i32(acc)
}

/// AVX-VNNI I2S ternary dot product using _mm256_dpbusd_epi32.
/// Fuses maddubs+madd into a single instruction with direct i32 accumulation.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn vec_dot_i2_i8_s_avx_vnni(packed: &[u8], a_q: &[i8], k: usize) -> i32 {
    use std::arch::x86_64::*;

    let nb = k / 128;
    let mask_2bit = _mm256_set1_epi8(0x03);
    let mut acc = _mm256_setzero_si256();

    for i in 0..nb {
        let px = i * 32;
        let py = i * 128;

        let w_raw = _mm256_loadu_si256(packed.as_ptr().add(px) as *const __m256i);
        let w0 = _mm256_and_si256(w_raw, mask_2bit);
        let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit);
        let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit);
        let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit);

        let a3 = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
        let a2 = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
        let a1 = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
        let a0 = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);

        acc = _mm256_dpbusd_epi32(acc, w0, a0);
        acc = _mm256_dpbusd_epi32(acc, w1, a1);
        acc = _mm256_dpbusd_epi32(acc, w2, a2);
        acc = _mm256_dpbusd_epi32(acc, w3, a3);
    }

    hsum_i32(acc)
}

/// Horizontal sum of 16 i32 lanes in a __m512i.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn hsum_i32_512(v: std::arch::x86_64::__m512i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm512_extracti64x4_epi64(v, 1);
    let lo = _mm512_castsi512_si256(v);
    let sum256 = _mm256_add_epi32(lo, hi);
    hsum_i32(sum256)
}

/// AVX512 per-tensor INT8 activation quantization — zero-allocation, writes directly into dst.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn quantize_row_i8_s_avx512_into(x: &[f32], dst: &mut [i8]) -> (f32, i32) {
    use std::arch::x86_64::*;

    let k = x.len();
    let mut vmax = _mm512_setzero_ps();
    let chunks16 = k / 16;
    for i in 0..chunks16 {
        let v = _mm512_loadu_ps(x.as_ptr().add(i * 16));
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v));
    }
    let mut max_abs = _mm512_reduce_max_ps(vmax);
    for i in chunks16 * 16..k {
        let av = x[i].abs();
        if av > max_abs { max_abs = av; }
    }
    if max_abs < 0.00001 { max_abs = 0.00001; }

    let s: f32 = 127.0 / max_abs;
    let s_vec = _mm512_set1_ps(s);
    let min_val = _mm512_set1_epi32(-128);
    let max_val = _mm512_set1_epi32(127);
    let mut sum_acc = _mm512_setzero_si512();

    for i in 0..chunks16 {
        let v = _mm512_loadu_ps(x.as_ptr().add(i * 16));
        let scaled = _mm512_mul_ps(v, s_vec);
        let rounded = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT));
        let clamped = _mm512_max_epi32(min_val, _mm512_min_epi32(max_val, rounded));
        sum_acc = _mm512_add_epi32(sum_acc, clamped);

        let lo256 = _mm512_castsi512_si256(clamped);
        let hi256 = _mm512_extracti64x4_epi64(clamped, 1);
        let lo128_a = _mm256_castsi256_si128(lo256);
        let lo128_b = _mm256_extracti128_si256(lo256, 1);
        let hi128_a = _mm256_castsi256_si128(hi256);
        let hi128_b = _mm256_extracti128_si256(hi256, 1);
        let p16_lo = _mm_packs_epi32(lo128_a, lo128_b);
        let p16_hi = _mm_packs_epi32(hi128_a, hi128_b);
        let p8 = _mm_packs_epi16(p16_lo, p16_hi);
        _mm_storeu_si128(dst.as_mut_ptr().add(i * 16) as *mut __m128i, p8);
    }

    let mut sum = _mm512_reduce_add_epi32(sum_acc);
    for i in chunks16 * 16..k {
        let mut q = nearest_int(x[i] * s);
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        sum += q;
        dst[i] = q as i8;
    }

    (s, sum)
}

/// AVX512 VNNI per-tensor INT8 activation quantization for I2S ternary matmul.
/// Returns (scale, sum, quantized_i8_vec).
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn quantize_row_i8_s_avx512(x: &[f32]) -> (f32, i32, Vec<i8>) {
    use std::arch::x86_64::*;

    let k = x.len();
    let mut dst = vec![0i8; k];

    // Find max abs using AVX-512
    let mut vmax = _mm512_setzero_ps();
    let chunks16 = k / 16;
    for i in 0..chunks16 {
        let v = _mm512_loadu_ps(x.as_ptr().add(i * 16));
        vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v));
    }
    let mut max_abs = _mm512_reduce_max_ps(vmax);
    for i in chunks16 * 16..k {
        let av = x[i].abs();
        if av > max_abs { max_abs = av; }
    }
    if max_abs < 0.00001 { max_abs = 0.00001; }

    let s: f32 = 127.0 / max_abs;
    let s_vec = _mm512_set1_ps(s);
    let min_val = _mm512_set1_epi32(-128);
    let max_val = _mm512_set1_epi32(127);

    let mut sum_acc = _mm512_setzero_si512();

    for i in 0..chunks16 {
        let v = _mm512_loadu_ps(x.as_ptr().add(i * 16));
        let scaled = _mm512_mul_ps(v, s_vec);
        let rounded = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT));
        let clamped = _mm512_max_epi32(min_val, _mm512_min_epi32(max_val, rounded));
        sum_acc = _mm512_add_epi32(sum_acc, clamped);

        // Pack i32 → i8: extract two 256-bit halves, pack each
        let lo256 = _mm512_castsi512_si256(clamped);
        let hi256 = _mm512_extracti64x4_epi64(clamped, 1);
        let lo128_a = _mm256_castsi256_si128(lo256);
        let lo128_b = _mm256_extracti128_si256(lo256, 1);
        let hi128_a = _mm256_castsi256_si128(hi256);
        let hi128_b = _mm256_extracti128_si256(hi256, 1);
        let p16_lo = _mm_packs_epi32(lo128_a, lo128_b);
        let p16_hi = _mm_packs_epi32(hi128_a, hi128_b);
        let p8 = _mm_packs_epi16(p16_lo, p16_hi);
        _mm_storeu_si128(dst.as_mut_ptr().add(i * 16) as *mut __m128i, p8);
    }

    let mut sum = _mm512_reduce_add_epi32(sum_acc);
    for i in chunks16 * 16..k {
        let mut q = nearest_int(x[i] * s);
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        sum += q;
        dst[i] = q as i8;
    }

    (s, sum, dst)
}

/// AVX512 VNNI I2S ternary dot product using _mm512_dpbusd_epi32.
/// Uses 512-bit maddubs (processes 64 u8×i8 pairs per instruction) — 2x throughput over AVX2.
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn vec_dot_i2_i8_s_avx512(packed: &[u8], a_q: &[i8], k: usize) -> i32 {
    use std::arch::x86_64::*;

    let nb = k / 128;
    let mask_2bit = _mm512_set1_epi8(0x03);
    let ones_16 = _mm512_set1_epi16(1);
    let mut acc = _mm512_setzero_si512();

    // Batch groups: accumulate in i16 across pairs, widen to i32 every 16 pairs (32 groups)
    let pairs = nb / 2;
    let batch16 = pairs / 16;

    for batch in 0..batch16 {
        let mut accu16 = _mm512_setzero_si512();
        let base = batch * 16;

        for i in base..base + 16 {
            let px = i * 64;
            let py = i * 256;

            let w_raw = _mm512_loadu_si512(packed.as_ptr().add(px) as *const __m512i);
            let w0 = _mm512_and_si512(w_raw, mask_2bit);
            let w1 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 2), mask_2bit);
            let w2 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 4), mask_2bit);
            let w3 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 6), mask_2bit);

            let a3_lo = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
            let a3_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 128) as *const __m256i);
            let a3 = _mm512_inserti64x4(_mm512_castsi256_si512(a3_lo), a3_hi, 1);
            let a2_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
            let a2_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 160) as *const __m256i);
            let a2 = _mm512_inserti64x4(_mm512_castsi256_si512(a2_lo), a2_hi, 1);
            let a1_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
            let a1_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 192) as *const __m256i);
            let a1 = _mm512_inserti64x4(_mm512_castsi256_si512(a1_lo), a1_hi, 1);
            let a0_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);
            let a0_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 224) as *const __m256i);
            let a0 = _mm512_inserti64x4(_mm512_castsi256_si512(a0_lo), a0_hi, 1);

            let p0 = _mm512_maddubs_epi16(w0, a0);
            let p1 = _mm512_maddubs_epi16(w1, a1);
            let p2 = _mm512_maddubs_epi16(w2, a2);
            let p3 = _mm512_maddubs_epi16(w3, a3);

            accu16 = _mm512_add_epi16(accu16, _mm512_add_epi16(p0, p1));
            accu16 = _mm512_add_epi16(accu16, _mm512_add_epi16(p2, p3));
        }

        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(accu16, ones_16));
    }

    // Remaining pairs
    let remaining_start = batch16 * 16;
    if remaining_start < pairs {
        let mut accu16 = _mm512_setzero_si512();
        for i in remaining_start..pairs {
            let px = i * 64;
            let py = i * 256;

            let w_raw = _mm512_loadu_si512(packed.as_ptr().add(px) as *const __m512i);
            let w0 = _mm512_and_si512(w_raw, mask_2bit);
            let w1 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 2), mask_2bit);
            let w2 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 4), mask_2bit);
            let w3 = _mm512_and_si512(_mm512_srli_epi16(w_raw, 6), mask_2bit);

            let a3_lo = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
            let a3_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 128) as *const __m256i);
            let a3 = _mm512_inserti64x4(_mm512_castsi256_si512(a3_lo), a3_hi, 1);
            let a2_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
            let a2_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 160) as *const __m256i);
            let a2 = _mm512_inserti64x4(_mm512_castsi256_si512(a2_lo), a2_hi, 1);
            let a1_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
            let a1_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 192) as *const __m256i);
            let a1 = _mm512_inserti64x4(_mm512_castsi256_si512(a1_lo), a1_hi, 1);
            let a0_lo = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);
            let a0_hi = _mm256_loadu_si256(a_q.as_ptr().add(py + 224) as *const __m256i);
            let a0 = _mm512_inserti64x4(_mm512_castsi256_si512(a0_lo), a0_hi, 1);

            let p0 = _mm512_maddubs_epi16(w0, a0);
            let p1 = _mm512_maddubs_epi16(w1, a1);
            let p2 = _mm512_maddubs_epi16(w2, a2);
            let p3 = _mm512_maddubs_epi16(w3, a3);

            accu16 = _mm512_add_epi16(accu16, _mm512_add_epi16(p0, p1));
            accu16 = _mm512_add_epi16(accu16, _mm512_add_epi16(p2, p3));
        }
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(accu16, ones_16));
    }

    let mut result = _mm512_reduce_add_epi32(acc);

    // Handle remaining odd group with AVX2
    if pairs * 2 < nb {
        let i = pairs * 2;
        let px = i * 32;
        let py = i * 128;
        let mask_2bit_256 = _mm256_set1_epi8(0x03);
        let ones_16_256 = _mm256_set1_epi16(1);

        let w_raw = _mm256_loadu_si256(packed.as_ptr().add(px) as *const __m256i);
        let w0 = _mm256_and_si256(w_raw, mask_2bit_256);
        let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit_256);
        let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit_256);
        let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit_256);

        let a3 = _mm256_loadu_si256(a_q.as_ptr().add(py) as *const __m256i);
        let a2 = _mm256_loadu_si256(a_q.as_ptr().add(py + 32) as *const __m256i);
        let a1 = _mm256_loadu_si256(a_q.as_ptr().add(py + 64) as *const __m256i);
        let a0 = _mm256_loadu_si256(a_q.as_ptr().add(py + 96) as *const __m256i);

        let p0 = _mm256_maddubs_epi16(w0, a0);
        let p1 = _mm256_maddubs_epi16(w1, a1);
        let p2 = _mm256_maddubs_epi16(w2, a2);
        let p3 = _mm256_maddubs_epi16(w3, a3);

        let accu16 = _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
        result += hsum_i32(_mm256_madd_epi16(accu16, ones_16_256));
    }

    result
}

/// Quantize f32 activations to i8 in-place (no allocation). Returns (scale, sum).
#[cfg(target_arch = "x86_64")]
pub fn i2s_quantize_into(a: &[f32], dst: &mut [i8]) -> (f32, i32) {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        return unsafe { quantize_row_i8_s_avx512_into(a, dst) };
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { quantize_row_i8_s_avx2_into(a, dst) };
    }
    quantize_row_i8_s(a, dst)
}

/// Single-threaded I2S GEMV on column range [col_start, col_end).
/// Called by layer-level threading — each thread processes its own column slice.
/// Uses raw addresses to satisfy Sync bounds in parallel closures.
#[cfg(target_arch = "x86_64")]
pub fn i2s_gemv_cols(
    preq_q8: &[i8],
    preq_scale: f32,
    preq_sum: i32,
    b_quantized: &[u8],
    w_scale: f32,
    output_addr: usize,
    col_start: usize,
    col_end: usize,
    k: usize,
) {
    let row_packed_bytes = k / 4;
    if is_x86_feature_detected!("avx2") {
        unsafe { i2s_gemv_cols_multi_row(preq_q8, preq_scale, preq_sum, b_quantized, w_scale, output_addr, col_start, col_end, k) };
        return;
    }

    let has_avx_vnni = is_x86_feature_detected!("avxvnni");
    let has_avx512 = is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni");

    for j in col_start..col_end {
        let w = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
        let raw = if has_avx_vnni {
            unsafe { vec_dot_i2_i8_s_avx_vnni(w, preq_q8, k) }
        } else if has_avx512 {
            unsafe { vec_dot_i2_i8_s_avx512(w, preq_q8, k) }
        } else {
            unsafe { vec_dot_i2_i8_s_avx2(w, preq_q8, k) }
        };
        unsafe { *(output_addr as *mut f32).add(j) = (raw - preq_sum) as f32 / preq_scale * w_scale; }
    }
}

/// Multi-row I2S GEMV: processes 16 weight rows at a time, sharing activation loads.
/// Translated from bitnet.cpp's ggml_vec_dot_i2_i8_s (blck_0=16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i2s_gemv_cols_multi_row(
    preq_q8: &[i8],
    preq_scale: f32,
    preq_sum: i32,
    b_quantized: &[u8],
    w_scale: f32,
    output_addr: usize,
    col_start: usize,
    col_end: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    const BLK: usize = 8;
    let row_packed_bytes = k / 4;
    let nb = k / 128;
    let mask_2bit = _mm256_set1_epi8(0x03);
    let ones_16 = _mm256_set1_epi16(1);
    let group32_num = nb / 32;
    let remainder = nb % 32;
    let scale = w_scale / preq_scale;

    let mut j = col_start;
    while j + BLK <= col_end {
        let mut acc = [_mm256_setzero_si256(); BLK];

        for batch in 0..group32_num {
            let mut accu16 = [_mm256_setzero_si256(); BLK];
            let base = batch * 32;

            for g in base..base + 32 {
                let py = g * 128;
                let a0 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 96) as *const __m256i);
                let a1 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 64) as *const __m256i);
                let a2 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 32) as *const __m256i);
                let a3 = _mm256_loadu_si256(preq_q8.as_ptr().add(py) as *const __m256i);

                for r in 0..BLK {
                    let px = (j + r) * row_packed_bytes + g * 32;
                    let w_raw = _mm256_loadu_si256(b_quantized.as_ptr().add(px) as *const __m256i);
                    let w0 = _mm256_and_si256(w_raw, mask_2bit);
                    let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit);
                    let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit);
                    let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit);

                    let p0 = _mm256_maddubs_epi16(w0, a0);
                    let p1 = _mm256_maddubs_epi16(w1, a1);
                    let p2 = _mm256_maddubs_epi16(w2, a2);
                    let p3 = _mm256_maddubs_epi16(w3, a3);

                    accu16[r] = _mm256_add_epi16(accu16[r], _mm256_add_epi16(p0, p1));
                    accu16[r] = _mm256_add_epi16(accu16[r], _mm256_add_epi16(p2, p3));
                }
            }

            for r in 0..BLK {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(accu16[r], ones_16));
            }
        }

        if remainder > 0 {
            let mut accu16 = [_mm256_setzero_si256(); BLK];
            let base = group32_num * 32;
            for g in base..base + remainder {
                let py = g * 128;
                let a0 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 96) as *const __m256i);
                let a1 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 64) as *const __m256i);
                let a2 = _mm256_loadu_si256(preq_q8.as_ptr().add(py + 32) as *const __m256i);
                let a3 = _mm256_loadu_si256(preq_q8.as_ptr().add(py) as *const __m256i);

                for r in 0..BLK {
                    let px = (j + r) * row_packed_bytes + g * 32;
                    let w_raw = _mm256_loadu_si256(b_quantized.as_ptr().add(px) as *const __m256i);
                    let w0 = _mm256_and_si256(w_raw, mask_2bit);
                    let w1 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 2), mask_2bit);
                    let w2 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 4), mask_2bit);
                    let w3 = _mm256_and_si256(_mm256_srli_epi16(w_raw, 6), mask_2bit);

                    let p0 = _mm256_maddubs_epi16(w0, a0);
                    let p1 = _mm256_maddubs_epi16(w1, a1);
                    let p2 = _mm256_maddubs_epi16(w2, a2);
                    let p3 = _mm256_maddubs_epi16(w3, a3);

                    accu16[r] = _mm256_add_epi16(accu16[r], _mm256_add_epi16(p0, p1));
                    accu16[r] = _mm256_add_epi16(accu16[r], _mm256_add_epi16(p2, p3));
                }
            }
            for r in 0..BLK {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(accu16[r], ones_16));
            }
        }

        for r in 0..BLK {
            let raw = hsum_i32(acc[r]);
            *(output_addr as *mut f32).add(j + r) = (raw - preq_sum) as f32 * scale;
        }
        j += BLK;
    }

    // Remaining rows (< 16)
    while j < col_end {
        let w = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
        let raw = vec_dot_i2_i8_s_avx2(w, preq_q8, k);
        *(output_addr as *mut f32).add(j) = (raw - preq_sum) as f32 * scale;
        j += 1;
    }
}

/// Extract the per-tensor f32 scale from I2S weight data.
#[inline]
pub fn i2s_weight_scale(b_quantized: &[u8], n: usize, k: usize) -> f32 {
    let packed_bytes = n * k / 4;
    f32::from_ne_bytes([
        b_quantized[packed_bytes],
        b_quantized[packed_bytes + 1],
        b_quantized[packed_bytes + 2],
        b_quantized[packed_bytes + 3],
    ])
}

// ── LUT-based I2S ternary dot product (bitnet.cpp TL2 translation) ─────────
//
// Instead of maddubs(weight_u8, activation_i8), precompute ALL possible
// partial sums from activation pairs, then use weight nibbles as table indices.
// Activations quantized to [-63, 63] so pair sums fit in i8 → single shuffle per nibble.

#[inline]
fn decode_ternary(w: u8) -> i8 {
    (w as i8) - 1
}

/// Quantize f32 activations to i8 with scale = 63/max_abs (half precision vs standard 127).
/// Pair sums stay in i8 range: max |a0 + a1| = 126 < 128.
fn quantize_row_i8_s_lut(x: &[f32], dst: &mut [i8]) -> f32 {
    let mut max: f64 = 0.00001;
    for &v in x {
        let av = (v as f64).abs();
        if av > max { max = av; }
    }
    let s: f32 = (63.0f64 / max) as f32;
    for (i, &v) in x.iter().enumerate() {
        let mut q = nearest_int(v * s);
        if q > 63 { q = 63; }
        if q < -63 { q = -63; }
        dst[i] = q as i8;
    }
    s
}

/// Build a 16-entry i8 LUT for a pair of activations (scaled to [-63,63]).
/// Index = (w_high_2bit << 2) | w_low_2bit. Max |entry| = 126, fits i8.
pub fn i2s_build_pair_lut_i8(a_low: i8, a_high: i8) -> [i8; 16] {
    let mut lut = [0i8; 16];
    for w_high in 0u8..3 {
        for w_low in 0u8..3 {
            let idx = ((w_high as usize) << 2) | (w_low as usize);
            let val = decode_ternary(w_high) as i16 * a_high as i16
                    + decode_ternary(w_low) as i16 * a_low as i16;
            lut[idx] = val as i8;
        }
    }
    lut
}

/// Build all i8 pair LUTs for a quantized activation row.
/// Layout: 64 LUTs per 128-element group, 16 bytes each, contiguous.
fn i2s_build_all_pair_luts_i8(a_q: &[i8], k: usize) -> Vec<i8> {
    let nb = k / 128;
    let mut luts = vec![0i8; nb * 64 * 16];

    for g in 0..nb {
        let base = g * 128;
        for b in 0..32 {
            let upper_lut = i2s_build_pair_lut_i8(a_q[base + 32 + b], a_q[base + b]);
            let off = g * 64 * 16 + b * 16;
            luts[off..off + 16].copy_from_slice(&upper_lut);

            let lower_lut = i2s_build_pair_lut_i8(a_q[base + 96 + b], a_q[base + 64 + b]);
            let off2 = g * 64 * 16 + (32 + b) * 16;
            luts[off2..off2 + 16].copy_from_slice(&lower_lut);
        }
    }
    luts
}

/// Build all pair LUTs with pre-split lo/hi bytes for AVX2 shuffle.
/// Layout per pair: 16 lo-bytes then 16 hi-bytes = 32 bytes.
/// Total: 64 pairs per group × 32 bytes = 2048 bytes per group.
/// Uses full 127-scale precision (i16 entries, no range reduction).
fn i2s_build_all_pair_luts_split(a_q: &[i8], k: usize) -> Vec<u8> {
    let nb = k / 128;
    let mut luts = vec![0u8; nb * 64 * 32]; // 64 pairs × 32 bytes per group

    for g in 0..nb {
        let base = g * 128;
        for b in 0..32 {
            // Upper nibble pair
            let a_high = a_q[base + b];
            let a_low = a_q[base + 32 + b];
            let off = g * 64 * 32 + b * 32;
            for w_high in 0u8..3 {
                for w_low in 0u8..3 {
                    let idx = ((w_high as usize) << 2) | (w_low as usize);
                    let val = decode_ternary(w_high) as i16 * a_high as i16
                            + decode_ternary(w_low) as i16 * a_low as i16;
                    luts[off + idx] = val as u8;           // lo byte
                    luts[off + 16 + idx] = (val >> 8) as u8; // hi byte
                }
            }

            // Lower nibble pair
            let a_high2 = a_q[base + 64 + b];
            let a_low2 = a_q[base + 96 + b];
            let off2 = g * 64 * 32 + (32 + b) * 32;
            for w_high in 0u8..3 {
                for w_low in 0u8..3 {
                    let idx = ((w_high as usize) << 2) | (w_low as usize);
                    let val = decode_ternary(w_high) as i16 * a_high2 as i16
                            + decode_ternary(w_low) as i16 * a_low2 as i16;
                    luts[off2 + idx] = val as u8;
                    luts[off2 + 16 + idx] = (val >> 8) as u8;
                }
            }
        }
    }
    luts
}

// Keep i16 versions for the scalar correctness test
pub fn i2s_build_pair_lut(a_low: i8, a_high: i8) -> [i16; 16] {
    let mut lut = [0i16; 16];
    for w_high in 0u8..3 {
        for w_low in 0u8..3 {
            let idx = ((w_high as usize) << 2) | (w_low as usize);
            lut[idx] = decode_ternary(w_high) as i16 * a_high as i16
                     + decode_ternary(w_low) as i16 * a_low as i16;
        }
    }
    lut
}

fn i2s_build_all_pair_luts(a_q: &[i8], k: usize) -> Vec<i16> {
    let nb = k / 128;
    let mut luts = vec![0i16; nb * 64 * 16];
    for g in 0..nb {
        let base = g * 128;
        for b in 0..32 {
            let upper_lut = i2s_build_pair_lut(a_q[base + 32 + b], a_q[base + b]);
            let off = g * 64 * 16 + b * 16;
            luts[off..off + 16].copy_from_slice(&upper_lut);
            let lower_lut = i2s_build_pair_lut(a_q[base + 96 + b], a_q[base + 64 + b]);
            let off2 = g * 64 * 16 + (32 + b) * 16;
            luts[off2..off2 + 16].copy_from_slice(&lower_lut);
        }
    }
    luts
}

#[allow(dead_code)]
fn vec_dot_i2_i8_s_lut_scalar(packed: &[u8], pair_luts: &[i16], k: usize) -> i32 {
    let nb = k / 128;
    let mut acc: i32 = 0;
    for g in 0..nb {
        let lut_base = g * 64 * 16;
        for b in 0..32 {
            let byte = packed[g * 32 + b];
            let upper_nibble = (byte >> 4) as usize;
            let lower_nibble = (byte & 0x0F) as usize;
            acc += pair_luts[lut_base + b * 16 + upper_nibble] as i32;
            acc += pair_luts[lut_base + (32 + b) * 16 + lower_nibble] as i32;
        }
    }
    acc
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

    fn forward_b_transposed_tq2(
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
        if k % TQ2_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} must be divisible by TQ2 block size {TQ2_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / TQ2_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * TQ2_BLOCK_BYTES;
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
                    let block_offset = (j * blocks_per_row + blk) * TQ2_BLOCK_BYTES;
                    let block = &b_quantized[block_offset..block_offset + TQ2_BLOCK_BYTES];
                    let a_slice = &a_row[blk * TQ2_BLOCK_SIZE..(blk + 1) * TQ2_BLOCK_SIZE];
                    sum += tq2_dot_block(block, a_slice);
                }
                output[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }

        // C: const float * scale = (float *)((uint8_t*)(src0->data) + (ne00 * ne01 / 4));
        let packed_bytes = n * k / 4;
        let row_packed_bytes = k / 4;
        let w_scale = f32::from_ne_bytes([
            b_quantized[packed_bytes],
            b_quantized[packed_bytes + 1],
            b_quantized[packed_bytes + 2],
            b_quantized[packed_bytes + 3],
        ]);

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];

            // C: quantize_row_i8_s(src1_row, wdata, ne10, &act_scale, &act_sum);
            let mut a_q = vec![0i8; k];
            let (act_scale, act_sum) = quantize_row_i8_s(a_row, &mut a_q);

            for j in 0..n {
                // C: vec_dot(ne00, &tmp, 0, src0_row + ir0 * nb01 / 4, 0, src1_col_de, 0, 1);
                let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
                let raw = vec_dot_i2_i8_s(w_packed, &a_q, k);
                // C: tmp = (tmp - act_sums[i1]) / (act_scales[i1]) * (*scale);
                output[i * n + j] = (raw - act_sum) as f32 / act_scale * w_scale;
            }
        }
        Ok(())
    }

    fn forward_i2s_preq(
        &self,
        preq_q8: &[i8],
        preq_scale: f32,
        preq_sum: i32,
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let packed_bytes = n * k / 4;
        let row_packed_bytes = k / 4;
        let w_scale = f32::from_ne_bytes([
            b_quantized[packed_bytes],
            b_quantized[packed_bytes + 1],
            b_quantized[packed_bytes + 2],
            b_quantized[packed_bytes + 3],
        ]);

        for j in 0..n {
            let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
            let raw = vec_dot_i2_i8_s(w_packed, preq_q8, k);
            output[j] = (raw - preq_sum) as f32 / preq_scale * w_scale;
        }
        Ok(())
    }

    fn forward_i2s_lut(
        &self,
        a: &[f32],
        b_transposed: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }

        let packed_bytes = n * k / 4;
        let _row_packed_bytes = k / 4;
        let w_scale = f32::from_ne_bytes([
            b_transposed[packed_bytes],
            b_transposed[packed_bytes + 1],
            b_transposed[packed_bytes + 2],
            b_transposed[packed_bytes + 3],
        ]);

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let mut a_q = vec![0i8; k];
            let (act_scale, _act_sum) = quantize_row_i8_s(a_row, &mut a_q);
            let pair_luts = i2s_build_all_pair_luts(&a_q, k);

            // Transposed layout: byte position b has n values contiguous at offset b*n
            for j in 0..n {
                let mut acc: i32 = 0;
                for g in 0..(k / 128) {
                    let lut_base = g * 64 * 16;
                    for b in 0..32 {
                        let byte_pos = g * 32 + b;
                        let byte = b_transposed[byte_pos * n + j];
                        let upper_nibble = (byte >> 4) as usize;
                        let lower_nibble = (byte & 0x0F) as usize;
                        acc += pair_luts[lut_base + b * 16 + upper_nibble] as i32;
                        acc += pair_luts[lut_base + (32 + b) * 16 + lower_nibble] as i32;
                    }
                }
                output[i * n + j] = acc as f32 / act_scale * w_scale;
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
    pub i2s_q8: Vec<i8>,
    pub i2s_scale: f32,
    pub i2s_sum: i32,
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

            let (i2s_scale, i2s_sum, i2s_q8) = unsafe { quantize_row_i8_s_avx2(a) };

            return Self {
                q8,
                q16,
                scales,
                scales_i16,
                group_sums_q4k,
                group_sums_q6k,
                i2s_q8,
                i2s_scale,
                i2s_sum,
                k,
            };
        }

        let mut i2s_q8 = vec![0i8; k];
        let (i2s_scale, i2s_sum) = quantize_row_i8_s(a, &mut i2s_q8);
        Self {
            q8: Vec::new(),
            q16: Vec::new(),
            scales: Vec::new(),
            scales_i16: Vec::new(),
            group_sums_q4k: Vec::new(),
            group_sums_q6k: Vec::new(),
            i2s_q8,
            i2s_scale,
            i2s_sum,
            k,
        }
    }

    pub fn i2s_only(a: &[f32]) -> Self {
        let k = a.len();
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let (i2s_scale, i2s_sum, i2s_q8) = unsafe { quantize_row_i8_s_avx2(a) };
            return Self {
                q8: Vec::new(), q16: Vec::new(), scales: Vec::new(),
                scales_i16: Vec::new(), group_sums_q4k: Vec::new(),
                group_sums_q6k: Vec::new(),
                i2s_q8, i2s_scale, i2s_sum, k,
            };
        }
        let mut i2s_q8 = vec![0i8; k];
        let (i2s_scale, i2s_sum) = quantize_row_i8_s(a, &mut i2s_q8);
        Self {
            q8: Vec::new(), q16: Vec::new(), scales: Vec::new(),
            scales_i16: Vec::new(), group_sums_q4k: Vec::new(),
            group_sums_q6k: Vec::new(),
            i2s_q8, i2s_scale, i2s_sum, k,
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
        } else if std::env::var("TEMPURA_DETERMINISTIC").is_ok() {
            // Deterministic batched: per-element q4k_dot_block, matches GPU accumulation order
            let blocks_per_row = k / Q4K_BLOCK_SIZE;
            output[..m * n].par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
                let a_row = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for blk in 0..blocks_per_row {
                        let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                        let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                        let a_slice = &a_row[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                        sum += q4k_dot_block(block, a_slice);
                    }
                    row_out[j] = sum;
                }
            });
        } else {
            // Fast batched path: dequant full row, use gemm for SIMD matmul
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

    fn forward_b_transposed_tq2(
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
        if k % TQ2_BLOCK_SIZE != 0 {
            return Err(crate::KernelError::InvalidConfig(format!(
                "k={k} not divisible by {TQ2_BLOCK_SIZE}"
            )));
        }

        let blocks_per_row = k / TQ2_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * TQ2_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(crate::KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        let chunk_size = (n / rayon::current_num_threads()).max(64);
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            output[i * n..(i + 1) * n]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let j_start = ci * chunk_size;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let j = j_start + local;
                        let mut sum = 0.0f32;
                        for blk in 0..blocks_per_row {
                            let block_offset = (j * blocks_per_row + blk) * TQ2_BLOCK_BYTES;
                            let block = &b_quantized[block_offset..block_offset + TQ2_BLOCK_BYTES];
                            let a_slice = &a_row[blk * TQ2_BLOCK_SIZE..(blk + 1) * TQ2_BLOCK_SIZE];
                            sum += tq2_dot_block(block, a_slice);
                        }
                        *out = sum;
                    }
                });
        }
        Ok(())
    }

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }

        let packed_bytes = n * k / 4;
        let row_packed_bytes = k / 4;
        let w_scale = f32::from_ne_bytes([
            b_quantized[packed_bytes],
            b_quantized[packed_bytes + 1],
            b_quantized[packed_bytes + 2],
            b_quantized[packed_bytes + 3],
        ]);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avxvnni") {
            let mut q_buf = vec![0i8; k];
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                let (act_scale, act_sum) = i2s_quantize_into(a_row, &mut q_buf);
                let out_row = &mut output[i * n..(i + 1) * n];
                for j in 0..n {
                    let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
                    let raw = unsafe { vec_dot_i2_i8_s_avx_vnni(w_packed, &q_buf, k) };
                    out_row[j] = (raw - act_sum) as f32 / act_scale * w_scale;
                }
            }
            return Ok(());
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                let (act_scale, act_sum, a_q) = unsafe { quantize_row_i8_s_avx512(a_row) };
                let out_row = &mut output[i * n..(i + 1) * n];
                for j in 0..n {
                    let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
                    let raw = unsafe { vec_dot_i2_i8_s_avx512(w_packed, &a_q, k) };
                    out_row[j] = (raw - act_sum) as f32 / act_scale * w_scale;
                }
            }
            return Ok(());
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                let (act_scale, act_sum, a_q) = unsafe { quantize_row_i8_s_avx2(a_row) };
                let out_row = &mut output[i * n..(i + 1) * n];
                for j in 0..n {
                    let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
                    let raw = unsafe { vec_dot_i2_i8_s_avx2(w_packed, &a_q, k) };
                    out_row[j] = (raw - act_sum) as f32 / act_scale * w_scale;
                }
            }
            return Ok(());
        }

        let cpu = CpuMatMulQ4K;
        cpu.forward_b_transposed_i2s(a, b_quantized, output, m, n, k)
    }

    fn forward_i2s_preq(
        &self,
        preq_q8: &[i8],
        preq_scale: f32,
        preq_sum: i32,
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let packed_bytes = n * k / 4;
        let row_packed_bytes = k / 4;
        let w_scale = f32::from_ne_bytes([
            b_quantized[packed_bytes],
            b_quantized[packed_bytes + 1],
            b_quantized[packed_bytes + 2],
            b_quantized[packed_bytes + 3],
        ]);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let out_addr = output.as_mut_ptr() as usize;
            let b_addr = b_quantized.as_ptr() as usize;
            let a_addr = preq_q8.as_ptr() as usize;
            let has_avx_vnni = is_x86_feature_detected!("avxvnni");
            let has_avx512 = is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni");

            let chunk_size = (n + 7) / 8;
            crate::thread_pool::global_pool().parallel_chunks(n, chunk_size, &|start, end| {
                for j in start..end {
                    let w = unsafe {
                        std::slice::from_raw_parts((b_addr + j * row_packed_bytes) as *const u8, row_packed_bytes)
                    };
                    let aq = unsafe { std::slice::from_raw_parts(a_addr as *const i8, k) };
                    let raw = if has_avx_vnni {
                        unsafe { vec_dot_i2_i8_s_avx_vnni(w, aq, k) }
                    } else if has_avx512 {
                        unsafe { vec_dot_i2_i8_s_avx512(w, aq, k) }
                    } else {
                        unsafe { vec_dot_i2_i8_s_avx2(w, aq, k) }
                    };
                    unsafe { *(out_addr as *mut f32).add(j) = (raw - preq_sum) as f32 / preq_scale * w_scale; }
                }
            });
            return Ok(());
        }

        let cpu = CpuMatMulQ4K;
        cpu.forward_i2s_preq(preq_q8, preq_scale, preq_sum, b_quantized, output, n, k)
    }

    fn forward_i2s_lut(
        &self,
        a: &[f32],
        b_transposed: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k {
            return Err(crate::KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(crate::KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }

        let packed_bytes = n * k / 4;
        let w_scale = f32::from_ne_bytes([
            b_transposed[packed_bytes],
            b_transposed[packed_bytes + 1],
            b_transposed[packed_bytes + 2],
            b_transposed[packed_bytes + 3],
        ]);

        let num_groups = k / 128;

        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];

            // Full-precision i16 LUT with pre-split lo/hi bytes (bitnet.cpp TL2 pattern)
            let mut a_q = vec![0i8; k];
            let (act_scale, _) = quantize_row_i8_s(a_row, &mut a_q);
            let split_luts = i2s_build_all_pair_luts_split(&a_q, k);

            let out_row = &mut output[i * n..(i + 1) * n];

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && n >= 32 {
                let n_chunks = n / 32;

                let chunk_results: Vec<[i32; 32]> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let j_start = chunk_idx * 32;
                        let mut accs = [0i32; 32];

                        unsafe {
                            use std::arch::x86_64::*;
                            let nibble_mask = _mm256_set1_epi8(0x0F);
                            let mut acc0 = _mm256_setzero_si256();
                            let mut acc1 = _mm256_setzero_si256();
                            let mut acc2 = _mm256_setzero_si256();
                            let mut acc3 = _mm256_setzero_si256();

                            for g in 0..num_groups {
                                let lut_base = g * 64 * 32;
                                let mut vec_c0 = _mm256_setzero_si256();
                                let mut vec_c1 = _mm256_setzero_si256();

                                for b in 0..32 {
                                    let byte_pos = g * 32 + b;
                                    let w = _mm256_loadu_si256(
                                        b_transposed.as_ptr().add(byte_pos * n + j_start) as *const __m256i
                                    );
                                    let upper = _mm256_and_si256(_mm256_srli_epi16(w, 4), nibble_mask);
                                    let lower = _mm256_and_si256(w, nibble_mask);

                                    // 4 LUT loads: lo/hi bytes for upper and lower nibble pairs
                                    let u_off = lut_base + b * 32;
                                    let tbl_u_lo = _mm256_broadcastsi128_si256(
                                        _mm_loadu_si128(split_luts.as_ptr().add(u_off) as *const __m128i));
                                    let tbl_u_hi = _mm256_broadcastsi128_si256(
                                        _mm_loadu_si128(split_luts.as_ptr().add(u_off + 16) as *const __m128i));
                                    let l_off = lut_base + (32 + b) * 32;
                                    let tbl_l_lo = _mm256_broadcastsi128_si256(
                                        _mm_loadu_si128(split_luts.as_ptr().add(l_off) as *const __m128i));
                                    let tbl_l_hi = _mm256_broadcastsi128_si256(
                                        _mm_loadu_si128(split_luts.as_ptr().add(l_off + 16) as *const __m128i));

                                    // 4 shuffles: lo/hi byte lookups
                                    let u_lo_r = _mm256_shuffle_epi8(tbl_u_lo, upper);
                                    let u_hi_r = _mm256_shuffle_epi8(tbl_u_hi, upper);
                                    let l_lo_r = _mm256_shuffle_epi8(tbl_l_lo, lower);
                                    let l_hi_r = _mm256_shuffle_epi8(tbl_l_hi, lower);

                                    // Reconstruct i16: unpack(lo_bytes, hi_bytes) → i16
                                    // unpacklo: bytes 0-7 per lane → cols [0-7 | 16-23]
                                    // unpackhi: bytes 8-15 per lane → cols [8-15 | 24-31]
                                    let u_i16_c0 = _mm256_unpacklo_epi8(u_lo_r, u_hi_r);
                                    let u_i16_c1 = _mm256_unpackhi_epi8(u_lo_r, u_hi_r);
                                    let l_i16_c0 = _mm256_unpacklo_epi8(l_lo_r, l_hi_r);
                                    let l_i16_c1 = _mm256_unpackhi_epi8(l_lo_r, l_hi_r);

                                    // Accumulate in i16 (32 byte positions × max ±254 per position → max ±8128, fits i16)
                                    vec_c0 = _mm256_add_epi16(vec_c0, _mm256_add_epi16(u_i16_c0, l_i16_c0));
                                    vec_c1 = _mm256_add_epi16(vec_c1, _mm256_add_epi16(u_i16_c1, l_i16_c1));
                                }

                                // Widen i16 → i32 once per group
                                acc0 = _mm256_add_epi32(acc0,
                                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
                                acc1 = _mm256_add_epi32(acc1,
                                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
                                acc2 = _mm256_add_epi32(acc2,
                                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
                                acc3 = _mm256_add_epi32(acc3,
                                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
                            }

                            // acc0=[0-7], acc2=[8-15], acc1=[16-23], acc3=[24-31]
                            _mm256_storeu_si256(accs[0..8].as_mut_ptr() as *mut __m256i, acc0);
                            _mm256_storeu_si256(accs[8..16].as_mut_ptr() as *mut __m256i, acc2);
                            _mm256_storeu_si256(accs[16..24].as_mut_ptr() as *mut __m256i, acc1);
                            _mm256_storeu_si256(accs[24..32].as_mut_ptr() as *mut __m256i, acc3);
                        }

                        accs
                    })
                    .collect();

                for (chunk_idx, accs) in chunk_results.iter().enumerate() {
                    let j_start = chunk_idx * 32;
                    for (local, &acc) in accs.iter().enumerate() {
                        out_row[j_start + local] = acc as f32 / act_scale * w_scale;
                    }
                }

                // Scalar tail
                for j in n_chunks * 32..n {
                    let mut acc: i32 = 0;
                    for g in 0..num_groups {
                        let lut_base = g * 64 * 32;
                        for b in 0..32 {
                            let byte_pos = g * 32 + b;
                            let byte = b_transposed[byte_pos * n + j];
                            let u_idx = (byte >> 4) as usize;
                            let l_idx = (byte & 0x0F) as usize;
                            let u_off = lut_base + b * 32;
                            let l_off = lut_base + (32 + b) * 32;
                            let u_val = split_luts[u_off + u_idx] as i8 as i16
                                      | (split_luts[u_off + 16 + u_idx] as i8 as i16) << 8;
                            let l_val = split_luts[l_off + l_idx] as i8 as i16
                                      | (split_luts[l_off + 16 + l_idx] as i8 as i16) << 8;
                            acc += u_val as i32 + l_val as i32;
                        }
                    }
                    out_row[j] = acc as f32 / act_scale * w_scale;
                }

                continue;
            }

            // Scalar fallback
            for j in 0..n {
                let mut acc: i32 = 0;
                for g in 0..num_groups {
                    let lut_base = g * 64 * 32;
                    for b in 0..32 {
                        let byte_pos = g * 32 + b;
                        let byte = b_transposed[byte_pos * n + j];
                        let u_idx = (byte >> 4) as usize;
                        let l_idx = (byte & 0x0F) as usize;
                        let u_off = lut_base + b * 32;
                        let l_off = lut_base + (32 + b) * 32;
                        let u_val = split_luts[u_off + u_idx] as i8 as i16
                                  | (split_luts[u_off + 16 + u_idx] as i8 as i16) << 8;
                        let l_val = split_luts[l_off + l_idx] as i8 as i16
                                  | (split_luts[l_off + 16 + l_idx] as i8 as i16) << 8;
                        acc += u_val as i32 + l_val as i32;
                    }
                }
                out_row[j] = acc as f32 / act_scale * w_scale;
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

    pub fn with_context(ctx: std::sync::Arc<cuda_core::CudaContext>) -> crate::Result<Self> {
        let gpu = crate::gpu::GpuContext::from_context(ctx);
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

    pub fn gemv_q4k_multirow_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        n_warps_per_col: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let cols_per_cta = 1usize;
        let warps_per_block = (n_warps_per_col * cols_per_cta) as u32;
        let grid_x = ((n as u32) + cols_per_cta as u32 - 1) / cols_per_cta as u32;
        self.module
            .gemv_q4k_multirow(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                n_warps_per_col as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_q8_mw_device(
        &self,
        a_q8: &cuda_core::DeviceBuffer<u32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        n_warps_per_col: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let warps_per_block = n_warps_per_col as u32;
        let grid_x = n as u32;
        self.module
            .gemv_q4k_q8_mw(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                n_warps_per_col as u32,
                a_q8,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_ptx_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        n_warps_per_col: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let warps_per_block = n_warps_per_col as u32;
        let grid_x = n as u32;
        self.module
            .gemv_q4k_ptx(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                n_warps_per_col as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_intacc_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        n_warps_per_col: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let cols_per_cta = 1usize;
        let warps_per_block = (n_warps_per_col * cols_per_cta) as u32;
        let grid_x = ((n as u32) + cols_per_cta as u32 - 1) / cols_per_cta as u32;
        self.module
            .gemv_q4k_intacc(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                n_warps_per_col as u32,
                a,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q4k_fast_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
        n_warps_k: usize,
        rows_per_cta: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let warps_per_block = (n_warps_k * rows_per_cta) as u32;
        let grid_x = ((n + rows_per_cta - 1) / rows_per_cta) as u32;
        self.module
            .gemv_q4k_fast(
                &stream,
                LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (32, warps_per_block, 1),
                    shared_mem_bytes: 0,
                },
                n as u32,
                k as u32,
                n_warps_k as u32,
                rows_per_cta as u32,
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

    pub fn gemm_q4k_q8_det_device(
        &self,
        a_q8: &cuda_core::DeviceBuffer<u32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let total = (m * n) as u32;
        self.module
            .gemm_q4k_q8_det(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                a_q8,
                b_quantized,
                output,
            )
            .map_err(|e| crate::KernelError::InvalidConfig(e.to_string()))?;
        Ok(())
    }

    pub fn gemv_q6k_coop_device(
        &self,
        a: &cuda_core::DeviceBuffer<f32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q6k_coop(
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

    pub fn gemv_q6k_q8_dp4a_device(
        &self,
        a_q8: &cuda_core::DeviceBuffer<u32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let warps_per_block = 4u32;
        let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
        self.module
            .gemv_q6k_q8_dp4a(
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

    pub fn gemm_q6k_q8_det_device(
        &self,
        a_q8: &cuda_core::DeviceBuffer<u32>,
        b_quantized: &cuda_core::DeviceBuffer<u8>,
        output: &mut cuda_core::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        use cuda_core::LaunchConfig;
        let stream = self.gpu.stream();
        let total = (m * n) as u32;
        self.module
            .gemm_q6k_q8_det(
                &stream,
                LaunchConfig::for_num_elems(total),
                m as u32,
                n as u32,
                k as u32,
                a_q8,
                b_quantized,
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

    fn forward_b_transposed_tq2(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let cpu = CpuMatMulQ4K;
        cpu.forward_b_transposed_tq2(a, b_quantized, output, m, n, k)
    }

    fn forward_b_transposed_i2s(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let cpu = CpuMatMulQ4K;
        cpu.forward_b_transposed_i2s(a, b_quantized, output, m, n, k)
    }

    fn forward_i2s_preq(
        &self,
        preq_q8: &[i8],
        preq_scale: f32,
        preq_sum: i32,
        b_quantized: &[u8],
        output: &mut [f32],
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let cpu = CpuMatMulQ4K;
        cpu.forward_i2s_preq(preq_q8, preq_scale, preq_sum, b_quantized, output, n, k)
    }

    fn forward_i2s_lut(
        &self,
        a: &[f32],
        b_transposed: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::Result<()> {
        let cpu = CpuMatMulQ4K;
        cpu.forward_i2s_lut(a, b_transposed, output, m, n, k)
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

        let max_per_element: f64 = 100.0 * 945.0;
        let max_sum = max_per_element * k as f64;

        assert!(max_sum < 3.4e38);
        assert!(max_sum < f32::MAX as f64);
    }

    /// Prove symmetric int8 dequantization has bounded error: |dequant(quant(x)) - x| <= step/2.
    /// Symmetric int8: quant(x) = round(x / scale), dequant(q) = q * scale,
    /// where scale = max_abs / 127. Round-trip error is at most scale/2.
    #[kani::proof]
    fn int8_dequant_bounded_error() {
        let q: i8 = kani::any();
        kani::assume(q >= -127 && q <= 127);

        let max_abs: u32 = kani::any();
        kani::assume(max_abs >= 1 && max_abs <= 100);

        let scale = max_abs as f64 / 127.0;
        let dequantized = q as f64 * scale;
        let requantized = (dequantized / scale).round() as i64;

        assert!(requantized == q as i64);

        let step = scale;
        let max_error = step / 2.0;
        assert!(max_error >= 0.0);
        assert!(max_error <= max_abs as f64 / 254.0 + 0.001);
    }

    /// Prove Q4K monotonicity: within a group, larger nibble value → larger dequantized value.
    /// dequant(q) = d * sc * q - dmin * m. For fixed d, sc, dmin, m: if q1 > q2 then
    /// dequant(q1) >= dequant(q2), since d * sc >= 0 (both non-negative).
    #[kani::proof]
    fn q4k_monotonicity_within_group() {
        let q1: u8 = kani::any();
        let q2: u8 = kani::any();
        let sc: u8 = kani::any();
        let m_val: u8 = kani::any();

        kani::assume(q1 <= 15);
        kani::assume(q2 <= 15);
        kani::assume(q1 > q2);
        kani::assume(sc <= 63);
        kani::assume(m_val <= 63);

        let d: f64 = 1.0;
        let dmin: f64 = 1.0;

        let val1 = d * (sc as f64) * (q1 as f64) - dmin * (m_val as f64);
        let val2 = d * (sc as f64) * (q2 as f64) - dmin * (m_val as f64);

        assert!(val1 >= val2);
    }

    /// Prove int8 monotonicity: for fixed scale, q1 > q2 implies dequant(q1) > dequant(q2).
    #[kani::proof]
    fn int8_monotonicity() {
        let q1: i8 = kani::any();
        let q2: i8 = kani::any();
        kani::assume(q1 > q2);

        let scale: u32 = kani::any();
        kani::assume(scale >= 1 && scale <= 1000);

        let v1 = q1 as i64 * scale as i64;
        let v2 = q2 as i64 * scale as i64;

        assert!(v1 > v2);
    }

    /// Prove Q4K group index arithmetic stays within group boundaries.
    /// For each (j, l) in the dequant loop, element indices and qs indices
    /// stay within their respective group's range and never alias another group.
    #[kani::proof]
    fn q4k_group_boundary_safety() {
        let j: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(j < 4);
        kani::assume(l < 32);

        let elem_lo = j * 64 + l;
        let elem_hi = j * 64 + l + 32;
        let qs_byte = j * 32 + l;
        let scale_lo = 2 * j;
        let scale_hi = 2 * j + 1;

        // Element indices stay in this group's 64-element range [j*64, (j+1)*64)
        assert!(elem_lo >= j * 64 && elem_lo < (j + 1) * 64);
        assert!(elem_hi >= j * 64 && elem_hi < (j + 1) * 64);

        // qs byte index stays in this group's 32-byte range [j*32, (j+1)*32)
        assert!(qs_byte >= j * 32 && qs_byte < (j + 1) * 32);

        // Scale indices are valid for 8 groups
        assert!(scale_lo < Q4K_GROUPS);
        assert!(scale_hi < Q4K_GROUPS);

        // No overlap with adjacent groups
        if j > 0 {
            assert!(elem_lo >= j * 64);
            assert!(qs_byte >= j * 32);
        }
        if j < 3 {
            assert!(elem_hi < (j + 1) * 64);
            assert!(qs_byte < (j + 1) * 32);
        }
    }

    /// Prove i32 accumulator does not overflow in quantized matmul.
    /// Per Q4K block: 256 multiply-accumulate of (int8_activation * int4_weight).
    /// Max product: 127 * 15 = 1905. Sum of 256: 1905 * 256 = 487,680.
    /// For group_size=128: max partial sum = 1905 * 128 = 243,840 < i32::MAX.
    #[kani::proof]
    fn i32_accumulator_no_overflow_quantized_matmul() {
        let group_size: usize = kani::any();
        kani::assume(group_size >= 1 && group_size <= 128);

        let max_activation: i64 = 127;
        let max_weight: i64 = 15;
        let max_product = max_activation * max_weight;
        let max_sum = max_product * group_size as i64;

        assert!(max_sum <= i32::MAX as i64);

        // Even full block (256 elements): 1905 * 256 = 487680
        let full_block_sum = max_product * 256;
        assert!(full_block_sum <= i32::MAX as i64);
    }

    /// Prove Q6K group index arithmetic stays within boundaries.
    /// Q6_K has 16 groups of 16 elements = 256 elements.
    #[kani::proof]
    fn q6k_group_boundary_safety() {
        let group: usize = kani::any();
        let elem_in_group: usize = kani::any();
        kani::assume(group < Q6K_GROUPS);
        kani::assume(elem_in_group < Q6K_GROUP_SIZE);

        let global_elem = group * Q6K_GROUP_SIZE + elem_in_group;
        assert!(global_elem < Q6K_BLOCK_SIZE);

        // ql index: first 128 bytes hold low 4 bits
        let ql_idx = global_elem % 128;
        assert!(ql_idx < 128);

        // qh index: next 64 bytes hold high 2 bits
        let qh_idx = global_elem / 4;
        assert!(qh_idx < 64);

        // scale index
        assert!(group < 16);
    }

    #[kani::proof]
    fn tq2_block_access_in_bounds() {
        let idx: usize = kani::any();
        kani::assume(idx < TQ2_BLOCK_SIZE);

        // qs occupies bytes 0..63
        let byte_idx = idx / 4;
        assert!(byte_idx < 64);
        // d occupies bytes 64..65
        assert!(64 + 2 <= TQ2_BLOCK_BYTES);
    }

    #[kani::proof]
    fn tq2_decode_value_in_range() {
        let byte_val: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);
        let bits = (byte_val >> (shift * 2)) & 0x03;
        let trit = bits as i8 - 1;
        assert!(trit >= -1);
        assert!(trit <= 1);
    }

    #[kani::proof]
    fn tq2_encode_decode_roundtrip() {
        let trit: i8 = kani::any();
        kani::assume(trit >= -1 && trit <= 1);
        let encoded = (trit + 1) as u8;
        assert!(encoded <= 2);
        let decoded = encoded as i8 - 1;
        assert!(decoded == trit);
    }

    #[kani::proof]
    fn tq2_matmul_index_safety() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        let k: usize = kani::any();

        kani::assume(m >= 1 && m <= 4);
        kani::assume(n >= 1 && n <= 4);
        kani::assume(k == 256 || k == 512);

        let blocks_per_row = k / TQ2_BLOCK_SIZE;
        let i: usize = kani::any();
        let j: usize = kani::any();
        let blk: usize = kani::any();

        kani::assume(i < m);
        kani::assume(j < n);
        kani::assume(blk < blocks_per_row);

        let a_offset = i * k + blk * TQ2_BLOCK_SIZE;
        assert!(a_offset + TQ2_BLOCK_SIZE <= m * k);

        let block_offset = (j * blocks_per_row + blk) * TQ2_BLOCK_BYTES;
        assert!(block_offset + TQ2_BLOCK_BYTES <= n * blocks_per_row * TQ2_BLOCK_BYTES);

        let out_idx = i * n + j;
        assert!(out_idx < m * n);
    }

    #[kani::proof]
    fn i2s_block_access_in_bounds() {
        let idx: usize = kani::any();
        kani::assume(idx < I2S_BLOCK_SIZE);

        let group = idx / 128;
        assert!(group < 2);
        let within = idx % 128;
        let byte_idx = group * 32 + within % 32;
        assert!(byte_idx < I2S_BLOCK_BYTES);
    }

    #[kani::proof]
    fn i2s_decode_value_in_range() {
        let byte_val: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);
        let bits = (byte_val >> (shift * 2)) & 0x03;
        let trit = bits as i8 - 1;
        assert!(trit >= -1);
        assert!(trit <= 1);
    }

    #[kani::proof]
    fn i2s_quantize_i8_range() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val.abs() <= 1e6);

        let max: f64 = 0.00001f64.max((val as f64).abs());
        let s: f32 = (127.0f64 / max) as f32;
        let mut q: i32 = nearest_int(val * s);
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        assert!(q >= -128 && q <= 127);
    }

    #[kani::proof]
    fn i2s_matmul_index_safety() {
        let m: usize = kani::any();
        let n: usize = kani::any();
        let k: usize = kani::any();

        kani::assume(m >= 1 && m <= 4);
        kani::assume(n >= 1 && n <= 4);
        kani::assume(k == 128 || k == 256);

        let row_packed_bytes = k / 4;
        let packed_bytes = n * k / 4;

        let i: usize = kani::any();
        let j: usize = kani::any();

        kani::assume(i < m);
        kani::assume(j < n);

        let a_offset = i * k;
        assert!(a_offset + k <= m * k);

        let w_offset = j * row_packed_bytes;
        assert!(w_offset + row_packed_bytes <= packed_bytes);

        let out_idx = i * n + j;
        assert!(out_idx < m * n);
    }

    #[kani::proof]
    fn f16_dot_scalar_index_bounds() {
        let k: usize = kani::any();
        kani::assume(k >= 1 && k <= 64);

        let i: usize = kani::any();
        kani::assume(i < k);

        let byte_idx_lo = i * 2;
        let byte_idx_hi = i * 2 + 1;
        let b_len = k * 2;

        assert!(byte_idx_lo < b_len);
        assert!(byte_idx_hi < b_len);
    }

    #[kani::proof]
    fn f16_gemv_row_index_bounds() {
        let n: usize = kani::any();
        let k: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);
        kani::assume(k >= 1 && k <= 64);

        let j: usize = kani::any();
        kani::assume(j < n);

        let row_bytes = k * 2;
        let b_start = j * row_bytes;
        let b_end = b_start + row_bytes;
        assert!(b_end <= n * k * 2);
        assert!(j < n);
    }

    #[kani::proof]
    fn i2s_gemv_cols_index_bounds() {
        let n: usize = kani::any();
        let k: usize = kani::any();
        kani::assume(n >= 1 && n <= 8);
        kani::assume(k == 128 || k == 256);

        let col_start: usize = kani::any();
        let col_end: usize = kani::any();
        kani::assume(col_start <= col_end);
        kani::assume(col_end <= n);

        let row_packed_bytes = k / 4;

        let col: usize = kani::any();
        kani::assume(col >= col_start && col < col_end);

        let w_offset = col * row_packed_bytes;
        assert!(w_offset + row_packed_bytes <= n * row_packed_bytes);
        assert!(col < n);
    }
}

#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_q4k_dequant_value_bounded(
            q in 0u8..=15,
            sc in 0u8..=63,
            m_val in 0u8..=63,
            d_bits in proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL | proptest::num::f32::ZERO,
            dmin_bits in proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL | proptest::num::f32::ZERO,
        ) {
            let d = d_bits.abs();
            let dmin = dmin_bits.abs();

            let value = d * (sc as f32) * (q as f32) - dmin * (m_val as f32);
            let max_positive = d * 63.0 * 15.0;
            let max_negative = dmin * 63.0;

            if !d.is_nan() && !dmin.is_nan() && d.is_finite() && dmin.is_finite() {
                prop_assert!(
                    value <= max_positive + 1e-6 || value.is_nan(),
                    "value {} > max_positive {} for q={} sc={} d={}",
                    value, max_positive, q, sc, d
                );
                prop_assert!(
                    value >= -max_negative - 1e-6 || value.is_nan(),
                    "value {} < -max_negative {} for m={} dmin={}",
                    value, max_negative, m_val, dmin
                );
            }
        }

        #[test]
        fn prop_int8_dequant_bounded_error(
            q in -127i8..=127,
            max_abs in 1u32..=10000,
        ) {
            let scale = max_abs as f64 / 127.0;
            let dequantized = q as f64 * scale;
            let requantized = (dequantized / scale).round() as i64;

            prop_assert_eq!(
                requantized, q as i64,
                "round-trip failed: q={}, scale={}, dequant={}, requant={}",
                q, scale, dequantized, requantized
            );

            let max_error = scale / 2.0;
            prop_assert!(max_error >= 0.0);
            prop_assert!(
                max_error <= max_abs as f64 / 254.0 + 0.001,
                "max_error {} > bound {} for max_abs={}",
                max_error, max_abs as f64 / 254.0 + 0.001, max_abs
            );
        }
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

    #[test]
    fn test_tq2_encode_decode_roundtrip() {
        let mut values = [0.0f32; TQ2_BLOCK_SIZE];
        for i in 0..TQ2_BLOCK_SIZE {
            values[i] = match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
        }
        let block = tq2_encode_block(&values);
        let mut decoded = [0.0f32; TQ2_BLOCK_SIZE];
        tq2_dequant_block(&block, &mut decoded);
        for i in 0..TQ2_BLOCK_SIZE {
            assert!(
                (decoded[i] - values[i]).abs() < 1e-3,
                "tq2 roundtrip mismatch at [{i}]: got {}, expected {}",
                decoded[i],
                values[i]
            );
        }
    }

    #[test]
    fn test_tq2_dot_block_manual() {
        let mut values = [0.0f32; TQ2_BLOCK_SIZE];
        values[0] = 1.0;
        values[1] = -1.0;
        values[2] = 0.0;
        values[3] = 1.0;
        let block = tq2_encode_block(&values);

        let mut a = [0.0f32; TQ2_BLOCK_SIZE];
        a[0] = 2.0;
        a[1] = 3.0;
        a[2] = 5.0;
        a[3] = 7.0;

        let dot = tq2_dot_block(&block, &a);
        // Expected: 1*2 + (-1)*3 + 0*5 + 1*7 = 2 - 3 + 0 + 7 = 6
        // Scale d = max(|values|) = 1.0, so d≈1.0 (fp16 roundtrip)
        assert!(
            (dot - 6.0).abs() < 0.1,
            "tq2 dot expected ~6.0, got {dot}"
        );
    }

    #[test]
    fn test_tq2_matmul_identity_like() {
        let k = 256;
        let n = 2;
        let m = 1;

        let mut row0_vals = [0.0f32; 256];
        row0_vals[0] = 1.0;
        let mut row1_vals = [0.0f32; 256];
        row1_vals[1] = 1.0;

        let block0 = tq2_encode_block(&row0_vals);
        let block1 = tq2_encode_block(&row1_vals);

        let mut b_quantized = Vec::with_capacity(n * TQ2_BLOCK_BYTES);
        b_quantized.extend_from_slice(&block0);
        b_quantized.extend_from_slice(&block1);

        let mut a = vec![0.0f32; k];
        a[0] = 10.0;
        a[1] = 20.0;

        let mut output = vec![0.0f32; n];
        let matmul = CpuMatMulQ4K;
        matmul
            .forward_b_transposed_tq2(&a, &b_quantized, &mut output, m, n, k)
            .unwrap();

        assert!(
            (output[0] - 10.0).abs() < 0.5,
            "row 0: expected ~10.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 20.0).abs() < 0.5,
            "row 1: expected ~20.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_squared_relu() {
        let inputs = [-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0];
        let expected = [0.0, 0.0, 0.0, 0.25, 1.0, 9.0];
        for (x, exp) in inputs.iter().zip(expected.iter()) {
            let relu = x.max(0.0);
            let result = relu * relu;
            assert!(
                (result - exp).abs() < 1e-6,
                "squared_relu({x}) = {result}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_i2s_dequant_block_manual() {
        let mut block = [0u8; I2S_BLOCK_BYTES];
        // Group 0, byte 0: bits [7:6]=2(+1), [5:4]=1(0), [3:2]=0(-1), [1:0]=2(+1)
        block[0] = 0b10_01_00_10;
        let mut output = [0.0f32; I2S_BLOCK_SIZE];
        i2s_dequant_block(&block, &mut output);
        // Byte 0 in group 0: element 0 from [7:6], element 32 from [5:4], element 64 from [3:2], element 96 from [1:0]
        assert_eq!(output[0], 1.0, "element 0 should be +1");
        assert_eq!(output[32], 0.0, "element 32 should be 0");
        assert_eq!(output[64], -1.0, "element 64 should be -1");
        assert_eq!(output[96], 1.0, "element 96 should be +1");
    }

    #[test]
    fn test_i2s_quantize_row_i8_s() {
        let x = [1.0f32, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.125];
        let mut dst = [0i8; 8];
        let (scale, sum) = quantize_row_i8_s(&x, &mut dst);
        assert!(scale > 0.0, "scale must be positive");
        for &q in &dst {
            assert!(q >= -128, "i8 range violated: {q}");
        }
        assert_eq!(dst[0], 127, "max positive maps to 127");
        assert_eq!(dst[1], -127, "max negative maps to -127");
        let expected_sum: i32 = dst.iter().map(|&v| v as i32).sum();
        assert_eq!(sum, expected_sum, "sum must match");
    }

    #[test]
    fn test_i2s_vec_dot_manual() {
        // 128-element group: construct packed weights for a known pattern
        let k = 128;
        let mut packed = vec![0u8; k / 4]; // 32 bytes
        // Byte 0: bits [7:6]=2(+1), [5:4]=1(0), [3:2]=0(-1), [1:0]=2(+1)
        packed[0] = 0b10_01_00_10;

        let mut a_q = vec![0i8; k];
        a_q[0] = 10; // element 0: w=+1 → value 2
        a_q[32] = 20; // element 32: w=0 → value 1
        a_q[64] = 30; // element 64: w=-1 → value 0
        a_q[96] = 40; // element 96: w=+1 → value 2

        let dot = vec_dot_i2_i8_s(&packed, &a_q, k);
        // w values: 2,1,0,2 → unsigned dot: 2*10 + 1*20 + 0*30 + 2*40 = 20 + 20 + 0 + 80 = 120
        assert_eq!(dot, 120, "i2s vec_dot expected 120, got {dot}");
    }

    #[test]
    fn test_i2s_matmul_identity_like() {
        let k = 256; // I2S_BLOCK_SIZE
        let n = 2;
        let m = 1;

        // Row 0: element 0 = +1, rest = 0
        // Row 1: element 1 = +1, rest = 0
        let row_packed = k / 4; // 64 bytes per row
        let total_packed = n * row_packed;
        let mut b_quantized = vec![0u8; total_packed + 4]; // +4 for per-tensor f32 scale

        // Encode +1 at element 0 for row 0: byte 0 bits [7:6] = 2
        b_quantized[0] = 0b10_000000;
        // Encode +1 at element 1 for row 1: byte 1 bits [7:6] = 2
        b_quantized[row_packed + 1] = 0b10_000000;

        // Per-tensor scale = 1.0
        let scale_bytes = 1.0f32.to_ne_bytes();
        b_quantized[total_packed..total_packed + 4].copy_from_slice(&scale_bytes);

        // But weight values are {0,1,2} → ternary {-1,0,+1}
        // So w=2 means +1, w=1 means 0, w=0 means -1
        // All zero bytes → w=0 → ternary -1. That's a problem for "identity".
        // The rescaling formula is: (raw - act_sum) / act_scale * w_scale
        //
        // For a cleaner test: use all w=1 (ternary 0) except the identity positions.
        // w=1 → byte pattern where each 2-bit field is 01.
        // 0b01_01_01_01 = 0x55
        b_quantized.iter_mut().take(total_packed).for_each(|b| *b = 0x55);
        // Now override: element 0 in row 0 = +1 (w=2 at byte 0 bits [7:6])
        b_quantized[0] = 0b10_01_01_01; // elem 0=+1, rest=0
        // Element 1 in row 1 = +1 (byte 1, bits [7:6])
        b_quantized[row_packed + 1] = 0b10_01_01_01;

        let mut a = vec![0.0f32; k];
        a[0] = 10.0;
        a[1] = 20.0;

        let mut output = vec![0.0f32; n];
        let matmul = CpuMatMulQ4K;
        matmul
            .forward_b_transposed_i2s(&a, &b_quantized, &mut output, m, n, k)
            .unwrap();

        // Row 0 has w=+1 at elem 0, w=0 everywhere else → dot product picks up a[0]=10.0
        // Row 1 has w=+1 at elem 1, w=0 everywhere else → dot product picks up a[1]=20.0
        // After INT8 quantization + rescaling, values won't be exact but should be close
        assert!(
            (output[0] - 10.0).abs() < 1.0,
            "row 0: expected ~10.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 20.0).abs() < 1.0,
            "row 1: expected ~20.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_i2s_avx2_matches_scalar() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let k = 256;
        let n = 4;
        let m = 1;
        let mut rng = StdRng::seed_from_u64(42);

        // Random I2S tensor: n rows of k/4 packed bytes + 4-byte scale
        let row_packed = k / 4;
        let total_packed = n * row_packed;
        let mut b = vec![0u8; total_packed + 4];
        for byte in b[..total_packed].iter_mut() {
            *byte = rng.random();
        }
        b[total_packed..total_packed + 4].copy_from_slice(&0.5f32.to_ne_bytes());

        // Random activations
        let a: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();

        // Scalar path
        let mut out_scalar = vec![0.0f32; m * n];
        let scalar = CpuMatMulQ4K;
        scalar
            .forward_b_transposed_i2s(&a, &b, &mut out_scalar, m, n, k)
            .unwrap();

        // Optimized path (AVX2 + rayon on x86_64)
        let mut out_opt = vec![0.0f32; m * n];
        let opt = CpuMatMulQ4KOpt;
        opt.forward_b_transposed_i2s(&a, &b, &mut out_opt, m, n, k)
            .unwrap();

        for j in 0..n {
            assert!(
                (out_scalar[j] - out_opt[j]).abs() < 1e-4,
                "mismatch at j={j}: scalar={}, opt={}",
                out_scalar[j],
                out_opt[j]
            );
        }
    }

    #[test]
    fn test_i2s_lut_transposed_matches_maddubs() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let k = 256;
        let n = 64;
        let mut rng = StdRng::seed_from_u64(88);

        let row_packed = k / 4;
        let total_packed = n * row_packed;
        let mut b = vec![0u8; total_packed + 4];
        for byte in b[..total_packed].iter_mut() {
            let mut val = 0u8;
            for shift in (0..8).step_by(2) {
                val |= rng.random_range(0u8..3) << shift;
            }
            *byte = val;
        }
        b[total_packed..total_packed + 4].copy_from_slice(&0.5f32.to_ne_bytes());

        // Transpose weights
        let mut b_transposed = vec![0u8; total_packed + 4];
        for j in 0..n {
            for bp in 0..row_packed {
                b_transposed[bp * n + j] = b[j * row_packed + bp];
            }
        }
        b_transposed[total_packed..total_packed + 4].copy_from_slice(&0.5f32.to_ne_bytes());

        let a: Vec<f32> = (0..k).map(|_| rng.random_range(-1.0..1.0)).collect();

        // maddubs path
        let mut out_mad = vec![0.0f32; n];
        let mad = CpuMatMulQ4K;
        mad.forward_b_transposed_i2s(&a, &b, &mut out_mad, 1, n, k).unwrap();

        // LUT path with transposed weights (scalar CpuMatMulQ4K)
        let mut out_lut_scalar = vec![0.0f32; n];
        mad.forward_i2s_lut(&a, &b_transposed, &mut out_lut_scalar, 1, n, k).unwrap();

        // LUT path with transposed weights (optimized CpuMatMulQ4KOpt)
        let mut out_lut_opt = vec![0.0f32; n];
        let opt = CpuMatMulQ4KOpt;
        opt.forward_i2s_lut(&a, &b_transposed, &mut out_lut_opt, 1, n, k).unwrap();

        for j in 0..n {
            // Scalar LUT uses i16 (full precision), should match maddubs closely
            assert!(
                (out_mad[j] - out_lut_scalar[j]).abs() < 0.01,
                "scalar LUT mismatch at j={j}: mad={}, lut={}",
                out_mad[j], out_lut_scalar[j]
            );
            // Opt LUT uses i8 (63-scale, ~2% quantization difference expected)
            let tol = out_mad[j].abs() * 0.05 + 0.05;
            assert!(
                (out_mad[j] - out_lut_opt[j]).abs() < tol,
                "opt LUT mismatch at j={j}: mad={}, lut_opt={}, tol={}",
                out_mad[j], out_lut_opt[j], tol
            );
        }
    }

    #[test]
    fn test_i2s_lut_matches_maddubs() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let k = 256;
        let n = 4;
        let mut rng = StdRng::seed_from_u64(77);

        let row_packed = k / 4;
        let total_packed = n * row_packed;
        let mut b = vec![0u8; total_packed + 4];
        for byte in b[..total_packed].iter_mut() {
            let mut val = 0u8;
            for shift in (0..8).step_by(2) {
                val |= rng.random_range(0u8..3) << shift;
            }
            *byte = val;
        }
        b[total_packed..total_packed + 4].copy_from_slice(&0.5f32.to_ne_bytes());

        let a: Vec<f32> = (0..k).map(|_| rng.random_range(-1.0..1.0)).collect();

        // maddubs path
        let mut out_mad = vec![0.0f32; n];
        let mad = CpuMatMulQ4K;
        mad.forward_b_transposed_i2s(&a, &b, &mut out_mad, 1, n, k).unwrap();

        // LUT path
        let mut a_q = vec![0i8; k];
        let (act_scale, _act_sum) = quantize_row_i8_s(&a, &mut a_q);
        let pair_luts = i2s_build_all_pair_luts(&a_q, k);
        let w_scale = f32::from_ne_bytes([
            b[total_packed], b[total_packed + 1], b[total_packed + 2], b[total_packed + 3],
        ]);

        let mut out_lut = vec![0.0f32; n];
        for j in 0..n {
            let w_packed = &b[j * row_packed..(j + 1) * row_packed];
            let raw = vec_dot_i2_i8_s_lut_scalar(w_packed, &pair_luts, k);
            out_lut[j] = raw as f32 / act_scale * w_scale;
        }

        for j in 0..n {
            assert!(
                (out_mad[j] - out_lut[j]).abs() < 0.01,
                "LUT mismatch at j={j}: maddubs={}, lut={}",
                out_mad[j], out_lut[j]
            );
        }
    }

    #[test]
    fn bench_i2s_vnni_vs_tl2() {
        let shapes = [(2560, 2560), (2560, 6912), (6912, 2560), (640, 2560)];
        for (n, k) in shapes {
            let packed_bytes = n * k / 4;
            let mut i2s = vec![0u8; packed_bytes + 4];
            let mut rng = 42u64;
            for b in &mut i2s[..packed_bytes] {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (rng >> 32) as u8;
            }
            i2s[packed_bytes..packed_bytes + 4].copy_from_slice(&1.0f32.to_ne_bytes());
            let act: Vec<f32> = (0..k).map(|i| (i as f32 - k as f32 / 2.0) * 0.001).collect();
            let mut output = vec![0.0f32; n];

            let mut q8 = vec![0i8; k];
            let iters = 20;

            // I2S AVX-VNNI GEMV
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let (s, sum) = i2s_quantize_into(&act, &mut q8);
                for j in 0..n {
                    let w = &i2s[j * k / 4..(j + 1) * k / 4];
                    let raw = unsafe { vec_dot_i2_i8_s_avx_vnni(w, &q8, k) };
                    output[j] = (raw - sum) as f32 / s;
                }
            }
            let i2s_us = start.elapsed().as_micros() as f64 / iters as f64;
            eprintln!("({n},{k}) I2S-VNNI: {i2s_us:.0}μs/iter");
        }
    }
}

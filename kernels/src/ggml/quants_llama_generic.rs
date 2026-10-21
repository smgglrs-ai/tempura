//! Direct Rust translation of llama.cpp ggml/src/ggml-cpu/quants.c
//! Generic (scalar, non-architecture-specific) fallback implementations.
//! Algorithm identical to llama.cpp; no SIMD, no optimization.

use super::blocks::*;
use super::helpers::{f16_to_f32, e8m0_to_f32_half, ue4m3_to_f32};
use super::iq_tables::*;

const QK_K: usize = 256;
const QK8_0: usize = 32;
const QK8_1: usize = 32;
const QK4_NL: usize = 32;
const QK2_0: usize = 64;
const QK1_0: usize = 128;
const QK_MXFP4: usize = 32;
const QK_NVFP4: usize = 64;
const QK_NVFP4_SUB: usize = 16;

pub fn ggml_vec_dot_q1_0_q8_0_generic(n: usize, x: &[BlockQ1_0], y: &[BlockQ8_0]) -> f32 {
    let qk = QK1_0;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d0 = f16_to_f32(x[i].d);
        let mut sumi = 0.0f32;

        for k in 0..4 {
            let yb = &y[i * 4 + k];
            let d1 = f16_to_f32(yb.d);
            let mut sumi_block = 0i32;

            let bits = &x[i].qs[k * 4..k * 4 + 4];
            let qy = &yb.qs;

            for b in 0..4usize {
                let mask = bits[b] as u32;
                let base = b * 8;
                sumi_block += if mask & 0x01 != 0 { qy[base + 0] as i32 } else { -(qy[base + 0] as i32) };
                sumi_block += if mask & 0x02 != 0 { qy[base + 1] as i32 } else { -(qy[base + 1] as i32) };
                sumi_block += if mask & 0x04 != 0 { qy[base + 2] as i32 } else { -(qy[base + 2] as i32) };
                sumi_block += if mask & 0x08 != 0 { qy[base + 3] as i32 } else { -(qy[base + 3] as i32) };
                sumi_block += if mask & 0x10 != 0 { qy[base + 4] as i32 } else { -(qy[base + 4] as i32) };
                sumi_block += if mask & 0x20 != 0 { qy[base + 5] as i32 } else { -(qy[base + 5] as i32) };
                sumi_block += if mask & 0x40 != 0 { qy[base + 6] as i32 } else { -(qy[base + 6] as i32) };
                sumi_block += if mask & 0x80 != 0 { qy[base + 7] as i32 } else { -(qy[base + 7] as i32) };
            }

            sumi += d1 * sumi_block as f32;
        }

        sumf += d0 * sumi;
    }

    sumf
}

pub fn ggml_vec_dot_q2_0_q8_0_generic(n: usize, x: &[BlockQ2_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK2_0;
    assert_eq!(n % QK2_0, 0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d0 = f16_to_f32(x[i].d);
        let mut sumi = 0.0f32;

        for k in 0..2usize {
            let yb = &y[i * 2 + k];
            let d1 = f16_to_f32(yb.d);
            let mut sumi_block = 0i32;

            let qs = &x[i].qs[k * 8..k * 8 + 8];
            let qy = &yb.qs;

            for b in 0..8usize {
                let byte = qs[b];
                sumi_block += ((((byte >> 0) & 3) as i32) - 1) * qy[b * 4 + 0] as i32;
                sumi_block += ((((byte >> 2) & 3) as i32) - 1) * qy[b * 4 + 1] as i32;
                sumi_block += ((((byte >> 4) & 3) as i32) - 1) * qy[b * 4 + 2] as i32;
                sumi_block += ((((byte >> 6) & 3) as i32) - 1) * qy[b * 4 + 3] as i32;
            }

            sumi += d1 * sumi_block as f32;
        }

        sumf += d0 * sumi;
    }

    sumf
}

pub fn ggml_vec_dot_q4_0_q8_0_generic(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let v0 = ((x[ib].qs[j] & 0x0F) as i32) - 8;
            let v1 = ((x[ib].qs[j] >> 4) as i32) - 8;
            sumi0 += v0 * y[ib].qs[j] as i32;
            sumi1 += v1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
    }

    sumf
}

pub fn ggml_vec_dot_q4_1_q8_1_generic(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    let qk = QK8_1;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let v0 = (x[ib].qs[j] & 0x0F) as i32;
            let v1 = (x[ib].qs[j] >> 4) as i32;
            sumi0 += v0 * y[ib].qs[j] as i32;
            sumi1 += v1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += (f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d)) * sumi as f32
              + f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);
    }

    sumf
}

pub fn ggml_vec_dot_mxfp4_q8_0_generic(n: usize, x: &[BlockMxfp4], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(n % QK_MXFP4, 0);
    let nb = n / QK_MXFP4;

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let d = f16_to_f32(y[ib].d) * e8m0_to_f32_half(x[ib].e);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..QK_MXFP4 / 2 {
            sumi1 += y[ib].qs[j] as i32 * KVALUES_MXFP4[(x[ib].qs[j] & 0xf) as usize] as i32;
            sumi2 += y[ib].qs[j + QK_MXFP4 / 2] as i32 * KVALUES_MXFP4[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }

    sumf
}

pub fn ggml_vec_dot_nvfp4_q8_0_generic(n: usize, x: &[BlockNvfp4], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(n % QK_NVFP4, 0);
    let nb = n / QK_NVFP4;

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        for s_idx in 0..4usize {
            let d = ue4m3_to_f32(x[ib].d[s_idx]);
            let q8_block = s_idx / 2;
            let q8_off = (s_idx % 2) * QK_NVFP4_SUB;
            let dy = f16_to_f32(y[2 * ib + q8_block].d);

            let mut sumi_lo = 0i32;
            let mut sumi_hi = 0i32;
            for j in 0..QK_NVFP4_SUB / 2 {
                let qv = x[ib].qs[s_idx * (QK_NVFP4_SUB / 2) + j];
                sumi_lo += y[2 * ib + q8_block].qs[q8_off + j] as i32
                         * KVALUES_MXFP4[(qv & 0xf) as usize] as i32;
                sumi_hi += y[2 * ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB / 2] as i32
                         * KVALUES_MXFP4[(qv >> 4) as usize] as i32;
            }

            sumf += dy * d * (sumi_lo + sumi_hi) as f32;
        }
    }

    sumf
}

pub fn ggml_vec_dot_q5_0_q8_0_generic(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let mut qh = 0u32;
        let qh_bytes = x[ib].qh;
        qh = u32::from_le_bytes(qh_bytes);

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let xh_0 = (((qh & (1u32 << (j + 0))) >> (j + 0)) << 4) as u8;
            let xh_1 = (((qh & (1u32 << (j + 16))) >> (j + 12))) as u8;

            let x0 = (((x[ib].qs[j] & 0x0F) | xh_0) as i8 as i32) - 16;
            let x1 = (((x[ib].qs[j] >> 4) | xh_1) as i8 as i32) - 16;

            sumi0 += x0 * y[ib].qs[j] as i32;
            sumi1 += x1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += (f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d)) * sumi as f32;
    }

    sumf
}

pub fn ggml_vec_dot_q5_1_q8_1_generic(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
    let qk = QK8_1;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let qh = u32::from_le_bytes(x[ib].qh);

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let xh_0 = (((qh >> (j + 0)) << 4) & 0x10) as u8;
            let xh_1 = (((qh >> (j + 12))) & 0x10) as u8;

            let x0 = ((x[ib].qs[j] & 0xF) | xh_0) as i32;
            let x1 = ((x[ib].qs[j] >> 4) | xh_1) as i32;

            sumi0 += x0 * y[ib].qs[j] as i32;
            sumi1 += x1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += (f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d)) * sumi as f32
              + f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);
    }

    sumf
}

pub fn ggml_vec_dot_q8_0_q8_0_generic(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    assert_eq!(n % qk, 0);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let mut sumi = 0i32;
        for j in 0..qk {
            sumi += x[ib].qs[j] as i32 * y[ib].qs[j] as i32;
        }
        sumf += sumi as f32 * (f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d));
    }

    sumf
}

pub fn ggml_vec_dot_tq1_0_q8_k_generic(n: usize, x: &[BlockTQ1_0], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;

    let pow3: [u8; 6] = [1, 3, 9, 27, 81, 243];

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let mut sum = 0i32;

        // First loop: j in 0..(48 - 48%32) = 0..32, step 32 -> j=0
        let qs_len = 48usize; // x[i].qs: [u8; 48]
        let first_limit = qs_len - qs_len % 32;
        let mut j = 0;
        while j < first_limit {
            for l in 0..5usize {
                for m in 0..32usize {
                    let q = x[i].qs[j + m].wrapping_mul(pow3[l]);
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    sum += (xi - 1) * y[i].qs[j * 5 + l * 32 + m] as i32;
                }
            }
            j += 32;
        }
        // Second loop: j from first_limit to qs_len, step 16
        j = first_limit;
        while j < qs_len {
            for l in 0..5usize {
                for m in 0..16usize {
                    let q = x[i].qs[j + m].wrapping_mul(pow3[l]);
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    sum += (xi - 1) * y[i].qs[j * 5 + l * 16 + m] as i32;
                }
            }
            j += 16;
        }

        // qh loop: 4 bytes, 4 levels each
        let qh_len = 4usize; // x[i].qh: [u8; 4]
        for l in 0..4usize {
            for j2 in 0..qh_len {
                let q = x[i].qh[j2].wrapping_mul(pow3[l]);
                let xi = ((q as u16 * 3) >> 8) as i32;
                sum += (xi - 1) * y[i].qs[qs_len * 5 + l * qh_len + j2] as i32;
            }
        }

        sumf += sum as f32 * (f16_to_f32(x[i].d) * y[i].d);
    }

    sumf
}

pub fn ggml_vec_dot_tq2_0_q8_k_generic(n: usize, x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let mut sumi = 0i32;

        let qs_len = 64usize; // x[i].qs: [u8; 64]
        let mut j = 0;
        while j < qs_len {
            for l in 0..4usize {
                for k in 0..32usize {
                    sumi += y[i].qs[j * 4 + l * 32 + k] as i32
                          * (((x[i].qs[j + k] >> (l * 2)) & 3) as i32 - 1);
                }
            }
            j += 32;
        }

        let d = y[i].d * f16_to_f32(x[i].d);
        sumf += sumi as f32 * d;
    }

    sumf
}

pub fn ggml_vec_dot_q2_k_q8_k_generic(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let q2 = &x[i].qs;
        let q8 = &y[i].qs;
        let sc = &x[i].scales;

        let mut summs = 0i32;
        for j in 0..16usize {
            summs += y[i].bsums[j] as i32 * (sc[j] >> 4) as i32;
        }

        let dall = y[i].d * f16_to_f32(x[i].d);
        let dmin = y[i].d * f16_to_f32(x[i].dmin);

        let mut isum = 0i32;
        let mut is = 0usize;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for _k in 0..QK_K / 128 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let d = (sc[is] & 0xF) as i32;
                is += 1;
                let mut isuml = 0i32;
                for l in 0..16usize {
                    isuml += q8[q8_off + l] as i32 * ((q2[q2_off + l] >> shift) & 3) as i32;
                }
                isum += d * isuml;

                let d2 = (sc[is] & 0xF) as i32;
                is += 1;
                isuml = 0;
                for l in 16..32usize {
                    isuml += q8[q8_off + l] as i32 * ((q2[q2_off + l] >> shift) & 3) as i32;
                }
                isum += d2 * isuml;

                shift += 2;
                q8_off += 32;
            }
            q2_off += 32;
        }

        sumf += dall * isum as f32 - dmin * summs as f32;
    }

    sumf
}

pub fn ggml_vec_dot_q3_k_q8_k_generic(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut auxs = [0u32; 4];

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q3 = &x[i].qs;
        let hm = &x[i].hmask;
        let q8 = &y[i].qs;

        for v in aux32.iter_mut() { *v = 0; }

        let mut a_off = 0usize;
        let mut m = 1u8;
        let mut j = 0usize;
        while j < QK_K {
            for l in 0..32usize { aux8[a_off + l] = (q3[l + j / 4] & 3) as i8; }
            for l in 0..32usize {
                if hm[l] & m == 0 { aux8[a_off + l] -= 4; }
            }
            a_off += 32; m <<= 1;

            for l in 0..32usize { aux8[a_off + l] = ((q3[l + j / 4] >> 2) & 3) as i8; }
            for l in 0..32usize {
                if hm[l] & m == 0 { aux8[a_off + l] -= 4; }
            }
            a_off += 32; m <<= 1;

            for l in 0..32usize { aux8[a_off + l] = ((q3[l + j / 4] >> 4) & 3) as i8; }
            for l in 0..32usize {
                if hm[l] & m == 0 { aux8[a_off + l] -= 4; }
            }
            a_off += 32; m <<= 1;

            for l in 0..32usize { aux8[a_off + l] = ((q3[l + j / 4] >> 6) & 3) as i8; }
            for l in 0..32usize {
                if hm[l] & m == 0 { aux8[a_off + l] -= 4; }
            }
            a_off += 32; m <<= 1;

            j += 128;
        }

        // Scale unpacking: memcpy(auxs, x[i].scales, 12)
        let sc = &x[i].scales; // [u8; 12]
        auxs[0] = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
        auxs[1] = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
        auxs[2] = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // scales is reinterpretation of auxs as [i8] (little-endian bytes)
        let scale_byte = |idx: usize| -> i8 {
            ((auxs[idx / 4] >> (8 * (idx % 4))) & 0xff) as i8
        };

        a_off = 0;
        let mut q8_off = 0usize;
        for jj in 0..QK_K / 16 {
            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            let sc_j = scale_byte(jj) as i32 - 32;
            for l in 0..8usize { aux32[l] += sc_j * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += sc_j * aux16[l] as i32; }
            q8_off += 8; a_off += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8usize { sums[l] += d * aux32[l] as f32; }
    }

    for l in 0..8usize { sumf += sums[l]; }
    sumf
}

pub fn ggml_vec_dot_q4_k_q8_k_generic(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].qs;
        let q8 = &y[i].qs;

        for v in aux32.iter_mut() { *v = 0; }

        // Unpack nibbles
        let mut a_off = 0usize;
        let mut q4_off = 0usize;
        let mut _jj = 0usize;
        while _jj < QK_K / 64 {
            for l in 0..32usize { aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8; }
            a_off += 32;
            for l in 0..32usize { aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8; }
            a_off += 32;
            q4_off += 32;
            _jj += 1;
        }

        // Scale/min unpacking (from x[i].scales[12])
        let sc_raw = &x[i].scales; // [u8; 12]
        utmp[0] = u32::from_le_bytes([sc_raw[0], sc_raw[1], sc_raw[2], sc_raw[3]]);
        utmp[1] = u32::from_le_bytes([sc_raw[4], sc_raw[5], sc_raw[6], sc_raw[7]]);
        utmp[2] = u32::from_le_bytes([sc_raw[8], sc_raw[9], sc_raw[10], sc_raw[11]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // scales = (const uint8_t*)&utmp[0], mins = (const uint8_t*)&utmp[2]
        let scale_byte = |idx: usize| -> u8 {
            ((utmp[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8
        };
        let min_byte = |idx: usize| -> u8 {
            // mins starts at utmp[2], so base offset is 8
            ((utmp[(idx + 8) / 4] >> (8 * ((idx + 8) % 4))) & 0xff) as u8
        };

        let mut sumi = 0i32;
        for jj in 0..QK_K / 16 {
            sumi += y[i].bsums[jj] as i32 * min_byte(jj / 2) as i32;
        }

        a_off = 0;
        let mut q8_off = 0usize;
        let mut is = 0usize;
        for _jj in 0..QK_K / 32 {
            let scale = scale_byte(is) as i32;
            is += 1;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8usize { sums[l] += d * aux32[l] as f32; }

        let dmin = f16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi as f32;
    }

    for l in 0..8usize { sumf += sums[l]; }
    sumf
}

pub fn ggml_vec_dot_q5_k_q8_k_generic(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut utmp = [0u32; 4];

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].qs;
        let hm = &x[i].qh;
        let q8 = &y[i].qs;

        for v in aux32.iter_mut() { *v = 0; }

        let mut a_off = 0usize;
        let mut q4_off = 0usize;
        let mut m = 1u8;
        let mut _jj = 0usize;
        while _jj < QK_K / 64 {
            for l in 0..32usize { aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8; }
            for l in 0..32usize { if hm[l] & m != 0 { aux8[a_off + l] += 16; } }
            a_off += 32; m <<= 1;

            for l in 0..32usize { aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8; }
            for l in 0..32usize { if hm[l] & m != 0 { aux8[a_off + l] += 16; } }
            a_off += 32; m <<= 1;

            q4_off += 32;
            _jj += 1;
        }

        let sc_raw = &x[i].scales; // [u8; 12]
        utmp[0] = u32::from_le_bytes([sc_raw[0], sc_raw[1], sc_raw[2], sc_raw[3]]);
        utmp[1] = u32::from_le_bytes([sc_raw[4], sc_raw[5], sc_raw[6], sc_raw[7]]);
        utmp[2] = u32::from_le_bytes([sc_raw[8], sc_raw[9], sc_raw[10], sc_raw[11]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let scale_byte = |idx: usize| -> u8 {
            ((utmp[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8
        };
        let min_byte = |idx: usize| -> u8 {
            ((utmp[(idx + 8) / 4] >> (8 * ((idx + 8) % 4))) & 0xff) as u8
        };

        let mut sumi = 0i32;
        for jj in 0..QK_K / 16 {
            sumi += y[i].bsums[jj] as i32 * min_byte(jj / 2) as i32;
        }

        a_off = 0;
        let mut q8_off = 0usize;
        let mut is = 0usize;
        for _jj in 0..QK_K / 32 {
            let scale = scale_byte(is) as i32;
            is += 1;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8usize { sums[l] += d * aux32[l] as f32; }

        let dmin = f16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi as f32;
    }

    for l in 0..8usize { sumf += sums[l]; }
    sumf
}

pub fn ggml_vec_dot_q6_k_q8_k_generic(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].ql;
        let qh = &x[i].qh;
        let q8 = &y[i].qs;

        for v in aux32.iter_mut() { *v = 0; }

        let mut a_off = 0usize;
        let mut q4_off = 0usize;
        let mut qh_off = 0usize;
        let mut j = 0usize;
        while j < QK_K {
            for l in 0..32usize {
                aux8[a_off + l + 0]  = (((q4[q4_off + l + 0]  & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8) - 32;
                aux8[a_off + l + 32] = (((q4[q4_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8) - 32;
                aux8[a_off + l + 64] = (((q4[q4_off + l + 0]  >> 4)  | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8) - 32;
                aux8[a_off + l + 96] = (((q4[q4_off + l + 32] >> 4)  | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8) - 32;
            }
            a_off += 128;
            q4_off += 64;
            qh_off += 32;
            j += 128;
        }

        a_off = 0;
        let mut q8_off = 0usize;
        let mut is = 0usize;
        for _jj in 0..QK_K / 16 {
            let scale = x[i].scales[is] as i32;
            is += 1;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;

            for l in 0..8usize { aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16; }
            for l in 0..8usize { aux32[l] += scale * aux16[l] as i32; }
            q8_off += 8; a_off += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8usize { sums[l] += d * aux32[l] as f32; }
    }

    for l in 0..8usize { sumf += sums[l]; }
    sumf
}

pub fn ggml_vec_dot_iq2_xxs_q8_k_generic(n: usize, x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q2 = &x[i].qs;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for _ib32 in 0..QK_K / 32 {
            // aux32[0] = q2[q2_off] | (q2[q2_off+1] << 16), aux32[1] = q2[q2_off+2] | (q2[q2_off+3] << 16)
            let aux32_0 = (q2[q2_off + 0] as u32) | ((q2[q2_off + 1] as u32) << 16);
            let aux32_1 = (q2[q2_off + 2] as u32) | ((q2[q2_off + 3] as u32) << 16);
            let aux8_bytes = [
                (aux32_0 & 0xff) as u8,
                ((aux32_0 >> 8) & 0xff) as u8,
                ((aux32_0 >> 16) & 0xff) as u8,
                ((aux32_0 >> 24) & 0xff) as u8,
            ];
            q2_off += 4;
            let ls = 2 * (aux32_1 >> 28) + 1;
            let mut sumi = 0i32;
            for l in 0..4usize {
                let grid = IQ2XXS_GRID[aux8_bytes[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                for j in 0..8usize {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1i32 };
                    sumi += grid[j] as i32 * q8[q8_off + j] as i32 * sign;
                }
                q8_off += 8;
            }
            bsum += sumi * ls as i32;
        }
        sumf += d * bsum as f32;
    }

    0.125f32 * sumf
}

pub fn ggml_vec_dot_iq2_xs_q8_k_generic(n: usize, x: &[BlockIQ2XS], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q2 = &x[i].qs;
        let sc = &x[i].scales;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for ib32 in 0..QK_K / 32 {
            let ls1 = (2 * (sc[ib32] & 0xf) + 1) as i32;
            let ls2 = (2 * (sc[ib32] >> 4) + 1) as i32;
            let mut sumi = 0i32;
            for l in 0..2usize {
                let grid = IQ2XS_GRID[(q2[q2_off + l] & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(q2[q2_off + l] >> 9) as usize];
                for j in 0..8usize {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1i32 };
                    sumi += grid[j] as i32 * q8[q8_off + j] as i32 * sign;
                }
                q8_off += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for l in 2..4usize {
                let grid = IQ2XS_GRID[(q2[q2_off + l] & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(q2[q2_off + l] >> 9) as usize];
                for j in 0..8usize {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1i32 };
                    sumi += grid[j] as i32 * q8[q8_off + j] as i32 * sign;
                }
                q8_off += 8;
            }
            bsum += sumi * ls2;
            q2_off += 4;
        }
        sumf += d * bsum as f32;
    }

    0.125f32 * sumf
}

pub fn ggml_vec_dot_iq2_s_q8_k_generic(n: usize, x: &[BlockIQ2S], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q8 = &y[i].qs;
        let qs = &x[i].qs; // [u8; 64]
        let qh = &x[i].qh; // [u8; 8]
        let signs_base = 32usize; // qs + QK_K/8 = qs + 32

        let mut bsum = 0i32;
        let mut qs_off = 0usize;
        let mut signs_off = signs_base;
        let mut q8_off = 0usize;
        for ib32 in 0..QK_K / 32 {
            let ls1 = (1 + 2 * (x[i].scales[ib32] & 0xf)) as i32;
            let ls2 = (1 + 2 * (x[i].scales[ib32] >> 4)) as i32;
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;

            for l in 0..2usize {
                let grid_idx = (qs[qs_off + l] as usize)
                             | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = IQ2S_GRID[grid_idx].to_le_bytes();
                for j in 0..8usize {
                    let sign = if qs[signs_off + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1i32 };
                    sumi1 += q8[q8_off + j] as i32 * grid[j] as i32 * sign;
                }
                q8_off += 8;
            }
            for l in 2..4usize {
                let grid_idx = (qs[qs_off + l] as usize)
                             | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = IQ2S_GRID[grid_idx].to_le_bytes();
                for j in 0..8usize {
                    let sign = if qs[signs_off + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1i32 };
                    sumi2 += q8[q8_off + j] as i32 * grid[j] as i32 * sign;
                }
                q8_off += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs_off += 4;
            signs_off += 4;
        }

        sumf += d * bsum as f32;
    }

    0.125f32 * sumf
}

pub fn ggml_vec_dot_iq3_xxs_q8_k_generic(n: usize, x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        // qs: [u8; 96], gas starts at qs + QK_K/4 = qs + 64
        let q3 = &x[i].qs[..64];
        let gas_base = 64usize;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q3_off = 0usize;
        let mut gas_off = gas_base;
        let mut q8_off = 0usize;
        for _ib32 in 0..QK_K / 32 {
            // memcpy(&aux32, gas, sizeof(uint32_t))
            let aux32 = u32::from_le_bytes([
                x[i].qs[gas_off],
                x[i].qs[gas_off + 1],
                x[i].qs[gas_off + 2],
                x[i].qs[gas_off + 3],
            ]);
            gas_off += 4;
            let ls = 2 * (aux32 >> 28) + 1;
            let mut sumi = 0i32;
            for l in 0..4usize {
                let grid1 = IQ3XXS_GRID[q3[q3_off + 2 * l + 0] as usize].to_le_bytes();
                let grid2 = IQ3XXS_GRID[q3[q3_off + 2 * l + 1] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                for j in 0..4usize {
                    let s0 = if signs & KMASK_IQ2XS[j + 0] != 0 { -1i32 } else { 1i32 };
                    let s4 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1i32 };
                    sumi += grid1[j] as i32 * q8[q8_off + j + 0] as i32 * s0;
                    sumi += grid2[j] as i32 * q8[q8_off + j + 4] as i32 * s4;
                }
                q8_off += 8;
            }
            q3_off += 8;
            bsum += sumi * ls as i32;
        }
        sumf += d * bsum as f32;
    }

    0.25f32 * sumf
}

pub fn ggml_vec_dot_iq3_s_q8_k_generic(n: usize, x: &[BlockIQ3S], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let qs = &x[i].qs;     // [u8; 64]
        let qh = &x[i].qh;     // [u8; 8]
        let signs = &x[i].signs; // [u8; 32]
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut q8_off = 0usize;
        let mut ib32 = 0usize;
        while ib32 < QK_K / 32 {
            let ls1 = (2 * (x[i].scales[ib32 / 2] & 0xf) + 1) as i32;
            let ls2 = (2 * (x[i].scales[ib32 / 2] >> 4) + 1) as i32;

            let mut sumi = 0i32;
            for l in 0..4usize {
                let g1_idx = (qs[qs_off + 2 * l + 0] as usize)
                           | (((qh[ib32 + 0] as usize) << (8 - 2 * l)) & 256);
                let g2_idx = (qs[qs_off + 2 * l + 1] as usize)
                           | (((qh[ib32 + 0] as usize) << (7 - 2 * l)) & 256);
                let grid1 = IQ3S_GRID[g1_idx].to_le_bytes();
                let grid2 = IQ3S_GRID[g2_idx].to_le_bytes();
                for j in 0..4usize {
                    let s0 = if signs[signs_off + l] & KMASK_IQ2XS[j + 0] != 0 { -1i32 } else { 1i32 };
                    let s4 = if signs[signs_off + l] & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1i32 };
                    sumi += grid1[j] as i32 * q8[q8_off + j + 0] as i32 * s0;
                    sumi += grid2[j] as i32 * q8[q8_off + j + 4] as i32 * s4;
                }
                q8_off += 8;
            }
            qs_off += 8;
            signs_off += 4;
            bsum += sumi * ls1;

            sumi = 0;
            for l in 0..4usize {
                let g1_idx = (qs[qs_off + 2 * l + 0] as usize)
                           | (((qh[ib32 + 1] as usize) << (8 - 2 * l)) & 256);
                let g2_idx = (qs[qs_off + 2 * l + 1] as usize)
                           | (((qh[ib32 + 1] as usize) << (7 - 2 * l)) & 256);
                let grid1 = IQ3S_GRID[g1_idx].to_le_bytes();
                let grid2 = IQ3S_GRID[g2_idx].to_le_bytes();
                for j in 0..4usize {
                    let s0 = if signs[signs_off + l] & KMASK_IQ2XS[j + 0] != 0 { -1i32 } else { 1i32 };
                    let s4 = if signs[signs_off + l] & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1i32 };
                    sumi += grid1[j] as i32 * q8[q8_off + j + 0] as i32 * s0;
                    sumi += grid2[j] as i32 * q8[q8_off + j + 4] as i32 * s4;
                }
                q8_off += 8;
            }
            qs_off += 8;
            signs_off += 4;
            bsum += sumi * ls2;

            ib32 += 2;
        }
        sumf += d * bsum as f32;
    }

    sumf
}

pub fn ggml_vec_dot_iq1_s_q8_k_generic(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q8 = &y[i].qs;
        let qs = &x[i].qs; // [u8; 32]
        let qh = &x[i].qh; // [u16; 8]

        let mut sumi = 0i32;
        let mut sumi1 = 0i32;
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        for ib in 0..QK_K / 32 {
            let ls = (2 * ((qh[ib] >> 12) & 7) + 1) as i32;
            let delta: i32 = if qh[ib] & 0x8000 != 0 { -1 } else { 1 };
            let mut lsum = 0i32;
            for l in 0..4usize {
                let grid_idx = (qs[qs_off + l] as usize)
                             | (((qh[ib] as usize >> (3 * l)) & 7) << 8);
                let grid = IQ1S_GRID[grid_idx].to_le_bytes();
                for j in 0..8usize {
                    lsum += q8[q8_off + j] as i32 * grid[j] as i8 as i32;
                }
                q8_off += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2 * ib + 0] as i32 + y[i].bsums[2 * ib + 1] as i32);
            qs_off += 4;
        }

        sumf += f16_to_f32(x[i].d) * y[i].d * (sumi as f32 + IQ1S_DELTA * sumi1 as f32);
    }

    sumf
}

pub fn ggml_vec_dot_iq1_m_q8_k_generic(n: usize, x: &[BlockIQ1M], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q8 = &y[i].qs;
        let qs = &x[i].qs; // [u8; 32]
        let qh = &x[i].qh; // [u8; 16]

        // const uint16_t * sc = (const uint16_t *)x[i].scales;  ([u8; 8] -> [u16; 4])
        let sc_bytes = &x[i].scales; // [u8; 8]
        let sc = [
            u16::from_le_bytes([sc_bytes[0], sc_bytes[1]]),
            u16::from_le_bytes([sc_bytes[2], sc_bytes[3]]),
            u16::from_le_bytes([sc_bytes[4], sc_bytes[5]]),
            u16::from_le_bytes([sc_bytes[6], sc_bytes[7]]),
        ];

        // iq1m_scale_t union: scale.u16 = ...; then interpret as f16
        let scale_u16 = (sc[0] >> 12)
                      | ((sc[1] >> 8) & 0x00f0)
                      | ((sc[2] >> 4) & 0x0f00)
                      | (sc[3] & 0xf000);
        let scale_f = f16_to_f32(scale_u16);

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        let mut qh_off = 0usize;
        for ib in 0..QK_K / 32 {
            let delta = [
                if qh[qh_off + 0] & 0x08 != 0 { -1i32 } else { 1 },
                if qh[qh_off + 0] & 0x80 != 0 { -1i32 } else { 1 },
                if qh[qh_off + 1] & 0x08 != 0 { -1i32 } else { 1 },
                if qh[qh_off + 1] & 0x80 != 0 { -1i32 } else { 1 },
            ];
            let mut sum1 = [0i32; 2];
            let mut sum2 = [0i32; 2];

            for l in 0..4usize {
                let qh_byte = qh[qh_off + l / 2];
                let grid_idx = (qs[qs_off + l] as usize)
                             | (((qh_byte as usize) << (8 - 4 * (l % 2))) & 0x700);
                let grid = IQ1S_GRID[grid_idx].to_le_bytes();
                let mut lsum1 = 0i32;
                let mut lsum2 = 0i32;
                for j in 0..8usize {
                    lsum1 += q8[q8_off + j] as i32 * grid[j] as i8 as i32;
                    lsum2 += q8[q8_off + j] as i32;
                }
                q8_off += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }

            let ls1 = (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1) as i32;
            let ls2 = (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as i32;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs_off += 4;
            qh_off += 2;
        }

        sumf += scale_f * y[i].d * (sumi1 as f32 + IQ1M_DELTA * sumi2 as f32);
    }

    sumf
}

pub fn ggml_vec_dot_iq4_nl_q8_0_generic(n: usize, x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(n % QK4_NL, 0);
    let nb = n / QK4_NL;

    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let d = f16_to_f32(y[ib].d) * f16_to_f32(x[ib].d);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..QK4_NL / 2 {
            sumi1 += y[ib].qs[j] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] & 0xf) as usize] as i32;
            sumi2 += y[ib].qs[j + QK4_NL / 2] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }

    sumf
}

pub fn ggml_vec_dot_iq4_xs_q8_k_generic(n: usize, x: &[BlockIQ4XS], y: &[BlockQ8K]) -> f32 {
    assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;

    let mut sumf = 0.0f32;
    for ibl in 0..nb {
        let d4d8 = f16_to_f32(x[ibl].d) * y[ibl].d;
        let mut h = x[ibl].scales_h;
        let qs = &x[ibl].qs;
        let q8 = &y[ibl].qs;
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        let mut ib = 0usize;
        while ib < QK_K / 32 {
            let ls1 = ((x[ibl].scales_l[ib / 2] & 0xf) | (((h << 4) & 0x30) as u8)) as i32;
            let ls2 = ((x[ibl].scales_l[ib / 2] >> 4) | (((h << 2) & 0x30) as u8)) as i32;
            h >>= 4;
            let d1 = d4d8 * (ls1 - 32) as f32;
            let d2 = d4d8 * (ls2 - 32) as f32;

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..16usize {
                sumi1 += q8[q8_off + j] as i32 * KVALUES_IQ4NL[(qs[qs_off + j] & 0xf) as usize] as i32;
                sumi2 += q8[q8_off + 16 + j] as i32 * KVALUES_IQ4NL[(qs[qs_off + j] >> 4) as usize] as i32;
            }
            sumf += d1 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;

            sumi1 = 0; sumi2 = 0;
            for j in 0..16usize {
                sumi1 += q8[q8_off + j] as i32 * KVALUES_IQ4NL[(qs[qs_off + j] & 0xf) as usize] as i32;
                sumi2 += q8[q8_off + 16 + j] as i32 * KVALUES_IQ4NL[(qs[qs_off + j] >> 4) as usize] as i32;
            }
            sumf += d2 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;

            ib += 2;
        }
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::helpers::{f32_to_f16, nearest_int};
    use crate::ggml::quantize::{quantize_row_q8_0, quantize_row_q8_k};

    fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    // Build a Q4K block from raw nibbles and scale parameters.
    // Uses a uniform scale so we can verify the result analytically.
    fn synthetic_q4k_q8k(n: usize) -> (Vec<BlockQ4K>, Vec<BlockQ8K>) {
        assert_eq!(n, QK_K);
        // Build a Q4K block: uniform scale d, zero min, all nibbles = 8 (midpoint)
        // This gives zero dot product trivially; instead use a ramp.
        let d_f = 0.01f32;
        let dmin_f = 0.0f32;

        // Pack nibbles: low = i % 16, high = (i/16) % 16 -- for i in 0..128 bytes
        let mut qs = [0u8; 128];
        for j in 0..128usize {
            let lo = (j & 0xf) as u8;
            let hi = ((j / 2) & 0xf) as u8;
            qs[j] = lo | (hi << 4);
        }

        // Scales: 8 scale groups and 8 min groups packed into 12 bytes
        // Use scales = [4,4,4,4, 4,4,4,4] and mins = [0; 8] encoded via get_scale_min_k4 inverse
        // Simplest: pack scales[0..8] = 4 and mins[0..8] = 0 using the known encoding
        // From ggml-common.h: for j < 4: scales[j] = scale | (min << 6)... complex.
        // Use known-good 12-byte packing for all-4 scales, all-0 mins:
        // scale=4 fits in 6 bits, min=0, so: for j<4: sc[j]=4, sc[j+4]=0|(0<<4)=0
        // The 12-byte encoding: sc[0..4]=4, sc[4..8]=0, sc[8..12]=0 won't be right without
        // the actual encoder. Instead, test via the existing vec_dot function.
        let scales = {
            // Encode: scales[0..8] = 4, mins[0..8] = 0
            // From encode_scale_min_k4: if j<4: sc[j] = (sc & 0x3f) | ((mn & 3) << 6)
            //                                  sc[j+4] = (sc >> 2) | ((mn & 0x3c) << 2)? No.
            // Simplest: use all zeros for scales so the dot product is 0.
            [0u8; 12]
        };

        let x = vec![BlockQ4K {
            d: f32_to_f16(d_f),
            dmin: f32_to_f16(dmin_f),
            scales,
            qs,
        }];

        // Q8K: uniform activation of 1 per element
        let acts = vec![1.0f32; QK_K];
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&acts, &mut y);
        (x, y)
    }

    #[test]
    fn q4k_generic_matches_existing() {
        // Construct a non-trivial Q4K block via known scale patterns and verify
        // generic output matches the existing vec_dot_q4_k_q8_k implementation.
        let n = QK_K;

        // Build a block with deterministic nibbles and non-trivial scales.
        // We'll use the scale packing from helpers::get_scale_min_k4 inverse:
        // For j < 4: packed[j] = sc & 0x3F, packed[j+4] bits 3:0 = mn & 0xF
        // and the upper bits go in the last 4 bytes. Use simple values:
        // scales = [10,10,10,10, 10,10,10,10], mins = [1,1,1,1, 1,1,1,1]
        // Encoded: for j<4: packed[j] = 10 | ((1&3) << 6) = 10 | 64 = 74
        //          packed[j+4] = 10>>2 | ((1>>2) << 4) = 2 | 0 = 2  ... this needs the
        // actual encoder. Since we don't have quantize_row_q4_k, use a cross-check with
        // the existing function instead: build any block, call both functions, compare.
        let mut x = [BlockQ4K {
            d: f32_to_f16(0.25),
            dmin: f32_to_f16(0.01),
            scales: [
                // Manually encode: scales[0..8]=[16;8], mins[0..8]=[8;8]
                // Using the 6-bit encoding format from ggml-common.h:
                // j<4: byte = (sc & 0xf) | ((mn & 0xf) << 4)  -- this is a simplification
                // We'll just use bytes that produce stable results in both paths.
                0x11, 0x22, 0x33, 0x44,  // lower nibbles are scales[0..4]
                0x55, 0x66, 0x77, 0x88,  // upper nibbles encode mins
                0x00, 0x00, 0x00, 0x00,  // extra scale bits
            ],
            qs: std::array::from_fn(|i| {
                let lo = (i & 0xf) as u8;
                let hi = ((i + 3) & 0xf) as u8;
                lo | (hi << 4)
            }),
        }];

        let acts: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&acts, &mut y);

        let existing = crate::ggml::vec_dot::vec_dot_q4_k_q8_k(n, &x, &y);
        let generic = ggml_vec_dot_q4_k_q8_k_generic(n, &x, &y);

        assert_eq!(
            existing, generic,
            "Q4K generic vs existing: existing={existing}, generic={generic}"
        );
    }

    #[test]
    fn q6k_generic_matches_existing() {
        let n = QK_K;

        let x = [BlockQ6K {
            ql: std::array::from_fn(|i| (i & 0x3f) as u8),
            qh: std::array::from_fn(|i| (i & 0x3) as u8),
            scales: std::array::from_fn(|i| ((i as i8) - 8)),
            d: f32_to_f16(0.1),
        }];

        let acts: Vec<f32> = (0..QK_K).map(|i| (i as f32) * 0.005 - 0.64).collect();
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&acts, &mut y);

        let existing = crate::ggml::vec_dot::vec_dot_q6_k_q8_k(n, &x, &y);
        let generic = ggml_vec_dot_q6_k_q8_k_generic(n, &x, &y);

        assert_eq!(
            existing, generic,
            "Q6K generic vs existing: existing={existing}, generic={generic}"
        );
    }

    #[test]
    fn q4_0_generic_correct() {
        let n = QK8_0;
        let floats: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let scale_f = floats.iter().cloned().fold(0.0f32, f32::max).abs()
                    .max(floats.iter().cloned().fold(0.0f32, f32::min).abs());
        let d = scale_f / 7.0;
        let d_f16 = f32_to_f16(d);
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let lo = nearest_int(floats[j] / d + 8.0).clamp(0, 15) as u8;
            let hi = nearest_int(floats[j + 16] / d + 8.0).clamp(0, 15) as u8;
            qs[j] = lo | (hi << 4);
        }
        let xb = [BlockQ4_0 { d: d_f16, qs }];

        let mut yb = [BlockQ8_0 { d: 0, qs: [0i8; 32] }];
        quantize_row_q8_0(&floats, &mut yb);

        let result = ggml_vec_dot_q4_0_q8_0_generic(n, &xb, &yb);

        // Reference: dequant x, dequant y, dot product
        let mut x_f32 = [0.0f32; 32];
        let d_f = f16_to_f32(d_f16);
        for j in 0..16 {
            x_f32[j] = ((xb[0].qs[j] & 0xF) as f32 - 8.0) * d_f;
            x_f32[j + 16] = ((xb[0].qs[j] >> 4) as f32 - 8.0) * d_f;
        }
        let mut y_f32 = [0.0f32; 32];
        let dy = f16_to_f32(yb[0].d);
        for j in 0..32 {
            y_f32[j] = yb[0].qs[j] as f32 * dy;
        }
        let expected = dot_f32(&x_f32, &y_f32);

        let rel_err = ((result - expected) / expected.abs().max(1e-6)).abs();
        assert!(
            rel_err < 1e-3,
            "Q4_0 generic: result={result}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn q8_0_generic_self_dot() {
        let n = QK8_0;
        let floats: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

        let mut xb = [BlockQ8_0 { d: 0, qs: [0i8; 32] }];
        quantize_row_q8_0(&floats, &mut xb);

        let result = ggml_vec_dot_q8_0_q8_0_generic(n, &xb, &xb);

        let d = f16_to_f32(xb[0].d);
        let mut x_f = [0.0f32; 32];
        for j in 0..32 { x_f[j] = xb[0].qs[j] as f32 * d; }
        let expected = dot_f32(&x_f, &x_f);

        let rel_err = ((result - expected) / expected.abs().max(1e-6)).abs();
        assert!(
            rel_err < 1e-4,
            "Q8_0 self-dot: result={result}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn tq2_0_generic_zero_weights_zero_dot() {
        // All TQ2_0 weights = 1 (ternary midpoint: bit-pair 01 -> (1-1)=0).
        // Encoding: all bytes = 0b01010101 = 0x55.
        let n = QK_K;
        let x = [BlockTQ2_0 {
            qs: [0x55u8; 64], // each byte: four 2-bit values, all = 1 -> trit = 0
            d: f32_to_f16(1.0),
        }];
        let acts: Vec<f32> = (0..QK_K).map(|i| i as f32).collect();
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&acts, &mut y);

        let result = ggml_vec_dot_tq2_0_q8_k_generic(n, &x, &y);
        // All trits = 0, so sum = 0
        assert_eq!(result, 0.0, "TQ2_0 midpoint weights should give zero dot product");
    }

    #[test]
    fn tq2_0_generic_all_positive() {
        // All TQ2_0 weights = 2 (bit-pair 10 -> (2-1)=+1), all Q8K qs = -127.
        // quantize_row_q8_k([1.0; 256]) gives qs[j]=-127, d_y=-1/127.
        // isum = 256 * (1 * -127) = -32512
        // result = d_x * d_y * isum = 0.5 * (-1/127) * (-32512) = 128.0
        let n = QK_K;
        let d_x = 0.5f32;
        let x = [BlockTQ2_0 {
            qs: [0b10101010u8; 64], // each bit-pair = 2 -> trit = +1
            d: f32_to_f16(d_x),
        }];
        let acts = vec![1.0f32; QK_K];
        let mut y = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        quantize_row_q8_k(&acts, &mut y);

        let result = ggml_vec_dot_tq2_0_q8_k_generic(n, &x, &y);

        // result = d_x * d_y * 256 * (1 * -127) = 0.5 * (-1/127) * (-32512) = 128
        let expected = 128.0f32;
        let abs_err = (result - expected).abs();
        assert!(
            abs_err < 0.01,
            "TQ2_0 all-positive: result={result}, expected={expected}"
        );
    }
}

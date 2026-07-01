use super::blocks::*;
use super::helpers::f16_to_f32;

// ── Q8_0 family (basic quants, QK = 32) ────────────────────────────────────

pub fn vec_dot_q4_0_q8_0(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..QK / 2 {
            let v0 = (x[ib].qs[j] & 0x0F) as i32 - 8;
            let v1 = (x[ib].qs[j] >> 4) as i32 - 8;
            sumi0 += v0 * y[ib].qs[j] as i32;
            sumi1 += v1 * y[ib].qs[j + QK / 2] as i32;
        }
        sumf += (sumi0 + sumi1) as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
    }
    sumf
}

pub fn vec_dot_q4_1_q8_1(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi = 0i32;
        for j in 0..QK / 2 {
            let v0 = (x[ib].qs[j] & 0x0F) as i32;
            let v1 = (x[ib].qs[j] >> 4) as i32;
            sumi += v0 * y[ib].qs[j] as i32;
            sumi += v1 * y[ib].qs[j + QK / 2] as i32;
        }
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d)
            + f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);
    }
    sumf
}

pub fn vec_dot_q5_0_q8_0(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let qh = u32::from_le_bytes(x[ib].qh);
        let mut sumi = 0i32;
        for j in 0..QK / 2 {
            let xh_0 = ((qh >> j) & 1) as i32;
            let xh_1 = ((qh >> (j + 16)) & 1) as i32;
            let v0 = ((x[ib].qs[j] & 0x0F) as i32 | (xh_0 << 4)) - 16;
            let v1 = ((x[ib].qs[j] >> 4) as i32 | (xh_1 << 4)) - 16;
            sumi += v0 * y[ib].qs[j] as i32;
            sumi += v1 * y[ib].qs[j + QK / 2] as i32;
        }
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
    }
    sumf
}

pub fn vec_dot_q5_1_q8_1(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let qh = u32::from_le_bytes(x[ib].qh);
        let mut sumi = 0i32;
        for j in 0..QK / 2 {
            let xh_0 = ((qh >> j) & 1) as i32;
            let xh_1 = ((qh >> (j + 16)) & 1) as i32;
            let v0 = (x[ib].qs[j] & 0x0F) as i32 | (xh_0 << 4);
            let v1 = (x[ib].qs[j] >> 4) as i32 | (xh_1 << 4);
            sumi += v0 * y[ib].qs[j] as i32;
            sumi += v1 * y[ib].qs[j + QK / 2] as i32;
        }
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d)
            + f16_to_f32(x[ib].m) * f16_to_f32(y[ib].s);
    }
    sumf
}

pub fn vec_dot_q8_0_q8_0(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi = 0i32;
        for j in 0..QK {
            sumi += x[ib].qs[j] as i32 * y[ib].qs[j] as i32;
        }
        sumf += sumi as f32 * f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
    }
    sumf
}

// ── Q8_K family (K-quants, QK_K = 256) ─────────────────────────────────────

pub fn vec_dot_q4_k_q8_k(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let q4 = &x[i].qs;
        let q8 = &y[i].qs;

        // unpack scales/mins from 12 bytes using ggml's bitmask approach
        let mut utmp = [0u32; 4];
        let scale_bytes: &[u8; 12] = &x[i].scales;
        utmp[0] = u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
        utmp[1] = u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]);
        utmp[2] = u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales = utmp[0].to_le_bytes()
            .iter().chain(utmp[1].to_le_bytes().iter())
            .copied().collect::<Vec<u8>>();
        let mins = utmp[2].to_le_bytes()
            .iter().chain(utmp[3].to_le_bytes().iter())
            .copied().collect::<Vec<u8>>();

        // unpack Q4 nibbles to aux8
        let mut aux8 = [0i8; QK];
        {
            let mut a_idx = 0usize;
            let mut q4_off = 0usize;
            for _j in 0..QK / 64 {
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
        }

        // compute sum of mins * bsums
        let mut sumi = 0i32;
        for j in 0..QK / 16 {
            sumi += y[i].bsums[j] as i32 * mins[j / 2] as i32;
        }

        // main dot product
        let mut aux32 = [0i32; 8];
        let mut a_idx = 0usize;
        let mut q8_idx = 0usize;
        let mut is = 0usize;
        for _j in 0..QK / 32 {
            let scale = scales[is] as i32;
            is += 1;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = f16_to_f32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi as f32;
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

pub fn vec_dot_q6_k_q8_k(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let ql = &x[i].ql;
        let qh = &x[i].qh;
        let q8 = &y[i].qs;
        let sc = &x[i].scales;

        let mut aux8 = [0i8; QK];
        {
            let mut a_idx = 0usize;
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            for _j in 0..QK / 128 {
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
        }

        let mut aux32 = [0i32; 8];
        let mut a_idx = 0usize;
        let mut q8_idx = 0usize;
        let mut is = 0usize;
        for _j in 0..QK / 16 {
            let scale = sc[is] as i32;
            is += 1;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
            for l in 0..8 {
                aux32[l] += scale * (q8[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
        }

        let d = f16_to_f32(x[i].d) * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

pub fn vec_dot_q2_k_q8_k(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let dmin = f16_to_f32(x[i].dmin) * y[i].d;

        let mut summs = 0i32;
        for j in 0..QK / 16 {
            summs += y[i].bsums[j] as i32 * (x[i].scales[j] >> 4) as i32;
        }
        sumf -= dmin * summs as f32;

        let mut isum = 0i32;
        let mut q_idx = 0usize;
        for j in 0..QK / 16 {
            let sc = (x[i].scales[j] & 0xF) as i32;
            for l in 0..16 {
                let byte_idx = q_idx / 4;
                let shift = (q_idx % 4) * 2;
                let q = ((x[i].qs[byte_idx] >> shift) & 3) as i32;
                isum += sc * q * y[i].qs[q_idx] as i32;
                q_idx += 1;
            }
        }
        sumf += d * isum as f32;
    }
    sumf
}

pub fn vec_dot_q3_k_q8_k(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;

        // unpack scales (same logic as dequantize)
        let mut scales = [0i32; 16];
        let aux = &x[i].scales;
        for j in 0..4 {
            scales[j] = (aux[j] & 0xF) as i32 - 8;
            scales[j + 4] = (aux[j] >> 4) as i32 - 8;
        }
        for j in 0..4 {
            scales[j + 8] = (aux[j + 4] & 0xF) as i32 - 8;
            scales[j + 12] = (aux[j + 4] >> 4) as i32 - 8;
        }
        for j in 0..4 {
            let a = aux[j + 8];
            let c0 = (a & 3) as i32;
            let c1 = ((a >> 2) & 3) as i32;
            let c2 = ((a >> 4) & 3) as i32;
            let c3 = ((a >> 6) & 3) as i32;
            scales[j] += c0 * 16 - if c0 >= 2 { 32 } else { 0 };
            scales[j + 4] += c1 * 16 - if c1 >= 2 { 32 } else { 0 };
            scales[j + 8] += c2 * 16 - if c2 >= 2 { 32 } else { 0 };
            scales[j + 12] += c3 * 16 - if c3 >= 2 { 32 } else { 0 };
        }

        let mut isum = 0i32;
        for j in 0..QK {
            let byte_idx = j / 4;
            let shift = (j % 4) * 2;
            let low2 = ((x[i].qs[byte_idx] >> shift) & 3) as i32;
            let hm_byte = j / 8;
            let hm_bit = j % 8;
            let high = ((x[i].hmask[hm_byte] >> hm_bit) & 1) as i32;
            let q = (low2 | (high << 2)) - 4;
            isum += scales[j / 16] * q * y[i].qs[j] as i32;
        }
        sumf += d * isum as f32;
    }
    sumf
}

pub fn vec_dot_q5_k_q8_k(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let dmin = f16_to_f32(x[i].dmin) * y[i].d;

        // unpack scales/mins
        let mut utmp = [0u32; 4];
        let sb: &[u8; 12] = &x[i].scales;
        utmp[0] = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
        utmp[1] = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
        utmp[2] = u32::from_le_bytes([sb[8], sb[9], sb[10], sb[11]]);

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales_bytes: Vec<u8> = utmp[0].to_le_bytes().iter().chain(utmp[1].to_le_bytes().iter()).copied().collect();
        let mins_bytes: Vec<u8> = utmp[2].to_le_bytes().iter().chain(utmp[3].to_le_bytes().iter()).copied().collect();

        let mut summs = 0i32;
        for j in 0..QK / 16 {
            summs += y[i].bsums[j] as i32 * mins_bytes[j / 2] as i32;
        }
        sumf -= dmin * summs as f32;

        let ql = &x[i].qs;
        let qh = &x[i].qh;

        let mut isum = 0i32;
        let mut is = 0usize;
        let mut ql_off = 0usize;
        let mut q8_off = 0usize;

        for _chunk in 0..QK / 64 {
            let sc0 = scales_bytes[is] as i32;
            let sc1 = scales_bytes[is + 1] as i32;
            is += 2;
            // iterate element by element with direct high-bit lookup
            for l in 0..32 {
                let byte_idx = ql_off + l;
                let q_lo = (ql[byte_idx] & 0xF) as i32;
                let hbit_idx = byte_idx;
                let hbit_byte = hbit_idx / 8;
                let hbit_shift = hbit_idx % 8;
                let hbit = if hbit_byte < qh.len() { ((qh[hbit_byte] >> hbit_shift) & 1) as i32 } else { 0 };
                let q = q_lo | (hbit << 4);
                isum += sc0 * q * y[i].qs[q8_off + l] as i32;
            }
            for l in 0..32 {
                let byte_idx = ql_off + l;
                let q_hi = (ql[byte_idx] >> 4) as i32;
                let hbit_idx = byte_idx + 128; // high nibbles correspond to positions 128..255
                let hbit_byte = hbit_idx / 8;
                let hbit_shift = hbit_idx % 8;
                let hbit = if hbit_byte < qh.len() { ((qh[hbit_byte] >> hbit_shift) & 1) as i32 } else { 0 };
                let q = q_hi | (hbit << 4);
                isum += sc1 * q * y[i].qs[q8_off + 32 + l] as i32;
            }
            ql_off += 32;
            q8_off += 64;
        }
        sumf += d * isum as f32;
    }
    sumf
}

// ── Ternary quants ─────────────────────────────────────────────────────────

pub fn vec_dot_tq2_0_q8_k(n: usize, x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut isum = 0i32;
        for j in 0..64 {
            let byte = x[i].qs[j];
            for b in 0..4 {
                let trit = ((byte >> (b * 2)) & 3) as i32 - 1;
                isum += trit * y[i].qs[j * 4 + b] as i32;
            }
        }
        sumf += d * isum as f32;
    }
    sumf
}

pub fn vec_dot_tq1_0_q8_k(n: usize, x: &[BlockTQ1_0], y: &[BlockQ8K]) -> f32 {
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let mut isum = 0i32;
        let mut idx = 0usize;
        for j in 0..48 {
            let mut byte = x[i].qs[j] as u32;
            for _b in 0..5 {
                if idx >= QK { break; }
                let trit = (byte % 3) as i32 - 1;
                byte /= 3;
                isum += trit * y[i].qs[idx] as i32;
                idx += 1;
            }
        }
        for j in 0..4 {
            let mut byte = x[i].qh[j] as u32;
            for _b in 0..4 {
                if idx >= QK { break; }
                let trit = (byte % 3) as i32 - 1;
                byte /= 3;
                isum += trit * y[i].qs[idx] as i32;
                idx += 1;
            }
        }
        sumf += d * isum as f32;
    }
    sumf
}

// ── IQ quants ──────────────────────────────────────────────────────────────

pub fn vec_dot_iq4_nl_q8_0(n: usize, x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    use super::iq_tables::KVALUES_IQ4NL;
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let d = f16_to_f32(x[ib].d) * f16_to_f32(y[ib].d);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..QK / 2 {
            sumi1 += y[ib].qs[j] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] & 0xF) as usize] as i32;
            sumi2 += y[ib].qs[j + QK / 2] as i32 * KVALUES_IQ4NL[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

pub fn vec_dot_mxfp4_q8_0(n: usize, x: &[BlockMxfp4], y: &[BlockQ8_0]) -> f32 {
    use super::iq_tables::KVALUES_MXFP4;
    const QK: usize = 32;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        // E8M0 shared exponent → scale
        let e8m0 = x[ib].e;
        let scale = if e8m0 == 0 { 0.0f32 } else { f32::from_bits((e8m0 as u32) << 23) };
        let d = f16_to_f32(y[ib].d) * scale * 0.5; // kvalues are doubled (int representation)
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..QK / 2 {
            sumi1 += y[ib].qs[j] as i32 * KVALUES_MXFP4[(x[ib].qs[j] & 0xF) as usize] as i32;
            sumi2 += y[ib].qs[j + QK / 2] as i32 * KVALUES_MXFP4[(x[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

pub fn vec_dot_nvfp4_q8_0(n: usize, x: &[BlockNvfp4], y: &[BlockQ8_0]) -> f32 {
    use super::iq_tables::KVALUES_MXFP4;
    const QK: usize = 64;
    const SUB: usize = 16;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        for s_idx in 0..4usize {
            let ue4m3_byte = x[ib].d[s_idx];
            let d_sub = super::dequantize::ue4m3_to_f32_pub(ue4m3_byte);
            let q8_block = s_idx / 2;
            let q8_off = (s_idx % 2) * SUB;
            let dy = f16_to_f32(y[2 * ib + q8_block].d);
            let mut sumi_lo = 0i32;
            let mut sumi_hi = 0i32;
            for j in 0..SUB / 2 {
                let qv = x[ib].qs[s_idx * (SUB / 2) + j];
                sumi_lo += y[2 * ib + q8_block].qs[q8_off + j] as i32 * KVALUES_MXFP4[(qv & 0xF) as usize] as i32;
                sumi_hi += y[2 * ib + q8_block].qs[q8_off + j + SUB / 2] as i32 * KVALUES_MXFP4[(qv >> 4) as usize] as i32;
            }
            sumf += dy * d_sub * (sumi_lo + sumi_hi) as f32;
        }
    }
    sumf
}

pub fn vec_dot_q1_0_q8_0(n: usize, x: &[BlockQ1_0], y: &[BlockQ8_0]) -> f32 {
    const QK: usize = 128; // Q1_0 block = 128 elements, spans 4 Q8_0 blocks
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d0 = f16_to_f32(x[i].d);
        let mut sumi = 0.0f32;
        for k in 0..4 {
            let yb = &y[i * 4 + k];
            let d1 = f16_to_f32(yb.d);
            let mut sumi_block = 0i32;
            let bits = &x[i].qs[k * 4..k * 4 + 4];
            let mut q8_idx = 0usize;
            for b in 0..4u8 {
                let mask = bits[b as usize];
                for bit in 0..8u8 {
                    let sign = if (mask >> bit) & 1 == 1 { 1i32 } else { -1 };
                    sumi_block += sign * yb.qs[q8_idx] as i32;
                    q8_idx += 1;
                }
            }
            sumi += d1 * sumi_block as f32;
        }
        sumf += d0 * sumi;
    }
    sumf
}

fn grid_u64_bytes(grid: &[u64], idx: usize) -> [u8; 8] {
    grid[idx].to_le_bytes()
}

fn grid_u32_bytes(grid: &[u32], idx: usize) -> [u8; 4] {
    grid[idx].to_le_bytes()
}

pub fn vec_dot_iq2_xxs_q8_k(n: usize, x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q2 = &x[i].qs;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q8_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let aux32_0 = q2[ib32 * 4] as u32 | ((q2[ib32 * 4 + 1] as u32) << 16);
            let aux32_1 = q2[ib32 * 4 + 2] as u32 | ((q2[ib32 * 4 + 3] as u32) << 16);
            let aux8 = [
                (aux32_0 & 0xFF) as u8,
                ((aux32_0 >> 8) & 0xFF) as u8,
                (aux32_1 & 0xFF) as u8,
                ((aux32_1 >> 8) & 0xFF) as u8,
            ];
            let ls = 2 * ((aux32_1 >> 28) as i32) + 1;
            let mut sumi = 0i32;
            for l in 0..4 {
                let grid = grid_u64_bytes(&IQ2XXS_GRID, aux8[l] as usize);
                let sign_bits = KSIGNS_IQ2XS[((aux32_1 >> (7 * l as u32)) & 127) as usize];
                for j in 0..8 {
                    let sign = if sign_bits & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    sumi += grid[j] as i32 * q8[q8_idx + j] as i32 * sign;
                }
                q8_idx += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

pub fn vec_dot_iq2_xs_q8_k(n: usize, x: &[BlockIQ2XS], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q2 = &x[i].qs;
        let sc = &x[i].scales;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q8_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let ls1 = (2 * (sc[ib32] & 0xF) + 1) as i32;
            let ls2 = (2 * (sc[ib32] >> 4) + 1) as i32;
            let mut sumi = 0i32;
            for l in 0..2 {
                let grid = grid_u64_bytes(&IQ2XS_GRID, (q2[ib32 * 4 + l] & 511) as usize);
                let sign_bits = KSIGNS_IQ2XS[(q2[ib32 * 4 + l] >> 9) as usize];
                for j in 0..8 {
                    let sign = if sign_bits & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    sumi += grid[j] as i32 * q8[q8_idx + j] as i32 * sign;
                }
                q8_idx += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for l in 2..4 {
                let grid = grid_u64_bytes(&IQ2XS_GRID, (q2[ib32 * 4 + l] & 511) as usize);
                let sign_bits = KSIGNS_IQ2XS[(q2[ib32 * 4 + l] >> 9) as usize];
                for j in 0..8 {
                    let sign = if sign_bits & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    sumi += grid[j] as i32 * q8[q8_idx + j] as i32 * sign;
                }
                q8_idx += 8;
            }
            bsum += sumi * ls2;
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

pub fn vec_dot_iq2_s_q8_k(n: usize, x: &[BlockIQ2S], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q8 = &y[i].qs;
        let qs = &x[i].qs;
        let qh = &x[i].qh;
        let signs = &qs[QK / 8..]; // signs start at qs + QK/8 = qs + 32

        let mut bsum = 0i32;
        let mut q8_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let ls1 = (1 + 2 * (x[i].scales[ib32] & 0xF)) as i32;
            let ls2 = (1 + 2 * (x[i].scales[ib32] >> 4)) as i32;
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for l in 0..2usize {
                let grid_idx = qs[qs_idx + l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = grid_u64_bytes(&IQ2S_GRID, grid_idx);
                for j in 0..8 {
                    let sign = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    sumi1 += q8[q8_idx + j] as i32 * grid[j] as i32 * sign;
                }
                q8_idx += 8;
            }
            for l in 2..4usize {
                let grid_idx = qs[qs_idx + l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = grid_u64_bytes(&IQ2S_GRID, grid_idx);
                for j in 0..8 {
                    let sign = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    sumi2 += q8[q8_idx + j] as i32 * grid[j] as i32 * sign;
                }
                q8_idx += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs_idx += 4;
            signs_idx += 4;
        }
        sumf += d * bsum as f32;
    }
    0.125 * sumf
}

pub fn vec_dot_iq3_xxs_q8_k(n: usize, x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let q3 = &x[i].qs[..QK / 4]; // first 64 bytes are grid indices
        let gas = &x[i].qs[QK / 4..]; // remaining bytes are scales+signs
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q8_idx = 0usize;
        let mut q3_idx = 0usize;
        let mut gas_idx = 0usize;
        for _ib32 in 0..QK / 32 {
            let aux32 = u32::from_le_bytes([gas[gas_idx], gas[gas_idx + 1], gas[gas_idx + 2], gas[gas_idx + 3]]);
            gas_idx += 4;
            let ls = (2 * (aux32 >> 28) + 1) as i32;
            let mut sumi = 0i32;
            for l in 0..4 {
                let grid1 = grid_u32_bytes(&IQ3XXS_GRID, q3[q3_idx + 2 * l] as usize);
                let grid2 = grid_u32_bytes(&IQ3XXS_GRID, q3[q3_idx + 2 * l + 1] as usize);
                let sign_bits = KSIGNS_IQ2XS[((aux32 >> (7 * l as u32)) & 127) as usize];
                for j in 0..4 {
                    let sign1 = if sign_bits & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    let sign2 = if sign_bits & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1 };
                    sumi += grid1[j] as i32 * q8[q8_idx + j] as i32 * sign1;
                    sumi += grid2[j] as i32 * q8[q8_idx + j + 4] as i32 * sign2;
                }
                q8_idx += 8;
            }
            q3_idx += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum as f32;
    }
    0.25 * sumf
}

pub fn vec_dot_iq3_s_q8_k(n: usize, x: &[BlockIQ3S], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let d = f16_to_f32(x[i].d) * y[i].d;
        let qs = &x[i].qs;
        let qh = &x[i].qh;
        let signs = &x[i].signs;
        let q8 = &y[i].qs;
        let mut bsum = 0i32;
        let mut q8_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = 0usize;
        for ib32 in (0..QK / 32).step_by(2) {
            let ls1 = (2 * (x[i].scales[ib32 / 2] & 0xF) + 1) as i32;
            let ls2 = (2 * (x[i].scales[ib32 / 2] >> 4) + 1) as i32;
            let mut sumi = 0i32;
            for l in 0..4 {
                let idx1 = qs[qs_idx + 2 * l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 256);
                let idx2 = qs[qs_idx + 2 * l + 1] as usize | ((qh[ib32] as usize) << (7 - 2 * l) & 256);
                let grid1 = grid_u32_bytes(&IQ3S_GRID, idx1);
                let grid2 = grid_u32_bytes(&IQ3S_GRID, idx2);
                for j in 0..4 {
                    let sign1 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    let sign2 = if signs[signs_idx + l] & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1 };
                    sumi += grid1[j] as i32 * q8[q8_idx + j] as i32 * sign1;
                    sumi += grid2[j] as i32 * q8[q8_idx + j + 4] as i32 * sign2;
                }
                q8_idx += 8;
            }
            qs_idx += 8;
            signs_idx += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for l in 0..4 {
                let idx1 = qs[qs_idx + 2 * l] as usize | ((qh[ib32 + 1] as usize) << (8 - 2 * l) & 256);
                let idx2 = qs[qs_idx + 2 * l + 1] as usize | ((qh[ib32 + 1] as usize) << (7 - 2 * l) & 256);
                let grid1 = grid_u32_bytes(&IQ3S_GRID, idx1);
                let grid2 = grid_u32_bytes(&IQ3S_GRID, idx2);
                for j in 0..4 {
                    let sign1 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1i32 } else { 1 };
                    let sign2 = if signs[signs_idx + l] & KMASK_IQ2XS[j + 4] != 0 { -1i32 } else { 1 };
                    sumi += grid1[j] as i32 * q8[q8_idx + j] as i32 * sign1;
                    sumi += grid2[j] as i32 * q8[q8_idx + j + 4] as i32 * sign2;
                }
                q8_idx += 8;
            }
            qs_idx += 8;
            signs_idx += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum as f32;
    }
    sumf
}

pub fn vec_dot_iq1_s_q8_k(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q8 = &y[i].qs;
        let qs = &x[i].qs;
        let qh = &x[i].qh;
        let mut sumi = 0i32;
        let mut sumi1 = 0i32;
        let mut q8_idx = 0usize;
        let mut qs_idx = 0usize;
        for ib in 0..QK / 32 {
            let ls = (2 * ((qh[ib] >> 12) & 7) + 1) as i32;
            let delta: i32 = if qh[ib] & 0x8000 != 0 { -1 } else { 1 };
            let mut lsum = 0i32;
            for l in 0..4 {
                let grid_idx = qs[qs_idx + l] as usize | (((qh[ib] >> (3 * l as u16)) & 7) as usize) << 8;
                let grid = grid_u64_bytes(&IQ1S_GRID, grid_idx);
                for j in 0..8 {
                    lsum += q8[q8_idx + j] as i32 * grid[j] as i8 as i32;
                }
                q8_idx += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2 * ib] as i32 + y[i].bsums[2 * ib + 1] as i32);
            qs_idx += 4;
        }
        sumf += f16_to_f32(x[i].d) * y[i].d * (sumi as f32 + IQ1S_DELTA * sumi1 as f32);
    }
    sumf
}

pub fn vec_dot_iq1_m_q8_k(n: usize, x: &[BlockIQ1M], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::*;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q8 = &y[i].qs;
        let qs = &x[i].qs;
        let qh = &x[i].qh;
        let sc = &x[i].scales;

        // reconstruct f16 scale from packed scales bytes
        let sc16 = [
            sc[0] as u16 | ((sc[1] as u16) << 8),
            sc[2] as u16 | ((sc[3] as u16) << 8),
            sc[4] as u16 | ((sc[5] as u16) << 8),
            sc[6] as u16 | ((sc[7] as u16) << 8),
        ];
        let scale_bits = (sc16[0] >> 12) | ((sc16[1] >> 8) & 0x00F0)
            | ((sc16[2] >> 4) & 0x0F00) | (sc16[3] & 0xF000);
        let scale_d = f16_to_f32(scale_bits);

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        let mut q8_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut qh_idx = 0usize;
        for ib in 0..QK / 32 {
            let delta = [
                if qh[qh_idx] & 0x08 != 0 { -1i32 } else { 1 },
                if qh[qh_idx] & 0x80 != 0 { -1i32 } else { 1 },
                if qh[qh_idx + 1] & 0x08 != 0 { -1i32 } else { 1 },
                if qh[qh_idx + 1] & 0x80 != 0 { -1i32 } else { 1 },
            ];
            let mut sum1 = [0i32; 2];
            let mut sum2 = [0i32; 2];
            for l in 0..4usize {
                let grid_idx = qs[qs_idx + l] as usize
                    | (((qh[qh_idx + l / 2] as u16) << (8 - 4 * (l % 2) as u16)) & 0x700) as usize;
                let grid = grid_u64_bytes(&IQ1S_GRID, grid_idx);
                let mut lsum1 = 0i32;
                let mut lsum2 = 0i32;
                for j in 0..8 {
                    lsum1 += q8[q8_idx + j] as i32 * grid[j] as i8 as i32;
                    lsum2 += q8[q8_idx + j] as i32;
                }
                q8_idx += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }

            let ls1 = (2 * ((sc16[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as i32;
            let ls2 = (2 * ((sc16[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as i32;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs_idx += 4;
            qh_idx += 2;
        }
        sumf += scale_d * y[i].d * (sumi1 as f32 + IQ1M_DELTA * sumi2 as f32);
    }
    sumf
}

pub fn vec_dot_iq4_xs_q8_k(n: usize, x: &[BlockIQ4XS], y: &[BlockQ8K]) -> f32 {
    use super::iq_tables::KVALUES_IQ4NL;
    const QK: usize = 256;
    let nb = n / QK;
    assert_eq!(n % QK, 0);
    let mut sumf = 0.0f32;
    for ibl in 0..nb {
        let d4d8 = f16_to_f32(x[ibl].d) * y[ibl].d;
        let mut h = x[ibl].scales_h;
        let qs = &x[ibl].qs;
        let q8 = &y[ibl].qs;
        let mut qs_idx = 0usize;
        let mut q8_idx = 0usize;
        for ib in (0..QK / 32).step_by(2) {
            let ls1 = ((x[ibl].scales_l[ib / 2] & 0xF) | ((h << 4) as u8 & 0x30)) as i32;
            let ls2 = ((x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) as u8 & 0x30)) as i32;
            h >>= 4;
            let d1 = d4d8 * (ls1 - 32) as f32;
            let d2 = d4d8 * (ls2 - 32) as f32;
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..16 {
                sumi1 += q8[q8_idx + j] as i32 * KVALUES_IQ4NL[(qs[qs_idx + j] & 0xF) as usize] as i32;
                sumi2 += q8[q8_idx + j + 16] as i32 * KVALUES_IQ4NL[(qs[qs_idx + j] >> 4) as usize] as i32;
            }
            sumf += d1 * (sumi1 + sumi2) as f32;
            qs_idx += 16;
            q8_idx += 32;
            sumi1 = 0;
            sumi2 = 0;
            for j in 0..16 {
                sumi1 += q8[q8_idx + j] as i32 * KVALUES_IQ4NL[(qs[qs_idx + j] & 0xF) as usize] as i32;
                sumi2 += q8[q8_idx + j + 16] as i32 * KVALUES_IQ4NL[(qs[qs_idx + j] >> 4) as usize] as i32;
            }
            sumf += d2 * (sumi1 + sumi2) as f32;
            qs_idx += 16;
            q8_idx += 32;
        }
    }
    sumf
}

// ── Raw-byte interface for type-erased dispatch ────────────────────────────

pub fn vec_dot(n: usize, result: &mut f32, vx: &[u8], vy: &[u8], weight_type: super::GgmlType) {
    use super::GgmlType::*;
    match weight_type {
        Q4_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ4_0, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_q4_0_q8_0(n, x, y);
        }
        Q4_1 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ4_1, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_1, n / 32) };
            *result = vec_dot_q4_1_q8_1(n, x, y);
        }
        Q5_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ5_0, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_q5_0_q8_0(n, x, y);
        }
        Q5_1 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ5_1, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_1, n / 32) };
            *result = vec_dot_q5_1_q8_1(n, x, y);
        }
        Q8_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ8_0, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_q8_0_q8_0(n, x, y);
        }
        Q4K => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ4K, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_q4_k_q8_k(n, x, y);
        }
        Q6K => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ6K, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_q6_k_q8_k(n, x, y);
        }
        Q2K => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ2K, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_q2_k_q8_k(n, x, y);
        }
        Q3K => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ3K, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_q3_k_q8_k(n, x, y);
        }
        Q5K => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ5K, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_q5_k_q8_k(n, x, y);
        }
        TQ2_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockTQ2_0, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_tq2_0_q8_k(n, x, y);
        }
        TQ1_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockTQ1_0, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_tq1_0_q8_k(n, x, y);
        }
        IQ4NL => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ4NL, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_iq4_nl_q8_0(n, x, y);
        }
        IQ4XS => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ4XS, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq4_xs_q8_k(n, x, y);
        }
        IQ2XXS => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ2XXS, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq2_xxs_q8_k(n, x, y);
        }
        IQ2XS => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ2XS, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq2_xs_q8_k(n, x, y);
        }
        IQ2S => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ2S, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq2_s_q8_k(n, x, y);
        }
        IQ3XXS => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ3XXS, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq3_xxs_q8_k(n, x, y);
        }
        IQ3S => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ3S, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq3_s_q8_k(n, x, y);
        }
        IQ1S => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ1S, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq1_s_q8_k(n, x, y);
        }
        IQ1M => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockIQ1M, n / 256) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8K, n / 256) };
            *result = vec_dot_iq1_m_q8_k(n, x, y);
        }
        MXFP4 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockMxfp4, n / 32) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_mxfp4_q8_0(n, x, y);
        }
        NVFP4 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockNvfp4, n / 64) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_nvfp4_q8_0(n, x, y);
        }
        Q1_0 => {
            let x = unsafe { std::slice::from_raw_parts(vx.as_ptr() as *const BlockQ1_0, n / 128) };
            let y = unsafe { std::slice::from_raw_parts(vy.as_ptr() as *const BlockQ8_0, n / 32) };
            *result = vec_dot_q1_0_q8_0(n, x, y);
        }
        _ => unimplemented!("vec_dot not implemented for {}", weight_type.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::helpers::f32_to_f16;
    use super::super::quantize::{quantize_row_q8_0, quantize_row_q8_k};
    use super::super::dequantize::{dequantize_row_q4_k, dequantize_row_q6_k, dequantize_row_q4_0};

    fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn q8_0_self_dot() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&src, &mut blocks);
        let dot = vec_dot_q8_0_q8_0(32, &blocks, &blocks);
        let expected = {
            let d = f16_to_f32(blocks[0].d);
            let mut s = 0i32;
            for j in 0..32 {
                s += blocks[0].qs[j] as i32 * blocks[0].qs[j] as i32;
            }
            s as f32 * d * d
        };
        assert!((dot - expected).abs() < 0.001, "dot={dot}, expected={expected}");
    }

    #[test]
    fn q4_0_dot_vs_dequant() {
        let w_data = BlockQ4_0 {
            d: f32_to_f16(0.5),
            qs: [0x48; 16], // nibbles: 8 and 4 → values 0 and -4
        };
        let mut w_f32 = [0.0f32; 32];
        dequantize_row_q4_0(&[w_data], &mut w_f32);

        let act: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let mut act_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }];
        quantize_row_q8_0(&act, &mut act_q8);
        let mut act_f32 = [0.0f32; 32];
        super::super::dequantize::dequantize_row_q8_0(&act_q8, &mut act_f32);

        let ref_dot = f32_dot(&w_f32, &act_f32);
        let q_dot = vec_dot_q4_0_q8_0(32, &[w_data], &act_q8);
        let rel_err = if ref_dot.abs() > 0.001 {
            ((q_dot - ref_dot) / ref_dot).abs()
        } else {
            (q_dot - ref_dot).abs()
        };
        assert!(rel_err < 0.05, "q_dot={q_dot}, ref_dot={ref_dot}, rel_err={rel_err}");
    }

    #[test]
    fn q4k_dot_vs_dequant() {
        // create a Q4K block with known values
        let w = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], // scale=1 for first 4 groups, min=0
            qs: [0x11; 128], // all nibbles = 1
        };
        let mut w_f32 = [0.0f32; 256];
        dequantize_row_q4_k(&[w], &mut w_f32);

        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);
        let mut act_f32 = [0.0f32; 256];
        for j in 0..256 {
            act_f32[j] = act_q8k[0].d * act_q8k[0].qs[j] as f32;
        }

        let ref_dot = f32_dot(&w_f32, &act_f32);
        let q_dot = vec_dot_q4_k_q8_k(256, &[w], &act_q8k);
        let rel_err = if ref_dot.abs() > 0.001 {
            ((q_dot - ref_dot) / ref_dot).abs()
        } else {
            (q_dot - ref_dot).abs()
        };
        assert!(rel_err < 0.1, "q_dot={q_dot}, ref_dot={ref_dot}, rel_err={rel_err}");
    }

    #[test]
    fn q6k_dot_vs_dequant() {
        let w = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [1; 16],
            d: f32_to_f16(1.0),
        };
        let mut w_f32 = [0.0f32; 256];
        dequantize_row_q6_k(&[w], &mut w_f32);

        let act: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);
        let mut act_f32 = [0.0f32; 256];
        for j in 0..256 {
            act_f32[j] = act_q8k[0].d * act_q8k[0].qs[j] as f32;
        }

        let ref_dot = f32_dot(&w_f32, &act_f32);
        let q_dot = vec_dot_q6_k_q8_k(256, &[w], &act_q8k);
        let rel_err = if ref_dot.abs() > 0.001 {
            ((q_dot - ref_dot) / ref_dot).abs()
        } else {
            (q_dot - ref_dot).abs()
        };
        assert!(rel_err < 0.1, "q_dot={q_dot}, ref_dot={ref_dot}, rel_err={rel_err}");
    }

    #[test]
    fn tq2_0_dot_identity() {
        // all trits = 1 (value = 0), dot should be ~0
        let w = BlockTQ2_0 {
            qs: [0x55; 64], // 0b01_01_01_01 = all trits 1 → all values 0
            d: f32_to_f16(1.0),
        };
        let act: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut act_q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        quantize_row_q8_k(&act, &mut act_q8k);
        let dot = vec_dot_tq2_0_q8_k(256, &[w], &act_q8k);
        assert!(dot.abs() < 0.01, "expected ~0, got {dot}");
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn q4_0_nibble_extraction() {
        let byte: u8 = kani::any();
        let j: usize = kani::any();
        kani::assume(j < 16);
        let v0 = (byte & 0x0F) as i32 - 8;
        let v1 = (byte >> 4) as i32 - 8;
        assert!(v0 >= -8 && v0 <= 7);
        assert!(v1 >= -8 && v1 <= 7);
        assert!(j < 16);
        assert!(j + 16 < 32);
    }

    #[kani::proof]
    fn q4k_scale_unpack_bounds() {
        let scales: [u8; 12] = kani::any();
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
        utmp[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
        utmp[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        // each byte in utmp[0..2] (scales) should be <= 63
        for byte in utmp[0].to_le_bytes() { assert!(byte <= 63); }
        for byte in utmp[2].to_le_bytes() { assert!(byte <= 63); }
        // each byte in utmp[1], utmp[3] (mins) should be <= 63
        for byte in utmp[1].to_le_bytes() { assert!(byte <= 63); }
        for byte in utmp[3].to_le_bytes() { assert!(byte <= 63); }
    }

    #[kani::proof]
    fn q6k_dequant_indices() {
        let half: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(half < 2 && l < 32);
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;
        assert!(ql_off + l < 128);
        assert!(ql_off + 32 + l < 128);
        assert!(qh_off + l < 64);
        assert!(y_off + l < 256);
        assert!(y_off + 32 + l < 256);
        assert!(y_off + 64 + l < 256);
        assert!(y_off + 96 + l < 256);
    }

    #[kani::proof]
    fn tq2_trit_range() {
        let byte: u8 = kani::any();
        let b: usize = kani::any();
        kani::assume(b < 4);
        let trit = ((byte >> (b * 2)) & 3) as i32;
        assert!(trit >= 0 && trit <= 3);
        let value = trit - 1;
        assert!(value >= -1 && value <= 2);
    }

    #[kani::proof]
    #[kani::unwind(17)]
    fn q4_0_accumulator_no_i32_overflow() {
        let qs: [u8; 16] = kani::any();
        let y_qs: [i8; 32] = kani::any();
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..16 {
            let v0 = (qs[j] & 0x0F) as i32 - 8;  // [-8, 7]
            let v1 = (qs[j] >> 4) as i32 - 8;     // [-8, 7]
            // max product: 8 * 128 = 1024, 16 terms: 16384
            sumi0 += v0 * y_qs[j] as i32;
            sumi1 += v1 * y_qs[j + 16] as i32;
        }
        // 16 * 8 * 128 = 16384, well within i32
        let total = sumi0 + sumi1;
        assert!(total >= -32768 && total <= 32768);
    }

    #[kani::proof]
    fn q4k_aux8_and_q8_indices() {
        let j: usize = kani::any();
        kani::assume(j < 256);
        // aux8 index: q4 nibbles are unpacked into 256-element array
        // for each 64-element chunk: 32 low nibbles + 32 high nibbles
        let chunk = j / 64;
        let within = j % 64;
        let q4_off = chunk * 32;
        if within < 32 {
            assert!(q4_off + within < 128); // qs array is [u8; 128]
        } else {
            assert!(q4_off + (within - 32) < 128);
        }
        // q8 index: matches j directly
        assert!(j < 256); // qs array is [i8; 256]
    }

    #[kani::proof]
    #[kani::unwind(17)]
    fn vec_dot_q4_0_q8_0_no_panic() {
        let x = BlockQ4_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let y = BlockQ8_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let _ = vec_dot_q4_0_q8_0(32, &[x], &[y]);
    }

    #[kani::proof]
    #[kani::unwind(33)]
    fn vec_dot_q8_0_q8_0_no_panic() {
        let x = BlockQ8_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let y = BlockQ8_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let _ = vec_dot_q8_0_q8_0(32, &[x], &[y]);
    }

    #[kani::proof]
    #[kani::unwind(65)]
    fn vec_dot_tq2_0_q8_k_no_panic() {
        let x = BlockTQ2_0 {
            qs: kani::any(),
            d: kani::any(),
        };
        let y = BlockQ8K {
            d: kani::any(),
            qs: kani::any(),
            bsums: kani::any(),
        };
        let _ = vec_dot_tq2_0_q8_k(256, &[x], &[y]);
    }
}

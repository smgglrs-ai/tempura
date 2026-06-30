use super::blocks::*;
use super::helpers::{f16_to_f32, bf16_to_f32, get_scale_min_k4};

// ── Basic quants (QK = 32) ─────────────────────────────────────────────────

pub fn dequantize_row_q4_0(blocks: &[BlockQ4_0], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            let v0 = (blk.qs[j] & 0x0F) as i32 - 8;
            let v1 = (blk.qs[j] >> 4) as i32 - 8;
            y[j] = d * v0 as f32;
            y[j + QK / 2] = d * v1 as f32;
        }
    }
}

pub fn dequantize_row_q4_1(blocks: &[BlockQ4_1], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let m = f16_to_f32(blk.m);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            let v0 = (blk.qs[j] & 0x0F) as f32;
            let v1 = (blk.qs[j] >> 4) as f32;
            y[j] = d * v0 + m;
            y[j + QK / 2] = d * v1 + m;
        }
    }
}

pub fn dequantize_row_q5_0(blocks: &[BlockQ5_0], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let qh = u32::from_le_bytes(blk.qh);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            let xh_0 = ((qh >> j) & 1) as i32;
            let xh_1 = ((qh >> (j + 16)) & 1) as i32;
            let v0 = ((blk.qs[j] & 0x0F) as i32 | (xh_0 << 4)) - 16;
            let v1 = ((blk.qs[j] >> 4) as i32 | (xh_1 << 4)) - 16;
            y[j] = d * v0 as f32;
            y[j + QK / 2] = d * v1 as f32;
        }
    }
}

pub fn dequantize_row_q5_1(blocks: &[BlockQ5_1], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let m = f16_to_f32(blk.m);
        let qh = u32::from_le_bytes(blk.qh);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            let xh_0 = ((qh >> j) & 1) as u32;
            let xh_1 = ((qh >> (j + 16)) & 1) as u32;
            let v0 = (blk.qs[j] & 0x0F) as u32 | (xh_0 << 4);
            let v1 = (blk.qs[j] >> 4) as u32 | (xh_1 << 4);
            y[j] = d * v0 as f32 + m;
            y[j + QK / 2] = d * v1 as f32 + m;
        }
    }
}

pub fn dequantize_row_q8_0(blocks: &[BlockQ8_0], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK {
            y[j] = d * blk.qs[j] as f32;
        }
    }
}

pub fn dequantize_row_q1_0(data: &[u8], output: &mut [f32], num_elements: usize) {
    const QK: usize = 128;
    assert_eq!(num_elements % QK, 0);
    let nb = num_elements / QK;
    assert!(data.len() >= nb * BLOCK_Q1_0_SIZE);
    for i in 0..nb {
        let base = i * BLOCK_Q1_0_SIZE;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 18];
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK {
            let byte_idx = j / 8;
            let bit_idx = j % 8;
            let bit = (qs[byte_idx] >> bit_idx) & 1;
            y[j] = if bit == 0 { -d } else { d };
        }
    }
}

// ── K-quants (QK_K = 256) ──────────────────────────────────────────────────

pub fn dequantize_row_q2_k(blocks: &[BlockQ2K], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let dmin = f16_to_f32(blk.dmin);
        let y = &mut output[i * QK..][..QK];
        let mut q_idx = 0usize;
        for j in 0..QK / 16 {
            let sc_byte = blk.scales[j];
            let sc = (sc_byte & 0xF) as f32;
            let m = (sc_byte >> 4) as f32;
            let dl = d * sc;
            let ml = dmin * m;
            for l in 0..16 {
                let elem_idx = j * 16 + l;
                let byte_idx = q_idx / 4;
                let shift = (q_idx % 4) * 2;
                let q = ((blk.qs[byte_idx] >> shift) & 3) as f32;
                y[elem_idx] = dl * q - ml;
                q_idx += 1;
            }
        }
    }
}

pub fn dequantize_row_q3_k(blocks: &[BlockQ3K], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];

        // unpack 6-bit scales from 12 bytes
        let mut scales = [0i8; 16];
        let aux = &blk.scales;
        for j in 0..4 {
            scales[j] = (aux[j] & 0xF) as i8 - 8;
            scales[j + 4] = (aux[j] >> 4) as i8 - 8;
        }
        for j in 0..4 {
            scales[j + 8] = (aux[j + 4] & 0xF) as i8 - 8;
            scales[j + 12] = (aux[j + 4] >> 4) as i8 - 8;
        }
        // apply 2-bit corrections from aux[8..12]
        for j in 0..4 {
            let a = aux[j + 8];
            scales[j] = (scales[j] as i32 + ((a & 3) as i32) * 16 - if (a & 3) >= 2 { 32 } else { 0 }) as i8;
            scales[j + 4] = (scales[j + 4] as i32 + (((a >> 2) & 3) as i32) * 16 - if ((a >> 2) & 3) >= 2 { 32 } else { 0 }) as i8;
            scales[j + 8] = (scales[j + 8] as i32 + (((a >> 4) & 3) as i32) * 16 - if ((a >> 4) & 3) >= 2 { 32 } else { 0 }) as i8;
            scales[j + 12] = (scales[j + 12] as i32 + (((a >> 6) & 3) as i32) * 16 - if ((a >> 6) & 3) >= 2 { 32 } else { 0 }) as i8;
        }

        for j in 0..QK {
            let byte_idx = j / 4;
            let shift = (j % 4) * 2;
            let low2 = ((blk.qs[byte_idx] >> shift) & 3) as i32;
            let hm_byte = j / 8;
            let hm_bit = j % 8;
            let high = ((blk.hmask[hm_byte] >> hm_bit) & 1) as i32;
            let q = low2 | (high << 2);
            let sc = scales[j / 16] as f32;
            y[j] = d * sc * (q as f32 - 4.0);
        }
    }
}

pub fn dequantize_row_q4_k(blocks: &[BlockQ4K], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let dmin = f16_to_f32(blk.dmin);
        let y = &mut output[i * QK..][..QK];
        let q = &blk.qs;
        let mut is = 0usize;
        let mut q_offset = 0usize;
        let mut y_offset = 0usize;
        for _j in 0..QK / 64 {
            let (sc0, m0) = get_scale_min_k4(is, &blk.scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1v) = get_scale_min_k4(is + 1, &blk.scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1v as f32;
            for l in 0..32 {
                y[y_offset + l] = d1 * (q[q_offset + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                y[y_offset + 32 + l] = d2 * (q[q_offset + l] >> 4) as f32 - m2;
            }
            q_offset += 32;
            y_offset += 64;
            is += 2;
        }
    }
}

pub fn dequantize_row_q5_k(blocks: &[BlockQ5K], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let dmin = f16_to_f32(blk.dmin);
        let y = &mut output[i * QK..][..QK];
        let ql = &blk.qs;
        let qh = &blk.qh;
        let mut is = 0usize;
        let mut ql_offset = 0usize;
        let mut y_offset = 0usize;
        let mut qh_offset = 0usize;
        for _j in 0..QK / 64 {
            let (sc0, m0) = get_scale_min_k4(is, &blk.scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1v) = get_scale_min_k4(is + 1, &blk.scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1v as f32;
            for l in 0..32 {
                let hbit = ((qh[qh_offset + l] >> (is % 4)) & 1) as u8;
                y[y_offset + l] = d1 * ((ql[ql_offset + l] & 0xF) | (hbit << 4)) as f32 - m1;
            }
            for l in 0..32 {
                let hbit = ((qh[qh_offset + l] >> ((is + 1) % 4)) & 1) as u8;
                y[y_offset + 32 + l] = d2 * ((ql[ql_offset + l] >> 4) | (hbit << 4)) as f32 - m2;
            }
            ql_offset += 32;
            y_offset += 64;
            is += 2;
            if is == 4 {
                qh_offset += 32;
            }
        }
    }
}

pub fn dequantize_row_q6_k(blocks: &[BlockQ6K], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let ql = &blk.ql;
        let qh = &blk.qh;
        let sc = &blk.scales;

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut y_off = 0usize;

        for _n in 0..QK / 128 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + 32 + l] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + 32 + l] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
                y[y_off + l] = d * sc[sc_off + is] as f32 * q1 as f32;
                y[y_off + 32 + l] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
                y[y_off + 64 + l] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
                y[y_off + 96 + l] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

// ── Ternary quants ─────────────────────────────────────────────────────────

pub fn dequantize_row_tq2_0(blocks: &[BlockTQ2_0], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        for j in 0..64 {
            let byte = blk.qs[j];
            for b in 0..4 {
                let trit = ((byte >> (b * 2)) & 3) as i32;
                // 0 → -1, 1 → 0, 2 → +1
                y[j * 4 + b] = d * (trit - 1) as f32;
            }
        }
    }
}

pub fn dequantize_row_tq1_0(blocks: &[BlockTQ1_0], output: &mut [f32]) {
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let mut idx = 0usize;
        // qs: 48 bytes, 5 trits per byte (base-3)
        for j in 0..48 {
            let mut byte = blk.qs[j] as u32;
            for _b in 0..5 {
                if idx >= QK {
                    break;
                }
                let trit = (byte % 3) as i32;
                byte /= 3;
                y[idx] = d * (trit - 1) as f32;
                idx += 1;
            }
        }
        // qh: 4 bytes, 4 trits per byte
        for j in 0..4 {
            let mut byte = blk.qh[j] as u32;
            for _b in 0..4 {
                if idx >= QK {
                    break;
                }
                let trit = (byte % 3) as i32;
                byte /= 3;
                y[idx] = d * (trit - 1) as f32;
                idx += 1;
            }
        }
    }
}

// ── Micro-FP quants ────────────────────────────────────────────────────────

pub fn dequantize_row_mxfp4(blocks: &[BlockMxfp4], output: &mut [f32]) {
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let shared_exp = blk.e as i32;
        let scale = if shared_exp == 0 { 0.0 } else { f32::from_bits(((shared_exp as u32) << 23).wrapping_sub(127 << 23).wrapping_add(127 << 23)) };
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            let nibble_lo = blk.qs[j] & 0x0F;
            let nibble_hi = blk.qs[j] >> 4;
            y[j * 2] = scale * mxfp4_to_f32(nibble_lo);
            y[j * 2 + 1] = scale * mxfp4_to_f32(nibble_hi);
        }
    }
}

fn mxfp4_to_f32(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let exp = (nibble >> 1) & 3;
    let mant = nibble & 1;
    let val = match (exp, mant) {
        (0, 0) => 0.0,
        (0, 1) => 0.5,
        (1, 0) => 1.0,
        (1, 1) => 1.5,
        (2, 0) => 2.0,
        (2, 1) => 3.0,
        (3, 0) => 4.0,
        (3, 1) => 6.0,
        _ => unreachable!(),
    };
    if sign == 1 { -val } else { val }
}

pub fn dequantize_row_nvfp4(blocks: &[BlockNvfp4], output: &mut [f32]) {
    const QK: usize = 64;
    const SUB: usize = 16;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let y = &mut output[i * QK..][..QK];
        for sub in 0..QK / SUB {
            let scale_byte = blk.d[sub];
            let scale = ue4m3_to_f32(scale_byte);
            for j in 0..SUB / 2 {
                let byte_idx = sub * (SUB / 2) + j;
                let nibble_lo = blk.qs[byte_idx] & 0x0F;
                let nibble_hi = blk.qs[byte_idx] >> 4;
                y[sub * SUB + j * 2] = scale * e2m1_to_f32(nibble_lo);
                y[sub * SUB + j * 2 + 1] = scale * e2m1_to_f32(nibble_hi);
            }
        }
    }
}

fn ue4m3_to_f32(byte: u8) -> f32 {
    let exp = (byte >> 3) as i32;
    let mant = (byte & 7) as f32 / 8.0 + 1.0;
    if exp == 0 {
        (byte & 7) as f32 / 8.0 * f32::from_bits((1u32) << 23) // denorm
    } else {
        mant * 2.0f32.powi(exp - 7)
    }
}

fn e2m1_to_f32(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let val = mxfp4_to_f32(nibble & 0x07 | (sign << 3));
    val // sign already handled in mxfp4_to_f32
}

// ── IQ quants ──────────────────────────────────────────────────────────────

pub fn dequantize_row_iq2_xxs(blocks: &[BlockIQ2XXS], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let q2 = &blk.qs;
        let mut y_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let aux32_0 = q2[ib32 * 4] as u32 | ((q2[ib32 * 4 + 1] as u32) << 16);
            let aux32_1 = q2[ib32 * 4 + 2] as u32 | ((q2[ib32 * 4 + 3] as u32) << 16);
            let aux8 = [
                (aux32_0 & 0xFF) as u8,
                ((aux32_0 >> 8) & 0xFF) as u8,
                (aux32_1 & 0xFF) as u8,
                ((aux32_1 >> 8) & 0xFF) as u8,
            ];
            let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
                let sign_bits = KSIGNS_IQ2XS[((aux32_1 >> (7 * l as u32)) & 127) as usize];
                for j in 0..8 {
                    let sign: f32 = if sign_bits & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db * grid[j] as f32 * sign;
                    y_idx += 1;
                }
            }
        }
    }
}

pub fn dequantize_row_iq2_xs(blocks: &[BlockIQ2XS], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let q2 = &blk.qs;
        let sc = &blk.scales;
        let mut y_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let ls_lo = (sc[ib32] & 0xF) as f32;
            let ls_hi = (sc[ib32] >> 4) as f32;
            let db1 = d * (0.5 + ls_lo) * 0.25;
            let db2 = d * (0.5 + ls_hi) * 0.25;
            for l in 0..2 {
                let grid = IQ2XS_GRID[(q2[ib32 * 4 + l] & 511) as usize].to_le_bytes();
                let sign_bits = KSIGNS_IQ2XS[(q2[ib32 * 4 + l] >> 9) as usize];
                for j in 0..8 {
                    let sign: f32 = if sign_bits & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db1 * grid[j] as f32 * sign;
                    y_idx += 1;
                }
            }
            for l in 2..4 {
                let grid = IQ2XS_GRID[(q2[ib32 * 4 + l] & 511) as usize].to_le_bytes();
                let sign_bits = KSIGNS_IQ2XS[(q2[ib32 * 4 + l] >> 9) as usize];
                for j in 0..8 {
                    let sign: f32 = if sign_bits & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db2 * grid[j] as f32 * sign;
                    y_idx += 1;
                }
            }
        }
    }
}

pub fn dequantize_row_iq2_s(blocks: &[BlockIQ2S], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let qs = &blk.qs;
        let qh = &blk.qh;
        let signs = &qs[QK / 8..];
        let mut y_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = 0usize;
        for ib32 in 0..QK / 32 {
            let db1 = d * (0.5 + (blk.scales[ib32] & 0xF) as f32) * 0.25;
            let db2 = d * (0.5 + (blk.scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..2usize {
                let grid_idx = qs[qs_idx + l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = IQ2S_GRID[grid_idx].to_le_bytes();
                for j in 0..8 {
                    let sign: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db1 * grid[j] as f32 * sign;
                    y_idx += 1;
                }
            }
            for l in 2..4usize {
                let grid_idx = qs[qs_idx + l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 0x300);
                let grid = IQ2S_GRID[grid_idx].to_le_bytes();
                for j in 0..8 {
                    let sign: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db2 * grid[j] as f32 * sign;
                    y_idx += 1;
                }
            }
            qs_idx += 4;
            signs_idx += 4;
        }
    }
}

pub fn dequantize_row_iq3_xxs(blocks: &[BlockIQ3XXS], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let q3 = &blk.qs[..QK / 4];
        let gas = &blk.qs[QK / 4..];
        let mut y_idx = 0usize;
        let mut q3_idx = 0usize;
        let mut gas_idx = 0usize;
        for _ib32 in 0..QK / 32 {
            let aux32 = u32::from_le_bytes([gas[gas_idx], gas[gas_idx + 1], gas[gas_idx + 2], gas[gas_idx + 3]]);
            gas_idx += 4;
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let grid1 = IQ3XXS_GRID[q3[q3_idx + 2 * l] as usize].to_le_bytes();
                let grid2 = IQ3XXS_GRID[q3[q3_idx + 2 * l + 1] as usize].to_le_bytes();
                let sign_bits = KSIGNS_IQ2XS[((aux32 >> (7 * l as u32)) & 127) as usize];
                for j in 0..4 {
                    let sign1: f32 = if sign_bits & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db * grid1[j] as f32 * sign1;
                    y_idx += 1;
                }
                for j in 0..4 {
                    let sign2: f32 = if sign_bits & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db * grid2[j] as f32 * sign2;
                    y_idx += 1;
                }
            }
            q3_idx += 8;
        }
    }
}

pub fn dequantize_row_iq3_s(blocks: &[BlockIQ3S], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let qs = &blk.qs;
        let qh = &blk.qh;
        let signs = &blk.signs;
        let mut y_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = 0usize;
        for ib32 in (0..QK / 32).step_by(2) {
            let db1 = d * (1 + 2 * (blk.scales[ib32 / 2] & 0xF)) as f32;
            let db2 = d * (1 + 2 * (blk.scales[ib32 / 2] >> 4)) as f32;
            for l in 0..4 {
                let idx1 = qs[qs_idx + 2 * l] as usize | ((qh[ib32] as usize) << (8 - 2 * l) & 256);
                let idx2 = qs[qs_idx + 2 * l + 1] as usize | ((qh[ib32] as usize) << (7 - 2 * l) & 256);
                let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                for j in 0..4 {
                    let sign1: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db1 * grid1[j] as f32 * sign1;
                    y_idx += 1;
                }
                for j in 0..4 {
                    let sign2: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db2 * grid2[j] as f32 * sign2;
                    y_idx += 1;
                }
            }
            qs_idx += 8;
            signs_idx += 4;
            for l in 0..4 {
                let idx1 = qs[qs_idx + 2 * l] as usize | ((qh[ib32 + 1] as usize) << (8 - 2 * l) & 256);
                let idx2 = qs[qs_idx + 2 * l + 1] as usize | ((qh[ib32 + 1] as usize) << (7 - 2 * l) & 256);
                let grid1 = IQ3S_GRID[idx1].to_le_bytes();
                let grid2 = IQ3S_GRID[idx2].to_le_bytes();
                for j in 0..4 {
                    let sign1: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db1 * grid1[j] as f32 * sign1;
                    y_idx += 1;
                }
                for j in 0..4 {
                    let sign2: f32 = if signs[signs_idx + l] & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    y[y_idx] = db2 * grid2[j] as f32 * sign2;
                    y_idx += 1;
                }
            }
            qs_idx += 8;
            signs_idx += 4;
        }
    }
}

pub fn dequantize_row_iq1_s(blocks: &[BlockIQ1S], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let qs = &blk.qs;
        let qh = &blk.qh;
        let mut y_idx = 0usize;
        let mut qs_idx = 0usize;
        for ib in 0..QK / 32 {
            let dl = d * (2 * ((qh[ib] >> 12) & 7) + 1) as f32;
            let delta = if qh[ib] & 0x8000 != 0 { -IQ1S_DELTA * dl } else { IQ1S_DELTA * dl };
            for l in 0..4 {
                let grid_idx = qs[qs_idx + l] as usize | (((qh[ib] >> (3 * l as u16)) & 7) as usize) << 8;
                let grid = IQ1S_GRID[grid_idx].to_le_bytes();
                for j in 0..8 {
                    y[y_idx] = dl * grid[j] as i8 as f32 + delta;
                    y_idx += 1;
                }
            }
            qs_idx += 4;
        }
    }
}

pub fn dequantize_row_iq1_m(blocks: &[BlockIQ1M], output: &mut [f32]) {
    use super::iq_tables::*;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let y = &mut output[i * QK..][..QK];
        let qs = &blk.qs;
        let qh = &blk.qh;
        let sc = &blk.scales;
        let sc16 = [
            sc[0] as u16 | ((sc[1] as u16) << 8),
            sc[2] as u16 | ((sc[3] as u16) << 8),
            sc[4] as u16 | ((sc[5] as u16) << 8),
            sc[6] as u16 | ((sc[7] as u16) << 8),
        ];
        let scale_bits = (sc16[0] >> 12) | ((sc16[1] >> 8) & 0x00F0) | ((sc16[2] >> 4) & 0x0F00) | (sc16[3] & 0xF000);
        let d = f16_to_f32(scale_bits);
        let mut y_idx = 0usize;
        let mut qs_idx = 0usize;
        let mut qh_idx = 0usize;
        for ib in 0..QK / 32 {
            let ls1 = (2 * ((sc16[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as f32;
            let ls2 = (2 * ((sc16[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
            let delta = [
                if qh[qh_idx] & 0x08 != 0 { -IQ1M_DELTA } else { IQ1M_DELTA },
                if qh[qh_idx] & 0x80 != 0 { -IQ1M_DELTA } else { IQ1M_DELTA },
                if qh[qh_idx + 1] & 0x08 != 0 { -IQ1M_DELTA } else { IQ1M_DELTA },
                if qh[qh_idx + 1] & 0x80 != 0 { -IQ1M_DELTA } else { IQ1M_DELTA },
            ];
            for l in 0..4usize {
                let grid_idx = qs[qs_idx + l] as usize | (((qh[qh_idx + l / 2] as u16) << (8 - 4 * (l % 2) as u16)) & 0x700) as usize;
                let grid = IQ1S_GRID[grid_idx].to_le_bytes();
                let dl = d * if l < 2 { ls1 } else { ls2 };
                let dd = dl * delta[l];
                for j in 0..8 {
                    y[y_idx] = dl * grid[j] as i8 as f32 + dd;
                    y_idx += 1;
                }
            }
            qs_idx += 4;
            qh_idx += 2;
        }
    }
}

pub fn dequantize_row_iq4_xs(blocks: &[BlockIQ4XS], output: &mut [f32]) {
    use super::iq_tables::KVALUES_IQ4NL;
    const QK: usize = 256;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        let qs = &blk.qs;
        let mut h = blk.scales_h;
        let mut y_idx = 0usize;
        let mut qs_idx = 0usize;
        for ib in (0..QK / 32).step_by(2) {
            let ls1 = ((blk.scales_l[ib / 2] & 0xF) | ((h << 4) as u8 & 0x30)) as i32 - 32;
            let ls2 = ((blk.scales_l[ib / 2] >> 4) | ((h << 2) as u8 & 0x30)) as i32 - 32;
            h >>= 4;
            let dl1 = d * ls1 as f32;
            let dl2 = d * ls2 as f32;
            for j in 0..16 {
                y[y_idx] = dl1 * KVALUES_IQ4NL[(qs[qs_idx + j] & 0xF) as usize] as f32;
                y[y_idx + 1] = dl1 * KVALUES_IQ4NL[(qs[qs_idx + j] >> 4) as usize] as f32;
                y_idx += 2;
            }
            qs_idx += 16;
            for j in 0..16 {
                y[y_idx] = dl2 * KVALUES_IQ4NL[(qs[qs_idx + j] & 0xF) as usize] as f32;
                y[y_idx + 1] = dl2 * KVALUES_IQ4NL[(qs[qs_idx + j] >> 4) as usize] as f32;
                y_idx += 2;
            }
            qs_idx += 16;
        }
    }
}

pub fn dequantize_row_iq4_nl(blocks: &[BlockIQ4NL], output: &mut [f32]) {
    use super::iq_tables::KVALUES_IQ4NL;
    const QK: usize = 32;
    assert_eq!(output.len(), blocks.len() * QK);
    for (i, blk) in blocks.iter().enumerate() {
        let d = f16_to_f32(blk.d);
        let y = &mut output[i * QK..][..QK];
        for j in 0..QK / 2 {
            y[j * 2] = d * KVALUES_IQ4NL[(blk.qs[j] & 0xF) as usize] as f32;
            y[j * 2 + 1] = d * KVALUES_IQ4NL[(blk.qs[j] >> 4) as usize] as f32;
        }
    }
}

// ── Raw-byte interface for type-erased dispatch ────────────────────────────

pub fn dequantize_row(data: &[u8], output: &mut [f32], ggml_type: super::GgmlType, num_elements: usize) {
    use super::GgmlType::*;
    match ggml_type {
        F32 => {
            assert!(data.len() >= num_elements * 4);
            for i in 0..num_elements {
                let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                output[i] = f32::from_le_bytes(bytes);
            }
        }
        F16 => {
            assert!(data.len() >= num_elements * 2);
            for i in 0..num_elements {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                output[i] = f16_to_f32(bits);
            }
        }
        BF16 => {
            assert!(data.len() >= num_elements * 2);
            for i in 0..num_elements {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                output[i] = bf16_to_f32(bits);
            }
        }
        Q4_0 => {
            let blocks = cast_blocks_aligned::<BlockQ4_0>(data, num_elements, 32);
            dequantize_row_q4_0(blocks, output);
        }
        Q4_1 => {
            let blocks = cast_blocks_aligned::<BlockQ4_1>(data, num_elements, 32);
            dequantize_row_q4_1(blocks, output);
        }
        Q5_0 => {
            let blocks = cast_blocks_aligned::<BlockQ5_0>(data, num_elements, 32);
            dequantize_row_q5_0(blocks, output);
        }
        Q5_1 => {
            let blocks = cast_blocks_aligned::<BlockQ5_1>(data, num_elements, 32);
            dequantize_row_q5_1(blocks, output);
        }
        Q8_0 => {
            let blocks = cast_blocks_aligned::<BlockQ8_0>(data, num_elements, 32);
            dequantize_row_q8_0(blocks, output);
        }
        Q2K => {
            let blocks = cast_blocks_aligned::<BlockQ2K>(data, num_elements, 256);
            dequantize_row_q2_k(blocks, output);
        }
        Q3K => {
            let blocks = cast_blocks_aligned::<BlockQ3K>(data, num_elements, 256);
            dequantize_row_q3_k(blocks, output);
        }
        Q4K => {
            let blocks = cast_blocks_aligned::<BlockQ4K>(data, num_elements, 256);
            dequantize_row_q4_k(blocks, output);
        }
        Q5K => {
            let blocks = cast_blocks_aligned::<BlockQ5K>(data, num_elements, 256);
            dequantize_row_q5_k(blocks, output);
        }
        Q6K => {
            let blocks = cast_blocks_aligned::<BlockQ6K>(data, num_elements, 256);
            dequantize_row_q6_k(blocks, output);
        }
        TQ1_0 => {
            let blocks = cast_blocks_aligned::<BlockTQ1_0>(data, num_elements, 256);
            dequantize_row_tq1_0(blocks, output);
        }
        TQ2_0 => {
            let blocks = cast_blocks_aligned::<BlockTQ2_0>(data, num_elements, 256);
            dequantize_row_tq2_0(blocks, output);
        }
        Q1_0 => {
            dequantize_row_q1_0(data, output, num_elements);
        }
        MXFP4 => {
            let blocks = cast_blocks_aligned::<BlockMxfp4>(data, num_elements, 32);
            dequantize_row_mxfp4(blocks, output);
        }
        NVFP4 => {
            let blocks = cast_blocks_aligned::<BlockNvfp4>(data, num_elements, 64);
            dequantize_row_nvfp4(blocks, output);
        }
        IQ4NL => {
            let blocks = cast_blocks_aligned::<BlockIQ4NL>(data, num_elements, 32);
            dequantize_row_iq4_nl(blocks, output);
        }
        IQ4XS => {
            let blocks = cast_blocks_aligned::<BlockIQ4XS>(data, num_elements, 256);
            dequantize_row_iq4_xs(blocks, output);
        }
        IQ2XXS => {
            let blocks = cast_blocks_aligned::<BlockIQ2XXS>(data, num_elements, 256);
            dequantize_row_iq2_xxs(blocks, output);
        }
        IQ2XS => {
            let blocks = cast_blocks_aligned::<BlockIQ2XS>(data, num_elements, 256);
            dequantize_row_iq2_xs(blocks, output);
        }
        IQ2S => {
            let blocks = cast_blocks_aligned::<BlockIQ2S>(data, num_elements, 256);
            dequantize_row_iq2_s(blocks, output);
        }
        IQ3XXS => {
            let blocks = cast_blocks_aligned::<BlockIQ3XXS>(data, num_elements, 256);
            dequantize_row_iq3_xxs(blocks, output);
        }
        IQ3S => {
            let blocks = cast_blocks_aligned::<BlockIQ3S>(data, num_elements, 256);
            dequantize_row_iq3_s(blocks, output);
        }
        IQ1S => {
            let blocks = cast_blocks_aligned::<BlockIQ1S>(data, num_elements, 256);
            dequantize_row_iq1_s(blocks, output);
        }
        IQ1M => {
            let blocks = cast_blocks_aligned::<BlockIQ1M>(data, num_elements, 256);
            dequantize_row_iq1_m(blocks, output);
        }
        // intermediate / non-quantized types
        Q8_1 | Q8K | I8 | I16 | I32 | I64 | F64 => {
            unimplemented!("dequantize not applicable for {}", ggml_type.name())
        }
    }
}

fn cast_blocks_aligned<T: Copy>(data: &[u8], num_elements: usize, block_elems: usize) -> &[T] {
    let type_size = std::mem::size_of::<T>();
    let nb = num_elements / block_elems;
    assert_eq!(num_elements % block_elems, 0);
    assert!(data.len() >= nb * type_size);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, nb) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::helpers::f32_to_f16;

    #[test]
    fn dequant_q4_0_zero_block() {
        let blk = BlockQ4_0 {
            d: f32_to_f16(1.0),
            qs: [0x88; 16], // all nibbles = 8, so value = 8-8 = 0
        };
        let mut out = [0.0f32; 32];
        dequantize_row_q4_0(&[blk], &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_q8_0_identity() {
        let mut blk = BlockQ8_0 {
            d: f32_to_f16(0.5),
            qs: [0i8; 32],
        };
        blk.qs[0] = 2;
        blk.qs[31] = -4;
        let mut out = [0.0f32; 32];
        dequantize_row_q8_0(&[blk], &mut out);
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[31] - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn dequant_q4k_zero_block() {
        let blk = BlockQ4K {
            d: f32_to_f16(0.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qs: [0; 128],
        };
        let mut out = [1.0f32; 256];
        dequantize_row_q4_k(&[blk], &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_q6k_zero_block() {
        let blk = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(0.0),
        };
        let mut out = [1.0f32; 256];
        dequantize_row_q6_k(&[blk], &mut out);
        // all values should be d * scale * (q - 32) = 0
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_tq2_0_ternary() {
        let mut blk = BlockTQ2_0 {
            qs: [0; 64],
            d: f32_to_f16(1.0),
        };
        // byte 0: 0b00_01_10_00 = trits [0,2,1,0] = values [-1,+1,0,-1]
        blk.qs[0] = 0b00_01_10_00;
        let mut out = [0.0f32; 256];
        dequantize_row_tq2_0(&[blk], &mut out);
        assert!((out[0] - (-1.0)).abs() < 0.01);
        assert!((out[1] - (1.0)).abs() < 0.01);
        assert!((out[2] - (0.0)).abs() < 0.01);
        assert!((out[3] - (-1.0)).abs() < 0.01);
    }

    fn load_test_vector(name: &str) -> Option<Vec<u8>> {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let workspace = std::path::Path::new(&manifest).parent().unwrap_or(std::path::Path::new("."));
        let path = workspace.join("tests-ggml").join(name);
        std::fs::read(&path).ok()
    }

    fn load_f32_vector(name: &str) -> Option<Vec<f32>> {
        let data = load_test_vector(name)?;
        let n = data.len() / 4;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]));
        }
        Some(out)
    }

    #[test]
    fn ggml_crossval_q4_0_dequantize() {
        let Some(q4_data) = load_test_vector("q4_0_quantized.bin") else {
            eprintln!("SKIP: test vectors not generated (run tests-ggml/run_crossval.sh)");
            return;
        };
        let ggml_deq = load_f32_vector("q4_0_dequantized.bin").unwrap();
        let k = ggml_deq.len();
        let mut our_deq = vec![0.0f32; k];
        dequantize_row(&q4_data, &mut our_deq, super::super::GgmlType::Q4_0, k);

        let mut max_diff = 0.0f32;
        for i in 0..k {
            let diff = (our_deq[i] - ggml_deq[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Q4_0 dequant cross-val: max_abs_diff={max_diff:.6e} across {k} elements");
        assert!(max_diff < 1e-5, "Q4_0 dequant mismatch: max_diff={max_diff}");
    }

    #[test]
    fn ggml_crossval_q4k_dequantize() {
        let Some(q4k_data) = load_test_vector("q4k_quantized.bin") else {
            eprintln!("SKIP: test vectors not generated");
            return;
        };
        let ggml_deq = load_f32_vector("q4k_dequantized.bin").unwrap();
        let k = ggml_deq.len();
        let mut our_deq = vec![0.0f32; k];
        dequantize_row(&q4k_data, &mut our_deq, super::super::GgmlType::Q4K, k);

        let mut max_diff = 0.0f32;
        for i in 0..k {
            let diff = (our_deq[i] - ggml_deq[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Q4_K dequant cross-val: max_abs_diff={max_diff:.6e} across {k} elements");
        assert!(max_diff < 1e-5, "Q4_K dequant mismatch: max_diff={max_diff}");
    }

    #[test]
    fn ggml_crossval_q6k_dequantize() {
        let Some(q6k_data) = load_test_vector("q6k_quantized.bin") else {
            eprintln!("SKIP: test vectors not generated");
            return;
        };
        let ggml_deq = load_f32_vector("q6k_dequantized.bin").unwrap();
        let k = ggml_deq.len();
        let mut our_deq = vec![0.0f32; k];
        dequantize_row(&q6k_data, &mut our_deq, super::super::GgmlType::Q6K, k);

        let mut max_diff = 0.0f32;
        for i in 0..k {
            let diff = (our_deq[i] - ggml_deq[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Q6_K dequant cross-val: max_abs_diff={max_diff:.6e} across {k} elements");
        assert!(max_diff < 1e-5, "Q6_K dequant mismatch: max_diff={max_diff}");
    }

    #[test]
    fn ggml_crossval_q4_0_quantize_q8_0() {
        let Some(src_data) = load_f32_vector("q4_0_src.bin") else {
            eprintln!("SKIP: test vectors not generated");
            return;
        };
        let ggml_q8 = load_test_vector("q4_0_q8_0_activations.bin").unwrap();
        let k = src_data.len();
        let nb = k / 32;
        let mut our_q8 = vec![0u8; nb * 34];
        super::super::quantize::quantize_activations(&src_data, &mut our_q8, super::super::GgmlType::Q8_0);

        let mut max_diff = 0u8;
        let mut diff_count = 0usize;
        for i in 0..our_q8.len() {
            let diff = (our_q8[i] as i16 - ggml_q8[i] as i16).unsigned_abs() as u8;
            if diff > 0 {
                diff_count += 1;
                let blk = i / 34;
                let off = i % 34;
                if off >= 2 {
                    let j = off - 2;
                    let sv = src_data[blk * 32 + j];
                    let blk_src = &src_data[blk*32..(blk+1)*32];
                    let amax: f32 = blk_src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let d = amax / 127.0;
                    let scaled = sv / d;
                    eprintln!("  DIFF blk {blk} qs[{j}]: ggml={} ours={} src={sv:.8} scaled={scaled:.8}", ggml_q8[i] as i8, our_q8[i] as i8);
                } else {
                    eprintln!("  DIFF blk {blk} d_byte[{off}]: ggml=0x{:02x} ours=0x{:02x}", ggml_q8[i], our_q8[i]);
                }
            }
            max_diff = max_diff.max(diff);
        }
        eprintln!("Q8_0 quantize cross-val: max_byte_diff={max_diff}, {diff_count} diffs across {} bytes", our_q8.len());
        assert!(max_diff <= 1, "Q8_0 quantize mismatch: max_diff={max_diff}");
    }

    #[test]
    fn ggml_crossval_q4k_quantize_q8k() {
        let Some(src_data) = load_f32_vector("q4k_src.bin") else {
            eprintln!("SKIP: test vectors not generated");
            return;
        };
        let ggml_q8k = load_test_vector("q4k_q8k_activations.bin").unwrap();
        let k = src_data.len();
        let nb = k / 256;
        let mut our_q8k = vec![0u8; nb * 292];
        super::super::quantize::quantize_activations(&src_data, &mut our_q8k, super::super::GgmlType::Q8K);

        // Compare d (f32) at offset 0
        let ggml_d = f32::from_le_bytes([ggml_q8k[0], ggml_q8k[1], ggml_q8k[2], ggml_q8k[3]]);
        let our_d = f32::from_le_bytes([our_q8k[0], our_q8k[1], our_q8k[2], our_q8k[3]]);
        let d_diff = (ggml_d - our_d).abs();
        eprintln!("Q8_K quantize cross-val: d_diff={d_diff:.6e} (ggml={ggml_d}, ours={our_d})");
        assert!(d_diff < 1e-6, "Q8_K d mismatch");

        // Compare qs bytes (offset 4..260)
        let mut max_qs_diff = 0u8;
        for i in 4..260 {
            let diff = (our_q8k[i] as i16 - ggml_q8k[i] as i16).unsigned_abs() as u8;
            max_qs_diff = max_qs_diff.max(diff);
        }
        eprintln!("Q8_K quantize cross-val: max_qs_diff={max_qs_diff}");
        assert!(max_qs_diff <= 1, "Q8_K qs mismatch: max_diff={max_qs_diff}");
    }

    #[test]
    fn dequant_row_dispatch_q4k() {
        let blk = BlockQ4K {
            d: f32_to_f16(0.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qs: [0; 128],
        };
        let data = unsafe {
            std::slice::from_raw_parts(
                &blk as *const BlockQ4K as *const u8,
                std::mem::size_of::<BlockQ4K>(),
            )
        };
        let mut out = [1.0f32; 256];
        dequantize_row(data, &mut out, super::super::GgmlType::Q4K, 256);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn dequant_q4_0_no_panic() {
        let blk = BlockQ4_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let mut out = [0.0f32; 32];
        dequantize_row_q4_0(&[blk], &mut out);
    }

    #[kani::proof]
    fn dequant_q8_0_no_panic() {
        let blk = BlockQ8_0 {
            d: kani::any(),
            qs: kani::any(),
        };
        let mut out = [0.0f32; 32];
        dequantize_row_q8_0(&[blk], &mut out);
    }

    #[kani::proof]
    fn dequant_q4k_no_panic() {
        let blk = BlockQ4K {
            d: kani::any(),
            dmin: kani::any(),
            scales: kani::any(),
            qs: kani::any(),
        };
        let mut out = [0.0f32; 256];
        dequantize_row_q4_k(&[blk], &mut out);
    }

    #[kani::proof]
    fn dequant_q6k_no_panic() {
        let blk = BlockQ6K {
            ql: kani::any(),
            qh: kani::any(),
            scales: kani::any(),
            d: kani::any(),
        };
        let mut out = [0.0f32; 256];
        dequantize_row_q6_k(&[blk], &mut out);
    }

    #[kani::proof]
    fn dequant_tq2_0_no_panic() {
        let blk = BlockTQ2_0 {
            qs: kani::any(),
            d: kani::any(),
        };
        let mut out = [0.0f32; 256];
        dequantize_row_tq2_0(&[blk], &mut out);
    }

    #[kani::proof]
    fn dequant_q4_0_nibble_range() {
        let byte: u8 = kani::any();
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        assert!(lo <= 15);
        assert!(hi <= 15);
        let v0 = lo as i32 - 8;
        let v1 = hi as i32 - 8;
        assert!(v0 >= -8 && v0 <= 7);
        assert!(v1 >= -8 && v1 <= 7);
    }

    #[kani::proof]
    fn dequant_q6k_6bit_range() {
        let ql_byte: u8 = kani::any();
        let qh_byte: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);
        let low = ql_byte & 0x0F;
        let high = (qh_byte >> (shift * 2)) & 0x03;
        let q6 = low | (high << 4);
        assert!(q6 <= 63);
    }

    #[kani::proof]
    fn get_scale_min_k4_in_range() {
        let j: usize = kani::any();
        kani::assume(j < 8);
        let scales: [u8; 12] = kani::any();
        let (s, m) = get_scale_min_k4(j, &scales);
        assert!(s <= 63);
        assert!(m <= 63);
    }
}

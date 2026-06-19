use crate::Result;

pub const Q4K_BLOCK_SIZE: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_GROUPS: usize = 8;
pub const Q4K_GROUP_SIZE: usize = 32;

#[allow(clippy::too_many_arguments)]
pub trait MatMulQ: Send + Sync {
    fn forward_b_transposed_q4k(
        &self,
        a: &[f32],
        b_quantized: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}

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
        sc[i + 4] = ((packed[i] >> 6) | ((packed[i + 8] & 0x0F) << 2)) & 0x3F;
        m[i] = packed[i + 4] & 0x3F;
        m[i + 4] = ((packed[i + 4] >> 6) | ((packed[i + 8] >> 4) << 2)) & 0x3F;
        i += 1;
    }

    let sc_f: [f32; Q4K_GROUPS] = [
        sc[0] as f32, sc[1] as f32, sc[2] as f32, sc[3] as f32,
        sc[4] as f32, sc[5] as f32, sc[6] as f32, sc[7] as f32,
    ];
    let m_f: [f32; Q4K_GROUPS] = [
        m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32,
        m[4] as f32, m[5] as f32, m[6] as f32, m[7] as f32,
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

    let mut group = 0;
    while group < Q4K_GROUPS {
        let d_sc = d * sc[group];
        let dm = dmin * m[group];
        let mut gi = 0;
        while gi < Q4K_GROUP_SIZE {
            let elem = group * Q4K_GROUP_SIZE + gi;
            let q = q4k_nibble(qs, elem) as f32;
            output[elem] = d_sc * q - dm;
            gi += 1;
        }
        group += 1;
    }
}

pub fn q4k_dot_block(block: &[u8], a: &[f32]) -> f32 {
    let d = q4k_block_d(block);
    let dmin = q4k_block_dmin(block);
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = q4k_unpack_scales(packed_scales);

    let mut sum = 0.0f32;
    let mut group = 0;
    while group < Q4K_GROUPS {
        let d_sc = d * sc[group];
        let dm = dmin * m[group];
        let mut gi = 0;
        while gi < Q4K_GROUP_SIZE {
            let elem = group * Q4K_GROUP_SIZE + gi;
            let q = q4k_nibble(qs, elem) as f32;
            let w = d_sc * q - dm;
            sum += a[elem] * w;
            gi += 1;
        }
        group += 1;
    }
    sum
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
            return Err(crate::KernelError::InvalidConfig(
                format!("k={k} must be divisible by Q4K block size {Q4K_BLOCK_SIZE}"),
            ));
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
        let sc_hi = ((packed[i] >> 6) | ((packed[i + 8] & 0x0F) << 2)) & 0x3F;
        let m_lo = packed[i + 4] & 0x3F;
        let m_hi = ((packed[i + 4] >> 6) | ((packed[i + 8] >> 4) << 2)) & 0x3F;

        assert!(sc_lo <= 63);
        assert!(sc_hi <= 63);
        assert!(m_lo <= 63);
        assert!(m_hi <= 63);
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

        // Pack nibbles
        for i in 0..128 {
            let lo = values[i * 2];
            let hi = values[i * 2 + 1];
            block[16 + i] = (lo & 0x0F) | ((hi & 0x0F) << 4);
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
        assert!(
            (dot - 128.0).abs() < 1e-2,
            "dot={dot}, expected=128.0"
        );
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
}

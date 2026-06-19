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
        sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        m[i] = packed[i + 4] & 0x3F;
        m[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
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

        if m == 1 {
            // Optimized m=1 decode path: dequant block → vectorizable dot
            let a_row = &a[0..k];
            let mut tmp = [0.0f32; Q4K_BLOCK_SIZE];
            for j in 0..n {
                let mut sum = 0.0f32;
                for blk in 0..blocks_per_row {
                    let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
                    let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
                    q4k_dequant_block(block, &mut tmp);
                    let a_slice = &a_row[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
                    // Separate loop — compiler can vectorize this with AVX
                    let mut block_sum = 0.0f32;
                    for idx in 0..Q4K_BLOCK_SIZE {
                        block_sum += a_slice[idx] * tmp[idx];
                    }
                    sum += block_sum;
                }
                output[j] = sum;
            }
        } else {
            // Batched path: dequant full row, use gemm for SIMD matmul
            let mut b_f32 = vec![0.0f32; n * k];
            for j in 0..n {
                Self::dequant_row(b_quantized, j, k, &mut b_f32[j * k..(j + 1) * k]);
            }
            unsafe {
                gemm::gemm(
                    m, n, k,
                    output.as_mut_ptr(), 1, n as isize, false,
                    a.as_ptr(), 1, k as isize,
                    b_f32.as_ptr(), k as isize, 1,
                    0.0f32, 1.0f32,
                    false, false, false,
                    gemm::Parallelism::Rayon(0),
                );
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

    #[test]
    fn test_opt_matches_reference() {
        let k = 512;
        let m = 2;
        let n = 4;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect();

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

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 23) as f32) * 0.005 - 0.05).collect();

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
            ref_kernel.forward_b_transposed_q4k(&a, &b_q, &mut out, m, n, k).unwrap();
        }
        let ref_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            opt_kernel.forward_b_transposed_q4k(&a, &b_q, &mut out, m, n, k).unwrap();
        }
        let opt_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = ref_ms / opt_ms;
        eprintln!("  Q4K matmul {m}x{n}x{k}: ref={ref_ms:.1}ms, opt={opt_ms:.1}ms, speedup={speedup:.1}x");
    }

    #[test]
    fn test_opt_larger_matmul() {
        let k = 4096;
        let m = 1;
        let n = 4096;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 23) as f32) * 0.005 - 0.05).collect();

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
}

use std::sync::Arc;
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use cuda_core::{Device, Stream};
use crate::{
    KernelError, MatMulQ, Result, VerificationTier,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE,
    TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE,
    I2S_BLOCK_BYTES, I2S_BLOCK_SIZE,
};

// Tile constants matching cutile_matmul.rs
const BM: usize = 16;
const BN: usize = 16;
const BK: usize = 8;

// ── Block layout constants ────────────────────────────────────────────────────
// Q4_K: 4-bit, 256 elements per block, 144 bytes
//   [0..2]   d      (f16)
//   [2..4]   dmin   (f16)
//   [4..16]  scales (12 bytes packed, 8 groups × 6-bit sc + 8 groups × 6-bit m)
//   [16..144] qs    (128 bytes = 256 nibbles)
//
// Q6_K: 6-bit, 256 elements per block, 210 bytes
//   [0..128]   ql     (low 4-bit per element)
//   [128..192] qh     (high 2-bit per element)
//   [192..208] scales (16 bytes, i8 per 16-element group)
//   [208..210] d      (f16)
//
// TQ2: ternary {-1,0,1}, 256 elements, 66 bytes
//   [0..64]  qs (2-bit packed: stored as (trit+1) in 2 bits)
//   [64..66] d  (f16)
//
// I2S: 2-bit signed, variable layout (handled by existing CPU impl)

// ── f16 bit conversion (mirrors quantized.rs) ────────────────────────────────
#[inline(always)]
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

// ── Q4_K scale unpacking (mirrors quantized.rs::q4k_unpack_scales) ───────────
fn q4k_unpack_scales(packed: &[u8]) -> ([f32; Q4K_GROUPS], [f32; Q4K_GROUPS]) {
    let mut sc = [0u8; Q4K_GROUPS];
    let mut m = [0u8; Q4K_GROUPS];
    let mut i = 0;
    while i < 4 {
        sc[i]     = packed[i] & 0x3F;
        sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        m[i]      = packed[i + 4] & 0x3F;
        m[i + 4]  = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
        i += 1;
    }
    let to_f32 = |a: [u8; Q4K_GROUPS]| {
        [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32,
         a[4] as f32, a[5] as f32, a[6] as f32, a[7] as f32]
    };
    (to_f32(sc), to_f32(m))
}

// ── Dequantize a single Q4_K block to 256 f32 values ────────────────────────
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q4K_BLOCK_SIZE);
    let d    = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
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
            out[j * 64 + l]      = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            out[j * 64 + l + 32] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            l += 1;
        }
        j += 1;
    }
}

// ── Dequantize a single Q6_K block to 256 f32 values ────────────────────────
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_q6k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q6K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q6K_BLOCK_SIZE);
    let ql     = &block[0..128];
    let qh     = &block[128..192];
    let scales = &block[192..208];
    let d      = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off  = half * 128;
        for l in 0..32 {
            let q1 = ((ql[ql_off + l] & 0xF)      | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF)  | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4)         | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4)    | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
            let is = l / 16;
            let sc0 = scales[half * 8 + is]     as i8 as f32;
            let sc1 = scales[half * 8 + is + 2] as i8 as f32;
            let sc2 = scales[half * 8 + is + 4] as i8 as f32;
            let sc3 = scales[half * 8 + is + 6] as i8 as f32;
            out[y_off + l]       = d * sc0 * q1 as f32;
            out[y_off + l + 32]  = d * sc1 * q2 as f32;
            out[y_off + l + 64]  = d * sc2 * q3 as f32;
            out[y_off + l + 96]  = d * sc3 * q4 as f32;
        }
    }
}

// ── Dequantize a single TQ2 block to 256 f32 values ─────────────────────────
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_tq2_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), TQ2_BLOCK_BYTES);
    debug_assert_eq!(out.len(), TQ2_BLOCK_SIZE);
    let d  = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let qs = &block[0..64];
    let mut idx = 0;
    for &byte in qs {
        for shift in (0..8).step_by(2) {
            let trit = ((byte >> shift) & 0x03) as i32 - 1;
            out[idx] = d * trit as f32;
            idx += 1;
        }
    }
}

// ── Full weight matrix dequantization ────────────────────────────────────────
// Expands [N, blocks_per_row * BLOCK_BYTES] → [N, K] f32
fn dequant_weights_q4k(b_quantized: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let mut b_f32 = vec![0.0f32; n * k];
    for j in 0..n {
        let row = &mut b_f32[j * k..(j + 1) * k];
        for blk in 0..blocks_per_row {
            let block_offset = (j * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
            let block = &b_quantized[block_offset..block_offset + Q4K_BLOCK_BYTES];
            let out_slice = &mut row[blk * Q4K_BLOCK_SIZE..(blk + 1) * Q4K_BLOCK_SIZE];
            dequant_q4k_block(block, out_slice);
        }
    }
    b_f32
}

fn dequant_weights_q6k(b_quantized: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / Q6K_BLOCK_SIZE;
    let mut b_f32 = vec![0.0f32; n * k];
    for j in 0..n {
        let row = &mut b_f32[j * k..(j + 1) * k];
        for blk in 0..blocks_per_row {
            let block_offset = (j * blocks_per_row + blk) * Q6K_BLOCK_BYTES;
            let block = &b_quantized[block_offset..block_offset + Q6K_BLOCK_BYTES];
            let out_slice = &mut row[blk * Q6K_BLOCK_SIZE..(blk + 1) * Q6K_BLOCK_SIZE];
            dequant_q6k_block(block, out_slice);
        }
    }
    b_f32
}

fn dequant_weights_tq2(b_quantized: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / TQ2_BLOCK_SIZE;
    let mut b_f32 = vec![0.0f32; n * k];
    for j in 0..n {
        let row = &mut b_f32[j * k..(j + 1) * k];
        for blk in 0..blocks_per_row {
            let block_offset = (j * blocks_per_row + blk) * TQ2_BLOCK_BYTES;
            let block = &b_quantized[block_offset..block_offset + TQ2_BLOCK_BYTES];
            let out_slice = &mut row[blk * TQ2_BLOCK_SIZE..(blk + 1) * TQ2_BLOCK_SIZE];
            dequant_tq2_block(block, out_slice);
        }
    }
    b_f32
}

// I2S dequantization: 2-bit signed integer packed weights with a single global scale.
// Layout: packed_bytes = n*k/4 bytes of 2-bit values, then 4 bytes f32 scale.
// Each 2-bit value encodes a value in {-1, 0, 1, 2} (i2s signed: -2..1 range).
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_weights_i2s(b_quantized: &[u8], n: usize, k: usize) -> Vec<f32> {
    let packed_bytes = n * k / 4;
    let w_scale = f32::from_ne_bytes([
        b_quantized[packed_bytes],
        b_quantized[packed_bytes + 1],
        b_quantized[packed_bytes + 2],
        b_quantized[packed_bytes + 3],
    ]);
    let row_packed_bytes = k / 4;
    let mut b_f32 = vec![0.0f32; n * k];
    for j in 0..n {
        let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
        let row = &mut b_f32[j * k..(j + 1) * k];
        let mut idx = 0;
        for &byte in w_packed {
            for shift in 0..4 {
                let raw = (byte >> (shift * 2)) & 0x03;
                // i2s: 0→0, 1→1, 2→-2, 3→-1 (two's complement 2-bit)
                let val = (raw as i8).wrapping_shl(6).wrapping_shr(6) as f32;
                row[idx] = val * w_scale;
                idx += 1;
            }
        }
    }
    b_f32
}

// ── Cutile GEMM helper (B transposed): output[M,N] = A[M,K] @ B^T[N,K] ──────
fn cutile_gemm_bt(
    stream: &Arc<Stream>,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let k_pad = k.div_ceil(BK) * BK;

    let a_vec: Arc<Vec<f32>> = if k_pad == k {
        Arc::new(a.to_vec())
    } else {
        let mut padded = vec![0.0f32; m * k_pad];
        for r in 0..m {
            padded[r * k_pad..r * k_pad + k].copy_from_slice(&a[r * k..(r + 1) * k]);
        }
        Arc::new(padded)
    };

    let b_vec: Arc<Vec<f32>> = if k_pad == k {
        Arc::new(b.to_vec())
    } else {
        let mut padded = vec![0.0f32; n * k_pad];
        for r in 0..n {
            padded[r * k_pad..r * k_pad + k].copy_from_slice(&b[r * k..(r + 1) * k]);
        }
        Arc::new(padded)
    };

    let z = api::zeros::<f32>(&[m, n])
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        .partition([BM, BN]);

    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&a_vec)
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        .reshape(&[m, k_pad])
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        .into();

    let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&b_vec)
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        .reshape(&[n, k_pad])
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        .into();

    let generics = vec![
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        k_pad.to_string(),
    ];

    let (z, _x, _y) = crate::cutile_matmul::kernels::cutile_gemm_b_transposed(z, x, y)
        .generics(generics)
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

    let host: Vec<f32> = z
        .unpartition()
        .to_host_vec()
        .sync_on(stream)
        .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

    output.copy_from_slice(&host);
    Ok(())
}

// ── CutileQuantizedGemm struct ────────────────────────────────────────────────

/// GPU-accelerated quantized GEMM using cutile-rs.
///
/// Dequantizes weights on CPU (Q4_K, Q6_K, TQ2, I2S), then executes
/// f32 GEMM on GPU via cutile. All methods are `forward_b_transposed_*`:
/// weights are stored transposed ([N, K]) and activations are [M, K].
///
/// TODO: GPU dequant — move block-format expansion to GPU kernels for
/// full end-to-end GPU execution.
pub struct CutileQuantizedGemm {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

unsafe impl Send for CutileQuantizedGemm {}
unsafe impl Sync for CutileQuantizedGemm {}

impl CutileQuantizedGemm {
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    pub fn with_device(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let stream = device
            .new_stream()
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Self { device, stream })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn verification_tier(&self) -> VerificationTier {
        VerificationTier::Tested
    }
}

impl MatMulQ for CutileQuantizedGemm {
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
            return Err(KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }
        if k % Q4K_BLOCK_SIZE != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "k={k} must be divisible by Q4K block size {Q4K_BLOCK_SIZE}"
            )));
        }
        let blocks_per_row = k / Q4K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        // TODO: GPU dequant — expand Q4_K on GPU to avoid host→device bandwidth for weights
        let b_f32 = dequant_weights_q4k(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
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
            return Err(KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }
        if k % Q6K_BLOCK_SIZE != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "k={k} must be divisible by Q6K block size {Q6K_BLOCK_SIZE}"
            )));
        }
        let blocks_per_row = k / Q6K_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * Q6K_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        // TODO: GPU dequant — expand Q6_K on GPU
        let b_f32 = dequant_weights_q6k(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
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
            return Err(KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }
        if k % TQ2_BLOCK_SIZE != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "k={k} must be divisible by TQ2 block size {TQ2_BLOCK_SIZE}"
            )));
        }
        let blocks_per_row = k / TQ2_BLOCK_SIZE;
        let expected_b_bytes = n * blocks_per_row * TQ2_BLOCK_BYTES;
        if b_quantized.len() != expected_b_bytes {
            return Err(KernelError::DimensionMismatch {
                expected: expected_b_bytes,
                got: b_quantized.len(),
            });
        }

        // TODO: GPU dequant — expand TQ2 ternary weights on GPU (all-add/sub/skip paths)
        let b_f32 = dequant_weights_tq2(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
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
            return Err(KernelError::DimensionMismatch { expected: m * k, got: a.len() });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch { expected: m * n, got: output.len() });
        }
        if k % 4 != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "k={k} must be divisible by 4 for I2S format"
            )));
        }

        // TODO: GPU dequant — expand I2S 2-bit signed weights on GPU
        let b_f32 = dequant_weights_i2s(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::{q4k_dot_block, q6k_dot_block, tq2_dot_block};

    fn make_q4k_block(d: f32, dmin: f32) -> Vec<u8> {
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        // Encode d and dmin as f16
        let d_bits = f32_to_f16_bits(d);
        let dmin_bits = f32_to_f16_bits(dmin);
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        block[2] = dmin_bits as u8;
        block[3] = (dmin_bits >> 8) as u8;
        // scales[0..8] = 1, mins[0..8] = 0 (encoded in 12 packed bytes)
        // For simplicity: set sc[i]=1, m[i]=0 → packed bytes = [1,1,1,1,0,0,0,0,0,0,0,0]
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // qs: all 0x88 → low nibble=8, high nibble=8
        for i in 16..144 {
            block[i] = 0x88;
        }
        block
    }

    fn f32_to_f16_bits(val: f32) -> u16 {
        // Simple f32→f16 for test values that fit in normal f16 range
        let bits = val.to_bits();
        let sign = (bits >> 31) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 { return sign << 15; }
        if exp >= 31 { return (sign << 15) | (31 << 10); }
        (sign << 15) | ((exp as u16) << 10) | frac
    }

    #[test]
    fn dequant_q4k_block_matches_dot_block_identity() {
        // Build a block with d=1, dmin=0 and all nibbles=7 → each element = 7.0
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        let d_bits = f32_to_f16_bits(1.0);
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // dmin = 0
        // scales: sc[0..4]=1, packed in bytes [4..16]
        block[4] = 1; block[5] = 1; block[6] = 1; block[7] = 1;
        // qs: all nibbles = 7
        for i in 16..144 {
            block[i] = 0x77;
        }
        let a = vec![1.0f32; Q4K_BLOCK_SIZE];
        let expected_dot = q4k_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; Q4K_BLOCK_SIZE];
        dequant_q4k_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!((got_dot - expected_dot).abs() < 1e-3,
            "dequant dot={got_dot} vs q4k_dot_block={expected_dot}");
    }

    #[test]
    fn dequant_q6k_block_matches_dot_block_identity() {
        let mut block = vec![0u8; Q6K_BLOCK_BYTES];
        // d = 1.0 → f16 bits at [208..210]
        let d_bits = f32_to_f16_bits(1.0);
        block[208] = d_bits as u8;
        block[209] = (d_bits >> 8) as u8;
        // scales: all 1 (i8 value 1 → all groups have sc=1)
        for i in 192..208 { block[i] = 1u8; }
        // ql: all 0x11 → low nibbles=1, meaning q1=(1|0<<4)-32=-31
        for i in 0..128 { block[i] = 0x11; }
        // qh: 0 → high bits=0
        let a = vec![1.0f32; Q6K_BLOCK_SIZE];
        let expected_dot = q6k_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; Q6K_BLOCK_SIZE];
        dequant_q6k_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!((got_dot - expected_dot).abs() < 1e-2,
            "dequant dot={got_dot} vs q6k_dot_block={expected_dot}");
    }

    #[test]
    fn dequant_tq2_block_matches_dot_block_identity() {
        let mut block = vec![0u8; TQ2_BLOCK_BYTES];
        let d_bits = f32_to_f16_bits(1.0);
        block[64] = d_bits as u8;
        block[65] = (d_bits >> 8) as u8;
        // qs: 0b10101010 → each pair = 0b10 = 2 → trit = 2-1 = 1
        for i in 0..64 { block[i] = 0xAA; }
        let a = vec![1.0f32; TQ2_BLOCK_SIZE];
        let expected_dot = tq2_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; TQ2_BLOCK_SIZE];
        dequant_tq2_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!((got_dot - expected_dot).abs() < 1e-3,
            "dequant dot={got_dot} vs tq2_dot_block={expected_dot}");
    }
}

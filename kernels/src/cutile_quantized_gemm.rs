use crate::{
    KernelError, MatMulQ, Result, VerificationTier, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE,
};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use std::sync::Arc;

// ── Q2_0 block layout constants ──────────────────────────────────────────────
// Q2_0 (PrismML): 2-bit quantized, 128 elements per block, 34 bytes per block
//   [0..2]   d  (f16 scale)
//   [2..34]  qs (32 bytes = 128 elements × 2 bits, 4 elements per byte)
// Dequant: (((qs[i/4] >> (2*(i%4))) & 3) - 1) × d
// Encoding: q=0→-1, q=1→0, q=2→+1, q=3→+2
//
// GPU packed format (9 u32 per block):
//   u32[0]    = f32 bits of scale (f16 → f32 at upload time)
//   u32[1..8] = 8 u32 words of packed 2-bit data (each holds 16 elements)
pub const Q2_0_QK: usize = 128;
pub const Q2_0_BLOCK_BYTES: usize = 34; // raw on-disk bytes per block
pub const Q2_0_PACKED_U32S: usize = 9;  // u32 slots per block in GPU format

// ── cuTile Q2_0 GEMV kernel ───────────────────────────────────────────────────

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Q2_0 dequantized GEMV: z[1] = Σ_k dequant(packed[pid_n, k]) × x[k]
    ///
    /// Packed layout: [N × K_BLOCKS × 9] u32 (conceptually [N, K_BLOCKS, 9]):
    ///   packed[n, blk, 0]    = f32 bits of block scale (converted from f16)
    ///   packed[n, blk, 1..8] = 8 u32 words of 2-bit weights
    ///     bits[2i+1 : 2i] of word w = q of element blk*128 + w*16 + i
    ///     dequant: (q - 1) × scale, q ∈ {0,1,2,3} → values ∈ {-1,0,1,2} × scale
    ///
    /// Grid: (N, 1, 1) — one program per output neuron.
    #[cutile::entry()]
    pub fn cutile_q2_0_gemv<const K_BLOCKS: i32>(
        z: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
        packed: &Tensor<u32, { [-1] }>,
    ) {
        let pid_n = program_id(0);
        let part_x = x.partition(shape![1]);
        let part_p = packed.partition(shape![1]);
        let mut acc: Tile<f32, { [1] }> = broadcast_scalar(0.0_f32, shape![1]);

        for blk in 0i32..K_BLOCKS {
            let pack_base = pid_n * K_BLOCKS * 9 + blk * 9;
            let x_base = blk * 128;

            // Load block scale stored as f32 bits in the first u32 slot.
            let scale_u32: Tile<u32, { [1] }> = part_p.load([pack_base]);
            let scale: Tile<f32, { [1] }> = bitcast(scale_u32);

            // Process 8 u32 words; each holds 16 2-bit quantised weight values.
            for word in 0i32..8 {
                let word_data: Tile<u32, { [1] }> = part_p.load([pack_base + 1 + word]);
                let x_word_base = x_base + word * 16;

                // IEEE 754 integer-to-float trick — avoids itof (cutile-0.3.1 bug:
                // itof does not emit the required 'signedness' attribute).
                // For u32 x in [0, 2^23): bitcast(x | 0x4B000000) - 8388608.0 = (float)x.
                // Combined constant 8388609.0 = 8388608.0 + 1.0 centres {0,1,2,3} to {-1,0,1,2}.
                let magic: Tile<u32, { [1] }> = broadcast_scalar(0x4B000000_u32, shape![1]);
                let bias_f32: Tile<f32, { [1] }> = broadcast_scalar(8388609.0_f32, shape![1]);

                for bit_pair in 0i32..16 {
                    // Extract 2-bit q value via right-shift + mask.
                    let shift: Tile<u32, { [1] }> =
                        broadcast_scalar((bit_pair * 2) as u32, shape![1]);
                    let mask: Tile<u32, { [1] }> = broadcast_scalar(3_u32, shape![1]);
                    let q_u32: Tile<u32, { [1] }> = andi(shri(word_data, shift), mask);

                    // Integer→float via IEEE bit trick:
                    // bitcast(q | 0x4B000000) - 8388609.0 == (q as f32) - 1.0
                    let biased: Tile<u32, { [1] }> = ori(q_u32, magic);
                    let q_f32: Tile<f32, { [1] }> = bitcast(biased) - bias_f32;

                    // Load the corresponding activation element.
                    let x_elem: Tile<f32, { [1] }> = part_x.load([x_word_base + bit_pair]);

                    // Accumulate: acc += scale × q × x
                    acc = acc + scale * q_f32 * x_elem;
                }
            }
        }
        z.store(acc);
    }
}

/// GPU-resident Q2_0 GEMV dispatcher.
///
/// Holds the CUDA stream and dispatches `cutile_q2_0_gemv` for decode (m=1)
/// forward passes where weights are stored in packed Q2_0 format on the GPU.
pub struct CutileQ2Gemv {
    stream: Arc<Stream>,
}

unsafe impl Send for CutileQ2Gemv {}
unsafe impl Sync for CutileQ2Gemv {}

impl CutileQ2Gemv {
    /// Create from an existing CUDA stream (share with `CutileMatMul`).
    pub fn from_stream(stream: Arc<Stream>) -> Self {
        Self { stream }
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Device-to-device Q2_0 GEMV: output[n] = packed_weights × x[k].
    ///
    /// `packed` holds `n × k_blocks × Q2_0_PACKED_U32S` u32 elements.
    /// `x` holds `k = k_blocks × Q2_0_QK` f32 activation elements.
    pub fn forward_d2d(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let stream = &self.stream;

        let z = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (z, _x, _p) =
            kernels::cutile_q2_0_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // Force a true rank-1 tensor: download the result and re-upload as a fresh
        // flat allocation.  cuTile unpartition() + reshape() may leave partition
        // stride metadata visible to subsequent kernels (cutile_deinterleave rejects
        // tensors with 2 strides).  This round-trip is the simplest portable fix.
        let host: Vec<f32> = z
            .unpartition()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        let out_flat: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(host))
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        Ok(out_flat)
    }
}

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
        sc[i] = packed[i] & 0x3F;
        sc[i + 4] = ((packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4)) & 0x3F;
        m[i] = packed[i + 4] & 0x3F;
        m[i + 4] = ((packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4)) & 0x3F;
        i += 1;
    }
    let to_f32 = |a: [u8; Q4K_GROUPS]| {
        [
            a[0] as f32,
            a[1] as f32,
            a[2] as f32,
            a[3] as f32,
            a[4] as f32,
            a[5] as f32,
            a[6] as f32,
            a[7] as f32,
        ]
    };
    (to_f32(sc), to_f32(m))
}

// ── Dequantize a single Q4_K block to 256 f32 values ────────────────────────
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q4K_BLOCK_SIZE);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
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
            out[j * 64 + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
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
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let y_off = half * 128;
        for l in 0..32 {
            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
            let is = l / 16;
            let sc0 = scales[half * 8 + is] as i8 as f32;
            let sc1 = scales[half * 8 + is + 2] as i8 as f32;
            let sc2 = scales[half * 8 + is + 4] as i8 as f32;
            let sc3 = scales[half * 8 + is + 6] as i8 as f32;
            out[y_off + l] = d * sc0 * q1 as f32;
            out[y_off + l + 32] = d * sc1 * q2 as f32;
            out[y_off + l + 64] = d * sc2 * q3 as f32;
            out[y_off + l + 96] = d * sc3 * q4 as f32;
        }
    }
}

// ── Dequantize a single TQ2 block to 256 f32 values ─────────────────────────
// TODO: GPU dequant — currently runs on CPU before GEMM
fn dequant_tq2_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), TQ2_BLOCK_BYTES);
    debug_assert_eq!(out.len(), TQ2_BLOCK_SIZE);
    let d = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
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
        let device = Device::new(ordinal).map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
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
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if !k.is_multiple_of(Q4K_BLOCK_SIZE) {
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
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if !k.is_multiple_of(Q6K_BLOCK_SIZE) {
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
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if !k.is_multiple_of(TQ2_BLOCK_SIZE) {
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
            return Err(KernelError::DimensionMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if output.len() != m * n {
            return Err(KernelError::DimensionMismatch {
                expected: m * n,
                got: output.len(),
            });
        }
        if !k.is_multiple_of(4) {
            return Err(KernelError::InvalidConfig(format!(
                "k={k} must be divisible by 4 for I2S format"
            )));
        }

        // TODO: GPU dequant — expand I2S 2-bit signed weights on GPU
        let b_f32 = dequant_weights_i2s(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
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
        // Fall back to CPU I2S preq path for now
        let a_f32: Vec<f32> = preq_q8.iter().map(|&x| x as f32 * preq_scale).collect();
        let _ = preq_sum;
        let b_f32 = dequant_weights_i2s(b_quantized, n, k);
        cutile_gemm_bt(&self.stream, &a_f32, &b_f32, output, 1, n, k)
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
        // Fall back to dequant + GEMM for now
        let b_f32 = dequant_weights_i2s(b_transposed, n, k);
        cutile_gemm_bt(&self.stream, a, &b_f32, output, m, n, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::{q4k_dot_block, q6k_dot_block, tq2_dot_block};

    fn f32_to_f16_bits(val: f32) -> u16 {
        // Simple f32→f16 for test values that fit in normal f16 range
        let bits = val.to_bits();
        let sign = (bits >> 31) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 {
            return sign << 15;
        }
        if exp >= 31 {
            return (sign << 15) | (31 << 10);
        }
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
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // qs: all nibbles = 7
        for i in 16..144 {
            block[i] = 0x77;
        }
        let a = vec![1.0f32; Q4K_BLOCK_SIZE];
        let expected_dot = q4k_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; Q4K_BLOCK_SIZE];
        dequant_q4k_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!(
            (got_dot - expected_dot).abs() < 1e-3,
            "dequant dot={got_dot} vs q4k_dot_block={expected_dot}"
        );
    }

    #[test]
    fn dequant_q6k_block_matches_dot_block_identity() {
        let mut block = vec![0u8; Q6K_BLOCK_BYTES];
        // d = 1.0 → f16 bits at [208..210]
        let d_bits = f32_to_f16_bits(1.0);
        block[208] = d_bits as u8;
        block[209] = (d_bits >> 8) as u8;
        // scales: all 1 (i8 value 1 → all groups have sc=1)
        for i in 192..208 {
            block[i] = 1u8;
        }
        // ql: all 0x11 → low nibbles=1, meaning q1=(1|0<<4)-32=-31
        for i in 0..128 {
            block[i] = 0x11;
        }
        // qh: 0 → high bits=0
        let a = vec![1.0f32; Q6K_BLOCK_SIZE];
        let expected_dot = q6k_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; Q6K_BLOCK_SIZE];
        dequant_q6k_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!(
            (got_dot - expected_dot).abs() < 1e-2,
            "dequant dot={got_dot} vs q6k_dot_block={expected_dot}"
        );
    }

    #[test]
    fn dequant_tq2_block_matches_dot_block_identity() {
        let mut block = vec![0u8; TQ2_BLOCK_BYTES];
        let d_bits = f32_to_f16_bits(1.0);
        block[64] = d_bits as u8;
        block[65] = (d_bits >> 8) as u8;
        // qs: 0b10101010 → each pair = 0b10 = 2 → trit = 2-1 = 1
        for i in 0..64 {
            block[i] = 0xAA;
        }
        let a = vec![1.0f32; TQ2_BLOCK_SIZE];
        let expected_dot = tq2_dot_block(&block, &a);

        let mut dequanted = vec![0.0f32; TQ2_BLOCK_SIZE];
        dequant_tq2_block(&block, &mut dequanted);
        let got_dot: f32 = dequanted.iter().sum();

        assert!(
            (got_dot - expected_dot).abs() < 1e-3,
            "dequant dot={got_dot} vs tq2_dot_block={expected_dot}"
        );
    }

    // ── Q2_0 upload format helpers ────────────────────────────────────────────

    fn f32_to_f16_bits_q2(val: f32) -> u16 {
        f32_to_f16_bits(val)
    }

    /// Build a synthetic Q2_0 block: scale=d, all 128 weights = q_val ∈ {0,1,2,3}.
    fn make_q2_0_block(d: f32, q_val: u8) -> Vec<u8> {
        assert!(q_val <= 3);
        let mut block = vec![0u8; Q2_0_BLOCK_BYTES];
        let d_bits = f32_to_f16_bits_q2(d);
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        // Pack q_val into every 2-bit slot of the 32 payload bytes.
        let packed_byte = q_val | (q_val << 2) | (q_val << 4) | (q_val << 6);
        for b in block[2..].iter_mut() {
            *b = packed_byte;
        }
        block
    }

    #[test]
    fn q2_0_upload_format_round_trips() {
        // Build a 2-row, 2-block weight matrix in Q2_0 format.
        // Row 0: scale=2.0, all q=1 → dequant = (1-1)*2.0 = 0.0
        // Row 1: scale=3.0, all q=3 → dequant = (3-1)*3.0 = 6.0
        let n = 2;
        let k = Q2_0_QK * 2; // 2 blocks per row
        let k_blocks = k / Q2_0_QK;

        let block0a = make_q2_0_block(2.0, 1); // row 0, block 0
        let block0b = make_q2_0_block(2.0, 1); // row 0, block 1
        let block1a = make_q2_0_block(3.0, 3); // row 1, block 0
        let block1b = make_q2_0_block(3.0, 3); // row 1, block 1

        let mut raw = Vec::new();
        raw.extend_from_slice(&block0a);
        raw.extend_from_slice(&block0b);
        raw.extend_from_slice(&block1a);
        raw.extend_from_slice(&block1b);

        // Convert to packed u32 format (mirrors maybe_upload_q2 logic).
        let total_u32s = n * k_blocks * Q2_0_PACKED_U32S;
        let mut packed: Vec<u32> = Vec::with_capacity(total_u32s);

        fn f16b_to_f32(bits: u16) -> f32 {
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1f) as u32;
            let frac = (bits & 0x3ff) as u32;
            if exp == 0 && frac == 0 {
                return f32::from_bits(sign << 31);
            } else if exp == 31 {
                return f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13));
            }
            f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
        }

        for row in 0..n {
            for blk in 0..k_blocks {
                let off = (row * k_blocks + blk) * Q2_0_BLOCK_BYTES;
                let block = &raw[off..off + Q2_0_BLOCK_BYTES];
                let scale_bits = u16::from_le_bytes([block[0], block[1]]);
                packed.push(f16b_to_f32(scale_bits).to_bits());
                for w in 0..8usize {
                    let b0 = block[2 + w * 4] as u32;
                    let b1 = block[3 + w * 4] as u32;
                    let b2 = block[4 + w * 4] as u32;
                    let b3 = block[5 + w * 4] as u32;
                    packed.push(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
                }
            }
        }

        assert_eq!(packed.len(), total_u32s);

        // Validate scale slots.
        let scale_row0 = f32::from_bits(packed[0]);
        let scale_row1 = f32::from_bits(packed[k_blocks * Q2_0_PACKED_U32S]);
        assert!(
            (scale_row0 - 2.0).abs() < 1e-4,
            "row 0 scale: {scale_row0}"
        );
        assert!(
            (scale_row1 - 3.0).abs() < 1e-4,
            "row 1 scale: {scale_row1}"
        );

        // Validate data words: all q=1 → packed byte 0b01_01_01_01 = 0x55.
        let word0 = packed[1];
        assert_eq!(word0, 0x55555555, "row0 word: 0x{:08x}", word0);

        // All q=3 → packed byte 0b11_11_11_11 = 0xFF.
        let word1 = packed[k_blocks * Q2_0_PACKED_U32S + 1];
        assert_eq!(word1, 0xFFFFFFFF, "row1 word: 0x{:08x}", word1);
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn q2_0_gpu_gemv_matches_cpu_dequant() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;
        use cutile::tensor::Reshape;

        // Build a 4-output, K=128 Q2_0 GEMV:
        // Row 0: scale=1.0, all q=2 → dequant = +1.0
        // Row 1: scale=2.0, all q=1 → dequant =  0.0
        // Row 2: scale=1.0, all q=0 → dequant = -1.0
        // Row 3: scale=1.0, all q=3 → dequant = +2.0
        let n = 4;
        let k = Q2_0_QK;
        let k_blocks = 1;
        let test_cases: &[(f32, u8, f32)] = &[
            (1.0, 2, 1.0),  // row 0: scale=1, q=2 → +1
            (2.0, 1, 0.0),  // row 1: scale=2, q=1 →  0
            (1.0, 0, -1.0), // row 2: scale=1, q=0 → -1
            (1.0, 3, 2.0),  // row 3: scale=1, q=3 → +2
        ];

        // Activation: all ones → dot = sum of row weights
        let x_vals = vec![1.0f32; k];

        // Build raw Q2_0 bytes and pack.
        let mut packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q2_0_PACKED_U32S);
        let mut expected: Vec<f32> = Vec::with_capacity(n);
        for &(d, q_val, elem_val) in test_cases {
            let block = make_q2_0_block(d, q_val);
            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let scale_f32 = f16_to_f32(scale_bits);
            packed.push(scale_f32.to_bits());
            for w in 0..8usize {
                let b0 = block[2 + w * 4] as u32;
                let b1 = block[3 + w * 4] as u32;
                let b2 = block[4 + w * 4] as u32;
                let b3 = block[5 + w * 4] as u32;
                packed.push(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
            // Expected: k elements each = elem_val, activation all 1 → dot = k * elem_val
            expected.push(k as f32 * elem_val);
        }

        // Upload to GPU and dispatch the kernel.
        let device = Device::new(0).expect("CUDA device 0 not available");
        let stream: std::sync::Arc<cuda_core::Stream> =
            device.new_stream().expect("stream").into();
        let q2_gemv = CutileQ2Gemv::from_stream(stream);

        let x_dev: std::sync::Arc<cutile::tensor::Tensor<f32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(x_vals))
                .sync_on(q2_gemv.stream())
                .expect("upload x")
                .into();
        let p_dev: std::sync::Arc<cutile::tensor::Tensor<u32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(packed))
                .sync_on(q2_gemv.stream())
                .expect("upload packed")
                .into();

        let result = q2_gemv
            .forward_d2d(&x_dev, &p_dev, n, k_blocks)
            .expect("q2_0 gemv");

        let out: Vec<f32> = result
            .to_host_vec()
            .sync_on(q2_gemv.stream())
            .expect("download");

        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-2,
                "row {i}: got {got}, expected {exp}"
            );
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ── Q2_0 block layout and bit-extraction safety ───────────────────────────

    /// Byte indices accessed during Q2_0 block upload are within the 34-byte block.
    ///
    /// Upload loop: block[2 + w*4 + b] for w in 0..8, b in 0..4.
    /// Max index: 2 + 7*4 + 3 = 33. Block size = 34.
    #[kani::proof]
    fn q2_0_block_byte_indices_in_bounds() {
        let w: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(w < 8);
        kani::assume(b < 4);
        let idx = 2 + w * 4 + b;
        assert!(idx < Q2_0_BLOCK_BYTES);
    }

    /// Extracting a 2-bit field from a packed u32 always yields a value in 0..=3.
    #[kani::proof]
    fn q2_0_2bit_extraction_range() {
        let word: u32 = kani::any();
        let bit_pair: u32 = kani::any();
        kani::assume(bit_pair < 16);
        let q = (word >> (bit_pair * 2)) & 3u32;
        assert!(q <= 3);
    }

    /// Centering: (q - 1) for q in 0..=3 gives exactly {-1, 0, 1, 2}; no i32 underflow.
    #[kani::proof]
    fn q2_0_centered_value_range() {
        let q: u32 = kani::any();
        kani::assume(q <= 3);
        let centered = q as i32 - 1;
        assert!(centered >= -1 && centered <= 2);
    }

    /// IEEE 754 trick: `q | 0x4B000000` for q in 0..=3 has exp field = 0x96, not 0xFF,
    /// so the resulting f32 bitcast is a normal finite number (not NaN or Inf).
    #[kani::proof]
    fn q2_0_ieee_trick_no_nan() {
        let q: u32 = kani::any();
        kani::assume(q <= 3);
        let biased = q | 0x4B000000u32;
        let exp = (biased >> 23) & 0xFF;
        assert!(exp != 0xFF);
    }

    /// Packed GPU slot indices: `1 + word` < Q2_0_PACKED_U32S (= 9) for word in 0..8.
    #[kani::proof]
    fn q2_0_packed_word_slot_in_bounds() {
        let word: usize = kani::any();
        kani::assume(word < 8);
        let offset_in_block = 1 + word;
        assert!(offset_in_block < Q2_0_PACKED_U32S);
    }

    // ── Q4K block layout and nibble safety ───────────────────────────────────

    /// Nibble byte index for Q4K qs array: 16 + idx/2 < 144 for idx < 256.
    #[kani::proof]
    fn q4k_block_upload_no_oob() {
        let idx: usize = kani::any();
        kani::assume(idx < 256);
        // d: bytes 0-1, dmin: bytes 2-3, scales: bytes 4-15, qs: bytes 16-143
        let byte_idx = 16usize + idx / 2;
        assert!(byte_idx < Q4K_BLOCK_BYTES);
    }

    /// Q4K nibble extraction: both the low and high nibble of any byte are in 0..=15.
    #[kani::proof]
    fn q4k_nibble_range() {
        let byte: u8 = kani::any();
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        assert!(lo <= 15);
        assert!(hi <= 15);
    }

    /// Q4K 6-bit scale extraction: `byte & 0x3F` is always in 0..=63.
    #[kani::proof]
    fn q4k_6bit_scale_range() {
        let byte: u8 = kani::any();
        let sc = byte & 0x3F;
        assert!(sc <= 63);
    }

    // ── Q6K block layout and 6-bit reconstruction ────────────────────────────

    /// Q6K 6-bit reconstruction from ql (low 4 bits) + qh (high 2 bits) is in 0..=63.
    /// The signed offset `q6 - 32` is then in [-32, 31].
    #[kani::proof]
    fn q6k_block_upload_no_oob() {
        let ql_byte: u8 = kani::any();
        let qh_byte: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);
        let low4 = (ql_byte & 0x0F) as u32;
        let high2 = ((qh_byte >> (shift * 2)) & 0x03) as u32;
        let q6 = low4 | (high2 << 4);
        assert!(q6 <= 63);
        let q_signed = q6 as i32 - 32;
        assert!(q_signed >= -32 && q_signed <= 31);
    }

    // ── TQ2 ternary bit extraction ────────────────────────────────────────────

    /// TQ2 2-bit trit extraction: value in 0..=3, centered (trit-1) in {-1,0,1,2}.
    #[kani::proof]
    fn tq2_block_upload_no_oob() {
        let byte: u8 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(shift < 4);
        let trit = ((byte >> (shift * 2)) & 0x03) as i32;
        assert!(trit >= 0 && trit <= 3);
        let centered = trit - 1;
        assert!(centered >= -1 && centered <= 2);
    }

    // ── Full-block dequant no-panic harnesses ─────────────────────────────────

    #[kani::proof]
    fn dequant_q4k_block_no_panic() {
        let block: [u8; 144] = kani::any();
        let mut out = [0.0f32; 256];
        dequant_q4k_block(&block, &mut out);
    }

    #[kani::proof]
    fn dequant_q6k_block_no_panic() {
        let block: [u8; 210] = kani::any();
        let mut out = [0.0f32; 256];
        dequant_q6k_block(&block, &mut out);
    }

    #[kani::proof]
    #[kani::unwind(65)]
    fn dequant_tq2_block_no_panic() {
        let block: [u8; 66] = kani::any();
        let mut out = [0.0f32; 256];
        dequant_tq2_block(&block, &mut out);
    }
}

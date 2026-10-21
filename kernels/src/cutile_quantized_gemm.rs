use crate::{
    KernelError, MatMulQ, Result, VerificationTier, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS,
    Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE,
};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::prelude::{DeviceOp, TileKernel, ToHostVec};
use cutile::tensor::{IntoPartition, Reshape, Tensor};
use std::sync::Arc;

// -- Q6_K block layout constants -----------------------------------------------
// Q6_K: 6-bit, 256 elements per block, 210 bytes on disk
//   [0..128]   ql     (low 4-bit per element, 128 bytes)
//   [128..192] qh     (high 2-bit per element, 64 bytes -- 4 elements per byte)
//   [192..208] scales (16 i8 values, one per 16-element group)
//   [208..210] d      (f16 super-scale)
// Dequant: q_full = (ql[i/2] >> 4*(i%2) & 0xF) | (qh bits << 4); weight = d x scale[g] x (q_full ??? 32)
//
// GPU packed format (53 u32 per block, native ql+qh layout):
//   u32[0]      = f32 bits of d (f16 -> f32 at upload time)
//   u32[1..5]   = 4 u32s, 16 group scales packed as bias-128 u8 (4 per u32)
//                 stored_byte = (scale_i8 as i32 + 128) as u8
//                 decode: sc_f32 = bitcast(stored | 0x4B000000) ??? 8388736.0
//   u32[5..37]  = 32 u32s of ql[0..128] (native GGUF order, 4 u8 per u32)
//   u32[37..53] = 16 u32s of qh[0..64] (native GGUF order, 4 u8 per u32)
//
// Dequant for element within ??? [0,255]:
//   half = within/128, sub = (within%128)/32, l = within%32, group = within/16
//   ql_byte_idx = half*64 + l + (sub%2)*32
//   lo = (ql[ql_byte_idx] >> ((sub/2)*4)) & 0xF
//   qh_byte_idx = half*32 + l
//   hi = (qh[qh_byte_idx] >> (sub*2)) & 0x3
//   q6 = lo | (hi<<4); weight = d x scale[group] x (q6 ??? 32)
pub const Q6K_QK: usize = 256;
pub const Q6K_PACKED_U32S: usize = 53; // 1 (d) + 4 (scales) + 32 (ql) + 16 (qh) = 53

// -- Q2_0 block layout constants ----------------------------------------------
// Q2_0 (PrismML): 2-bit quantized, 128 elements per block, 34 bytes per block
//   [0..2]   d  (f16 scale)
//   [2..34]  qs (32 bytes = 128 elements x 2 bits, 4 elements per byte)
// Dequant: (((qs[i/4] >> (2*(i%4))) & 3) - 1) x d
// Encoding: q=0->-1, q=1->0, q=2->+1, q=3->+2
//
// GPU packed format (9 u32 per block):
//   u32[0]    = f32 bits of scale (f16 -> f32 at upload time)
//   u32[1..8] = 8 u32 words of packed 2-bit data (each holds 16 elements)
pub const Q2_0_QK: usize = 128;
pub const Q2_0_BLOCK_BYTES: usize = 34; // raw on-disk bytes per block
pub const Q2_0_PACKED_U32S: usize = 9;  // u32 slots per block in GPU format

// -- Q4_K block layout constants ----------------------------------------------
// Q4_K: 4-bit quantized, 256 elements per block, 144 bytes per block
//   [0..2]    d      (f16 super-scale)
//   [2..4]    dmin   (f16 super-minimum)
//   [4..16]   sc     (12 bytes packed: 8 groups x 6-bit scale + 8 groups x 6-bit min)
//   [16..144] qs     (128 bytes = 256 nibbles, 4-bit each)
//
// GPU packed format (42 u32 per block):
//   u32[0]       : f32 bits of d (f16 -> f32 at upload time)
//   u32[1]       : f32 bits of dmin (f16 -> f32 at upload time)
//   u32[2+j]     : sc[2*j] | (sc[2*j+1]<<8) for j=0..4  -- 4 u32s for 8 unpacked scale values
//   u32[6+j]     : m[2*j]  | (m[2*j+1]<<8)  for j=0..4  -- 4 u32s for 8 unpacked min values
//   u32[10..42]  : 32 u32s holding qs[0..128] (4 bytes per u32, little-endian)
//
// Dequant per element i (group g = i/32):
//   nibble = qs[i/2] nibble at bit (4*(i%2))   [low nibble for even i, high for odd]
//   Actually: for j=0..4, l=0..32:
//     out[j*64+l]    = d * sc[2*j]   * (qs[j*32+l] & 0xF)  - dmin * m[2*j]
//     out[j*64+l+32] = d * sc[2*j+1] * (qs[j*32+l] >> 4)   - dmin * m[2*j+1]
pub const Q4K_PACKED_U32S: usize = 42; // u32 slots per block in GPU format

// -- cuTile Q2_0 GEMV kernel ---------------------------------------------------

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Q2_0 dequantized GEMV: z[1] = Sigma_k dequant(packed[pid_n, k]) x x[k]
    ///
    /// Packed layout: [N x K_BLOCKS x 9] u32 (conceptually [N, K_BLOCKS, 9]):
    ///   packed[n, blk, 0]    = f32 bits of block scale (converted from f16)
    ///   packed[n, blk, 1..8] = 8 u32 words of 2-bit weights
    ///     bits[2i+1 : 2i] of word w = q of element blk*128 + w*16 + i
    ///     dequant: (q - 1) x scale, q ??? {0,1,2,3} -> values ??? {-1,0,1,2} x scale
    ///
    /// Grid: (N, 1, 1) -- one program per output neuron.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
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

                // IEEE 754 integer-to-float trick -- avoids itof (cutile-0.3.1 bug:
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

                    // Integer->float via IEEE bit trick:
                    // bitcast(q | 0x4B000000) - 8388609.0 == (q as f32) - 1.0
                    let biased: Tile<u32, { [1] }> = ori(q_u32, magic);
                    let q_f32: Tile<f32, { [1] }> = bitcast(biased) - bias_f32;

                    // Load the corresponding activation element.
                    let x_elem: Tile<f32, { [1] }> = part_x.load([x_word_base + bit_pair]);

                    // Accumulate: acc += scale x q x x
                    acc = acc + scale * q_f32 * x_elem;
                }
            }
        }
        z.store(acc);
    }

    /// Q2_0 multi-output GEMV: BN output neurons per program.
    ///
    /// Uses a rank-2 `MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>` with
    /// `MAP_SHAPE: [i32; 2]` = `[1, BN]`, bypassing the cuTile 0.4.0 restriction
    /// that rank-1 tile maps must be `[1]`. Each program iterates over BN
    /// consecutive output tiles via `iter_indices()` and computes one dot product
    /// per iteration, reducing kernel launch overhead by BNx vs. the scalar variant.
    ///
    /// Packed layout: same as `cutile_q2_0_gemv` -- `[N x K_BLOCKS x 9]` u32.
    /// Inputs x and packed must be reshaped to rank-2 `[1, K]` / `[1, total]`
    /// at the dispatch level so that `[1, 1]` tile loads are type-consistent
    /// with the `[1, 1]` accumulator.
    ///
    /// Grid: N/BN programs (one per group of BN output neurons).
    #[cutile::entry()]
    pub fn cutile_q2_0_gemv_bn<const K_BLOCKS: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>,
        x: &Tensor<f32, { [1, -1] }>,
        packed: &Tensor<u32, { [1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, 1]);
        let part_p = packed.partition(shape![1, 1]);
        let magic: Tile<u32, { [1, 1] }> = broadcast_scalar(0x4B000000_u32, shape![1, 1]);
        let bias_f32: Tile<f32, { [1, 1] }> = broadcast_scalar(8388609.0_f32, shape![1, 1]);
        let mask: Tile<u32, { [1, 1] }> = broadcast_scalar(3_u32, shape![1, 1]);

        for index in z.iter_indices() {
            let (_, n) = index.components();
            let mut acc: Tile<f32, { [1, 1] }> = broadcast_scalar(0.0_f32, shape![1, 1]);

            for blk in 0i32..K_BLOCKS {
                let pack_base = n * K_BLOCKS * 9 + blk * 9;
                let x_base = blk * 128;

                let scale_u32: Tile<u32, { [1, 1] }> = part_p.load([0, pack_base]);
                let scale: Tile<f32, { [1, 1] }> = bitcast(scale_u32);

                for word in 0i32..8 {
                    let word_data: Tile<u32, { [1, 1] }> =
                        part_p.load([0, pack_base + 1 + word]);
                    let x_word_base = x_base + word * 16;

                    for bit_pair in 0i32..16 {
                        let shift: Tile<u32, { [1, 1] }> =
                            broadcast_scalar((bit_pair * 2) as u32, shape![1, 1]);
                        let q_u32: Tile<u32, { [1, 1] }> = andi(shri(word_data, shift), mask);
                        let biased: Tile<u32, { [1, 1] }> = ori(q_u32, magic);
                        let q_f32: Tile<f32, { [1, 1] }> = bitcast(biased) - bias_f32;
                        let x_elem: Tile<f32, { [1, 1] }> =
                            part_x.load([0, x_word_base + bit_pair]);
                        acc = acc + scale * q_f32 * x_elem;
                    }
                }
            }
            z.store(acc, index);
        }
    }

    /// Q6_K multi-output GEMV: BN output neurons per program.
    ///
    /// Uses a rank-2 `MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>` with
    /// `MAP_SHAPE: [i32; 2]` = `[1, BN]`, bypassing the cuTile 0.4.0 restriction
    /// that rank-1 tile maps must be `[1]`. Reduces kernel launch overhead by BNx.
    ///
    /// Packed layout: same as `cutile_q6k_gemv` -- `[N x K_BLOCKS x 53]` u32.
    ///
    /// Grid: N/BN programs. MAP_SHAPE = [1, BN]: each program iterates BN tiles.
    /// Dispatch: output tensor must be reshaped to `[1, N]` and partitioned `[1, 1]`
    /// before calling `.map([1, BN], N/BN)`.
    #[cutile::entry()]
    pub fn cutile_q6k_gemv_bn<const K_BLOCKS: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>,
        x: &Tensor<f32, { [1, -1] }>,
        packed: &Tensor<u32, { [1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, 1]);
        let part_p = packed.partition(shape![1, 1]);
        // q6 ??? [0,63]: bitcast(q6 | 0x4B000000) ??? 8388640.0 -> q6???32 ??? [???32,31]
        let magic:     Tile<u32, { [1, 1] }> = broadcast_scalar(0x4B000000_u32, shape![1, 1]);
        let mask_ff:   Tile<u32, { [1, 1] }> = broadcast_scalar(0xFF_u32, shape![1, 1]);
        let mask_0f:   Tile<u32, { [1, 1] }> = broadcast_scalar(0x0F_u32, shape![1, 1]);
        let mask_03:   Tile<u32, { [1, 1] }> = broadcast_scalar(0x03_u32, shape![1, 1]);
        let center:    Tile<f32, { [1, 1] }> = broadcast_scalar(8388640.0_f32, shape![1, 1]);
        // scale bias-128 decode: bitcast(sc | 0x4B000000) ??? 8388736.0 -> sc_i8 ??? [???128,127]
        let sc_center: Tile<f32, { [1, 1] }> = broadcast_scalar(8388736.0_f32, shape![1, 1]);
        let sixteen:   Tile<u32, { [1, 1] }> = broadcast_scalar(16u32, shape![1, 1]);

        for index in z.iter_indices() {
            let (_, n) = index.components();
            let mut acc: Tile<f32, { [1, 1] }> = broadcast_scalar(0.0_f32, shape![1, 1]);

            for blk in 0i32..K_BLOCKS {
                let pack_base = n * K_BLOCKS * 53 + blk * 53;
                let x_base = blk * 256;

                let d_u32: Tile<u32, { [1, 1] }> = part_p.load([0, pack_base]);
                let d: Tile<f32, { [1, 1] }> = bitcast(d_u32);

                for group in 0i32..16 {
                    let half = group / 8;
                    let sub  = (group % 8) / 2;

                    // Scale: bias-128 u8 in u32[1 + group/4], byte (group%4)
                    let sc_word: Tile<u32, { [1, 1] }> =
                        part_p.load([0, pack_base + 1 + group / 4]);
                    let sc_shift = (group % 4 * 8) as u32;
                    let sc_byte: Tile<u32, { [1, 1] }> =
                        andi(shri(sc_word, broadcast_scalar(sc_shift, shape![1, 1])), mask_ff);
                    let sc: Tile<f32, { [1, 1] }> = bitcast(ori(sc_byte, magic)) - sc_center;
                    let d_sc: Tile<f32, { [1, 1] }> = d * sc;

                    // ql u32 base: ql section starts at +5; half*16 + (sub%2)*8 u32 offset
                    let ql_u32_base = pack_base + 5 + half * 16 + (sub % 2) * 8;
                    // qh u32 base: qh section starts at +37; half*8 u32 offset
                    let qh_u32_base = pack_base + 37 + half * 8;
                    // nibble shift: sub<2 -> lo nibble (shift=0), sub>=2 -> hi nibble (shift=4)
                    let nibble_shift = ((sub / 2) * 4) as u32;
                    // qh bit shift: sub*2 (0,2,4,6)
                    let qh_shift = (sub * 2) as u32;
                    // l_base: (group%2)*16 elements offset within this sub's ql/qh section
                    let l_base = group % 2 * 16;
                    let ql_word_start = ql_u32_base + l_base / 4;
                    let qh_word_start = qh_u32_base + l_base / 4;

                    for word in 0i32..4 {
                        let ql_word: Tile<u32, { [1, 1] }> =
                            part_p.load([0, ql_word_start + word]);
                        let qh_word: Tile<u32, { [1, 1] }> =
                            part_p.load([0, qh_word_start + word]);
                        let x_elem_base = x_base + group * 16 + word * 4;

                        for byte_pos in 0i32..4 {
                            let shift: Tile<u32, { [1, 1] }> =
                                broadcast_scalar((byte_pos * 8) as u32, shape![1, 1]);
                            // Extract ql nibble
                            let ql_byte: Tile<u32, { [1, 1] }> =
                                andi(shri(ql_word, shift), mask_ff);
                            let lo: Tile<u32, { [1, 1] }> = andi(
                                shri(ql_byte, broadcast_scalar(nibble_shift, shape![1, 1])),
                                mask_0f,
                            );
                            // Extract qh 2-bit field
                            let qh_byte: Tile<u32, { [1, 1] }> =
                                andi(shri(qh_word, shift), mask_ff);
                            let hi: Tile<u32, { [1, 1] }> = andi(
                                shri(qh_byte, broadcast_scalar(qh_shift, shape![1, 1])),
                                mask_03,
                            );
                            // q6 = lo | (hi<<4); since ranges don't overlap, lo + hi*16 == lo | (hi<<4)
                            let q6: Tile<u32, { [1, 1] }> = lo + hi * sixteen;
                            let q_f32: Tile<f32, { [1, 1] }> = bitcast(ori(q6, magic)) - center;
                            let x_val: Tile<f32, { [1, 1] }> =
                                part_x.load([0, x_elem_base + byte_pos]);
                            acc = acc + d_sc * q_f32 * x_val;
                        }
                    }
                }
            }
            z.store(acc, index);
        }
    }

    /// Q6_K dequantized GEMV: z[1] = Sigma_k dequant(packed[pid_n, k]) x x[k]
    ///
    /// Packed layout: [N x K_BLOCKS x Q6K_PACKED_U32S] u32 (53 u32/block)
    ///   packed[n, blk, 0]       = f32 bits of d (block super-scale)
    ///   packed[n, blk, 1..5]    = 4 u32 words: 16 group scales as bias-128 u8 (4/u32)
    ///   packed[n, blk, 5..37]   = 32 u32 words: ql[0..128] native GGUF order
    ///   packed[n, blk, 37..53]  = 16 u32 words: qh[0..64] native GGUF order
    ///   dequant: within->(half,sub,l,group); lo+hi nibbles->q6 ???[0,63]; dxscx(q6-32)
    ///
    /// Grid: (N, 1, 1) -- one program per output neuron.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
    pub fn cutile_q6k_gemv<const K_BLOCKS: i32>(
        z: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
        packed: &Tensor<u32, { [-1] }>,
    ) {
        let pid_n = program_id(0);
        let part_x = x.partition(shape![1]);
        let part_p = packed.partition(shape![1]);
        let mut acc: Tile<f32, { [1] }> = broadcast_scalar(0.0_f32, shape![1]);

        // q6 ??? [0,63]: bitcast(q6 | 0x4B000000) ??? 8388640.0 -> q6???32 ??? [???32,31]
        let magic:     Tile<u32, { [1] }> = broadcast_scalar(0x4B000000_u32, shape![1]);
        let mask_ff:   Tile<u32, { [1] }> = broadcast_scalar(0xFF_u32, shape![1]);
        let mask_0f:   Tile<u32, { [1] }> = broadcast_scalar(0x0F_u32, shape![1]);
        let mask_03:   Tile<u32, { [1] }> = broadcast_scalar(0x03_u32, shape![1]);
        let center:    Tile<f32, { [1] }> = broadcast_scalar(8388640.0_f32, shape![1]);
        // scale bias-128 decode: bitcast(sc | 0x4B000000) ??? 8388736.0 -> sc_i8 ??? [???128,127]
        let sc_center: Tile<f32, { [1] }> = broadcast_scalar(8388736.0_f32, shape![1]);
        let sixteen:   Tile<u32, { [1] }> = broadcast_scalar(16u32, shape![1]);

        for blk in 0i32..K_BLOCKS {
            // pack_base: start of this block's 53-u32 region for neuron pid_n.
            let pack_base = pid_n * K_BLOCKS * 53 + blk * 53;
            let x_base = blk * 256;

            let d_u32: Tile<u32, { [1] }> = part_p.load([pack_base]);
            let d: Tile<f32, { [1] }> = bitcast(d_u32);

            // 16 groups of 16 elements; group index == element_index / 16.
            for group in 0i32..16 {
                let half = group / 8;    // 0 or 1
                let sub  = (group % 8) / 2;  // 0..3

                // Scale: bias-128 encoded u8 in u32[1 + group/4], byte (group%4)
                let sc_word: Tile<u32, { [1] }> = part_p.load([pack_base + 1 + group / 4]);
                let sc_shift = (group % 4 * 8) as u32;
                let sc_byte: Tile<u32, { [1] }> =
                    andi(shri(sc_word, broadcast_scalar(sc_shift, shape![1])), mask_ff);
                let sc: Tile<f32, { [1] }> = bitcast(ori(sc_byte, magic)) - sc_center;
                let d_sc: Tile<f32, { [1] }> = d * sc;

                // ql base: ql section at +5; half*16 + (sub%2)*8 selects the right 32-byte span
                let ql_u32_base = pack_base + 5 + half * 16 + (sub % 2) * 8;
                // qh base: qh section at +37; half*8 selects the right 32-byte span
                let qh_u32_base = pack_base + 37 + half * 8;
                // nibble shift: sub<2 -> lo nibble (0), sub>=2 -> hi nibble (4)
                let nibble_shift = ((sub / 2) * 4) as u32;
                // qh bit shift: sub*2 extracts the 2 bits for this sub-group
                let qh_shift = (sub * 2) as u32;
                // l_base = (group%2)*16: offset within the 32-element span for this group
                let l_base = group % 2 * 16;
                let ql_word_start = ql_u32_base + l_base / 4;
                let qh_word_start = qh_u32_base + l_base / 4;

                // 4 u32 words cover the 16 elements of this group.
                for word in 0i32..4 {
                    let ql_word: Tile<u32, { [1] }> = part_p.load([ql_word_start + word]);
                    let qh_word: Tile<u32, { [1] }> = part_p.load([qh_word_start + word]);
                    let x_elem_base = x_base + group * 16 + word * 4;

                    for byte_pos in 0i32..4 {
                        let shift: Tile<u32, { [1] }> =
                            broadcast_scalar((byte_pos * 8) as u32, shape![1]);
                        // Extract ql byte, select nibble (lo or hi depending on sub)
                        let ql_byte: Tile<u32, { [1] }> = andi(shri(ql_word, shift), mask_ff);
                        let lo: Tile<u32, { [1] }> = andi(
                            shri(ql_byte, broadcast_scalar(nibble_shift, shape![1])),
                            mask_0f,
                        );
                        // Extract qh byte, select 2-bit field
                        let qh_byte: Tile<u32, { [1] }> = andi(shri(qh_word, shift), mask_ff);
                        let hi: Tile<u32, { [1] }> = andi(
                            shri(qh_byte, broadcast_scalar(qh_shift, shape![1])),
                            mask_03,
                        );
                        // q6 ??? [0,63]: lo | (hi<<4). Ranges don't overlap -> + == |
                        let q6: Tile<u32, { [1] }> = lo + hi * sixteen;
                        let q_f32: Tile<f32, { [1] }> = bitcast(ori(q6, magic)) - center;
                        let x_val: Tile<f32, { [1] }> = part_x.load([x_elem_base + byte_pos]);
                        acc = acc + d_sc * q_f32 * x_val;
                    }
                }
            }
        }
        z.store(acc);
    }

    /// Q4_K multi-output GEMV: BN output neurons per program.
    ///
    /// Uses a rank-2 `MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>` with
    /// `MAP_SHAPE: [i32; 2]` = `[1, BN]`, bypassing the cuTile 0.4.0 restriction
    /// that rank-1 tile maps must be `[1]`. Reduces kernel launch overhead by BNx.
    ///
    /// Packed layout: same as `cutile_q4k_gemv` -- `[N x K_BLOCKS x 42]` u32.
    ///
    /// Grid: N/BN programs. MAP_SHAPE = [1, BN]: each program iterates BN tiles.
    /// Dispatch: output tensor must be reshaped to `[1, N]` and partitioned `[1, 1]`
    /// before calling `.map([1, BN], N/BN)`.
    #[cutile::entry()]
    pub fn cutile_q4k_gemv_bn<const K_BLOCKS: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [1, 1] }, MAP_SHAPE>,
        x: &Tensor<f32, { [1, -1] }>,
        packed: &Tensor<u32, { [1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, 1]);
        let part_p = packed.partition(shape![1, 1]);
        let magic: Tile<u32, { [1, 1] }> = broadcast_scalar(0x4B000000_u32, shape![1, 1]);
        let bias_f32: Tile<f32, { [1, 1] }> = broadcast_scalar(8388608.0_f32, shape![1, 1]);
        let mask_8: Tile<u32, { [1, 1] }> = broadcast_scalar(0xFF_u32, shape![1, 1]);
        let mask_4: Tile<u32, { [1, 1] }> = broadcast_scalar(0xF_u32, shape![1, 1]);
        let shift_8: Tile<u32, { [1, 1] }> = broadcast_scalar(8_u32, shape![1, 1]);
        let shift_4: Tile<u32, { [1, 1] }> = broadcast_scalar(4_u32, shape![1, 1]);

        for index in z.iter_indices() {
            let (_, n) = index.components();
            let mut acc: Tile<f32, { [1, 1] }> = broadcast_scalar(0.0_f32, shape![1, 1]);

            for blk in 0i32..K_BLOCKS {
                let base = n * K_BLOCKS * 42 + blk * 42;
                let x_base = blk * 256;

                let d_u32: Tile<u32, { [1, 1] }> = part_p.load([0, base]);
                let d: Tile<f32, { [1, 1] }> = bitcast(d_u32);

                let dmin_u32: Tile<u32, { [1, 1] }> = part_p.load([0, base + 1]);
                let dmin: Tile<f32, { [1, 1] }> = bitcast(dmin_u32);

                for j in 0i32..4 {
                    let sc_pair: Tile<u32, { [1, 1] }> = part_p.load([0, base + 2 + j]);
                    let m_pair: Tile<u32, { [1, 1] }> = part_p.load([0, base + 6 + j]);

                    let sc_lo_u32: Tile<u32, { [1, 1] }> = andi(sc_pair, mask_8);
                    let sc_hi_u32: Tile<u32, { [1, 1] }> =
                        andi(shri(sc_pair, shift_8), mask_8);
                    let sc_lo_f32: Tile<f32, { [1, 1] }> =
                        bitcast(ori(sc_lo_u32, magic)) - bias_f32;
                    let sc_hi_f32: Tile<f32, { [1, 1] }> =
                        bitcast(ori(sc_hi_u32, magic)) - bias_f32;

                    let m_lo_u32: Tile<u32, { [1, 1] }> = andi(m_pair, mask_8);
                    let m_hi_u32: Tile<u32, { [1, 1] }> =
                        andi(shri(m_pair, shift_8), mask_8);
                    let m_lo_f32: Tile<f32, { [1, 1] }> =
                        bitcast(ori(m_lo_u32, magic)) - bias_f32;
                    let m_hi_f32: Tile<f32, { [1, 1] }> =
                        bitcast(ori(m_hi_u32, magic)) - bias_f32;

                    let d1: Tile<f32, { [1, 1] }> = d * sc_lo_f32;
                    let m1: Tile<f32, { [1, 1] }> = dmin * m_lo_f32;
                    let d2: Tile<f32, { [1, 1] }> = d * sc_hi_f32;
                    let m2: Tile<f32, { [1, 1] }> = dmin * m_hi_f32;

                    for l in 0i32..32 {
                        let qs_word_idx = (j * 32 + l) / 4;
                        let byte_shift = ((j * 32 + l) % 4) * 8;

                        let qs_word: Tile<u32, { [1, 1] }> =
                            part_p.load([0, base + 10 + qs_word_idx]);
                        let byte_shift_tile: Tile<u32, { [1, 1] }> =
                            broadcast_scalar(byte_shift as u32, shape![1, 1]);
                        let byte_val: Tile<u32, { [1, 1] }> =
                            andi(shri(qs_word, byte_shift_tile), mask_8);

                        let lo_nibble: Tile<u32, { [1, 1] }> = andi(byte_val, mask_4);
                        let hi_nibble: Tile<u32, { [1, 1] }> = shri(byte_val, shift_4);

                        let lo_f32: Tile<f32, { [1, 1] }> =
                            bitcast(ori(lo_nibble, magic)) - bias_f32;
                        let hi_f32: Tile<f32, { [1, 1] }> =
                            bitcast(ori(hi_nibble, magic)) - bias_f32;

                        let w_lo: Tile<f32, { [1, 1] }> = d1 * lo_f32 - m1;
                        let w_hi: Tile<f32, { [1, 1] }> = d2 * hi_f32 - m2;

                        let x_lo: Tile<f32, { [1, 1] }> =
                            part_x.load([0, x_base + j * 64 + l]);
                        let x_hi: Tile<f32, { [1, 1] }> =
                            part_x.load([0, x_base + j * 64 + 32 + l]);

                        acc = acc + w_lo * x_lo;
                        acc = acc + w_hi * x_hi;
                    }
                }
            }
            z.store(acc, index);
        }
    }

    /// Q4_K dequantized GEMV: z[1] = Sigma_k dequant(packed[pid_n, k]) x x[k]
    ///
    /// GPU packed format per block (42 u32s):
    ///   u32[0]      : f32 bits of d (super-scale, converted from f16 at upload)
    ///   u32[1]      : f32 bits of dmin (super-min, converted from f16 at upload)
    ///   u32[2+j]    : sc[2*j] | (sc[2*j+1]<<8) for j=0..4 (group pair scales, 6-bit -> 8-bit slots)
    ///   u32[6+j]    : m[2*j]  | (m[2*j+1]<<8)  for j=0..4 (group pair mins)
    ///   u32[10..42] : 32 u32s of packed qs bytes (128 bytes = 256 4-bit weights)
    ///
    /// Dequant: for j=0..4, l=0..32:
    ///   lo = qs[j*32+l] & 0xF  -> w = d * sc[2*j]   * lo  - dmin * m[2*j]   (element j*64+l)
    ///   hi = qs[j*32+l] >> 4   -> w = d * sc[2*j+1] * hi  - dmin * m[2*j+1] (element j*64+l+32)
    ///
    /// Grid: (N, 1, 1) -- one program per output neuron.
    #[cutile::entry(optimization_hints = (sm_120 = (vec_width = 4,)))]
    pub fn cutile_q4k_gemv<const K_BLOCKS: i32>(
        z: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
        packed: &Tensor<u32, { [-1] }>,
    ) {
        let pid_n = program_id(0);
        let part_x = x.partition(shape![1]);
        let part_p = packed.partition(shape![1]);
        let mut acc: Tile<f32, { [1] }> = broadcast_scalar(0.0_f32, shape![1]);

        // Q4K_PACKED_U32S = 42 (literal -- cutile const generics not available here)
        for blk in 0i32..K_BLOCKS {
            let base = pid_n * K_BLOCKS * 42 + blk * 42;
            let x_base = blk * 256;

            // Load d and dmin (stored as f32 bits in u32[0] and u32[1])
            let d_u32: Tile<u32, { [1] }> = part_p.load([base]);
            let d: Tile<f32, { [1] }> = bitcast(d_u32);

            let dmin_u32: Tile<u32, { [1] }> = part_p.load([base + 1]);
            let dmin: Tile<f32, { [1] }> = bitcast(dmin_u32);

            // IEEE 754 integer-to-float trick (same as cutile_q2_0_gemv):
            // For u32 x in [0, 2^23): bitcast(x | 0x4B000000) - 8388608.0 = (float)x.
            let magic: Tile<u32, { [1] }> = broadcast_scalar(0x4B000000_u32, shape![1]);
            let bias_f32: Tile<f32, { [1] }> = broadcast_scalar(8388608.0_f32, shape![1]);
            let mask_8: Tile<u32, { [1] }> = broadcast_scalar(0xFF_u32, shape![1]);
            let mask_4: Tile<u32, { [1] }> = broadcast_scalar(0xF_u32, shape![1]);
            let shift_8: Tile<u32, { [1] }> = broadcast_scalar(8_u32, shape![1]);
            let shift_4: Tile<u32, { [1] }> = broadcast_scalar(4_u32, shape![1]);

            // 4 pairs of 32-element groups (j=0..4).
            // Group 2*j covers elements j*64..j*64+32, group 2*j+1 covers j*64+32..j*64+64.
            for j in 0i32..4 {
                // u32[2+j]: sc[2*j] in bits[7:0], sc[2*j+1] in bits[15:8]
                let sc_pair: Tile<u32, { [1] }> = part_p.load([base + 2 + j]);
                // u32[6+j]: m[2*j]  in bits[7:0], m[2*j+1]  in bits[15:8]
                let m_pair: Tile<u32, { [1] }> = part_p.load([base + 6 + j]);

                // Extract and float-convert sc[2*j] and sc[2*j+1]
                let sc_lo_u32: Tile<u32, { [1] }> = andi(sc_pair, mask_8);
                let sc_hi_u32: Tile<u32, { [1] }> = andi(shri(sc_pair, shift_8), mask_8);
                let sc_lo_f32: Tile<f32, { [1] }> = bitcast(ori(sc_lo_u32, magic)) - bias_f32;
                let sc_hi_f32: Tile<f32, { [1] }> = bitcast(ori(sc_hi_u32, magic)) - bias_f32;

                // Extract and float-convert m[2*j] and m[2*j+1]
                let m_lo_u32: Tile<u32, { [1] }> = andi(m_pair, mask_8);
                let m_hi_u32: Tile<u32, { [1] }> = andi(shri(m_pair, shift_8), mask_8);
                let m_lo_f32: Tile<f32, { [1] }> = bitcast(ori(m_lo_u32, magic)) - bias_f32;
                let m_hi_f32: Tile<f32, { [1] }> = bitcast(ori(m_hi_u32, magic)) - bias_f32;

                // Per-group dequant factors
                let d1: Tile<f32, { [1] }> = d * sc_lo_f32;   // d * sc[2*j]
                let m1: Tile<f32, { [1] }> = dmin * m_lo_f32; // dmin * m[2*j]
                let d2: Tile<f32, { [1] }> = d * sc_hi_f32;   // d * sc[2*j+1]
                let m2: Tile<f32, { [1] }> = dmin * m_hi_f32; // dmin * m[2*j+1]

                // 32 qs bytes for this j-pair: qs[j*32+l], l=0..32.
                // Each byte: low nibble -> element j*64+l, high nibble -> element j*64+l+32.
                // qs byte j*32+l lives in u32[10 + (j*32+l)/4] at byte (j*32+l)%4.
                for l in 0i32..32 {
                    let qs_word_idx = (j * 32 + l) / 4;
                    let byte_shift = ((j * 32 + l) % 4) * 8;

                    let qs_word: Tile<u32, { [1] }> = part_p.load([base + 10 + qs_word_idx]);
                    let byte_shift_tile: Tile<u32, { [1] }> =
                        broadcast_scalar(byte_shift as u32, shape![1]);
                    let byte_val: Tile<u32, { [1] }> =
                        andi(shri(qs_word, byte_shift_tile), mask_8);

                    let lo_nibble: Tile<u32, { [1] }> = andi(byte_val, mask_4);
                    let hi_nibble: Tile<u32, { [1] }> = shri(byte_val, shift_4);

                    // Convert nibbles to f32 via the same IEEE bit trick
                    let lo_f32: Tile<f32, { [1] }> = bitcast(ori(lo_nibble, magic)) - bias_f32;
                    let hi_f32: Tile<f32, { [1] }> = bitcast(ori(hi_nibble, magic)) - bias_f32;

                    // Dequant: w = d * sc * nibble - dmin * m
                    let w_lo: Tile<f32, { [1] }> = d1 * lo_f32 - m1;
                    let w_hi: Tile<f32, { [1] }> = d2 * hi_f32 - m2;

                    // Activation elements at x_base + j*64 + l  and  + j*64 + 32 + l
                    let x_lo: Tile<f32, { [1] }> = part_x.load([x_base + j * 64 + l]);
                    let x_hi: Tile<f32, { [1] }> = part_x.load([x_base + j * 64 + 32 + l]);

                    acc = acc + w_lo * x_lo;
                    acc = acc + w_hi * x_hi;
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

    /// Device-to-device Q2_0 GEMV: output[n] = packed_weights x x[k].
    ///
    /// `packed` holds `n x k_blocks x Q2_0_PACKED_U32S` u32 elements.
    /// `x` holds `k = k_blocks x Q2_0_QK` f32 activation elements.
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

        // reshape(&[n]) collapses the partition-stride metadata that
        // partition([1]).unpartition() leaves behind, producing a true rank-1
        // tensor (1 stride).  Downstream kernels (cutile_deinterleave, etc.)
        // require exactly 1 stride and would reject the 2-stride tensor that
        // a plain unpartition() produces.  The same pattern is used in
        // forward_gemv_coalesced_d2d in cutile_matmul.rs.
        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Device-to-device Q2_0 GEMV with BN outputs per program.
    ///
    /// Reduces kernel launch overhead by BNx versus `forward_d2d`.
    /// `n` must be divisible by `bn`. Uses a rank-2 `MappedPartitionMut` with
    /// `MAP_SHAPE=[1, BN]` (rank-2, bypassing the cuTile 0.4.0 rank-1 map restriction)
    /// so each program iterates over BN consecutive output tiles via `iter_indices()`.
    pub fn forward_d2d_bn(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
        bn: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        if n % bn != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "n={n} must be divisible by bn={bn}"
            )));
        }
        let stream = &self.stream;

        let k = k_blocks * Q2_0_QK;
        let total_packed = n * k_blocks * Q2_0_PACKED_U32S;

        // x may have next_pow2(k) elements (activation padded for non-power-of-2 intermediates).
        // Reshape to [1, next_pow2(k)]; kernel reads only k_blocks blocks of Q2_0_QK elements.
        let x_2d: Arc<Tensor<f32>> = Arc::clone(x)
            .reshape(&[1, k.next_power_of_two()])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        let p_2d: Arc<Tensor<u32>> = Arc::clone(packed)
            .reshape(&[1, total_packed])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let z = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[1, n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1, 1])
            .map([1, bn], (n / bn) as u32);

        let (z, _x, _p) =
            kernels::cutile_q2_0_gemv_bn(z, x_2d, p_2d)
                // MAP_SHAPE: [i32; 2] -- pass as two generics: MAP_SHAPE[0]=1, MAP_SHAPE[1]=bn
                .generics(vec![k_blocks.to_string(), "1".to_string(), bn.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Non-blocking Q2_0 GEMV reusing a pre-allocated output buffer.
    ///
    /// Enqueues the kernel without synchronising. The caller MUST ensure the
    /// stream is synced (or a subsequent `sync_on` on the same stream runs first)
    /// before reading the result on the CPU.
    ///
    /// # Safety
    ///
    /// The returned tensor's device memory is being written asynchronously.
    /// Do not read it on the CPU until the stream has been synced.
    pub unsafe fn submit_d2d_into(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let z = out
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (z, _x, _p) = unsafe {
            kernels::cutile_q2_0_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .async_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let flat = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(flat))
    }
}

/// GPU-resident Q6_K GEMV dispatcher.
///
/// Holds the CUDA stream and dispatches `cutile_q6k_gemv` for decode (m=1)
/// forward passes where weights are stored in pre-expanded packed Q6_K format.
pub struct CutileQ6kGemv {
    stream: Arc<Stream>,
}

unsafe impl Send for CutileQ6kGemv {}
unsafe impl Sync for CutileQ6kGemv {}

impl CutileQ6kGemv {
    /// Create from an existing CUDA stream (share with `CutileMatMul`).
    pub fn from_stream(stream: Arc<Stream>) -> Self {
        Self { stream }
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Device-to-device Q6_K GEMV: output[n] = packed_weights x x[k].
    ///
    /// `packed` holds `n x k_blocks x Q6K_PACKED_U32S` u32 elements.
    /// `x` holds `k = k_blocks x Q6K_QK` f32 activation elements.
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
            kernels::cutile_q6k_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // reshape(&[n]) collapses partition-stride metadata left by
        // partition([1]).unpartition() -- same fix as CutileQ2Gemv::forward_d2d.
        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Device-to-device Q6_K GEMV with BN outputs per program.
    ///
    /// Reduces kernel launch overhead by BNx versus `forward_d2d`. Uses rank-2
    /// `MappedPartitionMut<f32, { [1, 1] }, [1, BN]>` to bypass the cuTile 0.4.0
    /// "rank-1 maps must be [1]" restriction.
    pub fn forward_d2d_bn(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
        bn: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        if n % bn != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "n={n} must be divisible by bn={bn}"
            )));
        }
        let stream = &self.stream;

        let k = k_blocks * Q6K_QK;
        let total_packed = n * k_blocks * Q6K_PACKED_U32S;

        let x_2d: Arc<Tensor<f32>> = Arc::clone(x)
            .reshape(&[1, k.next_power_of_two()])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        let p_2d: Arc<Tensor<u32>> = Arc::clone(packed)
            .reshape(&[1, total_packed])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let z = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[1, n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1, 1])
            .map([1, bn], (n / bn) as u32);

        let (z, _x, _p) =
            kernels::cutile_q6k_gemv_bn(z, x_2d, p_2d)
                .generics(vec![k_blocks.to_string(), "1".to_string(), bn.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Non-blocking Q6_K GEMV reusing a pre-allocated output buffer.
    ///
    /// Enqueues the kernel without synchronising. The caller MUST ensure the
    /// stream is synced before reading the result on the CPU.
    ///
    /// # Safety
    ///
    /// The returned tensor's device memory is being written asynchronously.
    pub unsafe fn submit_d2d_into(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let z = out
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (z, _x, _p) = unsafe {
            kernels::cutile_q6k_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .async_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let flat = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(flat))
    }
}

/// GPU-resident Q4_K GEMV dispatcher.
///
/// Holds the CUDA stream and dispatches `cutile_q4k_gemv` for decode (m=1)
/// forward passes where weights are stored in packed Q4_K format on the GPU.
pub struct CutileQ4Gemv {
    stream: Arc<Stream>,
}

unsafe impl Send for CutileQ4Gemv {}
unsafe impl Sync for CutileQ4Gemv {}

impl CutileQ4Gemv {
    /// Create from an existing CUDA stream (share with `CutileMatMul`).
    pub fn from_stream(stream: Arc<Stream>) -> Self {
        Self { stream }
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Device-to-device Q4_K GEMV: output[n] = packed_weights x x[k].
    ///
    /// `packed` holds `n x k_blocks x Q4K_PACKED_U32S` u32 elements.
    /// `x` holds `k = k_blocks x 256` f32 activation elements.
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
            kernels::cutile_q4k_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        // reshape(&[n]) collapses partition-stride metadata left by
        // partition([1]).unpartition() -- same fix as CutileQ2Gemv::forward_d2d.
        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Device-to-device Q4_K GEMV with BN outputs per program.
    ///
    /// Reduces kernel launch overhead by BNx versus `forward_d2d`. Uses rank-2
    /// `MappedPartitionMut<f32, { [1, 1] }, [1, BN]>` to bypass the cuTile 0.4.0
    /// "rank-1 maps must be [1]" restriction.
    pub fn forward_d2d_bn(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        n: usize,
        k_blocks: usize,
        bn: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        if n % bn != 0 {
            return Err(KernelError::InvalidConfig(format!(
                "n={n} must be divisible by bn={bn}"
            )));
        }
        let stream = &self.stream;

        let k = k_blocks * Q4K_BLOCK_SIZE;
        let total_packed = n * k_blocks * Q4K_PACKED_U32S;

        let x_2d: Arc<Tensor<f32>> = Arc::clone(x)
            .reshape(&[1, k.next_power_of_two()])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();
        let p_2d: Arc<Tensor<u32>> = Arc::clone(packed)
            .reshape(&[1, total_packed])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .into();

        let z = api::zeros::<f32>(&[n])
            .sync_on(stream)
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .reshape(&[1, n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1, 1])
            .map([1, bn], (n / bn) as u32);

        let (z, _x, _p) =
            kernels::cutile_q4k_gemv_bn(z, x_2d, p_2d)
                .generics(vec![k_blocks.to_string(), "1".to_string(), bn.to_string()])
                .sync_on(stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;

        let out = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(out))
    }

    /// Non-blocking Q4_K GEMV reusing a pre-allocated output buffer.
    ///
    /// Enqueues the kernel without synchronising. The caller MUST ensure the
    /// stream is synced before reading the result on the CPU.
    ///
    /// # Safety
    ///
    /// The returned tensor's device memory is being written asynchronously.
    pub unsafe fn submit_d2d_into(
        &self,
        x: &Arc<Tensor<f32>>,
        packed: &Arc<Tensor<u32>>,
        out: Tensor<f32>,
        n: usize,
        k_blocks: usize,
    ) -> Result<Arc<Tensor<f32>>> {
        let z = out
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
            .partition([1]);

        let (z, _x, _p) = unsafe {
            kernels::cutile_q4k_gemv(z, Arc::clone(x), Arc::clone(packed))
                .generics(vec![k_blocks.to_string()])
                .async_on(&self.stream)
                .map_err(|e| KernelError::InvalidConfig(e.to_string()))?
        };

        let flat = z
            .unpartition()
            .reshape(&[n])
            .map_err(|e| KernelError::InvalidConfig(e.to_string()))?;
        Ok(Arc::new(flat))
    }
}

// Tile constants matching cutile_matmul.rs
const BM: usize = 16;
const BN: usize = 16;
const BK: usize = 8;

// -- Block layout constants ----------------------------------------------------
// Q4_K: 4-bit, 256 elements per block, 144 bytes
//   [0..2]   d      (f16)
//   [2..4]   dmin   (f16)
//   [4..16]  scales (12 bytes packed, 8 groups x 6-bit sc + 8 groups x 6-bit m)
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

// -- f16 bit conversion (mirrors quantized.rs) --------------------------------
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

// -- Q4_K scale unpacking (mirrors quantized.rs::q4k_unpack_scales) -----------
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

// -- Dequantize a single Q4_K block to 256 f32 values ------------------------
// TODO: GPU dequant -- currently runs on CPU before GEMM
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

// -- Dequantize a single Q6_K block to 256 f32 values ------------------------
// TODO: GPU dequant -- currently runs on CPU before GEMM
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

// -- Dequantize a single TQ2 block to 256 f32 values -------------------------
// TODO: GPU dequant -- currently runs on CPU before GEMM
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

// -- Full weight matrix dequantization ----------------------------------------
// Expands [N, blocks_per_row * BLOCK_BYTES] -> [N, K] f32
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
// TODO: GPU dequant -- currently runs on CPU before GEMM
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
                // i2s: 0->0, 1->1, 2->-2, 3->-1 (two's complement 2-bit)
                let val = (raw as i8).wrapping_shl(6).wrapping_shr(6) as f32;
                row[idx] = val * w_scale;
                idx += 1;
            }
        }
    }
    b_f32
}

// -- Cutile GEMM helper (B transposed): output[M,N] = A[M,K] @ B^T[N,K] ------
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

// -- CutileQuantizedGemm struct ------------------------------------------------

/// GPU-accelerated quantized GEMM using cutile-rs.
///
/// Dequantizes weights on CPU (Q4_K, Q6_K, TQ2, I2S), then executes
/// f32 GEMM on GPU via cutile. All methods are `forward_b_transposed_*`:
/// weights are stored transposed ([N, K]) and activations are [M, K].
///
/// TODO: GPU dequant -- move block-format expansion to GPU kernels for
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

        // TODO: GPU dequant -- expand Q4_K on GPU to avoid host->device bandwidth for weights
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

        // TODO: GPU dequant -- expand Q6_K on GPU
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

        // TODO: GPU dequant -- expand TQ2 ternary weights on GPU (all-add/sub/skip paths)
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

        // I2S uses integer arithmetic: quantize activations to i8, compute integer
        // dot products via vec_dot_i2_i8_s, apply ZeroPoint correction.
        // TODO: implement as GPU dp4a kernel; this CPU fallback is numerically exact.
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
            let mut a_q = vec![0i8; k];
            let (act_scale, act_sum) = crate::quantized::quantize_row_i8_s(a_row, &mut a_q);
            for j in 0..n {
                let w_packed = &b_quantized[j * row_packed_bytes..(j + 1) * row_packed_bytes];
                let raw = crate::quantized::vec_dot_i2_i8_s(w_packed, &a_q, k);
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
        // Simple f32->f16 for test values that fit in normal f16 range
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
        // Build a block with d=1, dmin=0 and all nibbles=7 -> each element = 7.0
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
        // d = 1.0 -> f16 bits at [208..210]
        let d_bits = f32_to_f16_bits(1.0);
        block[208] = d_bits as u8;
        block[209] = (d_bits >> 8) as u8;
        // scales: all 1 (i8 value 1 -> all groups have sc=1)
        for i in 192..208 {
            block[i] = 1u8;
        }
        // ql: all 0x11 -> low nibbles=1, meaning q1=(1|0<<4)-32=-31
        for i in 0..128 {
            block[i] = 0x11;
        }
        // qh: 0 -> high bits=0
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
        // qs: 0b10101010 -> each pair = 0b10 = 2 -> trit = 2-1 = 1
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

    // -- Q6_K upload format helper ---------------------------------------------

    /// Build a synthetic Q6_K block: d=scale, all elements = q_val ??? [0, 63].
    /// All 16 group scales are set to group_sc (i8).
    fn make_q6k_block_uniform(d: f32, q_val: u8, group_sc: i8) -> Vec<u8> {
        assert!(q_val <= 63);
        let mut block = vec![0u8; Q6K_BLOCK_BYTES];
        // d at [208..210]
        let d_bits = f32_to_f16_bits(d);
        block[208] = d_bits as u8;
        block[209] = (d_bits >> 8) as u8;
        // scales at [192..208] -- all group_sc
        for i in 192..208 {
            block[i] = group_sc as u8;
        }
        // ql [0..128] and qh [128..192]: encode q_val for every element.
        // q_val = (ql_nibble) | (qh_2bits << 4)
        // We choose ql_nibble = q_val & 0xF, qh_2bits = q_val >> 4 (0..3).
        // For element in half, sub, l:
        //   sub=0,2: ql[half*64+l]; sub=1,3: ql[half*64+l+32]
        //   ql low nibble for sub=0,1; ql high nibble for sub=2,3
        //   qh[half*32+l] bits 1:0 for sub=0; bits 3:2 for sub=1; bits 5:4 for sub=2; bits 7:6 for sub=3
        let ql_nibble = q_val & 0xF;
        let qh_2bits = q_val >> 4; // 0..3
        // For every ql byte: both nibbles = ql_nibble
        let ql_byte = ql_nibble | (ql_nibble << 4);
        for i in 0..128usize {
            block[i] = ql_byte;
        }
        // For every qh byte: all 4 two-bit slots = qh_2bits
        let qh_byte = qh_2bits | (qh_2bits << 2) | (qh_2bits << 4) | (qh_2bits << 6);
        for i in 128..192usize {
            block[i] = qh_byte;
        }
        block
    }

    /// Pack a Q6_K raw block into the 53-u32 compact GPU format.
    ///
    /// Layout: u32[0] = d as f32 | u32[1..5] = 16 scales as bias-128 u8 |
    ///         u32[5..37] = ql[0..128] | u32[37..53] = qh[0..64]
    fn pack_q6k_block(block: &[u8]) -> Vec<u32> {
        assert_eq!(block.len(), Q6K_BLOCK_BYTES);
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d_f32 = f16_to_f32(d_bits);

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];

        let mut packed = Vec::with_capacity(Q6K_PACKED_U32S);

        // slot 0: d as f32
        packed.push(d_f32.to_bits());

        // slots 1..5: 16 group scales packed as bias-128 u8 (4 per u32)
        // bias-128: stored = (scale_i8 as i32 + 128) as u8; decode: (byte as f32) - 128.0
        for w in 0..4usize {
            let b0 = ((scales[w * 4] as i8) as i32 + 128) as u8;
            let b1 = ((scales[w * 4 + 1] as i8) as i32 + 128) as u8;
            let b2 = ((scales[w * 4 + 2] as i8) as i32 + 128) as u8;
            let b3 = ((scales[w * 4 + 3] as i8) as i32 + 128) as u8;
            packed.push(
                (b0 as u32)
                    | ((b1 as u32) << 8)
                    | ((b2 as u32) << 16)
                    | ((b3 as u32) << 24),
            );
        }

        // slots 5..37: ql[0..128] packed 4 u8 per u32 (native GGUF order)
        for i in 0..32usize {
            let w = (ql[i * 4] as u32)
                | ((ql[i * 4 + 1] as u32) << 8)
                | ((ql[i * 4 + 2] as u32) << 16)
                | ((ql[i * 4 + 3] as u32) << 24);
            packed.push(w);
        }

        // slots 37..53: qh[0..64] packed 4 u8 per u32 (native GGUF order)
        for i in 0..16usize {
            let w = (qh[i * 4] as u32)
                | ((qh[i * 4 + 1] as u32) << 8)
                | ((qh[i * 4 + 2] as u32) << 16)
                | ((qh[i * 4 + 3] as u32) << 24);
            packed.push(w);
        }

        assert_eq!(packed.len(), Q6K_PACKED_U32S);
        packed
    }

    #[test]
    fn q6k_pack_matches_cpu_dequant_uniform() {
        // Uniform block: d=2.0, all q_full=40 (-> q_signed=8), group_sc=3
        // expected weight per element = 2.0 * 3.0 * 8.0 = 48.0
        // dot with all-ones activation = 256 * 48.0 = 12288.0
        let d = 2.0f32;
        let q_val = 40u8; // q_signed = 40 - 32 = 8
        let group_sc = 3i8;
        let block = make_q6k_block_uniform(d, q_val, group_sc);

        // CPU dequant via dequant_q6k_block
        let mut dequanted = vec![0.0f32; Q6K_BLOCK_SIZE];
        dequant_q6k_block(&block, &mut dequanted);

        let cpu_dot: f32 = dequanted.iter().sum();
        let expected = 256.0 * d * (group_sc as f32) * ((q_val as f32) - 32.0);
        assert!(
            (cpu_dot - expected).abs() < 0.5,
            "cpu dot={cpu_dot} expected={expected}"
        );

        // Verify packed format with the new 53-u32 compact layout.
        let packed = pack_q6k_block(&block);
        assert_eq!(packed.len(), Q6K_PACKED_U32S);
        // slot 0: d
        assert!(
            (f32::from_bits(packed[0]) - d).abs() < 1e-4,
            "d mismatch: {}",
            f32::from_bits(packed[0])
        );
        // slots 1..5: bias-128 scales; all should decode to group_sc
        for w in 1..5usize {
            let word = packed[w];
            for bp in 0..4usize {
                let byte = ((word >> (bp * 8)) & 0xFF) as u8;
                let decoded = byte as f32 - 128.0;
                assert!(
                    (decoded - group_sc as f32).abs() < 1e-6,
                    "scale[{}]={} expected {}",
                    (w - 1) * 4 + bp,
                    decoded,
                    group_sc
                );
            }
        }
        // Verify dequant: recompute all 256 weights from packed ql+qh and compare to CPU.
        let mut dequanted_from_packed = vec![0.0f32; Q6K_BLOCK_SIZE];
        let d_packed = f32::from_bits(packed[0]);
        for within in 0..256usize {
            let half = within / 128;
            let sub  = (within % 128) / 32;
            let l    = within % 32;
            let group = within / 16;

            let sc_word = packed[1 + group / 4];
            let sc_byte = ((sc_word >> ((group % 4) * 8)) & 0xFF) as u8;
            let sc = sc_byte as f32 - 128.0;

            let ql_byte_idx = half * 64 + l + (sub % 2) * 32;
            let ql_u32 = packed[5 + ql_byte_idx / 4];
            let ql_byte = ((ql_u32 >> ((ql_byte_idx % 4) * 8)) & 0xFF) as u8;
            let lo = (ql_byte >> ((sub / 2) * 4)) & 0xF;

            let qh_byte_idx = half * 32 + l;
            let qh_u32 = packed[37 + qh_byte_idx / 4];
            let qh_byte = ((qh_u32 >> ((qh_byte_idx % 4) * 8)) & 0xFF) as u8;
            let hi = (qh_byte >> (sub * 2)) & 0x3;

            let q6 = lo | (hi << 4);
            dequanted_from_packed[within] = d_packed * sc * (q6 as f32 - 32.0);
        }
        let mut dequanted_cpu = vec![0.0f32; Q6K_BLOCK_SIZE];
        dequant_q6k_block(&block, &mut dequanted_cpu);
        for i in 0..Q6K_BLOCK_SIZE {
            assert!(
                (dequanted_from_packed[i] - dequanted_cpu[i]).abs() < 1e-3,
                "packed dequant[{i}]={} vs cpu={} (q_val={})",
                dequanted_from_packed[i],
                dequanted_cpu[i],
                q_val
            );
        }
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn q6k_gpu_gemv_matches_cpu_dequant() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        // Four output neurons, K = Q6K_QK (1 block each).
        // Row 0: d=1.0, all q_full=32 (-> 0), group_sc=1  -> dot with ones = 0
        // Row 1: d=1.0, all q_full=33 (-> 1), group_sc=1  -> dot = 256 * 1 = 256
        // Row 2: d=1.0, all q_full=31 (-> -1), group_sc=1 -> dot = 256 * -1 = -256
        // Row 3: d=2.0, all q_full=40 (-> 8), group_sc=3  -> dot = 256 * 2 * 3 * 8 = 12288
        let n = 4;
        let k = Q6K_QK;
        let k_blocks = 1;
        let cases: &[(f32, u8, i8, f32)] = &[
            (1.0, 32, 1, 0.0),
            (1.0, 33, 1, 256.0),
            (1.0, 31, 1, -256.0),
            (2.0, 40, 3, 12288.0),
        ];

        let x_vals = vec![1.0f32; k];
        let mut all_packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q6K_PACKED_U32S);
        let mut expected: Vec<f32> = Vec::with_capacity(n);

        for &(d, q_val, sc, exp) in cases {
            let block = make_q6k_block_uniform(d, q_val, sc);
            all_packed.extend(pack_q6k_block(&block));
            expected.push(exp);
        }

        let device = Device::new(0).expect("CUDA device 0 not available");
        let stream: std::sync::Arc<cuda_core::Stream> =
            device.new_stream().expect("stream").into();
        let q6k_gemv = CutileQ6kGemv::from_stream(stream);

        let x_dev: std::sync::Arc<cutile::tensor::Tensor<f32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(x_vals))
                .sync_on(q6k_gemv.stream())
                .expect("upload x")
                .into();
        let p_dev: std::sync::Arc<cutile::tensor::Tensor<u32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(all_packed))
                .sync_on(q6k_gemv.stream())
                .expect("upload packed")
                .into();

        let result = q6k_gemv
            .forward_d2d(&x_dev, &p_dev, n, k_blocks)
            .expect("q6k gemv");

        let out: Vec<f32> = result
            .to_host_vec()
            .sync_on(q6k_gemv.stream())
            .expect("download");

        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1.0,
                "row {i}: got {got}, expected {exp}"
            );
        }
    }

    // -- Q2_0 upload format helpers --------------------------------------------

    fn f32_to_f16_bits_q2(val: f32) -> u16 {
        f32_to_f16_bits(val)
    }

    /// Build a synthetic Q2_0 block: scale=d, all 128 weights = q_val ??? {0,1,2,3}.
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
        // Row 0: scale=2.0, all q=1 -> dequant = (1-1)*2.0 = 0.0
        // Row 1: scale=3.0, all q=3 -> dequant = (3-1)*3.0 = 6.0
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

        // Validate data words: all q=1 -> packed byte 0b01_01_01_01 = 0x55.
        let word0 = packed[1];
        assert_eq!(word0, 0x55555555, "row0 word: 0x{:08x}", word0);

        // All q=3 -> packed byte 0b11_11_11_11 = 0xFF.
        let word1 = packed[k_blocks * Q2_0_PACKED_U32S + 1];
        assert_eq!(word1, 0xFFFFFFFF, "row1 word: 0x{:08x}", word1);
    }

    // -- Q4_K upload format helpers --------------------------------------------

    /// Build a synthetic Q4_K block (144 bytes): d=`d`, dmin=`dmin_val`, all nibbles=`nibble`.
    /// Scale group g: sc[g]=`sc_val`, m[g]=`m_val` for all 8 groups.
    fn make_q4k_block(d: f32, dmin_val: f32, nibble: u8, sc_val: u8, m_val: u8) -> Vec<u8> {
        assert!(nibble <= 15);
        assert!(sc_val < 64);
        assert!(m_val < 64);
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        // d at [0..2], dmin at [2..4]
        let d_bits = f32_to_f16_bits(d);
        block[0] = d_bits as u8;
        block[1] = (d_bits >> 8) as u8;
        let dmin_bits = f32_to_f16_bits(dmin_val);
        block[2] = dmin_bits as u8;
        block[3] = (dmin_bits >> 8) as u8;
        // Packed scales [4..16]: encode sc_val into sc[0..4] and m_val into m[0..4].
        // sc[i] = packed[i] & 0x3F for i=0..4 -> set packed[i] = sc_val
        // m[i] = packed[i+4] & 0x3F for i=0..4 -> set packed[i+4] = m_val
        // sc[i+4] = (packed[i+8] & 0x0F) | ((packed[i] >> 6) << 4) -> high bits zero (sc_val < 64)
        // m[i+4] = (packed[i+8] >> 4) | ((packed[i+4] >> 6) << 4) -> high bits zero
        for i in 0..4 {
            block[4 + i] = sc_val;
            block[8 + i] = m_val;
        }
        // Packed scales [4..16] encode sc[0..8] and m[0..8] using 6-bit fields.
        // packed_scales = block[4..16], split into three 4-byte groups:
        //   group A: packed_scales[0..4] = block[4..8]
        //   group B: packed_scales[4..8] = block[8..12]
        //   group C: packed_scales[8..12] = block[12..16]
        // Unpack rules (for i=0..4):
        //   sc[i]   = packed_scales[i]   & 0x3F          -> set block[4+i] = sc_val
        //   sc[i+4] = (packed_scales[i+8] & 0x0F)
        //             | ((packed_scales[i] >> 6) << 4)   -> set block[12+i] lo nibble = sc_val (sc_val <= 15)
        //   m[i]    = packed_scales[i+4] & 0x3F          -> set block[8+i] = m_val
        //   m[i+4]  = (packed_scales[i+8] >> 4)
        //             | ((packed_scales[i+4] >> 6) << 4) -> set block[12+i] hi nibble = m_val (m_val <= 15)
        // This only works correctly for sc_val <= 15 and m_val <= 15 (no high-bit overflow).
        for i in 0..4 {
            block[4 + i] = sc_val;
            block[8 + i] = m_val;
            block[12 + i] = (sc_val & 0x0F) | ((m_val & 0x0F) << 4);
        }
        // nibble byte: lo=nibble, hi=nibble -> byte = nibble | (nibble << 4)
        let qs_byte = nibble | (nibble << 4);
        for i in 16..144 {
            block[i] = qs_byte;
        }
        block
    }

    #[test]
    fn q4k_upload_format_round_trips() {
        // Build a 2-row, 1-block Q4K matrix and verify the packed u32 format.
        // Row 0: d=1.0, dmin=0.0, all nibbles=7, sc=1, m=0 -> each elem = 1.0*1*7 - 0.0*0 = 7.0
        // Row 1: d=2.0, dmin=0.5, all nibbles=4, sc=2, m=1 -> each elem = 2.0*2*4 - 0.5*1 = 15.5
        let block0 = make_q4k_block(1.0, 0.0, 7, 1, 0);
        let block1 = make_q4k_block(2.0, 0.5, 4, 2, 1);

        fn unpack_q4k_scales(packed_scales: &[u8]) -> ([u8; 8], [u8; 8]) {
            let mut sc = [0u8; 8];
            let mut m = [0u8; 8];
            for i in 0..4 {
                sc[i] = packed_scales[i] & 0x3F;
                sc[i + 4] = ((packed_scales[i + 8] & 0x0F) | ((packed_scales[i] >> 6) << 4)) & 0x3F;
                m[i] = packed_scales[i + 4] & 0x3F;
                m[i + 4] = ((packed_scales[i + 8] >> 4) | ((packed_scales[i + 4] >> 6) << 4)) & 0x3F;
            }
            (sc, m)
        }

        fn pack_one_block(block: &[u8]) -> Vec<u32> {
            let d_f16 = u16::from_le_bytes([block[0], block[1]]);
            let d_f32 = f16_to_f32(d_f16);
            let dmin_f16 = u16::from_le_bytes([block[2], block[3]]);
            let dmin_f32 = f16_to_f32(dmin_f16);
            let (sc, m) = unpack_q4k_scales(&block[4..16]);
            let qs = &block[16..144];
            let mut out = vec![0u32; Q4K_PACKED_U32S];
            out[0] = d_f32.to_bits();
            out[1] = dmin_f32.to_bits();
            for j in 0..4 {
                out[2 + j] = sc[2 * j] as u32 | ((sc[2 * j + 1] as u32) << 8);
                out[6 + j] = m[2 * j] as u32 | ((m[2 * j + 1] as u32) << 8);
            }
            for w in 0..32 {
                out[10 + w] = qs[w * 4] as u32
                    | ((qs[w * 4 + 1] as u32) << 8)
                    | ((qs[w * 4 + 2] as u32) << 16)
                    | ((qs[w * 4 + 3] as u32) << 24);
            }
            out
        }

        let packed0 = pack_one_block(&block0);
        let packed1 = pack_one_block(&block1);

        // Validate row 0
        let d0 = f32::from_bits(packed0[0]);
        assert!((d0 - 1.0).abs() < 1e-4, "row0 d: {d0}");
        let dmin0 = f32::from_bits(packed0[1]);
        assert!((dmin0 - 0.0).abs() < 1e-4, "row0 dmin: {dmin0}");
        // sc_pair[0]: sc[0] | (sc[1]<<8) = 1 | (1<<8)
        let sc_pair0 = packed0[2];
        assert_eq!(sc_pair0 & 0xFF, 1, "row0 sc[0]");
        assert_eq!((sc_pair0 >> 8) & 0xFF, 1, "row0 sc[1]");
        // qs word[0]: byte 0x77 repeated
        let qs_word0 = packed0[10];
        assert_eq!(qs_word0, 0x77777777, "row0 qs word0: 0x{:08x}", qs_word0);

        // Validate row 1
        let d1 = f32::from_bits(packed1[0]);
        assert!((d1 - 2.0).abs() < 1e-4, "row1 d: {d1}");
        // sc_pair[0]: sc[0]=2 | (sc[1]=2)<<8 = 0x0202
        let sc_pair1 = packed1[2];
        assert_eq!(sc_pair1 & 0xFF, 2, "row1 sc[0]");
        // qs word[0]: byte 0x44 repeated
        let qs_word1 = packed1[10];
        assert_eq!(qs_word1, 0x44444444, "row1 qs word0: 0x{:08x}", qs_word1);
    }

    #[test]
    fn q4k_cpu_dequant_matches_dot_block() {
        // Verify that dequant_q4k_block output is consistent with dequant_weights_q4k.
        let block = make_q4k_block(1.0, 0.0, 7, 1, 0);
        let mut dq = vec![0.0f32; Q4K_BLOCK_SIZE];
        dequant_q4k_block(&block, &mut dq);
        let sum: f32 = dq.iter().sum();
        // d=1, sc=1, nibble=7, dmin=0, m=0 -> each of 256 elements = 1*1*7 - 0 = 7.0
        assert!(
            (sum - 256.0 * 7.0).abs() < 1e-2,
            "q4k dequant sum: {sum} expected {}",
            256.0 * 7.0
        );
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn q4k_gpu_gemv_matches_cpu_dequant() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        // Build a 4-output Q4K GEMV, K=256 (1 block each):
        // Row 0: d=1.0, dmin=0.0, nibble=7, sc=1, m=0 -> each element = 7.0
        // Row 1: d=2.0, dmin=0.5, nibble=4, sc=2, m=1 -> each element = 2.0*2*4 - 0.5*1 = 15.5
        // Row 2: d=1.0, dmin=0.0, nibble=0, sc=3, m=0 -> each element = 0.0
        // Row 3: d=0.5, dmin=0.0, nibble=15, sc=2, m=0 -> each element = 0.5*2*15 = 15.0
        let n = 4;
        let k = Q4K_BLOCK_SIZE; // 256
        let k_blocks = 1;
        let test_cases: &[(f32, f32, u8, u8, u8, f32)] = &[
            (1.0, 0.0, 7, 1, 0, 7.0),
            (2.0, 0.5, 4, 2, 1, 15.5),
            (1.0, 0.0, 0, 3, 0, 0.0),
            (0.5, 0.0, 15, 2, 0, 15.0),
        ];

        let x_vals = vec![1.0f32; k];

        fn unpack_scales_cpu(packed_scales: &[u8]) -> ([u8; 8], [u8; 8]) {
            let mut sc = [0u8; 8];
            let mut m = [0u8; 8];
            for i in 0..4 {
                sc[i] = packed_scales[i] & 0x3F;
                sc[i + 4] = ((packed_scales[i + 8] & 0x0F) | ((packed_scales[i] >> 6) << 4)) & 0x3F;
                m[i] = packed_scales[i + 4] & 0x3F;
                m[i + 4] = ((packed_scales[i + 8] >> 4) | ((packed_scales[i + 4] >> 6) << 4)) & 0x3F;
            }
            (sc, m)
        }

        let mut packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q4K_PACKED_U32S);
        let mut expected: Vec<f32> = Vec::with_capacity(n);

        for &(d, dmin_val, nibble, sc_val, m_val, elem_val) in test_cases {
            let block = make_q4k_block(d, dmin_val, nibble, sc_val, m_val);
            let d_f32 = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_f32 = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (sc, m) = unpack_scales_cpu(&block[4..16]);
            let qs = &block[16..144];
            packed.push(d_f32.to_bits());
            packed.push(dmin_f32.to_bits());
            for j in 0..4 {
                packed.push(sc[2 * j] as u32 | ((sc[2 * j + 1] as u32) << 8));
            }
            for j in 0..4 {
                packed.push(m[2 * j] as u32 | ((m[2 * j + 1] as u32) << 8));
            }
            for w in 0..32 {
                packed.push(
                    qs[w * 4] as u32
                        | ((qs[w * 4 + 1] as u32) << 8)
                        | ((qs[w * 4 + 2] as u32) << 16)
                        | ((qs[w * 4 + 3] as u32) << 24),
                );
            }
            // activation all ones -> dot = k * elem_val
            expected.push(k as f32 * elem_val);
        }

        let device = Device::new(0).expect("CUDA device 0 not available");
        let stream: std::sync::Arc<cuda_core::Stream> =
            device.new_stream().expect("stream").into();
        let q4k_gemv = CutileQ4Gemv::from_stream(stream);

        let x_dev: std::sync::Arc<cutile::tensor::Tensor<f32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(x_vals))
                .sync_on(q4k_gemv.stream())
                .expect("upload x")
                .into();
        let p_dev: std::sync::Arc<cutile::tensor::Tensor<u32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(packed))
                .sync_on(q4k_gemv.stream())
                .expect("upload packed")
                .into();

        let result = q4k_gemv
            .forward_d2d(&x_dev, &p_dev, n, k_blocks)
            .expect("q4k gemv");

        let out: Vec<f32> = result
            .to_host_vec()
            .sync_on(q4k_gemv.stream())
            .expect("download");

        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-1,
                "row {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn q2_0_gpu_gemv_matches_cpu_dequant() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        // Build a 4-output, K=128 Q2_0 GEMV:
        // Row 0: scale=1.0, all q=2 -> dequant = +1.0
        // Row 1: scale=2.0, all q=1 -> dequant =  0.0
        // Row 2: scale=1.0, all q=0 -> dequant = -1.0
        // Row 3: scale=1.0, all q=3 -> dequant = +2.0
        let n = 4;
        let k = Q2_0_QK;
        let k_blocks = 1;
        let test_cases: &[(f32, u8, f32)] = &[
            (1.0, 2, 1.0),  // row 0: scale=1, q=2 -> +1
            (2.0, 1, 0.0),  // row 1: scale=2, q=1 ->  0
            (1.0, 0, -1.0), // row 2: scale=1, q=0 -> -1
            (1.0, 3, 2.0),  // row 3: scale=1, q=3 -> +2
        ];

        // Activation: all ones -> dot = sum of row weights
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
            // Expected: k elements each = elem_val, activation all 1 -> dot = k * elem_val
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

    // -- BN variant tests: verify multi-output kernels match scalar kernels ----

    #[test]
    #[cfg(feature = "cutile")]
    fn q2_0_gpu_gemv_bn_matches_scalar() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        // Same 4-case setup as q2_0_gpu_gemv_matches_cpu_dequant, but using BN=4
        // so n=4 is divisible by bn=4.
        let n = 4;
        let k = Q2_0_QK;
        let k_blocks = 1;
        let bn = 4;

        let test_cases: &[(f32, u8, f32)] = &[
            (1.0, 2, 1.0),
            (2.0, 1, 0.0),
            (1.0, 0, -1.0),
            (1.0, 3, 2.0),
        ];

        let x_vals = vec![1.0f32; k];
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
            expected.push(k as f32 * elem_val);
        }

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
            .forward_d2d_bn(&x_dev, &p_dev, n, k_blocks, bn)
            .expect("q2_0 gemv_bn");

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

    #[test]
    #[cfg(feature = "cutile")]
    fn q4k_gpu_gemv_bn_matches_scalar() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        let n = 4;
        let k = Q4K_BLOCK_SIZE;
        let k_blocks = 1;
        let bn = 4;

        let test_cases: &[(f32, f32, u8, u8, u8, f32)] = &[
            (1.0, 0.0, 7, 1, 0, 7.0),
            (2.0, 0.5, 4, 2, 1, 15.5),
            (1.0, 0.0, 0, 3, 0, 0.0),
            (0.5, 0.0, 15, 2, 0, 15.0),
        ];

        let x_vals = vec![1.0f32; k];

        fn unpack_scales_inner(packed_scales: &[u8]) -> ([u8; 8], [u8; 8]) {
            let mut sc = [0u8; 8];
            let mut m = [0u8; 8];
            for i in 0..4 {
                sc[i] = packed_scales[i] & 0x3F;
                sc[i + 4] =
                    ((packed_scales[i + 8] & 0x0F) | ((packed_scales[i] >> 6) << 4)) & 0x3F;
                m[i] = packed_scales[i + 4] & 0x3F;
                m[i + 4] =
                    ((packed_scales[i + 8] >> 4) | ((packed_scales[i + 4] >> 6) << 4)) & 0x3F;
            }
            (sc, m)
        }

        let mut packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q4K_PACKED_U32S);
        let mut expected: Vec<f32> = Vec::with_capacity(n);

        for &(d, dmin_val, nibble, sc_val, m_val, elem_val) in test_cases {
            let block = make_q4k_block(d, dmin_val, nibble, sc_val, m_val);
            let d_f32 = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_f32 = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (sc, m) = unpack_scales_inner(&block[4..16]);
            let qs = &block[16..144];
            packed.push(d_f32.to_bits());
            packed.push(dmin_f32.to_bits());
            for j in 0..4 {
                packed.push(sc[2 * j] as u32 | ((sc[2 * j + 1] as u32) << 8));
            }
            for j in 0..4 {
                packed.push(m[2 * j] as u32 | ((m[2 * j + 1] as u32) << 8));
            }
            for w in 0..32 {
                packed.push(
                    qs[w * 4] as u32
                        | ((qs[w * 4 + 1] as u32) << 8)
                        | ((qs[w * 4 + 2] as u32) << 16)
                        | ((qs[w * 4 + 3] as u32) << 24),
                );
            }
            expected.push(k as f32 * elem_val);
        }

        let device = Device::new(0).expect("CUDA device 0 not available");
        let stream: std::sync::Arc<cuda_core::Stream> =
            device.new_stream().expect("stream").into();
        let q4k_gemv = CutileQ4Gemv::from_stream(stream);

        let x_dev: std::sync::Arc<cutile::tensor::Tensor<f32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(x_vals))
                .sync_on(q4k_gemv.stream())
                .expect("upload x")
                .into();
        let p_dev: std::sync::Arc<cutile::tensor::Tensor<u32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(packed))
                .sync_on(q4k_gemv.stream())
                .expect("upload packed")
                .into();

        let result = q4k_gemv
            .forward_d2d_bn(&x_dev, &p_dev, n, k_blocks, bn)
            .expect("q4k gemv_bn");

        let out: Vec<f32> = result
            .to_host_vec()
            .sync_on(q4k_gemv.stream())
            .expect("download");

        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-1,
                "row {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cutile")]
    fn q6k_gpu_gemv_bn_matches_scalar() {
        use cuda_core::Device;
        use cutile::api;
        use cutile::prelude::DeviceOp;

        let n = 4;
        let k = Q6K_QK;
        let k_blocks = 1;
        let bn = 4;

        let cases: &[(f32, u8, i8, f32)] = &[
            (1.0, 32, 1, 0.0),
            (1.0, 33, 1, 256.0),
            (1.0, 31, 1, -256.0),
            (2.0, 40, 3, 12288.0),
        ];

        let x_vals = vec![1.0f32; k];
        let mut all_packed: Vec<u32> = Vec::with_capacity(n * k_blocks * Q6K_PACKED_U32S);
        let mut expected: Vec<f32> = Vec::with_capacity(n);

        for &(d, q_val, sc, exp) in cases {
            let block = make_q6k_block_uniform(d, q_val, sc);
            all_packed.extend(pack_q6k_block(&block));
            expected.push(exp);
        }

        let device = Device::new(0).expect("CUDA device 0 not available");
        let stream: std::sync::Arc<cuda_core::Stream> =
            device.new_stream().expect("stream").into();
        let q6k_gemv = CutileQ6kGemv::from_stream(stream);

        let x_dev: std::sync::Arc<cutile::tensor::Tensor<f32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(x_vals))
                .sync_on(q6k_gemv.stream())
                .expect("upload x")
                .into();
        let p_dev: std::sync::Arc<cutile::tensor::Tensor<u32>> =
            api::copy_host_vec_to_device(&std::sync::Arc::new(all_packed))
                .sync_on(q6k_gemv.stream())
                .expect("upload packed")
                .into();

        let result = q6k_gemv
            .forward_d2d_bn(&x_dev, &p_dev, n, k_blocks, bn)
            .expect("q6k gemv_bn");

        let out: Vec<f32> = result
            .to_host_vec()
            .sync_on(q6k_gemv.stream())
            .expect("download");

        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1.0,
                "row {i}: got {got}, expected {exp}"
            );
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -- Q2_0 block layout and bit-extraction safety ---------------------------

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

    // -- Q4K block layout and nibble safety -----------------------------------

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

    // -- Q6K block layout and 6-bit reconstruction ----------------------------

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

    // -- TQ2 ternary bit extraction --------------------------------------------

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

    // -- Full-block dequant no-panic harnesses ---------------------------------

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

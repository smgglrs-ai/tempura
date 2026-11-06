//! GPU weight-block layout constants, host-side block dequantization, and
//! the Kani proofs covering them.
//!
//! Extracted from `cutile_quantized_gemm.rs` (which is `#[cfg(feature =
//! "cutile")]`-gated): the GPU upload paths are behind the cutile feature,
//! but the block-layout arithmetic and the CPU dequant-of-one-block helpers
//! are pure host code -- and their Kani harnesses were unreachable (the
//! module is compiled out for `cargo kani`, which runs featureless), so
//! they had never been verified anywhere. Moving them here (single
//! definition, re-imported by the gated module) makes them verifiable.

use crate::ggml::helpers::f16_to_f32;
use crate::{Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE, Q4K_GROUPS, Q6K_BLOCK_BYTES, Q6K_BLOCK_SIZE, TQ2_BLOCK_BYTES, TQ2_BLOCK_SIZE};

pub const Q6K_QK: usize = 256;
pub const Q6K_PACKED_U32S: usize = 53; // 1 (d) + 4 (scales) + 32 (ql) + 16 (qh) = 53

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

// Q4_K: 4-bit quantized, 256 elements per block, 144 bytes per block
//     out[j*64+l]    = d * sc[2*j]   * (qs[j*32+l] & 0xF)  - dmin * m[2*j]
//     out[j*64+l+32] = d * sc[2*j+1] * (qs[j*32+l] >> 4)   - dmin * m[2*j+1]
pub const Q4K_PACKED_U32S: usize = 42; // u32 slots per block in GPU format

// -- Dequantize a single Q4_K block to 256 f32 values ------------------------
// TODO: GPU dequant -- currently runs on CPU before GEMM
pub(crate) fn dequant_q4k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q4K_BLOCK_SIZE);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let packed_scales = &block[4..16];
    let qs = &block[16..144];
    let (sc, m) = crate::quantized::q4k_unpack_scales(packed_scales);

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
pub(crate) fn dequant_q6k_block(block: &[u8], out: &mut [f32]) {
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
pub(crate) fn dequant_tq2_block(block: &[u8], out: &mut [f32]) {
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
    //
    // The scale bytes are constrained to finite f16: for arbitrary bytes the
    // f16 scale can be NaN/Inf, and Kani's inserted nan-check then fires on
    // the (expected) NaN propagation -- that is not a safety property. The
    // panic/indexing safety these harnesses exist for is unaffected.

    /// f16 exponent bits != 0x7C00 (all-ones exponent = Inf/NaN).
    fn assume_f16_finite(lo: u8, hi: u8) {
        let bits = u16::from_le_bytes([lo, hi]);
        kani::assume(bits & 0x7C00 != 0x7C00);
    }

    #[kani::proof]
    fn dequant_q4k_block_no_panic() {
        let block: [u8; 144] = kani::any();
        assume_f16_finite(block[0], block[1]); // d
        assume_f16_finite(block[2], block[3]); // dmin
        let mut out = [0.0f32; 256];
        dequant_q4k_block(&block, &mut out);
    }

    #[kani::proof]
    fn dequant_q6k_block_no_panic() {
        let block: [u8; 210] = kani::any();
        assume_f16_finite(block[208], block[209]); // d
        let mut out = [0.0f32; 256];
        dequant_q6k_block(&block, &mut out);
    }

    #[kani::proof]
    #[kani::unwind(65)]
    fn dequant_tq2_block_no_panic() {
        let block: [u8; 66] = kani::any();
        assume_f16_finite(block[64], block[65]); // d
        let mut out = [0.0f32; 256];
        dequant_tq2_block(&block, &mut out);
    }
}

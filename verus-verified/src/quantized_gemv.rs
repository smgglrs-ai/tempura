//! Verus-verified Q2_0 GPU GEMV dequantization proofs -- unbounded
//!
//! Proves integer arithmetic, bit-manipulation, and block-layout properties
//! of the Q2_0 packed GPU format and the related Q4K/Q6K/TQ2 CPU dequant
//! functions in cutile_quantized_gemm.rs.
//!
//! Excluded from Verus (require f32/f64 or runtime alloc reasoning):
//!   - dequant_*_block_no_panic (runtime slice operations)
//!   - q2_0_ieee_trick_f32_value (requires IEEE 754 float semantics)
//!   - q2_0_gemv_accumulator_no_overflow (f32 accumulation bounds)

use vstd::prelude::*;

verus! {

// ============================================================================
// Constants -- mirror cutile_quantized_gemm.rs and kernel-api/src/lib.rs
// ============================================================================

pub spec const Q2_0_BLOCK_BYTES: nat = 34;
pub spec const Q2_0_PACKED_U32S: nat = 9;
pub spec const Q2_0_QK: nat = 128;

pub spec const Q4K_BLOCK_BYTES: nat = 144;
pub spec const Q4K_BLOCK_SIZE: nat = 256;

pub spec const Q6K_BLOCK_BYTES: nat = 210;
pub spec const Q6K_BLOCK_SIZE: nat = 256;

pub spec const TQ2_BLOCK_BYTES: nat = 66;
pub spec const TQ2_BLOCK_SIZE: nat = 256;

// ============================================================================
// Q2_0 -- block layout and bit-field proofs
// ============================================================================

/// Scale field (bytes 0-1) and data words (bytes 2-33) are within the 34-byte block.
proof fn q2_0_block_layout_in_bounds()
    ensures
        // scale: f16 at bytes 0..2
        2 <= Q2_0_BLOCK_BYTES,
        // data: 32 payload bytes at bytes 2..34
        2 + 32 <= Q2_0_BLOCK_BYTES,
        // total consistent
        Q2_0_BLOCK_BYTES == 34,
{
}

/// Upload loop byte index: `2 + w*4 + b` for w < 8, b < 4 stays within 34-byte block.
proof fn q2_0_upload_byte_index_in_bounds(w: nat, b: nat)
    requires
        w < 8,
        b < 4,
    ensures
        2 + w * 4 + b < Q2_0_BLOCK_BYTES,
{
    assert(2 + w * 4 + b < 34) by (nonlinear_arith)
        requires w < 8, b < 4;
}

/// Packed GPU slot index: scale at slot 0, data words at slots 1..=8.
/// `1 + word` < Q2_0_PACKED_U32S for word < 8.
proof fn q2_0_packed_slot_in_bounds(word: nat)
    requires word < 8,
    ensures 1 + word < Q2_0_PACKED_U32S,
{
}

/// Element index to block index and word index mapping stays in bounds.
/// For element index i < Q2_0_QK (128):
///   byte index = i / 4  ??? [0, 31] (within 32 payload bytes)
///   bit offset = (i % 4) * 2  ??? {0, 2, 4, 6} (valid shift for u8)
proof fn q2_0_element_index_to_byte_in_bounds(i: nat)
    requires i < Q2_0_QK,
    ensures
        i / 4 < 32,
        (i % 4) * 2 < 8,
{
    assert(i / 4 < 32) by (nonlinear_arith) requires i < 128;
    assert((i % 4) * 2 < 8) by (nonlinear_arith) requires i < 128;
}

/// GPU word index: element i < 128 lives in word `(i / 16)` (0..7) at bit_pair `(i % 16)`.
/// word < 8 and bit_pair < 16 so both slot access and shift are in bounds.
proof fn q2_0_word_and_bitpair_in_bounds(i: nat)
    requires i < Q2_0_QK,
    ensures
        i / 16 < 8,
        i % 16 < 16,
{
    assert(i / 16 < 8) by (nonlinear_arith) requires i < 128;
    assert(i % 16 < 16) by (nonlinear_arith) requires i < 128;
}

// ============================================================================
// Q2_0 -- 2-bit value centering arithmetic
// ============================================================================

/// Two-bit mask gives 0..=3; no u32 overflow.
proof fn q2_0_2bit_mask_range(word: u32, shift: u32)
    requires shift < 32,
    ensures ({
        let q: u32 = (word >> shift) & 3u32;
        q <= 3
    }),
{
    assert(((word >> shift) & 3u32) <= 3u32) by (bit_vector);
}

/// For shift = bit_pair * 2 with bit_pair < 16, shift < 32 so the >> is defined.
proof fn q2_0_shift_in_bounds(bit_pair: nat)
    requires bit_pair < 16,
    ensures bit_pair * 2 < 32,
{
    assert(bit_pair * 2 < 32) by (nonlinear_arith) requires bit_pair < 16;
}

/// Center subtraction: for q ??? {0,1,2,3}, (q as int) - 1 ??? {-1, 0, 1, 2}.
/// This proves the subtraction does not underflow int and produces a
/// bounded signed value -- the dequantized weight level.
proof fn q2_0_centering_in_range(q: u32)
    requires q <= 3,
    ensures ({
        let centered: int = q as int - 1int;
        &&& centered >= -1int
        &&& centered <= 2int
    }),
{
}

/// IEEE 754 trick used in the GPU kernel:
/// `q | 0x4B000000` for q ??? {0,1,2,3} has exponent field = 0x96 = 150 (decimal).
/// 150 ??? 255, so the u32 bitcast produces a normal finite f32 (not NaN or Inf).
proof fn q2_0_ieee_trick_exp_not_0xff(q: u32)
    requires q <= 3,
    ensures ({
        let biased: u32 = q | 0x4B000000u32;
        (biased >> 23u32) & 0xFFu32 != 0xFFu32
    }),
{
    assert((q | 0x4B000000u32) >> 23u32 == 0x96u32) by (bit_vector)
        requires q <= 3u32;
    assert(0x96u32 & 0xFFu32 != 0xFFu32) by (bit_vector);
}

// ============================================================================
// Q4K -- block layout and nibble safety
// ============================================================================

/// Q4K block field layout is internally consistent and within 144 bytes.
proof fn q4k_block_layout_in_bounds()
    ensures
        // d: bytes 0-1
        2 <= Q4K_BLOCK_BYTES,
        // dmin: bytes 2-3
        4 <= Q4K_BLOCK_BYTES,
        // scales: bytes 4-15 (12 bytes)
        16 <= Q4K_BLOCK_BYTES,
        // qs: bytes 16-143 (128 bytes)
        144 <= Q4K_BLOCK_BYTES,
        Q4K_BLOCK_BYTES == 144,
{
}

/// Nibble byte index: for element idx < 256, access byte `16 + idx/2` < 144.
proof fn q4k_qs_byte_index_in_bounds(idx: nat)
    requires idx < Q4K_BLOCK_SIZE,
    ensures 16 + idx / 2 < Q4K_BLOCK_BYTES,
{
    assert(16 + idx / 2 < 144) by (nonlinear_arith) requires idx < 256;
}

/// Nibble extraction from any byte: low nibble (& 0x0F) <= 15, high nibble (>> 4) <= 15.
proof fn q4k_nibble_range(byte: u8)
    ensures ({
        let lo: u8 = byte & 0x0Fu8;
        let hi: u8 = byte >> 4u8;
        &&& lo <= 15
        &&& hi <= 15
    }),
{
    assert(byte & 0x0Fu8 <= 15) by (bit_vector);
    assert(byte >> 4u8 <= 15) by (bit_vector);
}

/// Scale unpacking: `byte & 0x3F` <= 63 for any byte -- 6-bit scale value stays in range.
proof fn q4k_6bit_scale_in_range(byte: u8)
    ensures byte & 0x3Fu8 <= 63,
{
    assert(byte & 0x3Fu8 <= 63) by (bit_vector);
}

/// Scale unpack indices for group i < 4 stay within the 12-byte packed_scales slice.
proof fn q4k_scale_unpack_indices_in_bounds(i: nat)
    requires i < 4,
    ensures
        i < 12,
        i + 4 < 12,
        i + 8 < 12,
{
}

// ============================================================================
// Q6K -- block layout and 6-bit reconstruction
// ============================================================================

/// Q6K block field layout is within 210 bytes.
proof fn q6k_block_layout_in_bounds()
    ensures
        // ql: bytes 0-127 (128 bytes)
        128 <= Q6K_BLOCK_BYTES,
        // qh: bytes 128-191 (64 bytes)
        192 <= Q6K_BLOCK_BYTES,
        // scales: bytes 192-207 (16 bytes)
        208 <= Q6K_BLOCK_BYTES,
        // d: bytes 208-209
        210 <= Q6K_BLOCK_BYTES,
        Q6K_BLOCK_BYTES == 210,
{
}

/// Q6K 6-bit reconstruction: low 4 bits from ql + high 2 bits from qh <= 63.
///
/// The low 4 bits come from `ql & 0x0F` (range [0,15]) and the high 2 bits
/// from `(qh >> shift*2) & 0x03` (range [0,3]), combined as `low | (high << 4)`.
/// All operations are u8; the result occupies at most 6 bits, so it is <= 63.
proof fn q6k_6bit_reconstruction_in_range(ql: u8, qh: u8, shift: u32)
    requires shift < 4,
    ensures ({
        let shift_amount: u8 = (shift * 2) as u8;
        let low: u8 = ql & 0x0Fu8;
        let high: u8 = (qh >> shift_amount) & 0x03u8;
        let q6: u8 = low | (high << 4u8);
        q6 <= 63u8
    }),
{
    // shift * 2 < 8 so casting to u8 is safe
    assert((shift * 2u32) < 8u32) by (nonlinear_arith) requires shift < 4u32;

    assert(ql & 0x0Fu8 <= 15u8) by (bit_vector);

    // forall: (b >> s) & 0x03 <= 3 for s < 8
    assert(forall |b: u8, s: u8| s < 8u8 ==>
        #[trigger] ((b >> s) & 0x03u8) <= 3u8) by (bit_vector);

    // forall: lo | (hi << 4) <= 63 when lo <= 15 and hi <= 3
    assert(forall |lo: u8, hi: u8| lo <= 15u8 && hi <= 3u8 ==>
        #[trigger] (lo | (hi << 4u8)) <= 63u8) by (bit_vector);
}

/// Signed Q6K value: q6 - 32 for q6 ??? [0, 63] is in [-32, 31].
proof fn q6k_signed_value_in_range(q6: u32)
    requires q6 <= 63,
    ensures ({
        let q_signed: int = q6 as int - 32int;
        &&& q_signed >= -32int
        &&& q_signed <= 31int
    }),
{
}

// ============================================================================
// TQ2 -- ternary bit extraction
// ============================================================================

/// TQ2 block field layout: qs at bytes 0-63, scale at bytes 64-65.
proof fn tq2_block_layout_in_bounds()
    ensures
        64 <= TQ2_BLOCK_BYTES,
        66 <= TQ2_BLOCK_BYTES,
        TQ2_BLOCK_BYTES == 66,
{
}

/// TQ2 2-bit trit extraction: (byte >> shift*2) & 3 ??? {0,1,2,3}.
proof fn tq2_trit_extraction_range(byte: u8, shift: u32)
    requires shift < 4,
    ensures ({
        let trit: u32 = ((byte >> (shift * 2u32)) & 0x03u8) as u32;
        trit <= 3
    }),
{
    assert(((byte >> (shift * 2u32)) & 0x03u8) <= 3) by (bit_vector)
        requires shift < 4u32;
}

/// TQ2 centering: (trit as int) - 1 ??? {-1, 0, 1, 2} for trit in 0..=3.
proof fn tq2_centering_in_range(trit: u32)
    requires trit <= 3,
    ensures ({
        let centered: int = trit as int - 1int;
        &&& centered >= -1int
        &&& centered <= 2int
    }),
{
}

/// TQ2 byte index: for element idx < 256, the byte index `idx / 4` < 64.
proof fn tq2_byte_index_in_bounds(idx: nat)
    requires idx < TQ2_BLOCK_SIZE,
    ensures idx / 4 < 64,
{
    assert(idx / 4 < 64) by (nonlinear_arith) requires idx < 256;
}

fn main() {}

} // verus!

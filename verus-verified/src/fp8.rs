//! Verus-verified FP8 E4M3 encoding properties — unbounded proofs
//!
//! Proves: sign bit isolation, exponent/mantissa field widths,
//! encoded value fits in u8, max finite value representation.

use vstd::prelude::*;

verus! {

// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits
// bias = 7, max_exp = 14 (exp=15 reserved), max val = 2^8 * 1.75 = 448

// ============================================================================
// Proof: sign bit extraction
// ============================================================================

proof fn fp8_sign_bit(byte: u8)
    ensures ({
        let sign = byte >> 7u8;
        sign <= 1
    }),
{
    assert(byte >> 7u8 <= 1u8) by (bit_vector);
}

// ============================================================================
// Proof: exponent field is 4 bits (0..15)
// ============================================================================

proof fn fp8_exponent_field(byte: u8)
    ensures ({
        let exp = (byte >> 3u8) & 0x0Fu8;
        exp <= 15
    }),
{
    assert((byte >> 3u8) & 0x0Fu8 <= 15u8) by (bit_vector);
}

// ============================================================================
// Proof: mantissa field is 3 bits (0..7)
// ============================================================================

proof fn fp8_mantissa_field(byte: u8)
    ensures ({
        let mant = byte & 0x07u8;
        mant <= 7
    }),
{
    assert(byte & 0x07u8 <= 7u8) by (bit_vector);
}

// ============================================================================
// Proof: sign | exponent | mantissa reconstructs the byte
// ============================================================================

proof fn fp8_fields_reconstruct(byte: u8)
    ensures ({
        let sign = byte & 0x80u8;
        let exp = byte & 0x78u8;
        let mant = byte & 0x07u8;
        sign | exp | mant == byte
    }),
{
    assert((byte & 0x80u8) | (byte & 0x78u8) | (byte & 0x07u8) == byte) by (bit_vector);
}

// ============================================================================
// Proof: max finite E4M3 encoding = 0x7E (sign=0, exp=1111, mant=110)
// ============================================================================

proof fn fp8_max_finite_encoding()
    ensures ({
        let max_byte: u8 = 0x7Eu8;
        let sign = max_byte >> 7u8;
        let exp = (max_byte >> 3u8) & 0x0Fu8;
        let mant = max_byte & 0x07u8;
        &&& sign == 0
        &&& exp == 15
        &&& mant == 6
    }),
{
    assert(0x7Eu8 >> 7u8 == 0u8) by (bit_vector);
    assert((0x7Eu8 >> 3u8) & 0x0Fu8 == 15u8) by (bit_vector);
    assert(0x7Eu8 & 0x07u8 == 6u8) by (bit_vector);
}

// ============================================================================
// Proof: NaN encoding = 0x7F (exp=15, mant=7) — all bits set except sign
// ============================================================================

proof fn fp8_nan_encoding()
    ensures ({
        let nan_byte: u8 = 0x7Fu8;
        let exp = (nan_byte >> 3u8) & 0x0Fu8;
        let mant = nan_byte & 0x07u8;
        &&& exp == 15
        &&& mant == 7
    }),
{
    assert((0x7Fu8 >> 3u8) & 0x0Fu8 == 15u8) by (bit_vector);
    assert(0x7Fu8 & 0x07u8 == 7u8) by (bit_vector);
}

// ============================================================================
// Proof: zero encoding (positive and negative)
// ============================================================================

proof fn fp8_zero_encoding()
    ensures
        0x00u8 & 0x7Fu8 == 0,
        0x80u8 & 0x7Fu8 == 0,
{
    assert(0x00u8 & 0x7Fu8 == 0u8) by (bit_vector);
    assert(0x80u8 & 0x7Fu8 == 0u8) by (bit_vector);
}

fn main() {}

} // verus!

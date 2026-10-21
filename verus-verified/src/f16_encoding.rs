//! Verus-verified f16 (half precision) encoding properties -- unbounded proofs
//!
//! Proves: field extraction, exponent bias conversion, special value encoding
//! (zero, inf, NaN), and bit field widths for IEEE 754 half precision.

use vstd::prelude::*;

verus! {

// f16: 1 sign + 5 exponent + 10 fraction = 16 bits, bias = 15

// ============================================================================
// f16 sign extraction
// ============================================================================

proof fn f16_sign_bit(bits: u16)
    ensures ({
        let sign = (bits >> 15u16) & 1u16;
        sign <= 1
    }),
{
    assert((bits >> 15u16) & 1u16 <= 1u16) by (bit_vector);
}

// ============================================================================
// f16 exponent field: 5 bits (0..31)
// ============================================================================

proof fn f16_exponent_field(bits: u16)
    ensures ({
        let exp = (bits >> 10u16) & 0x1Fu16;
        exp <= 31
    }),
{
    assert((bits >> 10u16) & 0x1Fu16 <= 31u16) by (bit_vector);
}

// ============================================================================
// f16 fraction field: 10 bits (0..1023)
// ============================================================================

proof fn f16_fraction_field(bits: u16)
    ensures ({
        let frac = bits & 0x3FFu16;
        frac <= 1023
    }),
{
    assert(bits & 0x3FFu16 <= 1023u16) by (bit_vector);
}

// ============================================================================
// f16 zero: exp==0, frac==0
// ============================================================================

proof fn f16_zero_encoding()
    ensures ({
        let pos_zero: u16 = 0x0000u16;
        let neg_zero: u16 = 0x8000u16;
        &&& (pos_zero >> 10u16) & 0x1Fu16 == 0
        &&& pos_zero & 0x3FFu16 == 0
        &&& (neg_zero >> 10u16) & 0x1Fu16 == 0
        &&& neg_zero & 0x3FFu16 == 0
    }),
{
    assert((0x0000u16 >> 10u16) & 0x1Fu16 == 0u16) by (bit_vector);
    assert(0x0000u16 & 0x3FFu16 == 0u16) by (bit_vector);
    assert((0x8000u16 >> 10u16) & 0x1Fu16 == 0u16) by (bit_vector);
    assert(0x8000u16 & 0x3FFu16 == 0u16) by (bit_vector);
}

// ============================================================================
// f16 infinity: exp==31, frac==0
// ============================================================================

proof fn f16_infinity_encoding()
    ensures ({
        let pos_inf: u16 = 0x7C00u16;
        let neg_inf: u16 = 0xFC00u16;
        &&& (pos_inf >> 10u16) & 0x1Fu16 == 31
        &&& pos_inf & 0x3FFu16 == 0
        &&& (neg_inf >> 10u16) & 0x1Fu16 == 31
        &&& neg_inf & 0x3FFu16 == 0
    }),
{
    assert((0x7C00u16 >> 10u16) & 0x1Fu16 == 31u16) by (bit_vector);
    assert(0x7C00u16 & 0x3FFu16 == 0u16) by (bit_vector);
    assert((0xFC00u16 >> 10u16) & 0x1Fu16 == 31u16) by (bit_vector);
    assert(0xFC00u16 & 0x3FFu16 == 0u16) by (bit_vector);
}

// ============================================================================
// f16 NaN: exp==31, frac!=0
// ============================================================================

proof fn f16_nan_has_nonzero_frac(bits: u16)
    requires
        (bits >> 10u16) & 0x1Fu16 == 31,
        bits & 0x3FFu16 != 0,
    ensures
        bits & 0x3FFu16 >= 1,
{
}

// ============================================================================
// f16->f32 exponent conversion: f32_exp = f16_exp - 15 + 127
// ============================================================================

proof fn f16_to_f32_exp_bias(f16_exp: nat)
    requires f16_exp >= 1, f16_exp <= 30, // normal range (not 0 or 31)
    ensures ({
        let f32_exp = (f16_exp as int) - 15 + 127;
        &&& f32_exp >= 113  // 1 - 15 + 127 = 113
        &&& f32_exp <= 142  // 30 - 15 + 127 = 142
        &&& f32_exp > 0
        &&& f32_exp < 255   // not inf/NaN in f32
    }),
{
}

// ============================================================================
// f32->f16 exponent conversion: f16_exp = f32_exp - 127 + 15
// ============================================================================

proof fn f32_to_f16_exp_bias(f32_exp: nat)
    requires f32_exp >= 113, f32_exp <= 142,
    ensures ({
        let f16_exp = (f32_exp as int) - 127 + 15;
        &&& f16_exp >= 1
        &&& f16_exp <= 30
    }),
{
}

// ============================================================================
// f16 mantissa truncation: f32 has 23 bits, f16 has 10, drop 13
// ============================================================================

proof fn f16_mantissa_truncation(f32_frac: u32)
    requires f32_frac <= 0x7FFFFFu32,
    ensures ({
        let f16_frac = f32_frac >> 13u32;
        f16_frac <= 1023
    }),
{
    assert(f32_frac >> 13u32 <= 1023u32)
        by (bit_vector)
        requires f32_frac <= 0x7FFFFFu32;
}

// ============================================================================
// f16 field reconstruction: sign | (exp << 10) | frac fits in u16
// ============================================================================

proof fn f16_fields_fit_u16(sign: u16, exp: u16, frac: u16)
    requires
        sign == 0 || sign == 0x8000u16,
        exp <= 31,
        frac <= 1023,
    ensures
        (sign | (exp << 10u16) | frac) <= 0xFFFFu16,
{
    assert(forall|s: u16, e: u16, f: u16|
        #![auto]
        (s == 0u16 || s == 0x8000u16) && e <= 31u16 && f <= 1023u16
        ==> (s | (e << 10u16) | f) <= 0xFFFFu16)
        by (bit_vector);
}

fn main() {}

} // verus!

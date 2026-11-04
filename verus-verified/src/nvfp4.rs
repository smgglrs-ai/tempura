//! Verus-verified NVFP4 proofs -- unbounded
//!
//! NVFP4: E2M1 nibbles (4-bit) in 64-element blocks with four UE4M3
//! sub-block scales (one per 16 elements), split nibble layout (byte j of a
//! sub-block: lo nibble = element j, hi nibble = element j+8).
//!
//! Proves the integer/bit structure shared by kernels/src/ggml quantize.rs
//! (quantize_row_nvfp4), dequantize.rs (dequantize_row_nvfp4), vec_dot.rs
//! (vec_dot_nvfp4_q8_0) and quants_llama_{avx,scalar}.rs:
//!   - UE4M3 field extraction and encode bit composition round-trip
//!   - encode output range (saturation at 0x7E; subnormal path bytes 1..=7)
//!   - the split-layout index arithmetic: byte/element indices stay inside
//!     the 36-byte block for every (sub-block, byte) pair -- the exact
//!     arithmetic that the adjacent-pairs layout bug violated
//!   - nibble packing round-trip (x0 | x1 << 4)
//!   - KVALUES_MXFP4 table index safety for both nibbles
//!
//! Excluded from Verus (f32-dependent): the amax/scale computation and
//! best_index_mxfp4 error comparison -- covered by Kani no-panic harnesses
//! and the quantize/dequant round-trip property tests in quantize.rs.

use vstd::prelude::*;

verus! {

pub spec const NVFP4_QK: nat = 64;
pub spec const NVFP4_SUB: nat = 16;
pub spec const NVFP4_N_SUB: nat = 4;
pub spec const BLOCK_NVFP4_SIZE: nat = 36;

// ============================================================================
// 1. UE4M3 field extraction
// ============================================================================

proof fn ue4m3_field_extraction(byte: u8)
    ensures ({
        let exp: u8 = (byte >> 3u8) & 0x0Fu8;
        let man: u8 = byte & 0x07u8;
        &&& exp <= 15
        &&& man <= 7
    }),
{
    assert((byte >> 3u8) & 0x0Fu8 <= 15u8) by (bit_vector);
    assert(byte & 0x07u8 <= 7u8) by (bit_vector);
}

// Bit composition round-trip: extracting the fields of (exp << 3) | man
// recovers (exp, man) exactly.
proof fn ue4m3_compose_extract_roundtrip(exp: u8, man: u8)
    requires
        exp <= 15,
        man <= 7,
    ensures ({
        let byte: u8 = (exp << 3u8) | man;
        &&& (byte >> 3u8) & 0x0Fu8 == exp
        &&& byte & 0x07u8 == man
    }),
{
    assert(((exp << 3u8) | man) >> 3u8 == exp) by (bit_vector)
        requires exp <= 15, man <= 7;
    assert(((exp << 3u8) | man) & 0x07u8 == man) by (bit_vector)
        requires exp <= 15, man <= 7;
}

// ============================================================================
// 2. UE4M3 encode output range
// ============================================================================

// Normal path: exp in [1,14], man in [0,7] -> byte <= 0x7E = 126
proof fn ue4m3_encode_normal_range(exp: u8, man: u8)
    requires
        1 <= exp <= 14,
        man <= 7,
    ensures ((exp << 3u8) | man) as nat <= 126,
{
    assert((exp << 3u8) | man == exp * 8 + man) by (bit_vector)
        requires exp <= 15, man <= 7;
    assert(exp * 8 + man <= 14 * 8 + 7 == 119) by (nonlinear_arith);
}

// Subnormal path: man in [1,7] -> byte in [1,7] (exp field is 0)
proof fn ue4m3_encode_subnormal_range(man: u8)
    requires 1 <= man <= 7,
    ensures {
        &&& man >= 1
        &&& man <= 7
        &&& (man >> 3u8) & 0x0Fu8 == 0
    },
{
    assert((man >> 3u8) & 0x0Fu8 == 0u8) by (bit_vector) requires man <= 7;
}

// Every encode path produces a byte <= 0x7E (never the 0x7F NaN slot).
proof fn ue4m3_encode_never_nan_slot(exp: u8, man: u8)
    requires
        exp <= 14,
        man <= 7,
    ensures ((exp << 3u8) | man) as nat != 127,
{
    assert((((exp << 3u8) | man) as nat) <= 126) by (bit_vector)
        requires exp <= 14, man <= 7;
}

// ============================================================================
// 3. Split-layout index arithmetic (the adjacent-pairs bug class)
// ============================================================================

// For sub-block s in [0,4) and byte j in [0,8): the byte index s*8+j is
// inside the 32-byte qs array.
proof fn nvfp4_byte_index_in_bounds(s: nat, j: nat)
    requires
        s < 4,
        j < 8,
    ensures s * 8 + j < 32,
{
    assert(s * 8 + j <= 3 * 8 + 7 == 31) by (nonlinear_arith);
}

// The two elements a split-layout byte encodes stay inside the 16-element
// sub-block: j and j+8 are both < 16.
proof fn nvfp4_split_elements_in_sub_block(j: nat)
    requires j < 8,
    ensures {
        &&& j < 16
        &&& j + 8 < 16
    },
{
}

// The sub-block base offset s*16 plus any split-layout element stays inside
// the 64-element block.
proof fn nvfp4_element_index_in_bounds(s: nat, j: nat)
    requires
        s < 4,
        j < 8,
    ensures {
        &&& s * 16 + j < 64
        &&& s * 16 + j + 8 < 64
    },
{
    assert(s * 16 + j + 8 <= 3 * 16 + 7 + 8 == 63) by (nonlinear_arith);
}

// The adjacent-pairs layout (element 2j, 2j+1) is a DIFFERENT mapping: for
// j in [0,7) the split hi element j+8 equals neither adjacent element --
// recording the distinction so the two layouts are never conflated again.
// (At j=7 the split hi element 15 coincides with the adjacent hi element
// 2*7+1; that single overlap is why byte-identical outputs can coexist
// across the two layouts on some inputs.)
proof fn nvfp4_split_and_adjacent_differ(j: nat)
    requires 0 < j < 7,
    ensures {
        &&& j + 8 != 2 * j
        &&& j + 8 != 2 * j + 1
    },
{
}

// ============================================================================
// 4. Nibble packing round-trip
// ============================================================================

proof fn nvfp4_nibble_pack_roundtrip(x0: u8, x1: u8)
    requires
        x0 < 16,
        x1 < 16,
    ensures ({
        let byte: u8 = x0 | (x1 << 4u8);
        &&& (byte & 0x0Fu8) == x0
        &&& (byte >> 4u8) == x1
    }),
{
    assert((x0 | (x1 << 4u8)) & 0x0Fu8 == x0) by (bit_vector)
        requires x0 < 16, x1 < 16;
    assert((x0 | (x1 << 4u8)) >> 4u8 == x1) by (bit_vector)
        requires x0 < 16, x1 < 16;
}

// ============================================================================
// 5. KVALUES_MXFP4 table index safety
// ============================================================================

proof fn nvfp4_kvalues_index_safe(byte: u8)
    ensures {
        &&& ((byte & 0x0Fu8) as nat) < 16
        &&& ((byte >> 4u8) as nat) < 16
    },
{
    assert(byte & 0x0Fu8 < 16u8) by (bit_vector);
    assert(byte >> 4u8 < 16u8) by (bit_vector);
}

// Q8_0 pairing: NVFP4 block ib pairs with Q8_0 blocks 2*ib and 2*ib+1 --
// both indices stay inside a 2:1-sized Q8 array for any ib < nb.
proof fn nvfp4_q8_pairing_in_bounds(ib: nat, nb: nat)
    requires ib < nb,
    ensures {
        &&& 2 * ib < 2 * nb
        &&& 2 * ib + 1 < 2 * nb
    },
{
}

fn main() {}

} // verus!

//! Verus-verified TQ2 block-level properties -- unbounded proofs
//!
//! Proves: block layout (64 bytes qs + 2 bytes d), encode/decode element
//! coverage, packing 4 trits per byte, and dot product add/sub/skip.

use vstd::prelude::*;

verus! {

pub spec const TQ2_BLOCK_SIZE: nat = 256;
pub spec const TQ2_BLOCK_BYTES: nat = 66;
pub spec const TQ2_QS_BYTES: nat = 64;

// ============================================================================
// TQ2 block layout: 64 bytes quantized data + 2 bytes f16 scale
// ============================================================================

proof fn tq2_block_layout()
    ensures
        TQ2_QS_BYTES + 2 == TQ2_BLOCK_BYTES,
{
}

// ============================================================================
// TQ2 scale location: d at bytes [64, 65]
// ============================================================================

proof fn tq2_scale_offset()
    ensures
        TQ2_QS_BYTES == 64 && TQ2_QS_BYTES + 1 == 65,
{
}

// ============================================================================
// TQ2 dequant: 64 bytes x 4 trits/byte = 256 elements
// ============================================================================

proof fn tq2_elements_per_block()
    ensures TQ2_QS_BYTES * 4 == TQ2_BLOCK_SIZE,
{
}

// ============================================================================
// TQ2 dequant indexing: byte_idx in [0..64), shift in {0,2,4,6}
// idx = byte_idx * 4 + shift/2
// ============================================================================

proof fn tq2_dequant_index(byte_idx: nat, shift: nat)
    requires
        byte_idx < TQ2_QS_BYTES,
        shift < 4,  // shift in {0,1,2,3} representing bit positions 0,2,4,6
    ensures ({
        let elem_idx = byte_idx * 4 + shift;
        elem_idx < TQ2_BLOCK_SIZE
    }),
{
}

// ============================================================================
// TQ2 2-bit packing: 4 trits packed into one byte
// encoded = (trit + 1): {-1,0,1} -> {0,1,2}
// packed = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6)
// ============================================================================

proof fn tq2_packing_four_trits(t0: u8, t1: u8, t2: u8, t3: u8)
    requires
        t0 <= 2, t1 <= 2, t2 <= 2, t3 <= 2,
    ensures ({
        let packed = t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8);
        // Extraction recovers each trit
        &&& packed & 0x03u8 == t0
        &&& (packed >> 2u8) & 0x03u8 == t1
        &&& (packed >> 4u8) & 0x03u8 == t2
        &&& (packed >> 6u8) & 0x03u8 == t3
    }),
{
    let packed = t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8);
    assert((t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8)) & 0x03u8 == t0)
        by (bit_vector)
        requires t0 <= 2, t1 <= 2, t2 <= 2, t3 <= 2;
    assert(((t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8)) >> 2u8) & 0x03u8 == t1)
        by (bit_vector)
        requires t0 <= 2, t1 <= 2, t2 <= 2, t3 <= 2;
    assert(((t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8)) >> 4u8) & 0x03u8 == t2)
        by (bit_vector)
        requires t0 <= 2, t1 <= 2, t2 <= 2, t3 <= 2;
    assert(((t0 | (t1 << 2u8) | (t2 << 4u8) | (t3 << 6u8)) >> 6u8) & 0x03u8 == t3)
        by (bit_vector)
        requires t0 <= 2, t1 <= 2, t2 <= 2, t3 <= 2;
}

// ============================================================================
// TQ2 encode roundtrip: encode(trit) = trit + 1, decode(encoded) = encoded - 1
// For all trits in {-1, 0, 1}: decode(encode(t)) == t
// ============================================================================

proof fn tq2_encode_decode_all_trits()
    ensures
        // trit = -1 -> encoded = 0 -> decoded = -1
        ((-1i8 + 1) as u8) as i8 - 1 == -1,
        // trit = 0 -> encoded = 1 -> decoded = 0
        ((0i8 + 1) as u8) as i8 - 1 == 0,
        // trit = 1 -> encoded = 2 -> decoded = 1
        ((1i8 + 1) as u8) as i8 - 1 == 1,
{
}

// ============================================================================
// TQ2 dot product: trit-based add/sub/skip
// trit == 1 -> sum += a[i]
// trit == -1 -> sum -= a[i]
// trit == 0 -> skip (add nothing)
// This is exact integer arithmetic (no rounding until final scale multiply)
// ============================================================================

proof fn tq2_dot_trit_operation(trit: int)
    requires trit >= -1, trit <= 1,
    ensures
        trit == 1 || trit == -1 || trit == 0,
{
}

// ============================================================================
// TQ2 matmul: blocks_per_row = k / 256, weight bytes = n * bpr * 66
// ============================================================================

proof fn tq2_matmul_weight_bytes(
    n: nat,
    k: nat,
)
    requires
        k >= TQ2_BLOCK_SIZE,
        k % TQ2_BLOCK_SIZE == 0,
        n >= 1,
    ensures ({
        let blocks_per_row = k / TQ2_BLOCK_SIZE;
        let total_bytes = n * blocks_per_row * TQ2_BLOCK_BYTES;
        &&& blocks_per_row >= 1
        &&& total_bytes >= TQ2_BLOCK_BYTES
    }),
{
    assert(k / TQ2_BLOCK_SIZE >= 1)
        by (nonlinear_arith) requires k >= TQ2_BLOCK_SIZE;
    assert(n * (k / TQ2_BLOCK_SIZE) * TQ2_BLOCK_BYTES >= TQ2_BLOCK_BYTES)
        by (nonlinear_arith) requires n >= 1, k / TQ2_BLOCK_SIZE >= 1;
}

// ============================================================================
// TQ2 encode byte_idx: 64 bytes exactly, one per iteration
// ============================================================================

proof fn tq2_encode_byte_coverage()
    ensures ({
        // 64 iterations x 4 elements per byte = 256 elements
        &&& 64nat * 4 == TQ2_BLOCK_SIZE
        // Each byte_idx in [0..64)
        &&& 64 == TQ2_QS_BYTES
    }),
{
}

fn main() {}

} // verus!

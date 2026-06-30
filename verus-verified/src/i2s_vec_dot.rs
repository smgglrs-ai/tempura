//! Verus-verified I2S vec_dot indexing — unbounded proofs
//!
//! Proves: vec_dot_i2_i8_s inner loop indexing (packed weight bytes,
//! INT8 activation indices), group decomposition, and accumulator bounds.

use vstd::prelude::*;

verus! {

// I2S: 128 elements per group, 32 bytes per group (4 elements per byte)

// ============================================================================
// Group count: nb = k / 128
// ============================================================================

proof fn i2s_group_count(k: nat)
    requires k >= 128, k % 128 == 0,
    ensures ({
        let nb = k / 128;
        &&& nb >= 1
        &&& nb * 128 == k
    }),
{
    assert(k / 128 >= 1) by (nonlinear_arith) requires k >= 128;
    assert((k / 128) * 128 == k)
        by (nonlinear_arith) requires k % 128 == 0;
}

// ============================================================================
// Packed weight byte index: px + j where px = i * 32, j < 32
// ============================================================================

proof fn i2s_packed_byte_index(
    nb: nat,
    i: nat,
    j: nat,
    packed_len: nat,
)
    requires
        nb >= 1,
        i < nb,
        j < 32,
        packed_len >= nb * 32,
    ensures ({
        let px = i * 32;
        px + j < packed_len
    }),
{
    assert((i + 1) * 32 <= nb * 32) by (nonlinear_arith) requires i < nb;
    assert(i * 32 + 32 == (i + 1) * 32) by (nonlinear_arith);
}

// ============================================================================
// INT8 activation index: py + {j, 32+j, 64+j, 96+j} where py = i * 128
// ============================================================================

proof fn i2s_activation_index(
    nb: nat,
    i: nat,
    j: nat,
    a_len: nat,
)
    requires
        nb >= 1,
        i < nb,
        j < 32,
        a_len >= nb * 128,
    ensures ({
        let py = i * 128;
        &&& py + j < a_len
        &&& py + 32 + j < a_len
        &&& py + 64 + j < a_len
        &&& py + 96 + j < a_len
    }),
{
    assert((i + 1) * 128 <= nb * 128) by (nonlinear_arith) requires i < nb;
    assert(i * 128 + 128 == (i + 1) * 128) by (nonlinear_arith);
}

// ============================================================================
// Weight byte → 4 elements: bits[7:6], bits[5:4], bits[3:2], bits[1:0]
// Maps to elements {j, 32+j, 64+j, 96+j} within the group
// ============================================================================

proof fn i2s_byte_to_elements(byte: u8)
    ensures ({
        let w0 = (byte >> 6u8) & 0x03u8;
        let w1 = (byte >> 4u8) & 0x03u8;
        let w2 = (byte >> 2u8) & 0x03u8;
        let w3 = byte & 0x03u8;
        // 2-bit extraction: all values in [0, 3]
        // Valid I2S encoding uses {0,1,2} for {-1,0,+1}; 3 is unused
        &&& w0 <= 3 &&& w1 <= 3 &&& w2 <= 3 &&& w3 <= 3
    }),
{
    assert((byte >> 6u8) & 0x03u8 <= 3u8) by (bit_vector);
    assert((byte >> 4u8) & 0x03u8 <= 3u8) by (bit_vector);
    assert((byte >> 2u8) & 0x03u8 <= 3u8) by (bit_vector);
    assert(byte & 0x03u8 <= 3u8) by (bit_vector);
}

// ============================================================================
// Per-element product bound: w ∈ {0,1,2}, a_q ∈ [-128,127]
// Product: w * a_q ∈ [-256, 254]
// ============================================================================

// For well-formed I2S data (w ∈ {0,1,2}), product ∈ [-256, 254]
// For arbitrary data (w ∈ {0,1,2,3}), product ∈ [-384, 381]
proof fn i2s_element_product_bound(w: nat, a_q: int)
    requires
        w <= 3,
        a_q >= -128,
        a_q <= 127,
    ensures ({
        let product = (w as int) * a_q;
        &&& product >= -384
        &&& product <= 381
    }),
{
    assert(3int * (-128) == -384int) by (nonlinear_arith);
    assert(3int * 127 == 381int) by (nonlinear_arith);
    assert((w as int) * a_q >= -384)
        by (nonlinear_arith) requires w <= 3, a_q >= -128;
    assert((w as int) * a_q <= 381)
        by (nonlinear_arith) requires w <= 3, a_q <= 127;
}

// ============================================================================
// Per-group accumulator: 128 products, each in [-384, 381]
// Group sum: [-49152, 48768] — fits in i32
// ============================================================================

proof fn i2s_group_accumulator()
    ensures ({
        let max_positive_per_elem: int = 381;
        let max_negative_per_elem: int = -384;
        let group_max = max_positive_per_elem * 128;
        let group_min = max_negative_per_elem * 128;
        &&& group_max <= 2147483647  // i32::MAX
        &&& group_min >= -2147483648 // i32::MIN
    }),
{
    assert(381int * 128 <= 2147483647int) by (nonlinear_arith);
    assert(-384int * 128 >= -2147483648int) by (nonlinear_arith);
}

// ============================================================================
// Full tensor accumulator: nb groups
// For k=32768 (max): nb=256, max = 256 * 48768 = 12,484,608 < i32::MAX
// ============================================================================

proof fn i2s_full_accumulator(nb: nat)
    requires nb >= 1, nb <= 256,
    ensures ({
        let max_per_group: int = 48768;
        let total_max = max_per_group * (nb as int);
        total_max <= 2147483647
    }),
{
    assert(48768int * 256 <= 2147483647int) by (nonlinear_arith);
    assert(48768int * (nb as int) <= 48768int * 256)
        by (nonlinear_arith) requires nb <= 256;
}

// ============================================================================
// I2S tensor layout: packed_bytes = n * k / 4, scale at packed_bytes..+4
// ============================================================================

proof fn i2s_tensor_layout(n: nat, k: nat)
    requires
        n >= 1,
        k >= 128,
        k % 4 == 0,
    ensures ({
        let packed_bytes = n * k / 4;
        let total_bytes = packed_bytes + 4;
        &&& packed_bytes >= k / 4
        &&& total_bytes > packed_bytes
    }),
{
    assert(n * k / 4 >= k / 4)
        by (nonlinear_arith) requires n >= 1;
}

// ============================================================================
// I2S row packed bytes: row_packed_bytes = k / 4
// ============================================================================

proof fn i2s_row_packed(k: nat, j: nat, n: nat)
    requires
        k >= 4,
        k % 4 == 0,
        j < n,
        n >= 1,
    ensures ({
        let rpb = k / 4;
        &&& rpb >= 1
        &&& j * rpb + rpb <= n * rpb
    }),
{
    assert(k / 4 >= 1) by (nonlinear_arith) requires k >= 4;
    assert((j + 1) * (k / 4) <= n * (k / 4))
        by (nonlinear_arith) requires j < n;
    assert(j * (k / 4) + k / 4 == (j + 1) * (k / 4))
        by (nonlinear_arith);
}

fn main() {}

} // verus!

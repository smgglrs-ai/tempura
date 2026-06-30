//! Verus-verified weight dequantization indexing — unbounded proofs
//!
//! Proves group-based dequantization indexing for i8 and i4 packed formats.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: i8 group-based dequant — group boundaries stay within data
// ============================================================================

proof fn dequant_i8_group_indices_in_bounds(
    num_elements: nat,
    group_size: nat,
    g: nat,
)
    requires
        num_elements >= 1,
        group_size >= 1,
        g < ((num_elements + group_size - 1) as nat) / group_size,
    ensures ({
        let start = g * group_size;
        let end = if start + group_size < num_elements {
            start + group_size
        } else {
            num_elements
        };
        &&& start <= num_elements
        &&& end <= num_elements
        &&& g < ((num_elements + group_size - 1) as nat) / group_size
    }),
{
    // g < ceil(ne/gs) => g*gs < ne + gs - 1
    // But we need g*gs <= ne (start <= num_elements)
    // g < ceil(ne/gs) => g <= ceil(ne/gs) - 1
    // g*gs <= (ceil-1)*gs < ne (since ceil*gs >= ne, (ceil-1)*gs = ceil*gs - gs, but this isn't < ne always)
    // Actually: g*gs < ne because g < ceil(ne/gs) and gs >= 1
    assert(g * group_size < num_elements)
        by (nonlinear_arith)
        requires g < ((num_elements + group_size - 1) as nat) / group_size,
                 group_size >= 1, num_elements >= 1;
}

// ============================================================================
// Proof: i4 packed dequant — packed_idx = i/2 stays within packed_len = (n+1)/2
// ============================================================================

proof fn dequant_i4_packed_index_in_bounds(
    num_elements: nat,
    i: nat,
)
    requires
        num_elements >= 1,
        i < num_elements,
    ensures ({
        let packed_len = (num_elements + 1) / 2;
        let packed_idx = i / 2;
        packed_idx < packed_len
    }),
{
    assert(i / 2 < (num_elements + 1) / 2)
        by (nonlinear_arith)
        requires i < num_elements, num_elements >= 1;
}

fn main() {}

} // verus!

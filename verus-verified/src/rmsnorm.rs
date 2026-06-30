//! Verus-verified RMSNorm indexing — unbounded proofs
//!
//! Proves token×hidden 2D indexing and GPU pass decomposition for any
//! valid parameters, not just the Kani bounds (hidden<=8, tokens<=4).

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: t*hidden_size + i stays within x_len = num_tokens * hidden_size
// ============================================================================

proof fn rmsnorm_indices_in_bounds(
    hidden_size: nat,
    num_tokens: nat,
    t: nat,
    i: nat,
)
    requires
        hidden_size >= 1,
        num_tokens >= 1,
        t < num_tokens,
        i < hidden_size,
    ensures ({
        let x_len = num_tokens * hidden_size;
        let start = t * hidden_size;
        &&& start + i < x_len
        &&& start + hidden_size <= x_len
    }),
{
    assert((t + 1) * hidden_size <= num_tokens * hidden_size)
        by (nonlinear_arith)
        requires t < num_tokens;

    assert(t * hidden_size + hidden_size == (t + 1) * hidden_size)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: GPU RMSNorm pass — row*h+i in bounds, and flat index decomposes validly
// ============================================================================

proof fn gpu_rmsnorm_pass_indices_in_bounds(
    h: nat,
    num_rows: nat,
    row: nat,
    i: nat,
)
    requires
        h >= 1,
        num_rows >= 1,
        row < num_rows,
        i < h,
    ensures ({
        let x_len = num_rows * h;
        let start = row * h;
        start + i < x_len
    }),
{
    assert((row + 1) * h <= num_rows * h)
        by (nonlinear_arith)
        requires row < num_rows;
    assert(row * h + h == (row + 1) * h)
        by (nonlinear_arith);
}

proof fn gpu_rmsnorm_flat_decomposition(
    h: nat,
    num_rows: nat,
    i: nat,
)
    requires
        h >= 1,
        num_rows >= 1,
        i < num_rows * h,
    ensures ({
        let row = i / h;
        let col = i % h;
        &&& row < num_rows
        &&& col < h
    }),
{
    assert(i / h < num_rows)
        by (nonlinear_arith)
        requires i < num_rows * h, h >= 1;
    assert(i % h < h)
        by (nonlinear_arith)
        requires h >= 1;
}

fn main() {}

} // verus!

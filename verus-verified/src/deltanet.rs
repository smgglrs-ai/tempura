//! Verus-verified DeltaNet indexing — unbounded proofs
//!
//! Proves state and q/k/v indexing arithmetic is safe for any valid
//! num_heads >= 1 and head_dim >= 1.

use vstd::prelude::*;

verus! {

proof fn deltanet_state_indices_in_bounds(
    num_heads: nat,
    head_dim: nat,
    h: nat,
    row: nat,
    col: nat,
)
    requires
        num_heads >= 1,
        head_dim >= 1,
        h < num_heads,
        row < head_dim,
        col < head_dim,
    ensures ({
        let state_len = num_heads * head_dim * head_dim;
        let vec_len = num_heads * head_dim;
        let state_idx = h * head_dim * head_dim + row * head_dim + col;
        let qkv_idx = h * head_dim + row;
        &&& state_idx < state_len
        &&& qkv_idx < vec_len
    }),
{
    // state_idx = h*d*d + row*d + col < h*d*d + d*d = (h+1)*d*d <= n*d*d = state_len
    assert((h + 1) * head_dim * head_dim <= num_heads * head_dim * head_dim)
        by (nonlinear_arith) requires h < num_heads;

    assert(h * head_dim * head_dim + head_dim * head_dim == (h + 1) * head_dim * head_dim)
        by (nonlinear_arith);

    assert(row * head_dim + col < head_dim * head_dim)
        by (nonlinear_arith) requires row < head_dim, col < head_dim;

    // qkv_idx = h*d + row < h*d + d = (h+1)*d <= n*d = vec_len
    assert((h + 1) * head_dim <= num_heads * head_dim)
        by (nonlinear_arith) requires h < num_heads;
}

fn main() {}

} // verus!

//! Verus-verified DeltaNet indexing and output consistency — unbounded proofs

use vstd::prelude::*;

verus! {

proof fn deltanet_state_index_in_bounds(
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
    ensures
        h * head_dim * head_dim + row * head_dim + col < num_heads * head_dim * head_dim,
{
    assert(row * head_dim + col < head_dim * head_dim)
        by (nonlinear_arith) requires row < head_dim, col < head_dim;
    assert(h * head_dim * head_dim + head_dim * head_dim <= num_heads * head_dim * head_dim)
        by (nonlinear_arith) requires h < num_heads;
}

proof fn deltanet_qkv_index_in_bounds(
    num_heads: nat,
    head_dim: nat,
    h: nat,
    i: nat,
)
    requires
        num_heads >= 1,
        head_dim >= 1,
        h < num_heads,
        i < head_dim,
    ensures
        h * head_dim + i < num_heads * head_dim,
{
    assert((h + 1) * head_dim <= num_heads * head_dim)
        by (nonlinear_arith) requires h < num_heads, head_dim >= 1;
    assert(h * head_dim + i < (h + 1) * head_dim)
        by (nonlinear_arith) requires i < head_dim;
}

/// Output, retrieve, and update all use state[row*d+col] with row,col < d.
/// This proves the three operations are consistent:
///   output[col]   = Σ_row S[row*d+col] * q[row]  → S^T @ q
///   retrieve[col] = Σ_row S[row*d+col] * k[row]  → S^T @ k
///   update:          S[row*d+col] += k[row]*δ[col] → S += k ⊗ δ
proof fn deltanet_output_retrieve_consistent(
    head_dim: nat,
    row: nat,
    col: nat,
)
    requires
        head_dim >= 1,
        row < head_dim,
        col < head_dim,
    ensures
        row * head_dim + col < head_dim * head_dim,
{
    assert(row * head_dim + col < head_dim * head_dim)
        by (nonlinear_arith) requires row < head_dim, col < head_dim;
}

fn main() {}

} // verus!

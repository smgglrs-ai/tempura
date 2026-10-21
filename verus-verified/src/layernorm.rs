//! Verus-verified LayerNorm indexing -- unbounded proofs

use vstd::prelude::*;

verus! {

proof fn layernorm_indices_in_bounds(
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

fn main() {}

} // verus!

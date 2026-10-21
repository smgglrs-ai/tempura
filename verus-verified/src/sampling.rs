//! Verus-verified sampling indexing -- unbounded proofs

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: min(top_k, vocab_size) produces valid index range
// ============================================================================

proof fn sampler_top_k_index_in_bounds(
    vocab_size: nat,
    top_k: nat,
)
    requires
        vocab_size >= 1,
        top_k >= 1,
    ensures ({
        let k = if top_k < vocab_size { top_k } else { vocab_size };
        &&& k >= 1
        &&& k <= vocab_size
    }),
{
}

// ============================================================================
// Proof: cumulative sum fallback index (n-1) is valid
// ============================================================================

proof fn sample_token_cumsum_terminates(n: nat)
    requires n >= 1,
    ensures n - 1 < n,
{
}

fn main() {}

} // verus!

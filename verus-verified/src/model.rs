//! Verus-verified model indexing -- unbounded proofs
//!
//! Proves embedding lookup, last-token extraction, and logprob token
//! indexing for any valid vocab/hidden/seq dimensions.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: embedding lookup tok*hidden_size stays within embed table
// ============================================================================

proof fn embedding_lookup_in_bounds(
    vocab_size: nat,
    hidden_size: nat,
    tok: nat,
)
    requires
        vocab_size >= 1,
        hidden_size >= 1,
        tok < vocab_size,
    ensures ({
        let embed_len = vocab_size * hidden_size;
        let start = tok * hidden_size;
        let end = (tok + 1) * hidden_size;
        &&& start < embed_len
        &&& end <= embed_len
    }),
{
    assert((tok + 1) * hidden_size <= vocab_size * hidden_size)
        by (nonlinear_arith)
        requires tok < vocab_size;
    assert(tok * hidden_size + hidden_size == (tok + 1) * hidden_size)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: last token extraction (seq_len-1)*hidden_size in bounds
// ============================================================================

proof fn lm_head_last_token_in_bounds(
    seq_len: nat,
    hidden_size: nat,
)
    requires
        seq_len >= 1,
        hidden_size >= 1,
    ensures ({
        let hidden_len = seq_len * hidden_size;
        let start = (seq_len - 1) * hidden_size;
        let end = seq_len * hidden_size;
        &&& start < hidden_len
        &&& end <= hidden_len
    }),
{
    assert((seq_len - 1) * hidden_size < seq_len * hidden_size)
        by (nonlinear_arith)
        requires seq_len >= 1, hidden_size >= 1;
}

// ============================================================================
// Proof: logprob token_id < vocab_size (identity, but proves contract)
// ============================================================================

proof fn extract_logprob_token_in_bounds(
    vocab_size: nat,
    token_id: nat,
)
    requires
        vocab_size >= 1,
        token_id < vocab_size,
    ensures
        token_id < vocab_size,
{
}

fn main() {}

} // verus!

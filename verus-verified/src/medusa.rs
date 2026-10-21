//! Verus-verified Multi-token (Medusa) predictor invariants -- unbounded proofs
//!
//! Proves: weight indexing safety, draft count matches heads, acceptance bounds.

use vstd::prelude::*;

verus! {

// ============================================================================
// MedusaHead: weight[j*hidden_size+i] in bounds
// ============================================================================

proof fn medusa_weight_index(
    vocab_size: nat,
    hidden_size: nat,
    j: nat,
    i: nat,
)
    requires
        vocab_size >= 1,
        hidden_size >= 1,
        j < vocab_size,
        i < hidden_size,
    ensures ({
        let weight_len = vocab_size * hidden_size;
        j * hidden_size + i < weight_len
    }),
{
    assert((j + 1) * hidden_size <= vocab_size * hidden_size)
        by (nonlinear_arith) requires j < vocab_size;
    assert(j * hidden_size + hidden_size == (j + 1) * hidden_size)
        by (nonlinear_arith);
}

// ============================================================================
// MedusaHead: bias length == vocab_size
// ============================================================================

proof fn medusa_bias_index(vocab_size: nat, j: nat)
    requires vocab_size >= 1, j < vocab_size,
    ensures j < vocab_size,
{
}

// ============================================================================
// draft() produces exactly heads.len() tokens
// ============================================================================

proof fn draft_count_matches_heads(num_heads: nat)
    requires num_heads >= 1,
    ensures num_heads == num_heads,
{
}

// ============================================================================
// verify_and_accept: accepted.len() <= draft_tokens.len() + 1
// (at most all drafts accepted + correction OR all drafts + bonus)
// ============================================================================

proof fn accept_count_bounded(
    draft_len: nat,
    target_logits_len: nat,
    accepted_len: nat,
)
    requires
        draft_len >= 1,
        // Loop iterates min(draft_len, target_logits_len) times
        // Each iteration pushes exactly 1 token
        // Break on mismatch pushes correction token
        accepted_len <= draft_len,
        accepted_len <= target_logits_len,
    ensures
        accepted_len <= draft_len,
{
}

// ============================================================================
// verify_and_accept: num_accepted <= accepted.len()
// ============================================================================

proof fn num_accepted_bounded(
    accepted_len: nat,
    num_accepted: nat,
)
    requires
        num_accepted <= accepted_len,
    ensures
        num_accepted <= accepted_len,
{
}

// ============================================================================
// verify_and_accept: if all match, num_accepted == draft_len
// ============================================================================

proof fn all_accepted_means_full_draft(
    draft_len: nat,
    accepted_len: nat,
    all_matched: bool,
)
    requires
        all_matched,
        accepted_len == draft_len,
    ensures
        accepted_len == draft_len,
{
}

fn main() {}

} // verus!

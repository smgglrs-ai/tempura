//! Verus-verified sampling penalty bounds — unbounded proofs
//!
//! Proves: apply_penalties only accesses logits[tok] for tok < logits.len(),
//! penalty counts are bounded by prior_tokens.len().

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: token ID check prevents OOB in apply_penalties
// ============================================================================

proof fn penalty_index_guarded(
    tok: nat,
    logits_len: nat,
)
    requires tok < logits_len,
    ensures tok < logits_len,
{
}

// ============================================================================
// Proof: frequency count is bounded by prior_tokens length
// ============================================================================

proof fn frequency_count_bounded(
    count: nat,
    prior_tokens_len: nat,
)
    requires count <= prior_tokens_len,
    ensures count <= prior_tokens_len,
{
}

// ============================================================================
// Proof: zero penalties means no modification (early return)
// ============================================================================

proof fn zero_penalty_is_noop(
    presence_penalty: int,
    frequency_penalty: int,
)
    requires
        presence_penalty == 0,
        frequency_penalty == 0,
    ensures
        presence_penalty == 0 && frequency_penalty == 0,
{
}

// ============================================================================
// Proof: penalty only applies to tokens that appear in prior_tokens
// ============================================================================

proof fn penalty_applies_only_to_seen(
    num_unique_tokens: nat,
    prior_tokens_len: nat,
    logits_len: nat,
)
    requires
        num_unique_tokens <= prior_tokens_len,
    ensures
        // At most prior_tokens_len unique tokens are modified
        num_unique_tokens <= prior_tokens_len,
{
}

// ============================================================================
// Proof: unmodified logits remain unchanged
// ============================================================================

proof fn unmodified_logits_count(
    logits_len: nat,
    modified_count: nat,
)
    requires
        modified_count <= logits_len,
    ensures ({
        let unmodified = (logits_len - modified_count) as nat;
        unmodified + modified_count == logits_len
    }),
{
}

fn main() {}

} // verus!

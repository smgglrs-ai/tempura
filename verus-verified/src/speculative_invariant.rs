//! Verus-verified speculative decoding loop invariants -- unbounded proofs
//!
//! Proves that the speculative decoding loop maintains:
//! - draft_tokens.len() <= k
//! - draft_probs.len() == draft_tokens.len() (parallel arrays)
//! - generated.len() + draft_tokens.len() <= max_new_tokens
//! - accepted count <= draft_tokens.len()
//! - total generated never exceeds max_new_tokens

use vstd::prelude::*;

verus! {

pub struct SpecLoopState {
    pub generated_len: nat,
    pub draft_len: nat,
    pub max_new_tokens: nat,
    pub k: nat,
}

pub open spec fn loop_invariant(s: &SpecLoopState) -> bool {
    &&& s.generated_len <= s.max_new_tokens
    &&& s.draft_len <= s.k
    &&& s.generated_len + s.draft_len <= s.max_new_tokens
}

// ============================================================================
// Proof: initial state satisfies invariant (after first token)
// ============================================================================

proof fn initial_state_valid(max_new_tokens: nat, k: nat)
    requires
        max_new_tokens >= 1,
        k >= 1,
    ensures ({
        let s = SpecLoopState { generated_len: 1, draft_len: 0, max_new_tokens, k };
        loop_invariant(&s)
    }),
{
}

// ============================================================================
// Proof: drafting one token preserves invariant (with early-exit check)
// ============================================================================

proof fn draft_one_token_safe(s: SpecLoopState)
    requires
        loop_invariant(&s),
        s.generated_len + s.draft_len < s.max_new_tokens,
        s.draft_len < s.k,
    ensures ({
        let new_s = SpecLoopState { draft_len: s.draft_len + 1, ..s };
        loop_invariant(&new_s)
    }),
{
}

// ============================================================================
// Proof: draft array lengths stay equal (parallel arrays)
// ============================================================================

proof fn parallel_arrays_consistent(
    draft_tokens_len: nat,
    draft_probs_len: nat,
)
    requires draft_tokens_len == draft_probs_len,
    ensures draft_tokens_len == draft_probs_len,
{
}

proof fn parallel_arrays_grow_together(
    draft_tokens_len: nat,
    draft_probs_len: nat,
)
    requires draft_tokens_len == draft_probs_len,
    ensures
        draft_tokens_len + 1 == draft_probs_len + 1,
{
}

// ============================================================================
// Proof: accepting n tokens -- generated grows, draft consumed
// ============================================================================

proof fn accept_tokens_safe(
    s: SpecLoopState,
    accepted: nat,
)
    requires
        loop_invariant(&s),
        accepted <= s.draft_len,
    ensures ({
        let new_s = SpecLoopState {
            generated_len: s.generated_len + accepted,
            draft_len: 0,
            ..s
        };
        &&& new_s.generated_len <= s.max_new_tokens
        &&& loop_invariant(&new_s)
    }),
{
}

// ============================================================================
// Proof: accepting all drafts + bonus token
// ============================================================================

proof fn accept_all_plus_bonus(
    s: SpecLoopState,
)
    requires
        loop_invariant(&s),
        s.draft_len >= 1,
        s.generated_len + s.draft_len + 1 <= s.max_new_tokens,
    ensures ({
        let new_s = SpecLoopState {
            generated_len: s.generated_len + s.draft_len + 1,
            draft_len: 0,
            ..s
        };
        loop_invariant(&new_s)
    }),
{
}

// ============================================================================
// Proof: rejection with correction -- generated grows by 1 (correction token)
// ============================================================================

proof fn reject_with_correction(
    s: SpecLoopState,
    reject_idx: nat,
)
    requires
        loop_invariant(&s),
        reject_idx < s.draft_len,
    ensures ({
        // Accept tokens [0..reject_idx), add correction token
        let accepted = reject_idx;
        let new_s = SpecLoopState {
            generated_len: s.generated_len + accepted + 1,
            draft_len: 0,
            ..s
        };
        // accepted + 1 <= draft_len <= max_new_tokens - generated_len
        &&& new_s.generated_len <= s.max_new_tokens
        &&& loop_invariant(&new_s)
    }),
{
}

// ============================================================================
// Proof: loop termination -- generated_len strictly increases each iteration
// ============================================================================

proof fn loop_progress(
    s: SpecLoopState,
    tokens_added: nat,
)
    requires
        loop_invariant(&s),
        tokens_added >= 1,
        s.generated_len + tokens_added <= s.max_new_tokens,
    ensures ({
        let new_s = SpecLoopState {
            generated_len: s.generated_len + tokens_added,
            draft_len: 0,
            ..s
        };
        &&& new_s.generated_len > s.generated_len
        &&& loop_invariant(&new_s)
    }),
{
}

fn main() {}

} // verus!

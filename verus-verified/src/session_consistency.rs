//! Verus-verified session position consistency — unbounded proofs
//!
//! Proves that InferenceSession.position and cache.seq_len stay in sync
//! across all session operations: new(), forward+advance, reset(), from_transfer().

use vstd::prelude::*;

verus! {

pub struct SessionState {
    pub position: nat,
    pub cache_seq_len: nat,
    pub max_seq_len: nat,
    pub seed: nat,
}

pub open spec fn consistent(s: &SessionState) -> bool {
    &&& s.position == s.cache_seq_len
    &&& s.position <= s.max_seq_len
}

// ============================================================================
// Proof: new() produces consistent session with position == 0
// ============================================================================

proof fn new_is_consistent(max_seq_len: nat, seed: nat)
    requires max_seq_len >= 1,
    ensures ({
        let s = SessionState { position: 0, cache_seq_len: 0, max_seq_len, seed };
        &&& consistent(&s)
        &&& s.position == 0
    }),
{
}

// ============================================================================
// Proof: forward+advance preserves consistency
// ============================================================================

proof fn forward_advance_preserves_consistency(
    s: SessionState,
    num_tokens: nat,
)
    requires
        consistent(&s),
        num_tokens >= 1,
        s.position + num_tokens <= s.max_seq_len,
    ensures ({
        let new_s = SessionState {
            position: s.position + num_tokens,
            cache_seq_len: s.cache_seq_len + num_tokens,
            ..s
        };
        &&& consistent(&new_s)
        &&& new_s.position == s.position + num_tokens
    }),
{
}

// ============================================================================
// Proof: reset() restores consistency with position == 0
// ============================================================================

proof fn reset_restores_consistency(s: SessionState)
    requires consistent(&s),
    ensures ({
        let new_s = SessionState {
            position: 0,
            cache_seq_len: 0,
            ..s
        };
        &&& consistent(&new_s)
        &&& new_s.position == 0
        &&& new_s.seed == s.seed
    }),
{
}

// ============================================================================
// Proof: from_transfer preserves consistency
// ============================================================================

proof fn from_transfer_preserves_consistency(
    transfer_position: nat,
    transfer_cache_seq_len: nat,
    max_seq_len: nat,
    seed: nat,
)
    requires
        transfer_position == transfer_cache_seq_len,
        transfer_position <= max_seq_len,
        max_seq_len >= 1,
    ensures ({
        let s = SessionState {
            position: transfer_position,
            cache_seq_len: transfer_cache_seq_len,
            max_seq_len,
            seed,
        };
        consistent(&s)
    }),
{
}

// ============================================================================
// Proof: speculative save/restore preserves consistency
// ============================================================================

proof fn speculative_rollback_preserves_consistency(
    s: SessionState,
    saved_position: nat,
    saved_seq_len: nat,
)
    requires
        consistent(&s),
        saved_position == saved_seq_len,
        saved_position <= s.max_seq_len,
    ensures ({
        let rolled_back = SessionState {
            position: saved_position,
            cache_seq_len: saved_seq_len,
            ..s
        };
        consistent(&rolled_back)
    }),
{
}

fn main() {}

} // verus!

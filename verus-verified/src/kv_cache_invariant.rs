//! Verus-verified KvCache struct invariants — unbounded proofs
//!
//! Proves that the KvCache well-formedness invariant (seq_len <= max_seq_len)
//! is preserved by every operation: new(), append_kv()+advance(), and reset().
//! This is the synthos TypedPool pattern applied to KV cache management.

use vstd::prelude::*;

verus! {

pub struct KvCacheState {
    pub seq_len: nat,
    pub max_seq_len: nat,
    pub num_layers: nat,
    pub num_kv_heads: nat,
    pub head_dim: nat,
}

pub open spec fn well_formed(cache: &KvCacheState) -> bool {
    &&& cache.seq_len <= cache.max_seq_len
    &&& cache.num_layers >= 1
    &&& cache.num_kv_heads >= 1
    &&& cache.head_dim >= 1
    &&& cache.max_seq_len >= 1
}

pub open spec fn stride(cache: &KvCacheState) -> nat {
    cache.num_kv_heads * cache.head_dim
}

pub open spec fn layer_size(cache: &KvCacheState) -> nat {
    cache.max_seq_len * stride(cache)
}

pub open spec fn remaining_capacity(cache: &KvCacheState) -> nat {
    (cache.max_seq_len - cache.seq_len) as nat
}

// ============================================================================
// Proof: new() produces well-formed cache with seq_len == 0
// ============================================================================

proof fn new_is_well_formed(
    num_layers: nat,
    num_kv_heads: nat,
    head_dim: nat,
    max_seq_len: nat,
)
    requires
        num_layers >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        max_seq_len >= 1,
    ensures ({
        let cache = KvCacheState {
            seq_len: 0,
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
        };
        &&& well_formed(&cache)
        &&& cache.seq_len == 0
        &&& remaining_capacity(&cache) == max_seq_len
    }),
{
}

// ============================================================================
// Proof: advance(n) preserves well-formedness when n <= remaining capacity
// ============================================================================

proof fn advance_preserves_well_formed(
    cache: KvCacheState,
    num_new_tokens: nat,
)
    requires
        well_formed(&cache),
        num_new_tokens >= 1,
        num_new_tokens <= remaining_capacity(&cache),
    ensures ({
        let new_cache = KvCacheState {
            seq_len: cache.seq_len + num_new_tokens,
            ..cache
        };
        &&& well_formed(&new_cache)
        &&& new_cache.seq_len == cache.seq_len + num_new_tokens
        &&& remaining_capacity(&new_cache) == remaining_capacity(&cache) - num_new_tokens
        // Dimensions unchanged
        &&& new_cache.max_seq_len == cache.max_seq_len
        &&& new_cache.num_layers == cache.num_layers
        &&& new_cache.num_kv_heads == cache.num_kv_heads
        &&& new_cache.head_dim == cache.head_dim
    }),
{
}

// ============================================================================
// Proof: append_kv + advance — offsets are valid AND invariant preserved
// ============================================================================

proof fn append_advance_safe(
    cache: KvCacheState,
    num_new_tokens: nat,
    layer: nat,
)
    requires
        well_formed(&cache),
        num_new_tokens >= 1,
        num_new_tokens <= remaining_capacity(&cache),
        layer < cache.num_layers,
    ensures ({
        let s = stride(&cache);
        let dst_offset = cache.seq_len * s;
        let src_len = num_new_tokens * s;
        let ls = layer_size(&cache);
        let new_cache = KvCacheState {
            seq_len: cache.seq_len + num_new_tokens,
            ..cache
        };
        &&& dst_offset + src_len <= ls
        &&& well_formed(&new_cache)
    }),
{
    let s = stride(&cache);
    assert((cache.seq_len + num_new_tokens) * s <= cache.max_seq_len * s)
        by (nonlinear_arith)
        requires cache.seq_len + num_new_tokens <= cache.max_seq_len;
    assert(cache.seq_len * s + num_new_tokens * s == (cache.seq_len + num_new_tokens) * s)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: reset() restores well-formedness with seq_len == 0
// ============================================================================

proof fn reset_restores_well_formed(cache: KvCacheState)
    requires
        well_formed(&cache),
    ensures ({
        let new_cache = KvCacheState { seq_len: 0, ..cache };
        &&& well_formed(&new_cache)
        &&& new_cache.seq_len == 0
        &&& remaining_capacity(&new_cache) == cache.max_seq_len
    }),
{
}

// ============================================================================
// Proof: capacity conservation — seq_len + remaining == max_seq_len always
// ============================================================================

proof fn capacity_conservation(cache: &KvCacheState)
    requires well_formed(cache),
    ensures
        cache.seq_len + remaining_capacity(cache) == cache.max_seq_len,
{
}

// ============================================================================
// Proof: multiple advances accumulate correctly
// ============================================================================

proof fn sequential_advances(
    cache: KvCacheState,
    n1: nat,
    n2: nat,
)
    requires
        well_formed(&cache),
        n1 + n2 <= remaining_capacity(&cache),
        n1 >= 1,
        n2 >= 1,
    ensures ({
        let after_first = KvCacheState { seq_len: cache.seq_len + n1, ..cache };
        let after_both = KvCacheState { seq_len: cache.seq_len + n1 + n2, ..cache };
        &&& well_formed(&after_first)
        &&& well_formed(&after_both)
        &&& after_both.seq_len == cache.seq_len + n1 + n2
    }),
{
}

fn main() {}

} // verus!

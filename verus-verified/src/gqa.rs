//! Verus-verified GQA (Grouped Query Attention) head mapping -- unbounded proofs
//!
//! Proves the core GQA correctness property: for every query head h,
//! the KV head h_kv = h / (num_heads / num_kv_heads) is valid (< num_kv_heads).
//! This is the fundamental invariant that makes GQA memory-safe.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: GQA head mapping -- every query head maps to a valid KV head
// ============================================================================

proof fn gqa_head_mapping_valid(
    num_heads: nat,
    num_kv_heads: nat,
    h: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        num_heads >= num_kv_heads,
        num_heads % num_kv_heads == 0,
        h < num_heads,
    ensures ({
        let ratio = num_heads / num_kv_heads;
        let h_kv = h / ratio;
        h_kv < num_kv_heads
    }),
{
    assert(num_heads / num_kv_heads >= 1)
        by (nonlinear_arith)
        requires num_heads >= num_kv_heads, num_kv_heads >= 1;
    assert((num_heads / num_kv_heads) * num_kv_heads == num_heads)
        by (nonlinear_arith)
        requires num_heads % num_kv_heads == 0, num_kv_heads >= 1;
    assert(h / (num_heads / num_kv_heads) < num_kv_heads)
        by (nonlinear_arith)
        requires h < num_heads, num_heads / num_kv_heads >= 1,
                 (num_heads / num_kv_heads) * num_kv_heads == num_heads;
}

// ============================================================================
// Proof: GQA ratio is always >= 1 (more Q heads than KV heads)
// ============================================================================

proof fn gqa_ratio_at_least_one(num_heads: nat, num_kv_heads: nat)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        num_heads >= num_kv_heads,
        num_heads % num_kv_heads == 0,
    ensures
        num_heads / num_kv_heads >= 1,
{
    assert(num_heads / num_kv_heads >= 1)
        by (nonlinear_arith)
        requires num_heads >= num_kv_heads, num_kv_heads >= 1;
}

// ============================================================================
// Proof: MHA is a special case of GQA (num_heads == num_kv_heads, ratio == 1)
// ============================================================================

proof fn mha_is_gqa_identity(num_heads: nat, h: nat)
    requires
        num_heads >= 1,
        h < num_heads,
    ensures ({
        let ratio = num_heads / num_heads;
        let h_kv = h / ratio;
        h_kv == h
    }),
{
    assert(num_heads / num_heads == 1)
        by (nonlinear_arith) requires num_heads >= 1;
}

// ============================================================================
// Proof: MQA is a special case (num_kv_heads == 1, all Q heads share one KV)
// ============================================================================

proof fn mqa_all_share_one_kv(num_heads: nat, h: nat)
    requires
        num_heads >= 1,
        h < num_heads,
    ensures ({
        let ratio = num_heads / 1;
        let h_kv = h / ratio;
        h_kv == 0
    }),
{
    assert(num_heads / 1 == num_heads)
        by (nonlinear_arith) requires num_heads >= 1;
    assert(h / num_heads == 0)
        by (nonlinear_arith) requires h < num_heads;
}

// ============================================================================
// Proof: consecutive Q heads in same group map to same KV head
// ============================================================================

proof fn same_group_same_kv(
    num_heads: nat,
    num_kv_heads: nat,
    h1: nat,
    h2: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        num_heads >= num_kv_heads,
        num_heads % num_kv_heads == 0,
        h1 < num_heads,
        h2 < num_heads,
        h1 / (num_heads / num_kv_heads) == h2 / (num_heads / num_kv_heads),
    ensures ({
        let ratio = num_heads / num_kv_heads;
        h1 / ratio == h2 / ratio
    }),
{
}

// ============================================================================
// Proof: KV buffer size with GQA -- num_kv_heads * head_dim (not num_heads * head_dim)
// ============================================================================

proof fn gqa_kv_buffer_size(
    num_heads: nat,
    num_kv_heads: nat,
    head_dim: nat,
    seq_len: nat,
    h_kv: nat,
    s: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        seq_len >= 1,
        num_heads % num_kv_heads == 0,
        h_kv < num_kv_heads,
        s < seq_len,
    ensures ({
        let kv_stride = num_kv_heads * head_dim;
        let kv_buf_size = seq_len * kv_stride;
        let offset = (s * num_kv_heads + h_kv) * head_dim;
        offset + head_dim <= kv_buf_size
    }),
{
    let kv_stride = num_kv_heads * head_dim;
    assert(s * num_kv_heads + h_kv < seq_len * num_kv_heads)
        by (nonlinear_arith) requires s < seq_len, h_kv < num_kv_heads;
    assert(s * num_kv_heads + h_kv + 1 <= seq_len * num_kv_heads)
        by (nonlinear_arith) requires s * num_kv_heads + h_kv < seq_len * num_kv_heads;
    assert((s * num_kv_heads + h_kv + 1) * head_dim <= seq_len * num_kv_heads * head_dim)
        by (nonlinear_arith)
        requires s * num_kv_heads + h_kv + 1 <= seq_len * num_kv_heads;
    assert(seq_len * num_kv_heads * head_dim == seq_len * (num_kv_heads * head_dim))
        by (nonlinear_arith);
    assert((s * num_kv_heads + h_kv) * head_dim + head_dim
        == (s * num_kv_heads + h_kv + 1) * head_dim)
        by (nonlinear_arith);
}

fn main() {}

} // verus!

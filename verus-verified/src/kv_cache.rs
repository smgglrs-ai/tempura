//! Verus-verified KV cache indexing -- unbounded proofs
//!
//! This module proves the KvCache indexing arithmetic from engine/src/kv_cache.rs
//! for ALL possible parameter values, not just the bounded instances Kani checks
//! (num_kv_heads<=4, head_dim<=8, max_seq_len<=8).
//!
//! Verification hierarchy for KV cache safety:
//!   - Kani: 5 bounded harnesses (small parameter ranges)
//!   - Verus (this file): Deductive proof for any valid parameters

use vstd::prelude::*;

verus! {

// ============================================================================
// Helper lemmas for nonlinear arithmetic on naturals
// ============================================================================

proof fn lemma_mul_monotone(a: nat, b: nat, c: nat)
    requires a <= b
    ensures a * c <= b * c
{
    assert(a * c <= b * c) by (nonlinear_arith)
        requires a <= b, c >= 0;
}

proof fn lemma_mul_distributes_add(a: nat, b: nat, c: nat)
    ensures (a + b) * c == a * c + b * c
{
    assert((a + b) * c == a * c + b * c) by (nonlinear_arith);
}

proof fn lemma_mod_bound(a: nat, b: nat)
    requires b >= 1
    ensures a % b < b
{
    assert(a % b < b) by (nonlinear_arith)
        requires b >= 1;
}

proof fn lemma_mul_strict_monotone(a: nat, b: nat, c: nat)
    requires a < b, c >= 1
    ensures a * c < b * c
{
    assert(a * c < b * c) by (nonlinear_arith)
        requires a < b, c >= 1;
}

proof fn lemma_div_monotone(a: nat, b: nat, c: nat)
    requires a < b, c >= 1
    ensures (a as int) / (c as int) <= (b as int) / (c as int)
{
    assert((a as int) / (c as int) <= (b as int) / (c as int)) by (nonlinear_arith)
        requires a < b, c >= 1;
}

// ============================================================================
// Specification: what makes KV cache parameters valid
// ============================================================================

pub open spec fn valid_cache_params(
    num_kv_heads: nat,
    head_dim: nat,
    max_seq_len: nat,
) -> bool {
    &&& num_kv_heads >= 1
    &&& head_dim >= 1
    &&& max_seq_len >= 1
}

pub open spec fn valid_seq_state(
    seq_len: nat,
    max_seq_len: nat,
) -> bool {
    seq_len <= max_seq_len
}

// ============================================================================
// Proof: append_kv offsets stay in bounds for any valid parameters
// ============================================================================

proof fn append_kv_offsets_in_bounds(
    num_kv_heads: nat,
    head_dim: nat,
    max_seq_len: nat,
    seq_len: nat,
    num_new_tokens: nat,
)
    requires
        valid_cache_params(num_kv_heads, head_dim, max_seq_len),
        valid_seq_state(seq_len, max_seq_len),
        num_new_tokens >= 1,
        seq_len + num_new_tokens <= max_seq_len,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let layer_size = max_seq_len * stride;
        let dst_offset = seq_len * stride;
        let src_len = num_new_tokens * stride;
        dst_offset + src_len <= layer_size
    }),
{
    let stride = num_kv_heads * head_dim;
    lemma_mul_distributes_add(seq_len, num_new_tokens, stride);
    lemma_mul_monotone(seq_len + num_new_tokens, max_seq_len, stride);
}

// ============================================================================
// Proof: k_slice / v_slice length stays within layer allocation
// ============================================================================

proof fn k_v_slice_in_bounds(
    num_kv_heads: nat,
    head_dim: nat,
    max_seq_len: nat,
    seq_len: nat,
)
    requires
        valid_cache_params(num_kv_heads, head_dim, max_seq_len),
        valid_seq_state(seq_len, max_seq_len),
    ensures
        seq_len * num_kv_heads * head_dim <= max_seq_len * num_kv_heads * head_dim,
{
    assert(seq_len * num_kv_heads * head_dim <= max_seq_len * num_kv_heads * head_dim)
        by (nonlinear_arith)
        requires seq_len <= max_seq_len;
}

// ============================================================================
// Proof: paged KV cache block table offset stays in pool
// ============================================================================

proof fn paged_block_table_offset_in_bounds(
    block_size: nat,
    num_kv_heads: nat,
    head_dim: nat,
    total_pages: nat,
    pos: nat,
    phys_page: nat,
)
    requires
        block_size >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        total_pages >= 1,
        phys_page < total_pages,
        pos < total_pages * block_size,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let page_stride = block_size * stride;
        let pool_size = total_pages * page_stride;
        let block_offset = pos % block_size;
        let page_base = phys_page * page_stride + block_offset * stride;
        page_base + stride <= pool_size
    }),
{
    let stride = num_kv_heads * head_dim;
    let page_stride = block_size * stride;
    let block_offset = pos % block_size;

    // block_offset < block_size
    lemma_mod_bound(pos, block_size);

    // block_offset + 1 <= block_size
    // => (block_offset + 1) * stride <= block_size * stride = page_stride
    lemma_mul_monotone(block_offset + 1, block_size, stride);

    // phys_page + 1 <= total_pages
    // => (phys_page + 1) * page_stride <= total_pages * page_stride
    lemma_mul_monotone(phys_page + 1, total_pages, page_stride);

    // phys_page * page_stride + page_stride <= total_pages * page_stride
    lemma_mul_distributes_add(phys_page, 1, page_stride);

    // block_offset * stride + stride <= page_stride
    lemma_mul_distributes_add(block_offset, 1, stride);

    // Combine: page_base + stride
    //   = phys_page * page_stride + block_offset * stride + stride
    //   <= phys_page * page_stride + page_stride   (since block_offset * stride + stride <= page_stride)
    //   <= total_pages * page_stride               (since (phys_page+1) * page_stride <= total_pages * page_stride)
}

// ============================================================================
// Proof: paged block index stays within max_blocks
// ============================================================================

proof fn paged_block_index_in_bounds(
    block_size: nat,
    max_seq_len: nat,
    pos: nat,
)
    requires
        block_size >= 1,
        max_seq_len >= 1,
        pos < max_seq_len,
    ensures ({
        let ceil_div_numer = (max_seq_len + block_size - 1) as nat;
        let max_blocks = ceil_div_numer / block_size;
        let block_idx = pos / block_size;
        block_idx < max_blocks
    }),
{
    let ceil_div_numer = (max_seq_len + block_size - 1) as nat;
    assert(pos / block_size < ceil_div_numer / block_size)
        by (nonlinear_arith)
        requires pos < max_seq_len, block_size >= 1, ceil_div_numer == max_seq_len + block_size - 1;
}

// ============================================================================
// Proof: remaining capacity conservation (seq_len + remaining = max_seq_len)
// ============================================================================

proof fn remaining_capacity_consistent(
    max_seq_len: nat,
    seq_len: nat,
)
    requires
        max_seq_len >= 1,
        seq_len <= max_seq_len,
    ensures ({
        let remaining = max_seq_len - seq_len;
        seq_len + remaining == max_seq_len
    }),
{
}

fn main() {}

} // verus!

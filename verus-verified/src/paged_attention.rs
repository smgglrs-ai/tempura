//! Verus-verified paged attention offset arithmetic -- unbounded proofs
//!
//! Generalizes the Kani harnesses `paged_row_offset_in_bounds` and
//! `paged_row_offset_identity_table` (kernels/src/attention.rs) to ALL
//! valid parameter values: the tile->page->offset chain
//!   block_idx = token / block_size
//!   page      = block_table[block_idx]
//!   offset    = (page * block_size + token % block_size) * stride
//!               + kv_h * head_dim
//! stays within the paged pool whenever every table entry is a valid page
//! id, and collapses to the contiguous-cache offset under an identity
//! table.  This is the arithmetic the cutile paged gather mirrors on GPU
//! (plan p20-paged-attention-kernel).

use vstd::prelude::*;
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_mod_bound};

verus! {

spec fn paged_offset(block_table: Seq<nat>, block_size: nat, stride: nat, token: nat, kv_h: nat, head_dim: nat) -> nat
    recommends
        block_size >= 1,
        token < block_table.len() * block_size,
{
    let block_idx = token / block_size;
    let page = block_table[block_idx as int];
    let block_offset = token % block_size;
    (page * block_size + block_offset) * stride + kv_h * head_dim
}

// ============================================================================
// Proof: the paged offset stays within the pool
// ============================================================================

proof fn lemma_paged_offset_in_bounds(
    block_table: Seq<nat>,
    block_size: nat,
    num_kv_heads: nat,
    head_dim: nat,
    num_pool_pages: nat,
    token: nat,
    kv_h: nat,
)
    requires
        block_size >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        block_table.len() >= 1,
        forall|i: int| 0 <= i < block_table.len() ==> block_table[i] < num_pool_pages,
        token < block_table.len() * block_size,
        kv_h < num_kv_heads,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let pool_len = num_pool_pages * block_size * stride;
        paged_offset(block_table, block_size, stride, token, kv_h, head_dim) + head_dim <= pool_len
    }),
{
    let block_idx = token / block_size;
    let page = block_table[block_idx as int];
    let block_offset = token % block_size;
    let stride = num_kv_heads * head_dim;

    // Fundamental div/mod: token == block_size * block_idx + block_offset
    // and block_offset < block_size.
    lemma_fundamental_div_mod(token as int, block_size as int);
    lemma_mod_bound(token as int, block_size as int);

    // block_idx < block_table.len(): block_size * block_idx <= token and
    // token < len * block_size, so block_size * block_idx < len * block_size.
    assert(block_size as int * (block_idx as int) < (block_table.len() as int) * (block_size as int))
        by (nonlinear_arith)
        requires
            token as int == (block_size as int) * (block_idx as int) + (block_offset as int),
            (token as int) < (block_table.len() as int) * (block_size as int);
    assert(block_idx < block_table.len()) by (nonlinear_arith)
        requires
            (block_size as int) * (block_idx as int) < (block_table.len() as int) * (block_size as int),
            block_size >= 1;

    // The row stays inside its page: page * block_size + block_offset
    // < (page + 1) * block_size <= num_pool_pages * block_size.
    let row = page * block_size + block_offset;
    assert((block_offset as int) + 1 <= (block_size as int));
    assert((page as int) + 1 <= (num_pool_pages as int));
    assert(((page as int) + 1) * (block_size as int)
        <= (num_pool_pages as int) * (block_size as int))
        by (nonlinear_arith)
        requires (page as int) + 1 <= (num_pool_pages as int);
    assert((page as int) * (block_size as int) + (block_offset as int) + 1
        <= ((page as int) + 1) * (block_size as int))
        by (nonlinear_arith)
        requires (block_offset as int) + 1 <= (block_size as int);
    assert((row as int) + 1 <= (num_pool_pages as int) * (block_size as int));

    // row * stride + (kv_h + 1) * head_dim <= (row + 1) * stride
    // <= num_pool_pages * block_size * stride.
    assert(row * stride + (kv_h + 1) * head_dim <= num_pool_pages * block_size * stride)
        by (nonlinear_arith)
        requires
            row + 1 <= num_pool_pages * block_size,
            stride == num_kv_heads * head_dim,
            kv_h < num_kv_heads,
            head_dim >= 1;

    // Bridge to the postcondition's shape.
    assert(paged_offset(block_table, block_size, stride, token, kv_h, head_dim)
        == row * stride + kv_h * head_dim);
    assert(row * stride + kv_h * head_dim + head_dim
        == row * stride + (kv_h + 1) * head_dim)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: identity table collapses to the contiguous offset
// ============================================================================

proof fn lemma_paged_offset_identity_table(
    block_size: nat,
    num_kv_heads: nat,
    head_dim: nat,
    num_blocks: nat,
    token: nat,
    kv_h: nat,
)
    requires
        block_size >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        token < num_blocks * block_size,
        kv_h < num_kv_heads,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let ident = Seq::new(num_blocks, |i: int| i as nat);
        paged_offset(ident, block_size, stride, token, kv_h, head_dim)
            == token * stride + kv_h * head_dim
    }),
{
    let block_idx = token / block_size;
    let block_offset = token % block_size;

    lemma_fundamental_div_mod(token as int, block_size as int);
    lemma_mod_bound(token as int, block_size as int);

    assert(block_idx < num_blocks) by (nonlinear_arith)
        requires
            token as int == (block_size as int) * (block_idx as int) + (block_offset as int),
            (token as int) < (num_blocks as int) * (block_size as int),
            block_size >= 1;

    let ident = Seq::new(num_blocks, |i: int| i as nat);
    assert(ident[block_idx as int] == block_idx);
    assert(ident.len() == num_blocks);

    // page == block_idx: block_idx * block_size + block_offset == token.
    assert((block_idx as int) * (block_size as int) + (block_offset as int) == token as int)
        by (nonlinear_arith)
        requires
            token as int == (block_size as int) * (block_idx as int) + (block_offset as int);
}

// ============================================================================
// Proof: distinct tokens in the same page map to distinct rows
// (no aliasing within a page -- two tokens of one logical sequence never
// share a K/V row, the write-safety side of the scatter)
// ============================================================================

proof fn lemma_paged_rows_distinct_within_page(
    block_size: nat,
    stride: nat,
    page: nat,
    t1: nat,
    t2: nat,
)
    requires
        block_size >= 1,
        stride >= 1,
        t1 < block_size,
        t2 < block_size,
        t1 != t2,
    ensures
        (page * block_size + t1) * stride != (page * block_size + t2) * stride,
{
    assert(page * block_size + t1 != page * block_size + t2) by (nonlinear_arith)
        requires t1 != t2;
    assert((page * block_size + t1) * stride != (page * block_size + t2) * stride)
        by (nonlinear_arith)
        requires
            page * block_size + t1 != page * block_size + t2,
            stride >= 1;
}

} // verus!

fn main() {}

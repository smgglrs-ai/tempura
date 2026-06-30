//! Verus-verified attention kernel indexing — unbounded proofs
//!
//! Proves 4D tensor indexing (batch × seq × heads × head_dim) for attention
//! kernels, batch isolation (non-overlapping memory regions), and GPU thread
//! decomposition — all for ANY valid parameter values.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: Q tensor offset ((b*seq+s)*num_heads+h)*head_dim stays in bounds
// ============================================================================

proof fn attention_q_offset_in_bounds(
    batch: nat,
    seq_len: nat,
    num_heads: nat,
    head_dim: nat,
    b: nat,
    s: nat,
    h: nat,
)
    requires
        batch >= 1,
        seq_len >= 1,
        num_heads >= 1,
        head_dim >= 1,
        b < batch,
        s < seq_len,
        h < num_heads,
    ensures ({
        let q_len = batch * seq_len * num_heads * head_dim;
        let q_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        q_offset + head_dim <= q_len
    }),
{
    // b*seq_len + s < batch*seq_len (since b < batch, s < seq_len)
    assert(b * seq_len + s < batch * seq_len)
        by (nonlinear_arith)
        requires b < batch, s < seq_len;

    // (b*seq_len + s)*num_heads + h < batch*seq_len*num_heads
    assert((b * seq_len + s) * num_heads + h < batch * seq_len * num_heads)
        by (nonlinear_arith)
        requires b * seq_len + s < batch * seq_len, h < num_heads;

    // ((b*seq+s)*nh+h)*hd + hd <= batch*seq*nh*hd
    assert(((b * seq_len + s) * num_heads + h + 1) * head_dim <= batch * seq_len * num_heads * head_dim)
        by (nonlinear_arith)
        requires (b * seq_len + s) * num_heads + h + 1 <= batch * seq_len * num_heads;

    assert(((b * seq_len + s) * num_heads + h) * head_dim + head_dim
        == ((b * seq_len + s) * num_heads + h + 1) * head_dim)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: KV tensor offset ((b*kv_len+s)*num_kv_heads+h)*head_dim in bounds
// ============================================================================

proof fn attention_kv_offset_in_bounds(
    batch: nat,
    kv_len: nat,
    num_kv_heads: nat,
    head_dim: nat,
    b: nat,
    kv_s: nat,
    kv_h: nat,
)
    requires
        batch >= 1,
        kv_len >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        b < batch,
        kv_s < kv_len,
        kv_h < num_kv_heads,
    ensures ({
        let k_len = batch * kv_len * num_kv_heads * head_dim;
        let k_offset = ((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim;
        k_offset + head_dim <= k_len
    }),
{
    assert(b * kv_len + kv_s < batch * kv_len)
        by (nonlinear_arith)
        requires b < batch, kv_s < kv_len;

    assert((b * kv_len + kv_s) * num_kv_heads + kv_h < batch * kv_len * num_kv_heads)
        by (nonlinear_arith)
        requires b * kv_len + kv_s < batch * kv_len, kv_h < num_kv_heads;

    assert(((b * kv_len + kv_s) * num_kv_heads + kv_h + 1) * head_dim
        <= batch * kv_len * num_kv_heads * head_dim)
        by (nonlinear_arith)
        requires (b * kv_len + kv_s) * num_kv_heads + kv_h + 1
            <= batch * kv_len * num_kv_heads;

    assert(((b * kv_len + kv_s) * num_kv_heads + kv_h) * head_dim + head_dim
        == ((b * kv_len + kv_s) * num_kv_heads + kv_h + 1) * head_dim)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: batches occupy non-overlapping memory regions
// ============================================================================

proof fn attention_batch_isolation_no_overlap(
    batch: nat,
    seq_len: nat,
    num_heads: nat,
    head_dim: nat,
    b1: nat,
    b2: nat,
)
    requires
        batch >= 2,
        seq_len >= 1,
        num_heads >= 1,
        head_dim >= 1,
        b1 < batch,
        b2 < batch,
        b1 != b2,
    ensures ({
        let stride = seq_len * num_heads * head_dim;
        let start1 = b1 * stride;
        let end1 = start1 + stride;
        let start2 = b2 * stride;
        let end2 = start2 + stride;
        end1 <= start2 || end2 <= start1
    }),
{
    let stride = seq_len * num_heads * head_dim;
    // WLOG b1 < b2 or b2 < b1. In either case, the smaller batch's
    // end = (smaller+1)*stride <= larger*stride = larger's start.
    assert((b1 + 1) * stride <= b2 * stride || (b2 + 1) * stride <= b1 * stride)
        by (nonlinear_arith)
        requires b1 != b2;

    assert(b1 * stride + stride == (b1 + 1) * stride)
        by (nonlinear_arith);
    assert(b2 * stride + stride == (b2 + 1) * stride)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: GPU thread decomposition — flat index i decomposes into valid (b,s,h,d)
// and reconstructed offset is in bounds
// ============================================================================

proof fn gpu_attention_thread_decomposition_in_bounds(
    batch: nat,
    seq_len: nat,
    num_heads: nat,
    head_dim: nat,
    i: nat,
)
    requires
        batch >= 1,
        seq_len >= 1,
        num_heads >= 1,
        head_dim >= 1,
        i < batch * seq_len * num_heads * head_dim,
    ensures ({
        let hd = head_dim;
        let nh = num_heads;
        let sl = seq_len;
        let d = i % hd;
        let h = (i / hd) % nh;
        let s = (i / hd / nh) % sl;
        let b = i / hd / nh / sl;
        &&& b < batch
        &&& s < seq_len
        &&& h < num_heads
        &&& d < head_dim
        &&& i < batch * seq_len * num_heads * head_dim
    }),
{
    let hd = head_dim;
    let nh = num_heads;
    let sl = seq_len;

    assert(i % hd < hd) by (nonlinear_arith) requires hd >= 1;
    assert((i / hd) % nh < nh) by (nonlinear_arith) requires nh >= 1;
    assert((i / hd / nh) % sl < sl) by (nonlinear_arith) requires sl >= 1;

    assert(i / hd < batch * sl * nh)
        by (nonlinear_arith)
        requires i < batch * sl * nh * hd, hd >= 1;
    assert(i / hd / nh < batch * sl)
        by (nonlinear_arith)
        requires i / hd < batch * sl * nh, nh >= 1;
    assert(i / hd / nh / sl < batch)
        by (nonlinear_arith)
        requires i / hd / nh < batch * sl, sl >= 1;
}

fn main() {}

} // verus!

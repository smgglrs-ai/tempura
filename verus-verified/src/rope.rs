//! Verus-verified RoPE indexing — unbounded proofs
//!
//! Proves 4D tensor indexing with half_dim pairing (RoPE applies rotations
//! to pairs of dimensions) and GPU thread decomposition.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: RoPE offset and paired-dim offset both stay in bounds
// ============================================================================

proof fn rope_indices_in_bounds(
    batch: nat,
    seq_len: nat,
    num_heads: nat,
    head_dim: nat,
    b: nat,
    s: nat,
    h: nat,
    d: nat,
)
    requires
        batch >= 1,
        seq_len >= 1,
        num_heads >= 1,
        head_dim >= 2,
        head_dim % 2 == 0,
        b < batch,
        s < seq_len,
        h < num_heads,
        d < head_dim / 2,
    ensures ({
        let x_len = batch * seq_len * num_heads * head_dim;
        let pos_len = batch * seq_len;
        let half_dim = head_dim / 2;
        let offset = ((b * seq_len + s) * num_heads + h) * head_dim;
        &&& b * seq_len + s < pos_len
        &&& offset + d < x_len
        &&& offset + d + half_dim < x_len
    }),
{
    let half_dim = head_dim / 2;

    // b*seq_len + s < batch*seq_len
    assert(b * seq_len + s < batch * seq_len)
        by (nonlinear_arith)
        requires b < batch, s < seq_len;

    // (b*seq+s)*nh+h < batch*seq*nh
    assert((b * seq_len + s) * num_heads + h < batch * seq_len * num_heads)
        by (nonlinear_arith)
        requires b * seq_len + s < batch * seq_len, h < num_heads;

    // ((b*seq+s)*nh+h+1)*hd <= batch*seq*nh*hd
    assert(((b * seq_len + s) * num_heads + h + 1) * head_dim
        <= batch * seq_len * num_heads * head_dim)
        by (nonlinear_arith)
        requires (b * seq_len + s) * num_heads + h + 1 <= batch * seq_len * num_heads;

    // offset + d + half_dim < offset + head_dim <= x_len
    assert(d + half_dim < head_dim)
        by (nonlinear_arith)
        requires d < half_dim, half_dim == head_dim / 2, head_dim % 2 == 0;

    assert(((b * seq_len + s) * num_heads + h) * head_dim + head_dim
        == ((b * seq_len + s) * num_heads + h + 1) * head_dim)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: GPU RoPE thread decomposition — flat index decomposes into valid (s,h,d)
// ============================================================================

proof fn gpu_rope_thread_decomposition_in_bounds(
    seq_len: nat,
    num_heads: nat,
    head_dim: nat,
    i: nat,
)
    requires
        seq_len >= 1,
        num_heads >= 1,
        head_dim >= 2,
        head_dim % 2 == 0,
        i < seq_len * num_heads * head_dim,
    ensures ({
        let hd = head_dim;
        let d_local = i % hd;
        let h = (i / hd) % num_heads;
        let s = i / hd / num_heads;
        let offset = (s * num_heads + h) * hd;
        &&& s < seq_len
        &&& h < num_heads
        &&& d_local < head_dim
        &&& offset + d_local < seq_len * num_heads * head_dim
    }),
{
    let hd = head_dim;
    assert(i % hd < hd) by (nonlinear_arith) requires hd >= 1;
    assert((i / hd) % num_heads < num_heads) by (nonlinear_arith) requires num_heads >= 1;
    assert(i / hd < seq_len * num_heads)
        by (nonlinear_arith)
        requires i < seq_len * num_heads * hd, hd >= 1;
    assert(i / hd / num_heads < seq_len)
        by (nonlinear_arith)
        requires i / hd < seq_len * num_heads, num_heads >= 1;

    // Reconstruct: offset + d_local == i
    assert(i == (i / hd / num_heads * num_heads + (i / hd) % num_heads) * hd + i % hd)
        by (nonlinear_arith)
        requires hd >= 1, num_heads >= 1;
}

fn main() {}

} // verus!

//! Verus-verified disaggregated serving dimensions -- unbounded proofs

use vstd::prelude::*;

verus! {

proof fn from_transfer_dimensions_consistent(
    num_layers: nat,
    num_kv_heads: nat,
    head_dim: nat,
    seq_len: nat,
    max_seq_len: nat,
)
    requires
        num_layers >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
        seq_len >= 1,
        max_seq_len >= seq_len,
    ensures ({
        let stride = num_kv_heads * head_dim;
        let expected_kv_len = seq_len * stride;
        let layer_size = max_seq_len * stride;
        let src_len = seq_len * stride;
        &&& src_len <= layer_size
        &&& src_len == expected_kv_len
    }),
{
    let stride = num_kv_heads * head_dim;
    assert(seq_len * stride <= max_seq_len * stride)
        by (nonlinear_arith)
        requires seq_len <= max_seq_len;
}

fn main() {}

} // verus!

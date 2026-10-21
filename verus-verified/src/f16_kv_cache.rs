//! Verus-verified f16 KV cache properties -- unbounded proofs

use vstd::prelude::*;

verus! {

proof fn f16_normal_to_f32_exp_valid(exp: nat)
    requires 1 <= exp && exp <= 30,
    ensures ({
        let f32_exp = exp + 112;
        &&& f32_exp >= 113
        &&& f32_exp <= 142
        &&& f32_exp < 255
    }),
{
}

proof fn f16_frac_shift_valid(frac: nat)
    requires frac < 1024,
    ensures frac * 8192 < 8388608,
{
    assert(frac * 8192 < 1024 * 8192) by (nonlinear_arith)
        requires frac < 1024;
}

proof fn kv_cache_f16_index_bounds(
    max_seq: nat, num_kv_heads: nat, head_dim: nat,
    seq_len: nat, token: nat, head: nat, dim: nat,
)
    requires
        max_seq >= 1, num_kv_heads >= 1, head_dim >= 1,
        seq_len <= max_seq, token < seq_len,
        head < num_kv_heads, dim < head_dim,
    ensures
        token * (num_kv_heads * head_dim) + head * head_dim + dim
            < max_seq * (num_kv_heads * head_dim),
{
    assert(num_kv_heads * head_dim >= 1) by (nonlinear_arith)
        requires num_kv_heads >= 1, head_dim >= 1;
    assert((head + 1) * head_dim <= num_kv_heads * head_dim) by (nonlinear_arith)
        requires head < num_kv_heads, head_dim >= 1;
    assert((head + 1) * head_dim == head * head_dim + head_dim) by (nonlinear_arith);
    assert(head * head_dim + dim < head * head_dim + head_dim) by (nonlinear_arith)
        requires dim < head_dim;
    assert((token + 1) * (num_kv_heads * head_dim) <= seq_len * (num_kv_heads * head_dim)) by (nonlinear_arith)
        requires token < seq_len, num_kv_heads * head_dim >= 1;
    assert(seq_len * (num_kv_heads * head_dim) <= max_seq * (num_kv_heads * head_dim)) by (nonlinear_arith)
        requires seq_len <= max_seq, num_kv_heads * head_dim >= 1;
    // Chain: offset = token*stride + head*hd + dim < (token+1)*stride <= seq*stride <= max*stride
    assert(token * (num_kv_heads * head_dim) + head * head_dim + dim
        < (token + 1) * (num_kv_heads * head_dim)) by (nonlinear_arith)
        requires head * head_dim + dim < num_kv_heads * head_dim;
}

proof fn mmap_range_containment(file_size: nat, offset: nat, length: nat)
    requires offset + length <= file_size,
    ensures offset <= file_size && offset + length <= file_size,
{
}

}

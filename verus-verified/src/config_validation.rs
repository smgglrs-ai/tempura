//! Verus-verified config validation -- unbounded proofs
//!
//! Proves: ModelConfig presets satisfy all dimensional constraints,
//! from_hf_json derived head_dim is correct, tensor parallel splits evenly.

use vstd::prelude::*;

verus! {

pub open spec fn config_valid(
    hidden_size: nat,
    num_heads: nat,
    num_kv_heads: nat,
    head_dim: nat,
    intermediate_size: nat,
    num_layers: nat,
    vocab_size: nat,
    max_seq_len: nat,
) -> bool {
    &&& hidden_size >= 1
    &&& num_heads >= 1
    &&& num_kv_heads >= 1
    &&& head_dim >= 1
    &&& intermediate_size >= 1
    &&& num_layers >= 1
    &&& vocab_size >= 1
    &&& max_seq_len >= 1
    &&& hidden_size == num_heads * head_dim
    &&& hidden_size % num_heads == 0
    &&& num_heads % num_kv_heads == 0
    &&& num_heads >= num_kv_heads
}

// ============================================================================
// Proof: micro_llama preset is valid (GQA with 16 Q heads, 4 KV heads)
// ============================================================================

proof fn micro_llama_valid()
    ensures config_valid(1024, 16, 4, 64, 5632, 12, 32000, 2048),
{
}

// ============================================================================
// Proof: derived head_dim = hidden_size / num_attention_heads
// ============================================================================

proof fn derived_head_dim(hidden_size: nat, num_heads: nat)
    requires
        hidden_size >= 1,
        num_heads >= 1,
        hidden_size >= num_heads,
        hidden_size % num_heads == 0,
    ensures ({
        let head_dim = hidden_size / num_heads;
        &&& head_dim >= 1
        &&& head_dim * num_heads == hidden_size
    }),
{
    assert(hidden_size / num_heads >= 1)
        by (nonlinear_arith) requires hidden_size >= num_heads, num_heads >= 1;
    assert((hidden_size / num_heads) * num_heads == hidden_size)
        by (nonlinear_arith)
        requires hidden_size % num_heads == 0, num_heads >= 1;
}

// ============================================================================
// Proof: tensor_parallel splits heads evenly
// ============================================================================

proof fn tensor_parallel_heads(
    num_heads: nat,
    num_kv_heads: nat,
    tp_size: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        tp_size >= 1,
        num_heads >= tp_size,
        num_kv_heads >= tp_size,
        num_heads % tp_size == 0,
        num_kv_heads % tp_size == 0,
    ensures ({
        let local_heads = num_heads / tp_size;
        let local_kv_heads = num_kv_heads / tp_size;
        &&& local_heads >= 1
        &&& local_kv_heads >= 1
        &&& local_heads * tp_size == num_heads
        &&& local_kv_heads * tp_size == num_kv_heads
    }),
{
    assert(num_heads / tp_size >= 1)
        by (nonlinear_arith) requires num_heads >= tp_size, tp_size >= 1;
    assert((num_heads / tp_size) * tp_size == num_heads)
        by (nonlinear_arith) requires num_heads % tp_size == 0, tp_size >= 1;
    assert(num_kv_heads / tp_size >= 1)
        by (nonlinear_arith) requires num_kv_heads >= tp_size, tp_size >= 1;
    assert((num_kv_heads / tp_size) * tp_size == num_kv_heads)
        by (nonlinear_arith) requires num_kv_heads % tp_size == 0, tp_size >= 1;
}

// ============================================================================
// Proof: GQA ratio preserved after tensor parallel split
// ============================================================================

proof fn tp_preserves_gqa_ratio(
    num_heads: nat,
    num_kv_heads: nat,
    tp_size: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        tp_size >= 1,
        num_heads >= tp_size,
        num_kv_heads >= tp_size,
        num_heads % tp_size == 0,
        num_kv_heads % tp_size == 0,
        num_heads % num_kv_heads == 0,
        num_heads >= num_kv_heads,
    ensures ({
        let local_heads = num_heads / tp_size;
        let local_kv_heads = num_kv_heads / tp_size;
        // GQA ratio is the same locally as globally
        local_heads / local_kv_heads == num_heads / num_kv_heads
    }),
{
    // This follows from: (a/c)/(b/c) == a/b when a%c==0, b%c==0
    assert((num_heads / tp_size) / (num_kv_heads / tp_size) == num_heads / num_kv_heads)
        by (nonlinear_arith)
        requires num_heads % tp_size == 0, num_kv_heads % tp_size == 0,
                 num_heads % num_kv_heads == 0, tp_size >= 1, num_kv_heads >= 1,
                 num_heads >= num_kv_heads;
}

// ============================================================================
// Proof: sliding window doesn't affect buffer sizes
// ============================================================================

proof fn sliding_window_within_max_seq(
    window: nat,
    max_seq_len: nat,
)
    requires window >= 1,
    ensures ({
        let effective = if window < max_seq_len { window } else { max_seq_len };
        effective <= max_seq_len
    }),
{
}

fn main() {}

} // verus!

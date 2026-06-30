//! Verus-verified ModelConfig well-formedness — unbounded proofs
//!
//! Proves dimensional consistency: head_dim = hidden_size / num_heads,
//! GQA ratio is integral, and presets satisfy all invariants.

use vstd::prelude::*;

verus! {

pub open spec fn config_well_formed(
    hidden_size: nat,
    num_attention_heads: nat,
    num_kv_heads: nat,
    head_dim: nat,
    num_layers: nat,
    vocab_size: nat,
    max_seq_len: nat,
    intermediate_size: nat,
) -> bool {
    &&& hidden_size >= 1
    &&& num_attention_heads >= 1
    &&& num_kv_heads >= 1
    &&& head_dim >= 1
    &&& num_layers >= 1
    &&& vocab_size >= 1
    &&& max_seq_len >= 1
    &&& intermediate_size >= 1
    &&& head_dim == hidden_size / num_attention_heads
    &&& hidden_size % num_attention_heads == 0
    &&& num_attention_heads % num_kv_heads == 0
}

// ============================================================================
// Proof: head_dim * num_attention_heads == hidden_size
// ============================================================================

proof fn head_dim_times_heads_equals_hidden(
    hidden_size: nat,
    num_attention_heads: nat,
)
    requires
        hidden_size >= 1,
        num_attention_heads >= 1,
        hidden_size % num_attention_heads == 0,
    ensures ({
        let head_dim = hidden_size / num_attention_heads;
        head_dim * num_attention_heads == hidden_size
    }),
{
    assert((hidden_size / num_attention_heads) * num_attention_heads == hidden_size)
        by (nonlinear_arith)
        requires hidden_size % num_attention_heads == 0, num_attention_heads >= 1;
}

// ============================================================================
// Proof: GQA ratio is integral and maps every head to a valid KV head
// ============================================================================

proof fn gqa_ratio_integral(
    num_attention_heads: nat,
    num_kv_heads: nat,
)
    requires
        num_attention_heads >= 1,
        num_kv_heads >= 1,
        num_attention_heads >= num_kv_heads,
        num_attention_heads % num_kv_heads == 0,
    ensures ({
        let gqa_ratio = num_attention_heads / num_kv_heads;
        &&& gqa_ratio >= 1
        &&& gqa_ratio * num_kv_heads == num_attention_heads
    }),
{
    assert((num_attention_heads / num_kv_heads) * num_kv_heads == num_attention_heads)
        by (nonlinear_arith)
        requires num_attention_heads % num_kv_heads == 0, num_kv_heads >= 1;

    assert(num_attention_heads / num_kv_heads >= 1)
        by (nonlinear_arith)
        requires num_attention_heads >= num_kv_heads, num_kv_heads >= 1;
}

proof fn gqa_head_mapping(
    num_attention_heads: nat,
    num_kv_heads: nat,
    h: nat,
)
    requires
        num_attention_heads >= 1,
        num_kv_heads >= 1,
        num_attention_heads >= num_kv_heads,
        num_attention_heads % num_kv_heads == 0,
        h < num_attention_heads,
    ensures ({
        let gqa_ratio = num_attention_heads / num_kv_heads;
        let h_kv = h / gqa_ratio;
        h_kv < num_kv_heads
    }),
{
    assert(num_attention_heads / num_kv_heads >= 1)
        by (nonlinear_arith)
        requires num_attention_heads >= num_kv_heads, num_kv_heads >= 1;
    assert((num_attention_heads / num_kv_heads) * num_kv_heads == num_attention_heads)
        by (nonlinear_arith)
        requires num_attention_heads % num_kv_heads == 0, num_kv_heads >= 1;
    assert(h / (num_attention_heads / num_kv_heads) < num_kv_heads)
        by (nonlinear_arith)
        requires h < num_attention_heads, num_attention_heads / num_kv_heads >= 1,
                 (num_attention_heads / num_kv_heads) * num_kv_heads == num_attention_heads;
}

// ============================================================================
// Proof: llama2_7b preset satisfies config_well_formed
// ============================================================================

proof fn llama2_7b_well_formed()
    ensures
        config_well_formed(4096, 32, 32, 128, 32, 32000, 4096, 11008),
{
}

// ============================================================================
// Proof: tiny_test preset satisfies config_well_formed
// ============================================================================

proof fn tiny_test_well_formed()
    ensures
        config_well_formed(64, 4, 4, 16, 2, 256, 32, 128),
{
}

// ============================================================================
// Proof: embedding table size is vocab_size * hidden_size
// ============================================================================

proof fn embedding_table_size(vocab_size: nat, hidden_size: nat, tok: nat)
    requires
        vocab_size >= 1,
        hidden_size >= 1,
        tok < vocab_size,
    ensures ({
        let table_size = vocab_size * hidden_size;
        let offset = tok * hidden_size;
        offset + hidden_size <= table_size
    }),
{
    assert((tok + 1) * hidden_size <= vocab_size * hidden_size)
        by (nonlinear_arith) requires tok < vocab_size;
    assert(tok * hidden_size + hidden_size == (tok + 1) * hidden_size)
        by (nonlinear_arith);
}

fn main() {}

} // verus!

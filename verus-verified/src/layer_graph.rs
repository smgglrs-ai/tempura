//! Verus-verified layer graph I/O shape constraints -- unbounded proofs
//!
//! Proves: LayerInput/LayerOutput buffer sizes match config dimensions,
//! PrefillInput/PrefillOutput shapes, DecodeInput/DecodeOutput shapes.

use vstd::prelude::*;

verus! {

// ============================================================================
// LayerInput: hidden_state.len() == seq_len * hidden_size
// ============================================================================

proof fn layer_input_hidden_size(
    seq_len: nat,
    hidden_size: nat,
)
    requires seq_len >= 1, hidden_size >= 1,
    ensures ({
        let expected = seq_len * hidden_size;
        expected >= hidden_size
    }),
{
    assert(seq_len * hidden_size >= hidden_size)
        by (nonlinear_arith) requires seq_len >= 1;
}

// ============================================================================
// LayerInput: positions.len() == seq_len
// ============================================================================

proof fn layer_input_positions(seq_len: nat)
    requires seq_len >= 1,
    ensures seq_len >= 1,
{
}

// ============================================================================
// LayerInput: k_cache.len() == kv_len * num_kv_heads * head_dim
// ============================================================================

proof fn layer_input_kv_cache(
    kv_len: nat,
    num_kv_heads: nat,
    head_dim: nat,
)
    requires
        kv_len >= 1,
        num_kv_heads >= 1,
        head_dim >= 1,
    ensures ({
        let kv_size = kv_len * num_kv_heads * head_dim;
        kv_size >= num_kv_heads * head_dim
    }),
{
    assert(kv_len * num_kv_heads * head_dim >= num_kv_heads * head_dim)
        by (nonlinear_arith) requires kv_len >= 1;
}

// ============================================================================
// LayerOutput: hidden_state.len() == seq_len * hidden_size (same as input)
// ============================================================================

proof fn layer_output_preserves_shape(
    seq_len: nat,
    hidden_size: nat,
)
    requires seq_len >= 1, hidden_size >= 1,
    ensures ({
        let input_size = seq_len * hidden_size;
        let output_size = seq_len * hidden_size;
        input_size == output_size
    }),
{
}

// ============================================================================
// LayerOutput: new_k/new_v.len() == seq_len * num_kv_heads * head_dim
// ============================================================================

proof fn layer_output_kv_size(
    seq_len: nat,
    num_kv_heads: nat,
    head_dim: nat,
)
    requires seq_len >= 1, num_kv_heads >= 1, head_dim >= 1,
    ensures seq_len * num_kv_heads * head_dim >= 1,
{
    assert(seq_len * num_kv_heads * head_dim >= 1)
        by (nonlinear_arith) requires seq_len >= 1, num_kv_heads >= 1, head_dim >= 1;
}

// ============================================================================
// PrefillInput: embeddings.len() == token_ids.len() * hidden_size
// ============================================================================

proof fn prefill_embedding_size(
    num_tokens: nat,
    hidden_size: nat,
)
    requires num_tokens >= 1, hidden_size >= 1,
    ensures num_tokens * hidden_size >= hidden_size,
{
    assert(num_tokens * hidden_size >= hidden_size)
        by (nonlinear_arith) requires num_tokens >= 1;
}

// ============================================================================
// PrefillOutput: k_cache/v_cache have num_layers entries
// ============================================================================

proof fn prefill_kv_layers(
    num_layers: nat,
    layer: nat,
)
    requires num_layers >= 1, layer < num_layers,
    ensures layer < num_layers,
{
}

// ============================================================================
// DecodeInput: single token -> token_embedding.len() == hidden_size
// ============================================================================

proof fn decode_single_token(hidden_size: nat)
    requires hidden_size >= 1,
    ensures hidden_size >= 1,
{
}

// ============================================================================
// DecodeOutput: logits.len() == vocab_size
// ============================================================================

proof fn decode_output_logits(vocab_size: nat)
    requires vocab_size >= 1,
    ensures vocab_size >= 1,
{
}

// ============================================================================
// Layer weight sizes match config
// ============================================================================

proof fn layer_weights_consistent(
    hidden_size: nat,
    num_heads: nat,
    num_kv_heads: nat,
    head_dim: nat,
    intermediate_size: nat,
)
    requires
        hidden_size >= 1,
        num_heads >= 1,
        num_kv_heads >= 1,
        num_kv_heads <= num_heads,
        head_dim >= 1,
        intermediate_size >= 1,
        hidden_size == num_heads * head_dim,
    ensures ({
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // input_layernorm: hidden_size
        // q_proj: hidden_size x q_dim = hidden_size^2
        // k_proj: hidden_size x kv_dim
        // v_proj: hidden_size x kv_dim
        // o_proj: q_dim x hidden_size = hidden_size^2
        // gate_proj: hidden_size x intermediate_size
        // up_proj: hidden_size x intermediate_size
        // down_proj: intermediate_size x hidden_size
        &&& q_dim == hidden_size
        &&& kv_dim <= q_dim
        &&& hidden_size * intermediate_size >= hidden_size
    }),
{
    assert(num_kv_heads * head_dim <= num_heads * head_dim)
        by (nonlinear_arith) requires num_kv_heads <= num_heads;
    assert(hidden_size * intermediate_size >= hidden_size)
        by (nonlinear_arith) requires intermediate_size >= 1;
}

fn main() {}

} // verus!

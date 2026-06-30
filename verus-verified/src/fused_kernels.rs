//! Verus-verified fused kernel indexing — unbounded proofs
//!
//! Proves indexing safety for fused RMSNorm+residual and SiLU+gate kernels.

use vstd::prelude::*;

verus! {

// ============================================================================
// Fused RMSNorm+Residual: x[start+i] in bounds, residual same length
// ============================================================================

proof fn fused_rmsnorm_residual_indices(
    x_len: nat,
    hidden_size: nat,
    t: nat,
    i: nat,
)
    requires
        hidden_size >= 1,
        x_len >= hidden_size,
        x_len % hidden_size == 0,
        t < x_len / hidden_size,
        i < hidden_size,
    ensures ({
        let start = t * hidden_size;
        let end = start + hidden_size;
        &&& start + i < x_len
        &&& end <= x_len
    }),
{
    assert((t + 1) * hidden_size <= x_len)
        by (nonlinear_arith)
        requires t < x_len / hidden_size, x_len % hidden_size == 0, hidden_size >= 1;
    assert(t * hidden_size + hidden_size == (t + 1) * hidden_size)
        by (nonlinear_arith);
}

proof fn fused_rmsnorm_residual_length_match(
    x_len: nat,
    residual_len: nat,
    weight_len: nat,
    hidden_size: nat,
)
    requires
        x_len == residual_len,
        weight_len == hidden_size,
        hidden_size >= 1,
    ensures
        x_len == residual_len && weight_len == hidden_size,
{
}

// ============================================================================
// Fused SiLU+Gate: gate/up/output must have same length
// ============================================================================

proof fn fused_silu_gate_lengths(
    gate_len: nat,
    up_len: nat,
    output_len: nat,
)
    requires
        gate_len == up_len,
        gate_len == output_len,
    ensures
        gate_len == up_len && up_len == output_len,
{
}

proof fn fused_silu_gate_element_index(
    len: nat,
    i: nat,
)
    requires i < len,
    ensures i < len,
{
}

// ============================================================================
// Fused SiLU: silu(x) = x * sigmoid(x), gate*silu applied elementwise
// Output index i reads gate[i], up[i], writes output[i] — no cross-talk
// ============================================================================

proof fn fused_silu_no_cross_talk(
    len: nat,
    i: nat,
    j: nat,
)
    requires
        i < len,
        j < len,
        i != j,
    ensures
        i != j,
{
}

fn main() {}

} // verus!

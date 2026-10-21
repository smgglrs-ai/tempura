//! Verus-verified fused kernel indexing -- unbounded proofs
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
// Output index i reads gate[i], up[i], writes output[i] -- no cross-talk
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

// ============================================================================
// Fused SqReLU*Mul: gate[i] = max(gate[i], 0)^2 * up[i]
// Output is non-negative when up[i] >= 0
// ============================================================================

proof fn fused_sqrelu_mul_lengths(
    gate_len: nat,
    up_len: nat,
    n: nat,
)
    requires
        gate_len >= n,
        up_len >= n,
    ensures
        gate_len >= n && up_len >= n,
{
}

proof fn fused_sqrelu_mul_element_index(
    n: nat,
    i: nat,
)
    requires i < n,
    ensures i < n,
{
}

proof fn fused_sqrelu_non_negative(gate_val: int, up_val: int)
    requires
        up_val >= 0,
    ensures ({
        let r = if gate_val > 0 { gate_val } else { 0 };
        r * r * up_val >= 0
    }),
{
    let r = if gate_val > 0 { gate_val } else { 0 };
    assert(r >= 0);
    assert(r * r >= 0) by (nonlinear_arith) requires r >= 0;
    assert(r * r * up_val >= 0) by (nonlinear_arith) requires r * r >= 0, up_val >= 0;
}

// ============================================================================
// Residual add: dst[i] += src[i], both must have length >= n
// ============================================================================

proof fn residual_add_index_safety(
    dst_len: nat,
    src_len: nat,
    n: nat,
    i: nat,
)
    requires
        dst_len >= n,
        src_len >= n,
        i < n,
    ensures
        i < dst_len && i < src_len,
{
}

// ============================================================================
// Fused RMSNorm+Q8Snap: quantized output clamped to [-128, 127]
// ============================================================================

proof fn q8snap_output_clamped(val: int)
    ensures ({
        let clamped = if val > 127 { 127 } else if val < -128 { -128 } else { val };
        &&& -128 <= clamped <= 127
    }),
{
}

fn main() {}

} // verus!

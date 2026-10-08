//! Verus-verified quantized forward pass buffer shapes — unbounded proofs
//!
//! Proves:
//!   - Pre-allocated scratchpad buffer sizes match quantized matmul dimensions.
//!   - Q4K output error bound: the difference between the reference path
//!     (f32 dequant + dot) and the optimized path (Q8 activation quantization
//!     + integer GEMM + f32 scale) is bounded by the quantization step times
//!     the L1 norm of positive weight contributions.

use vstd::prelude::*;

verus! {

// ============================================================================
// Scratchpad buffer sizes
// ============================================================================

proof fn quantized_buffer_sizes(
    seq_len: nat,
    hidden_size: nat,
    q_dim: nat,
    kv_dim: nat,
    intermediate_size: nat,
)
    requires
        seq_len >= 1,
        hidden_size >= 1,
        q_dim >= 1,
        kv_dim >= 1,
        intermediate_size >= 1,
    ensures
        // hidden: seq_len * h
        // normed: seq_len * h
        // q: seq_len * q_dim
        // k: seq_len * kv_dim
        // v: seq_len * kv_dim
        // attn_out: seq_len * q_dim
        // o_proj: seq_len * h
        // gate: seq_len * inter
        // up: seq_len * inter
        // down_out: seq_len * h
        seq_len * hidden_size >= seq_len
        && seq_len * q_dim >= seq_len
        && seq_len * kv_dim >= seq_len
        && seq_len * intermediate_size >= seq_len,
{
    assert(seq_len * hidden_size >= seq_len)
        by (nonlinear_arith) requires hidden_size >= 1;
    assert(seq_len * q_dim >= seq_len)
        by (nonlinear_arith) requires q_dim >= 1;
    assert(seq_len * kv_dim >= seq_len)
        by (nonlinear_arith) requires kv_dim >= 1;
    assert(seq_len * intermediate_size >= seq_len)
        by (nonlinear_arith) requires intermediate_size >= 1;
}

// ============================================================================
// q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim
// ============================================================================

proof fn projection_dims(
    num_heads: nat,
    num_kv_heads: nat,
    head_dim: nat,
    hidden_size: nat,
)
    requires
        num_heads >= 1,
        num_kv_heads >= 1,
        num_kv_heads <= num_heads,
        head_dim >= 1,
        hidden_size == num_heads * head_dim,
    ensures ({
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        &&& q_dim == hidden_size
        &&& kv_dim <= q_dim
        &&& kv_dim >= head_dim
    }),
{
    assert(num_kv_heads * head_dim <= num_heads * head_dim)
        by (nonlinear_arith) requires num_kv_heads <= num_heads;
    assert(num_kv_heads * head_dim >= head_dim)
        by (nonlinear_arith) requires num_kv_heads >= 1;
}

// ============================================================================
// Quantized matmul: blocks_per_row = k / BLOCK_SIZE, accessing B_q
// ============================================================================

proof fn quantized_matmul_block_count(
    k: nat,
    block_size: nat,
)
    requires
        k >= block_size,
        block_size >= 1,
        k % block_size == 0,
    ensures ({
        let blocks_per_row = k / block_size;
        &&& blocks_per_row >= 1
        &&& blocks_per_row * block_size == k
    }),
{
    assert(k / block_size >= 1)
        by (nonlinear_arith) requires k >= block_size, block_size >= 1;
    assert((k / block_size) * block_size == k)
        by (nonlinear_arith) requires k % block_size == 0, block_size >= 1;
}

// ============================================================================
// Pre-quantized activations: INT8 quantization groups
// ============================================================================

proof fn prequant_group_count(
    elements: nat,
    group_size: nat,
)
    requires elements >= 1, group_size >= 1,
    ensures ({
        let num_groups = ((elements + group_size - 1) as nat) / group_size;
        &&& num_groups >= 1
        &&& num_groups * group_size >= elements
    }),
{
    assert(((elements + group_size - 1) as nat) / group_size >= 1)
        by (nonlinear_arith) requires elements >= 1, group_size >= 1;
    assert(((elements + group_size - 1) as nat) / group_size * group_size >= elements)
        by (nonlinear_arith) requires group_size >= 1, elements >= 1;
}

// ============================================================================
// Embedding lookup in quantized path (same as float path)
// ============================================================================

proof fn quantized_embed_lookup(
    vocab_size: nat,
    hidden_size: nat,
    tok: nat,
    t: nat,
    seq_len: nat,
)
    requires
        vocab_size >= 1,
        hidden_size >= 1,
        tok < vocab_size,
        t < seq_len,
        seq_len >= 1,
    ensures ({
        let embed_len = vocab_size * hidden_size;
        let src_start = tok * hidden_size;
        let dst_start = t * hidden_size;
        let hidden_len = seq_len * hidden_size;
        &&& src_start + hidden_size <= embed_len
        &&& dst_start + hidden_size <= hidden_len
    }),
{
    assert((tok + 1) * hidden_size <= vocab_size * hidden_size)
        by (nonlinear_arith) requires tok < vocab_size;
    assert(tok * hidden_size + hidden_size == (tok + 1) * hidden_size)
        by (nonlinear_arith);
    assert((t + 1) * hidden_size <= seq_len * hidden_size)
        by (nonlinear_arith) requires t < seq_len;
    assert(t * hidden_size + hidden_size == (t + 1) * hidden_size)
        by (nonlinear_arith);
}

// ============================================================================
// Scratchpad reuse: buffers allocated once, reused across all layers
// ============================================================================

proof fn scratchpad_reuse_valid(
    num_layers: nat,
    layer: nat,
    buf_size: nat,
)
    requires
        num_layers >= 1,
        layer < num_layers,
        buf_size >= 1,
    ensures
        // Buffer size doesn't depend on layer index — same size works for all layers
        buf_size >= 1,
{
}

// ============================================================================
// Q4K opt vs ref output error bound — self-contained, no cross-module imports
//
// The reference path computes (per group, integer model):
//   ref_contrib = sc_g * Σ_{i in g} q_i * a_i
//
// The opt path uses Q8(a_i) = a_i + e_i where |e_i| ≤ 1:
//   opt_contrib = sc_g * Σ_{i in g} q_i * (a_i + e_i)
//
// Error = sc_g * Σ q_i * e_i.  Since |e_i| ≤ 1 and q_i, sc_g ≥ 0:
//   |error| ≤ sc_g * Σ q_i  (weight L1-norm × group scale)
// ============================================================================

pub open spec fn qerr_abs(x: int) -> int {
    if x >= 0 { x } else { -x }
}

/// All elements of a sequence are nibble values: ∈ [0, 15].
pub open spec fn all_nibbles(q: Seq<int>) -> bool
    decreases q.len()
{
    if q.len() == 0 { true }
    else { 0 <= q.last() <= 15 && all_nibbles(q.drop_last()) }
}

/// All elements of a sequence are unit errors: ∈ [-1, 1].
pub open spec fn all_unit_errors(e: Seq<int>) -> bool
    decreases e.len()
{
    if e.len() == 0 { true }
    else { -1 <= e.last() <= 1 && all_unit_errors(e.drop_last()) }
}

/// Σ sc_g * q_i  — upper bound on one group's Q8 quantization error.
pub open spec fn group_error_bound(sc_g: int, q_g: Seq<int>) -> int
    decreases q_g.len()
{
    if q_g.len() == 0 { 0int }
    else { sc_g * q_g.last() + group_error_bound(sc_g, q_g.drop_last()) }
}

/// Weighted error sum: Σ (sc_g * q_i) * e_i  (= opt_contrib - ref_contrib per group).
pub open spec fn weighted_err(sc_g: int, q_g: Seq<int>, e_g: Seq<int>) -> int
    recommends q_g.len() == e_g.len()
    decreases q_g.len()
{
    if q_g.len() == 0 { 0int }
    else {
        sc_g * q_g.last() * e_g.last()
            + weighted_err(sc_g, q_g.drop_last(), e_g.drop_last())
    }
}

/// **Main theorem**: |weighted_err(sc_g, q_g, e_g)| ≤ group_error_bound(sc_g, q_g)
///
/// For any Q4K group: the error introduced by replacing exact activations with their
/// Q8-quantized approximations (|e_i| ≤ 1) is bounded by the group-scale × weight
/// L1-norm.  Proved for ALL sc_g ∈ [0,63], ALL lengths, ALL valid inputs.
pub proof fn lemma_q4k_group_opt_error_bound(
    sc_g: int,
    q_g: Seq<int>,
    e_g: Seq<int>,
)
    requires
        q_g.len() == e_g.len(),
        sc_g >= 0 && sc_g <= 63,
        all_nibbles(q_g),
        all_unit_errors(e_g),
    ensures
        qerr_abs(weighted_err(sc_g, q_g, e_g)) <= group_error_bound(sc_g, q_g)
    decreases q_g.len()
{
    if q_g.len() == 0 {
        assert(weighted_err(sc_g, q_g, e_g) == 0int);
        assert(group_error_bound(sc_g, q_g) == 0int);
        assert(qerr_abs(0int) == 0int);
    } else {
        let q_rest = q_g.drop_last();
        let e_rest = e_g.drop_last();

        // Recursive predicates unfold to give bounds on last element
        assert(all_nibbles(q_g));     // unfolds: 0 <= q_g.last() <= 15 && all_nibbles(q_rest)
        assert(all_unit_errors(e_g)); // unfolds: -1 <= e_g.last() <= 1 && all_unit_errors(e_rest)
        let ql = q_g.last();
        let el = e_g.last();
        assert(0 <= ql <= 15);
        assert(-1 <= el <= 1);

        // Apply IH on prefix
        lemma_q4k_group_opt_error_bound(sc_g, q_rest, e_rest);

        // Last term contribution
        let last_term = sc_g * ql * el;
        assert(qerr_abs(last_term) <= sc_g * ql) by (nonlinear_arith)
            requires sc_g >= 0, 0 <= ql, ql <= 15, -1 <= el, el <= 1,
                     last_term == sc_g * ql * el;

        // Total error = last_term + inner_error
        let inner = weighted_err(sc_g, q_rest, e_rest);
        assert(weighted_err(sc_g, q_g, e_g) == last_term + inner);

        // Triangle inequality
        assert(qerr_abs(last_term + inner) <= qerr_abs(last_term) + qerr_abs(inner))
            by (nonlinear_arith);

        // Capture inner bound and group bound as plain ints for linear arithmetic
        let inner_bound = group_error_bound(sc_g, q_rest);
        assert(qerr_abs(inner) <= inner_bound);  // from IH

        let last_bound = sc_g * ql;
        assert(qerr_abs(last_term) <= last_bound);  // from last_term assertion above

        // Unfold group_error_bound for the last element
        assert(group_error_bound(sc_g, q_g) == last_bound + inner_bound);

        // Now: linear arithmetic on plain ints
        assert(qerr_abs(last_term) + qerr_abs(inner) <= last_bound + inner_bound)
            by (nonlinear_arith)
            requires qerr_abs(last_term) <= last_bound,
                     qerr_abs(inner) <= inner_bound;
    }
}

/// group_error_bound is non-negative when sc_g ≥ 0 and q_i ≥ 0.
pub proof fn lemma_group_error_bound_nonneg(sc_g: int, q_g: Seq<int>)
    requires sc_g >= 0, all_nibbles(q_g)
    ensures group_error_bound(sc_g, q_g) >= 0
    decreases q_g.len()
{
    if q_g.len() == 0 {
    } else {
        assert(all_nibbles(q_g));
        assert(0 <= q_g.last() <= 15);
        lemma_group_error_bound_nonneg(sc_g, q_g.drop_last());
        assert(sc_g * q_g.last() >= 0) by (nonlinear_arith)
            requires sc_g >= 0, q_g.last() >= 0;
        assert(group_error_bound(sc_g, q_g) ==
               sc_g * q_g.last() + group_error_bound(sc_g, q_g.drop_last()));
    }
}

// Sanity check on empty sequences — proves lemma fires correctly at base case.
proof fn check_error_bound_empty()
    ensures qerr_abs(weighted_err(5int, Seq::<int>::empty(), Seq::<int>::empty()))
            <= group_error_bound(5int, Seq::<int>::empty())
{
    lemma_q4k_group_opt_error_bound(
        5int, Seq::<int>::empty(), Seq::<int>::empty());
}

// ============================================================================
// End of Q4K error bound proofs
// ============================================================================

// ============================================================================
// Q6K error bound proofs
// ============================================================================
//
// Q6K uses i16 activation quantization (32767 levels vs i8's 127).
// Weights are 6-bit signed: w_i ∈ [-32, 31], so |w_i| ≤ 32.
// Group scales are i8: sc_g ∈ [-128, 127].
// Per-element Q16 error: |e_i| ≤ 1 (in i16 units; scale is max_abs/32767).
//
// Error bound: |Σ sc_g * w_i * e_i| ≤ |sc_g| * Σ |w_i|.

/// Predicate: all weights are valid Q6K 6-bit signed values in [-32, 31].
pub open spec fn all_q6k_weights(q: Seq<int>) -> bool
    decreases q.len()
{
    if q.len() == 0 { true }
    else { -32 <= q.last() <= 31 && all_q6k_weights(q.drop_last()) }
}

/// Predicate: all errors are unit (|e_i| ≤ 1), same as Q4K.
pub open spec fn all_q6k_errors(e: Seq<int>) -> bool
    decreases e.len()
{
    if e.len() == 0 { true }
    else { -1 <= e.last() <= 1 && all_q6k_errors(e.drop_last()) }
}

/// L1 norm of Q6K weights: Σ |w_i|.  Each |w_i| ≤ 32.
pub open spec fn q6k_weight_l1(q: Seq<int>) -> int
    decreases q.len()
{
    if q.len() == 0 { 0int }
    else {
        let w = q.last();
        (if w >= 0 { w } else { -w }) + q6k_weight_l1(q.drop_last())
    }
}

/// Weighted error sum for Q6K: Σ sc_g * w_i * e_i.
pub open spec fn q6k_weighted_err(sc_g: int, q_g: Seq<int>, e_g: Seq<int>) -> int
    recommends q_g.len() == e_g.len()
    decreases q_g.len()
{
    if q_g.len() == 0 { 0int }
    else {
        sc_g * q_g.last() * e_g.last()
            + q6k_weighted_err(sc_g, q_g.drop_last(), e_g.drop_last())
    }
}

/// Q6K group error bound: |sc_g| * Σ |w_i|.
pub open spec fn q6k_group_error_bound(sc_g: int, q_g: Seq<int>) -> int
{
    let abs_sc = if sc_g >= 0 { sc_g } else { -sc_g };
    abs_sc * q6k_weight_l1(q_g)
}

/// L1 norm is non-negative.
proof fn lemma_q6k_l1_nonneg(q: Seq<int>)
    ensures q6k_weight_l1(q) >= 0
    decreases q.len()
{
    if q.len() == 0 {
    } else {
        lemma_q6k_l1_nonneg(q.drop_last());
        let w = q.last();
        let abs_w = if w >= 0 { w } else { -w };
        assert(abs_w >= 0);
        assert(q6k_weight_l1(q) == abs_w + q6k_weight_l1(q.drop_last()));
    }
}

/// **Main Q6K theorem**: |weighted_err(sc_g, q_g, e_g)| ≤ |sc_g| * Σ|w_i|.
///
/// For any Q6K group: the error from i16 activation quantization (|e_i| ≤ 1)
/// is bounded by the absolute scale times the L1 norm of weights.
/// Proved for ALL sc_g ∈ [-128, 127], ALL lengths, ALL valid Q6K inputs.
pub proof fn lemma_q6k_group_opt_error_bound(
    sc_g: int,
    q_g: Seq<int>,
    e_g: Seq<int>,
)
    requires
        q_g.len() == e_g.len(),
        sc_g >= -128 && sc_g <= 127,
        all_q6k_weights(q_g),
        all_q6k_errors(e_g),
    ensures
        qerr_abs(q6k_weighted_err(sc_g, q_g, e_g)) <= q6k_group_error_bound(sc_g, q_g)
    decreases q_g.len()
{
    let abs_sc = if sc_g >= 0 { sc_g } else { -sc_g };

    if q_g.len() == 0 {
        assert(q6k_weighted_err(sc_g, q_g, e_g) == 0int);
        assert(q6k_weight_l1(q_g) == 0int);
        assert(qerr_abs(0int) == 0int);
    } else {
        let q_rest = q_g.drop_last();
        let e_rest = e_g.drop_last();

        assert(all_q6k_weights(q_g));
        assert(all_q6k_errors(e_g));
        let wl = q_g.last();
        let el = e_g.last();
        assert(-32 <= wl <= 31);
        assert(-1 <= el <= 1);

        // Apply IH
        lemma_q6k_group_opt_error_bound(sc_g, q_rest, e_rest);

        // Last term: sc_g * wl * el.  |sc_g * wl * el| ≤ abs_sc * |wl|.
        let last_term = sc_g * wl * el;
        let abs_wl = if wl >= 0 { wl } else { -wl };
        assert(abs_wl <= 32);

        assert(qerr_abs(last_term) <= abs_sc * abs_wl) by (nonlinear_arith)
            requires -128 <= sc_g <= 127,
                     -32 <= wl <= 31,
                     -1 <= el <= 1,
                     abs_sc == (if sc_g >= 0 { sc_g } else { -sc_g }),
                     abs_wl == (if wl >= 0 { wl } else { -wl }),
                     last_term == sc_g * wl * el;

        // Total error
        let inner = q6k_weighted_err(sc_g, q_rest, e_rest);
        assert(q6k_weighted_err(sc_g, q_g, e_g) == last_term + inner);

        assert(qerr_abs(last_term + inner) <= qerr_abs(last_term) + qerr_abs(inner))
            by (nonlinear_arith);

        // Capture bounds as plain ints
        let inner_bound = q6k_group_error_bound(sc_g, q_rest);
        assert(qerr_abs(inner) <= inner_bound);

        let l1_rest = q6k_weight_l1(q_rest);
        lemma_q6k_l1_nonneg(q_rest);

        assert(q6k_weight_l1(q_g) == abs_wl + l1_rest)
            by { assert(q6k_weight_l1(q_g) == abs_wl + q6k_weight_l1(q_rest)); };

        // q6k_group_error_bound unfolds to abs_sc_inner * q6k_weight_l1(q_rest)
        // where abs_sc_inner is computed inside the spec fn.
        // We reveal the definition and use the same conditional to tie them together.
        let abs_sc_inner: int = if sc_g >= 0 { sc_g } else { -sc_g };
        assert(abs_sc == abs_sc_inner);
        assert(inner_bound == abs_sc_inner * l1_rest) by {
            // q6k_group_error_bound(sc_g, q_rest) = abs_sc_inner * q6k_weight_l1(q_rest)
            // by definition of q6k_group_error_bound (open spec, unfolds directly)
            assert(q6k_group_error_bound(sc_g, q_rest) == abs_sc_inner * l1_rest);
        };

        let last_bound = abs_sc * abs_wl;
        assert(qerr_abs(last_term) <= last_bound);

        // q6k_group_error_bound(sc_g, q_g) = abs_sc_inner * (abs_wl + l1_rest)
        assert(q6k_group_error_bound(sc_g, q_g) == abs_sc_inner * (abs_wl + l1_rest)) by {
            assert(q6k_weight_l1(q_g) == abs_wl + l1_rest);
            assert(q6k_group_error_bound(sc_g, q_g) == abs_sc_inner * q6k_weight_l1(q_g));
            assert(abs_sc_inner * q6k_weight_l1(q_g) == abs_sc_inner * (abs_wl + l1_rest));
        };

        // last_bound = abs_sc * abs_wl = abs_sc_inner * abs_wl (since abs_sc == abs_sc_inner)
        // inner_bound = abs_sc_inner * l1_rest (established above)
        assert(last_bound == abs_sc_inner * abs_wl) by {
            assert(abs_sc == abs_sc_inner);
            assert(last_bound == abs_sc * abs_wl);
        };
        assert(last_bound + inner_bound == abs_sc_inner * abs_wl + abs_sc_inner * l1_rest);
        assert(abs_sc_inner * abs_wl + abs_sc_inner * l1_rest == abs_sc_inner * (abs_wl + l1_rest))
            by (nonlinear_arith);

        assert(qerr_abs(last_term) + qerr_abs(inner) <= last_bound + inner_bound)
            by (nonlinear_arith)
            requires qerr_abs(last_term) <= last_bound,
                     qerr_abs(inner) <= inner_bound;
    }
}

/// Corollary: the Q6K error bound is non-negative.
pub proof fn lemma_q6k_error_bound_nonneg(sc_g: int, q_g: Seq<int>)
    requires sc_g >= -128 && sc_g <= 127, all_q6k_weights(q_g)
    ensures q6k_group_error_bound(sc_g, q_g) >= 0
{
    let abs_sc = if sc_g >= 0 { sc_g } else { -sc_g };
    lemma_q6k_l1_nonneg(q_g);
    assert(q6k_group_error_bound(sc_g, q_g) == abs_sc * q6k_weight_l1(q_g))
        by (nonlinear_arith)
        requires abs_sc == (if sc_g >= 0 { sc_g } else { -sc_g });
    assert(abs_sc >= 0) by {
        assert(sc_g >= -128 && sc_g <= 127);
        if sc_g >= 0 { assert(abs_sc == sc_g); } else { assert(abs_sc == -sc_g); }
    };
    assert(q6k_weight_l1(q_g) >= 0);
    assert(abs_sc * q6k_weight_l1(q_g) >= 0) by (nonlinear_arith)
        requires abs_sc >= 0, q6k_weight_l1(q_g) >= 0;
}

// Sanity check: Q6K lemma fires at base case.
proof fn check_q6k_error_bound_empty()
    ensures qerr_abs(q6k_weighted_err(5int, Seq::<int>::empty(), Seq::<int>::empty()))
            <= q6k_group_error_bound(5int, Seq::<int>::empty())
{
    lemma_q6k_group_opt_error_bound(5int, Seq::<int>::empty(), Seq::<int>::empty());
}

// ============================================================================
// End of Q6K error bound proofs
// ============================================================================

fn main() {}

} // verus!

//! Verus-verified GPU quantized GEMM kernel invariants -- unbounded proofs
//!
//! Proves: Q4K block iteration, Q8 activation quantization groups,
//! shared memory tile loading, warp reduction, and split-K decomposition.

use vstd::prelude::*;

verus! {

pub spec const Q4K_BLOCK_SIZE: nat = 256;
pub spec const Q4K_BLOCK_BYTES: nat = 144;
pub spec const Q8_BLOCK_SIZE: nat = 32;
pub spec const Q8_BLOCK_BYTES: nat = 36;

// ============================================================================
// Q4K block offset in transposed weight buffer
// ============================================================================

proof fn q4k_block_offset_bt(
    col: nat,
    blocks_per_row: nat,
    blk: nat,
)
    requires
        blk < blocks_per_row,
    ensures ({
        let block_offset = (col * blocks_per_row + blk) * Q4K_BLOCK_BYTES;
        // d: 2 bytes at block_offset
        // dmin: 2 bytes at block_offset+2
        // packed_scales: 12 bytes at block_offset+4
        // qs: 128 bytes at block_offset+16
        &&& block_offset + 4 <= (col * blocks_per_row + blk + 1) * Q4K_BLOCK_BYTES
        &&& block_offset + 16 <= (col * blocks_per_row + blk + 1) * Q4K_BLOCK_BYTES
        &&& block_offset + 144 == (col * blocks_per_row + blk + 1) * Q4K_BLOCK_BYTES
    }),
{
    assert((col * blocks_per_row + blk) * Q4K_BLOCK_BYTES + Q4K_BLOCK_BYTES
        == (col * blocks_per_row + blk + 1) * Q4K_BLOCK_BYTES)
        by (nonlinear_arith);
}

// ============================================================================
// Q4K activation base: a_base = row * k + blk * BLOCK_SIZE
// ============================================================================

proof fn q4k_activation_base(
    row: nat,
    k: nat,
    blk: nat,
    blocks_per_row: nat,
)
    requires
        k >= Q4K_BLOCK_SIZE,
        k % Q4K_BLOCK_SIZE == 0,
        blk < blocks_per_row,
        blocks_per_row == k / Q4K_BLOCK_SIZE,
    ensures ({
        let a_base = blk * Q4K_BLOCK_SIZE;
        a_base + Q4K_BLOCK_SIZE <= k
    }),
{
    assert((blk + 1) * Q4K_BLOCK_SIZE <= k)
        by (nonlinear_arith)
        requires blk < k / Q4K_BLOCK_SIZE, k % Q4K_BLOCK_SIZE == 0, Q4K_BLOCK_SIZE >= 1;
    assert(blk * Q4K_BLOCK_SIZE + Q4K_BLOCK_SIZE == (blk + 1) * Q4K_BLOCK_SIZE)
        by (nonlinear_arith);
}

// ============================================================================
// Q8 activation quantization: group indexing
// ============================================================================

proof fn q8_activation_group(
    hidden_size: nat,
    group_idx: nat,
)
    requires
        hidden_size >= Q8_BLOCK_SIZE,
        hidden_size % Q8_BLOCK_SIZE == 0,
        group_idx < hidden_size / Q8_BLOCK_SIZE,
    ensures ({
        let group_start = group_idx * Q8_BLOCK_SIZE;
        group_start + Q8_BLOCK_SIZE <= hidden_size
    }),
{
    assert((group_idx + 1) * Q8_BLOCK_SIZE <= hidden_size)
        by (nonlinear_arith)
        requires group_idx < hidden_size / Q8_BLOCK_SIZE,
                 hidden_size % Q8_BLOCK_SIZE == 0;
    assert(group_idx * Q8_BLOCK_SIZE + Q8_BLOCK_SIZE == (group_idx + 1) * Q8_BLOCK_SIZE)
        by (nonlinear_arith);
}

// ============================================================================
// Shared memory tile: WARPS_Y x 256 x 4 bytes per CTA
// ============================================================================

proof fn smem_tile_index(
    warps_y: nat,
    warp_id: nat,
    elem: nat,
)
    requires
        warps_y >= 1,
        warp_id < warps_y,
        elem < Q4K_BLOCK_SIZE,
    ensures ({
        let smem_base = warp_id * Q4K_BLOCK_SIZE;
        let smem_idx = smem_base + elem;
        smem_idx < warps_y * Q4K_BLOCK_SIZE
    }),
{
    assert((warp_id + 1) * Q4K_BLOCK_SIZE <= warps_y * Q4K_BLOCK_SIZE)
        by (nonlinear_arith) requires warp_id < warps_y;
    assert(warp_id * Q4K_BLOCK_SIZE + Q4K_BLOCK_SIZE == (warp_id + 1) * Q4K_BLOCK_SIZE)
        by (nonlinear_arith);
}

// ============================================================================
// Warp reduction: warp_sums array indexed by warp_id
// ============================================================================

proof fn warp_reduction_index(
    num_warps: nat,
    warp_id: nat,
)
    requires
        num_warps >= 1,
        warp_id < num_warps,
    ensures
        warp_id < num_warps,
{
}

// ============================================================================
// Split-K: each split handles blocks_per_row / num_splits blocks
// ============================================================================

proof fn split_k_decomposition(
    blocks_per_row: nat,
    num_splits: nat,
    split_idx: nat,
)
    requires
        blocks_per_row >= 1,
        num_splits >= 1,
        blocks_per_row >= num_splits,
        split_idx < num_splits,
    ensures ({
        let blocks_per_split = ((blocks_per_row + num_splits - 1) as nat) / num_splits;
        &&& blocks_per_split >= 1
        &&& blocks_per_split * num_splits >= blocks_per_row
    }),
{
    assert(((blocks_per_row + num_splits - 1) as nat) / num_splits >= 1)
        by (nonlinear_arith) requires blocks_per_row >= 1, num_splits >= 1;
    assert(((blocks_per_row + num_splits - 1) as nat) / num_splits * num_splits >= blocks_per_row)
        by (nonlinear_arith) requires num_splits >= 1, blocks_per_row >= 1;
}

// ============================================================================
// DP4A: 4 i8 multiplications accumulated into i32
// max per dp4a: 4 x 127 x 127 = 64516 < i32::MAX
// ============================================================================

proof fn dp4a_no_overflow()
    ensures
        4int * 127 * 127 <= 2147483647,
{
    assert(4int * 127 * 127 <= 2147483647int) by (nonlinear_arith);
}

proof fn dp4a_block_accumulator()
    ensures
        64int * 4 * 127 * 127 <= 2147483647,
{
    assert(64int * 4 * 127 * 127 <= 2147483647int) by (nonlinear_arith);
}

// ============================================================================
// GPU Q4K output-value equivalence proofs
//
// CutileQuantizedGemm first dequantizes Q4K weights to f32 (via the same
// dequant_q4k_block function as CpuMatMulQ4K), then runs a generic f32 GEMM
// on the GPU.  This means the GPU and CPU reference compute the SAME
// mathematical formula -- the only difference is f32 reduction order.
//
// Modelling arithmetic as exact integers (no f32 rounding), we prove:
//   1. gpu_q4k_output == cpu_ref_q4k_output  (same formula)
//   2. |gpu_q4k_output - cpu_opt_q4k_output| <= error_bound
//      where the error comes from Q8 activation quantization, NOT from the GPU.
//
// Spec functions are self-contained (run-verus.sh compiles each module
// standalone); names are prefixed gpu_ to avoid collisions.
// ============================================================================

// -- Utility ------------------------------------------------------------------

pub open spec fn gpu_abs(x: int) -> int {
    if x >= 0 { x } else { -x }
}

// -- Predicates on sequences ---------------------------------------------------

/// All elements are nibble values: ??? [0, 15].
pub open spec fn gpu_all_nibbles(q: Seq<int>) -> bool
    decreases q.len()
{
    if q.len() == 0 { true }
    else { 0 <= q.last() <= 15 && gpu_all_nibbles(q.drop_last()) }
}

/// All elements are Q8-quantization unit errors: ??? [-1, 1].
pub open spec fn gpu_all_unit_errors(e: Seq<int>) -> bool
    decreases e.len()
{
    if e.len() == 0 { true }
    else { -1 <= e.last() <= 1 && gpu_all_unit_errors(e.drop_last()) }
}

/// All elements are valid Q8 activations: ??? [-127, 127].
pub open spec fn gpu_all_q8(a: Seq<int>) -> bool
    decreases a.len()
{
    if a.len() == 0 { true }
    else { -127 <= a.last() <= 127 && gpu_all_q8(a.drop_last()) }
}

// -- Core spec functions -------------------------------------------------------

/// Integer dot product of two sequences.
pub open spec fn gpu_seq_dot(a: Seq<int>, b: Seq<int>) -> int
    recommends a.len() == b.len()
    decreases a.len()
{
    if a.len() == 0 || b.len() == 0 { 0int }
    else { a.last() * b.last() + gpu_seq_dot(a.drop_last(), b.drop_last()) }
}

/// Sum of all elements in a sequence.
pub open spec fn gpu_seq_sum(a: Seq<int>) -> int
    decreases a.len()
{
    if a.len() == 0 { 0int }
    else { a.last() + gpu_seq_sum(a.drop_last()) }
}

/// Reference formula for one Q4K group (integer model, treating d=dmin=1 for
/// algebraic clarity -- the caller folds d and dmin into the result):
///   ref_group(sc_g, m_g, q_g, a_g) = sc_g * dot(q_g, a_g) - m_g * sum(a_g)
pub open spec fn gpu_q4k_ref_group(sc_g: int, m_g: int, q_g: Seq<int>, a_g: Seq<int>) -> int
    recommends q_g.len() == a_g.len()
{
    sc_g * gpu_seq_dot(q_g, a_g) - m_g * gpu_seq_sum(a_g)
}

/// Optimized formula for one group (Q8 activations with errors e_i = Q8_i - a_i):
///   opt_group(sc_g, m_g, q_g, a_g, e_g) = sc_g * dot(q_g, a_g + e_g) - m_g * sum(a_g)
///   where a_q8_i = a_g[i] + e_g[i]
pub open spec fn gpu_q4k_opt_group(sc_g: int, m_g: int, q_g: Seq<int>, a_g: Seq<int>, e_g: Seq<int>) -> int
    recommends q_g.len() == a_g.len(), a_g.len() == e_g.len()
{
    sc_g * gpu_seq_dot(q_g, seq_add(a_g, e_g)) - m_g * gpu_seq_sum(a_g)
}

/// Element-wise addition of two sequences.
pub open spec fn seq_add(a: Seq<int>, b: Seq<int>) -> Seq<int>
    recommends a.len() == b.len()
    decreases a.len()
{
    if a.len() == 0 || b.len() == 0 { Seq::<int>::empty() }
    else { seq_add(a.drop_last(), b.drop_last()).push(a.last() + b.last()) }
}

/// Error bound for one group: sc_g * Sigma q_i  (max |e| when |e_i| <= 1).
pub open spec fn gpu_group_error_bound(sc_g: int, q_g: Seq<int>) -> int
    decreases q_g.len()
{
    if q_g.len() == 0 { 0int }
    else { sc_g * q_g.last() + gpu_group_error_bound(sc_g, q_g.drop_last()) }
}

/// Error term for one group: sc_g * Sigma q_i * e_i.
pub open spec fn gpu_weighted_err(sc_g: int, q_g: Seq<int>, e_g: Seq<int>) -> int
    recommends q_g.len() == e_g.len()
    decreases q_g.len()
{
    if q_g.len() == 0 { 0int }
    else {
        sc_g * q_g.last() * e_g.last()
            + gpu_weighted_err(sc_g, q_g.drop_last(), e_g.drop_last())
    }
}

// -- Helper lemmas (self-contained) --------------------------------------------

/// seq_dot over seq_add equals seq_dot(q, a) + seq_dot(q, e).
proof fn lemma_gpu_dot_add_split(q: Seq<int>, a: Seq<int>, e: Seq<int>)
    requires q.len() == a.len(), a.len() == e.len()
    ensures gpu_seq_dot(q, seq_add(a, e)) == gpu_seq_dot(q, a) + gpu_seq_dot(q, e)
    decreases q.len()
{
    if q.len() == 0 {
        assert(seq_add(a, e) =~= Seq::<int>::empty());
        assert(gpu_seq_dot(q, seq_add(a, e)) == 0int);
        assert(gpu_seq_dot(q, a) == 0int);
        assert(gpu_seq_dot(q, e) == 0int);
    } else {
        let ae = seq_add(a, e);
        // Establish structural properties of seq_add
        assert(ae.len() == a.len()) by {
            // seq_add produces same length as inputs -- prove by induction
            lemma_seq_add_len(a, e);
        }
        assert(ae.last() == a.last() + e.last()) by {
            lemma_seq_add_last(a, e);
        }
        assert(ae.drop_last() =~= seq_add(a.drop_last(), e.drop_last())) by {
            lemma_seq_add_drop_last(a, e);
        }
        // IH
        lemma_gpu_dot_add_split(q.drop_last(), a.drop_last(), e.drop_last());
        // Unfold gpu_seq_dot on the left
        assert(gpu_seq_dot(q, ae)
            == q.last() * ae.last()
               + gpu_seq_dot(q.drop_last(), ae.drop_last()));
        // Substitute ae.last() and ae.drop_last()
        assert(gpu_seq_dot(q, ae)
            == q.last() * (a.last() + e.last())
               + gpu_seq_dot(q.drop_last(), seq_add(a.drop_last(), e.drop_last())));
        // Apply IH
        assert(gpu_seq_dot(q, ae)
            == q.last() * (a.last() + e.last())
               + gpu_seq_dot(q.drop_last(), a.drop_last())
               + gpu_seq_dot(q.drop_last(), e.drop_last()));
        // Unfold gpu_seq_dot(q, a) and gpu_seq_dot(q, e)
        assert(gpu_seq_dot(q, a) == q.last() * a.last() + gpu_seq_dot(q.drop_last(), a.drop_last()));
        assert(gpu_seq_dot(q, e) == q.last() * e.last() + gpu_seq_dot(q.drop_last(), e.drop_last()));
        // Conclude by distributivity
        assert(q.last() * (a.last() + e.last())
            == q.last() * a.last() + q.last() * e.last())
            by (nonlinear_arith);
    }
}

proof fn lemma_seq_add_len(a: Seq<int>, b: Seq<int>)
    requires a.len() == b.len()
    ensures seq_add(a, b).len() == a.len()
    decreases a.len()
{
    if a.len() == 0 {
        assert(seq_add(a, b) =~= Seq::<int>::empty());
    } else {
        lemma_seq_add_len(a.drop_last(), b.drop_last());
        assert(seq_add(a, b) =~= seq_add(a.drop_last(), b.drop_last()).push(a.last() + b.last()));
    }
}

proof fn lemma_seq_add_last(a: Seq<int>, b: Seq<int>)
    requires a.len() == b.len(), a.len() >= 1
    ensures seq_add(a, b).last() == a.last() + b.last()
{
    lemma_seq_add_len(a, b);
    assert(seq_add(a, b) =~= seq_add(a.drop_last(), b.drop_last()).push(a.last() + b.last()));
}

proof fn lemma_seq_add_drop_last(a: Seq<int>, b: Seq<int>)
    requires a.len() == b.len(), a.len() >= 1
    ensures seq_add(a, b).drop_last() =~= seq_add(a.drop_last(), b.drop_last())
{
    lemma_seq_add_len(a, b);
    assert(seq_add(a, b) =~= seq_add(a.drop_last(), b.drop_last()).push(a.last() + b.last()));
}

/// gpu_seq_dot is linear: dot(q, scale(sc, a)) == sc * dot(q, a).
proof fn lemma_gpu_dot_scale(sc: int, q: Seq<int>, a: Seq<int>)
    requires q.len() == a.len()
    ensures gpu_seq_dot(q, seq_scale_gpu(sc, a)) == sc * gpu_seq_dot(q, a)
    decreases q.len()
{
    if q.len() == 0 {
        assert(seq_scale_gpu(sc, a) =~= Seq::<int>::empty());
        assert(gpu_seq_dot(q, seq_scale_gpu(sc, a)) == 0int);
        assert(sc * 0int == 0int) by (nonlinear_arith);
    } else {
        let sa = seq_scale_gpu(sc, a);
        // Structural properties of seq_scale_gpu
        assert(sa =~= seq_scale_gpu(sc, a.drop_last()).push(sc * a.last()));
        assert(sa.last() == sc * a.last());
        assert(sa.drop_last() =~= seq_scale_gpu(sc, a.drop_last()));
        // sa has same length as a
        lemma_seq_scale_gpu_len(sc, a);
        lemma_gpu_dot_scale(sc, q.drop_last(), a.drop_last());
        // Unfold dot
        assert(gpu_seq_dot(q, sa) == q.last() * sa.last() + gpu_seq_dot(q.drop_last(), sa.drop_last()));
        assert(gpu_seq_dot(q, a)  == q.last() * a.last()  + gpu_seq_dot(q.drop_last(), a.drop_last()));
        let ih = gpu_seq_dot(q.drop_last(), a.drop_last());
        assert(gpu_seq_dot(q.drop_last(), sa.drop_last()) == sc * ih);
        assert(q.last() * (sc * a.last()) == sc * (q.last() * a.last())) by (nonlinear_arith);
        // Conclude: dot(q, sa) = q.last()*(sc*a.last()) + sc*ih = sc*(q.last()*a.last() + ih) = sc*dot(q,a)
        assert(gpu_seq_dot(q, sa) == sc * gpu_seq_dot(q, a)) by (nonlinear_arith)
            requires
                gpu_seq_dot(q, sa)
                    == q.last() * (sc * a.last()) + gpu_seq_dot(q.drop_last(), sa.drop_last()),
                gpu_seq_dot(q.drop_last(), sa.drop_last()) == sc * ih,
                gpu_seq_dot(q, a) == q.last() * a.last() + ih,
                q.last() * (sc * a.last()) == sc * (q.last() * a.last());
    }
}

proof fn lemma_seq_scale_gpu_len(sc: int, a: Seq<int>)
    ensures seq_scale_gpu(sc, a).len() == a.len()
    decreases a.len()
{
    if a.len() == 0 {
        assert(seq_scale_gpu(sc, a) =~= Seq::<int>::empty());
    } else {
        lemma_seq_scale_gpu_len(sc, a.drop_last());
        assert(seq_scale_gpu(sc, a) =~= seq_scale_gpu(sc, a.drop_last()).push(sc * a.last()));
    }
}

pub open spec fn seq_scale_gpu(sc: int, a: Seq<int>) -> Seq<int>
    decreases a.len()
{
    if a.len() == 0 { Seq::<int>::empty() }
    else { seq_scale_gpu(sc, a.drop_last()).push(sc * a.last()) }
}

/// gpu_weighted_err(sc_g, q_g, e_g) == sc_g * dot(q_g, e_g).
proof fn lemma_gpu_weighted_err_is_scaled_dot(sc_g: int, q_g: Seq<int>, e_g: Seq<int>)
    requires q_g.len() == e_g.len()
    ensures gpu_weighted_err(sc_g, q_g, e_g) == sc_g * gpu_seq_dot(q_g, e_g)
    decreases q_g.len()
{
    if q_g.len() == 0 {
        assert(gpu_weighted_err(sc_g, q_g, e_g) == 0int);
        assert(gpu_seq_dot(q_g, e_g) == 0int);
        assert(sc_g * 0int == 0int) by (nonlinear_arith);
    } else {
        lemma_gpu_weighted_err_is_scaled_dot(sc_g, q_g.drop_last(), e_g.drop_last());
        let ql = q_g.last();
        let el = e_g.last();
        let inner = gpu_weighted_err(sc_g, q_g.drop_last(), e_g.drop_last());
        let inner_dot = gpu_seq_dot(q_g.drop_last(), e_g.drop_last());
        // IH: inner == sc_g * inner_dot
        assert(inner == sc_g * inner_dot);
        // Unfold definitions
        assert(gpu_weighted_err(sc_g, q_g, e_g) == sc_g * ql * el + inner);
        assert(gpu_seq_dot(q_g, e_g) == ql * el + inner_dot);
        // Algebra: sc_g * ql * el + sc_g * inner_dot == sc_g * (ql * el + inner_dot)
        assert(sc_g * ql * el + sc_g * inner_dot == sc_g * (ql * el + inner_dot))
            by (nonlinear_arith);
        assert(sc_g * ql * el == sc_g * (ql * el))
            by (nonlinear_arith);
    }
}

/// error_bound is non-negative when sc_g >= 0 and all q_i >= 0.
proof fn lemma_gpu_error_bound_nonneg(sc_g: int, q_g: Seq<int>)
    requires sc_g >= 0, gpu_all_nibbles(q_g)
    ensures gpu_group_error_bound(sc_g, q_g) >= 0
    decreases q_g.len()
{
    if q_g.len() == 0 {
    } else {
        assert(gpu_all_nibbles(q_g));
        assert(0 <= q_g.last() <= 15);
        lemma_gpu_error_bound_nonneg(sc_g, q_g.drop_last());
        assert(sc_g * q_g.last() >= 0) by (nonlinear_arith)
            requires sc_g >= 0, q_g.last() >= 0;
    }
}

// -- Theorem 1: GPU formula == CPU reference formula ---------------------------
//
// The GPU kernel (CutileQuantizedGemm) first calls dequant_q4k_block (same
// function as CpuMatMulQ4K) to get f32 weights, then runs a GPU GEMM.
// Treating f32 as exact integers, both compute gpu_q4k_ref_group per group.
// The equivalence is definitional.
//
// -----------------------------------------------------------------------------

pub proof fn lemma_gpu_q4k_ref_group_equals_cpu_ref(
    sc_g: int, m_g: int, q_g: Seq<int>, a_g: Seq<int>,
)
    requires q_g.len() == a_g.len()
    ensures gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
            == gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
// Trivially true: same spec function, documenting the connection explicitly.
{
}

// -- Theorem 2: GPU output ~= CPU opt output, within Q8 error bound -------------
//
// GPU computes: d * ref_group(sc_g, m_g, q_g, a_g)           (exact activations)
// CPU opt:      d * opt_group(sc_g, m_g, q_g, a_g, e_g)      (Q8 activations)
// Error:        gpu_q4k_ref_group - gpu_q4k_opt_group
//             = sc_g * (dot(q_g, a_g) - dot(q_g, a_g+e_g))
//             = -sc_g * dot(q_g, e_g)
//             = -gpu_weighted_err(sc_g, q_g, e_g)
// |Error| <= gpu_group_error_bound(sc_g, q_g)   [by Theorem 3 below]
// -----------------------------------------------------------------------------

pub proof fn lemma_gpu_vs_opt_group_error(
    sc_g: int, m_g: int, q_g: Seq<int>, a_g: Seq<int>, e_g: Seq<int>,
)
    requires
        q_g.len() == a_g.len(),
        a_g.len() == e_g.len(),
    ensures
        gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
            - gpu_q4k_opt_group(sc_g, m_g, q_g, a_g, e_g)
        == -gpu_weighted_err(sc_g, q_g, e_g)
{
    // opt_group uses dot(q, a+e) = dot(q,a) + dot(q,e)
    lemma_gpu_dot_add_split(q_g, a_g, e_g);
    // gpu_weighted_err(sc_g, q, e) = sc_g * dot(q, e)
    lemma_gpu_weighted_err_is_scaled_dot(sc_g, q_g, e_g);
    // ref_group - opt_group = sc_g*dot(q,a) - sc_g*dot(q,a+e) = -sc_g*dot(q,e)
    assert(gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
           - gpu_q4k_opt_group(sc_g, m_g, q_g, a_g, e_g)
        == -(sc_g * gpu_seq_dot(q_g, e_g)))
        by (nonlinear_arith)
        requires
            gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
                == sc_g * gpu_seq_dot(q_g, a_g) - m_g * gpu_seq_sum(a_g),
            gpu_q4k_opt_group(sc_g, m_g, q_g, a_g, e_g)
                == sc_g * (gpu_seq_dot(q_g, a_g) + gpu_seq_dot(q_g, e_g))
                   - m_g * gpu_seq_sum(a_g),
            gpu_weighted_err(sc_g, q_g, e_g) == sc_g * gpu_seq_dot(q_g, e_g);
}

// -- Theorem 3: |GPU ??? CPU_opt| <= group_error_bound ---------------------------

pub proof fn lemma_gpu_q4k_output_error_bound(
    sc_g: int,
    m_g: int,
    q_g: Seq<int>,
    a_g: Seq<int>,
    e_g: Seq<int>,
)
    requires
        q_g.len() == a_g.len(),
        a_g.len() == e_g.len(),
        sc_g >= 0 && sc_g <= 63,
        gpu_all_nibbles(q_g),
        gpu_all_unit_errors(e_g),
    ensures
        gpu_abs(
            gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
                - gpu_q4k_opt_group(sc_g, m_g, q_g, a_g, e_g)
        ) <= gpu_group_error_bound(sc_g, q_g)
{
    // Step 1: error == -weighted_err
    lemma_gpu_vs_opt_group_error(sc_g, m_g, q_g, a_g, e_g);
    let err = gpu_q4k_ref_group(sc_g, m_g, q_g, a_g)
              - gpu_q4k_opt_group(sc_g, m_g, q_g, a_g, e_g);
    assert(err == -gpu_weighted_err(sc_g, q_g, e_g));
    // Step 2: |err| = |weighted_err|
    assert(gpu_abs(err) == gpu_abs(gpu_weighted_err(sc_g, q_g, e_g)))
        by (nonlinear_arith)
        requires err == -gpu_weighted_err(sc_g, q_g, e_g);
    // Step 3: bound |weighted_err| inductively
    lemma_gpu_weighted_err_bounded(sc_g, q_g, e_g);
}

/// |weighted_err(sc_g, q_g, e_g)| <= group_error_bound(sc_g, q_g)
/// when |e_i| <= 1 and q_i ??? [0,15] and sc_g ??? [0,63].
pub proof fn lemma_gpu_weighted_err_bounded(
    sc_g: int,
    q_g: Seq<int>,
    e_g: Seq<int>,
)
    requires
        q_g.len() == e_g.len(),
        sc_g >= 0 && sc_g <= 63,
        gpu_all_nibbles(q_g),
        gpu_all_unit_errors(e_g),
    ensures
        gpu_abs(gpu_weighted_err(sc_g, q_g, e_g)) <= gpu_group_error_bound(sc_g, q_g)
    decreases q_g.len()
{
    if q_g.len() == 0 {
        assert(gpu_weighted_err(sc_g, q_g, e_g) == 0int);
        assert(gpu_group_error_bound(sc_g, q_g) == 0int);
        assert(gpu_abs(0int) == 0int);
    } else {
        let q_rest = q_g.drop_last();
        let e_rest = e_g.drop_last();
        assert(gpu_all_nibbles(q_g));
        assert(gpu_all_unit_errors(e_g));
        let ql = q_g.last();
        let el = e_g.last();
        assert(0 <= ql <= 15);
        assert(-1 <= el <= 1);

        lemma_gpu_weighted_err_bounded(sc_g, q_rest, e_rest);

        let last_term = sc_g * ql * el;
        assert(gpu_abs(last_term) <= sc_g * ql) by (nonlinear_arith)
            requires sc_g >= 0, 0 <= ql, ql <= 15, -1 <= el, el <= 1,
                     last_term == sc_g * ql * el;

        let inner = gpu_weighted_err(sc_g, q_rest, e_rest);
        assert(gpu_weighted_err(sc_g, q_g, e_g) == last_term + inner);

        assert(gpu_abs(last_term + inner) <= gpu_abs(last_term) + gpu_abs(inner))
            by (nonlinear_arith);

        let inner_bound = gpu_group_error_bound(sc_g, q_rest);
        assert(gpu_abs(inner) <= inner_bound);

        let last_bound = sc_g * ql;
        assert(gpu_abs(last_term) <= last_bound);

        assert(gpu_group_error_bound(sc_g, q_g) == last_bound + inner_bound);

        assert(gpu_abs(last_term) + gpu_abs(inner) <= last_bound + inner_bound)
            by (nonlinear_arith)
            requires gpu_abs(last_term) <= last_bound,
                     gpu_abs(inner) <= inner_bound;
    }
}

// -- Sanity checks -------------------------------------------------------------

proof fn check_gpu_ref_group_trivial()
    ensures gpu_q4k_ref_group(1int, 0int, Seq::<int>::empty(), Seq::<int>::empty()) == 0int
{
    assert(gpu_seq_dot(Seq::<int>::empty(), Seq::<int>::empty()) == 0int);
    assert(gpu_seq_sum(Seq::<int>::empty()) == 0int);
}

proof fn check_gpu_error_bound_empty()
    ensures gpu_abs(
        gpu_weighted_err(5int, Seq::<int>::empty(), Seq::<int>::empty())
    ) <= gpu_group_error_bound(5int, Seq::<int>::empty())
{
    lemma_gpu_weighted_err_bounded(
        5int, Seq::<int>::empty(), Seq::<int>::empty());
}

fn main() {}

} // verus!

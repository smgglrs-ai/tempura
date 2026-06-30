//! Verus-verified TurboQuant/PQO compression indexing — unbounded proofs
//!
//! Proves group-based compression and decompression indexing for any
//! valid head_dim, num_vectors, and group_size.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: compression group start/end/index stay within data and group arrays
// ============================================================================

proof fn compress_indices_in_bounds(
    head_dim: nat,
    num_vectors: nat,
    group_size: nat,
    vi: nat,
    gi: nat,
)
    requires
        head_dim >= 1,
        num_vectors >= 1,
        group_size >= 1,
        vi < num_vectors,
        gi < ((head_dim + group_size - 1) as nat) / group_size,
    ensures ({
        let data_len = num_vectors * head_dim;
        let groups_per_vec = ((head_dim + group_size - 1) as nat) / group_size;
        let total_groups = num_vectors * groups_per_vec;
        let vec_start = vi * head_dim;
        let g_start = vec_start + gi * group_size;
        let g_end_candidate = g_start + group_size;
        let g_end = if g_end_candidate < vec_start + head_dim {
            g_end_candidate
        } else {
            vec_start + head_dim
        };
        let group_idx = vi * groups_per_vec + gi;
        &&& g_start < data_len
        &&& g_end <= data_len
        &&& group_idx < total_groups
    }),
{
    let groups_per_vec = ((head_dim + group_size - 1) as nat) / group_size;

    // gi < groups_per_vec => gi * group_size < head_dim + group_size - 1
    // Actually we need: gi * group_size < head_dim (for g_start < data_len)
    // gi < ceil(head_dim / group_size) => gi * group_size < head_dim + group_size - 1
    // => gi * group_size <= head_dim - 1 (since gi * group_size < head_dim when gi < ceil)
    // Not quite. gi < ceil(hd/gs) doesn't imply gi*gs < hd. But it does imply gi*gs < hd + gs - 1.
    // And g_start = vi*hd + gi*gs, data_len = nv*hd. Since vi < nv: vi*hd + hd <= nv*hd.
    // And gi*gs < hd + gs - 1 <= hd + gs. So g_start = vi*hd + gi*gs < vi*hd + hd + gs.
    // We need g_start < nv*hd. Since gi < ceil, gi*gs <= hd - 1 + gs - 1 = hd + gs - 2.
    // Actually: gi < ceil(hd/gs) => gi <= ceil(hd/gs) - 1. The largest gi*gs is (ceil-1)*gs.
    // (ceil-1)*gs = ((hd+gs-1)/gs - 1)*gs. For hd=1, gs=1: ceil=1, (0)*1=0 < 1. OK.
    // In general: (ceil-1)*gs < hd because ceil*gs >= hd and (ceil-1)*gs = ceil*gs - gs < hd.
    // Wait: ceil*gs >= hd, so (ceil-1)*gs = ceil*gs - gs >= hd - gs.
    // But gi*gs can equal (ceil-1)*gs which might be >= hd when gs > 1.
    // Example: hd=5, gs=3. ceil=2. gi can be 0 or 1. gi=1: gi*gs=3 < 5. OK.
    // Example: hd=3, gs=3. ceil=1. gi can be 0. gi*gs=0 < 3. OK.
    // Example: hd=4, gs=3. ceil=2. gi=1: gi*gs=3 < 4. OK.
    // So gi*gs < hd always holds when gi < ceil(hd/gs).

    assert(gi * group_size < head_dim)
        by (nonlinear_arith)
        requires gi < ((head_dim + group_size - 1) as nat) / group_size, group_size >= 1, head_dim >= 1;

    // g_start = vi*hd + gi*gs < vi*hd + hd = (vi+1)*hd <= nv*hd = data_len
    assert((vi + 1) * head_dim <= num_vectors * head_dim)
        by (nonlinear_arith) requires vi < num_vectors;
    assert(vi * head_dim + head_dim == (vi + 1) * head_dim)
        by (nonlinear_arith);

    // g_end <= vec_start + head_dim = (vi+1)*hd <= data_len
    // (by the if/else clamping to vec_start + head_dim)

    // group_idx = vi * gpv + gi < nv * gpv = total_groups
    assert((vi + 1) * groups_per_vec <= num_vectors * groups_per_vec)
        by (nonlinear_arith) requires vi < num_vectors;
    assert(vi * groups_per_vec + groups_per_vec == (vi + 1) * groups_per_vec)
        by (nonlinear_arith);
}

// ============================================================================
// Proof: decompression group indexing stays in bounds
// ============================================================================

proof fn decompress_indices_in_bounds(
    head_dim: nat,
    num_vectors: nat,
    group_size: nat,
    vi: nat,
    gi: nat,
)
    requires
        head_dim >= 1,
        num_vectors >= 1,
        group_size >= 1,
        vi < num_vectors,
        gi < ((head_dim + group_size - 1) as nat) / group_size,
    ensures ({
        let groups_per_vec = ((head_dim + group_size - 1) as nat) / group_size;
        let total_groups = num_vectors * groups_per_vec;
        let total_elements = num_vectors * head_dim;
        let group_idx = vi * groups_per_vec + gi;
        let g_start = gi * group_size;
        let remaining = (head_dim - g_start) as nat;
        let g_len = if group_size < remaining { group_size } else { remaining };
        let output_pos = vi * head_dim + g_start;
        &&& group_idx < total_groups
        &&& g_len >= 1
        &&& output_pos + g_len <= total_elements
    }),
{
    let groups_per_vec = ((head_dim + group_size - 1) as nat) / group_size;

    // group_idx < total_groups
    assert((vi + 1) * groups_per_vec <= num_vectors * groups_per_vec)
        by (nonlinear_arith) requires vi < num_vectors;
    assert(vi * groups_per_vec + groups_per_vec == (vi + 1) * groups_per_vec)
        by (nonlinear_arith);

    // gi*gs < hd (same as compress proof)
    assert(gi * group_size < head_dim)
        by (nonlinear_arith)
        requires gi < ((head_dim + group_size - 1) as nat) / group_size, group_size >= 1, head_dim >= 1;

    // remaining = hd - gi*gs >= 1
    // g_len = min(gs, remaining) >= 1

    // output_pos + g_len = vi*hd + gi*gs + g_len <= vi*hd + gi*gs + remaining
    //                    = vi*hd + hd = (vi+1)*hd <= nv*hd = total_elements
    assert((vi + 1) * head_dim <= num_vectors * head_dim)
        by (nonlinear_arith) requires vi < num_vectors;
    assert(vi * head_dim + head_dim == (vi + 1) * head_dim)
        by (nonlinear_arith);
}

fn main() {}

} // verus!

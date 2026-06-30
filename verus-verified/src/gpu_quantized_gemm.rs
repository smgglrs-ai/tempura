//! Verus-verified GPU quantized GEMM kernel invariants — unbounded proofs
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
// Shared memory tile: WARPS_Y × 256 × 4 bytes per CTA
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
// max per dp4a: 4 × 127 × 127 = 64516 < i32::MAX
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

fn main() {}

} // verus!

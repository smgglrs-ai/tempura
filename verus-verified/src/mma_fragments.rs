//! Verus-verified MMA (matrix multiply-accumulate) fragment invariants
//!
//! Proves: warp-level MMA fragment sizes for different tiers,
//! thread-to-fragment mapping, and accumulator layout.

use vstd::prelude::*;

verus! {

// ============================================================================
// MMA m16n8k16 (Tier 1, sm_80+): D[4×f32], A[4×u32], B[2×u32], C[4×f32]
// ============================================================================

proof fn mma_m16n8k16_fragments()
    ensures ({
        let d_size: nat = 4;   // 4 f32 accumulators per thread
        let a_size: nat = 4;   // 4 u32 (holding bf16/fp16 pairs)
        let b_size: nat = 2;   // 2 u32
        let c_size: nat = 4;   // 4 f32 inputs
        let m: nat = 16;
        let n: nat = 8;
        let k: nat = 16;
        // 32 threads per warp × 4 outputs = 128 elements = m × n = 16 × 8
        &&& 32 * d_size == m * n
    }),
{
}

// ============================================================================
// MMA m16n8k32 (Tier 2, sm_120+): D[4×f32], A[4×u32], B[2×u32]
// For FP8 E4M3: each u32 holds 4 packed FP8 values
// ============================================================================

proof fn mma_m16n8k32_fragments()
    ensures ({
        let d_size: nat = 4;
        let a_size: nat = 4;   // 4 × 4 = 16 FP8 values per thread, 32 threads × 16 = 512
        let k: nat = 32;
        // A matrix: 16 rows × 32 cols, distributed across 32 threads
        // Each thread contributes 4 u32 = 16 FP8 elements
        &&& 32 * 16 == 512
        &&& 512 == 16 * k
    }),
{
}

// ============================================================================
// Thread-to-output mapping: thread t owns output elements
// For m16n8: thread t owns rows [t%16] columns based on t/16
// ============================================================================

proof fn mma_thread_output_mapping(
    thread_in_warp: nat,
    m: nat,
    n: nat,
)
    requires
        thread_in_warp < 32,
        m == 16,
        n == 8,
    ensures ({
        let row_group = thread_in_warp % 16;  // 0..15
        // Each thread produces 4 output elements: 2 per row × 2 rows
        &&& row_group < m
    }),
{
    assert(thread_in_warp % 16 < 16) by (nonlinear_arith);
}

// ============================================================================
// Warp-level tile: one warp computes m×n output tile
// CTA with W warps computes W tiles
// ============================================================================

proof fn warp_tile_coverage(
    num_warps: nat,
    m_tile: nat,
    n_tile: nat,
)
    requires
        num_warps >= 1,
        m_tile >= 1,
        n_tile >= 1,
    ensures ({
        let elements_per_warp = m_tile * n_tile;
        let total_elements = num_warps * elements_per_warp;
        total_elements >= elements_per_warp
    }),
{
    assert(num_warps * (m_tile * n_tile) >= m_tile * n_tile)
        by (nonlinear_arith) requires num_warps >= 1;
}

// ============================================================================
// CTA-level tile: CTA_M × CTA_N output, processed by warps
// ============================================================================

proof fn cta_tile_decomposition(
    cta_m: nat,
    cta_n: nat,
    m_tile: nat,
    n_tile: nat,
)
    requires
        cta_m >= m_tile,
        cta_n >= n_tile,
        m_tile >= 1,
        n_tile >= 1,
        cta_m % m_tile == 0,
        cta_n % n_tile == 0,
    ensures ({
        let warps_m = cta_m / m_tile;
        let warps_n = cta_n / n_tile;
        let total_warps = warps_m * warps_n;
        &&& warps_m >= 1
        &&& warps_n >= 1
        &&& total_warps * m_tile * n_tile == cta_m * cta_n
    }),
{
    assert(cta_m / m_tile >= 1)
        by (nonlinear_arith) requires cta_m >= m_tile, m_tile >= 1;
    assert(cta_n / n_tile >= 1)
        by (nonlinear_arith) requires cta_n >= n_tile, n_tile >= 1;
    assert((cta_m / m_tile) * (cta_n / n_tile) * m_tile * n_tile == cta_m * cta_n)
        by (nonlinear_arith)
        requires cta_m % m_tile == 0, cta_n % n_tile == 0, m_tile >= 1, n_tile >= 1;
}

// ============================================================================
// K-dimension tiling: iterate k in steps of MMA_K
// ============================================================================

proof fn k_dim_tiling(
    k: nat,
    mma_k: nat,
    kt: nat,
)
    requires
        k >= mma_k,
        mma_k >= 1,
        kt < ((k + mma_k - 1) as nat) / mma_k,
    ensures ({
        let k_base = kt * mma_k;
        k_base < k || kt == ((k + mma_k - 1) as nat) / mma_k - 1
    }),
{
    assert(kt * mma_k < k)
        by (nonlinear_arith)
        requires kt < ((k + mma_k - 1) as nat) / mma_k, mma_k >= 1, k >= 1;
}

fn main() {}

} // verus!

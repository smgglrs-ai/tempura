//! Verus-verified matrix multiply indexing — unbounded proofs
//!
//! Proves standard GEMM indexing (A[i,p], B[p,j], O[i,j]) and tiled
//! GEMM shared memory indexing for any valid matrix dimensions.

use vstd::prelude::*;

verus! {

// ============================================================================
// Proof: standard matmul A[i*k+p], B[p*n+j], O[i*n+j] all in bounds
// ============================================================================

proof fn matmul_indices_in_bounds(
    m: nat,
    n: nat,
    k: nat,
    i: nat,
    j: nat,
    p: nat,
)
    requires
        m >= 1, n >= 1, k >= 1,
        i < m, j < n, p < k,
    ensures ({
        let a_len = m * k;
        let b_len = k * n;
        let o_len = m * n;
        &&& i * k + p < a_len
        &&& p * n + j < b_len
        &&& i * n + j < o_len
    }),
{
    // i*k+p < m*k: (i+1)*k <= m*k, and i*k + p < (i+1)*k
    assert((i + 1) * k <= m * k) by (nonlinear_arith) requires i < m;
    assert(i * k + k == (i + 1) * k) by (nonlinear_arith);

    assert((p + 1) * n <= k * n) by (nonlinear_arith) requires p < k;
    assert(p * n + n == (p + 1) * n) by (nonlinear_arith);

    assert((i + 1) * n <= m * n) by (nonlinear_arith) requires i < m;
    assert(i * n + n == (i + 1) * n) by (nonlinear_arith);
}

// ============================================================================
// Proof: tiled GEMM — shared memory index ty*TILE+tx < TILE*TILE,
// and global indices row*n+col < m*n when row<m, col<n
// ============================================================================

proof fn gpu_tiled_gemm_smem_index_in_bounds(
    m_size: nat,
    n_size: nat,
    k_size: nat,
    tile: nat,
    block_y: nat,
    block_x: nat,
    ty: nat,
    tx: nat,
    t: nat,
)
    requires
        m_size >= 1, n_size >= 1, k_size >= 1,
        tile >= 1,
        block_y < ((m_size + tile - 1) as nat) / tile,
        block_x < ((n_size + tile - 1) as nat) / tile,
        ty < tile, tx < tile,
        t < ((k_size + tile - 1) as nat) / tile,
    ensures ({
        let smem_size = tile * tile;
        let smem_idx = ty * tile + tx;
        let row = block_y * tile + ty;
        let col = block_x * tile + tx;
        let a_col = t * tile + tx;
        &&& smem_idx < smem_size
        &&& (row < m_size && col < n_size ==> row * n_size + col < m_size * n_size)
        &&& (row < m_size && a_col < k_size ==> row * k_size + a_col < m_size * k_size)
    }),
{
    // smem: ty*tile+tx < tile*tile since ty < tile, tx < tile
    assert((ty + 1) * tile <= tile * tile) by (nonlinear_arith) requires ty < tile;
    assert(ty * tile + tile == (ty + 1) * tile) by (nonlinear_arith);

    let row = block_y * tile + ty;
    let col = block_x * tile + tx;
    let a_col = t * tile + tx;

    // row*n_size+col < m_size*n_size when row < m_size, col < n_size
    if row < m_size && col < n_size {
        assert(row * n_size + col < m_size * n_size)
            by (nonlinear_arith) requires row < m_size, col < n_size, n_size >= 1;
    }

    // row*k_size+a_col < m_size*k_size when row < m_size, a_col < k_size
    if row < m_size && a_col < k_size {
        assert(row * k_size + a_col < m_size * k_size)
            by (nonlinear_arith) requires row < m_size, a_col < k_size, k_size >= 1;
    }
}

fn main() {}

} // verus!

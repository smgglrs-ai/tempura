use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const TILE: usize = 16;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Tiled GEMM: C = A × B
    /// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major
    #[kernel]
    pub fn sgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let row = thread::blockIdx_y() as usize * TILE + ty;
        let col = thread::blockIdx_x() as usize * TILE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE);
        let mut sum = 0.0f32;

        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE;
            let smem_idx = ty * TILE + tx;

            unsafe {
                let a_col = tile_start + tx;
                if row < m_size && a_col < k_size {
                    TILE_A[smem_idx] = a[row * k_size + a_col];
                } else {
                    TILE_A[smem_idx] = 0.0;
                }

                let b_row = tile_start + ty;
                if b_row < k_size && col < n_size {
                    TILE_B[smem_idx] = b[b_row * n_size + col];
                } else {
                    TILE_B[smem_idx] = 0.0;
                }
            }

            thread::sync_threads();

            unsafe {
                let mut i = 0usize;
                while i < TILE {
                    sum += TILE_A[ty * TILE + i] * TILE_B[i * TILE + tx];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = sum;
                }
            }
        }
    }

    /// Tiled GEMM with B transposed: C = A × Bᵀ
    /// A: [m, k] row-major, B: [n, k] row-major (transposed), C: [m, n] row-major
    #[kernel]
    pub fn sgemm_tiled_bt(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let row = thread::blockIdx_y() as usize * TILE + ty;
        let col = thread::blockIdx_x() as usize * TILE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        let num_tiles = k_size.div_ceil(TILE);
        let mut sum = 0.0f32;

        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE;
            let smem_idx = ty * TILE + tx;

            unsafe {
                let a_col = tile_start + tx;
                if row < m_size && a_col < k_size {
                    TILE_A[smem_idx] = a[row * k_size + a_col];
                } else {
                    TILE_A[smem_idx] = 0.0;
                }

                // B is [n, k] — to get B^T[tile_row, col], read B[col, tile_row]
                let b_k_idx = tile_start + ty;
                if col < n_size && b_k_idx < k_size {
                    TILE_B[smem_idx] = b[col * k_size + b_k_idx];
                } else {
                    TILE_B[smem_idx] = 0.0;
                }
            }

            thread::sync_threads();

            unsafe {
                let mut i = 0usize;
                while i < TILE {
                    sum += TILE_A[ty * TILE + i] * TILE_B[i * TILE + tx];
                    i += 1;
                }
            }

            thread::sync_threads();
            tile += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_size) } {
            if row < m_size {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = sum;
                }
            }
        }
    }
}

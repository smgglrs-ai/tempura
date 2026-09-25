#[cfg(feature = "openmp")]
pub mod openmp {
    use tempura_kernel_api::{Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES};

    use crate::ggml::blocks::{BlockQ4K, BlockQ6K, BlockQ8K};

    const Q8K_BLOCK_BYTES: usize = std::mem::size_of::<BlockQ8K>();

    extern "C" {
        fn tempura_omp_gemv_q4k(
            weights: *const u8,
            q8k: *const BlockQ8K,
            output: *mut f32,
            n: usize,
            blocks_per_row: usize,
            block_bytes: usize,
            dot_fn: unsafe extern "C" fn(*const u8, *const BlockQ8K, usize) -> f32,
        );

        fn tempura_omp_gemv_q6k(
            weights: *const u8,
            q8k: *const BlockQ8K,
            output: *mut f32,
            n: usize,
            blocks_per_row: usize,
            block_bytes: usize,
            dot_fn: unsafe extern "C" fn(*const u8, *const BlockQ8K, usize) -> f32,
        );

        fn tempura_omp_gemm_tiled_q4k(
            weights: *const u8,
            q8k_rows: *const u8,
            output: *mut f32,
            m: usize,
            n: usize,
            bpr: usize,
            q4k_block_bytes: usize,
            q8k_block_bytes: usize,
            dot_fn: unsafe extern "C" fn(*const u8, *const u8) -> f32,
        );

        fn tempura_omp_gemm_tiled_q6k(
            weights: *const u8,
            q8k_rows: *const u8,
            output: *mut f32,
            m: usize,
            n: usize,
            bpr: usize,
            q6k_block_bytes: usize,
            q8k_block_bytes: usize,
            dot_fn: unsafe extern "C" fn(*const u8, *const u8) -> f32,
        );
    }

    unsafe extern "C" fn q4k_dot_trampoline(
        weight_row: *const u8,
        q8k: *const BlockQ8K,
        blocks_per_row: usize,
    ) -> f32 {
        let k = blocks_per_row * 256;
        let x: &[BlockQ4K] =
            std::slice::from_raw_parts(weight_row as *const BlockQ4K, blocks_per_row);
        let y: &[BlockQ8K] = std::slice::from_raw_parts(q8k, blocks_per_row);
        crate::ggml::vec_dot_avx2::vec_dot_q4_k_q8_k_dispatch(k, x, y)
    }

    unsafe extern "C" fn q6k_dot_trampoline(
        weight_row: *const u8,
        q8k: *const BlockQ8K,
        blocks_per_row: usize,
    ) -> f32 {
        let k = blocks_per_row * 256;
        let x: &[BlockQ6K] =
            std::slice::from_raw_parts(weight_row as *const BlockQ6K, blocks_per_row);
        let y: &[BlockQ8K] = std::slice::from_raw_parts(q8k, blocks_per_row);
        crate::ggml::vec_dot_avx2::vec_dot_q6_k_q8_k_dispatch(k, x, y)
    }

    pub fn gemv_q4k(
        weights: &[u8],
        q8k: &[BlockQ8K],
        output: &mut [f32],
        n: usize,
        blocks_per_row: usize,
    ) {
        unsafe {
            tempura_omp_gemv_q4k(
                weights.as_ptr(),
                q8k.as_ptr(),
                output.as_mut_ptr(),
                n,
                blocks_per_row,
                Q4K_BLOCK_BYTES,
                q4k_dot_trampoline,
            );
        }
    }

    pub fn gemv_q6k(
        weights: &[u8],
        q8k: &[BlockQ8K],
        output: &mut [f32],
        n: usize,
        blocks_per_row: usize,
    ) {
        unsafe {
            tempura_omp_gemv_q6k(
                weights.as_ptr(),
                q8k.as_ptr(),
                output.as_mut_ptr(),
                n,
                blocks_per_row,
                Q6K_BLOCK_BYTES,
                q6k_dot_trampoline,
            );
        }
    }

    // Per-block trampolines for the K-tiled GEMM (one Q4K block × one Q8K block)
    unsafe extern "C" fn q4k_q8k_block_dot(w_blk: *const u8, q8k_blk: *const u8) -> f32 {
        let x = std::slice::from_raw_parts(w_blk as *const BlockQ4K, 1);
        let y = std::slice::from_raw_parts(q8k_blk as *const BlockQ8K, 1);
        crate::ggml::vec_dot_avx2::vec_dot_q4_k_q8_k_dispatch(256, x, y)
    }

    unsafe extern "C" fn q6k_q8k_block_dot(w_blk: *const u8, q8k_blk: *const u8) -> f32 {
        let x = std::slice::from_raw_parts(w_blk as *const BlockQ6K, 1);
        let y = std::slice::from_raw_parts(q8k_blk as *const BlockQ8K, 1);
        crate::ggml::vec_dot_avx2::vec_dot_q6_k_q8_k_dispatch(256, x, y)
    }

    /// K-tiled GEMM for Q4K × Q8K (prefill, m>1).
    /// `q8k_all`: flat byte buffer of m×bpr `BlockQ8K` blocks, pre-quantized by caller.
    pub fn gemm_tiled_q4k(
        weights: &[u8],
        q8k_all: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        bpr: usize,
    ) {
        unsafe {
            tempura_omp_gemm_tiled_q4k(
                weights.as_ptr(),
                q8k_all.as_ptr(),
                output.as_mut_ptr(),
                m,
                n,
                bpr,
                Q4K_BLOCK_BYTES,
                Q8K_BLOCK_BYTES,
                q4k_q8k_block_dot,
            );
        }
    }

    /// K-tiled GEMM for Q6K × Q8K (prefill, m>1).
    pub fn gemm_tiled_q6k(
        weights: &[u8],
        q8k_all: &[u8],
        output: &mut [f32],
        m: usize,
        n: usize,
        bpr: usize,
    ) {
        unsafe {
            tempura_omp_gemm_tiled_q6k(
                weights.as_ptr(),
                q8k_all.as_ptr(),
                output.as_mut_ptr(),
                m,
                n,
                bpr,
                Q6K_BLOCK_BYTES,
                Q8K_BLOCK_BYTES,
                q6k_q8k_block_dot,
            );
        }
    }
}

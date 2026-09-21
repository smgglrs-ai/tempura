#[cfg(feature = "openmp")]
pub mod openmp {
    use tempura_kernel_api::{Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES};

    use crate::ggml::blocks::{BlockQ4K, BlockQ6K, BlockQ8K};

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
    }

    unsafe extern "C" fn q4k_dot_trampoline(
        weight_row: *const u8,
        q8k: *const BlockQ8K,
        blocks_per_row: usize,
    ) -> f32 {
        let k = blocks_per_row * 256;
        let x: &[BlockQ4K] = std::slice::from_raw_parts(
            weight_row as *const BlockQ4K,
            blocks_per_row,
        );
        let y: &[BlockQ8K] = std::slice::from_raw_parts(q8k, blocks_per_row);
        crate::ggml::vec_dot_avx2::vec_dot_q4_k_q8_k_dispatch(k, x, y)
    }

    unsafe extern "C" fn q6k_dot_trampoline(
        weight_row: *const u8,
        q8k: *const BlockQ8K,
        blocks_per_row: usize,
    ) -> f32 {
        let k = blocks_per_row * 256;
        let x: &[BlockQ6K] = std::slice::from_raw_parts(
            weight_row as *const BlockQ6K,
            blocks_per_row,
        );
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
}

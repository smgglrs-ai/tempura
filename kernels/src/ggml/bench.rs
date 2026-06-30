// Run: cargo test --package tempura-kernels --release -- ggml_bench --nocapture --ignored

#[cfg(test)]
mod bench_tests {
    use super::super::blocks::*;
    use super::super::helpers::f32_to_f16;
    use super::super::quantize::quantize_row_q8_k;
    use super::super::mul_mat::{mul_mat_q, mul_mat_q_scalar};
    use super::super::GgmlType;
    use std::time::Instant;

    fn make_q4k_weights(n: usize, k: usize) -> Vec<u8> {
        let blocks_per_row = k / 256;
        let mut weights = vec![0u8; n * blocks_per_row * BLOCK_Q4K_SIZE];
        for i in 0..weights.len() {
            weights[i] = (i as u8).wrapping_mul(7).wrapping_add(13);
        }
        // fix d/dmin fields to valid f16
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let base = (row * blocks_per_row + blk) * BLOCK_Q4K_SIZE;
                let d = f32_to_f16(0.01).to_le_bytes();
                weights[base] = d[0];
                weights[base + 1] = d[1];
                let dmin = f32_to_f16(0.001).to_le_bytes();
                weights[base + 2] = dmin[0];
                weights[base + 3] = dmin[1];
            }
        }
        weights
    }

    fn make_activations(m: usize, k: usize) -> Vec<f32> {
        (0..m * k).map(|i| ((i as f32 * 0.01).sin() * 0.1)).collect()
    }

    #[test]
    #[ignore]
    fn ggml_bench_q4k_mul_mat() {
        let k = 4096;
        let n = 4096;
        let m = 1;
        let weights = make_q4k_weights(n, k);
        let activations = make_activations(m, k);

        // warmup
        let mut out = vec![0.0f32; m * n];
        mul_mat_q(GgmlType::Q4K, &weights, &activations, &mut out, m, n, k);

        let iters = 5;

        // benchmark mul_mat_q (AVX2 dispatch)
        let start = Instant::now();
        for _ in 0..iters {
            mul_mat_q(GgmlType::Q4K, &weights, &activations, &mut out, m, n, k);
        }
        let avx2_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // benchmark mul_mat_q_scalar
        let start = Instant::now();
        for _ in 0..iters {
            mul_mat_q_scalar(GgmlType::Q4K, &weights, &activations, &mut out, m, n, k);
        }
        let scalar_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = scalar_ms / avx2_ms;
        eprintln!("Q4K mul_mat {m}x{n}x{k}:");
        eprintln!("  AVX2:   {avx2_ms:.2} ms");
        eprintln!("  Scalar: {scalar_ms:.2} ms");
        eprintln!("  Speedup: {speedup:.2}x");

        // also benchmark CpuMatMulQ4KOpt for comparison
        use crate::quantized::{CpuMatMulQ4KOpt, PreQuantizedActivations};
        use tempura_kernel_api::MatMulQ;
        let q4k_opt = CpuMatMulQ4KOpt;
        let start = Instant::now();
        for _ in 0..iters {
            q4k_opt.forward_b_transposed_q4k(&activations, &weights, &mut out, m, n, k).unwrap();
        }
        let old_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let old_speedup = old_ms / avx2_ms;
        eprintln!("  CpuMatMulQ4KOpt: {old_ms:.2} ms");
        eprintln!("  ggml dispatch vs old: {old_speedup:.2}x");

        // Pure VNNI (skip dispatch overhead)
        if is_x86_feature_detected!("avxvnni") {
            let x = unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ4K, n * (k / 256)) };
            let mut q8k_blocks = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; k / 256];
            super::super::quantize::quantize_row_q8_k(&activations, &mut q8k_blocks);
            let start = Instant::now();
            for _ in 0..iters {
                let mut s = 0.0f32;
                for row in 0..n {
                    let rx = &x[row * (k / 256)..(row + 1) * (k / 256)];
                    s += unsafe { super::super::vec_dot_avx2::vec_dot_q4_k_q8_k_vnni(k, rx, &q8k_blocks) };
                }
                out[0] = s;
            }
            let vnni_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            eprintln!("  Pure VNNI vec_dot: {vnni_ms:.2} ms ({:.2}x vs old)", old_ms / vnni_ms);

            // Pure AVX2 dequant approach
            let start = Instant::now();
            for _ in 0..iters {
                let mut s = 0.0f32;
                for row in 0..n {
                    let rx = &x[row * (k / 256)..(row + 1) * (k / 256)];
                    s += unsafe { super::super::vec_dot_avx2::vec_dot_q4_k_q8_k_avx2(k, rx, &q8k_blocks) };
                }
                out[0] = s;
            }
            let avx2_pure_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            eprintln!("  Pure AVX2 vec_dot: {avx2_pure_ms:.2} ms ({:.2}x vs old)", old_ms / avx2_pure_ms);
        }
    }

    #[test]
    #[ignore]
    fn ggml_bench_q6k_mul_mat() {
        let k = 4096;
        let n = 4096;
        let m = 1;
        let blocks_per_row = k / 256;
        let mut weights = vec![0u8; n * blocks_per_row * BLOCK_Q6K_SIZE];
        for i in 0..weights.len() {
            weights[i] = (i as u8).wrapping_mul(11).wrapping_add(7);
        }
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let base = (row * blocks_per_row + blk) * BLOCK_Q6K_SIZE;
                let d = f32_to_f16(0.01).to_le_bytes();
                weights[base + 208] = d[0];
                weights[base + 209] = d[1];
            }
        }

        let activations = make_activations(m, k);
        let mut out = vec![0.0f32; m * n];

        mul_mat_q(GgmlType::Q6K, &weights, &activations, &mut out, m, n, k);

        let iters = 5;

        let start = Instant::now();
        for _ in 0..iters {
            mul_mat_q(GgmlType::Q6K, &weights, &activations, &mut out, m, n, k);
        }
        let avx2_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            mul_mat_q_scalar(GgmlType::Q6K, &weights, &activations, &mut out, m, n, k);
        }
        let scalar_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!("Q6K mul_mat {m}x{n}x{k}:");
        eprintln!("  AVX2:   {avx2_ms:.2} ms");
        eprintln!("  Scalar: {scalar_ms:.2} ms");
        eprintln!("  Speedup: {:.2}x", scalar_ms / avx2_ms);
    }
}

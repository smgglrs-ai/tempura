#![cfg(feature = "cutile")]

use tempura_kernels::cutile_matmul::CutileMatMul;
use tempura_kernels::matmul::CpuMatMul;
use tempura_kernels::MatMul;

#[test]
fn test_cutile_identity() {
    let cutile_mm = CutileMatMul::new().expect("Failed to initialize CutileMatMul");
    let cpu_mm = CpuMatMul;

    let n = 16;
    let mut identity = vec![0.0f32; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }

    let mut b = vec![0.0f32; n * n];
    for (i, val) in b.iter_mut().enumerate() {
        *val = (i as f32 + 1.0) * 0.1;
    }

    let mut cpu_out = vec![0.0f32; n * n];
    let mut gpu_out = vec![0.0f32; n * n];

    cpu_mm
        .forward(&identity, &b, &mut cpu_out, n, n, n)
        .unwrap();
    cutile_mm
        .forward(&identity, &b, &mut gpu_out, n, n, n)
        .unwrap();

    for i in 0..(n * n) {
        let diff = (gpu_out[i] - cpu_out[i]).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at index {i}: gpu={}, cpu={}, diff={diff}",
            gpu_out[i],
            cpu_out[i]
        );
    }
}

#[test]
fn test_cutile_against_cpu_reference() {
    let cutile_mm = CutileMatMul::new().expect("Failed to initialize CutileMatMul");
    let cpu_mm = CpuMatMul;

    // Test across square and non-square shapes, including non-divisible K
    let test_shapes = vec![
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 64),
        (37, 53, 64), // non-divisible M and N with BK=8 divisible K
        (2, 2, 2),    // tiny shape with K < BK (tests K-padding)
        (31, 47, 35), // non-divisible M, N, and K (tests K-padding)
    ];

    for (m, n, k) in test_shapes {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.125)
            .collect();

        let mut cpu_out = vec![0.0f32; m * n];
        let mut gpu_out = vec![0.0f32; m * n];

        cpu_mm.forward(&a, &b, &mut cpu_out, m, n, k).unwrap();
        cutile_mm.forward(&a, &b, &mut gpu_out, m, n, k).unwrap();

        for i in 0..(m * n) {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff < 1e-3,
                "Shape ({m}, {n}, {k}) mismatch at {i}: gpu={}, cpu={}, diff={diff}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }
}

#[test]
fn test_cutile_b_transposed() {
    let cutile_mm = CutileMatMul::new().expect("Failed to initialize CutileMatMul");
    let cpu_mm = CpuMatMul;

    let shapes = vec![
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (32, 64, 48),
        (2, 2, 2),    // tiny shape with K < BK
        (29, 43, 37), // non-divisible M, N, and K
    ];

    for (m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        // B has shape [N, K] for transposed forward
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();

        let mut cpu_out = vec![0.0f32; m * n];
        let mut gpu_out = vec![0.0f32; m * n];

        cpu_mm
            .forward_b_transposed(&a, &b, &mut cpu_out, m, n, k)
            .unwrap();
        cutile_mm
            .forward_b_transposed(&a, &b, &mut gpu_out, m, n, k)
            .unwrap();

        for i in 0..(m * n) {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff < 1e-3,
                "Transposed shape ({m}, {n}, {k}) mismatch at {i}: gpu={}, cpu={}, diff={diff}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }
}

/// Exercises the dedicated `cutile_gemv` split-K kernel through `forward_b_transposed`
/// with m=1. This is the decode hot path that the new kernel replaces.
///
/// The kernel uses element-wise multiply + reduce_sum over [N_BLOCK=64, BK=64]
/// tiles instead of MMA with BM=1 (which forces scalar FMA fallback on hardware).
#[test]
fn test_cutile_gemv_m1() {
    let cutile_mm = CutileMatMul::new().expect("Failed to initialize CutileMatMul");
    let cpu_mm = CpuMatMul;

    // m=1 decode-realistic shapes: LLaMA-7B style dimensions.
    // N_BLOCK=64 and BK=64 must divide into n and k_pad respectively.
    let shapes = vec![
        (1, 64, 64),     // minimal: exactly one N_BLOCK, one BK chunk
        (1, 128, 64),    // two N_BLOCK tiles, one BK chunk
        (1, 256, 128),   // larger N, two BK chunks
        (1, 2560, 4096), // LLaMA-7B q_proj: n=2560, k=4096
        (1, 64, 37),     // non-power-of-2 K (tests k_pad rounding)
        (1, 192, 97),    // non-divisible N and K
    ];

    for (m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();

        let mut cpu_out = vec![0.0f32; m * n];
        let mut gpu_out = vec![0.0f32; m * n];

        cpu_mm
            .forward_b_transposed(&a, &b, &mut cpu_out, m, n, k)
            .unwrap();
        cutile_mm
            .forward_b_transposed(&a, &b, &mut gpu_out, m, n, k)
            .unwrap();

        for i in 0..(m * n) {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff < 1e-2,
                "GEMV shape ({m}, {n}, {k}) mismatch at {i}: gpu={}, cpu={}, diff={diff}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }
}

/// Exercises the coalesced GEMV kernel (`cutile_gemv_coalesced`) through the
/// `forward_gemv_coalesced_d2d` / `forward_gemv_coalesced_d2d_into` host API.
///
/// The coalesced kernel launches one program per output element.  Each program
/// loads BK=128 contiguous floats from one row of the weight matrix, giving
/// fully coalesced HBM access instead of the N_BLOCK-row strided pattern used
/// by the previous `cutile_gemv`.
#[test]
fn test_cutile_gemv_coalesced_m1() {
    use std::time::Instant;

    let cutile_mm = CutileMatMul::new().expect("Failed to initialize CutileMatMul");
    let cpu_mm = CpuMatMul;

    // Shapes where k_pad is divisible by BK_GEMV_COALESCED=128.
    let shapes = vec![
        (1, 128, 128),   // minimal: one program per output, one BK iteration
        (1, 256, 128),   // two outputs
        (1, 512, 256),   // wider N and K
        (1, 2560, 4096), // LLaMA-7B q_proj: n=2560, k=4096
        (1, 4096, 4096), // square decode step (hidden_dim x hidden_dim)
    ];

    for (m, n, k) in &shapes {
        let (m, n, k) = (*m, *n, *k);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();

        let mut cpu_out = vec![0.0f32; m * n];
        let mut gpu_out = vec![0.0f32; m * n];

        cpu_mm
            .forward_b_transposed(&a, &b, &mut cpu_out, m, n, k)
            .unwrap();
        cutile_mm
            .forward_b_transposed(&a, &b, &mut gpu_out, m, n, k)
            .unwrap();

        for i in 0..(m * n) {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            assert!(
                diff < 1e-2,
                "coalesced GEMV shape ({m}, {n}, {k}) mismatch at {i}: gpu={}, cpu={}, diff={diff}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }

    // Timing measurement for the decode-critical shape (1, 2560, 4096).
    // Routes through forward_b_transposed -> forward_gemv -> cutile_gemv_coalesced.
    // Includes host-to-device transfer; use the d2d methods in engine benchmarks
    // for kernel-only timing without the upload overhead.
    {
        let (m, n, k) = (1usize, 2560usize, 4096usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| (i % 11) as f32 * 0.01).collect();
        let mut out = vec![0.0f32; m * n];

        // Warm-up: one call to trigger JIT compilation.
        cutile_mm
            .forward_b_transposed(&a_data, &b_data, &mut out, m, n, k)
            .expect("warmup");

        // Timed run: 20 iterations.
        const ITERS: u32 = 20;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            cutile_mm
                .forward_b_transposed(&a_data, &b_data, &mut out, m, n, k)
                .expect("timed gemv");
        }
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / ITERS as f64;
        // Weight bytes only (activation is tiny compared to weight matrix).
        let weight_bytes = (n * k * std::mem::size_of::<f32>()) as f64;
        let bandwidth_tbs = weight_bytes / (per_call_us * 1e-6) / 1e12;

        println!(
            "cutile_gemv_coalesced shape ({m}, {n}, {k}): {per_call_us:.1}??s/call \
             (includes H2D transfer), weight-only bandwidth {bandwidth_tbs:.3} TB/s \
             (RTX 5090 peak 1.79 TB/s)"
        );
    }
}

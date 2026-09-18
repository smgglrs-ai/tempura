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

    cpu_mm.forward(&identity, &b, &mut cpu_out, n, n, n).unwrap();
    cutile_mm.forward(&identity, &b, &mut gpu_out, n, n, n).unwrap();

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
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
            .collect();
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

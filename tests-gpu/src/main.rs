use oxide_kernels::rmsnorm::{CpuRMSNorm, CudaRMSNorm, RMSNorm};
use oxide_kernels::rope::{CpuRoPE, CudaRoPE, RoPE};

fn test_rope_cpu_vs_gpu() {
    println!("=== RoPE: CPU vs GPU ===");

    let cpu = CpuRoPE;
    let gpu = CudaRoPE::new().expect("Failed to init CudaRoPE");

    let head_dim = 16;
    let seq_len = 4;
    let num_heads = 4;
    let theta = 10000.0f32;
    let n = seq_len * num_heads * head_dim;

    let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let mut x_cpu = x_orig.clone();
    cpu.forward(&mut x_cpu, 1, seq_len, num_heads, head_dim, &positions, theta)
        .unwrap();

    let mut x_gpu = x_orig.clone();
    gpu.forward(&mut x_gpu, 1, seq_len, num_heads, head_dim, &positions, theta)
        .unwrap();

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..n {
        let diff = (x_cpu[i] - x_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-4 {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    x_cpu[i], x_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  Elements: {n}");
    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-4): {mismatches}");

    if mismatches == 0 {
        println!("  PASS: CPU and GPU match within 1e-4\n");
    } else {
        println!("  FAIL: {mismatches} elements differ\n");
        std::process::exit(1);
    }
}

fn test_rope_position_zero_identity() {
    println!("=== RoPE: Position-zero identity (GPU) ===");

    let gpu = CudaRoPE::new().unwrap();
    let head_dim = 8;
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x = original.clone();
    let positions = vec![0u32];

    gpu.forward(&mut x, 1, 1, 1, head_dim, &positions, 10000.0)
        .unwrap();

    let mut ok = true;
    for (i, (a, b)) in x.iter().zip(original.iter()).enumerate() {
        if (a - b).abs() >= 1e-4 {
            eprintln!("  MISMATCH at [{i}]: got {a}, expected {b}");
            ok = false;
        }
    }

    if ok {
        println!("  PASS: Position-zero preserves identity\n");
    } else {
        println!("  FAIL\n");
        std::process::exit(1);
    }
}

fn test_rmsnorm_cpu_vs_gpu() {
    println!("=== RMSNorm: CPU vs GPU ===");

    let cpu = CpuRMSNorm;
    let gpu = CudaRMSNorm::new().expect("Failed to init CudaRMSNorm");

    let hidden_size = 64;
    let num_tokens = 4;
    let n = num_tokens * hidden_size;
    let eps = 1e-6f32;

    let x_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + i as f32 * 0.01).collect();

    let mut x_cpu = x_orig.clone();
    cpu.forward(&mut x_cpu, &weight, hidden_size, eps).unwrap();

    let mut x_gpu = x_orig.clone();
    gpu.forward(&mut x_gpu, &weight, hidden_size, eps).unwrap();

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..n {
        let diff = (x_cpu[i] - x_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-4 {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    x_cpu[i], x_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  Elements: {n}");
    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-4): {mismatches}");

    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL\n");
        std::process::exit(1);
    }
}

fn main() {
    test_rope_cpu_vs_gpu();
    test_rope_position_zero_identity();
    test_rmsnorm_cpu_vs_gpu();
    println!("All GPU tests passed.");
}

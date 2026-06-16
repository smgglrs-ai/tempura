use oxide_kernels::attention::{Attention, CpuAttention, CudaAttention};
use oxide_kernels::matmul::{CpuMatMul, CudaMatMul, MatMul};
use oxide_kernels::rmsnorm::{CpuRMSNorm, CudaRMSNorm, RMSNorm};
use oxide_kernels::rope::{CpuRoPE, CudaRoPE, RoPE};
use std::time::Instant;

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

fn test_matmul_cpu_vs_gpu(m: usize, n: usize, k: usize) {
    println!("=== MatMul: CPU vs GPU ({m}×{k} × {k}×{n}) ===");

    let cpu = CpuMatMul;
    let gpu = CudaMatMul::new().expect("Failed to init CudaMatMul");

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

    let mut out_cpu = vec![0.0f32; m * n];
    let t0 = Instant::now();
    cpu.forward(&a, &b, &mut out_cpu, m, n, k).unwrap();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Warmup GPU
    let mut out_gpu = vec![0.0f32; m * n];
    gpu.forward(&a, &b, &mut out_gpu, m, n, k).unwrap();

    // Timed GPU run
    let mut out_gpu = vec![0.0f32; m * n];
    let t1 = Instant::now();
    gpu.forward(&a, &b, &mut out_gpu, m, n, k).unwrap();
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..m * n {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-3 {
            if mismatches < 3 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    out_cpu[i], out_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

    println!("  CPU: {cpu_ms:.3}ms, GPU: {gpu_ms:.3}ms, speedup: {speedup:.1}x");
    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-3): {mismatches}");

    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL\n");
        std::process::exit(1);
    }
}

fn test_matmul_bt_cpu_vs_gpu() {
    println!("=== MatMul B-transposed: CPU vs GPU (256×512 × 512×256) ===");

    let cpu = CpuMatMul;
    let gpu = CudaMatMul::new().unwrap();

    let m = 256;
    let n = 256;
    let k = 512;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
    let b_t: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

    let mut out_cpu = vec![0.0f32; m * n];
    cpu.forward_b_transposed(&a, &b_t, &mut out_cpu, m, n, k).unwrap();

    let mut out_gpu = vec![0.0f32; m * n];
    gpu.forward_b_transposed(&a, &b_t, &mut out_gpu, m, n, k).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        max_diff = max_diff.max((out_cpu[i] - out_gpu[i]).abs());
    }

    println!("  Max abs diff: {max_diff:.8}");
    if max_diff < 1e-3 {
        println!("  PASS\n");
    } else {
        println!("  FAIL\n");
        std::process::exit(1);
    }
}

fn test_attention_timed(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    println!("=== Attention: {label} (seq={seq_len}, kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let cpu = CpuAttention;
    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim).map(|i| ((i % 7) as f32) * 0.1).collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim).map(|i| ((i % 11) as f32) * 0.1).collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim).map(|i| ((i % 13) as f32) * 0.1).collect();

    let mut out_cpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let t0 = Instant::now();
    cpu.forward(&q, &k, &v, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Warmup
    let mut out_gpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    gpu.forward(&q, &k, &v, &mut out_gpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();

    let mut out_gpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let t1 = Instant::now();
    gpu.forward(&q, &k, &v, &mut out_gpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let max_diff: f32 = out_cpu.iter().zip(&out_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

    println!("  CPU: {cpu_ms:.3}ms, GPU: {gpu_ms:.3}ms, speedup: {speedup:.1}x");
    println!("  Max abs diff: {max_diff:.8}");
    if max_diff < 1e-3 { println!("  PASS\n"); } else { println!("  FAIL\n"); std::process::exit(1); }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  oxide-serve GPU Validation Suite (RTX 5090)        ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // --- Correctness tests ---
    test_rope_cpu_vs_gpu();
    test_rope_position_zero_identity();
    test_rmsnorm_cpu_vs_gpu();

    // --- MatMul: small sizes (transfer-dominated) ---
    test_matmul_cpu_vs_gpu(64, 64, 64);
    test_matmul_cpu_vs_gpu(256, 256, 256);

    // --- MatMul: inference-realistic sizes (MicroLlama projections) ---
    // q_proj: input[1, 1024] × weight[1024, 1024]
    test_matmul_cpu_vs_gpu(1, 1024, 1024);
    // gate_proj: input[1, 1024] × weight[5632, 1024] → [1, 5632]
    test_matmul_cpu_vs_gpu(1, 5632, 1024);
    // lm_head: hidden[1, 1024] × weight[32000, 1024] → [1, 32000]
    test_matmul_cpu_vs_gpu(1, 32000, 1024);
    // Batched prefill: input[128, 1024] × weight[1024, 1024]
    test_matmul_cpu_vs_gpu(128, 1024, 1024);

    // --- MatMul B-transposed ---
    test_matmul_bt_cpu_vs_gpu();

    // --- Attention: small (correctness) ---
    test_attention_timed("MHA small", 4, 4, 4, 4, 16);
    test_attention_timed("GQA small", 2, 4, 4, 2, 8);

    // --- Attention: MicroLlama-scale ---
    // MicroLlama: 16 heads, 4 kv heads, head_dim=64
    test_attention_timed("GQA MicroLlama", 8, 128, 16, 4, 64);

    println!("All GPU tests passed.");
}

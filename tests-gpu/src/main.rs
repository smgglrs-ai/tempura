mod test_pack;

use half::f16;
use std::time::Instant;
use tempura_kernels::attention::{Attention, CpuAttention, CudaAttention};
use tempura_kernels::matmul::{CpuMatMul, CudaMatMul, MatMul};
use tempura_kernels::quantized::{ActivationQ8, CpuMatMulQ4K, CpuMatMulQ4KOpt, CudaMatMulQ4K, MatMulQ};
use tempura_kernels::rmsnorm::{CpuRMSNorm, CudaRMSNorm, RMSNorm};
use tempura_kernels::rope::{CpuRoPE, CudaRoPE, RoPE};
use tempura_kernels::{Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};

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

fn test_rmsnorm_cpu_vs_gpu_large() {
    println!("=== RMSNorm: CPU vs GPU (hidden=4096, tokens=8) ===");

    let cpu = CpuRMSNorm;
    let gpu = CudaRMSNorm::new().expect("Failed to init CudaRMSNorm");

    let hidden_size = 4096;
    let num_tokens = 8;
    let n = num_tokens * hidden_size;
    let eps = 1e-6f32;

    let x_orig: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
    let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + (i as f32 * 0.0001).cos()).collect();

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

fn test_flash_attention(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    tolerance: f32,
) {
    println!("=== Flash Attention: {label} (seq={seq_len}, kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let cpu = CpuAttention;
    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    let mut out_cpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    cpu.forward(
        &q, &k, &v, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
        scale,
    )
    .unwrap();

    let mut out_gpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    gpu.forward(
        &q, &k, &v, &mut out_gpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
        scale,
    )
    .unwrap();

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..out_cpu.len() {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= tolerance {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    out_cpu[i], out_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>={tolerance}): {mismatches}");

    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!(
            "  FAIL ({mismatches}/{} elements)\n",
            out_cpu.len()
        );
        std::process::exit(1);
    }
}

fn bench_flash_attention(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    println!("=== Flash Attention Bench: {label} ===");

    let cpu = CpuAttention;
    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    let mut out_cpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let t0 = Instant::now();
    cpu.forward(
        &q, &k, &v, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
        scale,
    )
    .unwrap();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Warmup
    let mut out_gpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    gpu.forward(
        &q, &k, &v, &mut out_gpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
        scale,
    )
    .unwrap();

    let iters = 10;
    let mut out_gpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let t1 = Instant::now();
    for _ in 0..iters {
        gpu.forward(
            &q, &k, &v, &mut out_gpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
            scale,
        )
        .unwrap();
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let max_diff: f32 = out_cpu
        .iter()
        .zip(&out_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

    println!("  CPU: {cpu_ms:.3}ms, GPU: {gpu_ms:.3}ms ({iters} iters avg), speedup: {speedup:.1}x");
    println!("  Max abs diff: {max_diff:.8}");
    if max_diff < 0.05 {
        println!("  PASS\n");
    } else {
        println!("  WARN: large diff (FP16 accumulation expected)\n");
    }
}

fn bench_flash_vs_naive(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    println!("=== Flash vs Naive GPU: {label} (seq={seq_len}, kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total = batch * seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    let stream = gpu.gpu().stream();
    let q_dev = tempura_kernels::gpu::to_device(&stream, &q).unwrap();
    let k_dev = tempura_kernels::gpu::to_device(&stream, &k).unwrap();
    let v_dev = tempura_kernels::gpu::to_device(&stream, &v).unwrap();

    let iters = 20;
    let warmup = 5;

    // Benchmark naive GPU
    let mut out_naive = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device_naive(
            &q_dev, &k_dev, &v_dev, &mut out_naive, batch, seq_len, kv_len,
            num_heads, num_kv_heads, head_dim, scale,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device_naive(
            &q_dev, &k_dev, &v_dev, &mut out_naive, batch, seq_len, kv_len,
            num_heads, num_kv_heads, head_dim, scale,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    let naive_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Benchmark flash GPU
    let mut out_flash = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device(
            &q_dev, &k_dev, &v_dev, &mut out_flash, batch, seq_len, kv_len,
            num_heads, num_kv_heads, head_dim, scale,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device(
            &q_dev, &k_dev, &v_dev, &mut out_flash, batch, seq_len, kv_len,
            num_heads, num_kv_heads, head_dim, scale,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    let flash_us = t1.elapsed().as_micros() as f64 / iters as f64;

    // Compare outputs
    let out_n = tempura_kernels::gpu::to_host(&stream, &out_naive).unwrap();
    let out_f = tempura_kernels::gpu::to_host(&stream, &out_flash).unwrap();
    let max_diff: f32 = out_n.iter().zip(&out_f).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);

    let speedup = if flash_us > 0.0 { naive_us / flash_us } else { 0.0 };
    println!("  Naive GPU:  {naive_us:.1}µs");
    println!("  Flash GPU:  {flash_us:.1}µs");
    println!("  Speedup:    {speedup:.2}x");
    println!("  Max diff (flash vs naive): {max_diff:.8}");
    println!();
}

fn bench_decode_splitk(
    label: &str,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    println!("=== Decode Split-K: {label} (kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let cpu = CpuAttention;
    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let seq_len = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total = batch * seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..total)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    // CPU reference
    let mut out_cpu = vec![0.0f32; total];
    cpu.forward(&q, &k, &v, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();

    // GPU decode (split-K, dispatched automatically for seq_len=1)
    let stream = gpu.gpu().stream();
    let q_dev = tempura_kernels::gpu::to_device(&stream, &q).unwrap();
    let k_dev = tempura_kernels::gpu::to_device(&stream, &k).unwrap();
    let v_dev = tempura_kernels::gpu::to_device(&stream, &v).unwrap();

    let iters = 20;
    let warmup = 5;

    let mut out_dev = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device(&q_dev, &k_dev, &v_dev, &mut out_dev, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device(&q_dev, &k_dev, &v_dev, &mut out_dev, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let decode_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Compare vs CPU
    let out_gpu = tempura_kernels::gpu::to_host(&stream, &out_dev).unwrap();
    let max_diff: f32 = out_cpu.iter().zip(&out_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);

    // Also time the naive path for comparison
    let mut out_naive = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device_naive(&q_dev, &k_dev, &v_dev, &mut out_naive, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device_naive(&q_dev, &k_dev, &v_dev, &mut out_naive, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let naive_us = t1.elapsed().as_micros() as f64 / iters as f64;

    let speedup_vs_naive = if decode_us > 0.0 { naive_us / decode_us } else { 0.0 };

    println!("  Decode split-K: {decode_us:.1}µs");
    println!("  Naive GPU:      {naive_us:.1}µs");
    println!("  Speedup vs naive: {speedup_vs_naive:.1}x");
    println!("  Max diff vs CPU:  {max_diff:.8}");
    if max_diff < 0.01 {
        println!("  PASS\n");
    } else {
        println!("  FAIL (diff too large)\n");
        std::process::exit(1);
    }
}

fn test_flash_attention_f16kv(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    tolerance: f32,
) {
    println!("=== Flash F16 KV: {label} (seq={seq_len}, kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let cpu = CpuAttention;
    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k_f32: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v_f32: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    let mut out_cpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    cpu.forward(
        &q, &k_f32, &v_f32, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads,
        head_dim, scale,
    )
    .unwrap();

    let k_f16: Vec<u16> = k_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let v_f16: Vec<u16> = v_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();

    let stream = gpu.gpu().stream();
    let q_dev = tempura_kernels::gpu::to_device(&stream, &q).unwrap();
    let k_dev = tempura_kernels::gpu::to_device_u16(&stream, &k_f16).unwrap();
    let v_dev = tempura_kernels::gpu::to_device_u16(&stream, &v_f16).unwrap();
    let total = batch * seq_len * num_heads * head_dim;
    let mut out_dev = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();

    gpu.forward_device_f16kv(
        &q_dev, &k_dev, &v_dev, &mut out_dev, batch, seq_len, kv_len, num_heads, num_kv_heads,
        head_dim, scale,
    )
    .unwrap();

    let out_gpu = tempura_kernels::gpu::to_host(&stream, &out_dev).unwrap();

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..out_cpu.len() {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= tolerance {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    out_cpu[i], out_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>={tolerance}): {mismatches}");
    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL ({mismatches}/{} elements)\n", out_cpu.len());
        std::process::exit(1);
    }
}

fn bench_flash_f32_vs_f16kv(
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    println!("=== Flash F32 vs F16 KV: {label} (seq={seq_len}, kv={kv_len}, h={num_heads}/{num_kv_heads}, d={head_dim}) ===");

    let gpu = CudaAttention::new().unwrap();
    let batch = 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total = batch * seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
        .collect();
    let k_f32: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let v_f32: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| ((i % 13) as f32) * 0.1 - 0.6)
        .collect();

    let k_f16: Vec<u16> = k_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let v_f16: Vec<u16> = v_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();

    let stream = gpu.gpu().stream();
    let q_dev = tempura_kernels::gpu::to_device(&stream, &q).unwrap();
    let k_dev_f32 = tempura_kernels::gpu::to_device(&stream, &k_f32).unwrap();
    let v_dev_f32 = tempura_kernels::gpu::to_device(&stream, &v_f32).unwrap();
    let k_dev_f16 = tempura_kernels::gpu::to_device_u16(&stream, &k_f16).unwrap();
    let v_dev_f16 = tempura_kernels::gpu::to_device_u16(&stream, &v_f16).unwrap();

    let iters = 20;
    let warmup = 5;

    // Benchmark flash f32
    let mut out_f32 = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device(&q_dev, &k_dev_f32, &v_dev_f32, &mut out_f32, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device(&q_dev, &k_dev_f32, &v_dev_f32, &mut out_f32, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let f32_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Benchmark flash f16kv
    let mut out_f16 = tempura_kernels::gpu::alloc_device(&stream, total).unwrap();
    for _ in 0..warmup {
        gpu.forward_device_f16kv(&q_dev, &k_dev_f16, &v_dev_f16, &mut out_f16, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        gpu.forward_device_f16kv(&q_dev, &k_dev_f16, &v_dev_f16, &mut out_f16, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale).unwrap();
    }
    stream.synchronize().unwrap();
    let f16_us = t1.elapsed().as_micros() as f64 / iters as f64;

    let speedup = if f16_us > 0.0 { f32_us / f16_us } else { 0.0 };
    println!("  Flash F32:   {f32_us:.1}µs");
    println!("  Flash F16KV: {f16_us:.1}µs");
    println!("  Speedup:     {speedup:.2}x");
    println!();
}

fn test_q4k_cpu_vs_gpu(m: usize, n: usize, k: usize) {
    println!("=== Q4K MatMul: CPU vs GPU  m={m} n={n} k={k} ===");

    assert!(k % Q4K_BLOCK_SIZE == 0, "k must be multiple of 256");

    let cpu = CpuMatMulQ4K;
    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();

    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let mut out_cpu = vec![0.0f32; m * n];
    let mut out_gpu = vec![0.0f32; m * n];

    let t_cpu = Instant::now();
    cpu.forward_b_transposed_q4k(&a, &b_quantized, &mut out_cpu, m, n, k)
        .unwrap();
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

    let t_gpu = Instant::now();
    gpu.forward_b_transposed_q4k(&a, &b_quantized, &mut out_gpu, m, n, k)
        .unwrap();
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..m * n {
        let abs_diff = (out_cpu[i] - out_gpu[i]).abs();
        let denom = out_cpu[i].abs().max(1e-6);
        let rel_diff = abs_diff / denom;
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        if rel_diff > 1e-4 {
            if mismatches < 3 {
                eprintln!(
                    "  MISMATCH [{i}]: cpu={:.4}, gpu={:.4}, abs={:.6}, rel={:.6}",
                    out_cpu[i], out_gpu[i], abs_diff, rel_diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  CPU: {cpu_ms:.2}ms  GPU: {gpu_ms:.2}ms");
    println!("  Max abs diff: {max_abs_diff:.6}  Max rel diff: {max_rel_diff:.8}");
    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL: {mismatches} elements with relative error > 1e-4\n");
        std::process::exit(1);
    }
}

fn bench_q4k_resident(m: usize, n: usize, k: usize, label: &str) {
    println!("=== Q4K Resident Bench: {label}  m={m} n={n} k={k} ===");

    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    // Warmup
    gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();
        stream.synchronize().unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[5];
    let gflops = 2.0 * m as f64 * n as f64 * k as f64 / (median / 1000.0) / 1e9;

    println!("  Kernel-only (no transfer): {median:.3}ms  {gflops:.0} GFLOPS\n");
}

fn bench_q4k_splitk(n: usize, k: usize, num_splits: usize, label: &str) {
    println!("=== Q4K Split-K Bench: {label}  n={n} k={k} splits={num_splits} ===");

    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut partials = tempura_kernels::gpu::alloc_device(&stream, n * num_splits).unwrap();

    // Warmup
    gpu.forward_q4k_splitk_device(&a_dev, &b_dev, &mut c_dev, n, k, num_splits, &mut partials).unwrap();

    // Correctness check vs non-split
    let mut c_ref = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_ref, 1, n, k).unwrap();
    let out_splitk = tempura_kernels::gpu::to_host(&stream, &c_dev).unwrap();
    let out_ref = tempura_kernels::gpu::to_host(&stream, &c_ref).unwrap();
    let max_diff: f32 = out_splitk.iter().zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_rel: f32 = out_splitk.iter().zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
        .fold(0.0f32, f32::max);

    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.forward_q4k_splitk_device(&a_dev, &b_dev, &mut c_dev, n, k, num_splits, &mut partials).unwrap();
        stream.synchronize().unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[5];
    let gflops = 2.0 * n as f64 * k as f64 / (median / 1000.0) / 1e9;

    println!("  Split-K: {median:.3}ms  {gflops:.0} GFLOPS  (max_rel_diff={max_rel:.2e})");
    if max_rel > 1e-3 {
        println!("  WARN: split-K vs non-split relative diff {max_rel:.6}");
    }
    println!();
}

fn bench_q4k_warp(n: usize, k: usize, label: &str) {
    println!("=== Q4K Warp-Coop Bench: {label}  n={n} k={k} ===");

    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_warp = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut c_ref = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    // Reference: naive kernel
    gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_ref, 1, n, k).unwrap();

    // Warmup warp kernel
    gpu.gemv_q4k_warp_device(&a_dev, &b_dev, &mut c_warp, n, k).unwrap();

    // Correctness check
    let out_warp = tempura_kernels::gpu::to_host(&stream, &c_warp).unwrap();
    let out_ref = tempura_kernels::gpu::to_host(&stream, &c_ref).unwrap();
    let max_rel: f32 = out_warp.iter().zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
        .fold(0.0f32, f32::max);

    // Benchmark naive (resident)
    let mut naive_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_ref, 1, n, k).unwrap();
        stream.synchronize().unwrap();
        naive_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    naive_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let naive_ms = naive_times[5];

    // Benchmark warp kernel
    let mut warp_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_warp_device(&a_dev, &b_dev, &mut c_warp, n, k).unwrap();
        stream.synchronize().unwrap();
        warp_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    warp_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let warp_ms = warp_times[5];

    let gflops = 2.0 * n as f64 * k as f64 / (warp_ms / 1000.0) / 1e9;
    let speedup = naive_ms / warp_ms;
    let weight_bytes = n as f64 * (k as f64 / 256.0) * 144.0;
    let bw_gbps = weight_bytes / (warp_ms / 1000.0) / 1e9;

    println!("  Naive:  {naive_ms:.3}ms");
    println!("  Warp:   {warp_ms:.3}ms  {gflops:.0} GFLOPS  {bw_gbps:.0} GB/s");
    println!("  Speedup: {speedup:.2}x  (max_rel_diff={max_rel:.2e})\n");
}

fn bench_q4k_coop(n: usize, k: usize, label: &str) {
    println!("=== Q4K Coop Bench: {label}  n={n} k={k} ===");

    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_coop = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut c_ref = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    // Reference: naive kernel
    gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_ref, 1, n, k).unwrap();

    // Warmup coop kernel
    gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_coop, n, k).unwrap();

    // Correctness check
    let out_coop = tempura_kernels::gpu::to_host(&stream, &c_coop).unwrap();
    let out_ref = tempura_kernels::gpu::to_host(&stream, &c_ref).unwrap();
    let max_abs: f32 = out_coop.iter().zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_rel: f32 = out_coop.iter().zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
        .fold(0.0f32, f32::max);

    // Benchmark naive (resident)
    let mut naive_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.forward_q4k_device(&a_dev, &b_dev, &mut c_ref, 1, n, k).unwrap();
        stream.synchronize().unwrap();
        naive_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    naive_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let naive_ms = naive_times[5];

    // Benchmark coop kernel
    let mut coop_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_coop, n, k).unwrap();
        stream.synchronize().unwrap();
        coop_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    coop_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let coop_ms = coop_times[5];

    let gflops = 2.0 * n as f64 * k as f64 / (coop_ms / 1000.0) / 1e9;
    let speedup = naive_ms / coop_ms;
    let weight_bytes = n as f64 * (k as f64 / 256.0) * 144.0;
    let bw_gbps = weight_bytes / (coop_ms / 1000.0) / 1e9;

    println!("  Naive:  {naive_ms:.3}ms");
    println!("  Coop:   {coop_ms:.3}ms  {gflops:.0} GFLOPS  {bw_gbps:.0} GB/s");
    println!("  Speedup: {speedup:.2}x  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}");
    if max_rel > 1e-3 {
        eprintln!("  WARN: coop vs naive relative diff {max_rel:.6}");
    }
    println!();
}

fn bench_q4k_dp4a(n: usize, k: usize, label: &str) {
    println!("=== Q4K dp4a Bench: {label}  n={n} k={k} ===");

    let cpu = CpuMatMulQ4K;
    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    // Pre-quantize activation to Q8
    let a_q8 = ActivationQ8::from_f32(&a);

    // CPU reference (f32 path)
    let mut cpu_out = vec![0.0f32; n];
    cpu.forward_b_transposed_q4k(&a, &b_quantized, &mut cpu_out, 1, n, k).unwrap();

    // GPU dp4a path
    let a_q8_dev = tempura_kernels::gpu::to_device_u32(&stream, &a_q8.data).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_dp4a = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut c_coop = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    gpu.gemv_q4k_q8_dp4a_device(&a_q8_dev, &b_dev, &mut c_dp4a, n, k).unwrap();
    let gpu_out = tempura_kernels::gpu::to_host(&stream, &c_dp4a).unwrap();

    // DualPipeline: compare dp4a vs CPU f32
    let max_abs: f32 = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_rel: f32 = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6)).fold(0.0f32, f32::max);

    // Bench coop v1 (f32 activation)
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_coop, n, k).unwrap();
    let mut coop_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_coop, n, k).unwrap();
        stream.synchronize().unwrap();
        coop_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    coop_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let coop_ms = coop_times[5];

    // Bench dp4a
    let mut dp4a_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_q8_dp4a_device(&a_q8_dev, &b_dev, &mut c_dp4a, n, k).unwrap();
        stream.synchronize().unwrap();
        dp4a_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    dp4a_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dp4a_ms = dp4a_times[5];

    let speedup = coop_ms / dp4a_ms;
    let weight_bytes = n as f64 * (k as f64 / 256.0) * 144.0;
    let bw = weight_bytes / (dp4a_ms / 1000.0) / 1e9;

    println!("  Coop v1 (f32): {coop_ms:.3}ms");
    println!("  dp4a (Q8):     {dp4a_ms:.3}ms  {bw:.0} GB/s");
    println!("  Speedup: {speedup:.2}x");
    println!("  DualPipeline vs CPU: max_abs={max_abs:.2e}  max_rel={max_rel:.2e}\n");
}

fn bench_q4k_coop_v2(n: usize, k: usize, label: &str) {
    println!("=== Q4K Coop v2 Bench: {label}  n={n} k={k} ===");

    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_quantized).unwrap();
    let mut c_v2 = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut c_v1 = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    // Reference: coop v1
    gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_v1, n, k).unwrap();
    gpu.gemv_q4k_coop_v2_device(&a_dev, &b_dev, &mut c_v2, n, k).unwrap();

    let out_v1 = tempura_kernels::gpu::to_host(&stream, &c_v1).unwrap();
    let out_v2 = tempura_kernels::gpu::to_host(&stream, &c_v2).unwrap();
    let max_rel: f32 = out_v1.iter().zip(out_v2.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
        .fold(0.0f32, f32::max);

    // Bench v1
    let mut v1_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_coop_device(&a_dev, &b_dev, &mut c_v1, n, k).unwrap();
        stream.synchronize().unwrap();
        v1_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    v1_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let v1_ms = v1_times[5];

    // Bench v2
    let mut v2_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_coop_v2_device(&a_dev, &b_dev, &mut c_v2, n, k).unwrap();
        stream.synchronize().unwrap();
        v2_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    v2_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let v2_ms = v2_times[5];

    let speedup = v1_ms / v2_ms;
    let weight_bytes = n as f64 * (k as f64 / 256.0) * 144.0;
    let bw = weight_bytes / (v2_ms / 1000.0) / 1e9;

    println!("  Coop v1: {v1_ms:.3}ms");
    println!("  Coop v2: {v2_ms:.3}ms  {bw:.0} GB/s");
    println!("  Speedup: {speedup:.2}x  max_rel={max_rel:.2e}\n");
}

fn bench_fused_gate_up(n: usize, k: usize, label: &str) {
    println!("=== Fused gate+up+SiLU Bench: {label}  n={n} k={k} ===");

    let cpu = CpuMatMulQ4K;
    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");
    let ctx = gpu.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_gate: Vec<u8> = (0..b_bytes).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    let b_up: Vec<u8> = (0..b_bytes).map(|i| ((i * 13 + 5) % 256) as u8).collect();

    // --- CPU sequential reference (DualPipeline) ---
    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];
    cpu.forward_b_transposed_q4k(&a, &b_gate, &mut gate_out, 1, n, k).unwrap();
    cpu.forward_b_transposed_q4k(&a, &b_up, &mut up_out, 1, n, k).unwrap();
    let mut cpu_ref = vec![0.0f32; n];
    for i in 0..n {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        cpu_ref[i] = silu * up_out[i];
    }

    // --- GPU fused ---
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let bg_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_gate).unwrap();
    let bu_dev = tempura_kernels::gpu::to_device_u8(&stream, &b_up).unwrap();
    let mut out_dev = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    gpu.fused_gate_up_silu_device(&a_dev, &bg_dev, &bu_dev, &mut out_dev, n, k).unwrap();
    let gpu_out = tempura_kernels::gpu::to_host(&stream, &out_dev).unwrap();

    // DualPipeline comparison
    let max_abs: f32 = cpu_ref.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_rel: f32 = cpu_ref.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6)).fold(0.0f32, f32::max);

    // Benchmark: fused vs 2× coop + silu
    let mut sep_times = Vec::new();
    let mut cg = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    let mut cu = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();
    gpu.gemv_q4k_coop_device(&a_dev, &bg_dev, &mut cg, n, k).unwrap();
    gpu.gemv_q4k_coop_device(&a_dev, &bu_dev, &mut cu, n, k).unwrap();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.gemv_q4k_coop_device(&a_dev, &bg_dev, &mut cg, n, k).unwrap();
        gpu.gemv_q4k_coop_device(&a_dev, &bu_dev, &mut cu, n, k).unwrap();
        tempura_kernels::gpu::silu_mul_device(&ctx, &stream, &cg, &cu, &mut out_dev).unwrap();
        stream.synchronize().unwrap();
        sep_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    sep_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sep_ms = sep_times[5];

    let mut fused_times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        gpu.fused_gate_up_silu_device(&a_dev, &bg_dev, &bu_dev, &mut out_dev, n, k).unwrap();
        stream.synchronize().unwrap();
        fused_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    fused_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fused_ms = fused_times[5];

    let speedup = sep_ms / fused_ms;
    let weight_bytes = 2.0 * n as f64 * (k as f64 / 256.0) * 144.0;
    let bw = weight_bytes / (fused_ms / 1000.0) / 1e9;

    println!("  Separate (2×coop + silu): {sep_ms:.3}ms");
    println!("  Fused:                    {fused_ms:.3}ms  {bw:.0} GB/s");
    println!("  Speedup: {speedup:.2}x");
    println!("  DualPipeline: max_abs={max_abs:.2e}  max_rel={max_rel:.2e}");
    if max_rel > 1e-3 {
        eprintln!("  WARN: fused vs CPU sequential relative diff {max_rel:.6}");
    }
    println!();
}

fn bench_q4k_gpu(m: usize, n: usize, k: usize, label: &str) {
    println!("=== Q4K Bench: {label}  m={m} n={n} k={k} ===");

    let cpu_opt = CpuMatMulQ4KOpt;
    let gpu = CudaMatMulQ4K::new().expect("Failed to init CudaMatMulQ4K");

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let b_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    let b_quantized: Vec<u8> = (0..b_bytes).map(|i| (i % 256) as u8).collect();
    let mut output = vec![0.0f32; m * n];

    // Warmup GPU (includes module load + first kernel launch)
    gpu.forward_b_transposed_q4k(&a, &b_quantized, &mut output, m, n, k).unwrap();

    // Benchmark GPU (3 runs, take median)
    let mut gpu_times = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        gpu.forward_b_transposed_q4k(&a, &b_quantized, &mut output, m, n, k).unwrap();
        gpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu_median = gpu_times[2];

    // Benchmark CPU optimized (3 runs, take median)
    let mut cpu_times = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        cpu_opt.forward_b_transposed_q4k(&a, &b_quantized, &mut output, m, n, k).unwrap();
        cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu_median = cpu_times[1];

    let speedup = cpu_median / gpu_median;
    let gflops_equiv = 2.0 * m as f64 * n as f64 * k as f64 / (gpu_median / 1000.0) / 1e9;

    println!("  CPU (opt): {cpu_median:.2}ms  GPU: {gpu_median:.2}ms  Speedup: {speedup:.1}x");
    println!("  GPU equiv GFLOPS: {gflops_equiv:.1}\n");
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura-serve GPU Validation Suite (RTX 5090)        ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // --- Correctness tests ---
    test_rope_cpu_vs_gpu();
    test_rope_position_zero_identity();
    test_rmsnorm_cpu_vs_gpu();
    test_rmsnorm_cpu_vs_gpu_large();

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

    // --- Slice columns (fused projection support) ---
    test_slice_columns();

    // --- Pack f16x2 diagnostic ---
    test_pack::run_pack_test();
    test_pack::run_four_output_test();
    test_pack::run_mma_raw_test();

    // --- MMA GEMM (tensor core) ---
    test_matmul_mma_bt_identity();
    test_matmul_mma_bt(32, 32, 16);
    test_matmul_mma_bt(64, 64, 64);
    test_matmul_mma_bt(128, 128, 128);

    // --- Flash Attention (MMA tensor core) ---
    // head_dim=128 triggers the flash kernel path
    test_flash_attention("MHA small", 4, 4, 4, 4, 128, 0.05);
    test_flash_attention("GQA small", 4, 16, 8, 2, 128, 0.05);
    test_flash_attention("causal prefill", 64, 64, 1, 1, 128, 0.05);
    test_flash_attention("decode single-token", 1, 256, 8, 4, 128, 0.05);
    test_flash_attention("multi-tile KV", 32, 256, 4, 2, 128, 0.05);
    test_flash_attention("long context", 128, 512, 8, 4, 128, 0.05);

    // Benchmark: flash vs CPU
    bench_flash_attention("LLaMA-7B decode", 1, 2048, 32, 8, 128);
    bench_flash_attention("LLaMA-7B prefill-128", 128, 128, 32, 8, 128);

    // Benchmark: flash vs naive GPU (the real comparison)
    bench_flash_vs_naive("decode s=1 kv=256", 1, 256, 32, 8, 128);
    bench_flash_vs_naive("decode s=1 kv=2048", 1, 2048, 32, 8, 128);
    bench_flash_vs_naive("decode s=1 kv=4096", 1, 4096, 32, 8, 128);
    bench_flash_vs_naive("decode s=1 kv=8192", 1, 8192, 32, 8, 128);
    bench_flash_vs_naive("prefill s=128 kv=128", 128, 128, 32, 8, 128);
    bench_flash_vs_naive("prefill s=256 kv=256", 256, 256, 32, 8, 128);
    bench_flash_vs_naive("prefill s=512 kv=512", 512, 512, 32, 8, 128);
    bench_flash_vs_naive("prefill s=1024 kv=1024", 1024, 1024, 32, 8, 128);
    bench_flash_vs_naive("prefill s=2048 kv=2048", 2048, 2048, 32, 8, 128);
    bench_flash_vs_naive("prefill s=4096 kv=4096", 4096, 4096, 32, 8, 128);

    // --- Decode split-K vs prefill-as-decode benchmark ---
    bench_decode_splitk("decode kv=64", 64, 32, 8, 128);
    bench_decode_splitk("decode kv=256", 256, 32, 8, 128);
    bench_decode_splitk("decode kv=1024", 1024, 32, 8, 128);
    bench_decode_splitk("decode kv=2048", 2048, 32, 8, 128);
    bench_decode_splitk("decode kv=4096", 4096, 32, 8, 128);
    bench_decode_splitk("decode kv=8192", 8192, 32, 8, 128);

    // --- Flash Attention F16 KV correctness ---
    test_flash_attention_f16kv("MHA small f16kv", 4, 4, 4, 4, 128, 0.05);
    test_flash_attention_f16kv("GQA f16kv", 4, 16, 8, 2, 128, 0.05);
    test_flash_attention_f16kv("causal prefill f16kv", 64, 64, 1, 1, 128, 0.05);
    test_flash_attention_f16kv("decode f16kv", 1, 256, 8, 4, 128, 0.05);

    // --- Flash F32 vs F16 KV bandwidth benchmark ---
    bench_flash_f32_vs_f16kv("prefill s=128", 128, 128, 32, 8, 128);
    bench_flash_f32_vs_f16kv("prefill s=256", 256, 256, 32, 8, 128);
    bench_flash_f32_vs_f16kv("prefill s=512", 512, 512, 32, 8, 128);
    bench_flash_f32_vs_f16kv("prefill s=1024", 1024, 1024, 32, 8, 128);
    bench_flash_f32_vs_f16kv("prefill s=2048", 2048, 2048, 32, 8, 128);
    bench_flash_f32_vs_f16kv("prefill s=4096", 4096, 4096, 32, 8, 128);
    bench_flash_f32_vs_f16kv("decode s=1 kv=2048", 1, 2048, 32, 8, 128);

    // --- Quantized GEMM Q4K: CPU vs GPU correctness ---
    test_q4k_cpu_vs_gpu(1, 16, 256);
    test_q4k_cpu_vs_gpu(1, 64, 256);
    test_q4k_cpu_vs_gpu(1, 256, 512);
    test_q4k_cpu_vs_gpu(1, 1024, 1024);

    // --- Quantized GEMM Q4K: performance benchmarks ---
    bench_q4k_gpu(1, 3584, 3584, "qkv_proj (decode)");
    bench_q4k_gpu(1, 18944, 3584, "gate_proj (decode)");
    bench_q4k_gpu(1, 3584, 18944, "down_proj (decode)");
    bench_q4k_gpu(1, 151936, 3584, "lm_head (decode)");
    bench_q4k_gpu(128, 3584, 3584, "qkv_proj (prefill-128)");

    // --- Quantized GEMM Q4K: resident weights (kernel-only, no transfer) ---
    bench_q4k_resident(1, 3584, 3584, "qkv_proj (decode)");
    bench_q4k_resident(1, 18944, 3584, "gate_proj (decode)");
    bench_q4k_resident(1, 3584, 18944, "down_proj (decode)");
    bench_q4k_resident(1, 151936, 3584, "lm_head (decode)");
    bench_q4k_resident(128, 3584, 3584, "qkv_proj (prefill-128)");

    // --- Split-K: decode matmul bottleneck optimization ---
    bench_q4k_splitk(3584, 18944, 4, "down_proj 4-split");
    bench_q4k_splitk(3584, 18944, 8, "down_proj 8-split");
    bench_q4k_splitk(3584, 18944, 16, "down_proj 16-split");
    bench_q4k_splitk(3584, 18944, 32, "down_proj 32-split");
    bench_q4k_splitk(3584, 3584, 8, "qkv_proj 8-split");
    bench_q4k_splitk(18944, 3584, 8, "gate_proj 8-split");

    // --- Warp-cooperative Q4K GEMV ---
    bench_q4k_warp(3584, 3584, "qkv_proj (decode)");
    bench_q4k_warp(18944, 3584, "gate_proj (decode)");
    bench_q4k_warp(3584, 18944, "down_proj (decode)");
    bench_q4k_warp(151936, 3584, "lm_head (decode)");

    // --- Intra-block cooperative Q4K GEMV ---
    bench_q4k_coop(3584, 3584, "qkv_proj (decode)");
    bench_q4k_coop(18944, 3584, "gate_proj (decode)");
    bench_q4k_coop(3584, 18944, "down_proj (decode)");
    bench_q4k_coop(151936, 3584, "lm_head (decode)");

    // --- dp4a: Q4K × Q8 integer dot product ---
    bench_q4k_dp4a(3584, 3584, "qkv_proj (decode)");
    bench_q4k_dp4a(18944, 3584, "gate_proj (decode)");
    bench_q4k_dp4a(3584, 18944, "down_proj (decode)");
    bench_q4k_dp4a(151936, 3584, "lm_head (decode)");

    // --- Coop v2: shared-memory activation tile ---
    bench_q4k_coop_v2(3584, 3584, "qkv_proj (decode)");
    bench_q4k_coop_v2(18944, 3584, "gate_proj (decode)");
    bench_q4k_coop_v2(3584, 18944, "down_proj (decode)");
    bench_q4k_coop_v2(151936, 3584, "lm_head (decode)");

    // --- Fused gate+up+SiLU (DualPipeline validated) ---
    bench_fused_gate_up(18944, 3584, "Qwen3-8B FFN");

    println!("All GPU tests passed.");
}

fn test_matmul_mma_bt_identity() {
    println!("=== MatMul MMA BT: identity test (16×16 × 16×16) ===");

    // A = I_16 (16×16 identity), B = I_16 (16×16 identity, B-transposed)
    // C = A × B^T = I_16 × I_16 = I_16 → C[i,j] = 1 if i==j else 0
    let m = 16usize;
    let n = 16;
    let k = 16;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; n * k];
    let mut i = 0;
    while i < 16 {
        a[i * k + i] = 1.0;
        b[i * k + i] = 1.0;
        i += 1;
    }

    let cpu = CpuMatMul;
    let mut out_cpu = vec![0.0f32; m * n];
    cpu.forward_b_transposed(&a, &b, &mut out_cpu, m, n, k).unwrap();

    let gpu_ctx = tempura_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = tempura_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(32), (m as u32).div_ceil(32), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = tempura_kernels::gpu::to_host(&stream, &c_dev).unwrap();

    println!("  CPU[0..8]:  {:?}", &out_cpu[0..8]);
    println!("  GPU[0..8]:  {:?}", &out_gpu[0..8]);
    println!("  CPU[16..24]: {:?}", &out_cpu[16..24]);
    println!("  GPU[16..24]: {:?}", &out_gpu[16..24]);

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    let mut i = 0;
    while i < m * n {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-3 {
            mismatches += 1;
        }
        i += 1;
    }

    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-3): {mismatches}");
    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL ({mismatches}/{} elements)\n", m * n);
        std::process::exit(1);
    }
}

fn test_mma_single_tile() {
    println!("=== MMA single-tile diagnostic (m=16, n=8, k=16) ===");

    let m = 16usize;
    let n = 8usize;
    let k = 16usize;

    let mut a = vec![0.0f32; m * k];
    for i in 0..16 {
        a[i * k + i] = 1.0;
    }

    let mut b = vec![0.0f32; n * k];
    for i in 0..8 {
        b[i * k + i] = 1.0;
    }

    let cpu = CpuMatMul;
    let mut out_cpu = vec![0.0f32; m * n];
    cpu.forward_b_transposed(&a, &b, &mut out_cpu, m, n, k).unwrap();

    let gpu_ctx = tempura_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = tempura_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(32), (m as u32).div_ceil(32), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = tempura_kernels::gpu::to_host(&stream, &c_dev).unwrap();

    println!("  Expected: identity top-left 8x8, zeros elsewhere");
    println!("  Row-by-row comparison (CPU | GPU):");
    for row in 0..m {
        let cpu_row: Vec<String> = (0..n).map(|c| format!("{:5.2}", out_cpu[row * n + c])).collect();
        let gpu_row: Vec<String> = (0..n).map(|c| format!("{:5.2}", out_gpu[row * n + c])).collect();
        let marker = if cpu_row != gpu_row { " <-- DIFF" } else { "" };
        println!("    r{row:02}: CPU=[{}]  GPU=[{}]{marker}", cpu_row.join(", "), gpu_row.join(", "));
    }

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..m * n {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-3 {
            mismatches += 1;
        }
    }

    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-3): {mismatches}");

    println!("\n  --- Random data test ---");
    let a2: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 37) as f32 * 0.1).collect();
    let b2: Vec<f32> = (0..n * k).map(|i| ((i * 11 + 5) % 41) as f32 * 0.1).collect();

    let mut out_cpu2 = vec![0.0f32; m * n];
    cpu.forward_b_transposed(&a2, &b2, &mut out_cpu2, m, n, k).unwrap();

    let a2_dev = tempura_kernels::gpu::to_device(&stream, &a2).unwrap();
    let b2_dev = tempura_kernels::gpu::to_device(&stream, &b2).unwrap();
    let mut c2_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a2_dev, &b2_dev, &mut c2_dev)
        .unwrap();
    let out_gpu2 = tempura_kernels::gpu::to_host(&stream, &c2_dev).unwrap();

    println!("  Row-by-row (first 4 cols):");
    let mut row = 0usize;
    while row < m {
        println!(
            "    r{row:02}: CPU=[{:.0}, {:.0}, {:.0}, {:.0}] GPU=[{:.0}, {:.0}, {:.0}, {:.0}]",
            out_cpu2[row * n], out_cpu2[row * n + 1], out_cpu2[row * n + 2], out_cpu2[row * n + 3],
            out_gpu2[row * n], out_gpu2[row * n + 1], out_gpu2[row * n + 2], out_gpu2[row * n + 3],
        );
        row += 1;
    }
    let mut max_diff2 = 0.0f32;
    for i in 0..m * n {
        max_diff2 = max_diff2.max((out_cpu2[i] - out_gpu2[i]).abs());
    }
    println!("  Max abs diff: {max_diff2:.8}");

    if mismatches == 0 && max_diff2 < 1e-2 {
        println!("  PASS\n");
    } else {
        println!("  FAIL (identity mismatches={mismatches}, sequential max_diff={max_diff2:.6})\n");
    }
}

fn test_matmul_mma_bt(m: usize, n: usize, k: usize) {
    println!("=== MatMul MMA BT: ({m}×{k} × {n}×{k}) ===");

    let cpu = CpuMatMul;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

    let mut out_cpu = vec![0.0f32; m * n];
    cpu.forward_b_transposed(&a, &b, &mut out_cpu, m, n, k).unwrap();

    let gpu_ctx = tempura_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();

    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = tempura_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let block = 32u32;
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), (m as u32).div_ceil(block), 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = tempura_kernels::gpu::to_host(&stream, &c_dev).unwrap();

    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..m * n {
        let diff = (out_cpu[i] - out_gpu[i]).abs();
        max_diff = max_diff.max(diff);
        if diff >= 1e-2 {
            if mismatches < 5 {
                eprintln!(
                    "  MISMATCH at [{i}]: cpu={:.6}, gpu={:.6}, diff={:.6}",
                    out_cpu[i], out_gpu[i], diff
                );
            }
            mismatches += 1;
        }
    }

    println!("  Max abs diff: {max_diff:.8}");
    println!("  Mismatches (>=1e-2): {mismatches}");

    if mismatches == 0 {
        println!("  PASS\n");
    } else {
        println!("  FAIL ({mismatches}/{} elements)\n", m * n);
        std::process::exit(1);
    }
}

fn test_slice_columns() {
    println!("=== Slice columns: fused QKV extraction ===");

    let gpu_ctx = tempura_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();

    // Simulate fused QKV output: [4 rows, 10 total_cols]
    // Q: cols 0..4, K: cols 4..7, V: cols 7..10
    let rows = 4usize;
    let total_cols = 10usize;
    let data: Vec<f32> = (0..rows * total_cols).map(|i| i as f32).collect();

    let src_dev = tempura_kernels::gpu::to_device(&stream, &data).unwrap();

    // Extract Q (cols 0..4)
    let q_cols = 4usize;
    let mut q_dev = tempura_kernels::gpu::alloc_device(&stream, rows * q_cols).unwrap();
    tempura_kernels::gpu::slice_columns_device(
        &ctx, &stream, &src_dev, &mut q_dev, rows, total_cols, 0, q_cols,
    )
    .unwrap();
    let q_host = tempura_kernels::gpu::to_host(&stream, &q_dev).unwrap();

    // Extract K (cols 4..7)
    let k_cols = 3usize;
    let mut k_dev = tempura_kernels::gpu::alloc_device(&stream, rows * k_cols).unwrap();
    tempura_kernels::gpu::slice_columns_device(
        &ctx, &stream, &src_dev, &mut k_dev, rows, total_cols, q_cols, k_cols,
    )
    .unwrap();
    let k_host = tempura_kernels::gpu::to_host(&stream, &k_dev).unwrap();

    // Verify Q: row r, col c -> data[r * 10 + c]
    for r in 0..rows {
        for c in 0..q_cols {
            let expected = (r * total_cols + c) as f32;
            let got = q_host[r * q_cols + c];
            assert_eq!(expected, got, "Q mismatch at [{r},{c}]");
        }
    }

    // Verify K: row r, col c -> data[r * 10 + 4 + c]
    for r in 0..rows {
        for c in 0..k_cols {
            let expected = (r * total_cols + q_cols + c) as f32;
            let got = k_host[r * k_cols + c];
            assert_eq!(expected, got, "K mismatch at [{r},{c}]");
        }
    }

    println!("  PASS");
}

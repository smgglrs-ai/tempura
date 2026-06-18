mod test_pack;

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

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  oxide-serve GPU Validation Suite (RTX 5090)        ║");
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

    let gpu_ctx = oxide_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();
    let a_dev = oxide_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = oxide_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = oxide_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = oxide_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(32), (m as u32).div_ceil(32), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = oxide_kernels::gpu::to_host(&stream, &c_dev).unwrap();

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

    let gpu_ctx = oxide_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();
    let a_dev = oxide_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = oxide_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = oxide_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = oxide_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(32), (m as u32).div_ceil(32), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = oxide_kernels::gpu::to_host(&stream, &c_dev).unwrap();

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

    let a2_dev = oxide_kernels::gpu::to_device(&stream, &a2).unwrap();
    let b2_dev = oxide_kernels::gpu::to_device(&stream, &b2).unwrap();
    let mut c2_dev = oxide_kernels::gpu::alloc_device(&stream, m * n).unwrap();
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a2_dev, &b2_dev, &mut c2_dev)
        .unwrap();
    let out_gpu2 = oxide_kernels::gpu::to_host(&stream, &c2_dev).unwrap();

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

    let gpu_ctx = oxide_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();

    let a_dev = oxide_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = oxide_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = oxide_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    let mma_module = oxide_kernels::gpu::matmul_mma_kernel::kernels::load(&ctx).unwrap();
    let block = 32u32;
    let config = cuda_core::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), (m as u32).div_ceil(block), 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    mma_module
        .sgemm_mma_32x32(&stream, config, m as u32, n as u32, k as u32, &a_dev, &b_dev, &mut c_dev)
        .unwrap();

    let out_gpu = oxide_kernels::gpu::to_host(&stream, &c_dev).unwrap();

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

    let gpu_ctx = oxide_kernels::gpu::GpuContext::new(0).unwrap();
    let ctx = gpu_ctx.context().clone();
    let stream = ctx.default_stream();

    // Simulate fused QKV output: [4 rows, 10 total_cols]
    // Q: cols 0..4, K: cols 4..7, V: cols 7..10
    let rows = 4usize;
    let total_cols = 10usize;
    let data: Vec<f32> = (0..rows * total_cols).map(|i| i as f32).collect();

    let src_dev = oxide_kernels::gpu::to_device(&stream, &data).unwrap();

    // Extract Q (cols 0..4)
    let q_cols = 4usize;
    let mut q_dev = oxide_kernels::gpu::alloc_device(&stream, rows * q_cols).unwrap();
    oxide_kernels::gpu::slice_columns_device(
        &ctx, &stream, &src_dev, &mut q_dev, rows, total_cols, 0, q_cols,
    )
    .unwrap();
    let q_host = oxide_kernels::gpu::to_host(&stream, &q_dev).unwrap();

    // Extract K (cols 4..7)
    let k_cols = 3usize;
    let mut k_dev = oxide_kernels::gpu::alloc_device(&stream, rows * k_cols).unwrap();
    oxide_kernels::gpu::slice_columns_device(
        &ctx, &stream, &src_dev, &mut k_dev, rows, total_cols, q_cols, k_cols,
    )
    .unwrap();
    let k_host = oxide_kernels::gpu::to_host(&stream, &k_dev).unwrap();

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

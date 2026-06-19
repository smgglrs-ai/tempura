use tempura_kernels::attention::{CpuAttention, CudaAttention, Attention};
use tempura_kernels::matmul::{CpuMatMul, CudaMatMul, MatMul};
use tempura_kernels::rmsnorm::{CpuRMSNorm, CudaRMSNorm, RMSNorm};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 100;

fn bench_matmul(gpu: &CudaMatMul, cpu: &CpuMatMul, m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.002).cos()).collect();

    let _ctx = gpu.gpu().context().clone();
    let stream = gpu.gpu().stream();
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a).unwrap();
    let b_dev = tempura_kernels::gpu::to_device(&stream, &b).unwrap();
    let mut c_dev = tempura_kernels::gpu::alloc_device(&stream, m * n).unwrap();

    for _ in 0..WARMUP {
        gpu.forward_bt_device(&a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();
    }
    stream.synchronize().unwrap();

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward_bt_device(&a_dev, &b_dev, &mut c_dev, m, n, k).unwrap();
    }
    stream.synchronize().unwrap();
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let mut c_cpu = vec![0.0f32; m * n];
    let cpu_start = Instant::now();
    for _ in 0..ITERS.min(10) {
        cpu.forward_b_transposed(&a, &b, &mut c_cpu, m, n, k).unwrap();
    }
    let cpu_us = cpu_start.elapsed().as_micros() as f64 / ITERS.min(10) as f64;

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gpu_gflops = flops / (gpu_us * 1e3);
    let cpu_gflops = flops / (cpu_us * 1e3);

    println!(
        "  MatMul {m:>5}×{k:>5}×{n:>5}  GPU: {:>8.1}µs ({:>7.1} GFLOPS)  CPU: {:>8.0}µs ({:>5.1} GFLOPS)  speedup: {:.1}×",
        gpu_us, gpu_gflops, cpu_us, cpu_gflops, cpu_us / gpu_us
    );
}

fn bench_rmsnorm(gpu: &CudaRMSNorm, cpu: &CpuRMSNorm, hidden_size: usize, num_tokens: usize) {
    let n = num_tokens * hidden_size;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + i as f32 * 0.0001).collect();
    let eps = 1e-6f32;

    let stream = gpu.gpu().stream();
    let x_dev = tempura_kernels::gpu::to_device(&stream, &x).unwrap();
    let w_dev = tempura_kernels::gpu::to_device(&stream, &weight).unwrap();
    let mut out_dev = tempura_kernels::gpu::alloc_device(&stream, n).unwrap();

    for _ in 0..WARMUP {
        gpu.forward_device(&x_dev, &w_dev, &mut out_dev, hidden_size, eps).unwrap();
    }
    stream.synchronize().unwrap();

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward_device(&x_dev, &w_dev, &mut out_dev, hidden_size, eps).unwrap();
    }
    stream.synchronize().unwrap();
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let mut x_cpu = x.clone();
    let cpu_start = Instant::now();
    for _ in 0..ITERS {
        x_cpu.copy_from_slice(&x);
        cpu.forward(&mut x_cpu, &weight, hidden_size, eps).unwrap();
    }
    let cpu_us = cpu_start.elapsed().as_micros() as f64 / ITERS as f64;

    let bytes = n as f64 * 4.0 * 3.0; // read x, read weight, write output
    let gpu_gbps = bytes / (gpu_us * 1e3);

    println!(
        "  RMSNorm h={hidden_size:>5} t={num_tokens:>3}  GPU: {:>8.1}µs ({:.1} GB/s)  CPU: {:>8.0}µs  speedup: {:.1}×",
        gpu_us, gpu_gbps, cpu_us, cpu_us / gpu_us
    );
}

fn bench_attention(
    gpu: &CudaAttention,
    cpu: &CpuAttention,
    label: &str,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let batch = 1;
    let q: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let k: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();
    let v: Vec<f32> = (0..batch * kv_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.03).sin())
        .collect();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let stream = gpu.gpu().stream();
    let q_dev = tempura_kernels::gpu::to_device(&stream, &q).unwrap();
    let k_dev = tempura_kernels::gpu::to_device(&stream, &k).unwrap();
    let v_dev = tempura_kernels::gpu::to_device(&stream, &v).unwrap();
    let mut out_dev =
        tempura_kernels::gpu::alloc_device(&stream, batch * seq_len * num_heads * head_dim).unwrap();

    for _ in 0..WARMUP {
        gpu.forward_device(
            &q_dev, &k_dev, &v_dev, &mut out_dev, batch, seq_len, kv_len, num_heads,
            num_kv_heads, head_dim, scale,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward_device(
            &q_dev, &k_dev, &v_dev, &mut out_dev, batch, seq_len, kv_len, num_heads,
            num_kv_heads, head_dim, scale,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let mut out_cpu = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let cpu_start = Instant::now();
    let cpu_iters = ITERS.min(10);
    for _ in 0..cpu_iters {
        cpu.forward(
            &q, &k, &v, &mut out_cpu, batch, seq_len, kv_len, num_heads, num_kv_heads, head_dim,
            scale,
        )
        .unwrap();
    }
    let cpu_us = cpu_start.elapsed().as_micros() as f64 / cpu_iters as f64;

    // FLOPs: 2*seq*kv*heads*head_dim (QK) + 2*seq*kv*heads*head_dim (score*V)
    let flops = 4.0 * batch as f64 * seq_len as f64 * kv_len as f64 * num_heads as f64
        * head_dim as f64;
    let gpu_gflops = flops / (gpu_us * 1e3);

    println!(
        "  Attn {label:<25}  GPU: {:>8.1}µs ({:>7.1} GFLOPS)  CPU: {:>8.0}µs  speedup: {:.1}×",
        gpu_us, gpu_gflops, cpu_us, cpu_us / gpu_us
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura-serve Kernel Microbenchmarks                  ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Warmup: {WARMUP} iters, Measured: {ITERS} iters\n");

    let gpu_matmul = CudaMatMul::new().unwrap();
    let cpu_matmul = CpuMatMul;
    let gpu_rmsnorm = CudaRMSNorm::new().unwrap();
    let cpu_rmsnorm = CpuRMSNorm;
    let gpu_attention = CudaAttention::new().unwrap();
    let cpu_attention = CpuAttention;

    println!("─── MatMul (B-transposed) ───");
    bench_matmul(&gpu_matmul, &cpu_matmul, 1, 4096, 4096);
    bench_matmul(&gpu_matmul, &cpu_matmul, 1, 11008, 4096);
    bench_matmul(&gpu_matmul, &cpu_matmul, 1, 32000, 4096);
    bench_matmul(&gpu_matmul, &cpu_matmul, 128, 4096, 4096);
    bench_matmul(&gpu_matmul, &cpu_matmul, 512, 4096, 4096);
    println!();

    println!("─── RMSNorm ───");
    bench_rmsnorm(&gpu_rmsnorm, &cpu_rmsnorm, 4096, 1);
    bench_rmsnorm(&gpu_rmsnorm, &cpu_rmsnorm, 4096, 8);
    bench_rmsnorm(&gpu_rmsnorm, &cpu_rmsnorm, 4096, 128);
    println!();

    println!("─── Attention (GQA, 32 heads / 8 kv heads, d=128) ───");
    bench_attention(&gpu_attention, &cpu_attention, "decode s=1 kv=128", 1, 128, 32, 8, 128);
    bench_attention(&gpu_attention, &cpu_attention, "decode s=1 kv=512", 1, 512, 32, 8, 128);
    bench_attention(&gpu_attention, &cpu_attention, "decode s=1 kv=2048", 1, 2048, 32, 8, 128);
    bench_attention(&gpu_attention, &cpu_attention, "prefill s=128 kv=128", 128, 128, 32, 8, 128);
    bench_attention(&gpu_attention, &cpu_attention, "prefill s=512 kv=512", 512, 512, 32, 8, 128);
    println!();

    println!("Done.");
}

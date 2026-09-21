use std::time::Instant;
use tempura_kernels::cutile_flash_attention::CutileFlashAttention;
use tempura_kernels::cutile_matmul::CutileMatMul;
use tempura_kernels::cutile_quantized_gemm::CutileQuantizedGemm;
use tempura_kernels::cutile_rmsnorm::CutileRmsNorm;
use tempura_kernels::matmul::CpuMatMul;
use tempura_kernels::rmsnorm::CpuRMSNorm;
use tempura_kernels::{Attention, MatMul, MatMulQ, RMSNorm, Q4K_BLOCK_BYTES, Q4K_BLOCK_SIZE};

const WARMUP: usize = 10;
const ITERS: usize = 100;

fn bench_matmul(gpu: &CutileMatMul, cpu: &CpuMatMul, m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut c = vec![0.0f32; m * n];

    for _ in 0..WARMUP {
        gpu.forward_b_transposed(&a, &b, &mut c, m, n, k).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward_b_transposed(&a, &b, &mut c, m, n, k).unwrap();
    }
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let mut c_cpu = vec![0.0f32; m * n];
    let cpu_start = Instant::now();
    for _ in 0..ITERS.min(10) {
        cpu.forward_b_transposed(&a, &b, &mut c_cpu, m, n, k)
            .unwrap();
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

fn bench_rmsnorm(gpu: &CutileRmsNorm, cpu: &CpuRMSNorm, hidden_size: usize, num_tokens: usize) {
    let n = num_tokens * hidden_size;
    let x_orig: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0 + i as f32 * 0.0001).collect();
    let eps = 1e-6f32;

    let mut x = x_orig.clone();
    for _ in 0..WARMUP {
        x.copy_from_slice(&x_orig);
        gpu.forward(&mut x, &weight, hidden_size, eps).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        x.copy_from_slice(&x_orig);
        gpu.forward(&mut x, &weight, hidden_size, eps).unwrap();
    }
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let mut x_cpu = x_orig.clone();
    let cpu_start = Instant::now();
    for _ in 0..ITERS {
        x_cpu.copy_from_slice(&x_orig);
        cpu.forward(&mut x_cpu, &weight, hidden_size, eps).unwrap();
    }
    let cpu_us = cpu_start.elapsed().as_micros() as f64 / ITERS as f64;

    let bytes = n as f64 * 4.0 * 3.0;
    let gpu_gbps = bytes / (gpu_us * 1e3);

    println!(
        "  RMSNorm h={hidden_size:>5} t={num_tokens:>3}  GPU: {:>8.1}µs ({:.1} GB/s)  CPU: {:>8.0}µs  speedup: {:.1}×",
        gpu_us, gpu_gbps, cpu_us, cpu_us / gpu_us
    );
}

fn bench_attention(
    gpu: &CutileFlashAttention,
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
    let mut out = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for _ in 0..WARMUP {
        gpu.forward(
            &q,
            &k,
            &v,
            &mut out,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward(
            &q,
            &k,
            &v,
            &mut out,
            batch,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();
    }
    let gpu_us = start.elapsed().as_micros() as f64 / ITERS as f64;

    let flops =
        4.0 * batch as f64 * seq_len as f64 * kv_len as f64 * num_heads as f64 * head_dim as f64;
    let gpu_gflops = flops / (gpu_us * 1e3);

    println!(
        "  Attn {label:<25}  GPU: {:>8.1}µs ({:>7.1} GFLOPS)",
        gpu_us, gpu_gflops
    );
}

fn make_synthetic_q4k(n: usize, k: usize) -> Vec<u8> {
    assert!(
        k.is_multiple_of(Q4K_BLOCK_SIZE),
        "k must be divisible by Q4K_BLOCK_SIZE"
    );
    let blocks_per_row = k / Q4K_BLOCK_SIZE;
    let total_bytes = n * blocks_per_row * Q4K_BLOCK_BYTES;
    // Fill with a recognizable pattern; values don't need to be exact Q4K encoding
    // since we just want timing data
    (0..total_bytes).map(|i| (i % 251) as u8).collect()
}

fn bench_q4k_gemv(gpu: &CutileQuantizedGemm, label: &str, m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let b_q4k = make_synthetic_q4k(n, k);
    let mut out = vec![0.0f32; m * n];

    for _ in 0..WARMUP {
        gpu.forward_b_transposed_q4k(&a, &b_q4k, &mut out, m, n, k)
            .unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        gpu.forward_b_transposed_q4k(&a, &b_q4k, &mut out, m, n, k)
            .unwrap();
    }
    let us = start.elapsed().as_micros() as f64 / ITERS as f64;

    // Weight bytes read per token: n * k / 2 (4-bit) but we measure end-to-end
    let weight_bytes = n as f64 * k as f64 * 0.5;
    let gbps = weight_bytes / (us * 1e3);

    println!(
        "  Q4K GEMV {label:<20}  {:>8.1}µs  ({:.1} GB/s effective weight BW)",
        us, gbps
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura cuTile Kernel Microbenchmarks                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  Warmup: {WARMUP} iters, Measured: {ITERS} iters\n");

    let gpu_matmul = CutileMatMul::new().unwrap();
    let cpu_matmul = CpuMatMul;
    let gpu_rmsnorm = CutileRmsNorm::new().unwrap();
    let cpu_rmsnorm = CpuRMSNorm;
    let gpu_attention = CutileFlashAttention::new().unwrap();
    let gpu_q4k = CutileQuantizedGemm::new().unwrap();

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

    println!("─── Flash Attention (GQA, 32 heads / 8 kv heads, d=128) ───");
    bench_attention(&gpu_attention, "decode s=1 kv=128", 1, 128, 32, 8, 128);
    bench_attention(&gpu_attention, "decode s=1 kv=512", 1, 512, 32, 8, 128);
    bench_attention(&gpu_attention, "decode s=1 kv=2048", 1, 2048, 32, 8, 128);
    bench_attention(&gpu_attention, "prefill s=128 kv=128", 128, 128, 32, 8, 128);
    bench_attention(&gpu_attention, "prefill s=512 kv=512", 512, 512, 32, 8, 128);
    println!();

    println!("─── Q4K Quantized GEMV (dequant-then-GEMM path) ───");
    bench_q4k_gemv(&gpu_q4k, "h=4096 n=4096", 1, 4096, 4096);
    bench_q4k_gemv(&gpu_q4k, "h=4096 n=11008", 1, 11008, 4096);
    bench_q4k_gemv(&gpu_q4k, "h=4096 n=32000", 1, 32000, 4096);
    println!();

    println!("Done.");
}

use std::time::Instant;
use tempura_kernels::attention::{
    CpuAttention, CpuAttentionF16, CpuFlashAttention, CpuFlashAttentionF16, CudaAttention,
};
use tempura_kernels::{Attention, AttentionMode};

struct Config {
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn make_data(cfg: &Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.3)
        .collect();
    let k: Vec<f32> = (0..cfg.batch * cfg.kv_len * cfg.num_kv_heads * cfg.head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
        .collect();
    let v: Vec<f32> = (0..cfg.batch * cfg.kv_len * cfg.num_kv_heads * cfg.head_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();
    (q, k, v)
}

fn run_cpu(
    attn: &dyn Attention,
    cfg: &Config,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f32,
    warmup: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    let n = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
    let mut out = vec![0.0f32; n];

    for _ in 0..warmup {
        attn.forward(
            q, k, v, &mut out, cfg.batch, cfg.seq_len, cfg.kv_len,
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, scale,
        )
        .unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        attn.forward(
            q, k, v, &mut out, cfg.batch, cfg.seq_len, cfg.kv_len,
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, scale,
        )
        .unwrap();
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (out, us)
}

fn run_gpu(
    attn: &CudaAttention,
    cfg: &Config,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f32,
    mode: AttentionMode,
    warmup: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    let n = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
    let ctx = attn.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = attn.gpu().stream();

    let q_dev = cuda_core::DeviceBuffer::from_host(&stream, q).unwrap();
    let k_dev = cuda_core::DeviceBuffer::from_host(&stream, k).unwrap();
    let v_dev = cuda_core::DeviceBuffer::from_host(&stream, v).unwrap();
    let mut out_dev = cuda_core::DeviceBuffer::zeroed(&stream, n).unwrap();

    for _ in 0..warmup {
        attn.forward_device_with_mode(
            &q_dev, &k_dev, &v_dev, &mut out_dev, cfg.batch, cfg.seq_len, cfg.kv_len,
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, scale, mode,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        attn.forward_device_with_mode(
            &q_dev, &k_dev, &v_dev, &mut out_dev, cfg.batch, cfg.seq_len, cfg.kv_len,
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, scale, mode,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let out = out_dev.to_host_vec(&stream).unwrap();
    (out[..n].to_vec(), us)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() {
    let configs = vec![
        ("decode kv=64", Config { batch: 1, seq_len: 1, kv_len: 64, num_heads: 32, num_kv_heads: 8, head_dim: 128 }),
        ("decode kv=256", Config { batch: 1, seq_len: 1, kv_len: 256, num_heads: 32, num_kv_heads: 8, head_dim: 128 }),
        ("decode kv=1024", Config { batch: 1, seq_len: 1, kv_len: 1024, num_heads: 32, num_kv_heads: 8, head_dim: 128 }),
        ("prefill s=16", Config { batch: 1, seq_len: 16, kv_len: 16, num_heads: 32, num_kv_heads: 8, head_dim: 128 }),
    ];

    let gpu_attn = CudaAttention::new().expect("Failed to init CudaAttention");
    let warmup = 3;
    let iters = 20;

    for (label, cfg) in &configs {
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let (q, k, v) = make_data(cfg);

        println!("=== {label} (s={}, kv={}, h={}/{}, d={}) ===",
                 cfg.seq_len, cfg.kv_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        println!("{:<24} {:>12} {:>10}", "", "max_diff", "time(μs)");

        let (ref_out, ref_us) = run_cpu(&CpuAttention, cfg, &q, &k, &v, scale, warmup, iters);
        println!("{:<24} {:>12} {:>10.1}", "CPU  f32  naive", "ref", ref_us);

        let cases_cpu: Vec<(&str, Box<dyn Attention>)> = vec![
            ("CPU  f16  naive", Box::new(CpuAttentionF16)),
            ("CPU  f32  flash", Box::new(CpuFlashAttention)),
            ("CPU  f16  flash", Box::new(CpuFlashAttentionF16)),
        ];
        for (name, attn) in &cases_cpu {
            let (out, us) = run_cpu(attn.as_ref(), cfg, &q, &k, &v, scale, warmup, iters);
            let d = max_diff(&ref_out, &out);
            println!("{:<24} {:>12.6} {:>10.1}", name, d, us);
        }

        let gpu_modes = vec![
            ("GPU  f32  naive", AttentionMode::NaiveF32),
            ("GPU  f16  naive", AttentionMode::NaiveF16),
            ("GPU  f32  flash", AttentionMode::FlashF32),
            ("GPU  f16  flash", AttentionMode::FlashF16),
        ];
        for (name, mode) in &gpu_modes {
            let (out, us) = run_gpu(&gpu_attn, cfg, &q, &k, &v, scale, *mode, warmup, iters);
            let d = max_diff(&ref_out, &out);
            println!("{:<24} {:>12.6} {:>10.1}", name, d, us);
        }
        println!();
    }

    // RoPE CPU vs GPU precision test
    {
        use tempura_kernels::rope::{CpuRoPE, CudaRoPE};
        use tempura_kernels::RoPE;

        let ctx = gpu_attn.gpu().context().clone();
        let cpu_rope = CpuRoPE;
        let gpu_rope = CudaRoPE::with_context(ctx).unwrap();

        for (label, hd, theta) in [("Qwen3", 128usize, 1000000.0f32), ("small", 16, 10000.0)] {
            let seq_len = 4;
            let nh = 32;
            let n = seq_len * nh * hd;
            let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
            let pos: Vec<u32> = (0..seq_len as u32).collect();

            let mut xc = x.clone();
            cpu_rope.forward(&mut xc, 1, seq_len, nh, hd, &pos, theta).unwrap();

            let mut xg = x.clone();
            gpu_rope.forward(&mut xg, 1, seq_len, nh, hd, &pos, theta).unwrap();

            let md: f32 = xc.iter().zip(&xg).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
            let nd = xc.iter().zip(&xg).filter(|(a,b)| a != b).count();
            println!("RoPE {label} (hd={hd}, theta={theta:.0}): max_diff={md:.10}, diff_elems={nd}/{n}");
        }
    }
}

fn test_rope_precision() {
    use tempura_kernels::rope::{CpuRoPE, CudaRoPE};
    use tempura_kernels::RoPE;

    let head_dim = 128;
    let seq_len = 4;
    let num_heads = 32;
    let theta = 1000000.0f32;
    let n = seq_len * num_heads * head_dim;

    let x_orig: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    let cpu = CpuRoPE;
    let gpu = CudaRoPE::new().unwrap();

    let mut x_cpu = x_orig.clone();
    cpu.forward(&mut x_cpu, 1, seq_len, num_heads, head_dim, &positions, theta).unwrap();

    let mut x_gpu = x_orig.clone();
    gpu.forward(&mut x_gpu, 1, seq_len, num_heads, head_dim, &positions, theta).unwrap();

    let max_diff: f32 = x_cpu.iter().zip(&x_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let num_diff: usize = x_cpu.iter().zip(&x_gpu).filter(|(a, b)| a != b).count();
    println!("\n=== RoPE precision (hd=128, theta=1e6, seq=4, heads=32) ===");
    println!("  max_diff: {max_diff:.10}");
    println!("  differing elements: {num_diff} / {n}");

    // Also test with small head_dim for comparison
    let head_dim_s = 16;
    let n_s = seq_len * num_heads * head_dim_s;
    let x_s: Vec<f32> = (0..n_s).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
    let mut x_cpu_s = x_s.clone();
    let mut x_gpu_s = x_s.clone();
    cpu.forward(&mut x_cpu_s, 1, seq_len, num_heads, head_dim_s, &positions, 10000.0).unwrap();
    gpu.forward(&mut x_gpu_s, 1, seq_len, num_heads, head_dim_s, &positions, 10000.0).unwrap();
    let max_diff_s: f32 = x_cpu_s.iter().zip(&x_gpu_s).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let num_diff_s: usize = x_cpu_s.iter().zip(&x_gpu_s).filter(|(a, b)| a != b).count();
    println!("\n=== RoPE precision (hd=16, theta=10k, seq=4, heads=32) ===");
    println!("  max_diff: {max_diff_s:.10}");
    println!("  differing elements: {num_diff_s} / {n_s}");
}

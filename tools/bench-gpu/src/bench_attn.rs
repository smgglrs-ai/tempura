use std::time::Instant;
use tempura_kernels::attention::{CpuAttention, CpuFlashAttention};
use tempura_kernels::cutile_flash_attention::CutileFlashAttention;
use tempura_kernels::Attention;

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

fn run_attn(
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
            q,
            k,
            v,
            &mut out,
            cfg.batch,
            cfg.seq_len,
            cfg.kv_len,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            scale,
        )
        .unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        attn.forward(
            q,
            k,
            v,
            &mut out,
            cfg.batch,
            cfg.seq_len,
            cfg.kv_len,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            scale,
        )
        .unwrap();
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (out, us)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let configs = vec![
        (
            "decode kv=64",
            Config {
                batch: 1,
                seq_len: 1,
                kv_len: 64,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
        (
            "decode kv=256",
            Config {
                batch: 1,
                seq_len: 1,
                kv_len: 256,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
        (
            "decode kv=1024",
            Config {
                batch: 1,
                seq_len: 1,
                kv_len: 1024,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
        (
            "prefill s=16",
            Config {
                batch: 1,
                seq_len: 16,
                kv_len: 16,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
        (
            "prefill s=64",
            Config {
                batch: 1,
                seq_len: 64,
                kv_len: 64,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
        (
            "prefill s=128",
            Config {
                batch: 1,
                seq_len: 128,
                kv_len: 128,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
            },
        ),
    ];

    let cutile_attn = CutileFlashAttention::new().expect("Failed to init CutileFlashAttention");
    let warmup = 3;
    let iters = 20;

    for (label, cfg) in &configs {
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let (q, k, v) = make_data(cfg);

        println!(
            "=== {label} (s={}, kv={}, h={}/{}, d={}) ===",
            cfg.seq_len, cfg.kv_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        );
        println!("{:<28} {:>12} {:>10}", "", "max_diff", "time(µs)");

        let (ref_out, ref_us) = run_attn(&CpuAttention, cfg, &q, &k, &v, scale, warmup, iters);
        println!("{:<28} {:>12} {:>10.1}", "CPU  f32  naive", "ref", ref_us);

        let (flash_out, flash_us) =
            run_attn(&CpuFlashAttention, cfg, &q, &k, &v, scale, warmup, iters);
        let d = max_diff(&ref_out, &flash_out);
        println!("{:<28} {:>12.6} {:>10.1}", "CPU  f32  flash", d, flash_us);

        let (cutile_out, cutile_us) = run_attn(&cutile_attn, cfg, &q, &k, &v, scale, warmup, iters);
        let d = max_diff(&ref_out, &cutile_out);
        let speedup = ref_us / cutile_us;
        println!(
            "{:<28} {:>12.6} {:>10.1}  ({:.1}× vs CPU naive)",
            "GPU  cutile flash", d, cutile_us, speedup
        );
        println!();
    }

    // RoPE precision spot check
    {
        use tempura_kernels::cutile_rope::CutileRoPE;
        use tempura_kernels::rope::CpuRoPE;
        use tempura_kernels::RoPE;

        let cpu_rope = CpuRoPE;
        let gpu_rope = CutileRoPE::new().unwrap();

        println!("─── RoPE precision (cuTile vs CPU) ───");
        for (label, hd, theta) in [("Qwen3", 128usize, 1_000_000.0f32), ("small", 16, 10_000.0)] {
            let seq_len = 4;
            let nh = 32;
            let n = seq_len * nh * hd;
            let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
            let pos: Vec<u32> = (0..seq_len as u32).collect();

            let mut x_cpu = x.clone();
            cpu_rope
                .forward(&mut x_cpu, 1, seq_len, nh, hd, &pos, theta)
                .unwrap();

            let mut x_gpu = x.clone();
            gpu_rope
                .forward(&mut x_gpu, 1, seq_len, nh, hd, &pos, theta)
                .unwrap();

            let md: f32 = x_cpu
                .iter()
                .zip(&x_gpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let nd = x_cpu.iter().zip(&x_gpu).filter(|(a, b)| a != b).count();
            println!(
                "  RoPE {label} (hd={hd}, theta={theta:.0}): max_diff={md:.10}  diff_elems={nd}/{n}"
            );
        }
    }

    println!("\nDone.");
}

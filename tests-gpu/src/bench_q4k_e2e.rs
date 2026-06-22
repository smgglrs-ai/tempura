use std::path::Path;
use std::time::Instant;

use tempura_engine::weights::GpuWeights;
use tempura_kernels::quantized::CudaMatMulQ4K;

fn main() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Q4K GPU Forward Pass Benchmark (real GGUF weights)   ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let gguf_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let default = format!(
                "{}/.local/share/navra/models/qwen3-8b.gguf",
                std::env::var("HOME").unwrap()
            );
            eprintln!("No GGUF path given, using default: {default}");
            default
        });

    if !Path::new(&gguf_path).exists() {
        eprintln!("GGUF file not found: {gguf_path}");
        std::process::exit(1);
    }

    println!("Loading GGUF: {gguf_path}");
    let t = Instant::now();
    let gguf = tempura_engine::gguf_loader::GgufLoader::from_file_quantized(Path::new(&gguf_path))
        .expect("Failed to load GGUF");
    let config = gguf.config.clone();
    let qweights = gguf.to_quantized_weights();
    drop(gguf);
    println!("  Loaded in {:.1}s", t.elapsed().as_secs_f64());

    println!(
        "  Model: {} layers, h={}, heads={}/{}, d={}, inter={}, vocab={}",
        config.num_layers, config.hidden_size, config.num_attention_heads,
        config.num_kv_heads, config.head_dim, config.intermediate_size, config.vocab_size,
    );

    // Upload to GPU
    println!("\nUploading weights to GPU...");
    let t = Instant::now();
    let q4k = CudaMatMulQ4K::new().expect("CudaMatMulQ4K init");
    let ctx = q4k.gpu().context().clone();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    let gpu_weights = GpuWeights::upload_from_quantized(&qweights, &ctx)
        .expect("GPU weight upload");
    println!("  Uploaded in {:.1}s", t.elapsed().as_secs_f64());

    // Count weight memory on GPU
    let mut total_q_bytes = 0usize;
    let mut total_f32_bytes = 0usize;
    let mut n_q4k = 0;
    let mut n_q6k = 0;
    let mut n_f32 = 0;

    let h = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let n_kv = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;

    // Check what we have
    for layer in 0..config.num_layers {
        let prefix = format!("model.layers.{layer}");
        for proj in &["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
            let name = format!("{prefix}.{proj}");
            if gpu_weights.is_quantized(&name) {
                n_q4k += 1; // might be q6k but we count together
            } else if gpu_weights.get(&name).is_some() {
                n_f32 += 1;
            }
        }
    }
    println!("  Quantized tensors: {n_q4k}, F32 tensors: {n_f32}");

    // Simulate decode forward pass: matmuls only (m=1)
    // Per layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    println!("\n─── Decode Matmul Pass (m=1, all layers) ───");

    macro_rules! run_mm {
        ($name:expr, $a:expr, $out:expr, $n:expr, $k:expr) => {
            if let Some(w) = gpu_weights.get_quantized_device($name) {
                q4k.gemv_q4k_coop_device($a, w, $out, $n, $k).unwrap();
            }
        };
    }

    // Activation buffer (f32, on GPU)
    let a_dummy: Vec<f32> = (0..h).map(|i| ((i % 97) as f32 - 48.0) * 0.001).collect();
    let a_dev = tempura_kernels::gpu::to_device(&stream, &a_dummy).unwrap();
    let a_inter: Vec<f32> = vec![0.0f32; inter];
    let a_inter_dev = tempura_kernels::gpu::to_device(&stream, &a_inter).unwrap();

    let mut out_q = tempura_kernels::gpu::alloc_device(&stream, q_dim).unwrap();
    let mut out_kv = tempura_kernels::gpu::alloc_device(&stream, kv_dim).unwrap();
    let mut out_h = tempura_kernels::gpu::alloc_device(&stream, h).unwrap();
    let mut out_inter = tempura_kernels::gpu::alloc_device(&stream, inter).unwrap();

    // Warmup
    for layer in 0..2.min(config.num_layers) {
        let prefix = format!("model.layers.{layer}");
        run_mm!(&format!("{prefix}.self_attn.q_proj.weight"), &a_dev, &mut out_q, q_dim, h);
        run_mm!(&format!("{prefix}.mlp.gate_proj.weight"), &a_dev, &mut out_inter, inter, h);
        run_mm!(&format!("{prefix}.mlp.down_proj.weight"), &a_inter_dev, &mut out_h, h, inter);
    }
    stream.synchronize().unwrap();

    // Timed run
    let iters = 10;
    let mut lm_out = tempura_kernels::gpu::alloc_device(&stream, config.vocab_size).unwrap();
    let mut times = Vec::new();
    for _ in 0..iters {
        let start = Instant::now();
        for layer in 0..config.num_layers {
            let prefix = format!("model.layers.{layer}");
            run_mm!(&format!("{prefix}.self_attn.q_proj.weight"), &a_dev, &mut out_q, q_dim, h);
            run_mm!(&format!("{prefix}.self_attn.k_proj.weight"), &a_dev, &mut out_kv, kv_dim, h);
            run_mm!(&format!("{prefix}.self_attn.v_proj.weight"), &a_dev, &mut out_kv, kv_dim, h);
            run_mm!(&format!("{prefix}.self_attn.o_proj.weight"), &a_dev, &mut out_h, h, h);
            run_mm!(&format!("{prefix}.mlp.gate_proj.weight"), &a_dev, &mut out_inter, inter, h);
            run_mm!(&format!("{prefix}.mlp.up_proj.weight"), &a_dev, &mut out_inter, inter, h);
            run_mm!(&format!("{prefix}.mlp.down_proj.weight"), &a_inter_dev, &mut out_h, h, inter);
        }
        run_mm!("lm_head.weight", &a_dev, &mut lm_out, config.vocab_size, h);
        stream.synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iters / 2];
    let tok_s = 1000.0 / median;

    println!("  Matmul-only forward: {median:.2}ms");
    println!("  Projected tok/s:     {tok_s:.0}");
    println!("  llama.cpp reference: 209 tok/s (4.78ms full forward)");
    println!("  Ratio: {:.2}x", tok_s / 209.0);
    println!();

    // Per-layer breakdown
    println!("─── Per-Layer Breakdown (layer 0) ───");
    let prefix = "model.layers.0";
    let projs: Vec<(&str, usize, usize)> = vec![
        ("self_attn.q_proj.weight", q_dim, h),
        ("self_attn.k_proj.weight", kv_dim, h),
        ("self_attn.v_proj.weight", kv_dim, h),
        ("self_attn.o_proj.weight", h, h),
        ("mlp.gate_proj.weight", inter, h),
        ("mlp.up_proj.weight", inter, h),
        ("mlp.down_proj.weight", h, inter),
    ];
    for (proj, n_out, k_size) in &projs {
        let name = format!("{prefix}.{proj}");
        if let Some(w) = gpu_weights.get_quantized_device(&name) {
            let a_buf = if *k_size == inter { &a_inter_dev } else { &a_dev };
            let mut out_buf = tempura_kernels::gpu::alloc_device(&stream, *n_out).unwrap();
            let mut t_arr = Vec::new();
            for _ in 0..20 {
                let t = Instant::now();
                q4k.gemv_q4k_coop_device(a_buf, w, &mut out_buf, *n_out, *k_size).unwrap();
                stream.synchronize().unwrap();
                t_arr.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            t_arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("  {proj:>35}: {:.3}ms (n={n_out}, k={k_size})", t_arr[10]);
        }
    }

    println!("\nDone.");
}

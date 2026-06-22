use std::path::Path;
use std::time::Instant;

use tempura_engine::weights::GpuWeights;
use tempura_kernels::dispatch::{CachedDispatch, GpuArch, GpuKernelDispatcher};
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

    // --- Multi-warp kernel comparison ---
    println!("─── Multi-Warp Kernel (8 warps/col) ───");
    let mut mr_times = Vec::new();
    let mut lm_out_mr = tempura_kernels::gpu::alloc_device(&stream, config.vocab_size).unwrap();
    // Warmup
    for layer in 0..2.min(config.num_layers) {
        let prefix = format!("model.layers.{layer}");
        let name = format!("{prefix}.self_attn.q_proj.weight");
        if let Some(w) = gpu_weights.get_quantized_device(&name) {
            q4k.gemv_q4k_multirow_device(&a_dev, w, &mut out_q, q_dim, h, 8).unwrap();
        }
    }
    stream.synchronize().unwrap();

    for _ in 0..iters {
        let start = Instant::now();
        for layer in 0..config.num_layers {
            let prefix = format!("model.layers.{layer}");
            macro_rules! run_mr {
                ($proj:expr, $a:expr, $out:expr, $n:expr, $k:expr, $nw:expr) => {
                    let name = format!("{}.{}", prefix, $proj);
                    if let Some(w) = gpu_weights.get_quantized_device(&name) {
                        q4k.gemv_q4k_multirow_device($a, w, $out, $n, $k, $nw).unwrap();
                    }
                };
            }
            run_mr!("self_attn.q_proj.weight", &a_dev, &mut out_q, q_dim, h, 8);
            run_mr!("self_attn.k_proj.weight", &a_dev, &mut out_kv, kv_dim, h, 4);
            run_mr!("self_attn.v_proj.weight", &a_dev, &mut out_kv, kv_dim, h, 4);
            run_mr!("self_attn.o_proj.weight", &a_dev, &mut out_h, h, h, 8);
            run_mr!("mlp.gate_proj.weight", &a_dev, &mut out_inter, inter, h, 4);
            run_mr!("mlp.up_proj.weight", &a_dev, &mut out_inter, inter, h, 4);
            run_mr!("mlp.down_proj.weight", &a_inter_dev, &mut out_h, h, inter, 8);
        }
        if let Some(w) = gpu_weights.get_quantized_device("lm_head.weight") {
            q4k.gemv_q4k_multirow_device(&a_dev, w, &mut lm_out_mr, config.vocab_size, h, 4).unwrap();
        }
        stream.synchronize().unwrap();
        mr_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    mr_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mr_median = mr_times[iters / 2];
    let mr_tps = 1000.0 / mr_median;
    let speedup = median / mr_median;

    println!("  Multi-warp forward:  {mr_median:.2}ms → {mr_tps:.0} tok/s");
    println!("  vs coop (1 warp):    {median:.2}ms → {tok_s:.0} tok/s");
    println!("  Speedup: {speedup:.2}x");
    println!("  llama.cpp reference: 209 tok/s");
    println!("  Ratio: {:.2}x", mr_tps / 209.0);

    // --- Cached dispatcher (zero-overhead dispatch) ---
    println!("─── Cached Dispatcher ───");
    let arch = GpuArch::detect(q4k.gpu());
    println!("  GPU arch: {:?}", arch);
    let cached = CachedDispatch::from_model_dims(
        arch,
        config.num_layers, h, n_heads, n_kv, head_dim, inter, config.vocab_size,
    ).expect("cached dispatch init");

    let mut cd_times = Vec::new();
    let mut lm_out_cd = tempura_kernels::gpu::alloc_device(&stream, config.vocab_size).unwrap();
    // Warmup
    for layer in 0..2.min(config.num_layers) {
        let prefix = format!("model.layers.{layer}");
        let name = format!("{prefix}.self_attn.q_proj.weight");
        if let Some(w) = gpu_weights.get_quantized_device(&name) {
            cached.gemv(&name, &a_dev, w, &mut out_q, q_dim, h).unwrap();
        }
    }
    stream.synchronize().unwrap();

    // Pre-resolve weight DeviceBuffer refs + indices (done once, not per iteration)
    // Pre-resolve weight refs (done once, zero allocation in hot path)
    struct LayerW<'a> {
        q: Option<&'a cuda_core::DeviceBuffer<u8>>,
        k: Option<&'a cuda_core::DeviceBuffer<u8>>,
        v: Option<&'a cuda_core::DeviceBuffer<u8>>,
        o: Option<&'a cuda_core::DeviceBuffer<u8>>,
        gate: Option<&'a cuda_core::DeviceBuffer<u8>>,
        up: Option<&'a cuda_core::DeviceBuffer<u8>>,
        down: Option<&'a cuda_core::DeviceBuffer<u8>>,
    }
    let layer_weights: Vec<LayerW> = (0..config.num_layers).map(|l| {
        let p = format!("model.layers.{l}");
        LayerW {
            q: gpu_weights.get_quantized_device(&format!("{p}.self_attn.q_proj.weight")),
            k: gpu_weights.get_quantized_device(&format!("{p}.self_attn.k_proj.weight")),
            v: gpu_weights.get_quantized_device(&format!("{p}.self_attn.v_proj.weight")),
            o: gpu_weights.get_quantized_device(&format!("{p}.self_attn.o_proj.weight")),
            gate: gpu_weights.get_quantized_device(&format!("{p}.mlp.gate_proj.weight")),
            up: gpu_weights.get_quantized_device(&format!("{p}.mlp.up_proj.weight")),
            down: gpu_weights.get_quantized_device(&format!("{p}.mlp.down_proj.weight")),
        }
    }).collect();
    let lm_w = gpu_weights.get_quantized_device("lm_head.weight");

    macro_rules! cd {
        ($cached:expr, $idx:expr, $a:expr, $w:expr, $out:expr, $n:expr, $k:expr) => {
            if let Some(w) = $w { $cached.gemv_idx($idx, $a, w, $out, $n, $k).unwrap(); }
        };
    }

    for _ in 0..iters {
        let start = Instant::now();
        for layer in 0..config.num_layers {
            let base = cached.layer_base_idx(layer);
            let lw = &layer_weights[layer];
            cd!(cached, base, &a_dev, lw.q, &mut out_q, q_dim, h);
            cd!(cached, base+1, &a_dev, lw.k, &mut out_kv, kv_dim, h);
            cd!(cached, base+2, &a_dev, lw.v, &mut out_kv, kv_dim, h);
            cd!(cached, base+3, &a_dev, lw.o, &mut out_h, h, h);
            cd!(cached, base+4, &a_dev, lw.gate, &mut out_inter, inter, h);
            cd!(cached, base+5, &a_dev, lw.up, &mut out_inter, inter, h);
            cd!(cached, base+6, &a_inter_dev, lw.down, &mut out_h, h, inter);
        }
        if let Some(w) = lm_w {
            cached.gemv_idx(cached.lm_head_idx(), &a_dev, w, &mut lm_out_cd, config.vocab_size, h).unwrap();
        }
        stream.synchronize().unwrap();
        cd_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    cd_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cd_median = cd_times[iters / 2];
    let cd_tps = 1000.0 / cd_median;

    println!("  Cached dispatch:     {cd_median:.2}ms → {cd_tps:.0} tok/s");
    println!("  llama.cpp reference: 209 tok/s");
    println!("  Ratio: {:.2}x", cd_tps / 209.0);

    // --- Multi-row fast kernel sweep ---
    println!("─── Fast Kernel (multi-row × multi-warp) ───");
    for (rpc, nwk) in [(1, 8), (2, 4), (4, 2), (2, 8), (4, 4)] {
        let label = format!("rows={rpc} warps_k={nwk}");
        let mut ft = Vec::new();
        // Warmup
        for layer in 0..2.min(config.num_layers) {
            let p = format!("model.layers.{layer}");
            if let Some(w) = gpu_weights.get_quantized_device(&format!("{p}.self_attn.q_proj.weight")) {
                let _ = q4k.gemv_q4k_fast_device(&a_dev, w, &mut out_q, q_dim, h, nwk, rpc);
            }
        }
        stream.synchronize().unwrap();

        for _ in 0..iters {
            let start = Instant::now();
            for layer in 0..config.num_layers {
                let p = format!("model.layers.{layer}");
                macro_rules! fast {
                    ($proj:expr, $a:expr, $out:expr, $no:expr, $ki:expr) => {
                        if let Some(w) = gpu_weights.get_quantized_device(&format!("{}.{}", p, $proj)) {
                            q4k.gemv_q4k_fast_device($a, w, $out, $no, $ki, nwk, rpc).unwrap();
                        }
                    };
                }
                fast!("self_attn.q_proj.weight", &a_dev, &mut out_q, q_dim, h);
                fast!("self_attn.k_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
                fast!("self_attn.v_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
                fast!("self_attn.o_proj.weight", &a_dev, &mut out_h, h, h);
                fast!("mlp.gate_proj.weight", &a_dev, &mut out_inter, inter, h);
                fast!("mlp.up_proj.weight", &a_dev, &mut out_inter, inter, h);
                fast!("mlp.down_proj.weight", &a_inter_dev, &mut out_h, h, inter);
            }
            if let Some(w) = gpu_weights.get_quantized_device("lm_head.weight") {
                q4k.gemv_q4k_fast_device(&a_dev, w, &mut lm_out, config.vocab_size, h, nwk, rpc).unwrap();
            }
            stream.synchronize().unwrap();
            ft.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        ft.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ft[iters / 2];
        let tps = 1000.0 / med;
        println!("  {label:>20}: {med:.2}ms → {tps:.0} tok/s ({:.2}x llama.cpp)", tps / 209.0);
    }

    // --- Integer-accumulation kernel ---
    println!("─── IntAcc Kernel (batched scale application) ───");
    let mut ia_times = Vec::new();
    // Warmup
    for layer in 0..2.min(config.num_layers) {
        let p = format!("model.layers.{layer}");
        if let Some(w) = gpu_weights.get_quantized_device(&format!("{p}.self_attn.q_proj.weight")) {
            q4k.gemv_q4k_intacc_device(&a_dev, w, &mut out_q, q_dim, h, 8).unwrap();
        }
    }
    stream.synchronize().unwrap();
    for _ in 0..iters {
        let start = Instant::now();
        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");
            macro_rules! ia {
                ($proj:expr, $a:expr, $out:expr, $no:expr, $ki:expr) => {
                    if let Some(w) = gpu_weights.get_quantized_device(&format!("{}.{}", p, $proj)) {
                        q4k.gemv_q4k_intacc_device($a, w, $out, $no, $ki, 8).unwrap();
                    }
                };
            }
            ia!("self_attn.q_proj.weight", &a_dev, &mut out_q, q_dim, h);
            ia!("self_attn.k_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
            ia!("self_attn.v_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
            ia!("self_attn.o_proj.weight", &a_dev, &mut out_h, h, h);
            ia!("mlp.gate_proj.weight", &a_dev, &mut out_inter, inter, h);
            ia!("mlp.up_proj.weight", &a_dev, &mut out_inter, inter, h);
            ia!("mlp.down_proj.weight", &a_inter_dev, &mut out_h, h, inter);
        }
        if let Some(w) = gpu_weights.get_quantized_device("lm_head.weight") {
            q4k.gemv_q4k_intacc_device(&a_dev, w, &mut lm_out, config.vocab_size, h, 8).unwrap();
        }
        stream.synchronize().unwrap();
        ia_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    ia_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ia_med = ia_times[iters / 2];
    let ia_tps = 1000.0 / ia_med;
    println!("  IntAcc (8 warps):    {ia_med:.2}ms → {ia_tps:.0} tok/s ({:.2}x llama.cpp)", ia_tps / 209.0);

    // --- PTX inner loop kernel ---
    println!("─── PTX Kernel (bfe.u32 + batched scale) ───");
    let mut ptx_times = Vec::new();
    for layer in 0..2.min(config.num_layers) {
        let p = format!("model.layers.{layer}");
        if let Some(w) = gpu_weights.get_quantized_device(&format!("{p}.self_attn.q_proj.weight")) {
            q4k.gemv_q4k_ptx_device(&a_dev, w, &mut out_q, q_dim, h, 8).unwrap();
        }
    }
    stream.synchronize().unwrap();
    for _ in 0..iters {
        let start = Instant::now();
        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");
            macro_rules! ptx {
                ($proj:expr, $a:expr, $out:expr, $no:expr, $ki:expr) => {
                    if let Some(w) = gpu_weights.get_quantized_device(&format!("{}.{}", p, $proj)) {
                        q4k.gemv_q4k_ptx_device($a, w, $out, $no, $ki, 8).unwrap();
                    }
                };
            }
            ptx!("self_attn.q_proj.weight", &a_dev, &mut out_q, q_dim, h);
            ptx!("self_attn.k_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
            ptx!("self_attn.v_proj.weight", &a_dev, &mut out_kv, kv_dim, h);
            ptx!("self_attn.o_proj.weight", &a_dev, &mut out_h, h, h);
            ptx!("mlp.gate_proj.weight", &a_dev, &mut out_inter, inter, h);
            ptx!("mlp.up_proj.weight", &a_dev, &mut out_inter, inter, h);
            ptx!("mlp.down_proj.weight", &a_inter_dev, &mut out_h, h, inter);
        }
        if let Some(w) = gpu_weights.get_quantized_device("lm_head.weight") {
            q4k.gemv_q4k_ptx_device(&a_dev, w, &mut lm_out, config.vocab_size, h, 8).unwrap();
        }
        stream.synchronize().unwrap();
        ptx_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    ptx_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ptx_med = ptx_times[iters / 2];
    let ptx_tps = 1000.0 / ptx_med;
    println!("  PTX (8 warps):       {ptx_med:.2}ms → {ptx_tps:.0} tok/s ({:.2}x llama.cpp)", ptx_tps / 209.0);

    println!("\nDone.");
}

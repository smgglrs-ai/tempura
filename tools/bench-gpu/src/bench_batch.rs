/// bench-batch — measure batched continuous decode throughput
///
/// Simulates B concurrent decode requests sharing a single GPU forward pass.
/// At B=5: 5×32=160 flash-attention blocks on RTX 5090 (170 SMs) → full SM utilization.
///
/// Usage:
///   RUSTUP_TOOLCHAIN=nightly-2026-10-11 cargo run -p tempura-bench-gpu --release \
///     --bin bench-batch -- /path/to/model.gguf

use std::path::Path;
use std::time::Instant;

use tempura_engine::cutile_kv_cache::CutileKvCache;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;

const WARMUP_STEPS: usize = 10;
const BENCH_STEPS:  usize = 50;

fn bench_batch(wrapper: &CutileInferenceWrapper, batch: usize) -> f64 {
    let cfg = &wrapper.model.config;

    // Allocate one KV cache per request using the model's GPU stream.
    // We use forward_batch_decode_stateful which accepts external KV caches.
    // For now, fresh per-call caches suffice for throughput measurement.
    let mut cur_pos = 0usize;

    // First call — prefill at position 0.
    {
        let toks: Vec<u32> = vec![1u32; batch];
        let pos:  Vec<u32> = vec![cur_pos as u32; batch];
        let _    = wrapper.forward_batch_decode_fresh(&toks, &pos);
        cur_pos += 1;
    }

    // Warmup.
    for _ in 0..WARMUP_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos as u32; batch];
        let _    = wrapper.forward_batch_decode_fresh(&toks, &pos);
        cur_pos += 1;
    }

    // Timed.
    let t0 = Instant::now();
    for _ in 0..BENCH_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos as u32; batch];
        let _    = wrapper.forward_batch_decode_fresh(&toks, &pos);
        cur_pos += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (batch * BENCH_STEPS) as f64 / elapsed
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            "/home/fdupont/.cache/tempura/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string()
        });

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura Batched Decode Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════╝\n");
    println!("  Model: {model_path}");
    println!("  Warmup: {WARMUP_STEPS} steps, Bench: {BENCH_STEPS} steps per batch size\n");

    println!("Loading model…");
    let t0 = Instant::now();
    let loader = GgufLoader::from_file_quantized(Path::new(&model_path))
        .expect("failed to open GGUF");
    let config = loader.config.clone();
    let qw = loader.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config);
    wrapper.load_quantized(qw);
    println!("  Loaded in {:.2}s\n", t0.elapsed().as_secs_f64());

    let n_heads = wrapper.model.config.num_attention_heads;

    println!("  {:>6}  {:>14}  {:>16}  {:>14}", "Batch", "total tok/s", "per-req tok/s", "attn blocks");
    println!("  {}", "-".repeat(58));

    for &batch in &[1usize, 2, 4, 5, 8] {
        let tok_s = bench_batch(&wrapper, batch);
        let per_req = tok_s / batch as f64;
        let blocks = batch * n_heads;
        println!("  {:>6}  {:>14.0}  {:>16.0}  {:>14}",
            batch, tok_s, per_req, blocks);
    }

    println!("\n  Kernel: q4k_gemv_sm120_batch (sm_120 PTX, static smem, grid=(N,M,1)).");
    println!("  Replaces oxide sm_80 batch kernels that crashed on Blackwell (error 719).");
    println!("  Remaining gap at high M: flash attention scales as O(M*kv_len*n_heads).");
    println!();
    println!("  Reference (single request, CUDA graph): 130 tok/s");
    println!("  llama.cpp single request:                209 tok/s");
}

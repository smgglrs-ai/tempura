/// bench-batch -- measure batched continuous decode throughput
///
/// Simulates B concurrent decode requests sharing a single GPU forward pass.
/// At B=5: 5x32=160 flash-attention blocks on RTX 5090 (170 SMs) -> full SM utilization.
///
/// Uses persistent KV caches so the CUDA graph captures ONCE (first decode step)
/// and all subsequent steps replay directly without re-capture overhead (~3ms saved/step).
///
/// Usage:
///   RUSTUP_TOOLCHAIN=nightly-2026-10-11 cargo run -p tempura-bench-gpu --release \
///     --bin bench-batch -- /path/to/model.gguf

use std::path::Path;
use std::time::Instant;

use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;

const WARMUP_STEPS: usize = 10;
const BENCH_STEPS:  usize = 50;

fn bench_batch(wrapper: &CutileInferenceWrapper, batch: usize) -> f64 {
    // Allocate M persistent KV caches once.
    let mut kv_caches = wrapper.new_batch_kv_caches(batch);

    // Prefill: position 0, one token per request.
    let mut cur_pos = 0u32;
    {
        let toks: Vec<u32> = vec![1u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let _    = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
        cur_pos += 1;
    }

    // Warmup: first step captures the CUDA graph, subsequent steps replay it.
    for _ in 0..WARMUP_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let _    = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
        cur_pos += 1;
    }

    // Timed: all steps are graph replays (no re-capture).
    let t0 = Instant::now();
    for _ in 0..BENCH_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let _    = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
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

    println!("================================================================================");
    println!("  tempura Batched Decode Benchmark                      ");
    println!("================================================================================\n");
    println!("  Model: {model_path}");
    println!("  Warmup: {WARMUP_STEPS} steps (step 1 captures CUDA graph, 2-{WARMUP_STEPS} replay)");
    println!("  Bench:  {BENCH_STEPS} steps (all graph replays)\n");

    println!("Loading model...");
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

    let mxfp4_info = if wrapper.mxfp4_tiled_count() > 0 {
        format!("MXFP4 ({} tiled weights, active={})",
            wrapper.mxfp4_tiled_count(),
            wrapper.mxfp4_active())
    } else {
        "MXFP4 unavailable (non-sm_120a hardware or no tiled weights)".to_string()
    };
    println!("  MXFP4 dispatch: {mxfp4_info}");

    for &batch in &[1usize, 2, 4, 5, 8, 16] {
        if batch > wrapper.max_batch_graph() {
            println!("  {:>6}  (exceeds max_batch_graph={})", batch, wrapper.max_batch_graph());
            continue;
        }
        let tok_s = bench_batch(&wrapper, batch);
        let per_req = tok_s / batch as f64;
        let blocks = batch * n_heads;
        println!("  {:>6}  {:>14.0}  {:>16.0}  {:>14}",
            batch, tok_s, per_req, blocks);
    }

    println!("\n  Projection dispatch: MXFP4 sm_120a (Q,K,O,gate,up) + Q6K (down) + Q4K (V).");
    println!("  Persistent KV caches: CUDA graph captures once, all bench steps replay.");
    println!("  Remaining gap at high M: flash attention scales as O(M*kv_len*n_heads).");
    println!();
    println!("  Reference (single request, CUDA graph): 130 tok/s");
    println!("  llama.cpp single request:                209 tok/s");
}

// bench-nograph -- throughput of the no-graph ragged batch decode path
// (the SERVING path: forward_batch_with_kvs with unequal positions and a
// non-f32 KV pool, both of which exclude the equal-position CUDA graphs).
//
// For each m in 1..=6 (TEMPURA_NOGRAPH_MAX_M): prefill m sessions with
// prompts of DIFFERENT lengths (forces ragged positions), warm up, then
// time TEMPURA_NOGRAPH_STEPS decode steps through one batch per step.
// Reports per-step wall time and aggregate tok/s.
//
// Run (laptop -> fishtank):
//   scripts/gpu-remote.sh bench nograph Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
//
// Env:
//   TEMPURA_NOGRAPH_STEPS   timed decode steps per m (default 48)
//   TEMPURA_NOGRAPH_MAX_M   max batch size (default 6)
//   TEMPURA_KV_DTYPE        pool dtype (default f16, the serving default)

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;
use tempura_engine::model::InferenceModel;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

/// Prompt bodies of increasing length so the decoded positions are ragged.
const PROMPTS: [&str; 6] = [
    "Explain paged attention.",
    "Explain paged attention in LLM serving.",
    "Explain paged attention in LLM serving, covering block tables.",
    "Explain paged attention in LLM serving, covering block tables and copy-on-write prefix sharing.",
    "Explain paged attention in LLM serving, covering block tables, copy-on-write prefix sharing, and eviction policy, in some detail.",
    "Explain paged attention in LLM serving, covering block tables, copy-on-write prefix sharing, eviction policy, and the interaction with continuous batching, at length and in depth.",
];

fn resolve_model_path(arg: Option<String>) -> PathBuf {
    let store = {
        let cache = std::env::var("XDG_CACHE_HOME")
            .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_default()));
        Path::new(&cache).join("tempura/models")
    };
    match arg {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if pb.exists() { pb } else { store.join(&p) }
        }
        None => store.join(DEFAULT_MODEL),
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn main() {
    let model_path = resolve_model_path(std::env::args().nth(1));
    let steps: usize = std::env::var("TEMPURA_NOGRAPH_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let max_m: usize = std::env::var("TEMPURA_NOGRAPH_MAX_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
        .min(6);

    println!("=== bench-nograph: no-graph ragged batch decode throughput ===");
    println!("  Model: {}", model_path.display());
    println!("  Steps per m: {steps}, max m: {max_m}");

    let t0 = Instant::now();
    let loader = GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF");
    let config = loader.config.clone();
    let tokenizer = Tokenizer::from_gguf_metadata(&loader.metadata).expect("tokenizer");
    let qw = loader.to_quantized_weights();
    let mut model = CutileInferenceWrapper::new(config.clone());
    model.load_quantized(qw);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let pool = model.new_kv_pool().expect("kv pool");

    let enc = |p: &str| -> Vec<u32> {
        let mut t = tokenizer.encode(p).expect("encode");
        if let Some(bos) = tokenizer.bos_token_id() {
            t.insert(0, bos);
        }
        t
    };
    let toks_all: Vec<Vec<u32>> = PROMPTS.iter().map(|p| enc(p)).collect();

    let mut results: Vec<(usize, f64, f64)> = Vec::new();
    // Server-condition bisect knobs: EQUAL uses one prompt for every
    // session (equal decode positions, like the serving engine), INIT
    // admits runs at 256 tokens so they grow in place like admissions.
    let equal = std::env::var("TEMPURA_NOGRAPH_EQUAL").is_ok();
    let init_tokens: usize = std::env::var("TEMPURA_NOGRAPH_INIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for m in 1..=max_m {
        // Fresh sessions each m (the pool frees finished runs on drop).
        let mut sessions = Vec::with_capacity(m);
        let mut kvs = Vec::with_capacity(m);
        let mut toks = Vec::with_capacity(m);
        let mut lens = Vec::new();
        for r in 0..m {
            let t = if equal { &toks_all[0] } else { &toks_all[r] };
            lens.push(t.len());
            let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
            let mut kv = if init_tokens > 0 {
                model
                    .new_session_kv_cache(&pool, init_tokens)
                    .expect("pooled kv alloc")
            } else {
                model
                    .new_session_kv_cache(&pool, t.len() + 512)
                    .expect("pooled kv alloc")
            };
            let logits = model
                .forward_with_kv(&mut session, t, &mut kv)
                .expect("prefill failed");
            sessions.push(session);
            kvs.push(kv);
            toks.push(argmax(&logits) as u32);
        }
        let ragged = lens.windows(2).any(|w| w[0] != w[1]);
        // Warmup (3 steps): first call sizes all cached scratch for this m.
        for _ in 0..3 {
            let poss: Vec<u32> = sessions.iter().map(|s| s.position as u32).collect();
            let rows = model
                .forward_batch_with_kvs(&toks, &poss, &mut kvs)
                .expect("batch decode failed");
            for (i, s) in sessions.iter_mut().enumerate() {
                s.position += 1;
                s.cache.advance(1);
                toks[i] = argmax(&rows[i]) as u32;
            }
        }
        let t = Instant::now();
        for _ in 0..steps {
            let poss: Vec<u32> = sessions.iter().map(|s| s.position as u32).collect();
            let rows = model
                .forward_batch_with_kvs(&toks, &poss, &mut kvs)
                .expect("batch decode failed");
            for (i, s) in sessions.iter_mut().enumerate() {
                s.position += 1;
                s.cache.advance(1);
                toks[i] = argmax(&rows[i]) as u32;
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        let per_step_ms = elapsed / steps as f64 * 1e3;
        let agg_tps = (steps * m) as f64 / elapsed;
        println!(
            "  m={}: {:.1} ms/step, {:.1} tok/s aggregate (ragged={}, prompt lens {:?})",
            m, per_step_ms, agg_tps, ragged, lens
        );
        results.push((m, per_step_ms, agg_tps));
        drop(sessions);
        drop(kvs);
    }
    println!("\n=== summary (m, ms/step, aggregate tok/s) ===");
    for (m, ms, tps) in &results {
        println!("  {m}\t{ms:.1}\t{tps:.1}");
    }
}

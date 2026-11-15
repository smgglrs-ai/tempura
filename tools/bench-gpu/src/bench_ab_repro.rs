// bench-ab-repro -- deterministic single-process repro for the 6-client
// kv-swap A/B divergence (found 2026-11-15: run A varies across server
// restarts; sessions diverge at their SECOND decode step exactly when
// ANOTHER session's prefill interleaves; the last-admitted session, whose
// decode never sees a later prefill, is bit-identical across restarts).
//
// Scenario matrix (same process, same pool):
//   S0: session A prefilled, N decode steps (m=1 batch path), checksums.
//   S0b: identical repeat of S0 (fresh session/run) -- intra-process
//        determinism baseline.
//   S1: session A' prefilled, ONE decode step, then session B is admitted
//        + prefilled (the serving engine's interleave), then A' continues
//        decoding.  Compare A' checksums vs S0.
//
// If S0==S0b but S1 diverges from S0 at the step after B's prefill, the
// prefill path perturbs concurrent decode state -- a causal, deterministic
// repro of the serving flakiness.
//
// Run (laptop -> fishtank):
//   scripts/gpu-remote.sh bench ab-repro Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
//
// Env knobs (bisect switches pass through to the engine):
//   TEMPURA_AB_STEPS   decode steps per scenario (default 12)
//   + all TEMPURA_NO_* kernel bisect switches.

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;
use tempura_engine::model::InferenceModel;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

const PROMPT_A: &str = "Explain in detail how paged attention works in an LLM inference server, covering page tables, block sizes, and why it helps memory utilization. Write at least 400 words.";
const PROMPT_B: &str = "Write a long technical explanation of the differences between prefill and decode phases in transformer inference, including KV cache behavior and batching strategies. Write at least 400 words.";

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

fn cs(logits: &[f32]) -> u64 {
    logits.iter().fold(0u64, |a, v| a.wrapping_add(v.to_bits() as u64))
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

/// Prefill `tokens` into a fresh session + pooled run, returning
/// (session, kv, logits of the last position).
fn prefill(
    model: &CutileInferenceWrapper,
    pool: &std::sync::Arc<std::sync::Mutex<tempura_engine::cutile_kv_cache::GpuKvPagePool>>,
    config: &tempura_engine::config::ModelConfig,
    tokens: &[u32],
) -> (InferenceSession, tempura_engine::cutile_kv_cache::CutileKvCache, Vec<f32>) {
    let mut session = InferenceSession::new(config, TraceLevel::Off, 0);
    let mut kv = model
        .new_session_kv_cache(pool, tokens.len() + 64)
        .expect("pooled kv alloc");
    let logits = model
        .forward_with_kv(&mut session, tokens, &mut kv)
        .expect("prefill failed");
    (session, kv, logits)
}

/// Decode `steps` tokens through the serving batch path (m=1), returning
/// per-step logits checksums plus the moved-back KV run.
fn decode(
    model: &CutileInferenceWrapper,
    session: &mut InferenceSession,
    mut kv: tempura_engine::cutile_kv_cache::CutileKvCache,
    first_token: u32,
    steps: usize,
) -> (
    Vec<u64>,
    tempura_engine::cutile_kv_cache::CutileKvCache,
    Vec<Vec<f32>>,
) {
    let mut out = Vec::with_capacity(steps);
    let mut logits_hist: Vec<Vec<f32>> = Vec::with_capacity(steps);
    let mut tok = first_token;
    for _ in 0..steps {
        let pos = session.position as u32;
        let mut arr = [kv];
        let rows = model
            .forward_batch_with_kvs(&[tok], &[pos], &mut arr)
            .expect("decode step failed");
        let [kv0] = arr;
        kv = kv0;
        // Mirror the engine's decode bookkeeping: the batch forward
        // advances the KV run internally; the SESSION position is ours.
        session.position += 1;
        session.cache.advance(1);
        let logits = &rows[0];
        out.push(cs(logits));
        logits_hist.push(logits.clone());
        tok = argmax(logits) as u32;
    }
    (out, kv, logits_hist)
}

fn main() {
    let model_path = resolve_model_path(std::env::args().nth(1));
    let steps: usize = std::env::var("TEMPURA_AB_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    println!("=== bench-ab-repro: prefill/decode interleave determinism ===");
    println!("  Model: {}", model_path.display());
    println!("  Steps: {steps}");

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
    let tok_a = enc(PROMPT_A);
    let tok_b = enc(PROMPT_B);
    println!("  prompt A: {} tokens, prompt B: {} tokens", tok_a.len(), tok_b.len());

    // -- S0: baseline (no interleave) --------------------------------------
    let (mut s0, kv0, logits0) = prefill(&model, &pool, &config, &tok_a);
    let first0 = argmax(&logits0) as u32;
    let (cs0, _, lg0) = decode(&model, &mut s0, kv0, first0, steps);
    println!("\nS0 (no interleave):   {:?}", cs0);

    // -- S0b: repeat (intra-process determinism) ---------------------------
    let (mut s0b, kv0b, logits0b) = prefill(&model, &pool, &config, &tok_a);
    let first0b = argmax(&logits0b) as u32;
    let (cs0b, _, _) = decode(&model, &mut s0b, kv0b, first0b, steps);
    println!("S0b (repeat):         {:?}", cs0b);
    report("S0 vs S0b (repeat)", &cs0, &cs0b);

    // -- S1: interleave B's prefill after A's FIRST decode step ------------
    // -- S1: interleave B's prefill after A's FIRST decode step ------------
    let (mut s1, mut kv1, logits1) = prefill(&model, &pool, &config, &tok_a);
    let first1 = argmax(&logits1) as u32;
    {
        let pos = s1.position as u32;
        let mut arr = [kv1];
        let rows = model
            .forward_batch_with_kvs(&[first1], &[pos], &mut arr)
            .expect("decode step failed");
        let [kv0] = arr;
        kv1 = kv0;
        s1.position += 1;
        s1.cache.advance(1);
        let tok = argmax(&rows[0]) as u32;
        // -- the interleave: admit + prefill session B (serving behavior) --
        let (_sb, _kvb, _lb) = prefill(&model, &pool, &config, &tok_b);
        // continue A's decode
        let (mut all, _, _) = decode(&model, &mut s1, kv1, tok, steps - 1);
        let mut full = vec![cs(&rows[0])];
        full.append(&mut all);
        println!("S1 (B prefilled after step 1): {:?}", full);
        report("S0 vs S1 (interleave)", &cs0, &full);
    }

    // -- S2/S3: batch-composition dependence -------------------------------
    // Decode A jointly with 1 (m=2) / 2 (m=3) companion sessions (prefilled
    // with prompt B, decoding their own greedy tokens).  If A's per-step
    // checksums change with m, the no-graph batch path is
    // composition-dependent -- the serving A/B gate requires invariance.
    for m in [2usize, 3usize] {
        let mut sessions: Vec<InferenceSession> = Vec::with_capacity(m);
        let mut kvs: Vec<tempura_engine::cutile_kv_cache::CutileKvCache> =
            Vec::with_capacity(m);
        let mut toks: Vec<u32> = Vec::with_capacity(m);
        // A first (index 0) -- its checksums are the comparison target.
        let (sa, kva, la) = prefill(&model, &pool, &config, &tok_a);
        sessions.push(sa);
        kvs.push(kva);
        toks.push(argmax(&la) as u32);
        for _ in 1..m {
            let (sc, kvc, lc) = prefill(&model, &pool, &config, &tok_b);
            sessions.push(sc);
            kvs.push(kvc);
            toks.push(argmax(&lc) as u32);
        }
        let mut out = Vec::with_capacity(steps);
        let mut first_diff_logged = false;
        for step in 0..steps {
            let poss: Vec<u32> = sessions.iter().map(|s| s.position as u32).collect();
            let rows = model
                .forward_batch_with_kvs(&toks, &poss, &mut kvs)
                .expect("batch decode failed");
            for (i, s) in sessions.iter_mut().enumerate() {
                s.position += 1;
                s.cache.advance(1);
                toks[i] = argmax(&rows[i]) as u32;
            }
            out.push(cs(&rows[0]));
            // Delta report vs S0 (same input tokens while checksums still
            // match): magnitude + count of differing logits localizes HOW
            // far the corruption propagated.
            if step < lg0.len() && !first_diff_logged {
                let mut ndiff = 0usize;
                let mut maxd = 0f32;
                for (a, b) in lg0[step].iter().zip(rows[0].iter()) {
                    if a.to_bits() != b.to_bits() {
                        ndiff += 1;
                        maxd = maxd.max((a - b).abs());
                    }
                }
                if ndiff > 0 {
                    println!(
                        "  S{} step {}: {} / {} logits differ, max|delta| = {maxd:.3e}",
                        m,
                        step + 1,
                        ndiff,
                        lg0[step].len()
                    );
                    first_diff_logged = true;
                }
            }
        }
        println!("S{} (m={} batch):      {:?}", m, m, out);
        report(&format!("S0 vs S{} (composition)", m), &cs0, &out);
    }
}

fn report(name: &str, cs0: &[u64], other: &[u64]) {
    if cs0 == other {
        println!("  {name}: IDENTICAL");
    } else {
        let first = cs0.iter().zip(other).position(|(a, b)| a != b);
        match first {
            Some(i) => println!("  {name}: DIVERGED at step {} (1-based)", i + 1),
            None => println!("  {name}: DIVERGED (length)"),
        }
    }
}


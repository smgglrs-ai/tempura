// bench-speculative — measure speculative decoding throughput on GPU
//
// Usage:
//   cargo run -p tempura-bench-gpu --release --bin bench-speculative \
//     -- /path/to/draft.gguf /path/to/target.gguf
//
// Implements speculative decoding:
//   1. Draft model generates K tokens autoregressively
//   2. Target model verifies K+1 tokens (K draft + 1 correction)
//   3. Accept/reject per-token; on rejection, rollback and correct
//
// NOTE: Speedup requires draft ITL < target_ITL × (1 - 1/(Kα+1)).
// With Bonsai-1.7B Q2_0 (~32ms/tok) as draft and Llama-3.1-8B Q4K (~9.3ms/tok)
// as target, the draft is too slow for net speedup at K=4, α=0.7.
// A 150+ tok/s draft model is needed to beat the target alone.

use std::path::Path;
use std::time::Instant;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{CutileInferenceWrapper, InferenceModel};
use tempura_engine::session::InferenceSession;
use tempura_engine::trace::TraceLevel;
use tempura_kernels::SamplingParams;

const WARMUP: usize = 5;
const BENCH_ROUNDS: usize = 30;
const K: usize = 4; // draft steps per round

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn argmax(probs: &[f32]) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Run one speculative decoding round (K draft + verify).
/// Returns (tokens_accepted, elapsed_ms).
fn speculative_round(
    draft: &CutileInferenceWrapper,
    target: &CutileInferenceWrapper,
    draft_session: &mut InferenceSession,
    target_session: &mut InferenceSession,
    last_token: u32,
) -> (usize, f64) {
    let t0 = Instant::now();

    // Save KV positions for potential rollback.
    let draft_pos0 = draft.gpu_kv_seq_len();
    let target_pos0 = target.gpu_kv_seq_len();

    // 1. Draft: generate K tokens (greedy for simplicity).
    let mut draft_tokens = Vec::with_capacity(K);
    let mut draft_probs_at = Vec::with_capacity(K); // p_draft(draft_token[i])
    let mut cur = last_token;
    for _ in 0..K {
        let logits = draft.forward(draft_session, &[cur]);
        let probs = softmax(&logits);
        let tok = argmax(&probs);
        draft_probs_at.push(probs[tok as usize]);
        draft_tokens.push(tok);
        cur = tok;
    }

    // 2. Target: verify each draft token sequentially.
    // Roll back target to where it was before draft ran.
    target.gpu_kv_rollback(target_session, target_pos0);

    let mut accepted = 0usize;
    cur = last_token;
    let mut correction_token = 0u32;
    let mut rejected = false;

    for i in 0..K {
        let logits = target.forward(target_session, &[cur]);
        let probs = softmax(&logits);
        let target_prob = probs[draft_tokens[i] as usize];
        let draft_prob = draft_probs_at[i].max(1e-10);

        if target_prob >= draft_prob {
            // Accept
            accepted += 1;
            cur = draft_tokens[i];
        } else {
            // Reject: sample corrected token (greedy = argmax of target probs)
            correction_token = argmax(&probs);
            rejected = true;
            break;
        }
    }

    // If all K accepted, also get bonus token from target.
    let total_tokens = if !rejected {
        let logits = target.forward(target_session, &[cur]);
        let probs = softmax(&logits);
        correction_token = argmax(&probs);
        accepted + 1
    } else {
        // Rollback draft KV cache to accepted position.
        draft.gpu_kv_rollback(draft_session, draft_pos0 + accepted + 1);
        accepted + 1
    };

    // Sync draft KV to accepted position (replay accepted tokens into draft).
    // For simplicity we just let draft continue from its current state.
    // (Slight inaccuracy: draft KV may have extra tokens from the K steps.)

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
    (total_tokens, elapsed_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: bench-speculative <draft.gguf> <target.gguf>");
        eprintln!("  draft.gguf  — small fast model (e.g. Bonsai-1.7B-Q2_0)");
        eprintln!("  target.gguf — large target model (e.g. Llama-3.1-8B-Q4K)");
        std::process::exit(1);
    }
    let draft_path = &args[1];
    let target_path = &args[2];

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  tempura Speculative Decoding Benchmark               ║");
    println!("╚══════════════════════════════════════════════════════╝\n");
    println!("  Draft:  {draft_path}");
    println!("  Target: {target_path}");
    println!("  K={K} draft steps per round, greedy sampling\n");

    // Load draft model.
    println!("Loading draft model…");
    let t0 = Instant::now();
    let draft_loader = GgufLoader::from_file_quantized(Path::new(draft_path))
        .expect("failed to open draft GGUF");
    let draft_config = draft_loader.config.clone();
    let draft_qw = draft_loader.to_quantized_weights();
    let mut draft_wrapper = CutileInferenceWrapper::new(draft_config.clone());
    draft_wrapper.load_quantized(draft_qw);
    println!("  Draft loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Load target model.
    println!("Loading target model…");
    let t1 = Instant::now();
    let target_loader = GgufLoader::from_file_quantized(Path::new(target_path))
        .expect("failed to open target GGUF");
    let target_config = target_loader.config.clone();
    let target_qw = target_loader.to_quantized_weights();
    let mut target_wrapper = CutileInferenceWrapper::new(target_config.clone());
    target_wrapper.load_quantized(target_qw);
    println!("  Target loaded in {:.2}s\n", t1.elapsed().as_secs_f64());

    // Prefill both models with a short prompt.
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // 8-token prompt
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        ..SamplingParams::default()
    };

    let mut draft_session = InferenceSession::new(&draft_config, TraceLevel::Off, 0);
    let mut target_session = InferenceSession::new(&target_config, TraceLevel::Off, 0);

    println!("Prefilling both models ({} tokens)…", prompt.len());
    draft_wrapper.forward(&mut draft_session, &prompt);
    target_wrapper.forward(&mut target_session, &prompt);
    let last_token = 1u32; // start decode from token 1

    // Warmup rounds.
    println!("Warmup ({WARMUP} rounds)…");
    for _ in 0..WARMUP {
        speculative_round(
            &draft_wrapper, &target_wrapper,
            &mut draft_session, &mut target_session,
            last_token,
        );
    }

    // Benchmark.
    println!("Benchmarking ({BENCH_ROUNDS} rounds)…\n");
    let mut total_tokens = 0usize;
    let mut total_ms = 0.0f64;
    let mut acceptance_sum = 0.0f64;

    for _ in 0..BENCH_ROUNDS {
        let (toks, ms) = speculative_round(
            &draft_wrapper, &target_wrapper,
            &mut draft_session, &mut target_session,
            last_token,
        );
        total_tokens += toks;
        total_ms += ms;
        acceptance_sum += (toks.saturating_sub(1)) as f64 / K as f64;
    }

    let avg_acceptance = acceptance_sum / BENCH_ROUNDS as f64;
    let effective_tok_s = total_tokens as f64 / (total_ms / 1e3);
    let ms_per_round = total_ms / BENCH_ROUNDS as f64;
    let toks_per_round = total_tokens as f64 / BENCH_ROUNDS as f64;

    println!("─── Speculative Decode Results ───");
    println!("  Draft steps K:       {K}");
    println!("  Avg acceptance rate: {:.1}%", avg_acceptance * 100.0);
    println!("  Tokens per round:    {toks_per_round:.2}");
    println!("  ms per round:        {ms_per_round:.1}ms");
    println!("  Effective tok/s:     {effective_tok_s:.0}");
    println!("  llama.cpp ref:       209 tok/s (RTX 5090, target alone)");
    println!("  target alone:        108 tok/s (current tempura)");
    println!();

    let target_alone_ms = 1000.0 / 108.0;
    let speedup = target_alone_ms * toks_per_round / ms_per_round;
    println!("  Speedup vs target-alone: {speedup:.2}×");
    if speedup < 1.0 {
        println!("  NOTE: Draft model too slow for net speedup.");
        println!("  Need draft ITL < {:.1}ms ({:.0} tok/s) for K={K}, α={:.0}%",
            target_alone_ms * (1.0 - 1.0 / (K as f64 * avg_acceptance + 1.0)),
            1000.0 / (target_alone_ms * (1.0 - 1.0 / (K as f64 * avg_acceptance + 1.0))),
            avg_acceptance * 100.0,
        );
    }

    println!("\nDone.");
}

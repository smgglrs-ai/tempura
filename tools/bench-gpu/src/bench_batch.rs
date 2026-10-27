/// bench-batch -- measure batched continuous decode throughput
///
/// Simulates B concurrent decode requests sharing a single GPU forward pass.
/// At B=5: 5x32=160 flash-attention blocks on RTX 5090 (170 SMs) -> full SM utilization.
///
/// Uses persistent KV caches so the CUDA graph captures ONCE (first decode step)
/// and all subsequent steps replay directly without re-capture overhead (~3ms saved/step).
///
/// Verification (2026-10-27 standard): every step's logits are scanned for NaNs
/// and the final step is fingerprinted (NaN-tolerant bits_sum + total_cmp argmax,
/// same scheme as bench-real-model). The whole schedule is then replayed on
/// FRESH caches -- forcing the CUDA graph to re-capture -- and the fingerprints
/// must match bitwise. Numbers count as verified only when both passes are
/// NaN-free and identical.
///
/// Usage:
///   cargo run -p tempura-bench-gpu --release --bin bench-batch -- /path/to/model.gguf

use std::path::Path;
use std::time::Instant;

use tempura_engine::config::ModelConfig;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::CutileInferenceWrapper;
use tempura_engine::weights::QuantizedWeights;

const WARMUP_STEPS: usize = 10;
const BENCH_STEPS:  usize = 50;
/// Verification replay: the full prefill + warmup + bench schedule, re-run.
const VERIFY_STEPS: usize = 1 + WARMUP_STEPS + BENCH_STEPS;

/// NaN-tolerant logits fingerprint over a whole batch. Broken kernels are
/// DIAGNOSED (explicit NaN count) instead of silently discarding logits.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Fingerprint {
    nan_count: usize,
    bits_sum:  u64,
    argmax:    usize,
    n_logits:  usize,
}

fn fingerprint(logits_batch: &[Vec<f32>]) -> Fingerprint {
    let mut fp = Fingerprint { nan_count: 0, bits_sum: 0, argmax: 0, n_logits: 0 };
    for logits in logits_batch {
        fp.nan_count += logits.iter().filter(|v| v.is_nan()).count();
        fp.bits_sum = fp.bits_sum
            .wrapping_add(logits.iter().map(|v| v.to_bits() as u64).sum::<u64>());
        fp.n_logits += logits.len();
        if let Some((i, _)) = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) {
            fp.argmax = i;
        }
    }
    fp
}

fn nan_count(logits_batch: &[Vec<f32>]) -> usize {
    logits_batch
        .iter()
        .map(|l| l.iter().filter(|v| v.is_nan()).count())
        .sum()
}

struct BatchResult {
    tok_s:     f64,
    nan_run1:  usize,          // NaNs across EVERY step of the timed run
    nan_run2:  usize,          // NaNs across every step of the verification replay
    final_fp:  Fingerprint,    // final step of the timed run
    verify_fp: Fingerprint,    // final step of the fresh-cache replay
}

impl BatchResult {
    /// Verified = both passes NaN-free, non-empty logits, bitwise-identical
    /// fingerprints (engine is deterministic by design).
    fn verified(&self) -> bool {
        self.nan_run1 == 0
            && self.nan_run2 == 0
            && self.final_fp.nan_count == 0
            && self.verify_fp.nan_count == 0
            && self.final_fp.n_logits > 0
            && self.final_fp == self.verify_fp
    }
}

fn bench_batch(wrapper: &CutileInferenceWrapper, batch: usize) -> BatchResult {
    // Allocate M persistent KV caches once.
    let mut kv_caches = wrapper.new_batch_kv_caches(batch);
    let mut nan_run1 = 0usize;
    let mut last: Vec<Vec<f32>> = Vec::new();
    // Per-step fingerprints (divergence localization: TEMPURA_BENCH_STEP_FP=1).
    let trace_steps = std::env::var("TEMPURA_BENCH_STEP_FP").is_ok();
    let mut step_fps: Vec<u64> = Vec::new();

    // Prefill: position 0, one token per request.
    let mut cur_pos = 0u32;
    {
        let toks: Vec<u32> = vec![1u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let logits = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
        nan_run1 += nan_count(&logits);
        if trace_steps {
            step_fps.push(fingerprint(&logits).bits_sum);
        }
        cur_pos += 1;
    }

    // Warmup: first step captures the CUDA graph, subsequent steps replay it.
    for _ in 0..WARMUP_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let logits = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
        nan_run1 += nan_count(&logits);
        if trace_steps {
            step_fps.push(fingerprint(&logits).bits_sum);
        }
        cur_pos += 1;
    }

    // Timed: all steps are graph replays (no re-capture).
    let t0 = Instant::now();
    for _ in 0..BENCH_STEPS {
        let toks: Vec<u32> = vec![2u32; batch];
        let pos:  Vec<u32> = vec![cur_pos; batch];
        let logits = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_caches);
        nan_run1 += nan_count(&logits);
        if trace_steps {
            step_fps.push(fingerprint(&logits).bits_sum);
        }
        last = logits;
        cur_pos += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tok_s = (batch * BENCH_STEPS) as f64 / elapsed;

    // WHY drop before the replay: KV caches are ~1 GiB/seq; at M=16 holding
    // both generations (32 GiB + model) exceeds RTX 5090 VRAM and the
    // verification pass OOMs. The cached CUDA graph only compares these
    // pointers (never dereferences) when deciding to re-capture, so freeing
    // them is safe.
    drop(kv_caches);

    // Verification replay: identical token/position schedule on fresh caches.
    // The KV pointers change, so the cached CUDA graph re-captures -- proving
    // capture path and replay path agree. Deterministic by design -> the final
    // fingerprints must be bitwise identical.
    let mut kv_verify = wrapper.new_batch_kv_caches(batch);
    let mut nan_run2 = 0usize;
    let mut v_pos = 0u32;
    let mut v_last: Vec<Vec<f32>> = Vec::new();
    let mut v_step_fps: Vec<u64> = Vec::new();
    for i in 0..VERIFY_STEPS {
        let tok = if i == 0 { 1u32 } else { 2u32 };
        let toks: Vec<u32> = vec![tok; batch];
        let pos:  Vec<u32> = vec![v_pos; batch];
        let logits = wrapper.forward_batch_decode_stateful(&toks, &pos, &mut kv_verify);
        nan_run2 += nan_count(&logits);
        if trace_steps {
            v_step_fps.push(fingerprint(&logits).bits_sum);
        }
        v_last = logits;
        v_pos += 1;
    }

    if trace_steps {
        // First step where the two runs diverge (schedule is identical, so
        // any difference is the bug). 0 = capture step (prefill), 1..10 =
        // warmup (1 captures the graph), 11..60 = timed replays.
        let mut diverged_at = None;
        for (i, (a, b)) in step_fps.iter().zip(v_step_fps.iter()).enumerate() {
            if a != b {
                diverged_at = Some(i);
                break;
            }
        }
        match diverged_at {
            Some(i) => {
                println!("         [step-fp] first divergence at step {i} (kv_len={}):", i + 1);
                let lo = i.saturating_sub(2);
                let hi = (i + 3).min(step_fps.len());
                for s in lo..hi {
                    let mark = if s == i { " <-- diverge" } else { "" };
                    println!("           step {s:>3}: run1={} run2={}{mark}",
                        step_fps[s], v_step_fps[s]);
                }
                // TEMPURA_BENCH_DUMP: element-wise diff of the diverging step's
                // logits (run 1 vs run 2), localizing which output region moves.
                if std::env::var("TEMPURA_BENCH_DUMP").is_ok() && !last.is_empty() && !v_last.is_empty() {
                    for r in 0..last.len().min(v_last.len()) {
                        let a = &last[r];
                        let b = &v_last[r];
                        let n = a.len().min(b.len());
                        let (mut ndiff, mut first_idx) = (0usize, None);
                        for j in 0..n {
                            if a[j].to_bits() != b[j].to_bits() {
                                ndiff += 1;
                                if first_idx.is_none() { first_idx = Some(j); }
                            }
                        }
                        println!("           [dump] req {r}: {ndiff}/{} logits differ, first at idx {:?}",
                            n, first_idx);
                        if let Some(j) = first_idx {
                            println!("             run1[{j}]={:?} run2[{j}]={:?}", a[j], b[j]);
                            println!("             run1[j-2..j+3]={:?} run2[j-2..j+3]={:?}",
                                &a[j.saturating_sub(2)..(j + 3).min(n)],
                                &b[j.saturating_sub(2)..(j + 3).min(n)]);
                        }
                    }
                }
            }
            None => println!("         [step-fp] no step-level divergence ({} steps)", step_fps.len()),
        }
        // All requests share identical inputs, so per-request bits_sums of
        // the final step must be EQUAL -- inequality is cross-request
        // contamination inside the batch kernels.
        let per_req: Vec<u64> = last.iter()
            .map(|l| l.iter().map(|v| v.to_bits() as u64).sum::<u64>())
            .collect();
        let all_same = per_req.windows(2).all(|w| w[0] == w[1]);
        println!("         [per-req] final step per-request bits_sums {}",
            if all_same { "all equal".to_string() } else { format!("DIVERGE: {per_req:?}") });
    }

    BatchResult {
        tok_s,
        nan_run1,
        nan_run2,
        final_fp: fingerprint(&last),
        verify_fp: fingerprint(&v_last),
    }
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
    println!("  Bench:  {BENCH_STEPS} steps (all graph replays)");
    println!("  Verify: {VERIFY_STEPS}-step replay on fresh caches (graph re-capture) + NaN scan\n");

    println!("Loading model...");
    let t0 = Instant::now();
    let model_p = Path::new(&model_path);
    let (mut config, qw) = if model_p.is_dir() {
        // BF16 safetensors directory: load config.json + raw BF16 weights.
        let config_path = model_p.join("config.json");
        let cfg = ModelConfig::from_hf_config(&config_path)
            .expect("failed to read config.json");
        let qw = QuantizedWeights::from_safetensors_bf16_dir(model_p)
            .expect("failed to load safetensors");
        (cfg, qw)
    } else {
        let loader = GgufLoader::from_file_quantized(model_p)
            .expect("failed to open GGUF");
        let cfg = loader.config.clone();
        let qw = loader.to_quantized_weights();
        (cfg, qw)
    };
    // Allow overriding max_seq_len via env var so large models fit in VRAM during bench.
    // Default: keep model's native max_seq_len for standard models, cap at 8192 for 27B+.
    if let Ok(s) = std::env::var("TEMPURA_BENCH_MAX_SEQ_LEN") {
        let cap: usize = s.parse().expect("TEMPURA_BENCH_MAX_SEQ_LEN must be an integer");
        if cap < config.max_seq_len {
            eprintln!("[bench] capping max_seq_len {} -> {} (TEMPURA_BENCH_MAX_SEQ_LEN)", config.max_seq_len, cap);
            config.max_seq_len = cap;
        }
    }
    let mut wrapper = CutileInferenceWrapper::new(config);
    wrapper.load_quantized(qw);
    println!("  Loaded in {:.2}s\n", t0.elapsed().as_secs_f64());

    let n_heads = wrapper.model.config.num_attention_heads;

    println!("  {:>6}  {:>11}  {:>13}  {:>11}  {:>9}", "Batch", "total tok/s", "per-req tok/s", "attn blocks", "verified");
    println!("  {}", "-".repeat(64));

    let mxfp4_info = if wrapper.mxfp4_tiled_count() > 0 {
        format!("MXFP4 ({} tiled weights, active={})",
            wrapper.mxfp4_tiled_count(),
            wrapper.mxfp4_active())
    } else {
        "MXFP4 unavailable (non-sm_120a hardware or no tiled weights)".to_string()
    };
    println!("  MXFP4 dispatch: {mxfp4_info}");

    // For MXFP4-only models (BF16 safetensors), M=1 non-graph path has no MXFP4 dispatch yet.
    // M>=2 uses the MXFP4 CUDA graph path which is fully supported.
    let mxfp4_only = wrapper.mxfp4_tiled_count() > 0 && !model_p.is_file();
    let mut all_verified = true;
    // TEMPURA_BENCH_ONLY_M=N: bench a single batch size (divergence hunting).
    let only_m: Option<usize> = std::env::var("TEMPURA_BENCH_ONLY_M")
        .ok()
        .and_then(|s| s.parse().ok());
    let batch_sizes: Vec<usize> = match only_m {
        Some(m) => vec![m],
        None => vec![1usize, 2, 4, 5, 8, 16],
    };
    for &batch in &batch_sizes {
        if batch > wrapper.max_batch_graph() {
            println!("  {:>6}  (exceeds max_batch_graph={})", batch, wrapper.max_batch_graph());
            continue;
        }
        if mxfp4_only && batch == 1 {
            println!("  {:>6}  (skipped: M=1 non-graph path has no MXFP4 dispatch; use M>=2)", batch);
            continue;
        }
        let r = bench_batch(&wrapper, batch);
        let per_req = r.tok_s / batch as f64;
        let blocks = batch * n_heads;
        let verified = if r.verified() { "yes" } else { "NO" };
        if !r.verified() { all_verified = false; }
        println!("  {:>6}  {:>11.0}  {:>13.0}  {:>11}  {:>9}",
            batch, r.tok_s, per_req, blocks, verified);
        println!("         fp: bits_sum={} argmax={} n={} | nan={}/{} det={}",
            r.final_fp.bits_sum, r.final_fp.argmax, r.final_fp.n_logits,
            r.nan_run1, r.nan_run2,
            if r.final_fp == r.verify_fp { "OK" } else { "MISMATCH" });
    }

    println!("\n  Persistent KV caches: CUDA graph captures once, all bench steps replay.");
    println!("  Remaining gap at high M: flash attention scales as O(M*kv_len*n_heads).");
    if all_verified {
        println!("  All batch sizes verified: NaN-free + deterministic (fresh-cache replay identical).");
    } else {
        println!("  WARNING: some batch sizes UNVERIFIED (NaNs, empty logits, or nondeterminism).");
        println!("           Bisect with TEMPURA_NO_SM120_KPAR / TEMPURA_NO_SM120_Q6K /");
        println!("           TEMPURA_NO_OXIDE_ATTN / TEMPURA_NO_DUAL_BLOCK.");
    }
    println!();
    println!("  Reference (single request, CUDA graph): 136 tok/s (verified NaN-free 2026-10-27)");
    println!("  llama.cpp single request:                209 tok/s");
}

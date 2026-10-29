// bench-quality -- end-to-end generation-quality validation of the GPU path
// vs the CPU reference for the same GGUF weights: greedy continuation of a
// fixed prompt on both paths, per-position argmax token agreement, logits
// delta, and decoded-text diff.
//
// This is the missing quality check behind the 2026-10-27 retraction: the
// fixed MXFP4 paths pass determinism + NaN + random-data CPU-reference
// kernel checks, but no end-to-end text-generation comparison vs the CPU
// existed (see AGENTS.md "Generation-quality validation of the MXFP4
// paths"). Numerics checks catch wrong kernels; only generation catches
// wrong models.
//
// Run (laptop -> fishtank):
//   scripts/gpu-remote.sh bench quality Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
// Local (needs a GPU):
//   cargo run -p tempura-bench-gpu --release --bin bench-quality -- \
//     /path/to/model.gguf
//
// Env knobs:
//   TEMPURA_QUALITY_PROMPT     prompt text (default: a fixed story prompt)
//   TEMPURA_QUALITY_TOKENS     generated tokens per side (default 64)
//   TEMPURA_QUALITY_MIN_AGREE  pass threshold, 0..1 (default 0.80)
// Exit code 1 if argmax agreement is below the threshold (or any GPU NaN).

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{CutileInferenceWrapper, InferenceModel, SampleParams};
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;
use tempura_engine::{ModelConfig, TransformerModel};

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const DEFAULT_PROMPT: &str = "Write a short story about a robot who learns to paint.";
const DEFAULT_TOKENS: usize = 64;
const DEFAULT_MIN_AGREE: f64 = 0.80;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Bare filenames resolve against the model store (~/.cache/tempura/models),
/// same root the gpu-remote workflow uses.
fn resolve_model_path(arg: Option<String>) -> PathBuf {
    let store = {
        let cache = std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| {
            format!("{}/.cache", std::env::var("HOME").unwrap_or_default())
        });
        Path::new(&cache).join("tempura/models")
    };
    match arg {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if pb.exists() {
                pb
            } else {
                let cand = store.join(&p);
                if cand.exists() {
                    cand
                } else {
                    pb // let the loader report the missing file
                }
            }
        }
        None => store.join(DEFAULT_MODEL),
    }
}

/// total_cmp argmax -- NaN-tolerant idiom shared with bench-real-model.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// top1 - top2 of a logit vector; small margin = ambiguous argmax, so a
/// cross-path token mismatch there is expected numerics, not a bug.
fn top_margin(logits: &[f32]) -> f32 {
    let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > top1 {
            top2 = top1;
            top1 = v;
        } else if v > top2 {
            top2 = v;
        }
    }
    top1 - top2
}

fn forward_catch(
    model: &dyn InferenceModel,
    session: &mut InferenceSession,
    tokens: &[u32],
) -> Option<Vec<f32>> {
    std::panic::catch_unwind(AssertUnwindSafe(|| model.forward(session, tokens))).ok()
}

struct GreedyRun {
    /// Generated token ids, position i produced from logits[i].
    tokens: Vec<u32>,
    /// Logits vector per step; logits[i] is the distribution token[i] was
    /// argmaxed from (logits[0] = prefill output).
    logits: Vec<Vec<f32>>,
    nan_steps: usize,
    /// Total NaN logit entries across all steps (nan_steps only counts
    /// steps with >= 1 NaN).
    nan_entries: usize,
    /// True when the input tokens were teacher-forced from the reference
    /// run instead of the run's own argmax.
    teacher_forced: bool,
    elapsed_s: f64,
}

/// Greedy continuation driven manually (not InferenceModel::generate) so
/// per-position logits are retained for cross-path comparison.
///
/// `follow`: when Some, the token fed at step i+1 is follow[i+1] (teacher
/// forcing) -- both paths then see identical inputs, so per-step logit
/// deltas isolate numerics instead of cascading through divergent argmax
/// choices. The run's own argmax is still recorded for comparison.
fn greedy_generate(
    model: &dyn InferenceModel,
    config: &ModelConfig,
    prompt: &[u32],
    max_new: usize,
    eos: Option<u32>,
    follow: Option<&[u32]>,
) -> GreedyRun {
    let mut session = InferenceSession::new(config, TraceLevel::Off, 0);
    let t0 = Instant::now();
    let mut run = GreedyRun {
        tokens: Vec::with_capacity(max_new),
        logits: Vec::with_capacity(max_new),
        nan_steps: 0,
        nan_entries: 0,
        teacher_forced: follow.is_some(),
        elapsed_s: 0.0,
    };
    let mut logits =
        forward_catch(model, &mut session, prompt).expect("prefill forward failed");
    while run.tokens.len() < max_new {
        let nan_count = logits.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            run.nan_steps += 1;
            run.nan_entries += nan_count;
        }
        let tok = argmax(&logits) as u32;
        run.tokens.push(tok);
        run.logits.push(std::mem::take(&mut logits));
        if Some(tok) == eos && follow.is_none() {
            break;
        }
        // Teacher forcing keeps both paths on the reference token sequence;
        // EOS only ends the run when the reference itself ended there. The
        // next token to FEED is the one just pushed (its logits come next):
        // feed follow[len-1], NOT follow[len] (an off-by-one here skips the
        // first generated token and invalidates the whole comparison).
        let next = match follow {
            Some(seq) => seq.get(run.tokens.len() - 1).copied(),
            None => Some(tok),
        };
        let next = match next {
            Some(t) => t,
            None => break,
        };
        logits =
            forward_catch(model, &mut session, &[next]).expect("decode forward failed");
    }
    run.elapsed_s = t0.elapsed().as_secs_f64();
    run
}

struct StepDiff {
    pos: usize,
    cpu_tok: u32,
    gpu_tok: u32,
    agree: bool,
    logit_max_abs: f32,
    cpu_margin: f32,
    cpu_top: f32,
    gpu_top: f32,
}

struct Comparison {
    steps: Vec<StepDiff>,
    agree: usize,
    first_div: Option<usize>,
    max_logit_abs: f32,
    sum_logit_abs: f32,
    /// Entries that are NaN on exactly one side -- f32::max ignores NaN, so
    /// these are invisible to max_logit_abs and counted separately.
    nan_diffs: usize,
}

fn compare_runs(cpu: &GreedyRun, gpu: &GreedyRun) -> Comparison {
    let n = cpu.tokens.len().min(gpu.tokens.len());
    let mut steps = Vec::with_capacity(n);
    let mut agree = 0;
    let mut first_div = None;
    let mut max_logit_abs = 0.0f32;
    let mut sum_logit_abs = 0.0f32;
    let mut nan_diffs = 0usize;
    for i in 0..n {
        let cl = &cpu.logits[i];
        let gl = &gpu.logits[i];
        let mut logit_max_abs = 0.0f32;
        for (&a, &b) in cl.iter().zip(gl) {
            if a.is_nan() || b.is_nan() {
                if a.is_nan() != b.is_nan() {
                    nan_diffs += 1;
                }
            } else {
                logit_max_abs = logit_max_abs.max((a - b).abs());
            }
        }
        max_logit_abs = max_logit_abs.max(logit_max_abs);
        sum_logit_abs += logit_max_abs;
        let cpu_tok = cpu.tokens[i];
        let gpu_tok = gpu.tokens[i];
        let step_agree = cpu_tok == gpu_tok;
        if step_agree {
            agree += 1;
        } else if first_div.is_none() {
            first_div = Some(i);
        }
        steps.push(StepDiff {
            pos: i,
            cpu_tok,
            gpu_tok,
            agree: step_agree,
            logit_max_abs,
            cpu_margin: top_margin(cl),
            cpu_top: cl[argmax(cl)],
            gpu_top: gl[argmax(gl)],
        });
    }
    Comparison {
        steps,
        agree,
        first_div,
        max_logit_abs,
        sum_logit_abs,
        nan_diffs,
    }
}

fn main() {
    let model_path = resolve_model_path(std::env::args().nth(1));
    let prompt_text = std::env::var("TEMPURA_QUALITY_PROMPT")
        .unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let max_new = env_usize("TEMPURA_QUALITY_TOKENS", DEFAULT_TOKENS);
    let min_agree = env_f64("TEMPURA_QUALITY_MIN_AGREE", DEFAULT_MIN_AGREE);

    println!("=== tempura GPU generation-quality validation (GPU vs CPU) ===\n");
    println!("  Model: {}", model_path.display());
    println!("  Prompt: {prompt_text:?}");
    println!("  Tokens per side: {max_new}, min agreement: {:.0}%", min_agree * 100.0);

    // --- Load GGUF (GPU side; to_quantized_weights consumes the loader) ---
    println!("Loading GGUF (GPU side)...");
    let t0 = Instant::now();
    let loader_gpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF file");
    let config = loader_gpu.config.clone();
    let metadata = loader_gpu.metadata.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());
    println!(
        "  Config: {} layers, h={}, heads={}/{}, d={}, vocab={}",
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
    );

    let tokenizer = Tokenizer::from_gguf_metadata(&metadata).ok();
    match &tokenizer {
        Some(_) => println!("  Tokenizer: extracted from GGUF metadata"),
        None => println!("  Tokenizer: NOT available, falling back to synthetic tokens"),
    }
    let prompt: Vec<u32> = match &tokenizer {
        Some(tok) => tok.encode(&prompt_text).expect("prompt tokenization failed"),
        None => {
            // Same synthetic fallback as the other benches: quality numbers
            // are meaningless without a tokenizer, but the argmax agreement
            // comparison is still valid.
            (0..32u32)
                .map(|i| 1 + (i % (config.vocab_size as u32 - 1)))
                .collect()
        }
    };
    let eos = tokenizer.as_ref().and_then(|t| t.eos_token_id());
    println!("  Prompt tokens: {}", prompt.len());
    if std::env::var("TEMPURA_DEBUG_TOKENS").is_ok() {
        println!("  Prompt token ids: {:?}", &prompt[..prompt.len().min(40)]);
    }

    // --- GPU model ---
    println!("\nUploading weights to GPU...");
    let t1 = Instant::now();
    let qw_gpu = loader_gpu.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config.clone());
    wrapper.load_quantized(qw_gpu);
    println!("  Loaded in {:.2}s", t1.elapsed().as_secs_f64());

    // --- CPU reference model (second parse: load_quantized frees the host
    // copies after upload, so the CPU side needs its own weights) ---
    println!("\nBuilding CPU reference model...");
    let t2 = Instant::now();
    let loader_cpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to re-open GGUF file");
    let qw_cpu = loader_cpu.to_quantized_weights();
    let mut cpu_model =
        TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    cpu_model.quantized_weights = Some(qw_cpu);
    println!("  Ready in {:.2}s", t2.elapsed().as_secs_f64());

    // --- Greedy generation on both paths ---
    // Default: teacher-force the GPU run onto the CPU token sequence so
    // per-step logit deltas isolate numerics instead of cascading through
    // divergent argmax choices.  Free-running agreement is a coin flip
    // whenever the prompt lands the model in near-tie territory (2026-10-28:
    // first divergence at a margin-0.014 top1/top2 gap, after which the two
    // runs hold different contexts and every later position "disagrees" with
    // logit deltas of 10-20 that say nothing about kernel health).
    // TEMPURA_QUALITY_FREE=1 restores the old free-running end-to-end mode
    // (still printed below either way -- the GPU run records its own argmax).
    let teacher = std::env::var("TEMPURA_QUALITY_FREE").as_deref() != Ok("1");
    println!("\n--- CPU reference greedy generation ---");
    let cpu_run = greedy_generate(&cpu_model, &config, &prompt, max_new, eos, None);
    println!(
        "  CPU: {} tokens in {:.1}s ({:.1} tok/s), NaN entries: {}",
        cpu_run.tokens.len(),
        cpu_run.elapsed_s,
        cpu_run.tokens.len() as f64 / cpu_run.elapsed_s.max(1e-9),
        cpu_run.nan_entries,
    );

    println!("\n--- GPU greedy generation ---");
    let follow = if teacher { Some(&cpu_run.tokens[..]) } else { None };
    let gpu_run = greedy_generate(&wrapper, &config, &prompt, max_new, eos, follow);
    println!(
        "  GPU: {} tokens in {:.1}s ({:.1} tok/s), NaN entries: {}, {}",
        gpu_run.tokens.len(),
        gpu_run.elapsed_s,
        gpu_run.tokens.len() as f64 / gpu_run.elapsed_s.max(1e-9),
        gpu_run.nan_entries,
        if gpu_run.teacher_forced {
            "teacher-forced onto the CPU token sequence"
        } else {
            "free-running"
        },
    );

    // TEMPURA_QUALITY_GPU_SELFTEST=1: recompute each teacher-forced position
    // via a fresh GPU prefill (host-dequant, TEMPURA_NO_GPU_PREFILL respected)
    // of [prompt + tokens so far] and compare last-row logits against the
    // decode path -- a GPU-internal measure of decode-path error with no CPU
    // reference involved.
    let gpu_selftest = std::env::var("TEMPURA_QUALITY_GPU_SELFTEST").as_deref() == Ok("1");
    let mut selftest_deltas: Vec<f32> = Vec::new();
    if gpu_selftest {
        println!("\n--- GPU decode-vs-prefill self-test ---");
        for j in 0..gpu_run.logits.len() {
            let mut prefix = prompt.clone();
            prefix.extend_from_slice(&cpu_run.tokens[..j]);
            let mut sess = InferenceSession::new(&config, TraceLevel::Off, 0);
            let pl = forward_catch(&wrapper, &mut sess, &prefix)
                .expect("selftest prefill failed");
            let dl = &gpu_run.logits[j];
            let mut max_abs = 0.0f32;
            let mut nan_diffs = 0usize;
            for (a, b) in pl.iter().zip(dl) {
                if a.is_nan() || b.is_nan() {
                    if a.is_nan() != b.is_nan() {
                        nan_diffs += 1;
                    }
                } else {
                    max_abs = max_abs.max((a - b).abs());
                }
            }
            selftest_deltas.push(max_abs);
            // Prefill-vs-CPU for the same position: if this grows with j, the
            // reference side (longer prefills) is what drifts, not decode.
            let cl = &cpu_run.logits[j];
            let mut pf_max = 0.0f32;
            for (a, b) in pl.iter().zip(cl) {
                if !a.is_nan() && !b.is_nan() {
                    pf_max = pf_max.max((a - b).abs());
                }
            }
            println!(
                "  pos {j:>3}: prefill-vs-decode max_abs={max_abs:.4} prefill-vs-cpu={pf_max:.4} nan_diffs={nan_diffs}"
            );
        }
    }

    // --- Comparison ---
    let cmp = compare_runs(&cpu_run, &gpu_run);
    let n = cmp.steps.len();
    let agreement = if n > 0 {
        cmp.agree as f64 / n as f64
    } else {
        0.0
    };
    let avg_logit_abs = cmp.sum_logit_abs / n.max(1) as f32;

    println!("\n--- Argmax agreement (GPU vs CPU) ---");
    if let Some(s0) = cmp.steps.first() {
        println!(
            "  Prefill logits: max_abs={:.4} argmax {}",
            s0.logit_max_abs,
            if s0.agree { "MATCH" } else { "MISMATCH" },
        );
    }
    println!(
        "  Agreement:      {}/{} positions ({agreement:.1}%)",
        cmp.agree, n
    );
    match cmp.first_div {
        Some(p) => println!("  First divergence at generated position {p}"),
        None => println!("  First divergence: none (identical continuations)"),
    }
    println!(
        "  Logits delta:   max_abs={:.4} avg_abs={:.4}",
        cmp.max_logit_abs, avg_logit_abs
    );
    println!(
        "  GPU NaN steps:  {} ({} entries, {})",
        gpu_run.nan_steps,
        gpu_run.nan_entries,
        if gpu_run.nan_steps == 0 { "clean" } else { "FAIL" },
    );
    println!(
        "  NaN-mismatched logits entries (one side NaN): {}",
        cmp.nan_diffs
    );

    let mismatches: Vec<&StepDiff> = cmp.steps.iter().filter(|s| !s.agree).collect();
    if !mismatches.is_empty() {
        println!("\n  Mismatched positions (cpu_margin = top1-top2 of the CPU");
        println!("  distribution; small margin means the argmax was ambiguous):");
        for s in mismatches.iter().take(20) {
            println!(
                "    pos {:>3}: cpu={} gpu={} cpu_top={:.3} gpu_top={:.3} margin={:.3} logit_max_abs={:.4}",
                s.pos, s.cpu_tok, s.gpu_tok, s.cpu_top, s.gpu_top, s.cpu_margin, s.logit_max_abs
            );
        }
        if mismatches.len() > 20 {
            println!("    ... and {} more", mismatches.len() - 20);
        }
    }

    if let Some(tok) = &tokenizer {
        println!("\n--- Continuations ---");
        let cpu_text = tok.decode(&cpu_run.tokens).unwrap_or_default();
        let gpu_text = tok.decode(&gpu_run.tokens).unwrap_or_default();
        println!("  CPU: {cpu_text}");
        println!("  GPU: {gpu_text}");
    }

    // --- GPU greedy fast-path check ---------------------------------------
    // InferenceModel::generate with temperature=0 routes through
    // forward_greedy (GPU argmax, no logits download) when available. The
    // generated tokens must match a FREE-RUNNING manual continuation
    // (forward + CPU argmax) on the same model -- this validates the
    // kernel's tie-breaking (Rust max_by: last max wins) end-to-end.
    // (The main gpu_run above is teacher-forced by default and diverges
    // from any free-running path at near-ties by design.)
    let fast_path_ok = {
        let free_gpu = greedy_generate(&wrapper, &config, &prompt, max_new, eos, None);
        let params = SampleParams {
            temperature: 0.0,
            ..SampleParams::default()
        };
        let mut session = InferenceSession::new(&config, TraceLevel::Off, 0);
        let gen = wrapper.generate(&mut session, &prompt, max_new, &params, eos);
        let n = gen.len().min(free_gpu.tokens.len());
        let diff = (0..n).filter(|&i| gen[i] != free_gpu.tokens[i]).count();
        let fast = diff == 0 && n == free_gpu.tokens.len() && n > 0;
        println!(
            "\n  greedy fast-path (generate vs free-run manual continuation): \
             {n}/{} tokens, {diff} diffs -> {}",
            free_gpu.tokens.len(),
            if fast { "MATCH" } else { "MISMATCH" }
        );
        fast
    };
    println!();

    // --- Verdict + JSON report ---
    let passed = agreement >= min_agree && fast_path_ok
        && gpu_run.nan_steps == 0
        && cmp.nan_diffs == 0;
    println!("\n--- Verdict ---");
    println!(
        "  {} (agreement {:.1}% vs threshold {:.0}%, NaN {} steps/{} entries, {} NaN-mismatched)",
        if passed { "PASS" } else { "FAIL" },
        agreement * 100.0,
        min_agree * 100.0,
        gpu_run.nan_steps,
        gpu_run.nan_entries,
        cmp.nan_diffs,
    );

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"model\": {:?},\n", model_path.display().to_string()));
    json.push_str(&format!("  \"prompt\": {:?},\n", prompt_text));
    json.push_str(&format!("  \"tokens_compared\": {n},\n"));
    json.push_str(&format!("  \"agreement\": {agreement:.4},\n"));
    json.push_str(&format!("  \"first_divergence\": {},\n",
        cmp.first_div.map(|p| p.to_string()).unwrap_or_else(|| "null".into())));
    json.push_str(&format!("  \"logits_max_abs\": {:.6},\n", cmp.max_logit_abs));
    json.push_str(&format!("  \"logits_avg_abs\": {:.6},\n", avg_logit_abs));
    json.push_str(&format!(
        "  \"gpu_nan_steps\": {},\n  \"gpu_nan_entries\": {},\n  \"nan_diffs\": {},\n",
        gpu_run.nan_steps, gpu_run.nan_entries, cmp.nan_diffs
    ));
    json.push_str(&format!("  \"teacher_forced\": {},\n", gpu_run.teacher_forced));
    json.push_str(&format!("  \"cpu_tokens\": {:?},\n", cpu_run.tokens));
    json.push_str(&format!("  \"gpu_tokens\": {:?},\n", gpu_run.tokens));
    json.push_str(&format!("  \"mismatches\": [\n"));
    for (i, s) in cmp.steps.iter().enumerate() {
        let comma = if i + 1 < cmp.steps.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"pos\": {}, \"cpu_tok\": {}, \"gpu_tok\": {}, \"agree\": {}, \"cpu_margin\": {:.4}, \"logit_max_abs\": {:.4}}}{}\n",
            s.pos, s.cpu_tok, s.gpu_tok, s.agree, s.cpu_margin, s.logit_max_abs, comma
        ));
    }
    json.push_str("  ],\n");
    json.push_str(&format!("  \"passed\": {passed}\n"));
    json.push_str("}\n");
    std::fs::write("bench-quality.json", &json)
        .expect("failed to write bench-quality.json");
    println!("\n  Report written to bench-quality.json");

    if !passed {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY; 4]), 0);
        // total_cmp orders NaN as greatest, matching the bench fingerprint
        // idiom -- a NaN-poisoned vector must still return an index, not panic.
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 1);
    }

    #[test]
    fn top_margin_is_top1_minus_top2() {
        assert_eq!(top_margin(&[1.0, 3.0, 2.0]), 1.0);
        assert_eq!(top_margin(&[5.0]), f32::INFINITY);
        assert_eq!(top_margin(&[2.0, 2.0]), 0.0);
    }

    fn run(tokens: &[u32], top: f32) -> GreedyRun {
        GreedyRun {
            tokens: tokens.to_vec(),
            logits: tokens
                .iter()
                .map(|&t| vec![top + t as f32, top + 0.5])
                .collect(),
            nan_steps: 0,
            nan_entries: 0,
            teacher_forced: false,
            elapsed_s: 0.0,
        }
    }

    #[test]
    fn compare_counts_agreement_and_first_divergence() {
        let cpu = run(&[7, 8, 9, 10], 10.0);
        let gpu = run(&[7, 8, 4, 10], 10.5);
        let c = compare_runs(&cpu, &gpu);
        assert_eq!(c.agree, 3);
        assert_eq!(c.first_div, Some(2));
        assert!(c.max_logit_abs > 0.0);
    }

    #[test]
    fn compare_counts_nan_mismatched_entries() {
        let cpu = run(&[1, 2], 1.0);
        let mut gpu = run(&[1, 2], 1.0);
        gpu.logits[0][0] = f32::NAN;
        let c = compare_runs(&cpu, &gpu);
        assert_eq!(c.nan_diffs, 1);
        // A NaN on one side must not corrupt the finite-entry max_abs
        // (f32::max would silently ignore the NaN diff).
        assert_eq!(c.max_logit_abs, 0.0);
    }

    #[test]
    fn identical_runs_agree_fully() {
        let cpu = run(&[1, 2, 3], 1.0);
        let c = compare_runs(&cpu, &cpu);
        assert_eq!(c.agree, 3);
        assert_eq!(c.first_div, None);
        assert_eq!(c.max_logit_abs, 0.0);
    }
}

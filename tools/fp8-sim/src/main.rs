// fp8-sim -- CPU-side go/no-go harness for the mid-precision (e4m3) prefill
// path (2026-11-18 investigation). Runs the MMLU-style log-prob evaluation
// TWICE on the SAME model+prompts: once on the unmodified CPU quantized
// reference path, once with tempura-kernels' Fp8SimMatMulQ active (every
// Q4K/Q6K prefill projection rounding activations and/or dequantized
// weights through e4m3 -- the numerics class of a Blackwell e4m3 x e4m3
// mma with f32 accumulation). Reports per-side accuracy, choice agreement,
// and the last-position logit band on a prompt subset.
//
// Usage:
//   cargo run -p tempura-fp8-sim --release -- \
//     [model.gguf] [mmlu_sample.jsonl]
//
// Env knobs:
//   TEMPURA_FP8_SIM_N       question cap (default: all)
//   TEMPURA_FP8_SIM_MODE    simulation mode, same grammar as
//                           TEMPURA_FP8_SIM in tempura-kernels
//                           (default "act+weight")
//   TEMPURA_FP8_SIM_BAND_N  prompts with full-logit band reporting
//                           (default 16)
//
// KV-compression sweep mode (TurboQuant go/no-go, 2026-11-23):
//   TEMPURA_KV_SIM_SWEEP    comma list of bits:group configs, e.g.
//                           "2:32,3:64,4:128". Replaces the fp8 mode
//                           machinery: runs the reference (sim off) once,
//                           then every config with the polar-quant KV
//                           round-trip active at the storage boundary
//                           (quantize-on-write serving semantics), and
//                           prints the ladder table (accuracy / choice
//                           agreement / logit band vs reference).
//   TEMPURA_KV_SIM_PPL      optional corpus file (WikiText-2 raw): adds a
//                           perplexity leg per config (CPU path, llama.cpp
//                           scoring convention) with the ppl ratio vs the
//                           reference run.
//   TEMPURA_KV_SIM_PPL_CHUNKS  max ppl chunks (default 8, the pooled-gate
//                           convention)
//   TEMPURA_KV_SIM_PPL_CTX     ppl chunk size (default 512)

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::eval::{mmlu_format_prompt, token_logprob, MMLU_LETTERS};
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{InferenceModel, TransformerModel};
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;
use tempura_kernels::fp8_sim;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn resolve_model_path(arg: Option<String>) -> PathBuf {
    let store = {
        let cache = std::env::var("XDG_CACHE_HOME")
            .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_default()));
        Path::new(&cache).join("tempura/models")
    };
    match arg {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if pb.exists() {
                pb
            } else if let (c, true) = (store.join(&p), store.join(&p).exists()) {
                c
            } else {
                pb
            }
        }
        None => store.join(DEFAULT_MODEL),
    }
}

fn resolve_corpus_path(arg: Option<String>) -> PathBuf {
    if let Some(p) = arg {
        return PathBuf::from(p);
    }
    let cache = std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| {
        format!("{}/.cache", std::env::var("HOME").unwrap_or_default())
    });
    Path::new(&cache).join("tempura/eval/mmlu/mmlu_sample.jsonl")
}

struct Question {
    prompt: String,
    answer: usize,
}

fn load_questions(path: &Path, cap: usize) -> Vec<Question> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if out.len() >= cap {
            break;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).expect("bad jsonl line in mmlu sample");
        let subject = v["subject"].as_str().unwrap_or("unknown");
        let question = v["question"].as_str().unwrap_or_default();
        let choices: Vec<String> = v["choices"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| c.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let answer = v["answer"].as_u64().unwrap_or(0) as usize;
        if choices.len() != 4 || question.is_empty() {
            continue;
        }
        out.push(Question {
            prompt: mmlu_format_prompt(subject, question, &choices),
            answer: answer.min(3),
        });
    }
    out
}

fn letter_token_ids(tok: &Tokenizer, bos: Option<u32>) -> [usize; 4] {
    let mut ids = [0usize; 4];
    for (i, l) in MMLU_LETTERS.iter().enumerate() {
        let mut enc = tok.encode(l).expect("letter tokenization failed");
        if enc.len() > 1 && Some(enc[0]) == bos {
            enc.remove(0);
        }
        assert_eq!(enc.len(), 1, "letter {l:?} tokenized to {enc:?}");
        ids[i] = enc[0] as usize;
    }
    ids
}

struct RunResult {
    picks: Vec<usize>,
    /// full last-position logits for the first band_n prompts
    band_logits: Vec<Vec<f32>>,
}

fn evaluate(
    model: &dyn InferenceModel,
    config: &tempura_engine::ModelConfig,
    prompts: &[Vec<u32>],
    letters: &[usize; 4],
    band_n: usize,
    label: &str,
) -> RunResult {
    let mut picks = Vec::with_capacity(prompts.len());
    let mut band_logits = Vec::new();
    let t = Instant::now();
    for (i, prompt) in prompts.iter().enumerate() {
        let mut session = InferenceSession::new(config, TraceLevel::Off, 0);
        let logits = model.forward(&mut session, prompt);
        let nan = logits.iter().filter(|v| v.is_nan()).count();
        if nan > 0 {
            eprintln!("[{label}] question {i}: {nan} NaN logits");
        }
        if i < band_n {
            band_logits.push(logits.clone());
        }
        let lps: Vec<f64> = letters.iter().map(|&l| token_logprob(&logits, l)).collect();
        let mut best = 0;
        for (c, &lp) in lps.iter().enumerate() {
            if lp > lps[best] {
                best = c;
            }
        }
        picks.push(best);
        if (i + 1) % 25 == 0 {
            eprintln!("  [{label}] {}/{} ({:.1}s)", i + 1, prompts.len(), t.elapsed().as_secs_f64());
        }
    }
    eprintln!("  [{label}] done in {:.1}s", t.elapsed().as_secs_f64());
    RunResult { picks, band_logits }
}

fn accuracy(picks: &[usize], answers: &[usize]) -> f64 {
    let n = picks.len().min(answers.len());
    if n == 0 {
        return 0.0;
    }
    (0..n).filter(|&i| picks[i] == answers[i]).count() as f64 / n as f64
}

fn agreement(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    (0..n).filter(|&i| a[i] == b[i]).count() as f64 / n as f64
}

/// Logit band between two runs' stashed logits: max/mean |delta| and the
/// same relative to the reference max |logit| (the bench-real-model
/// "max |logit| rel" convention used for the MXFP4 band of 0.10-0.13).
fn logit_band(ref_l: &[Vec<f32>], sim_l: &[Vec<f32>]) -> (f64, f64, f64) {
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut count = 0usize;
    let mut max_rel = 0.0f64;
    for (r, s) in ref_l.iter().zip(sim_l.iter()) {
        let ref_peak = r.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        for (a, b) in r.iter().zip(s.iter()) {
            let d = (*a - *b).abs() as f64;
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d;
            count += 1;
        }
        let rel = max_abs / ref_peak as f64;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, sum_abs / count.max(1) as f64, max_rel)
}

fn parse_mode(spec: &str) -> u8 {
    let mut mode = 0u8;
    for part in spec.split('+') {
        match part.trim() {
            "off" | "" => {}
            "act" => mode |= fp8_sim::FP8_SIM_ACT,
            "weight" => mode |= fp8_sim::FP8_SIM_WEIGHT,
            "wscale" => mode |= fp8_sim::FP8_SIM_WSCALE,
            "arowscale" => mode |= fp8_sim::FP8_SIM_AROWSCALE,
            "ftz" => mode |= fp8_sim::FP8_SIM_FTZ,
            "noop" => mode |= fp8_sim::FP8_SIM_NOOP,
            "q8act" => mode |= fp8_sim::FP8_SIM_Q8ACT,
            "mmq" => mode |= fp8_sim::FP8_SIM_Q8ACT | fp8_sim::FP8_SIM_MMQF16,
            other => panic!("unknown mode part {other:?}"),
        }
    }
    mode
}

fn main() {
    // KV-compression sweep mode: structured early-return path.
    if let Ok(sweep) = std::env::var("TEMPURA_KV_SIM_SWEEP") {
        kv_sim_sweep_main(&sweep);
        return;
    }

    let mut args = std::env::args().skip(1);
    let model_path = resolve_model_path(args.next());
    let sample_path = resolve_corpus_path(args.next());
    let n = env_usize("TEMPURA_FP8_SIM_N", usize::MAX);
    let band_n = env_usize("TEMPURA_FP8_SIM_BAND_N", 16);
    let mode_spec =
        std::env::var("TEMPURA_FP8_SIM_MODE").unwrap_or_else(|_| "act+weight".to_string());
    let sim_mode = parse_mode(&mode_spec);

    println!("=== tempura fp8-sim (e4m3 prefill simulation) ===\n");
    println!("  Model:  {}", model_path.display());
    println!("  Sample: {}", sample_path.display());
    println!("  Mode:   {mode_spec} (bits {sim_mode:#04x})");

    let questions = load_questions(&sample_path, n);
    println!("  Questions: {}", questions.len());
    assert!(!questions.is_empty(), "empty MMLU sample");

    println!("\nLoading GGUF...");
    let t0 = Instant::now();
    let loader = GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF file");
    let config = loader.config.clone();
    let metadata = loader.metadata.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());

    let tokenizer =
        Tokenizer::from_gguf_metadata(&metadata).expect("MMLU needs the real tokenizer");
    let bos = tokenizer.bos_token_id();
    let letters = letter_token_ids(&tokenizer, bos);

    let prompts: Vec<Vec<u32>> = questions
        .iter()
        .map(|q| tokenizer.encode(&q.prompt).expect("prompt tokenization failed"))
        .collect();
    let avg_len = prompts.iter().map(|p| p.len()).sum::<usize>() / prompts.len();
    println!("  Prompts tokenized (avg {avg_len} tokens)");

    let t1 = Instant::now();
    let qw = loader.to_quantized_weights();
    let mut model = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    model.quantized_weights = Some(qw);
    println!("  Weights ready in {:.2}s", t1.elapsed().as_secs_f64());

    let answers: Vec<usize> = questions.iter().map(|q| q.answer).collect();

    // Phase 1: unmodified reference (mode OFF -- bit-identical to the
    // stock CPU quantized path; the dispatch is a pass-through).
    fp8_sim::fp8_sim_set_mode(0);
    println!("\n--- reference (fp8 sim OFF) ---");
    let reference = evaluate(&model, &config, &prompts, &letters, band_n, "ref");
    println!(
        "  accuracy {:.1}%",
        accuracy(&reference.picks, &answers) * 100.0
    );

    // Phase 2: simulation.
    fp8_sim::fp8_sim_set_mode(sim_mode);
    println!("\n--- simulation ({mode_spec}) ---");
    let sim = evaluate(&model, &config, &prompts, &letters, band_n, "sim");
    println!("  accuracy {:.1}%", accuracy(&sim.picks, &answers) * 100.0);

    fp8_sim::fp8_sim_set_mode(0);

    // Comparison.
    let (max_abs, mean_abs, max_rel) = logit_band(&reference.band_logits, &sim.band_logits);
    println!("\n=== comparison ===");
    println!(
        "  questions:        {}",
        reference.picks.len().min(sim.picks.len())
    );
    println!(
        "  ref accuracy:     {:.1}%",
        accuracy(&reference.picks, &answers) * 100.0
    );
    println!(
        "  sim accuracy:     {:.1}%",
        accuracy(&sim.picks, &answers) * 100.0
    );
    println!(
        "  choice agreement: {:.1}%  (gate 95%; MXFP4 GPU measured 58.0%, exact 96.8-98.0%)",
        agreement(&reference.picks, &sim.picks) * 100.0
    );
    println!("  logit band (first {band_n} prompts):");
    println!("    max |delta|:    {max_abs:.4}");
    println!("    mean |delta|:   {mean_abs:.5}");
    println!(
        "    max rel to peak |logit|: {max_rel:.4}  (MXFP4 GPU band ~0.10-0.13; exact ~0)"
    );
    let flips: Vec<usize> = (0..reference.picks.len())
        .filter(|&i| reference.picks[i] != sim.picks[i])
        .collect();
    if !flips.is_empty() {
        println!("    flipped questions: {flips:?}");
    }
}

// -- KV-compression sweep (TurboQuant go/no-go) -------------------------------

struct KvSimConfig {
    bits: u8,
    group: usize,
}

fn parse_sweep(spec: &str) -> Vec<KvSimConfig> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (b, g) = part
            .split_once(':')
            .unwrap_or_else(|| panic!("bad sweep config {part:?} (want bits:group)"));
        let bits: u8 = b.trim().parse().expect("bits must be an integer");
        let group: usize = g.trim().parse().expect("group must be an integer");
        assert!((1..=8).contains(&bits), "bits out of range: {bits}");
        assert!(group > 0, "group must be > 0");
        out.push(KvSimConfig { bits, group });
    }
    assert!(!out.is_empty(), "empty TEMPURA_KV_SIM_SWEEP");
    out
}

/// Bits/key per element of a config (payload + per-group min/max f32
/// scales), and the resulting pool density in KiB/token for the
/// 32-layer x 2 x 1024 GQA pool (e4m3 = 64 KiB/token, f16 = 128).
fn density_kib_per_token(c: &KvSimConfig) -> (f64, f64) {
    let payload = c.bits as f64 / 8.0;
    let scales = 8.0 / c.group as f64; // 2 x f32 per group
    let bytes_per_elem = payload + scales;
    let density = bytes_per_elem * 1024.0 * 32.0 * 2.0 / 1024.0;
    (bytes_per_elem, density)
}

fn kv_sim_sweep_main(sweep_spec: &str) {
    use tempura_engine::eval::{perplexity, PplOptions};
    use tempura_kernels::kv_quant_sim;

    let configs = parse_sweep(sweep_spec);
    let n = env_usize("TEMPURA_FP8_SIM_N", usize::MAX);
    let band_n = env_usize("TEMPURA_FP8_SIM_BAND_N", 16);
    let ppl_path = std::env::var("TEMPURA_KV_SIM_PPL").ok();
    let ppl_chunks = env_usize("TEMPURA_KV_SIM_PPL_CHUNKS", 8);
    let ppl_ctx = env_usize("TEMPURA_KV_SIM_PPL_CTX", 512);

    let mut args = std::env::args().skip(1);
    let model_path = resolve_model_path(args.next());
    let sample_path = resolve_corpus_path(args.next());

    println!("=== tempura fp8-sim (KV polar-quant sweep) ===\n");
    println!("  Model:  {}", model_path.display());
    println!("  Sample: {}", sample_path.display());
    println!("  Sweep:  {sweep_spec}");
    if let Some(ref p) = ppl_path {
        println!("  Ppl:    {p} ({} chunks x {} ctx)", ppl_chunks, ppl_ctx);
    }

    let questions = load_questions(&sample_path, n);
    println!("  Questions: {}", questions.len());
    assert!(!questions.is_empty(), "empty MMLU sample");

    println!("\nLoading GGUF...");
    let t0 = Instant::now();
    let loader = GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF file");
    let config = loader.config.clone();
    let metadata = loader.metadata.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());

    let tokenizer =
        Tokenizer::from_gguf_metadata(&metadata).expect("MMLU needs the real tokenizer");
    let bos = tokenizer.bos_token_id();
    let letters = letter_token_ids(&tokenizer, bos);

    let prompts: Vec<Vec<u32>> = questions
        .iter()
        .map(|q| tokenizer.encode(&q.prompt).expect("prompt tokenization failed"))
        .collect();
    println!("  Prompts tokenized (avg {} tokens)", prompts.iter().map(|p| p.len()).sum::<usize>() / prompts.len());

    let t1 = Instant::now();
    let qw = loader.to_quantized_weights();
    let mut model = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    model.quantized_weights = Some(qw);
    println!("  Weights ready in {:.2}s", t1.elapsed().as_secs_f64());

    let answers: Vec<usize> = questions.iter().map(|q| q.answer).collect();

    // Corpus for the ppl leg (tokenized once, shared by every run).
    let corpus_tokens = ppl_path.as_ref().map(|p| {
        let text = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("failed to read {p}: {e}"));
        tokenizer.encode(&text).expect("corpus tokenization failed")
    });

    // Reference: sim off -- the stock CPU quantized path (f32 KV).
    kv_quant_sim::kv_sim_off();
    println!("\n--- reference (KV sim OFF) ---");
    let reference = evaluate(&model, &config, &prompts, &letters, band_n, "ref");
    let ref_acc = accuracy(&reference.picks, &answers);
    println!("  accuracy {:.1}%", ref_acc * 100.0);
    let ref_ppl = corpus_tokens.as_ref().map(|toks| {
        let opts = PplOptions {
            n_ctx: ppl_ctx,
            max_chunks: Some(ppl_chunks),
            bos,
            progress: true,
            collect_logprobs: false,
        };
        perplexity(&model, &config, toks, &opts).expect("corpus too short for one chunk")
    });
    if let Some(ref r) = ref_ppl {
        println!("  ppl {:.4} ({} chunks, {} positions, {} NaN steps)", r.ppl, r.chunks, r.count, r.nan_steps);
    }

    println!("\n=== KV polar-quant ladder (gate: agreement >= 95%, ppl ratio <= 1.01) ===");
    println!(
        "{:>10} {:>10} {:>12} {:>12} {:>11} {:>11} {:>10}",
        "bits", "group", "B/elem", "KiB/tok", "acc", "agreement", "ppl ratio"
    );
    let mut results: Vec<String> = Vec::new();
    for c in &configs {
        kv_quant_sim::kv_sim_set(c.bits, c.group);
        let label = format!("b{}g{}", c.bits, c.group);
        let sim = evaluate(&model, &config, &prompts, &letters, band_n, &label);
        let sim_acc = accuracy(&sim.picks, &answers);
        let agr = agreement(&reference.picks, &sim.picks);
        let (max_abs, _mean_abs, max_rel) = logit_band(&reference.band_logits, &sim.band_logits);

        let sim_ppl = corpus_tokens.as_ref().map(|toks| {
            let opts = PplOptions {
                n_ctx: ppl_ctx,
                max_chunks: Some(ppl_chunks),
                bos,
                progress: false,
                collect_logprobs: false,
            };
            perplexity(&model, &config, toks, &opts).expect("corpus too short for one chunk")
        });
        kv_quant_sim::kv_sim_off();

        let (bpe, kib) = density_kib_per_token(c);
        let ppl_ratio = match (&ref_ppl, &sim_ppl) {
            (Some(r), Some(s)) => format!("{:.4}", s.ppl / r.ppl),
            _ => "-".to_string(),
        };
        println!(
            "{:>10} {:>10} {:>12.4} {:>12.1} {:>10.1}% {:>10.1}% {:>10}",
            c.bits, c.group, bpe, kib, sim_acc * 100.0, agr * 100.0, ppl_ratio
        );
        println!(
            "           logit band: max |d| {max_abs:.4}, max rel-to-peak {max_rel:.4}; ppl {}",
            sim_ppl.as_ref().map(|r| format!("{:.4}", r.ppl)).unwrap_or_else(|| "-".into())
        );
        results.push(format!(
            "bits={} group={} B/elem={:.4} KiB/tok={:.1} acc={:.1}% agr={:.1}% ppl={} ratio={}",
            c.bits,
            c.group,
            bpe,
            kib,
            sim_acc * 100.0,
            agr * 100.0,
            sim_ppl.as_ref().map(|r| format!("{:.4}", r.ppl)).unwrap_or_else(|| "-".into()),
            ppl_ratio
        ));
    }

    println!("\n=== summary ===");
    for r in &results {
        println!("  {r}");
    }
}

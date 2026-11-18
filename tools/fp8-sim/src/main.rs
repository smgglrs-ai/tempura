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
            other => panic!("unknown mode part {other:?}"),
        }
    }
    mode
}

fn main() {
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

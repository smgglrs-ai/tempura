// bench-selfconsistency -- GSM8K-style self-consistency generation: quality
// benchmark (c) from the 2026-10-30 plan. For each problem, sample K
// completions at temperature 0.7 (seeded, deterministic per sample), extract
// the final numeric answer from each, majority-vote. Robust to argmax noise:
// near-tie flips average out across samples (Wang et al. 2022) -- this is
// what production systems do.
//
// Sides: tempura GPU (direct) and llama.cpp via llama-server's OpenAI-
// compatible API (model loaded once; set TEMPURA_SC_LLAMA_URL, default
// http://127.0.0.1:8081, started separately:
//   llama-server -m model.gguf -ngl 0 --port 8081
// ). tempura CPU is opt-in (TEMPURA_SC_CPU=1; scalar decode is ~3 tok/s,
// hours for the full sample).
//
// Run (GPU host, directly):
//   ./target/release/bench-selfconsistency /path/to/model.gguf [sample.jsonl]
//
// Env knobs:
//   TEMPURA_SC_PROBLEMS     problems to evaluate (default: all in sample)
//   TEMPURA_SC_SAMPLES      samples per problem (default 5)
//   TEMPURA_SC_MAX_TOKENS   generation cap (default 200)
//   TEMPURA_SC_SEED         base seed (default 42; sample k uses base+k)
//   TEMPURA_SC_MAX_GAP      accuracy gap gate vs llama (default 0.15)
//   TEMPURA_SC_LLAMA_URL    llama-server URL (default http://127.0.0.1:8081)
//   TEMPURA_SC_CPU=1        also run the tempura CPU model
//   TEMPURA_NO_LLAMA=1      skip the llama.cpp side
//   TEMPURA_SC_NO_GPU=1     skip the tempura GPU side (CPU-only hosts)

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::eval::{extract_answer, majority};
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{InferenceModel, SampleParams, TransformerModel};
#[cfg(feature = "cutile")]
use tempura_engine::model::CutileInferenceWrapper;
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::{ChatMessageInput, Tokenizer};
use tempura_engine::trace::TraceLevel;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_MAX_TOKENS: usize = 200;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_MAX_GAP: f64 = 0.15;

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
                    pb
                }
            }
        }
        None => store.join(DEFAULT_MODEL),
    }
}

fn resolve_sample_path(arg: Option<String>) -> PathBuf {
    if let Some(p) = arg {
        return PathBuf::from(p);
    }
    let cache = std::env::var("XDG_CACHE_HOME")
        .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_default()));
    Path::new(&cache).join("tempura/eval/gsm8k/gsm8k_sample.jsonl")
}

struct Problem {
    prompt: String,
    gold: String,
}

fn accuracy(results: &[Option<String>], problems: &[Problem]) -> f64 {
    let n = results.len().min(problems.len());
    if n == 0 {
        return 0.0;
    }
    let hit = (0..n)
        .filter(|&i| {
            results[i]
                .as_ref()
                .and_then(|r| r.parse::<f64>().ok())
                .and_then(|r| problems[i].gold.parse::<f64>().ok().map(|g| (r, g)))
                .map(|(r, g)| (r - g).abs() < 1e-6)
                .unwrap_or(false)
        })
        .count();
    hit as f64 / n as f64
}

fn run_side(
    model: &dyn InferenceModel,
    config: &tempura_engine::ModelConfig,
    tokenizer: &Tokenizer,
    problems: &[Problem],
    samples: usize,
    max_tokens: usize,
    base_seed: u64,
    tag: &str,
) -> Vec<Option<String>> {
    let params = SampleParams {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        ..SampleParams::default()
    };
    problems
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let prompt = tokenizer
                .apply_chat_template(
                    &[ChatMessageInput {
                        role: "user".to_string(),
                        content: p.prompt.clone(),
                    }],
                    true,
                )
                .unwrap_or_else(|_| p.prompt.clone());
            let ids = tokenizer.encode(&prompt).expect("encode failed");
            let eos = tokenizer.eos_token_id();
            let mut sample_answers = Vec::with_capacity(samples);
            for k in 0..samples {
                let mut session =
                    InferenceSession::new(config, TraceLevel::Off, base_seed + k as u64);
                let out = model.generate(&mut session, &ids, max_tokens, &params, eos);
                let text = tokenizer.decode(&out).unwrap_or_default();
                sample_answers.push(extract_answer(&text));
            }
            if i % 5 == 0 {
                eprintln!("  [{tag}] problem {}/{}", i + 1, problems.len());
            }
            majority(&sample_answers)
        })
        .collect()
}

/// One chat completion via llama-server's OpenAI-compatible endpoint.
fn llama_chat(
    url: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    seed: u64,
) -> Result<String, String> {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": max_tokens,
        "seed": seed,
    });
    let out = std::process::Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "600",
            "-X",
            "POST",
            &format!("{url}/v1/chat/completions"),
            "-H",
            "Content-Type: application/json",
            "-d",
            &body.to_string(),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("json: {e}: {text}"))?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("no content: {text}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = resolve_model_path(args.next());
    let sample_path = resolve_sample_path(args.next());
    let n_problems = env_usize("TEMPURA_SC_PROBLEMS", usize::MAX);
    let samples = env_usize("TEMPURA_SC_SAMPLES", DEFAULT_SAMPLES);
    let max_tokens = env_usize("TEMPURA_SC_MAX_TOKENS", DEFAULT_MAX_TOKENS);
    let base_seed = env_usize("TEMPURA_SC_SEED", DEFAULT_SEED as usize) as u64;
    let max_gap = env_f64("TEMPURA_SC_MAX_GAP", DEFAULT_MAX_GAP);

    println!("=== tempura GSM8K self-consistency (K={samples}, T=0.7) ===\n");
    println!("  Model:  {}", model_path.display());
    println!("  Sample: {}", sample_path.display());

    let text = std::fs::read_to_string(&sample_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", sample_path.display()));
    let mut problems: Vec<Problem> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("bad jsonl line");
        let question = v["question"].as_str().unwrap_or_default().trim().to_string();
        let gold = v["gold"].as_str().unwrap_or_default().trim().to_string();
        if question.is_empty() || gold.is_empty() {
            continue;
        }
        let prompt = format!(
            "{question}\n\nSolve this step by step, then give only the final numeric answer on the last line in the form \"Answer: <number>\"."
        );
        problems.push(Problem { prompt, gold });
    }
    problems.truncate(n_problems);
    println!("  Problems: {}", problems.len());
    assert!(!problems.is_empty(), "empty sample");

    println!("\nLoading GGUF...");
    let t0 = Instant::now();
    let loader = GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF");
    let config = loader.config.clone();
    let metadata = loader.metadata.clone();
    let tokenizer = Tokenizer::from_gguf_metadata(&metadata).expect("needs a tokenizer");
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());

    #[cfg(feature = "cutile")]
    let mut wrapper: Option<CutileInferenceWrapper> = None;
    #[cfg(feature = "cutile")]
    if std::env::var("TEMPURA_SC_NO_GPU").is_err() {
        let qw = loader.to_quantized_weights();
        let mut w = CutileInferenceWrapper::new(config.clone());
        w.load_quantized(qw);
        wrapper = Some(w);
    }

    let mut cpu_model = None;
    if std::env::var("TEMPURA_SC_CPU").is_ok() {
        let loader_cpu =
            GgufLoader::from_file_quantized(&model_path).expect("failed to re-open GGUF");
        let qw_cpu = loader_cpu.to_quantized_weights();
        let mut m = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
        m.quantized_weights = Some(qw_cpu);
        cpu_model = Some(m);
    }

    let mut gpu_acc = f64::NAN;
    #[cfg(feature = "cutile")]
    if let Some(wrapper) = &wrapper {
        println!("\n--- tempura GPU ---");
        let t = Instant::now();
        let gpu_results = run_side(
            wrapper,
            &config,
            &tokenizer,
            &problems,
            samples,
            max_tokens,
            base_seed,
            "gpu",
        );
        gpu_acc = accuracy(&gpu_results, &problems);
        println!(
            "  majority-vote accuracy {:.1}%  ({:.1}s, {samples} samples/problem)",
            gpu_acc * 100.0,
            t.elapsed().as_secs_f64()
        );
    } else {
        println!("\n(tempura GPU side skipped: TEMPURA_SC_NO_GPU=1)");
    }
    #[cfg(not(feature = "cutile"))]
    println!("\n(tempura GPU side skipped: built without the cutile feature)");

    let mut cpu_acc = f64::NAN;
    if let Some(m) = &cpu_model {
        println!("\n--- tempura CPU ---");
        let t = Instant::now();
        let cpu_results = run_side(m, &config, &tokenizer, &problems, samples, max_tokens, base_seed, "cpu");
        cpu_acc = accuracy(&cpu_results, &problems);
        println!(
            "  majority-vote accuracy {:.1}%  ({:.1}s)",
            cpu_acc * 100.0,
            t.elapsed().as_secs_f64()
        );
    }

    // --- llama.cpp via llama-server ---
    let mut llama_acc = f64::NAN;
    if std::env::var("TEMPURA_NO_LLAMA").is_err() {
        let url = std::env::var("TEMPURA_SC_LLAMA_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8081".to_string());
        println!("\n--- llama.cpp (llama-server at {url}) ---");
        match llama_chat(&url, "Reply with just: ok", 8, 0.0, 1) {
            Ok(_) => {
                let t = Instant::now();
                let mut results = Vec::with_capacity(problems.len());
                for (i, p) in problems.iter().enumerate() {
                    let mut sample_answers = Vec::with_capacity(samples);
                    for k in 0..samples {
                        match llama_chat(&url, &p.prompt, max_tokens, 0.7, base_seed + k as u64) {
                            Ok(text) => sample_answers.push(extract_answer(&text)),
                            Err(e) => {
                                eprintln!("  llama request failed: {e}");
                                sample_answers.push(None);
                            }
                        }
                    }
                    results.push(majority(&sample_answers));
                    if i % 5 == 0 {
                        eprintln!("  [llama] problem {}/{}", i + 1, problems.len());
                    }
                }
                llama_acc = accuracy(&results, &problems);
                println!(
                    "  majority-vote accuracy {:.1}%  ({:.1}s)",
                    llama_acc * 100.0,
                    t.elapsed().as_secs_f64()
                );
            }
            Err(e) => println!("  (llama-server not reachable at {url}: {e}; start it to enable)"),
        }
    }

    // --- Comparison / verdict ---
    println!("\n--- Verdict ---");
    let mut passed = true;
    let fmt = |a: f64| if a.is_nan() { "skipped".to_string() } else { format!("{:.1}%", a * 100.0) };
    if !gpu_acc.is_nan() && !llama_acc.is_nan() {
        let gap = (gpu_acc - llama_acc).abs();
        println!(
            "  GPU {:.1}% vs llama {:.1}% (gap {:.1} pts, gate {:.0})",
            gpu_acc * 100.0,
            llama_acc * 100.0,
            gap * 100.0,
            max_gap * 100.0
        );
        passed &= gap <= max_gap;
    }
    if !gpu_acc.is_nan() && !cpu_acc.is_nan() {
        let gap = (gpu_acc - cpu_acc).abs();
        println!("  GPU {:.1}% vs CPU {:.1}% (gap {:.1} pts)", gpu_acc * 100.0, cpu_acc * 100.0, gap * 100.0);
        passed &= gap <= max_gap;
    }
    println!(
        "  sides: GPU {}, CPU {}, llama {}",
        fmt(gpu_acc),
        fmt(cpu_acc),
        fmt(llama_acc)
    );
    if gpu_acc.is_nan() && cpu_acc.is_nan() && llama_acc.is_nan() {
        passed = false;
        println!("  (no side ran -- nothing to compare)");
    }
    println!("  {}", if passed { "PASS" } else { "FAIL" });

    let json = format!(
        "{{\n  \"model\": {:?},\n  \"sample\": {:?},\n  \"problems\": {},\n  \"samples_per_problem\": {samples},\n  \"gpu_accuracy\": {},\n  \"cpu_accuracy\": {},\n  \"llama_accuracy\": {},\n  \"passed\": {passed}\n}}\n",
        model_path.display().to_string(),
        sample_path.display().to_string(),
        problems.len(),
        if gpu_acc.is_nan() { "null".into() } else { format!("{gpu_acc:.4}") },
        if cpu_acc.is_nan() { "null".into() } else { format!("{cpu_acc:.4}") },
        if llama_acc.is_nan() { "null".into() } else { format!("{llama_acc:.4}") },
    );
    std::fs::write("bench-selfconsistency.json", json)
        .expect("failed to write bench-selfconsistency.json");
    println!("  Report written to bench-selfconsistency.json");

    if !passed {
        std::process::exit(1);
    }
}

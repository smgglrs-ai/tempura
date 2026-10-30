// bench-mmlu -- MMLU-style multiple-choice log-prob evaluation: quality
// benchmark (b) from the 2026-10-30 plan. For each question, prefill the
// standard harness prompt and read the log-probs of the four answer-letter
// tokens (" A".." D") from the last-position logits; the predicted answer
// is the argmax letter. Robust to argmax noise: well-separated options do
// not flip from small numerical differences, so implementations are
// compared on CHOICE agreement, not text.
//
// Sides: tempura GPU, tempura CPU, llama.cpp (batch mode of
// tests-ggml/llama_logitprobe: one prompt per line, one last-position
// logit vector per prompt).
//
// Run (laptop -> fishtank):
//   scripts/gpu-remote.sh bench mmlu Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
// Local (needs a GPU):
//   cargo run -p tempura-bench-gpu --release --bin bench-mmlu -- \
//     /path/to/model.gguf [mmlu_sample.jsonl]
//
// Env knobs:
//   TEMPURA_MMLU_MIN_AGREE  GPU-vs-CPU choice-agreement gate (default 0.85)
//   TEMPURA_NO_LLAMA=1       skip the llama.cpp reference run
//   LLAMA_PPL                llama_logitprobe binary path

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::eval::{mmlu_format_prompt, token_logprob, MMLU_LETTERS};
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{CutileInferenceWrapper, InferenceModel, TransformerModel};
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::session::InferenceSession;
use tempura_engine::tokenizer::Tokenizer;
use tempura_engine::trace::TraceLevel;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const DEFAULT_MIN_AGREE: f64 = 0.85;

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
    let cache = std::env::var("XDG_CACHE_HOME")
        .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_default()));
    Path::new(&cache).join("tempura/eval/mmlu/mmlu_sample.jsonl")
}

struct Question {
    #[allow(dead_code)]
    subject: String,
    prompt: String,
    answer: usize,
}

fn load_questions(path: &Path) -> Vec<Question> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).expect("bad jsonl line in mmlu sample");
        let subject = v["subject"].as_str().unwrap_or("unknown").to_string();
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
        let prompt = mmlu_format_prompt(&subject, question, &choices);
        out.push(Question {
            subject,
            prompt,
            answer: answer.min(3),
        });
    }
    out
}

/// Letter token ids: encode(" A") and strip a prepended BOS if present
/// (encode always adds one for add_bos vocabs; the letter continues a
/// prompt, so it must not carry another BOS).
fn letter_token_ids(tok: &Tokenizer, bos: Option<u32>) -> [usize; 4] {
    let mut ids = [0usize; 4];
    for (i, l) in MMLU_LETTERS.iter().enumerate() {
        let mut enc = tok.encode(l).expect("letter tokenization failed");
        if enc.len() > 1 && Some(enc[0]) == bos {
            enc.remove(0);
        }
        assert_eq!(
            enc.len(),
            1,
            "letter {l:?} tokenized to {enc:?} (expected a single token)"
        );
        ids[i] = enc[0] as usize;
    }
    ids
}

fn evaluate(
    model: &dyn InferenceModel,
    config: &tempura_engine::ModelConfig,
    prompts: &[Vec<u32>],
    letters: &[usize; 4],
) -> Vec<usize> {
    let mut picks = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let mut session = InferenceSession::new(config, TraceLevel::Off, 0);
        let logits = model.forward(&mut session, prompt);
        let nan = logits.iter().filter(|v| v.is_nan()).count();
        if nan > 0 {
            eprintln!("[mmlu] question {i}: {nan} NaN logits");
        }
        let lps: Vec<f64> = letters.iter().map(|&l| token_logprob(&logits, l)).collect();
        let mut best = 0;
        for (c, &lp) in lps.iter().enumerate() {
            if lp > lps[best] {
                best = c;
            }
        }
        picks.push(best);
    }
    picks
}

fn accuracy(picks: &[usize], answers: &[usize]) -> f64 {
    let n = picks.len().min(answers.len());
    if n == 0 {
        return 0.0;
    }
    let hit = (0..n).filter(|&i| picks[i] == answers[i]).count();
    hit as f64 / n as f64
}

fn agreement(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let hit = (0..n).filter(|&i| a[i] == b[i]).count();
    hit as f64 / n as f64
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = resolve_model_path(args.next());
    let sample_path = resolve_corpus_path(args.next());
    let min_agree = env_f64("TEMPURA_MMLU_MIN_AGREE", DEFAULT_MIN_AGREE);

    println!("=== tempura MMLU log-prob evaluation (GPU vs CPU vs llama.cpp) ===\n");
    println!("  Model:  {}", model_path.display());
    println!("  Sample: {}", sample_path.display());

    let questions = load_questions(&sample_path);
    println!("  Questions: {}", questions.len());
    assert!(!questions.is_empty(), "empty MMLU sample");

    println!("\nLoading GGUF...");
    let t0 = Instant::now();
    let loader_gpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF file");
    let config = loader_gpu.config.clone();
    let metadata = loader_gpu.metadata.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());

    let tokenizer =
        Tokenizer::from_gguf_metadata(&metadata).expect("MMLU needs the real tokenizer");
    let bos = tokenizer.bos_token_id();
    let letters = letter_token_ids(&tokenizer, bos);
    println!("  Letter tokens: {letters:?}");

    let prompts: Vec<Vec<u32>> = questions
        .iter()
        .map(|q| tokenizer.encode(&q.prompt).expect("prompt tokenization failed"))
        .collect();
    let avg_len = prompts.iter().map(|p| p.len()).sum::<usize>() / prompts.len();
    println!("  Prompts tokenized (avg {avg_len} tokens)");

    // Write the prompt-id file for the llama.cpp batch reference run.
    let llama_prompt_file = std::env::temp_dir().join("tempura_mmlu_prompts.txt");
    std::fs::write(
        &llama_prompt_file,
        prompts
            .iter()
            .map(|p| {
                p.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .expect("failed to write llama prompt file");

    let t1 = Instant::now();
    let qw_gpu = loader_gpu.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config.clone());
    wrapper.load_quantized(qw_gpu);
    println!("  GPU loaded in {:.2}s", t1.elapsed().as_secs_f64());

    let t2 = Instant::now();
    let loader_cpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to re-open GGUF file");
    let qw_cpu = loader_cpu.to_quantized_weights();
    let mut cpu_model = TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    cpu_model.quantized_weights = Some(qw_cpu);
    println!("  CPU ready in {:.2}s", t2.elapsed().as_secs_f64());

    let answers: Vec<usize> = questions.iter().map(|q| q.answer).collect();

    println!("\n--- tempura CPU ---");
    let cpu_picks = if std::env::var("TEMPURA_MMLU_NO_CPU").is_ok() {
        println!("  (skipped: TEMPURA_MMLU_NO_CPU=1)");
        Vec::new()
    } else {
        let t = Instant::now();
        let picks = evaluate(&cpu_model, &config, &prompts, &letters);
        println!(
            "  accuracy {:.1}%  ({:.1}s)",
            accuracy(&picks, &answers) * 100.0,
            t.elapsed().as_secs_f64()
        );
        picks
    };

    println!("\n--- tempura GPU ---");
    let t = Instant::now();
    let gpu_picks = evaluate(&wrapper, &config, &prompts, &letters);
    println!(
        "  accuracy {:.1}%  ({:.1}s)",
        accuracy(&gpu_picks, &answers) * 100.0,
        t.elapsed().as_secs_f64()
    );

    // --- llama.cpp reference ---
    let llama_picks = if std::env::var("TEMPURA_NO_LLAMA").is_err() {
        let bin = std::env::var("LLAMA_PPL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/Code/github.com/llama.cpp/build/bin/llama-perplexity")
        });
        // llama_logitprobe lives in tests-ggml/ (built on the GPU host).
        let probe = std::env::var("LLAMA_LOGITPROBE").unwrap_or_else(|_| "/tmp/llama-logitprobe".to_string());
        if Path::new(&probe).exists() {
            println!("\n--- llama.cpp (CPU) ---");
            let t = Instant::now();
            let out_file = std::env::temp_dir().join("tempura_mmlu_llama_logits.bin");
            let status = std::process::Command::new(&probe)
                .arg(&model_path)
                .arg(&llama_prompt_file)
                .arg(&out_file)
                .arg("batch")
                .env("LD_LIBRARY_PATH", {
                    let home = std::env::var("HOME").unwrap_or_default();
                    format!("{home}/Code/github.com/llama.cpp/build/bin")
                })
                .output()
                .expect("failed to run llama_logitprobe");
            if !status.status.success() {
                eprintln!("  llama_logitprobe failed: {}", String::from_utf8_lossy(&status.stderr));
                None
            } else {
                let bytes = std::fs::read(&out_file).expect("failed to read llama logits");
                let nv = config.vocab_size;
                assert!(bytes.len() >= prompts.len() * nv * 4, "llama logits file too small");
                let mut picks = Vec::with_capacity(prompts.len());
                for (i, _) in prompts.iter().enumerate() {
                    let logits: Vec<f32> = bytes[i * nv * 4..(i + 1) * nv * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let lps: Vec<f64> =
                        letters.iter().map(|&l| token_logprob(&logits, l)).collect();
                    let mut best = 0;
                    for (c, &lp) in lps.iter().enumerate() {
                        if lp > lps[best] {
                            best = c;
                        }
                    }
                    picks.push(best);
                }
                println!(
                    "  accuracy {:.1}%  ({:.1}s)",
                    accuracy(&picks, &answers) * 100.0,
                    t.elapsed().as_secs_f64()
                );
                let _ = bin; // LLAMA_PPL unused here; probe path is LLAMA_LOGITPROBE
                Some(picks)
            }
        } else {
            println!(
                "\n  (llama-logitprobe not found at {probe}; set LLAMA_LOGITPROBE to enable)"
            );
            None
        }
    } else {
        None
    };

    // --- Comparison ---
    println!("\n--- Comparison ---");
    let gpu_cpu = if cpu_picks.is_empty() {
        f64::NAN
    } else {
        agreement(&gpu_picks, &cpu_picks)
    };
    if !cpu_picks.is_empty() {
        println!("  GPU/CPU choice agreement: {:.1}%", gpu_cpu * 100.0);
    }
    if let Some(lp) = &llama_picks {
        println!("  CPU/llama.cpp agreement:   {:.1}%", agreement(&cpu_picks, lp) * 100.0);
        println!("  GPU/llama.cpp agreement:   {:.1}%", agreement(&gpu_picks, lp) * 100.0);
    }

    let passed = if cpu_picks.is_empty() {
        llama_picks
            .as_ref()
            .map(|lp| agreement(&gpu_picks, lp) >= min_agree)
            .unwrap_or(true)
    } else {
        gpu_cpu >= min_agree
    };
    println!(
        "\n--- Verdict ---\n  {} (GPU/CPU agreement {:.1}% vs gate {:.0}%)",
        if passed { "PASS" } else { "FAIL" },
        gpu_cpu * 100.0,
        min_agree * 100.0,
    );

    let json = format!(
        "{{\n  \"model\": {:?},\n  \"sample\": {:?},\n  \"questions\": {},\n  \"cpu_accuracy\": {:.4},\n  \"gpu_accuracy\": {:.4},\n  \"llama_accuracy\": {},\n  \"gpu_cpu_agreement\": {:.4},\n  \"passed\": {passed}\n}}\n",
        model_path.display().to_string(),
        sample_path.display().to_string(),
        questions.len(),
        accuracy(&cpu_picks, &answers),
        accuracy(&gpu_picks, &answers),
        llama_picks
            .as_ref()
            .map(|p| format!("{:.4}", accuracy(p, &answers)))
            .unwrap_or_else(|| "null".into()),
        gpu_cpu,
    );
    std::fs::write("bench-mmlu.json", json).expect("failed to write bench-mmlu.json");
    println!("  Report written to bench-mmlu.json");

    if !passed {
        std::process::exit(1);
    }
}

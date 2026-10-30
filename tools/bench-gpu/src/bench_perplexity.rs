// bench-perplexity -- corpus-level perplexity comparison: tempura GPU vs
// tempura CPU vs llama.cpp, on the same GGUF and the same corpus file
// (WikiText-2 raw test set by default).
//
// This is quality benchmark (a) from the 2026-10-30 plan: perplexity is the
// most implementation-robust quality metric -- no argmax, no sampling, just
// "are our logits as good as theirs?" averaged over the whole corpus.
//
// The scoring algorithm (engine/src/eval.rs) replicates llama.cpp's
// tools/perplexity default exactly: non-overlapping n_ctx chunks, BOS at
// every chunk start, positions [n_ctx/2, n_ctx-2] scored, f64 log-softmax
// NLL. llama.cpp batch-evaluates each chunk; we prefill the first half and
// decode the second (causal attention makes per-position logits identical,
// the difference is only which kernel path computes them).
//
// Run (laptop -> fishtank):
//   scripts/gpu-remote.sh bench perplexity Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
// Local (needs a GPU):
//   cargo run -p tempura-bench-gpu --release --bin bench-perplexity -- \
//     /path/to/model.gguf /path/to/wiki.test.raw
//
// Env knobs:
//   TEMPURA_PPL_CTX        chunk/context size (default 512, llama.cpp -c)
//   TEMPURA_PPL_CHUNKS     max chunks, -1 = whole corpus (default 32)
//   TEMPURA_PPL_MAX_RATIO  GPU-vs-CPU pass gate on ppl ratio (default 1.10)
//   TEMPURA_NO_LLAMA=1     skip the llama.cpp reference run
//   LLAMA_PPL              llama-perplexity binary (default: llama.cpp build)

use std::path::{Path, PathBuf};
use std::time::Instant;
use tempura_engine::eval::{perplexity, PplOptions, PplResult};
use tempura_engine::gguf_loader::GgufLoader;
use tempura_engine::model::{CutileInferenceWrapper, TransformerModel};
use tempura_engine::pipeline::KernelPipeline;
use tempura_engine::tokenizer::Tokenizer;

const DEFAULT_MODEL: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const DEFAULT_MAX_CHUNKS: i64 = 32;
const DEFAULT_CTX: usize = 512;

fn env_i64(name: &str, default: i64) -> i64 {
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

fn resolve_corpus_path(arg: Option<String>) -> PathBuf {
    if let Some(p) = arg {
        return PathBuf::from(p);
    }
    let cache = std::env::var("XDG_CACHE_HOME")
        .unwrap_or_else(|_| format!("{}/.cache", std::env::var("HOME").unwrap_or_default()));
    let default = Path::new(&cache).join("tempura/eval/wikitext-2-raw/wiki.test.raw");
    if default.exists() {
        default
    } else {
        // llama.cpp keeps a copy under its tests/ in some checkouts.
        let home = std::env::var("HOME").unwrap_or_default();
        let alt = Path::new(&home).join("Code/github.com/llama.cpp/tests/wikitext-2-raw/wiki.test.raw");
        if alt.exists() {
            alt
        } else {
            default
        }
    }
}

/// Run llama-perplexity on the same model + corpus and parse the final PPL.
fn llama_perplexity(
    bin: &str,
    model: &Path,
    corpus: &Path,
    n_ctx: usize,
    max_chunks: Option<usize>,
) -> Option<f64> {
    let mut cmd = std::process::Command::new(bin);
    cmd.arg("-m").arg(model)
        .arg("-f").arg(corpus)
        .arg("-c").arg(n_ctx.to_string())
        .arg("-ngl").arg("0") // CPU reference -- quality, not speed
        .arg("--no-warmup");
    if let Some(n) = max_chunks {
        cmd.arg("--chunks").arg(n.to_string());
    }
    let out = cmd.output().ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        eprintln!("  llama-perplexity exited with {:?}:", out.status.code());
        for line in text.lines().filter(|l| l.contains("error") || l.contains("ERR")).take(5) {
            eprintln!("    {line}");
        }
        return None;
    }
    // Modern: "Final estimate: PPL = 12.3456"; older: "perplexity: 12.3456 [114/114]"
    let mut ppl = None;
    for line in text.lines().rev() {
        if let Some(rest) = line.split("PPL =").nth(1) {
            if let Ok(v) = rest.trim().split_whitespace().next().unwrap_or("").parse::<f64>() {
                ppl = Some(v);
                break;
            }
        }
        if let Some(rest) = line.split("perplexity:").nth(1) {
            if let Ok(v) = rest.trim().split_whitespace().next().unwrap_or("").parse::<f64>() {
                ppl = Some(v);
                break;
            }
        }
    }
    ppl
}

fn report_line(name: &str, r: &PplResult) {
    println!(
        "  {name:<12} ppl {:>10.4}  (avg nll {:.4}, {} tokens, {} chunks, {} NaN steps, {:.1}s)",
        r.ppl,
        r.avg_nll(),
        r.count,
        r.chunks,
        r.nan_steps,
        r.elapsed_s
    );
}

fn main() {
    let mut args = std::env::args().skip(1).filter(|a| !a.starts_with("--"));
    let model_path = resolve_model_path(args.next());
    let corpus_path = resolve_corpus_path(args.next());
    let n_ctx = env_i64("TEMPURA_PPL_CTX", DEFAULT_CTX as i64).max(64) as usize;
    let max_chunks_raw = env_i64("TEMPURA_PPL_CHUNKS", DEFAULT_MAX_CHUNKS);
    let max_chunks = if max_chunks_raw < 0 {
        None
    } else {
        Some(max_chunks_raw as usize)
    };
    let max_ratio = env_f64("TEMPURA_PPL_MAX_RATIO", 1.10);

    println!("=== tempura perplexity comparison (GPU vs CPU vs llama.cpp) ===\n");
    println!("  Model:  {}", model_path.display());
    println!("  Corpus: {}", corpus_path.display());
    println!("  n_ctx: {n_ctx}, max chunks: {}", match max_chunks {
        Some(n) => n.to_string(),
        None => "all".to_string(),
    });

    let corpus = std::fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read corpus {}: {e}", corpus_path.display()));

    // --- Load GGUF (GPU side) ---
    println!("\nLoading GGUF...");
    let t0 = Instant::now();
    let loader_gpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to open GGUF file");
    let config = loader_gpu.config.clone();
    let metadata = loader_gpu.metadata.clone();
    println!("  Parsed in {:.2}s", t0.elapsed().as_secs_f64());

    let tokenizer = Tokenizer::from_gguf_metadata(&metadata)
        .expect("tokenizer extraction failed (perplexity needs the real tokenizer)");
    let bos = tokenizer.bos_token_id();
    println!("  Tokenizer: ok (bos={bos:?})");

    let t1 = Instant::now();
    let corpus_tokens = tokenizer.encode(&corpus).expect("corpus tokenization failed");
    println!(
        "  Corpus: {} chars -> {} tokens in {:.1}s ({} full chunks of {n_ctx})",
        corpus.len(),
        corpus_tokens.len(),
        t1.elapsed().as_secs_f64(),
        corpus_tokens.len() / n_ctx,
    );

    let opts = PplOptions {
        n_ctx,
        max_chunks,
        bos,
    };

    // --- GPU model ---
    println!("\nUploading weights to GPU...");
    let t2 = Instant::now();
    let qw_gpu = loader_gpu.to_quantized_weights();
    let mut wrapper = CutileInferenceWrapper::new(config.clone());
    wrapper.load_quantized(qw_gpu);
    println!("  Loaded in {:.2}s", t2.elapsed().as_secs_f64());

    // --- CPU reference model ---
    println!("Building CPU reference model...");
    let t3 = Instant::now();
    let loader_cpu =
        GgufLoader::from_file_quantized(&model_path).expect("failed to re-open GGUF file");
    let qw_cpu = loader_cpu.to_quantized_weights();
    let mut cpu_model =
        TransformerModel::new(config.clone(), KernelPipeline::cpu_native());
    cpu_model.quantized_weights = Some(qw_cpu);
    println!("  Ready in {:.2}s", t3.elapsed().as_secs_f64());

    // --- Perplexity on both paths ---
    println!("\n--- tempura CPU ---");
    let cpu_res = perplexity(&cpu_model, &config, &corpus_tokens, &opts)
        .expect("corpus too short for one chunk");
    report_line("tempura CPU", &cpu_res);

    println!("\n--- tempura GPU ---");
    let gpu_res = perplexity(&wrapper, &config, &corpus_tokens, &opts)
        .expect("corpus too short for one chunk");
    report_line("tempura GPU", &gpu_res);

    // --- llama.cpp reference ---
    let llama_ppl = if std::env::var("TEMPURA_NO_LLAMA").is_err() {
        let bin = std::env::var("LLAMA_PPL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/Code/github.com/llama.cpp/build/bin/llama-perplexity")
        });
        if Path::new(&bin).exists() {
            println!("\n--- llama.cpp (CPU) ---");
            print!("  running llama-perplexity...");
            use std::io::Write;
            std::io::stdout().flush().ok();
            match llama_perplexity(&bin, &model_path, &corpus_path, n_ctx, max_chunks) {
                Some(p) => {
                    println!(" ppl {p:.4}");
                    Some(p)
                }
                None => {
                    println!(" (failed to parse output)");
                    None
                }
            }
        } else {
            println!("\n  (llama-perplexity not found at {bin}; set LLAMA_PPL to enable)");
            None
        }
    } else {
        None
    };

    // --- Comparison ---
    let gpu_cpu_ratio = gpu_res.ppl / cpu_res.ppl;
    println!("\n--- Comparison ---");
    println!(
        "  tempura GPU / tempura CPU: {:.4}x  (1.00x = identical logits quality)",
        gpu_cpu_ratio
    );
    if let (Some(lp), cr) = (llama_ppl, cpu_res.ppl) {
        println!("  tempura CPU / llama.cpp:    {:.4}x", cr / lp);
        println!("  tempura GPU / llama.cpp:    {:.4}x", gpu_res.ppl / lp);
    }

    let passed = gpu_res.nan_steps == 0
        && cpu_res.nan_steps == 0
        && gpu_cpu_ratio <= max_ratio
        && gpu_cpu_ratio >= 1.0 / max_ratio;
    println!(
        "\n--- Verdict ---\n  {} (GPU/CPU ratio {:.4} vs gate {:.2}, NaN steps: gpu {} cpu {})",
        if passed { "PASS" } else { "FAIL" },
        gpu_cpu_ratio,
        max_ratio,
        gpu_res.nan_steps,
        cpu_res.nan_steps,
    );

    let json = format!(
        "{{\n  \"model\": {:?},\n  \"corpus\": {:?},\n  \"n_ctx\": {n_ctx},\n  \"max_chunks\": {},\n  \"cpu\": {{\"ppl\": {:.6}, \"nll\": {:.6}, \"count\": {}}},\n  \"gpu\": {{\"ppl\": {:.6}, \"nll\": {:.6}, \"count\": {}}},\n  \"llama_cpp_ppl\": {},\n  \"gpu_cpu_ratio\": {:.6},\n  \"gpu_nan_steps\": {},\n  \"passed\": {passed}\n}}\n",
        model_path.display().to_string(),
        corpus_path.display().to_string(),
        max_chunks.map(|n| n.to_string()).unwrap_or_else(|| "null".into()),
        cpu_res.ppl, cpu_res.nll, cpu_res.count,
        gpu_res.ppl, gpu_res.nll, gpu_res.count,
        llama_ppl.map(|v| format!("{v:.6}")).unwrap_or_else(|| "null".into()),
        gpu_cpu_ratio,
        gpu_res.nan_steps,
    );
    std::fs::write("bench-perplexity.json", json)
        .expect("failed to write bench-perplexity.json");
    println!("  Report written to bench-perplexity.json");

    if !passed {
        std::process::exit(1);
    }
}

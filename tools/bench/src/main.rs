use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "tempura-bench", about = "Throughput benchmark for tempura server")]
struct Args {
    #[arg(long, default_value = "http://localhost:8080")]
    host: String,

    #[arg(long, default_value = "default")]
    model: String,

    #[arg(long, default_value = "128,512", value_delimiter = ',')]
    pp: Vec<usize>,

    #[arg(long, default_value = "64,128", value_delimiter = ',')]
    tg: Vec<usize>,

    #[arg(long, default_value_t = 3)]
    runs: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long)]
    output: Option<String>,

    #[arg(long)]
    compare: Option<String>,

    #[arg(long, default_value_t = 5.0)]
    threshold: f64,

    #[arg(long, default_value = "table")]
    format: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct RunResult {
    tokens_per_sec: f64,
    elapsed_ms: u64,
    tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct BenchResult {
    label: String,
    pp: Option<usize>,
    tg: Option<usize>,
    runs: Vec<RunResult>,
    mean: f64,
    stddev: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchOutput {
    timestamp: String,
    host: String,
    model: String,
    results: Vec<BenchResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    seed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

fn synthetic_prompt(target_tokens: usize, seed: u64) -> String {
    let words = ["hello", "world", "foo", "bar", "baz", "qux", "the", "quick", "brown", "fox"];
    let chars_needed = (target_tokens as f64 / 0.75) as usize;
    let mut s = String::with_capacity(chars_needed + 32);
    let mut idx = seed as usize % words.len();
    while s.len() < chars_needed {
        s.push_str(words[idx % words.len()]);
        s.push(' ');
        idx = idx.wrapping_add(1);
    }
    s.truncate(chars_needed);
    s
}

fn stats(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance.sqrt())
}

async fn bench_pp(client: &Client, host: &str, model: &str, pp: usize, runs: usize, warmup: usize, seed: u64) -> BenchResult {
    let prompt = synthetic_prompt(pp, seed);
    let req = ChatRequest {
        model: model.to_string(),
        messages: vec![Message { role: "user".into(), content: prompt }],
        max_tokens: 1,
        temperature: 0.0,
        stream: false,
        seed,
    };

    let url = format!("{host}/v1/chat/completions");
    let mut all_runs = Vec::new();

    for i in 0..(warmup + runs) {
        let start = Instant::now();
        let res = client.post(&url).json(&req).send().await;
        let elapsed = start.elapsed();

        match res {
            Ok(r) if r.status().is_success() => {
                if i >= warmup {
                    let elapsed_ms = elapsed.as_millis() as u64;
                    let tokens_per_sec = pp as f64 / elapsed.as_secs_f64();
                    eprintln!("  pp={pp} run {}/{runs}: {tokens_per_sec:.1} tok/s", i - warmup + 1);
                    all_runs.push(RunResult { tokens_per_sec, elapsed_ms, tokens: pp });
                }
            }
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                eprintln!("  pp={pp} run {i}: HTTP {status}: {body}");
            }
            Err(e) => eprintln!("  pp={pp} run {i}: {e}"),
        }
    }

    let tps: Vec<f64> = all_runs.iter().map(|r| r.tokens_per_sec).collect();
    let (mean, stddev) = if tps.is_empty() { (0.0, 0.0) } else { stats(&tps) };
    BenchResult { label: format!("pp{pp}"), pp: Some(pp), tg: None, runs: all_runs, mean, stddev }
}

async fn bench_tg(client: &Client, host: &str, model: &str, tg: usize, runs: usize, warmup: usize, seed: u64) -> BenchResult {
    let req = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
        "max_tokens": tg,
        "temperature": 0.0,
        "stream": true,
        "seed": seed,
    });

    let url = format!("{host}/v1/chat/completions");
    let mut all_runs = Vec::new();

    for i in 0..(warmup + runs) {
        let start = Instant::now();
        let res = client.post(&url).json(&req).send().await;

        match res {
            Ok(r) if r.status().is_success() => {
                let mut first_token: Option<Instant> = None;
                let mut last_token: Option<Instant> = None;
                let mut token_count = 0usize;

                let bytes = r.bytes().await.unwrap_or_default();
                for line in String::from_utf8_lossy(&bytes).lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" { break; }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            let content = v["choices"][0]["delta"]["content"].as_str().unwrap_or("");
                            if !content.is_empty() {
                                let now = Instant::now();
                                if first_token.is_none() { first_token = Some(now); }
                                last_token = Some(now);
                                token_count += 1;
                            }
                        }
                    }
                }

                if i >= warmup {
                    if let (Some(first), Some(last)) = (first_token, last_token) {
                        let gen_secs = (last - first).as_secs_f64().max(1e-6);
                        let tokens_per_sec = token_count as f64 / gen_secs;
                        let elapsed_ms = start.elapsed().as_millis() as u64;
                        eprintln!("  tg={tg} run {}/{runs}: {tokens_per_sec:.1} tok/s ({token_count} tokens)", i - warmup + 1);
                        all_runs.push(RunResult { tokens_per_sec, elapsed_ms, tokens: token_count });
                    } else {
                        eprintln!("  tg={tg} run {i}: no tokens generated");
                    }
                }
            }
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                eprintln!("  tg={tg} run {i}: HTTP {status}: {body}");
            }
            Err(e) => eprintln!("  tg={tg} run {i}: {e}"),
        }
    }

    let tps: Vec<f64> = all_runs.iter().map(|r| r.tokens_per_sec).collect();
    let (mean, stddev) = if tps.is_empty() { (0.0, 0.0) } else { stats(&tps) };
    BenchResult { label: format!("tg{tg}"), pp: None, tg: Some(tg), runs: all_runs, mean, stddev }
}

fn format_table(results: &[BenchResult], model: &str) -> String {
    if results.is_empty() { return String::new(); }
    let mut header = format!("| {:<20} |", "model");
    let mut sep = format!("| {:<20} |", "-".repeat(20));
    let mut row = format!("| {:<20} |", model);
    for r in results {
        header.push_str(&format!(" {:>8} |", r.label));
        sep.push_str(&format!(" {:>8} |", "-".repeat(8)));
        if r.mean > 0.0 {
            row.push_str(&format!(" {:>8.1} |", r.mean));
        } else {
            row.push_str(&format!(" {:>8} |", "n/a"));
        }
    }
    format!("{header}\n{sep}\n{row}\n")
}

fn format_markdown(results: &[BenchResult], model: &str) -> String {
    format_table(results, model)
}

struct Regression {
    label: String,
    baseline: f64,
    current: f64,
    pct: f64,
}

fn compare_to_baseline(current: &BenchOutput, baseline: &BenchOutput, threshold: f64) -> Vec<Regression> {
    let mut regressions = Vec::new();
    for cr in &current.results {
        if let Some(br) = baseline.results.iter().find(|r| r.label == cr.label) {
            if br.mean > 0.0 && cr.mean > 0.0 {
                let pct = (cr.mean - br.mean) / br.mean * 100.0;
                if pct < -threshold {
                    regressions.push(Regression {
                        label: cr.label.clone(),
                        baseline: br.mean,
                        current: cr.mean,
                        pct,
                    });
                }
            }
        }
    }
    regressions
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("failed to build HTTP client");

    // Verify server reachable
    match client.get(format!("{}/health", args.host)).send().await {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => {
            eprintln!("Server unhealthy: HTTP {}", r.status());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Cannot reach server at {}: {e}", args.host);
            std::process::exit(1);
        }
    }

    let mut results: Vec<BenchResult> = Vec::new();

    for &pp in &args.pp {
        eprintln!("Benchmarking pp={pp}...");
        let r = bench_pp(&client, &args.host, &args.model, pp, args.runs, args.warmup, args.seed).await;
        results.push(r);
    }

    for &tg in &args.tg {
        eprintln!("Benchmarking tg={tg}...");
        let r = bench_tg(&client, &args.host, &args.model, tg, args.runs, args.warmup, args.seed).await;
        results.push(r);
    }

    let output = BenchOutput {
        timestamp: chrono::Utc::now().to_rfc3339(),
        host: args.host.clone(),
        model: args.model.clone(),
        results: results.clone(),
    };

    match args.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&output).unwrap()),
        "markdown" => print!("{}", format_markdown(&results, &args.model)),
        _ => print!("{}", format_table(&results, &args.model)),
    }

    if let Some(path) = &args.output {
        std::fs::write(path, serde_json::to_string_pretty(&output).unwrap())
            .expect("failed to write output file");
        eprintln!("Results written to {path}");
    }

    if let Some(baseline_path) = &args.compare {
        let baseline_json = std::fs::read_to_string(baseline_path)
            .expect("failed to read baseline file");
        let baseline: BenchOutput = serde_json::from_str(&baseline_json)
            .expect("failed to parse baseline JSON");

        let regressions = compare_to_baseline(&output, &baseline, args.threshold);
        if regressions.is_empty() {
            eprintln!("No regressions detected (threshold: {:.1}%)", args.threshold);
        } else {
            eprintln!("REGRESSIONS DETECTED:");
            for r in &regressions {
                eprintln!(
                    "  {}: {:.1} → {:.1} tok/s ({:+.1}%)",
                    r.label, r.baseline, r.current, r.pct
                );
            }
            std::process::exit(1);
        }
    }
}

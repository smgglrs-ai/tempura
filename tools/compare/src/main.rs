use clap::{Parser, ValueEnum};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const BUILTIN_PROMPTS: &[&str] = &[
    "The quick brown fox",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
    "What is 2+2?",
    "Explain quantum entanglement briefly.",
    "In 1969, Neil Armstrong",
    "The meaning of life is",
    "SELECT * FROM users WHERE",
    "<system>You are helpful.</system><user>Hi</user><assistant>",
    "日本語のテキスト：",
];

#[derive(Clone, ValueEnum, Debug)]
enum Mode {
    Generate,
    Tokenize,
    Logits,
}

#[derive(Parser, Debug)]
#[command(
    name = "tempura-compare",
    about = "Compare outputs of two inference servers"
)]
struct Args {
    #[arg(long)]
    server_a: String,
    #[arg(long)]
    server_b: String,
    #[arg(long, default_value = "tempura")]
    label_a: String,
    #[arg(long, default_value = "reference")]
    label_b: String,
    #[arg(long, default_value = "default")]
    model: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    tokens: u32,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value = "generate")]
    mode: Mode,
    #[arg(long, default_value_t = 5e-4)]
    tolerance: f64,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    prompts_builtin: bool,
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Status {
    Match,
    Diverge,
    Error,
}

#[derive(Debug, Serialize)]
struct PromptResult {
    prompt: String,
    status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    a: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_diff_char: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct LogitsResponse {
    logits: Vec<f32>,
}

async fn check_server(client: &Client, url: &str, label: &str) -> bool {
    match client.get(format!("{url}/health")).send().await {
        Ok(r) if r.status().is_success() => true,
        Ok(r) => {
            eprintln!("  {label}: /health returned {}", r.status());
            false
        }
        Err(e) => {
            eprintln!("  {label}: connection failed — {e}");
            false
        }
    }
}

async fn compare_generate(client: &Client, args: &Args, prompt: &str) -> PromptResult {
    let body = serde_json::json!({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": false,
    });

    async fn fetch(client: &Client, url: &str, body: &serde_json::Value) -> Result<String, String> {
        let resp = client
            .post(format!("{url}/v1/chat/completions"))
            .json(body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let cr: ChatResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(cr
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default())
    }

    let (ra, rb) = tokio::join!(
        fetch(client, &args.server_a, &body),
        fetch(client, &args.server_b, &body),
    );

    match (ra, rb) {
        (Err(e), _) | (_, Err(e)) => PromptResult {
            prompt: prompt.to_string(),
            status: Status::Error,
            a: None,
            b: None,
            first_diff_char: None,
            max_abs_error: None,
            error: Some(e),
        },
        (Ok(a), Ok(b)) => {
            let first_diff = a
                .chars()
                .zip(b.chars())
                .position(|(ca, cb)| ca != cb)
                .or_else(|| {
                    if a.len() != b.len() {
                        Some(a.len().min(b.len()))
                    } else {
                        None
                    }
                });
            PromptResult {
                prompt: prompt.to_string(),
                status: if first_diff.is_none() {
                    Status::Match
                } else {
                    Status::Diverge
                },
                a: Some(a),
                b: Some(b),
                first_diff_char: first_diff,
                max_abs_error: None,
                error: None,
            }
        }
    }
}

async fn compare_tokenize(client: &Client, args: &Args, prompt: &str) -> PromptResult {
    let body = serde_json::json!({"model": args.model, "content": prompt});

    async fn fetch(
        client: &Client,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<Vec<u32>, String> {
        let resp = client
            .post(format!("{url}/tokenize"))
            .json(body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let tr: TokenizeResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(tr.tokens)
    }

    let (ra, rb) = tokio::join!(
        fetch(client, &args.server_a, &body),
        fetch(client, &args.server_b, &body),
    );

    match (ra, rb) {
        (Err(e), _) | (_, Err(e)) => PromptResult {
            prompt: prompt.to_string(),
            status: Status::Error,
            a: None,
            b: None,
            first_diff_char: None,
            max_abs_error: None,
            error: Some(e),
        },
        (Ok(a), Ok(b)) => {
            let first_diff = a
                .iter()
                .zip(b.iter())
                .position(|(ta, tb)| ta != tb)
                .or_else(|| {
                    if a.len() != b.len() {
                        Some(a.len().min(b.len()))
                    } else {
                        None
                    }
                });
            PromptResult {
                prompt: prompt.to_string(),
                status: if first_diff.is_none() {
                    Status::Match
                } else {
                    Status::Diverge
                },
                a: Some(format!("{a:?}")),
                b: Some(format!("{b:?}")),
                first_diff_char: first_diff,
                max_abs_error: None,
                error: None,
            }
        }
    }
}

async fn compare_logits(
    client: &Client,
    args: &Args,
    prompt: &str,
    tolerance: f64,
) -> PromptResult {
    let body_tok = serde_json::json!({"model": args.model, "content": prompt});

    let tokens_resp = client
        .post(format!("{}/tokenize", args.server_a))
        .json(&body_tok)
        .send()
        .await;

    let tokens = match tokens_resp {
        Err(e) => {
            return PromptResult {
                prompt: prompt.to_string(),
                status: Status::Error,
                a: None,
                b: None,
                first_diff_char: None,
                max_abs_error: None,
                error: Some(format!("tokenize failed: {e}")),
            }
        }
        Ok(r) => match r.json::<TokenizeResponse>().await {
            Err(e) => {
                return PromptResult {
                    prompt: prompt.to_string(),
                    status: Status::Error,
                    a: None,
                    b: None,
                    first_diff_char: None,
                    max_abs_error: None,
                    error: Some(format!("tokenize parse: {e}")),
                }
            }
            Ok(t) => t.tokens,
        },
    };

    let logits_body = serde_json::json!({"model": args.model, "tokens": tokens});

    async fn fetch_logits(
        client: &Client,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<Vec<f32>, String> {
        let resp = client
            .post(format!("{url}/tempura/v1/debug/logits"))
            .json(body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!(
                "HTTP {} (endpoint may not be supported)",
                resp.status()
            ));
        }
        let lr: LogitsResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(lr.logits)
    }

    let (ra, rb) = tokio::join!(
        fetch_logits(client, &args.server_a, &logits_body),
        fetch_logits(client, &args.server_b, &logits_body),
    );

    match (ra, rb) {
        (Err(e), _) | (_, Err(e)) => PromptResult {
            prompt: prompt.to_string(),
            status: Status::Error,
            a: None,
            b: None,
            first_diff_char: None,
            max_abs_error: None,
            error: Some(e),
        },
        (Ok(a), Ok(b)) => {
            let max_err = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() as f64)
                .fold(0f64, f64::max);

            let mut diffs: Vec<(usize, f64)> = a
                .iter()
                .zip(b.iter())
                .enumerate()
                .map(|(i, (x, y))| (i, (x - y).abs() as f64))
                .collect();
            diffs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
            diffs.truncate(5);

            let summary_a = format!("len={}, top5_diffs={diffs:?}", a.len());
            let summary_b = format!("max_abs_err={max_err:.2e}, pass={}", max_err <= tolerance);

            PromptResult {
                prompt: prompt.to_string(),
                status: if max_err <= tolerance {
                    Status::Match
                } else {
                    Status::Diverge
                },
                a: Some(summary_a),
                b: Some(summary_b),
                first_diff_char: None,
                max_abs_error: Some(max_err),
                error: None,
            }
        }
    }
}

fn print_divergence(result: &PromptResult, index: usize, label_a: &str, label_b: &str) {
    let a = result.a.as_deref().unwrap_or("");
    let b = result.b.as_deref().unwrap_or("");
    let preview = |s: &str| {
        let s = s.replace('\n', "↵");
        if s.len() > 60 {
            format!("{}…", &s[..60])
        } else {
            s.to_string()
        }
    };
    println!("  [{index}] {:?} — diverged", &result.prompt);
    if let Some(pos) = result.first_diff_char {
        println!("    {label_a:12}: \"{}\"", preview(a));
        println!("    {label_b:12}: \"{}\"", preview(b));
        let marker_pos = pos.min(60);
        println!("    {:12}   {}^", "", " ".repeat(marker_pos));
    } else if let Some(err) = result.max_abs_error {
        println!("    max_abs_error: {err:.2e}");
        println!("    {label_a:12}: {}", preview(a));
        println!("    {label_b:12}: {}", preview(b));
    } else if let Some(e) = &result.error {
        println!("    error: {e}");
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();

    println!("Checking server connectivity…");
    let ok_a = check_server(&client, &args.server_a, &args.label_a).await;
    let ok_b = check_server(&client, &args.server_b, &args.label_b).await;
    if !ok_a || !ok_b {
        eprintln!("Both servers must be reachable. Aborting.");
        std::process::exit(2);
    }
    println!("Both servers reachable.\n");

    let prompts: Vec<String> = if let Some(ref text) = args.prompt {
        vec![text.clone()]
    } else if let Some(ref path) = args.prompt_file {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| {
                eprintln!("Cannot read {}: {e}", path.display());
                std::process::exit(2);
            })
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(String::from)
            .collect()
    } else if args.prompts_builtin || args.prompt.is_none() {
        BUILTIN_PROMPTS.iter().map(|s| s.to_string()).collect()
    } else {
        BUILTIN_PROMPTS.iter().map(|s| s.to_string()).collect()
    };

    println!("tempura-compare: {} vs {}", args.label_a, args.label_b);
    println!("{}", "=".repeat(50));
    println!("Prompts: {}", prompts.len());
    println!(
        "Mode: {:?}, tokens: {}, temperature: {}",
        args.mode, args.tokens, args.temperature
    );
    println!();

    let mut results: Vec<PromptResult> = Vec::new();
    let mut diverged = false;

    for (i, prompt) in prompts.iter().enumerate() {
        let result = match args.mode {
            Mode::Generate => compare_generate(&client, &args, prompt).await,
            Mode::Tokenize => compare_tokenize(&client, &args, prompt).await,
            Mode::Logits => compare_logits(&client, &args, prompt, args.tolerance).await,
        };

        match result.status {
            Status::Match => println!("✓  [{i}] {:?}", &prompt[..prompt.len().min(50)]),
            Status::Diverge => {
                println!("✗  [{i}] {:?}", &prompt[..prompt.len().min(50)]);
                diverged = true;
            }
            Status::Error => {
                println!(
                    "!  [{i}] {:?} — error: {}",
                    &prompt[..prompt.len().min(50)],
                    result.error.as_deref().unwrap_or("unknown")
                );
                diverged = true;
            }
        }

        results.push(result);

        if args.fail_fast && diverged {
            break;
        }
    }

    let matches = results
        .iter()
        .filter(|r| matches!(r.status, Status::Match))
        .count();
    let total = results.len();

    println!();
    println!("✓  {matches}/{total} prompts match exactly");
    if diverged {
        let n_div = total - matches;
        println!("✗  {n_div}/{total} prompts diverge");
        println!();
        println!("Divergences:");
        for (i, r) in results.iter().enumerate() {
            if !matches!(r.status, Status::Match) {
                print_divergence(r, i, &args.label_a, &args.label_b);
            }
        }
    }

    if let Some(ref out) = args.output {
        let json = serde_json::to_string_pretty(&results).unwrap();
        std::fs::write(out, json).unwrap_or_else(|e| eprintln!("Write error: {e}"));
        println!("\nReport written to {}", out.display());
    }

    std::process::exit(if diverged { 1 } else { 0 });
}

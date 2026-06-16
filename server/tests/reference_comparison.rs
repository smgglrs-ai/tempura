use std::process::Command;

fn ollama_available() -> bool {
    Command::new("ollama")
        .arg("list")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn oxide_available() -> bool {
    std::net::TcpStream::connect("127.0.0.1:8090").is_ok()
}

fn ollama_generate(prompt: &str, max_tokens: u32) -> Option<String> {
    let body = serde_json::json!({
        "model": "tinyllama",
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "seed": 42
        }
    });

    let output = Command::new("curl")
        .args([
            "-s",
            "http://localhost:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            &body.to_string(),
        ])
        .output()
        .ok()?;

    let resp: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    resp["response"].as_str().map(|s| s.to_string())
}

fn oxide_generate(prompt: &str, max_tokens: u32) -> Option<String> {
    let body = serde_json::json!({
        "model": "tinyllama",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42
    });

    let output = Command::new("curl")
        .args([
            "-s",
            "http://localhost:8090/v1/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            &body.to_string(),
        ])
        .output()
        .ok()?;

    let resp: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    resp["choices"][0]["text"].as_str().map(|s| s.to_string())
}

#[test]
#[ignore]
fn test_oxide_vs_llama_cpp_reference() {
    if !ollama_available() {
        eprintln!("SKIP: ollama not available (run `ollama serve` and `ollama pull tinyllama`)");
        return;
    }
    if !oxide_available() {
        eprintln!(
            "SKIP: oxide-serve not running on :8090 (run with --model tinyllama --port 8090)"
        );
        return;
    }

    let prompts = ["The capital of France is", "1 + 1 =", "Once upon a time"];

    let expected_keywords = [vec!["paris"], vec!["2"], vec!["a", "the", "there", "in"]];

    for (prompt, keywords) in prompts.iter().zip(expected_keywords.iter()) {
        let ollama_out = ollama_generate(prompt, 15).expect("ollama generate failed");
        let oxide_out = oxide_generate(prompt, 15).expect("oxide generate failed");

        eprintln!("Prompt: {prompt:?}");
        eprintln!("  llama.cpp: {ollama_out:?}");
        eprintln!("  oxide:     {oxide_out:?}");

        // Both outputs should be non-empty
        assert!(
            !oxide_out.trim().is_empty(),
            "oxide produced empty output for {prompt:?}"
        );
        assert!(
            !ollama_out.trim().is_empty(),
            "ollama produced empty output for {prompt:?}"
        );

        // Both should contain at least one expected keyword (semantic correctness)
        let oxide_lower = oxide_out.to_lowercase();
        let ollama_lower = ollama_out.to_lowercase();

        let oxide_has_keyword = keywords.iter().any(|kw| oxide_lower.contains(kw));
        let ollama_has_keyword = keywords.iter().any(|kw| ollama_lower.contains(kw));

        assert!(
            ollama_has_keyword,
            "llama.cpp missing keywords {keywords:?} for {prompt:?}: {ollama_out:?}"
        );
        assert!(
            oxide_has_keyword,
            "oxide missing keywords {keywords:?} for {prompt:?}: {oxide_out:?}"
        );
    }
}

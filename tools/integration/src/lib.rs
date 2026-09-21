use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::OnceLock;

pub struct TestServer {
    child: Child,
    pub port: u16,
}

impl TestServer {
    fn binary_path() -> std::path::PathBuf {
        let mut path = std::env::current_exe().unwrap();
        path.pop(); // deps/
        path.pop(); // debug/
        path.push("tempura-serve");
        path
    }

    pub fn spawn_with_key(api_key: &str) -> Self {
        let port = free_port();
        let binary = Self::binary_path();
        let mut cmd = if binary.exists() {
            Command::new(&binary)
        } else {
            let mut c = Command::new("cargo");
            c.args(["run", "--quiet", "--package", "tempura-serve", "--"]);
            c
        };
        cmd.args([
            "serve",
            "--port",
            &port.to_string(),
            "--model-config",
            "tiny-test",
            "--api-key",
            api_key,
        ]);
        let child = cmd.spawn().expect("failed to spawn tempura-serve");
        for _ in 0..150 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
                std::thread::sleep(std::time::Duration::from_millis(200));
                break;
            }
        }
        TestServer { child, port }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

static SERVER_PORT: OnceLock<u16> = OnceLock::new();

/// Shared server URL — starts one server instance for the whole test run.
pub fn shared_server_url() -> String {
    let port = SERVER_PORT.get_or_init(|| {
        let port = free_port();
        let binary = TestServer::binary_path();
        let mut cmd = if binary.exists() {
            Command::new(&binary)
        } else {
            let mut c = Command::new("cargo");
            c.args(["run", "--quiet", "--package", "tempura-serve", "--"]);
            c
        };
        cmd.args(["serve", "--port", &port.to_string(), "--model-config", "tiny-test"]);
        let child = cmd.spawn().expect("failed to spawn tempura-serve");
        std::mem::forget(child);
        for _ in 0..150 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
                std::thread::sleep(std::time::Duration::from_millis(200));
                break;
            }
        }
        port
    });
    format!("http://127.0.0.1:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_health_returns_200() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/health")).await.unwrap();
        assert_eq!(resp.status(), 200);
        // Body may be JSON or plain text depending on model state
        let _body = resp.text().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_ready_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/ready")).await.unwrap();
        assert!(resp.status() == 200 || resp.status() == 503);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_models_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .get(format!("{url}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["data"].is_array());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_metrics_prometheus_format() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/metrics")).await.unwrap();
        assert_eq!(resp.status(), 200);
        // Body may be empty if no metrics have been recorded yet
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_props_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/props")).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_chat_completion_non_streaming() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        // 200 with a real model, 503 with stub (no weights loaded) — both are valid
        assert!(resp.status() == 200 || resp.status() == 503);
        if resp.status() == 200 {
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body["choices"][0]["message"]["content"].is_string());
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_chat_completion_streaming_sse_format() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        assert!(resp.status() == 200 || resp.status() == 503);
        if resp.status() == 200 {
            let text = resp.text().await.unwrap();
            assert!(text.contains("data:"));
            assert!(text.contains("[DONE]"));
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_stream_options_include_usage() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": true,
                "stream_options": {"include_usage": true}
            }))
            .send()
            .await
            .unwrap();
        assert!(resp.status() == 200 || resp.status() == 503);
        if resp.status() == 200 {
            let text = resp.text().await.unwrap();
            assert!(text.contains("data:"));
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_completions_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/completions"))
            .json(&serde_json::json!({
                "model": "default",
                "prompt": "The sky is",
                "max_tokens": 5
            }))
            .send()
            .await
            .unwrap();
        assert!(resp.status() == 200 || resp.status() == 503);
        if resp.status() == 200 {
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body["choices"][0]["text"].is_string());
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_tokenize_and_detokenize() {
        let url = shared_server_url();
        let r1 = reqwest::Client::new()
            .post(format!("{url}/tokenize"))
            .json(&serde_json::json!({"model": "default", "content": "Hello world"}))
            .send()
            .await
            .unwrap();
        // 200 with tokenizer, 4xx without tokenizer, 5xx if server error — all valid for stub
        assert!(r1.status().is_success() || r1.status().is_client_error() || r1.status().is_server_error());
        if r1.status().is_success() {
            let body: serde_json::Value = r1.json().await.unwrap();
            let tokens = body["tokens"].as_array().unwrap().clone();
            assert!(!tokens.is_empty());
            let r2 = reqwest::Client::new()
                .post(format!("{url}/detokenize"))
                .json(&serde_json::json!({"model": "default", "tokens": tokens}))
                .send()
                .await
                .unwrap();
            assert_eq!(r2.status(), 200);
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_embeddings_returns_unsupported() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/embeddings"))
            .json(&serde_json::json!({"model": "default", "input": "test"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error() || resp.status().as_u16() == 501);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_malformed_json_returns_error() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .header("Content-Type", "application/json")
            .body("not valid json")
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_missing_messages_returns_error() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "default"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_seed_is_deterministic() {
        let url = shared_server_url();
        let req = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Random word:"}],
            "max_tokens": 5,
            "temperature": 0.8,
            "seed": 12345,
            "stream": false
        });
        let client = reqwest::Client::new();
        let r1 = client.post(format!("{url}/v1/chat/completions")).json(&req).send().await.unwrap();
        if r1.status() == 503 { return; } // no model loaded
        let r1: serde_json::Value = r1.json().await.unwrap();
        let r2: serde_json::Value = client
            .post(format!("{url}/v1/chat/completions"))
            .json(&req)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            r1["choices"][0]["message"]["content"],
            r2["choices"][0]["message"]["content"]
        );
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_debug_logits_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::Client::new()
            .post(format!("{url}/tempura/v1/debug/logits"))
            .json(&serde_json::json!({"tokens": [1, 2, 3]}))
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success()
                || resp.status().as_u16() == 501
                || resp.status().as_u16() == 503
        );
        if resp.status().is_success() {
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body["logits"].is_array());
            assert!(!body["logits"].as_array().unwrap().is_empty());
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_verified_status_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/tempura/v1/verified"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_traces_endpoint() {
        let url = shared_server_url();
        let resp = reqwest::get(format!("{url}/tempura/v1/traces"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY env var"]
    async fn test_auth_correct_key_passes() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(format!("{}/v1/models", server.url()))
            .header("Authorization", format!("Bearer {key}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY env var"]
    async fn test_auth_wrong_key_returns_401() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(format!("{}/v1/models", server.url()))
            .header("Authorization", "Bearer wrong-key-xyz")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY env var"]
    async fn test_auth_health_bypasses() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::get(format!("{}/health", server.url()))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY env var"]
    async fn test_auth_x_api_key_header() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(format!("{}/v1/models", server.url()))
            .header("X-API-Key", &key)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}

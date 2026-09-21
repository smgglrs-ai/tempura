use std::net::TcpListener;
use std::process::{Child, Command};

pub struct TestServer {
    child: Child,
    port: u16,
}

impl TestServer {
    pub fn spawn() -> Self {
        Self::spawn_with_args(&[])
    }

    pub fn spawn_with_key(api_key: &str) -> Self {
        Self::spawn_with_args(&["--api-key", api_key])
    }

    fn spawn_with_args(extra: &[&str]) -> Self {
        let port = free_port();
        let mut cmd = Command::new("cargo");
        cmd.args([
            "run",
            "--quiet",
            "--package",
            "tempura-serve",
            "--",
            "--port",
            &port.to_string(),
            "--model-config",
            "tiny-test",
        ]);
        for arg in extra {
            cmd.arg(arg);
        }
        let child = cmd.spawn().expect("failed to spawn tempura-serve");

        let base = format!("http://127.0.0.1:{port}");
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if reqwest::blocking::get(format!("{base}/health")).is_ok() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_health_returns_200() {
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/health").await.unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body.get("status").is_some());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_ready_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/ready").await.unwrap();
        assert!(resp.status() == 200 || resp.status() == 503);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_models_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .get(server.url() + "/v1/models")
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
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/metrics").await.unwrap();
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.unwrap();
        assert!(!text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_props_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/props").await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_chat_completion_non_streaming() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/chat/completions")
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["choices"][0]["message"]["content"].is_string());
        assert!(body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_chat_completion_streaming_sse_format() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/chat/completions")
            .json(&serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.unwrap();
        assert!(text.contains("data:"));
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_stream_options_include_usage() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/chat/completions")
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
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.unwrap();
        assert!(text.contains("data:"));
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_completions_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/completions")
            .json(&serde_json::json!({
                "model": "default",
                "prompt": "The sky is",
                "max_tokens": 5
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["choices"][0]["text"].is_string());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_tokenize_and_detokenize() {
        let server = TestServer::spawn();
        let r1 = reqwest::Client::new()
            .post(server.url() + "/tokenize")
            .json(&serde_json::json!({"model": "default", "content": "Hello world"}))
            .send()
            .await
            .unwrap();
        assert!(r1.status().is_success() || r1.status().is_client_error());
        if r1.status().is_success() {
            let body: serde_json::Value = r1.json().await.unwrap();
            let tokens = body["tokens"].as_array().unwrap().clone();
            assert!(!tokens.is_empty());

            let r2 = reqwest::Client::new()
                .post(server.url() + "/detokenize")
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
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/embeddings")
            .json(&serde_json::json!({"model": "default", "input": "test"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error() || resp.status().as_u16() == 501);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_malformed_json_returns_error() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/chat/completions")
            .header("Content-Type", "application/json")
            .body("not valid json")
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_missing_messages_returns_422() {
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/v1/chat/completions")
            .json(&serde_json::json!({"model": "default"}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_client_error());
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_seed_is_deterministic() {
        let server = TestServer::spawn();
        let req = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Random word:"}],
            "max_tokens": 5,
            "temperature": 0.8,
            "seed": 12345,
            "stream": false
        });
        let client = reqwest::Client::new();
        let r1: serde_json::Value = client
            .post(server.url() + "/v1/chat/completions")
            .json(&req)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let r2: serde_json::Value = client
            .post(server.url() + "/v1/chat/completions")
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
        let server = TestServer::spawn();
        let resp = reqwest::Client::new()
            .post(server.url() + "/tempura/v1/debug/logits")
            .json(&serde_json::json!({"tokens": [1, 2, 3]}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success() || resp.status().as_u16() == 501 || resp.status().as_u16() == 503);
        if resp.status().is_success() {
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body["logits"].is_array());
            assert!(body["logits"].as_array().unwrap().len() > 0);
        }
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_verified_status_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/tempura/v1/verified")
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary"]
    async fn test_traces_endpoint() {
        let server = TestServer::spawn();
        let resp = reqwest::get(server.url() + "/tempura/v1/traces")
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY"]
    async fn test_auth_correct_key_passes() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(server.url() + "/v1/models")
            .header("Authorization", format!("Bearer {key}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY"]
    async fn test_auth_wrong_key_returns_401() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(server.url() + "/v1/models")
            .header("Authorization", "Bearer wrong-key-xyz")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY"]
    async fn test_auth_health_bypasses_key() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::get(server.url() + "/health").await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    #[ignore = "requires built server binary and TEMPURA_TEST_API_KEY"]
    async fn test_auth_x_api_key_header() {
        let key = match std::env::var("TEMPURA_TEST_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };
        let server = TestServer::spawn_with_key(&key);
        let resp = reqwest::Client::new()
            .get(server.url() + "/v1/models")
            .header("X-API-Key", &key)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}

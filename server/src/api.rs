use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use oxide_engine::session::InferenceSession;
use oxide_engine::trace::TraceLevel;
use oxide_kernels::SamplingParams;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::sse::{ChatCompletionChunk, ChunkChoice, ChunkDelta};
use crate::AppState;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::Single(s) => vec![s],
            StringOrArray::Array(v) => v.into_iter().take(4).collect(),
        }
    }
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/oxide/v1/traces", get(list_traces))
        .route(
            "/oxide/v1/traces/{trace_id}",
            get(get_trace).delete(delete_trace),
        )
        .route("/oxide/v1/compare", post(compare_pipelines))
        .route("/oxide/v1/verified", get(verification_status))
        .route("/oxide/v1/replay", post(replay))
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health))
        .route("/ready", get(readiness))
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub stop: Option<StringOrArray>,
    #[serde(default = "default_n")]
    pub n: u32,
}

fn default_n() -> u32 {
    1
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobInfo>,
}

#[derive(Debug, Serialize)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprobInfo>,
}

#[derive(Debug, Serialize)]
pub struct TopLogprobInfo {
    pub token: String,
    pub logprob: f32,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<u8>,
    #[serde(default)]
    pub stop: Option<StringOrArray>,
    #[serde(default = "default_n")]
    pub n: u32,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

fn default_max_tokens() -> u32 {
    256
}
fn default_temperature() -> f32 {
    1.0
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let tokenizer = state.tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No tokenizer loaded".into(),
    ))?;

    if state.model.weights.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let prompt: String = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    if req.n < 1 || req.n > 128 {
        return Err((
            StatusCode::BAD_REQUEST,
            "n must be between 1 and 128".into(),
        ));
    }
    if req.stream && req.n > 1 {
        return Err((
            StatusCode::BAD_REQUEST,
            "n > 1 is not supported with streaming".into(),
        ));
    }

    let params = SamplingParams {
        temperature: req.temperature,
        top_k: 50,
        top_p: req.top_p.unwrap_or(1.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
    };

    if req.stream {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = req.model.clone();
        let model = state.model.clone();
        let tokenizer = tokenizer.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            let mut session = InferenceSession::new(&model.config, TraceLevel::Off, 42);
            let id_clone = id.clone();
            let model_clone = model_name.clone();

            model.generate_with_callback(
                &mut session,
                &token_ids,
                params.top_k.min(256),
                &params,
                None,
                |token| {
                    let text = tokenizer.decode(&[token]).unwrap_or_default();
                    let chunk = ChatCompletionChunk {
                        id: id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: 0,
                        model: model_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap();
                    let event = Event::default().data(data);
                    tx.blocking_send(Ok(event)).is_ok()
                },
            );

            let done = Event::default().data("[DONE]");
            let _ = tx.blocking_send(Ok(done));
        });

        let stream = ReceiverStream::new(rx);
        Ok(Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response())
    } else {
        let start = std::time::Instant::now();
        metrics::counter!("oxide_request_count", "endpoint" => "chat_completions").increment(1);

        let want_logprobs = req.logprobs.unwrap_or(false);
        let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
        let stop_sequences: Vec<String> = req.stop.map(|s| s.into_vec()).unwrap_or_default();
        let prompt_tokens = token_ids.len() as u32;

        let mut choices = Vec::new();
        let mut total_completion_tokens = 0u32;

        for i in 0..req.n {
            let seed = 42u64.wrapping_add(i as u64);
            let mut session = InferenceSession::new(&state.model.config, TraceLevel::Off, seed);

            let (generated, token_logprobs) = if want_logprobs {
                state.model.generate_with_logprobs(
                    &mut session,
                    &token_ids,
                    req.max_tokens as usize,
                    &params,
                    None,
                    top_n.max(1),
                )
            } else {
                let tokens = state.model.generate(
                    &mut session,
                    &token_ids,
                    req.max_tokens as usize,
                    &params,
                    None,
                );
                (tokens, Vec::new())
            };

            let full_text = tokenizer
                .decode(&generated)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let (text, finish_reason) = check_stop_sequences(&full_text, &stop_sequences);
            total_completion_tokens += generated.len() as u32;

            let logprobs_response = if want_logprobs {
                Some(ChoiceLogprobs {
                    content: token_logprobs
                        .iter()
                        .map(|lp| TokenLogprobInfo {
                            token: format!("token_{}", lp.token_id),
                            logprob: lp.logprob,
                            top_logprobs: lp
                                .top_logprobs
                                .iter()
                                .map(|t| TopLogprobInfo {
                                    token: format!("token_{}", t.token_id),
                                    logprob: t.logprob,
                                })
                                .collect(),
                        })
                        .collect(),
                })
            } else {
                None
            };

            choices.push(ChatChoice {
                index: i,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text.to_string(),
                },
                finish_reason: finish_reason.to_string(),
                logprobs: logprobs_response,
            });
        }

        metrics::counter!("oxide_tokens_generated").increment(total_completion_tokens as u64);
        metrics::histogram!("oxide_request_duration_seconds", "endpoint" => "chat_completions")
            .record(start.elapsed().as_secs_f64());

        Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: 0,
            model: req.model,
            choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens: total_completion_tokens,
                total_tokens: prompt_tokens + total_completion_tokens,
            },
        })
        .into_response())
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let tokenizer = state.tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No tokenizer loaded".into(),
    ))?;

    if state.model.weights.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let token_ids = tokenizer
        .encode(&req.prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let params = SamplingParams {
        temperature: req.temperature,
        top_k: 50,
        top_p: 1.0,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
    };

    let start = std::time::Instant::now();
    metrics::counter!("oxide_request_count", "endpoint" => "completions").increment(1);

    let mut session = InferenceSession::new(&state.model.config, TraceLevel::Off, 42);
    let want_logprobs = req.logprobs.is_some();
    let top_n = req.logprobs.unwrap_or(0).min(20) as usize;

    let (generated, token_logprobs) = if want_logprobs {
        state.model.generate_with_logprobs(
            &mut session,
            &token_ids,
            req.max_tokens as usize,
            &params,
            None,
            top_n.max(1),
        )
    } else {
        let tokens = state.model.generate(
            &mut session,
            &token_ids,
            req.max_tokens as usize,
            &params,
            None,
        );
        (tokens, Vec::new())
    };

    let full_text = tokenizer
        .decode(&generated)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let stop_sequences: Vec<String> = req.stop.map(|s| s.into_vec()).unwrap_or_default();
    let (text, finish_reason) = check_stop_sequences(&full_text, &stop_sequences);

    let prompt_tokens = token_ids.len() as u32;
    let completion_tokens = generated.len() as u32;

    metrics::counter!("oxide_tokens_generated").increment(completion_tokens as u64);
    metrics::histogram!("oxide_request_duration_seconds", "endpoint" => "completions")
        .record(start.elapsed().as_secs_f64());

    let logprobs_response = if want_logprobs {
        Some(ChoiceLogprobs {
            content: token_logprobs
                .iter()
                .map(|lp| TokenLogprobInfo {
                    token: format!("token_{}", lp.token_id),
                    logprob: lp.logprob,
                    top_logprobs: lp
                        .top_logprobs
                        .iter()
                        .map(|t| TopLogprobInfo {
                            token: format!("token_{}", t.token_id),
                            logprob: t.logprob,
                        })
                        .collect(),
                })
                .collect(),
        })
    } else {
        None
    };

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: 0,
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: text.to_string(),
            finish_reason: finish_reason.to_string(),
            logprobs: logprobs_response,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            owned_by: "oxide-serve".to_string(),
        }],
    })
}

async fn list_traces(State(state): State<AppState>) -> impl IntoResponse {
    let store = state.trace_store.lock().await;
    Json(store.list())
}

async fn get_trace(
    State(state): State<AppState>,
    axum::extract::Path(trace_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let store = state.trace_store.lock().await;
    store
        .get(&trace_id)
        .map(|t| Json(t.clone()))
        .ok_or(StatusCode::NOT_FOUND)
}

async fn delete_trace(
    State(state): State<AppState>,
    axum::extract::Path(trace_id): axum::extract::Path<String>,
) -> StatusCode {
    let mut store = state.trace_store.lock().await;
    if store.remove(&trace_id) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

#[derive(Debug, Deserialize)]
struct CompareRequest {
    pub token_ids: Vec<u32>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub seed: Option<u64>,
}

async fn compare_pipelines(
    State(state): State<AppState>,
    Json(req): Json<CompareRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if state.model.weights.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let params = SamplingParams {
        temperature: req.temperature,
        top_k: 50,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let seed = req.seed.unwrap_or(42);

    let result = oxide_engine::dual_pipeline::compare_pipelines(
        state.model.as_ref(),
        state.model.as_ref(),
        &state.model.config,
        &req.token_ids,
        &params,
        seed,
    );

    Ok(Json(result))
}

async fn verification_status() -> impl IntoResponse {
    Json(oxide_kernels::verification_manifest::manifest())
}

async fn replay(
    State(state): State<AppState>,
    Json(req): Json<crate::replay::ReplayRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if state.model.weights.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let token_ids = if let Some(ref ids) = req.token_ids {
        ids.clone()
    } else if let Some(ref trace_id) = req.trace_id {
        let store = state.trace_store.lock().await;
        let trace = store
            .get(trace_id)
            .ok_or((StatusCode::NOT_FOUND, format!("Trace {trace_id} not found")))?;
        // Use the seed from the trace for replay
        let _seed = trace.seed;
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "Replay from trace_id requires stored token_ids (not yet implemented)".into(),
        ));
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Must provide either token_ids or trace_id".into(),
        ));
    };

    let seed = req.seed.unwrap_or(42);
    let params = SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k.unwrap_or(50),
        top_p: req.top_p.unwrap_or(1.0),
        ..SamplingParams::default()
    };

    let (original, replay_tokens) =
        crate::replay::replay_from_params(&state.model, &token_ids, seed, req.max_tokens, &params);

    Ok(Json(crate::replay::build_replay_result(
        original,
        replay_tokens,
    )))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics_handle.render()
}

async fn health() -> &'static str {
    "ok"
}

fn check_stop_sequences<'a>(text: &'a str, stop_sequences: &[String]) -> (&'a str, &'static str) {
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            return (&text[..pos], "stop");
        }
    }
    (text, "length")
}

#[derive(Serialize)]
struct ReadinessResponse {
    ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

async fn readiness(State(state): State<AppState>) -> impl IntoResponse {
    let mut missing = Vec::new();
    if state.model.weights.is_none() {
        missing.push("model weights");
    }
    if state.tokenizer.is_none() {
        missing.push("tokenizer");
    }

    if missing.is_empty() {
        (
            StatusCode::OK,
            Json(ReadinessResponse {
                ready: true,
                reason: None,
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                ready: false,
                reason: Some(format!("not loaded: {}", missing.join(", "))),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state(with_tokenizer: bool) -> AppState {
        let config = oxide_engine::ModelConfig::tiny_test();
        let pipeline = oxide_engine::pipeline::KernelPipeline::cpu_verified();
        let model = oxide_engine::TransformerModel::new(config, pipeline);

        let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let metrics_handle = prometheus_builder.install_recorder().unwrap_or_else(|_| {
            // recorder already installed in another test
            metrics_exporter_prometheus::PrometheusBuilder::new()
                .build_recorder()
                .handle()
        });

        AppState {
            model: std::sync::Arc::new(model),
            tokenizer: if with_tokenizer {
                None // no real tokenizer available in tests
            } else {
                None
            },
            model_id: "test".to_string(),
            scheduler: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::batch::BatchScheduler::new(1, 128, 1),
            )),
            trace_store: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::trace_store::TraceStore::new(10),
            )),
            metrics_handle,
        }
    }

    #[tokio::test]
    async fn test_ready_returns_503_without_model() {
        let state = test_state(false);
        let app = router().with_state(state);
        let req = Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ready"], false);
        assert!(v["reason"].as_str().unwrap().contains("model weights"));
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let state = test_state(false);
        let app = router().with_state(state);
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn test_check_stop_sequences_match() {
        let (text, reason) =
            check_stop_sequences("Hello world! END more text", &["END".to_string()]);
        assert_eq!(text, "Hello world! ");
        assert_eq!(reason, "stop");
    }

    #[test]
    fn test_check_stop_sequences_no_match() {
        let (text, reason) = check_stop_sequences("Hello world!", &["END".to_string()]);
        assert_eq!(text, "Hello world!");
        assert_eq!(reason, "length");
    }

    #[test]
    fn test_check_stop_sequences_empty() {
        let (text, reason) = check_stop_sequences("Hello world!", &[]);
        assert_eq!(text, "Hello world!");
        assert_eq!(reason, "length");
    }

    #[test]
    fn test_string_or_array_single() {
        let s: StringOrArray = serde_json::from_str(r#""stop""#).unwrap();
        assert_eq!(s.into_vec(), vec!["stop".to_string()]);
    }

    #[test]
    fn test_string_or_array_array() {
        let s: StringOrArray = serde_json::from_str(r#"["a", "b", "c"]"#).unwrap();
        assert_eq!(
            s.into_vec(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}

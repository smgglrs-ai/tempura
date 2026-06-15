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
        .route("/health", get(health))
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

    let params = SamplingParams {
        temperature: req.temperature,
        top_k: 50,
        top_p: req.top_p.unwrap_or(1.0),
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
        let mut session = InferenceSession::new(&state.model.config, TraceLevel::Off, 42);
        let generated = state.model.generate(
            &mut session,
            &token_ids,
            req.max_tokens as usize,
            &params,
            None,
        );

        let text = tokenizer
            .decode(&generated)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let prompt_tokens = token_ids.len() as u32;
        let completion_tokens = generated.len() as u32;

        Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: 0,
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
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
    };

    let mut session = InferenceSession::new(&state.model.config, TraceLevel::Off, 42);
    let generated = state.model.generate(
        &mut session,
        &token_ids,
        req.max_tokens as usize,
        &params,
        None,
    );

    let text = tokenizer
        .decode(&generated)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let prompt_tokens = token_ids.len() as u32;
    let completion_tokens = generated.len() as u32;

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: 0,
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: "stop".to_string(),
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
    };

    let (original, replay_tokens) =
        crate::replay::replay_from_params(&state.model, &token_ids, seed, req.max_tokens, &params);

    Ok(Json(crate::replay::build_replay_result(
        original,
        replay_tokens,
    )))
}

async fn health() -> &'static str {
    "ok"
}

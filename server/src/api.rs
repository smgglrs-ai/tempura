use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use oxide_engine::session::InferenceSession;
use oxide_engine::trace::TraceLevel;
use oxide_kernels::SamplingParams;
use serde::{Deserialize, Serialize};

use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
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
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
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
    }))
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

async fn health() -> &'static str {
    "ok"
}

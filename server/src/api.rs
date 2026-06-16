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

use crate::batch::InferenceRequest;
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
        .route("/health/inference", get(health_inference))
        .route("/ready", get(readiness))
        .route("/v1/responses", post(responses))
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
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
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<oxide_engine::tool_call::ToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<oxide_engine::tool_call::ToolChoice>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub cache_salt: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

fn default_n() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<oxide_engine::tool_call::ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
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

impl ChatMessage {
    fn content_str(&self) -> String {
        self.content.clone().unwrap_or_default()
    }
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
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
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
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub cache_salt: Option<String>,
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

async fn acquire_slot(
    state: &AppState,
    request_id: &str,
    token_ids: Vec<u32>,
    max_new_tokens: u32,
    temperature: f32,
    top_p: f32,
    priority: u8,
) -> Result<(), (StatusCode, String)> {
    let mut scheduler = state.scheduler.lock().await;
    scheduler.submit(InferenceRequest {
        id: request_id.to_string(),
        token_ids,
        max_new_tokens,
        temperature,
        top_p,
        priority,
    });
    scheduler.admit_new_requests();

    record_scheduler_gauges(&scheduler);

    let admitted = scheduler
        .active_sequences()
        .iter()
        .any(|s| s.request_id == request_id);

    if admitted {
        Ok(())
    } else {
        scheduler.withdraw_pending(request_id);
        Err((
            StatusCode::TOO_MANY_REQUESTS,
            "Server is at capacity, please retry".into(),
        ))
    }
}

async fn release_slot(state: &AppState, request_id: &str) {
    let mut scheduler = state.scheduler.lock().await;
    scheduler.complete_sequence(request_id);
    record_scheduler_gauges(&scheduler);
}

fn resolve_timeout(state: &AppState, timeout_secs: Option<u64>) -> std::time::Duration {
    match timeout_secs {
        Some(secs) => std::time::Duration::from_secs(secs),
        None => state.request_timeout,
    }
}

fn record_scheduler_gauges(scheduler: &crate::batch::BatchScheduler) {
    metrics::gauge!("oxide_scheduler_active_sequences").set(scheduler.active_count() as f64);
    metrics::gauge!("oxide_scheduler_queue_depth")
        .set((scheduler.pending_count() + scheduler.preempted_count()) as f64);
    metrics::gauge!("oxide_kv_cache_slots_in_use").set(scheduler.allocated_slot_count() as f64);
    metrics::gauge!("oxide_kv_cache_slots_total").set(scheduler.max_batch_size() as f64);
}

#[tracing::instrument(skip(state, req), fields(model = %req.model))]
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let tokenizer = state.tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No tokenizer loaded".into(),
    ))?;

    if !state.model.has_weights() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let chat_messages: Vec<oxide_engine::ChatMessageInput> = req
        .messages
        .iter()
        .map(|m| oxide_engine::ChatMessageInput {
            role: m.role.clone(),
            content: m.content_str(),
        })
        .collect();

    let prompt = tokenizer
        .apply_chat_template(&chat_messages, true)
        .unwrap_or_else(|_| {
            req.messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content_str()))
                .collect::<Vec<_>>()
                .join("\n")
        });

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

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    acquire_slot(
        &state,
        &request_id,
        token_ids.clone(),
        req.max_tokens,
        req.temperature,
        req.top_p.unwrap_or(1.0),
        5,
    )
    .await?;

    let timeout = resolve_timeout(&state, req.timeout_secs);

    if req.stream {
        let model = state.model.clone();
        let tokenizer = tokenizer.clone();
        let scheduler = state.scheduler.clone();
        let rid = request_id.clone();
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let prompt_tokens = token_ids.len() as u32;
        let base_seed = req.seed.unwrap_or(42);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

        metrics::counter!("oxide_request_count", "endpoint" => "chat_completions").increment(1);

        tokio::task::spawn_blocking(move || {
            let stream_start = std::time::Instant::now();
            let mut session = InferenceSession::new(model.config(), TraceLevel::Off, base_seed);
            let id_clone = rid.clone();
            let model_name = req.model.clone();
            let deadline = std::time::Instant::now() + timeout;
            let mut token_count = 0u64;
            let mut last_token_time = stream_start;
            let mut first_token_recorded = false;

            let mut on_token = |token: u32| -> bool {
                let now = std::time::Instant::now();
                if now >= deadline {
                    metrics::counter!("oxide_request_timeout", "endpoint" => "chat_completions")
                        .increment(1);
                    return false;
                }

                if !first_token_recorded {
                    metrics::histogram!("oxide_time_to_first_token_seconds")
                        .record(now.duration_since(stream_start).as_secs_f64());
                    first_token_recorded = true;
                } else {
                    metrics::histogram!("oxide_inter_token_latency_seconds")
                        .record(now.duration_since(last_token_time).as_secs_f64());
                }
                last_token_time = now;
                token_count += 1;

                let text = tokenizer.decode(&[token]).unwrap_or_default();
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(text),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                let data = serde_json::to_string(&chunk).unwrap();
                let event = Event::default().data(data);
                tx.blocking_send(Ok(event)).is_ok()
            };

            model.generate_with_callback(
                &mut session,
                &token_ids,
                params.top_k.min(256),
                &params,
                None,
                &mut on_token,
            );

            metrics::counter!("oxide_tokens_generated").increment(token_count);
            metrics::histogram!("oxide_request_duration_seconds", "endpoint" => "chat_completions")
                .record(stream_start.elapsed().as_secs_f64());

            if include_usage {
                let usage_chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: model_name.clone(),
                    choices: vec![],
                    usage: Some(crate::sse::ChunkUsage {
                        prompt_tokens,
                        completion_tokens: token_count as u32,
                        total_tokens: prompt_tokens + token_count as u32,
                    }),
                };
                let data = serde_json::to_string(&usage_chunk).unwrap();
                let _ = tx.blocking_send(Ok(Event::default().data(data)));
            }

            let done = Event::default().data("[DONE]");
            let _ = tx.blocking_send(Ok(done));

            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let mut sched = scheduler.lock().await;
                sched.complete_sequence(&rid);
                record_scheduler_gauges(&sched);
            });
        });

        let stream = ReceiverStream::new(rx);
        Ok(Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response())
    } else {
        let start = std::time::Instant::now();
        let deadline = start + timeout;
        metrics::counter!("oxide_request_count", "endpoint" => "chat_completions").increment(1);

        let want_logprobs = req.logprobs.unwrap_or(false);
        let top_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
        let stop_sequences: Vec<String> = req.stop.map(|s| s.into_vec()).unwrap_or_default();
        let prompt_tokens = token_ids.len() as u32;

        let mut choices = Vec::new();
        let mut total_completion_tokens = 0u32;

        let result = (|| -> Result<(), (StatusCode, String)> {
            for i in 0..req.n {
                if std::time::Instant::now() >= deadline {
                    metrics::counter!("oxide_request_timeout", "endpoint" => "chat_completions")
                        .increment(1);
                    return Err((StatusCode::GATEWAY_TIMEOUT, "Request timed out".into()));
                }
                let base_seed = req.seed.unwrap_or(42);
                let seed = base_seed.wrapping_add(i as u64);
                let mut session =
                    InferenceSession::new(state.model.config(), TraceLevel::Off, seed);

                let mut guided_decoder = build_guided_decoder(&req.response_format, tokenizer)
                    .or_else(|| build_tool_guided_decoder(&req.tools, &req.tool_choice, tokenizer));

                let (generated, token_logprobs) = if want_logprobs {
                    state.model.generate_with_logprobs(
                        &mut session,
                        &token_ids,
                        req.max_tokens as usize,
                        &params,
                        None,
                        top_n.max(1),
                    )
                } else if let Some(ref mut decoder) = guided_decoder {
                    let tokens = state.model.generate_guided(
                        &mut session,
                        &token_ids,
                        req.max_tokens as usize,
                        &params,
                        None,
                        decoder.as_mut(),
                    );
                    (tokens, Vec::new())
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

                let (msg, fr) = if let Some(ref tools) = req.tools {
                    if !tools.is_empty() && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none())
                    {
                        let parser = oxide_engine::tool_call::ToolCallParser::new(tools);
                        if let Some(tool_calls) = parser.parse(text) {
                            (
                                ChatMessage {
                                    role: "assistant".to_string(),
                                    content: None,
                                    tool_calls: Some(tool_calls),
                                    tool_call_id: None,
                                },
                                "tool_calls".to_string(),
                            )
                        } else {
                            (
                                ChatMessage {
                                    role: "assistant".to_string(),
                                    content: Some(text.to_string()),
                                    tool_calls: None,
                                    tool_call_id: None,
                                },
                                finish_reason.to_string(),
                            )
                        }
                    } else {
                        (
                            ChatMessage {
                                role: "assistant".to_string(),
                                content: Some(text.to_string()),
                                tool_calls: None,
                                tool_call_id: None,
                            },
                            finish_reason.to_string(),
                        )
                    }
                } else {
                    (
                        ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(text.to_string()),
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason.to_string(),
                    )
                };

                choices.push(ChatChoice {
                    index: i,
                    message: msg,
                    finish_reason: fr,
                    logprobs: logprobs_response,
                });
            }
            Ok(())
        })();

        release_slot(&state, &request_id).await;

        result?;

        metrics::counter!("oxide_tokens_generated").increment(total_completion_tokens as u64);
        metrics::histogram!("oxide_request_duration_seconds", "endpoint" => "chat_completions")
            .record(start.elapsed().as_secs_f64());

        Ok(Json(ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: 0,
            model: req.model,
            system_fingerprint: format!("oxide-serve-{}", env!("CARGO_PKG_VERSION")),
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

#[tracing::instrument(skip(state, req), fields(model = %req.model))]
async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let tokenizer = state.tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No tokenizer loaded".into(),
    ))?;

    if !state.model.has_weights() {
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

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let timeout = resolve_timeout(&state, req.timeout_secs);

    acquire_slot(
        &state,
        &request_id,
        token_ids.clone(),
        req.max_tokens,
        req.temperature,
        1.0,
        5,
    )
    .await?;

    let start = std::time::Instant::now();
    metrics::counter!("oxide_request_count", "endpoint" => "completions").increment(1);

    let want_logprobs = req.logprobs.is_some();
    let top_n = req.logprobs.unwrap_or(0).min(20) as usize;
    let seed = req.seed.unwrap_or(42);

    let (generated, token_logprobs) = if want_logprobs {
        state.model.generate_with_logprobs(
            &mut InferenceSession::new(state.model.config(), TraceLevel::Off, seed),
            &token_ids,
            req.max_tokens as usize,
            &params,
            None,
            top_n.max(1),
        )
    } else {
        let tokens = state.model.generate(
            &mut InferenceSession::new(state.model.config(), TraceLevel::Off, seed),
            &token_ids,
            req.max_tokens as usize,
            &params,
            None,
        );
        (tokens, Vec::new())
    };

    if start.elapsed() >= timeout {
        release_slot(&state, &request_id).await;
        metrics::counter!("oxide_request_timeout", "endpoint" => "completions").increment(1);
        return Err((StatusCode::GATEWAY_TIMEOUT, "Request timed out".into()));
    }

    let full_text = tokenizer.decode(&generated).map_err(|e| {
        let state = state.clone();
        let rid = request_id.clone();
        tokio::task::spawn(async move { release_slot(&state, &rid).await });
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    release_slot(&state, &request_id).await;

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
        id: request_id,
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
    if !state.model.has_weights() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let seed = req.seed.unwrap_or(42);

    let config = state.model.config().clone();
    let mut session1 = InferenceSession::new(&config, TraceLevel::Off, seed);
    let logits1 = state.model.forward(&mut session1, &req.token_ids);

    let mut session2 = InferenceSession::new(&config, TraceLevel::Off, seed);
    let logits2 = state.model.forward(&mut session2, &req.token_ids);

    let max_diff = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    Ok(Json(serde_json::json!({
        "logits_max_abs_diff": max_diff,
        "token_match": true,
        "note": "self-comparison (same pipeline)"
    })))
}

async fn verification_status() -> impl IntoResponse {
    Json(oxide_kernels::verification_manifest::manifest())
}

async fn replay(
    State(state): State<AppState>,
    Json(req): Json<crate::replay::ReplayRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if !state.model.has_weights() {
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

    let (original, replay_tokens) = crate::replay::replay_from_params(
        state.model.as_ref(),
        &token_ids,
        seed,
        req.max_tokens,
        &params,
    );

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

async fn health_inference(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if !state.model.has_weights() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"healthy": false, "error": "model weights not loaded"})),
        ));
    }

    let probe_id = format!("health-{}", uuid::Uuid::new_v4());
    let acquire_result = acquire_slot(&state, &probe_id, vec![1], 1, 1.0, 1.0, 10).await;

    if acquire_result.is_err() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"healthy": false, "error": "scheduler at capacity"})),
        ));
    }

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(5);

    let model = state.model.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut session = InferenceSession::new(model.config(), TraceLevel::Off, 0);
        let params = SamplingParams::default();
        model.generate(&mut session, &[1], 1, &params, None)
    })
    .await;

    release_slot(&state, &probe_id).await;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    if start.elapsed() >= timeout {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"healthy": false, "error": "inference timeout"})),
        ));
    }

    match result {
        Ok(tokens) if !tokens.is_empty() => Ok(Json(
            serde_json::json!({"healthy": true, "latency_ms": latency_ms}),
        )),
        _ => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"healthy": false, "error": "inference produced no output"})),
        )),
    }
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    model: String,
    input: ResponsesInput,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_output_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    tools: Option<Vec<oxide_engine::tool_call::ToolDefinition>>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponsesMessage>),
}

#[derive(Debug, Deserialize)]
struct ResponsesMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponsesResponse {
    id: String,
    object: String,
    created_at: u64,
    model: String,
    output: Vec<ResponsesOutputItem>,
    usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
struct ResponsesOutputItem {
    #[serde(rename = "type")]
    item_type: String,
    id: String,
    role: String,
    content: Vec<ResponsesContent>,
}

#[derive(Debug, Serialize)]
struct ResponsesContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct ResponsesUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

async fn responses(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Result<Json<ResponsesResponse>, (StatusCode, String)> {
    let tokenizer = state.tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "No tokenizer loaded".into(),
    ))?;

    if !state.model.has_weights() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "No model weights loaded".into(),
        ));
    }

    let mut messages = Vec::new();
    if let Some(instructions) = &req.instructions {
        messages.push(oxide_engine::ChatMessageInput {
            role: "system".to_string(),
            content: instructions.clone(),
        });
    }

    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(oxide_engine::ChatMessageInput {
                role: "user".to_string(),
                content: text.clone(),
            });
        }
        ResponsesInput::Messages(msgs) => {
            for m in msgs {
                messages.push(oxide_engine::ChatMessageInput {
                    role: m.role.clone(),
                    content: m.content.clone(),
                });
            }
        }
    }

    let prompt = tokenizer
        .apply_chat_template(&messages, true)
        .unwrap_or_else(|_| {
            messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        });

    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let params = SamplingParams {
        temperature: req.temperature,
        top_k: 50,
        top_p: 1.0,
        ..SamplingParams::default()
    };

    let seed = req.seed.unwrap_or(42);
    let mut session = InferenceSession::new(state.model.config(), TraceLevel::Off, seed);
    let prompt_tokens = token_ids.len() as u32;

    let generated = state.model.generate(
        &mut session,
        &token_ids,
        req.max_output_tokens as usize,
        &params,
        None,
    );

    let text = tokenizer
        .decode(&generated)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let completion_tokens = generated.len() as u32;

    Ok(Json(ResponsesResponse {
        id: format!("resp_{}", uuid::Uuid::new_v4()),
        object: "response".to_string(),
        created_at: 0,
        model: req.model,
        output: vec![ResponsesOutputItem {
            item_type: "message".to_string(),
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            role: "assistant".to_string(),
            content: vec![ResponsesContent {
                content_type: "output_text".to_string(),
                text,
            }],
        }],
        usage: ResponsesUsage {
            input_tokens: prompt_tokens,
            output_tokens: completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

fn build_guided_decoder(
    response_format: &Option<ResponseFormat>,
    tokenizer: &oxide_engine::tokenizer::Tokenizer,
) -> Option<Box<dyn oxide_kernels::GuidedDecoder>> {
    let rf = response_format.as_ref()?;
    match rf.format_type.as_str() {
        "json_object" => {
            let vocab = tokenizer.vocab_strings();
            Some(oxide_kernels::guided::create_json_decoder(vocab))
        }
        "json_schema" => {
            let vocab = tokenizer.vocab_strings();
            let schema = rf
                .json_schema
                .as_ref()
                .and_then(|s| s.schema.as_ref())
                .cloned()
                .unwrap_or(serde_json::json!({}));
            Some(oxide_kernels::guided::create_json_schema_decoder(
                vocab, &schema,
            ))
        }
        _ => None,
    }
}

fn build_tool_guided_decoder(
    tools: &Option<Vec<oxide_engine::tool_call::ToolDefinition>>,
    tool_choice: &Option<oxide_engine::tool_call::ToolChoice>,
    tokenizer: &oxide_engine::tokenizer::Tokenizer,
) -> Option<Box<dyn oxide_kernels::GuidedDecoder>> {
    let tc = tool_choice.as_ref()?;
    let tools = tools.as_ref()?;
    if tools.is_empty() || tc.is_none() {
        return None;
    }

    if tc.is_required() {
        let vocab = tokenizer.vocab_strings();
        return Some(oxide_kernels::guided::create_json_decoder(vocab));
    }

    if let Some(func_name) = tc.specific_function() {
        let schema = tools
            .iter()
            .find(|t| t.function.name == func_name)
            .and_then(|t| t.function.parameters.as_ref())
            .cloned()
            .unwrap_or(serde_json::json!({"type": "object"}));
        let vocab = tokenizer.vocab_strings();
        return Some(oxide_kernels::guided::create_json_schema_decoder(
            vocab, &schema,
        ));
    }

    None
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
    if !state.model.has_weights() {
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
            metrics_exporter_prometheus::PrometheusBuilder::new()
                .build_recorder()
                .handle()
        });

        AppState {
            model: std::sync::Arc::new(model),
            tokenizer: if with_tokenizer { None } else { None },
            model_id: "test".to_string(),
            scheduler: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::batch::BatchScheduler::new(1, 128, 1),
            )),
            trace_store: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::trace_store::TraceStore::new(10),
            )),
            metrics_handle,
            request_timeout: std::time::Duration::from_secs(300),
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

    #[tokio::test]
    async fn test_scheduler_gates_concurrent_requests() {
        let config = oxide_engine::ModelConfig::tiny_test();
        let pipeline = oxide_engine::pipeline::KernelPipeline::cpu_verified();
        let model = oxide_engine::TransformerModel::new(config, pipeline);

        let scheduler = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::batch::BatchScheduler::new(1, 128, 1),
        ));

        let prometheus_builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let metrics_handle = prometheus_builder.install_recorder().unwrap_or_else(|_| {
            metrics_exporter_prometheus::PrometheusBuilder::new()
                .build_recorder()
                .handle()
        });

        let state = AppState {
            model: std::sync::Arc::new(model),
            tokenizer: None,
            model_id: "test".to_string(),
            scheduler: scheduler.clone(),
            trace_store: std::sync::Arc::new(tokio::sync::Mutex::new(
                crate::trace_store::TraceStore::new(10),
            )),
            metrics_handle,
            request_timeout: std::time::Duration::from_secs(300),
        };

        // First request acquires the only slot
        acquire_slot(&state, "req-1", vec![1, 2], 10, 1.0, 1.0, 5)
            .await
            .unwrap();

        {
            let sched = scheduler.lock().await;
            assert_eq!(sched.active_count(), 1);
            assert_eq!(sched.allocated_slot_count(), 1);
        }

        // Second request should be rejected (max_batch_size=1, num_slots=1)
        let result = acquire_slot(&state, "req-2", vec![3, 4], 10, 1.0, 1.0, 5).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);

        // Release first slot
        release_slot(&state, "req-1").await;

        {
            let sched = scheduler.lock().await;
            assert_eq!(sched.active_count(), 0);
            assert_eq!(sched.allocated_slot_count(), 0);
        }

        // Now third request succeeds
        acquire_slot(&state, "req-3", vec![5, 6], 10, 1.0, 1.0, 5)
            .await
            .unwrap();

        {
            let sched = scheduler.lock().await;
            assert_eq!(sched.active_count(), 1);
        }

        release_slot(&state, "req-3").await;
    }
}

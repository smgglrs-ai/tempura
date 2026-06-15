use axum::response::sse::{Event, KeepAlive};
use serde::Serialize;
use tokio_stream::Stream;

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub fn keep_alive() -> KeepAlive {
    KeepAlive::default()
}

pub fn chunk_to_event(chunk: &ChatCompletionChunk) -> Result<Event, serde_json::Error> {
    let data = serde_json::to_string(chunk)?;
    Ok(Event::default().data(data))
}

pub fn done_event() -> Event {
    Event::default().data("[DONE]")
}

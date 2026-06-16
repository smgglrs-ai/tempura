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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk(content: Option<&str>, finish_reason: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1700000000,
            model: "test-model".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: content.map(|s| s.to_string()),
                },
                finish_reason: finish_reason.map(|s| s.to_string()),
            }],
        }
    }

    #[test]
    fn test_chunk_serialization_with_content() {
        let chunk = sample_chunk(Some("Hello"), None);
        let json = serde_json::to_string(&chunk).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["id"], "chatcmpl-test");
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hello");
        assert!(v["choices"][0]["delta"]["role"].is_null());
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_chunk_serialization_with_role() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1700000000,
            model: "m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert!(v["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn test_chunk_serialization_with_finish_reason() {
        let chunk = sample_chunk(None, Some("stop"));
        let json = serde_json::to_string(&chunk).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_chunk_to_event_succeeds() {
        let chunk = sample_chunk(Some("token"), None);
        let event = chunk_to_event(&chunk);
        assert!(event.is_ok());
    }

    #[test]
    fn test_done_event_does_not_panic() {
        let _event = done_event();
    }
}

use std::collections::HashMap;
use std::path::Path;

use minijinja::{context, Environment};
use serde::Deserialize;

use crate::gguf_loader::GgufValue;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("encoding error: {0}")]
    Encode(String),
    #[error("decoding error: {0}")]
    Decode(String),
    #[error("chat template error: {0}")]
    Template(String),
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<StringOrToken>,
    eos_token: Option<StringOrToken>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrToken {
    Simple(String),
    Object { content: String },
}

impl StringOrToken {
    fn as_str(&self) -> &str {
        match self {
            StringOrToken::Simple(s) => s,
            StringOrToken::Object { content } => content,
        }
    }
}

const CHATML_TEMPLATE: &str = "\
{%- for message in messages %}\
<|im_start|>{{ message.role }}\n\
{{ message.content }}<|im_end|>\n\
{%- endfor %}\
{%- if add_generation_prompt %}\
<|im_start|>assistant\n\
{% endif %}";

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template_source: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(template: String, bos_token: String, eos_token: String) -> Self {
        Self {
            template_source: template,
            bos_token,
            eos_token,
        }
    }

    pub fn chatml() -> Self {
        Self {
            template_source: CHATML_TEMPLATE.to_string(),
            bos_token: String::new(),
            eos_token: String::new(),
        }
    }

    pub fn apply(
        &self,
        messages: &[ChatMessageInput],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template_source)
            .map_err(|e| TokenizerError::Template(e.to_string()))?;

        let messages_val: Vec<minijinja::Value> = messages
            .iter()
            .map(|m| {
                context! {
                    role => m.role,
                    content => m.content,
                }
            })
            .collect();

        let tmpl = env
            .get_template("chat")
            .map_err(|e| TokenizerError::Template(e.to_string()))?;

        tmpl.render(context! {
            messages => messages_val,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        })
        .map_err(|e| TokenizerError::Template(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessageInput {
    pub role: String,
    pub content: String,
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    chat_template: ChatTemplate,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::Load(e.to_string()))?;

        let chat_template = if let Some(parent) = path.parent() {
            let config_path = parent.join("tokenizer_config.json");
            if config_path.exists() {
                Self::load_chat_template(&config_path)?
            } else {
                ChatTemplate::chatml()
            }
        } else {
            ChatTemplate::chatml()
        };

        Ok(Self {
            inner,
            chat_template,
        })
    }

    pub fn from_gguf_metadata(
        metadata: &HashMap<String, GgufValue>,
    ) -> Result<Self, TokenizerError> {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| {
                TokenizerError::Load("missing tokenizer.ggml.tokens in GGUF metadata".into())
            })?;

        if tokens.is_empty() {
            return Err(TokenizerError::Load(
                "empty token list in GGUF metadata".into(),
            ));
        }

        let merges = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array());

        let mut vocab = tokenizers::models::bpe::Vocab::new();
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert((*token).to_string(), id as u32);
        }

        let merge_pairs: tokenizers::models::bpe::Merges = merges
            .unwrap_or_default()
            .iter()
            .filter_map(|m| {
                let (a, b) = m.split_once(' ')?;
                Some((a.to_string(), b.to_string()))
            })
            .collect();

        let bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, merge_pairs)
            .byte_fallback(true)
            .build()
            .map_err(|e| TokenizerError::Load(format!("failed to build BPE from GGUF: {e}")))?;

        let inner = tokenizers::Tokenizer::new(bpe);

        let bos_token = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .and_then(|id| tokens.get(id as usize))
            .map(|s| s.to_string())
            .unwrap_or_default();

        let eos_token = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .and_then(|id| tokens.get(id as usize))
            .map(|s| s.to_string())
            .unwrap_or_default();

        let chat_template = ChatTemplate::new(CHATML_TEMPLATE.to_string(), bos_token, eos_token);

        Ok(Self {
            inner,
            chat_template,
        })
    }

    fn load_chat_template(config_path: &Path) -> Result<ChatTemplate, TokenizerError> {
        let data = std::fs::read_to_string(config_path)
            .map_err(|e| TokenizerError::Load(format!("tokenizer_config.json: {e}")))?;
        let config: TokenizerConfig = serde_json::from_str(&data)
            .map_err(|e| TokenizerError::Load(format!("tokenizer_config.json parse: {e}")))?;

        let template = config
            .chat_template
            .unwrap_or_else(|| CHATML_TEMPLATE.to_string());
        let bos = config
            .bos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        let eos = config
            .eos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();

        Ok(ChatTemplate::new(template, bos, eos))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn vocab_strings(&self) -> Vec<String> {
        let size = self.vocab_size();
        (0..size as u32)
            .map(|id| self.decode(&[id]).unwrap_or_default())
            .collect()
    }

    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessageInput],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        self.chat_template.apply(messages, add_generation_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_fallback_single_turn() {
        let tmpl = ChatTemplate::chatml();
        let messages = vec![ChatMessageInput {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = tmpl.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_chatml_multi_turn() {
        let tmpl = ChatTemplate::chatml();
        let messages = vec![
            ChatMessageInput {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessageInput {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
            ChatMessageInput {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
            },
            ChatMessageInput {
                role: "user".to_string(),
                content: "How are you?".to_string(),
            },
        ];
        let result = tmpl.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chatml_no_generation_prompt() {
        let tmpl = ChatTemplate::chatml();
        let messages = vec![ChatMessageInput {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = tmpl.apply(&messages, false).unwrap();
        assert!(!result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_style_template() {
        let llama3_template = "\
{%- set ns = namespace(found=false) -%}\
{%- for message in messages -%}\
{%- if message['role'] == 'system' -%}\
{{ bos_token }}system\n{{ message['content'] }}\n\
{%- endif -%}\
{%- endfor -%}\
{%- for message in messages -%}\
{%- if message['role'] == 'user' -%}\
{{ bos_token }}user\n{{ message['content'] }}\n\
{%- elif message['role'] == 'assistant' -%}\
{{ bos_token }}assistant\n{{ message['content'] }}{{ eos_token }}\n\
{%- endif -%}\
{%- endfor -%}\
{%- if add_generation_prompt %}\
{{ bos_token }}assistant\n\
{% endif %}";

        let tmpl = ChatTemplate::new(
            llama3_template.to_string(),
            "<|begin|>".to_string(),
            "<|end|>".to_string(),
        );
        let messages = vec![
            ChatMessageInput {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessageInput {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        let result = tmpl.apply(&messages, true).unwrap();
        assert!(result.contains("<|begin|>system\nBe concise."));
        assert!(result.contains("<|begin|>user\nHello"));
        assert!(result.ends_with("<|begin|>assistant\n"));
    }

    #[test]
    fn test_special_tokens_in_template() {
        let tmpl = ChatTemplate::new(
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
                .to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
        );
        let messages = vec![ChatMessageInput {
            role: "user".to_string(),
            content: "test".to_string(),
        }];
        let result = tmpl.apply(&messages, false).unwrap();
        assert_eq!(result, "<s>test</s>");
    }

    #[test]
    fn test_from_gguf_metadata_basic() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![
                GgufValue::String("hello".to_string()),
                GgufValue::String("world".to_string()),
                GgufValue::String("<s>".to_string()),
                GgufValue::String("</s>".to_string()),
            ]),
        );
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(2));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(3));

        let tok = Tokenizer::from_gguf_metadata(&metadata).unwrap();
        assert_eq!(tok.vocab_size(), 4);
    }

    #[test]
    fn test_from_gguf_metadata_with_merges() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![
                GgufValue::String("h".to_string()),
                GgufValue::String("e".to_string()),
                GgufValue::String("l".to_string()),
                GgufValue::String("o".to_string()),
                GgufValue::String("he".to_string()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufValue::Array(vec![GgufValue::String("h e".to_string())]),
        );

        let tok = Tokenizer::from_gguf_metadata(&metadata).unwrap();
        assert_eq!(tok.vocab_size(), 5);

        let ids = tok.encode("he").unwrap();
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn test_from_gguf_metadata_missing_tokens() {
        let metadata = HashMap::new();
        let result = Tokenizer::from_gguf_metadata(&metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_gguf_metadata_empty_tokens() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![]),
        );
        let result = Tokenizer::from_gguf_metadata(&metadata);
        assert!(result.is_err());
    }
}

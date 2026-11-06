use std::collections::HashMap;
use std::path::Path;

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
{% if enable_thinking %}<think>\n{% endif %}\
{% endif %}";

const CHATML_TEMPLATE_PLAIN: &str = "\
{%- for message in messages %}\
<|im_start|>{{ message.role }}\n\
{{ message.content }}<|im_end|>\n\
{%- endfor %}\
{%- if add_generation_prompt %}\
<|im_start|>assistant\n\
{% endif %}";

pub struct ChatTemplate {
    inner: hf_chat_template::ChatTemplate,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").finish()
    }
}

impl ChatTemplate {
    pub fn new(template: String, bos_token: String, eos_token: String) -> Self {
        let cfg = hf_chat_template::TokenizerConfig {
            bos_token: Some(hf_chat_template::TokenField::Str(bos_token)),
            eos_token: Some(hf_chat_template::TokenField::Str(eos_token)),
            ..Default::default()
        };
        let inner = hf_chat_template::ChatTemplate::builder(&template)
            .special_tokens_from(&cfg)
            .build()
            .unwrap_or_else(|e| {
                eprintln!("[chat-template] GGUF template failed: {e}, falling back to ChatML");
                hf_chat_template::ChatTemplate::from_str(CHATML_TEMPLATE).unwrap()
            });
        Self { inner }
    }

    pub fn chatml() -> Self {
        Self {
            inner: hf_chat_template::ChatTemplate::from_str(CHATML_TEMPLATE).unwrap(),
        }
    }

    pub fn apply(
        &self,
        messages: &[ChatMessageInput],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let msgs: Vec<hf_chat_template::Message> = messages
            .iter()
            .map(|m| hf_chat_template::Message::new(&m.role, &m.content))
            .collect();
        let mut input = hf_chat_template::RenderInput {
            messages: msgs,
            add_generation_prompt,
            ..Default::default()
        };
        input
            .extra
            .insert("enable_thinking".into(), serde_json::Value::Bool(true));
        self.inner
            .render(&input)
            .map_err(|e| TokenizerError::Template(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessageInput {
    pub role: String,
    pub content: String,
}

pub struct Tokenizer {
    /// HF tokenizers pipeline (BPE-family vocabs). `None` for SPM vocabs,
    /// which use the self-contained llama.cpp port in `spm` instead (the
    /// HF Tokenizer cannot host a custom Model: closed ModelWrapper enum).
    inner: Option<tokenizers::Tokenizer>,
    spm: Option<crate::spm_tokenizer::SpmModel>,
    chat_template: ChatTemplate,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    add_bos: bool,
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
            inner: Some(inner),
            spm: None,
            chat_template,
            bos_token_id: None,
            eos_token_id: None,
            add_bos: false,
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

        let scores = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array());

        let model_type = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| match v {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("");

        // SentencePiece path (tokenizer.ggml.model == "llama" with scores,
        // no merges): llama.cpp tokenizes SPM vocabs with score-greedy
        // bigram merges over escaped-space text (llama-vocab.cpp
        // llm_tokenizer_spm_session) -- i.e. unigram segmentation. The HF
        // tokenizers equivalent is Unigram + Metaspace + byte_fallback. The
        // previous always-BPE construction could NEVER form multi-char
        // pieces for these vocabs (no merges), so every character fell to
        // byte fallback: Mistral-7B v0.3 corpus tokenized to 1.29M tokens
        // vs llama.cpp's 329k and perplexity came out 8641 vs 5.96
        // (found 2026-11-06; Llama-3.1 masked it -- real BPE with merges).
        let is_spm = model_type == "llama" && scores.is_some();

        // SPM: self-contained llama.cpp port (see spm_tokenizer.rs for why
        // neither BPE-without-merges nor HF Unigram is correct here).
        let spm = if is_spm {
            let token_types_slice = metadata
                .get("tokenizer.ggml.token_type")
                .and_then(|v| v.as_u32_array());
            let add_dummy_prefix = match metadata.get("tokenizer.ggml.add_dummy_prefix") {
                Some(GgufValue::Bool(b)) => *b,
                _ => true,
            };
            Some(
                crate::spm_tokenizer::SpmModel::new(
                    &tokens,
                    scores.as_deref(),
                    token_types_slice.as_deref(),
                )
                .with_add_dummy_prefix(add_dummy_prefix),
            )
        } else {
            None
        };

        let mut inner = if is_spm {
            None
        } else {
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
                .map_err(|e| {
                    TokenizerError::Load(format!("failed to build BPE from GGUF: {e}"))
                })?;

            Some(tokenizers::Tokenizer::new(bpe))
        };

        // Register special tokens (type != 1) so they're matched as whole strings
        // before the pre-tokenizer splits them. BYTE tokens (type 6) are
        // excluded: byte_fallback emits their ids internally, and matching
        // the literal "<0xNN>" text to them is not SPM semantics.
        let token_types = metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_u32_array());
        if let Some(ref types) = token_types {
            if let Some(ref mut inner) = inner {
                let mut special_tokens = Vec::new();
                for (id, &tt) in types.iter().enumerate() {
                    if tt != 1 && tt != 6 && id < tokens.len() {
                        let tok_str = tokens[id].to_string();
                        if !tok_str.is_empty() {
                            let mut at = tokenizers::AddedToken::from(tok_str, true);
                            at = at.normalized(false);
                            special_tokens.push(at);
                        }
                    }
                }
                if !special_tokens.is_empty() {
                    inner.add_special_tokens(&special_tokens);
                }
            }
        }

        let is_gpt2_bpe = model_type == "gpt2";

        if let Some(ref mut inner) = inner {
            if is_gpt2_bpe {
                inner.with_pre_tokenizer(Some(
                    tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
                ));
                inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
            }
        }

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

        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());

        // BOS prepending: honour the explicit GGUF flag; when it is absent,
        // default per architecture the way llama.cpp does (llama.cpp's
        // LLM_ARCH defaults). Llama-3.1 HF GGUFs ship bos_token_id without
        // add_bos_token -- without this default the prompt loses its BOS and
        // generation degrades (worst on short prompts; found via the llama.cpp
        // arbiter in the bench-quality small-m prefill hunt, 2026-10-28).
        let add_bos = match metadata.get("tokenizer.ggml.add_bos_token") {
            Some(GgufValue::Bool(b)) => *b,
            _ => {
                let arch = metadata
                    .get("general.architecture")
                    .and_then(|v| match v {
                        GgufValue::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .unwrap_or("");
                matches!(arch, "llama" | "mistral" | "gemma" | "gemma2" | "gemma3")
            }
        };

        let gguf_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| match v {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            });
        // Detect thinking model: has <think> as a special token
        let is_thinking = token_types.as_ref().is_some_and(|types| {
            tokens
                .iter()
                .enumerate()
                .any(|(id, tok)| *tok == "<think>" && id < types.len() && types[id] != 1)
        });

        let template_str = gguf_template.unwrap_or_else(|| {
            if is_thinking {
                CHATML_TEMPLATE.to_string()
            } else {
                CHATML_TEMPLATE_PLAIN.to_string()
            }
        });
        let chat_template = ChatTemplate::new(template_str, bos_token, eos_token);

        Ok(Self {
            inner,
            spm,
            chat_template,
            bos_token_id,
            eos_token_id,
            add_bos,
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
        let ids: Vec<u32> = if let Some(spm) = &self.spm {
            spm.tokenize_text(text)
        } else {
            self.inner
                .as_ref()
                .expect("tokenizer has neither HF inner nor SPM model")
                .encode(text, false)
                .map_err(|e| TokenizerError::Encode(e.to_string()))?
                .get_ids()
                .to_vec()
        };
        if self.add_bos {
            if let Some(bos) = self.bos_token_id {
                let mut result = Vec::with_capacity(ids.len() + 1);
                result.push(bos);
                result.extend_from_slice(&ids);
                return Ok(result);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        if let Some(spm) = &self.spm {
            return Ok(spm.decode(ids));
        }
        self.inner
            .as_ref()
            .expect("tokenizer has neither HF inner nor SPM model")
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        if let Some(spm) = &self.spm {
            return spm.vocab_len();
        }
        self.inner
            .as_ref()
            .expect("tokenizer has neither HF inner nor SPM model")
            .get_vocab_size(true)
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
        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
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

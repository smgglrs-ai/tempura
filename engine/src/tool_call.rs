use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(m) if m == "none")
    }

    pub fn is_required(&self) -> bool {
        matches!(self, ToolChoice::Mode(m) if m == "required")
    }

    pub fn specific_function(&self) -> Option<&str> {
        match self {
            ToolChoice::Specific { function, .. } => Some(&function.name),
            _ => None,
        }
    }
}

pub trait ToolCallParserTrait {
    fn parse(&self, text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>>;
}

pub struct ToolCallParser<'a> {
    tools: &'a [ToolDefinition],
    parser: Box<dyn ToolCallParserTrait>,
}

impl<'a> ToolCallParser<'a> {
    pub fn new(tools: &'a [ToolDefinition]) -> Self {
        Self {
            tools,
            parser: Box::new(GenericParser),
        }
    }

    pub fn for_model(tools: &'a [ToolDefinition], model_family: &str) -> Self {
        let parser: Box<dyn ToolCallParserTrait> = match model_family {
            "hermes" => Box::new(HermesParser),
            "mistral" => Box::new(MistralParser),
            "llama3" | "llama-3" => Box::new(Llama3Parser),
            _ => Box::new(GenericParser),
        };
        Self { tools, parser }
    }

    pub fn parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        self.parser.parse(text, self.tools)
    }
}

// --- Generic Parser (JSON objects) ---

struct GenericParser;

impl ToolCallParserTrait for GenericParser {
    fn parse(&self, text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
        let trimmed = text.trim();

        if let Some(calls) = try_parse_json_array(trimmed, tools) {
            return Some(calls);
        }
        if let Some(call) = try_parse_json_single(trimmed, tools) {
            return Some(vec![call]);
        }
        try_extract_json_from_text(trimmed, tools)
    }
}

// --- Hermes Parser (<tool_call> XML tags) ---

struct HermesParser;

impl ToolCallParserTrait for HermesParser {
    fn parse(&self, text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();
        let mut search = text;
        let mut idx = 0;

        while let Some(start) = search.find("<tool_call>") {
            let after_tag = &search[start + 11..];
            if let Some(end) = after_tag.find("</tool_call>") {
                let content = after_tag[..end].trim();
                if let Ok(raw) = serde_json::from_str::<RawToolCall>(content) {
                    if tools.iter().any(|t| t.function.name == raw.name) {
                        calls.push(raw_to_tool_call(raw, idx));
                        idx += 1;
                    }
                }
                search = &after_tag[end + 12..];
            } else {
                break;
            }
        }

        if calls.is_empty() {
            GenericParser.parse(text, tools)
        } else {
            Some(calls)
        }
    }
}

// --- Mistral Parser ([TOOL_CALLS] marker) ---

struct MistralParser;

impl ToolCallParserTrait for MistralParser {
    fn parse(&self, text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
        if let Some(pos) = text.find("[TOOL_CALLS]") {
            let after = text[pos + 12..].trim();
            if let Some(calls) = try_parse_json_array(after, tools) {
                return Some(calls);
            }
            if let Some(call) = try_parse_json_single(after, tools) {
                return Some(vec![call]);
            }
        }
        GenericParser.parse(text, tools)
    }
}

// --- LLaMA 3 Parser (JSON blocks, <|python_tag|>) ---

struct Llama3Parser;

impl ToolCallParserTrait for Llama3Parser {
    fn parse(&self, text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
        let content = if let Some(pos) = text.find("<|python_tag|>") {
            text[pos + 14..].trim()
        } else {
            text.trim()
        };

        if let Some(calls) = try_parse_json_array(content, tools) {
            return Some(calls);
        }
        if let Some(call) = try_parse_json_single(content, tools) {
            return Some(vec![call]);
        }
        try_extract_json_from_text(content, tools)
    }
}

// --- Shared helpers ---

#[derive(Debug, Deserialize)]
struct RawToolCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

fn raw_to_tool_call(raw: RawToolCall, index: usize) -> ToolCall {
    let arguments = match raw.arguments {
        serde_json::Value::String(s) => s,
        other => serde_json::to_string(&other).unwrap_or_default(),
    };
    ToolCall {
        id: format!("call_{index}"),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: raw.name,
            arguments,
        },
    }
}

fn try_parse_json_array(text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
    if !text.starts_with('[') {
        return None;
    }
    let raw: Vec<RawToolCall> = serde_json::from_str(text).ok()?;
    let calls: Vec<ToolCall> = raw
        .into_iter()
        .filter(|r| tools.iter().any(|t| t.function.name == r.name))
        .enumerate()
        .map(|(i, r)| raw_to_tool_call(r, i))
        .collect();
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn try_parse_json_single(text: &str, tools: &[ToolDefinition]) -> Option<ToolCall> {
    if !text.starts_with('{') {
        return None;
    }
    let raw: RawToolCall = serde_json::from_str(text).ok()?;
    if tools.iter().any(|t| t.function.name == raw.name) {
        Some(raw_to_tool_call(raw, 0))
    } else {
        None
    }
}

fn try_extract_json_from_text(text: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut search = text;
    let mut idx = 0;

    while let Some(start) = search.find('{') {
        let remaining = &search[start..];
        if let Some(end) = find_matching_brace(remaining) {
            let json_str = &remaining[..=end];
            if let Ok(raw) = serde_json::from_str::<RawToolCall>(json_str) {
                if tools.iter().any(|t| t.function.name == raw.name) {
                    calls.push(raw_to_tool_call(raw, idx));
                    idx += 1;
                }
            }
            search = &remaining[end + 1..];
        } else {
            break;
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

pub fn detect_model_family(tokenizer_config: Option<&serde_json::Value>) -> &'static str {
    if let Some(config) = tokenizer_config {
        if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
            if template.contains("<tool_call>") || template.contains("hermes") {
                return "hermes";
            }
            if template.contains("[TOOL_CALLS]") || template.contains("[AVAILABLE_TOOLS]") {
                return "mistral";
            }
            if template.contains("<|python_tag|>") || template.contains("ipython") {
                return "llama3";
            }
        }
    }
    "generic"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "get_weather".to_string(),
                    description: Some("Get weather for a location".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    })),
                },
            },
            ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "search".to_string(),
                    description: Some("Search the web".to_string()),
                    parameters: None,
                },
            },
        ]
    }

    #[test]
    fn test_generic_parse_single() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_generic_parse_array() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"[{"name": "get_weather", "arguments": {"location": "Paris"}}, {"name": "search", "arguments": {"query": "news"}}]"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generic_unknown_function() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        assert!(parser
            .parse(r#"{"name": "unknown", "arguments": {}}"#)
            .is_none());
    }

    #[test]
    fn test_generic_plain_text() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        assert!(parser.parse("Hello, how are you?").is_none());
    }

    #[test]
    fn test_hermes_parser() {
        let tools = sample_tools();
        let parser = ToolCallParser::for_model(&tools, "hermes");
        let text = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "London"}}</tool_call>"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_hermes_multiple_calls() {
        let tools = sample_tools();
        let parser = ToolCallParser::for_model(&tools, "hermes");
        let text = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>
<tool_call>{"name": "search", "arguments": {"query": "news"}}</tool_call>"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_mistral_parser() {
        let tools = sample_tools();
        let parser = ToolCallParser::for_model(&tools, "mistral");
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Berlin"}}]"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama3_parser() {
        let tools = sample_tools();
        let parser = ToolCallParser::for_model(&tools, "llama3");
        let text = r#"<|python_tag|>{"name": "get_weather", "arguments": {"location": "Tokyo"}}"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama3_fallback_to_json() {
        let tools = sample_tools();
        let parser = ToolCallParser::for_model(&tools, "llama3");
        let text = r#"{"name": "search", "arguments": {"query": "test"}}"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_detect_model_family() {
        assert_eq!(
            detect_model_family(Some(
                &serde_json::json!({"chat_template": "some <tool_call> template"})
            )),
            "hermes"
        );
        assert_eq!(
            detect_model_family(Some(
                &serde_json::json!({"chat_template": "[AVAILABLE_TOOLS] stuff"})
            )),
            "mistral"
        );
        assert_eq!(
            detect_model_family(Some(
                &serde_json::json!({"chat_template": "<|python_tag|> code"})
            )),
            "llama3"
        );
        assert_eq!(detect_model_family(None), "generic");
    }

    #[test]
    fn test_tool_choice_modes() {
        let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        assert!(!auto.is_none());
        assert!(!auto.is_required());

        let none: ToolChoice = serde_json::from_str(r#""none""#).unwrap();
        assert!(none.is_none());

        let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();
        assert!(required.is_required());

        let specific: ToolChoice =
            serde_json::from_str(r#"{"type": "function", "function": {"name": "test"}}"#).unwrap();
        assert_eq!(specific.specific_function(), Some("test"));
    }

    #[test]
    fn test_string_arguments() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"{"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall {
            id: "call_0".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "test".to_string(),
                arguments: r#"{"x": 1}"#.to_string(),
            },
        };
        let json = serde_json::to_string(&call).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "test");
    }
}

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

pub struct ToolCallParser<'a> {
    tools: &'a [ToolDefinition],
}

impl<'a> ToolCallParser<'a> {
    pub fn new(tools: &'a [ToolDefinition]) -> Self {
        Self { tools }
    }

    pub fn parse(&self, text: &str) -> Option<Vec<ToolCall>> {
        let trimmed = text.trim();

        if let Some(calls) = self.try_parse_array(trimmed) {
            return Some(calls);
        }

        if let Some(call) = self.try_parse_single(trimmed) {
            return Some(vec![call]);
        }

        if let Some(calls) = self.try_extract_from_text(trimmed) {
            return Some(calls);
        }

        None
    }

    fn try_parse_array(&self, text: &str) -> Option<Vec<ToolCall>> {
        if !text.starts_with('[') {
            return None;
        }
        let raw: Vec<RawToolCall> = serde_json::from_str(text).ok()?;
        let calls: Vec<ToolCall> = raw
            .into_iter()
            .filter(|r| self.tools.iter().any(|t| t.function.name == r.name))
            .enumerate()
            .map(|(i, r)| self.raw_to_tool_call(r, i))
            .collect();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn try_parse_single(&self, text: &str) -> Option<ToolCall> {
        if !text.starts_with('{') {
            return None;
        }
        let raw: RawToolCall = serde_json::from_str(text).ok()?;
        if self.tools.iter().any(|t| t.function.name == raw.name) {
            Some(self.raw_to_tool_call(raw, 0))
        } else {
            None
        }
    }

    fn try_extract_from_text(&self, text: &str) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();
        let mut search = text;
        let mut idx = 0;

        while let Some(start) = search.find('{') {
            let remaining = &search[start..];
            if let Some(end) = find_matching_brace(remaining) {
                let json_str = &remaining[..=end];
                if let Ok(raw) = serde_json::from_str::<RawToolCall>(json_str) {
                    if self.tools.iter().any(|t| t.function.name == raw.name) {
                        calls.push(self.raw_to_tool_call(raw, idx));
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

    fn raw_to_tool_call(&self, raw: RawToolCall, index: usize) -> ToolCall {
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
}

#[derive(Debug, Deserialize)]
struct RawToolCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
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
    fn test_parse_single_tool_call() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_parse_array_of_tool_calls() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"[{"name": "get_weather", "arguments": {"location": "Paris"}}, {"name": "search", "arguments": {"query": "news"}}]"#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "search");
    }

    #[test]
    fn test_parse_unknown_function_returns_none() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"{"name": "unknown_func", "arguments": {}}"#;
        assert!(parser.parse(text).is_none());
    }

    #[test]
    fn test_parse_plain_text_returns_none() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        assert!(parser.parse("Hello, how are you?").is_none());
    }

    #[test]
    fn test_parse_embedded_in_text() {
        let tools = sample_tools();
        let parser = ToolCallParser::new(&tools);
        let text = r#"I'll check the weather. {"name": "get_weather", "arguments": {"location": "London"}} Let me know."#;
        let calls = parser.parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
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
}

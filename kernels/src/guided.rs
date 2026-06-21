pub use crate::GuidedDecoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    ExpectValue,
    InString,
    InStringEscape,
    InNumber,
    InLiteral,
    AfterValue,
    ExpectKey,
    ExpectColon,
}

pub struct JsonGuidedDecoder {
    vocab: Vec<String>,
    stack: Vec<u8>,
    state: JsonState,
    literal_remaining: Vec<u8>,
    generated_text: String,
    complete: bool,
    last_masked: usize,
}

impl JsonGuidedDecoder {
    pub fn new(vocab: Vec<String>) -> Self {
        Self {
            vocab,
            stack: Vec::new(),
            state: JsonState::ExpectValue,
            literal_remaining: Vec::new(),
            generated_text: String::new(),
            complete: false,
            last_masked: 0,
        }
    }

    pub fn from_schema(vocab: Vec<String>, _schema: &serde_json::Value) -> Self {
        Self::new(vocab)
    }

    fn is_valid_continuation(&self, text: &str) -> bool {
        let mut stack = self.stack.clone();
        let mut state = self.state;
        let mut literal_remaining = self.literal_remaining.clone();

        for ch in text.chars() {
            match Self::advance_state(state, ch, &mut stack, &mut literal_remaining) {
                Some(new_state) => state = new_state,
                None => return false,
            }
        }
        true
    }

    fn advance_state(
        state: JsonState,
        ch: char,
        stack: &mut Vec<u8>,
        literal_remaining: &mut Vec<u8>,
    ) -> Option<JsonState> {
        match state {
            JsonState::ExpectValue => match ch {
                '"' => Some(JsonState::InString),
                '{' => {
                    stack.push(b'}');
                    Some(JsonState::ExpectKey)
                }
                '[' => {
                    stack.push(b']');
                    Some(JsonState::ExpectValue)
                }
                '0'..='9' | '-' => Some(JsonState::InNumber),
                't' => {
                    *literal_remaining = b"rue".to_vec();
                    Some(JsonState::InLiteral)
                }
                'f' => {
                    *literal_remaining = b"alse".to_vec();
                    Some(JsonState::InLiteral)
                }
                'n' => {
                    *literal_remaining = b"ull".to_vec();
                    Some(JsonState::InLiteral)
                }
                ' ' | '\n' | '\r' | '\t' => Some(JsonState::ExpectValue),
                ']' if stack.last() == Some(&b']') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                _ => None,
            },
            JsonState::InString => match ch {
                '"' => Some(JsonState::AfterValue),
                '\\' => Some(JsonState::InStringEscape),
                '\x00'..='\x1f' => None,
                _ => Some(JsonState::InString),
            },
            JsonState::InStringEscape => match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => Some(JsonState::InString),
                _ => None,
            },
            JsonState::InNumber => match ch {
                '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => Some(JsonState::InNumber),
                ',' => {
                    if stack.last() == Some(&b']') {
                        Some(JsonState::ExpectValue)
                    } else if stack.last() == Some(&b'}') {
                        Some(JsonState::ExpectKey)
                    } else {
                        None
                    }
                }
                '}' if stack.last() == Some(&b'}') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                ']' if stack.last() == Some(&b']') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                ' ' | '\n' | '\r' | '\t' => Some(JsonState::AfterValue),
                _ => None,
            },
            JsonState::InLiteral => {
                if literal_remaining.is_empty() {
                    return Self::advance_state(
                        JsonState::AfterValue,
                        ch,
                        stack,
                        literal_remaining,
                    );
                }
                if ch as u8 == literal_remaining[0] {
                    literal_remaining.remove(0);
                    if literal_remaining.is_empty() {
                        Some(JsonState::AfterValue)
                    } else {
                        Some(JsonState::InLiteral)
                    }
                } else {
                    None
                }
            }
            JsonState::AfterValue => match ch {
                ',' => {
                    if stack.last() == Some(&b']') {
                        Some(JsonState::ExpectValue)
                    } else if stack.last() == Some(&b'}') {
                        Some(JsonState::ExpectKey)
                    } else {
                        None
                    }
                }
                '}' if stack.last() == Some(&b'}') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                ']' if stack.last() == Some(&b']') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                ' ' | '\n' | '\r' | '\t' => Some(JsonState::AfterValue),
                _ => None,
            },
            JsonState::ExpectKey => match ch {
                '"' => Some(JsonState::InString),
                ' ' | '\n' | '\r' | '\t' => Some(JsonState::ExpectKey),
                '}' if stack.last() == Some(&b'}') => {
                    stack.pop();
                    Some(JsonState::AfterValue)
                }
                _ => None,
            },
            JsonState::ExpectColon => match ch {
                ':' => Some(JsonState::ExpectValue),
                ' ' | '\n' | '\r' | '\t' => Some(JsonState::ExpectColon),
                _ => None,
            },
        }
    }

    fn update_internal_state(&mut self, text: &str) {
        for ch in text.chars() {
            if let Some(new_state) =
                Self::advance_state(self.state, ch, &mut self.stack, &mut self.literal_remaining)
            {
                if self.state == JsonState::InString
                    && new_state == JsonState::AfterValue
                    && self.stack.last() == Some(&b'}')
                    && self.needs_colon_after(ch)
                {
                    self.state = JsonState::ExpectColon;
                } else {
                    self.state = new_state;
                }
            }
            self.generated_text.push(ch);
        }
        self.complete = self.stack.is_empty()
            && (self.state == JsonState::AfterValue
                || (self.state == JsonState::InNumber && !self.generated_text.is_empty()));
    }

    fn forced_string(&self) -> String {
        let mut forced = String::new();

        if self.state == JsonState::InLiteral && !self.literal_remaining.is_empty() {
            for &b in &self.literal_remaining {
                forced.push(b as char);
            }
        }

        if !forced.is_empty() {
            return forced;
        }

        if self.state == JsonState::ExpectColon {
            forced.push(':');
        }

        if !forced.is_empty() {
            return forced;
        }

        if self.state == JsonState::AfterValue && self.stack.is_empty() {
            return forced;
        }

        forced
    }

    fn needs_colon_after(&self, _closing_quote: char) -> bool {
        // Count quotes in current object context to determine if this is a key or value
        let mut quote_count = 0u32;
        let mut depth = 0i32;
        for ch in self.generated_text.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    if depth == self.stack.len() as i32 + 1 {
                        quote_count = 0;
                    }
                }
                '}' => depth -= 1,
                '"' if depth == self.stack.len() as i32 => {
                    quote_count += 1;
                }
                ',' if depth == self.stack.len() as i32 => {
                    quote_count = 0;
                }
                _ => {}
            }
        }
        // quote_count is odd means the closing quote completes a key (open=1st, close=2nd=key done)
        quote_count % 2 == 1
    }
}

impl GuidedDecoder for JsonGuidedDecoder {
    fn mask_logits(&mut self, token_ids_so_far: &[u32], logits: &mut [f32]) {
        if self.complete {
            return;
        }

        // Reconstruct current state from previously generated tokens
        if token_ids_so_far.len() == 1 || self.generated_text.is_empty() {
            self.stack.clear();
            self.state = JsonState::ExpectValue;
            self.literal_remaining.clear();
            self.generated_text.clear();
            self.complete = false;

            let prev_tokens: Vec<String> = token_ids_so_far
                [..token_ids_so_far.len().saturating_sub(1)]
                .iter()
                .filter_map(|&tok| self.vocab.get(tok as usize).cloned())
                .collect();
            for text in &prev_tokens {
                self.update_internal_state(text);
            }
        }

        let mut masked = 0;
        for (i, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue;
            }
            if let Some(text) = self.vocab.get(i) {
                if text.is_empty() || !self.is_valid_continuation(text) {
                    *logit = f32::NEG_INFINITY;
                    masked += 1;
                }
            } else {
                *logit = f32::NEG_INFINITY;
                masked += 1;
            }
        }
        self.last_masked = masked;

        // Update state with the last token that will be selected
        // (this is called before sampling, so we defer state update)
    }

    fn is_complete(&self) -> bool {
        self.complete
    }

    fn tokens_masked_last_step(&self) -> usize {
        self.last_masked
    }

    fn forced_tokens(&self) -> Option<Vec<u32>> {
        if self.complete {
            return None;
        }

        let forced_str = self.forced_string();
        if forced_str.is_empty() {
            return None;
        }

        let mut tokens = Vec::new();
        let mut remaining = forced_str.as_str();
        while !remaining.is_empty() {
            let mut best_len = 0;
            let mut best_tok = None;
            for (i, text) in self.vocab.iter().enumerate() {
                if !text.is_empty() && remaining.starts_with(text.as_str()) && text.len() > best_len
                {
                    best_len = text.len();
                    best_tok = Some(i as u32);
                }
            }
            match best_tok {
                Some(tok) => {
                    tokens.push(tok);
                    remaining = &remaining[best_len..];
                }
                None => break,
            }
        }

        if tokens.is_empty() {
            None
        } else {
            Some(tokens)
        }
    }
}

pub fn create_json_decoder(vocab: Vec<String>) -> Box<dyn GuidedDecoder> {
    Box::new(JsonGuidedDecoder::new(vocab))
}

pub fn create_json_schema_decoder(
    vocab: Vec<String>,
    schema: &serde_json::Value,
) -> Box<dyn GuidedDecoder> {
    Box::new(JsonGuidedDecoder::from_schema(vocab, schema))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_vocab() -> Vec<String> {
        vec![
            "{".into(),    // 0
            "}".into(),    // 1
            "\"".into(),   // 2
            ":".into(),    // 3
            ",".into(),    // 4
            " ".into(),    // 5
            "a".into(),    // 6
            "b".into(),    // 7
            "1".into(),    // 8
            "2".into(),    // 9
            "[".into(),    // 10
            "]".into(),    // 11
            "true".into(), // 12
            "null".into(), // 13
            "\n".into(),   // 14
        ]
    }

    #[test]
    fn test_json_state_machine_object() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);

        // At start, we expect a value: {, [, ", digit, t, f, n, whitespace
        let mut logits = vec![0.0f32; 15];
        decoder.mask_logits(&[], &mut logits);

        assert!(logits[0] != f32::NEG_INFINITY, "{{ should be allowed");
        assert!(logits[10] != f32::NEG_INFINITY, "[ should be allowed");
        assert!(logits[2] != f32::NEG_INFINITY, "\" should be allowed");
        assert!(logits[12] != f32::NEG_INFINITY, "true should be allowed");
        assert!(logits[3] == f32::NEG_INFINITY, ": should be masked");
    }

    #[test]
    fn test_valid_continuation_simple() {
        let vocab = simple_vocab();
        let decoder = JsonGuidedDecoder::new(vocab);

        assert!(decoder.is_valid_continuation("{"));
        assert!(decoder.is_valid_continuation("\""));
        assert!(decoder.is_valid_continuation("["));
        assert!(!decoder.is_valid_continuation("}"));
        assert!(!decoder.is_valid_continuation(":"));
    }

    #[test]
    fn test_is_complete_after_object() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("{}");
        assert!(decoder.is_complete());
    }

    #[test]
    fn test_is_complete_not_mid_object() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("{\"a\"");
        assert!(!decoder.is_complete());
    }

    #[test]
    fn test_logit_masking_count() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        let mut logits = vec![1.0f32; 15];
        decoder.mask_logits(&[], &mut logits);
        assert!(decoder.tokens_masked_last_step() > 0);
        let allowed = logits.iter().filter(|&&l| l != f32::NEG_INFINITY).count();
        assert!(allowed > 0, "at least some tokens must be allowed");
    }

    #[test]
    fn test_string_is_valid_json() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("\"ab\"");
        assert!(decoder.is_complete());
    }

    #[test]
    fn test_number_is_valid_json() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("12");
        assert!(decoder.is_complete());
    }

    #[test]
    fn test_array_is_valid_json() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("[1,2]");
        assert!(decoder.is_complete());
    }

    #[test]
    fn test_literal_empty_remaining_no_panic() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("true");
        assert!(decoder.is_complete());
        assert!(decoder.literal_remaining.is_empty());
    }

    #[test]
    fn test_all_valid_json_forms_complete() {
        let vocab = simple_vocab();
        let cases = vec![
            "{}",
            "[]",
            "\"ab\"",
            "12",
            "true",
            "null",
            "{\"a\":1}",
            "[1,2]",
        ];
        for json in cases {
            let mut decoder = JsonGuidedDecoder::new(vocab.clone());
            decoder.update_internal_state(json);
            assert!(decoder.is_complete(), "{json} should be complete");
        }
    }

    #[test]
    fn test_invalid_transitions_return_none() {
        let mut stack = Vec::new();
        let mut literal = Vec::new();
        assert!(JsonGuidedDecoder::advance_state(
            JsonState::ExpectValue,
            '}',
            &mut stack,
            &mut literal
        )
        .is_none());
        assert!(JsonGuidedDecoder::advance_state(
            JsonState::ExpectColon,
            'a',
            &mut stack,
            &mut literal
        )
        .is_none());
        assert!(JsonGuidedDecoder::advance_state(
            JsonState::InStringEscape,
            'z',
            &mut stack,
            &mut literal
        )
        .is_none());
    }

    #[test]
    fn test_forced_tokens_literal() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("t");
        // After 't', the grammar forces "rue"
        assert_eq!(decoder.state, JsonState::InLiteral);
        assert_eq!(decoder.literal_remaining, b"rue");
        let forced = decoder.forced_string();
        assert_eq!(forced, "rue");
    }

    #[test]
    fn test_forced_tokens_colon() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("{\"a\"");
        assert_eq!(decoder.state, JsonState::ExpectColon);
        let forced = decoder.forced_string();
        assert_eq!(forced, ":");
        let tokens = decoder.forced_tokens();
        assert!(tokens.is_some());
        assert_eq!(tokens.unwrap(), vec![3]); // ":" is token 3
    }

    #[test]
    fn test_forced_tokens_none_in_string() {
        let vocab = simple_vocab();
        let mut decoder = JsonGuidedDecoder::new(vocab);
        decoder.update_internal_state("{\"");
        assert_eq!(decoder.state, JsonState::InString);
        assert!(decoder.forced_tokens().is_none());
    }

    #[test]
    fn test_no_panic_on_any_ascii_continuation() {
        let vocab = simple_vocab();
        let decoder = JsonGuidedDecoder::new(vocab);
        for byte in 0u8..=127 {
            let s = String::from(byte as char);
            let _ = decoder.is_valid_continuation(&s);
        }
    }
}

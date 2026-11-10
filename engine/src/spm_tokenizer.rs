//! Exact port of llama.cpp's SentencePiece tokenizer
//! (llm_tokenizer_spm_session in ggml/src/llama-vocab.cpp): greedy
//! score-priority bigram merging over UTF-8 symbols with byte fallback.
//!
//! HF tokenizers' Unigram (Viterbi over score sums) produces DIFFERENT
//! segmentations for these vocabs: in Mistral-7B v0.3, "▁R"+"ob"+"ert"
//! (-140-339-671 = -1150) beats "▁Robert" (-5347), but llama.cpp's greedy
//! merge yields "▁Robert". Since llama.cpp is this project's designated
//! ground truth, its algorithm is implemented directly. The HF
//! `tokenizers::Tokenizer` cannot host a custom Model (closed ModelWrapper
//! enum), so this is a self-contained pipeline mirroring llama.cpp's
//! fragment buffer: special-token partition -> whitespace escaping ->
//! bigram merge -> resegment with byte fallback.

use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
pub(crate) struct SpmModel {
    /// Normal (GGUF token_type 1) pieces: text -> (id, score).
    pieces: HashMap<String, (u32, f32)>,
    /// Byte-fallback token ids, parsed from "<0xNN>" (token_type 6).
    byte_tokens: [u32; 256],
    /// id -> token string, all tokens.
    all_tokens: Vec<String>,
    /// Special tokens (types 2/3/4), longest-first for partitioning.
    special_tokens: Vec<(String, u32)>,
    /// tokenizer.ggml.add_dummy_prefix.
    /// USER_DEFINED (type 4) tokens, longest-first: llama.cpp matches them
    /// as RAW-TEXT prefixes before whitespace escaping (Gemma's "  " and
    /// newline pieces) -- user_defined_token_matcher in llama-vocab.cpp.
    user_defined: Vec<(String, u32)>,
    add_dummy_prefix: bool,
    unk_id: Option<u32>,
}

impl SpmModel {
    pub(crate) fn new(
        tokens: &[&str],
        scores: Option<&[f32]>,
        token_types: Option<&[u32]>,
    ) -> Self {
        let mut pieces = HashMap::with_capacity(tokens.len());
        let mut byte_tokens = [0u32; 256];
        let mut special_tokens = Vec::new();
        let mut user_defined = Vec::new();
        for (id, token) in tokens.iter().enumerate() {
            let score = scores.and_then(|s| s.get(id)).copied().unwrap_or(-10.0);
            let tt = token_types.and_then(|t| t.get(id)).copied().unwrap_or(1);
            match tt {
                1 => {
                    pieces.insert((*token).to_string(), (id as u32, score));
                }
                // USER_DEFINED (4): matched as RAW-text prefixes before
                // escaping (see user_defined) AND present in the merge map
                // (llama.cpp's token_to_id has no type filter).
                4 => {
                    pieces.insert((*token).to_string(), (id as u32, score));
                    user_defined.push(((*token).to_string(), id as u32));
                }
                6 => {
                    if let Some(hex) = token
                        .strip_prefix("<0x")
                        .and_then(|h| h.strip_suffix('>'))
                    {
                        if let Ok(b) = u8::from_str_radix(hex, 16) {
                            byte_tokens[b as usize] = id as u32;
                        }
                    }
                }
                2 | 3 => special_tokens.push(((*token).to_string(), id as u32)),
                _ => {}
            }
        }
        // Longest-first so partitioning prefers the longest special token.
        special_tokens.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        // Longest-first raw-prefix matching (llama.cpp's
        // user_defined_token_matcher.get_longest_prefix).
        user_defined.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let unk_id = tokens.iter().position(|t| *t == "<unk>").map(|i| i as u32);
        Self {
            pieces,
            byte_tokens,
            all_tokens: tokens.iter().map(|s| s.to_string()).collect(),
            special_tokens,
            user_defined,
            add_dummy_prefix: true,
            unk_id,
        }
    }

    pub(crate) fn with_add_dummy_prefix(mut self, add: bool) -> Self {
        self.add_dummy_prefix = add;
        self
    }

    /// Full encode pipeline, mirroring llama.cpp: partition on special
    /// tokens, escape whitespace (llama_escape_whitespace: ' ' -> U+2581,
    /// plus the dummy prefix), then the bigram merge over each raw segment.
    pub(crate) fn tokenize_text(&self, text: &str) -> Vec<u32> {
        let mut output = Vec::new();
        // Special-token partition (tokenizer_st_partition): raw text between
        // specials is escaped and merged; specials emit their ids verbatim.
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (special, id) in &self.special_tokens {
                if let Some(pos) = rest.find(special.as_str()) {
                    if pos > 0 {
                        self.tokenize_raw(&rest[..pos], &mut output);
                    }
                    output.push(*id);
                    rest = &rest[pos + special.len()..];
                    continue 'outer;
                }
            }
            self.tokenize_raw(rest, &mut output);
            break;
        }
        output
    }

    /// llama_escape_whitespace + llm_tokenizer_spm_session::tokenize, with
    /// USER_DEFINED raw-text prefix matching first (llama.cpp's
    /// normalize_prefix checks user_defined_token_matcher BEFORE escaping --
    /// Gemma's "  " / newline pieces carry real whitespace/newlines and can
    /// never match the escaped text).
    fn tokenize_raw(&self, text: &str, output: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Fast path: no user-defined tokens -- exactly the old pipeline.
        if self.user_defined.is_empty() {
            self.escape_merge(text, true, output);
            return;
        }
        let bytes = text.as_bytes();
        let mut chunk_start = 0usize;
        let mut i = 0usize;
        let mut first_chunk = true;
        while i < bytes.len() {
            let mut matched = None;
            for (s, id) in &self.user_defined {
                if text[i..].starts_with(s.as_str()) {
                    matched = Some((s.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if i > chunk_start {
                    self.escape_merge(&text[chunk_start..i], first_chunk, output);
                    first_chunk = false;
                }
                output.push(id);
                i += len;
                chunk_start = i;
            } else {
                i += utf8_len(bytes[i]).min(bytes.len() - i);
            }
        }
        if chunk_start < text.len() {
            self.escape_merge(&text[chunk_start..], first_chunk, output);
        }
    }

    /// Escape whitespace (+ optional dummy prefix on the first chunk only --
    /// llama.cpp prepends it once per input, not per user-defined split) and
    /// run the bigram merge.
    fn escape_merge(&self, text: &str, prepend_dummy: bool, output: &mut Vec<u32>) {
        let escaped = text.replace(' ', "\u{2581}");
        let escaped = if prepend_dummy && self.add_dummy_prefix {
            format!("\u{2581}{}", escaped)
        } else {
            escaped
        };
        self.merge_tokenize(&escaped, output);
    }

    /// The bigram merge (port of llm_tokenizer_spm_session::tokenize).
    fn merge_tokenize(&self, sentence: &str, output: &mut Vec<u32>) {
        let bytes = sentence.as_bytes();
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offs = 0usize;
        while offs < bytes.len() {
            let len = utf8_len(bytes[offs]).min(bytes.len() - offs);
            let idx = symbols.len();
            symbols.push(Symbol {
                start: offs,
                len,
                prev: idx as isize - 1,
                next: -1,
            });
            if idx > 0 {
                symbols[idx - 1].next = idx as isize;
            }
            offs += len;
        }

        let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();
        let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();

        for i in 1..symbols.len() {
            self.try_add_bigram(
                sentence,
                &symbols,
                &mut queue,
                &mut rev_merge,
                i as isize - 1,
                i as isize,
            );
        }

        while let Some(bigram) = queue.pop() {
            let (l, r) = (bigram.left, bigram.right);
            if symbols[l].len == 0
                || symbols[r].len == 0
                || symbols[l].len + symbols[r].len != bigram.size
            {
                continue; // stale
            }
            let new_next = symbols[r].next;
            symbols[l].len += symbols[r].len;
            symbols[r].len = 0;
            symbols[l].next = new_next;
            if new_next >= 0 {
                symbols[new_next as usize].prev = l as isize;
            }
            let prev = symbols[l].prev;
            let next = symbols[l].next;
            self.try_add_bigram(sentence, &symbols, &mut queue, &mut rev_merge, prev, l as isize);
            self.try_add_bigram(sentence, &symbols, &mut queue, &mut rev_merge, l as isize, next);
        }

        // Walk the final symbol list; resegment non-tokens via rev_merge,
        // byte-fallback anything else. Explicit stack: merge chains can be
        // deep (the C++ is deliberately non-recursive).
        let mut todo: Vec<Symbol> = Vec::new();
        let mut i = 0isize;
        while i != -1 {
            todo.push(symbols[i as usize]);
            while let Some(cur) = todo.pop() {
                if cur.len == 0 {
                    continue;
                }
                let text = &sentence[cur.start..cur.start + cur.len];
                if let Some((id, _)) = self.pieces.get(text) {
                    output.push(*id);
                    continue;
                }
                if let Some(&(l, r)) = rev_merge.get(text) {
                    // C++ pushes right then left so the LEFT is processed
                    // first (LIFO).
                    todo.push(symbols[r]);
                    todo.push(symbols[l]);
                    continue;
                }
                // byte fallback
                for &b in sentence[cur.start..cur.start + cur.len].as_bytes() {
                    let id = self.byte_tokens[b as usize];
                    if id != 0 {
                        output.push(id);
                    } else if let Some(unk) = self.unk_id {
                        output.push(unk);
                    }
                }
            }
            i = symbols[i as usize].next;
        }
    }

    fn try_add_bigram(
        &self,
        sentence: &str,
        symbols: &[Symbol],
        queue: &mut BinaryHeap<Bigram>,
        rev_merge: &mut HashMap<String, (usize, usize)>,
        left: isize,
        right: isize,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let start = symbols[l].start;
        let end = symbols[r].start + symbols[r].len;
        let text = &sentence[start..end];
        if let Some((_, score)) = self.pieces.get(text) {
            queue.push(Bigram {
                left: l,
                right: r,
                score: *score,
                size: text.len(),
            });
            rev_merge.insert(text.to_string(), (l, r));
        }
    }

    pub(crate) fn vocab_len(&self) -> usize {
        self.all_tokens.len()
    }

    /// Decode: pieces joined with U+2581 -> ' ', byte tokens to raw bytes,
    /// special ids skipped (skip_special_tokens semantics). The dummy
    /// prefix is dropped (SentencePiece roundtrip semantics:
    /// decode(encode(x)) == x).
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::with_capacity(ids.len() * 4);
        for &id in ids {
            let Some(tok) = self.all_tokens.get(id as usize) else {
                continue;
            };
            if let Some(hex) = tok.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b as char);
                    continue;
                }
            }
            if self.special_tokens.iter().any(|(s, _)| s == tok) {
                continue;
            }
            out.push_str(&tok.replace('\u{2581}', " "));
        }
        if self.add_dummy_prefix && out.starts_with(' ') {
            out.remove(0);
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    start: usize, // byte offset into the (escaped) segment
    len: usize,
    prev: isize,
    next: isize,
}

#[derive(Debug)]
struct Bigram {
    left: usize,
    right: usize,
    score: f32,
    size: usize, // total byte length at push time (staleness check)
}

// BinaryHeap pops the MAXIMUM. We want: highest score first, ties broken by
// the LOWEST left index (C++ comparator: l.score < r.score || (== && l.left
// > r.left) -- a max-heap on score, min on left).
impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.score == other.score
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

fn utf8_len(first_byte: u8) -> usize {
    match first_byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        // Invalid lead byte: one byte (the C++ clamps via min() too).
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab with realistic merge chains: every multi-char piece must be
    /// reachable through adjacent-pair merges whose intermediates are also
    /// pieces (that is how SPM vocabs are trained, and why llama.cpp's
    /// greedy merge finds whole words).
    fn model() -> SpmModel {
        let toks: Vec<&str> = vec![
            "<unk>", "</s>", "[INST]", "[/INST]",
            "▁the", "▁quick", "▁Robert",
            "▁t", "▁th", "he", "▁R", "ob", "er", "ert", "▁Rob",
            "▁q", "▁qu", "▁qui", "ck",
            "▁", "▁▁", "▁▁▁",
            "<0x41>", "<0x42>",
        ];
        let scores: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0,
            -13.0, -20.0, -30.0,
            -40.0, -45.0, -50.0, -60.0, -70.0, -75.0, -80.0, -90.0,
            -100.0, -105.0, -106.0, -107.0,
            -1e9, -1e9, -1e9,
            -2000.0, -2000.0,
        ];
        let types: Vec<u32> = vec![
            2, 3, 4, 4,
            1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1,
            6, 6,
        ];
        let tokens: Vec<&str> = toks;
        SpmModel::new(&tokens, Some(&scores), Some(&types))
    }

    #[test]
    fn encodes_words_with_dummy_prefix() {
        let m = model();
        // "▁the" forms via ▁t -> ▁th -> ▁the; "▁quick" via ▁q -> ▁qu ->
        // ▁qui + ck -> ▁quick.
        assert_eq!(m.tokenize_text("the quick"), vec![4, 5]);
    }

    #[test]
    fn greedy_merge_forms_whole_piece_via_chain() {
        // Score sums favor "▁R"+"ob"+"ert" (-210) over "▁Robert" (-30) --
        // Viterbi/Unigram would fragment; the greedy merge follows the
        // ▁R -> ▁Rob -> ▁Robert chain like llama.cpp.
        let m = model();
        assert_eq!(m.tokenize_text("Robert"), vec![6]);
    }

    #[test]
    fn consecutive_spaces_merge_to_multi_marker() {
        let m = model();
        // "  " -> dummy prefix + 2 escapes = "▁▁▁" -> single token (the
        // -1e9 multi-space pieces still merge when nothing better is
        // available, leftmost first -- llama.cpp behavior).
        assert_eq!(m.tokenize_text("  "), vec![21]);
    }

    #[test]
    fn special_tokens_partition() {
        let m = model();
        // "the " -> [▁the, ▁]; " quick" -> [▁, ▁quick]
        assert_eq!(m.tokenize_text("the [INST] quick"), vec![4, 19, 2, 19, 5]);
    }

    #[test]
    fn byte_fallback_for_unknown() {
        let m = model();
        // A/B only exist as byte tokens; "▁" is a piece.
        assert_eq!(m.tokenize_text("AB"), vec![19, 22, 23]);
    }

    #[test]
    fn decode_roundtrip() {
        let m = model();
        let ids = m.tokenize_text("the quick");
        assert_eq!(m.decode(&ids), "the quick");
    }
}

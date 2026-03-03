//! DeepSeek-R1 reasoning parser.
//!
//! Handles multiple reasoning tag formats:
//! - `<think>...</think>` blocks with DeepSeek-specific edge cases
//! - `<|begin_of_thought|>...<|end_of_thought|>` format
//!
//! DeepSeek-specific edge cases:
//! - The model may start generating with `<think>` implicitly (no explicit tag)
//! - The `</think>` tag may appear without a corresponding `<think>` at the start
//! - Multiple think blocks may appear throughout the response
//! - Empty think blocks should be treated as no thinking

use regex::Regex;
use std::sync::LazyLock;

use crate::reasoning_parser::ReasoningParser;
use crate::types::ReasoningParseResult;

static THINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").unwrap());

static THOUGHT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>").unwrap());

const THOUGHT_OPEN: &str = "<|begin_of_thought|>";
const THOUGHT_CLOSE: &str = "<|end_of_thought|>";

/// Parser for DeepSeek-R1 style reasoning.
pub struct DeepSeekR1Parser {
    in_think: bool,
    think_buffer: String,
    content_buffer: String,
    raw_buffer: String,
    /// Whether we've seen any content yet (to detect implicit think start).
    started: bool,
    /// Which tag format we detected during streaming.
    tag_format: TagFormat,
}

#[derive(Clone, Copy, PartialEq)]
enum TagFormat {
    Unknown,
    Think,   // <think>...</think>
    Thought, // <|begin_of_thought|>...<|end_of_thought|>
}

impl DeepSeekR1Parser {
    pub fn new() -> Self {
        Self {
            in_think: false,
            think_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
            started: false,
            tag_format: TagFormat::Unknown,
        }
    }

    /// Extract thinking parts from `<|begin_of_thought|>...<|end_of_thought|>` tags.
    fn extract_thought_parts(text: &str) -> Vec<String> {
        let mut parts = Vec::new();
        for cap in THOUGHT_RE.captures_iter(text) {
            let content = cap.get(1).unwrap().as_str().trim();
            if !content.is_empty() {
                parts.push(content.to_string());
            }
        }
        parts
    }

    /// Strip `<|begin_of_thought|>...<|end_of_thought|>` from text.
    fn strip_thought_tags(text: &str) -> String {
        THOUGHT_RE.replace_all(text, "").to_string()
    }
}

impl Default for DeepSeekR1Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for DeepSeekR1Parser {
    fn parse(&self, text: &str) -> ReasoningParseResult {
        let text_trimmed = text.trim_start();

        // Check for <|begin_of_thought|>...<|end_of_thought|> format first
        if text.contains(THOUGHT_OPEN) {
            let thought_parts = Self::extract_thought_parts(text);
            let content = Self::strip_thought_tags(text);
            let content = content.trim().to_string();

            let thinking = if thought_parts.is_empty() {
                None
            } else {
                Some(thought_parts.join("\n\n"))
            };

            return ReasoningParseResult { thinking, content };
        }

        // DeepSeek-R1 edge case: response starts with </think> (implicit think start)
        if let Some(after_close) = text_trimmed.strip_prefix("</think>") {
            let content = THINK_RE.replace_all(after_close, "").to_string();
            let content = content.trim().to_string();

            // Parse any additional think blocks in the remaining text
            let mut thinking_parts = Vec::new();
            for cap in THINK_RE.captures_iter(after_close) {
                let think_content = cap.get(1).unwrap().as_str().trim();
                if !think_content.is_empty() {
                    thinking_parts.push(think_content.to_string());
                }
            }

            return ReasoningParseResult {
                thinking: if thinking_parts.is_empty() {
                    None
                } else {
                    Some(thinking_parts.join("\n\n"))
                },
                content,
            };
        }

        // Standard parsing: extract <think>...</think> blocks
        let mut thinking_parts = Vec::new();

        for cap in THINK_RE.captures_iter(text) {
            let think_content = cap.get(1).unwrap().as_str().trim();
            if !think_content.is_empty() {
                thinking_parts.push(think_content.to_string());
            }
        }

        let content = THINK_RE.replace_all(text, "").to_string();
        let content = content.trim().to_string();

        // DeepSeek edge case: unclosed <think> tag at the end
        if let Some(last_open) = content.rfind("<think>") {
            if !content[last_open..].contains("</think>") {
                let trailing_think = &content[last_open + "<think>".len()..];
                if !trailing_think.trim().is_empty() {
                    thinking_parts.push(trailing_think.trim().to_string());
                }
                let content = content[..last_open].trim().to_string();
                return ReasoningParseResult {
                    thinking: if thinking_parts.is_empty() {
                        None
                    } else {
                        Some(thinking_parts.join("\n\n"))
                    },
                    content,
                };
            }
        }

        let thinking = if thinking_parts.is_empty() {
            None
        } else {
            Some(thinking_parts.join("\n\n"))
        };

        ReasoningParseResult { thinking, content }
    }

    fn parse_streaming(&mut self, delta: &str) -> Option<ReasoningParseResult> {
        self.raw_buffer.push_str(delta);

        // Detect tag format on first content
        if !self.started {
            self.started = true;

            if self.raw_buffer.starts_with(THOUGHT_OPEN) {
                self.tag_format = TagFormat::Thought;
                self.in_think = true;
                self.raw_buffer = self.raw_buffer[THOUGHT_OPEN.len()..].to_string();
            } else if self.raw_buffer.starts_with("<think>") {
                self.tag_format = TagFormat::Think;
                self.in_think = true;
                self.raw_buffer = self.raw_buffer["<think>".len()..].to_string();
            }
        }

        // If format not yet known, detect from markers in buffer
        if self.tag_format == TagFormat::Unknown {
            if self.raw_buffer.contains(THOUGHT_OPEN) {
                self.tag_format = TagFormat::Thought;
            } else if self.raw_buffer.contains("<think>") {
                self.tag_format = TagFormat::Think;
            }
        }

        let (open_tag, close_tag): (&str, &str) = match self.tag_format {
            TagFormat::Thought => (THOUGHT_OPEN, THOUGHT_CLOSE),
            TagFormat::Think | TagFormat::Unknown => ("<think>", "</think>"),
        };

        loop {
            if self.in_think {
                if let Some(end_pos) = self.raw_buffer.find(close_tag) {
                    let think_part = &self.raw_buffer[..end_pos];
                    self.think_buffer.push_str(think_part);
                    self.in_think = false;
                    self.raw_buffer = self.raw_buffer[end_pos + close_tag.len()..].to_string();
                } else {
                    // Check for partial tag
                    let mut partial = false;
                    for i in 1..close_tag.len() {
                        if self.raw_buffer.ends_with(&close_tag[..i]) {
                            partial = true;
                            break;
                        }
                    }
                    if partial {
                        return None;
                    }
                    self.think_buffer.push_str(&self.raw_buffer);
                    self.raw_buffer.clear();
                    return None;
                }
            } else if let Some(start_pos) = self.raw_buffer.find(open_tag) {
                let content_part = &self.raw_buffer[..start_pos];
                self.content_buffer.push_str(content_part);
                self.in_think = true;
                self.raw_buffer = self.raw_buffer[start_pos + open_tag.len()..].to_string();
            } else {
                // Check for partial opening tag
                let mut partial_len = 0;
                for i in 1..open_tag.len() {
                    if self.raw_buffer.ends_with(&open_tag[..i]) {
                        partial_len = i;
                        break;
                    }
                }
                if partial_len > 0 {
                    let safe_end = self.raw_buffer.len() - partial_len;
                    self.content_buffer.push_str(&self.raw_buffer[..safe_end]);
                    self.raw_buffer = self.raw_buffer[safe_end..].to_string();
                    break;
                }
                self.content_buffer.push_str(&self.raw_buffer);
                self.raw_buffer.clear();
                break;
            }
        }

        let thinking = if self.think_buffer.is_empty() {
            None
        } else {
            Some(self.think_buffer.clone())
        };

        Some(ReasoningParseResult {
            thinking,
            content: self.content_buffer.clone(),
        })
    }

    fn reset(&mut self) {
        self.in_think = false;
        self.think_buffer.clear();
        self.content_buffer.clear();
        self.raw_buffer.clear();
        self.started = false;
        self.tag_format = TagFormat::Unknown;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_think() {
        let parser = DeepSeekR1Parser::new();
        let text = "<think>Let me reason about this.</think>The answer is 42.";
        let result = parser.parse(text);
        assert_eq!(result.thinking.unwrap(), "Let me reason about this.");
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_implicit_close_at_start() {
        let parser = DeepSeekR1Parser::new();
        let text = "</think>The answer is 42.";
        let result = parser.parse(text);
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_empty_think() {
        let parser = DeepSeekR1Parser::new();
        let text = "<think></think>The answer is 42.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_no_think() {
        let parser = DeepSeekR1Parser::new();
        let text = "Just a normal response.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Just a normal response.");
    }

    #[test]
    fn test_begin_end_of_thought_format() {
        let parser = DeepSeekR1Parser::new();
        let text = "<|begin_of_thought|>Let me reason step by step.\n1. First\n2. Second<|end_of_thought|>The answer is 42.";
        let result = parser.parse(text);
        assert!(result.thinking.is_some());
        let thinking = result.thinking.unwrap();
        assert!(thinking.contains("step by step"));
        assert!(thinking.contains("First"));
        assert_eq!(result.content, "The answer is 42.");
    }

    #[test]
    fn test_begin_end_of_thought_empty() {
        let parser = DeepSeekR1Parser::new();
        let text = "<|begin_of_thought|><|end_of_thought|>Quick answer.";
        let result = parser.parse(text);
        assert!(result.thinking.is_none());
        assert_eq!(result.content, "Quick answer.");
    }

    #[test]
    fn test_streaming_thought_format() {
        let mut parser = DeepSeekR1Parser::new();

        let r1 = parser.parse_streaming("<|begin_of_thought|>");
        let _r2 = parser.parse_streaming("reasoning here");
        let _r3 = parser.parse_streaming("<|end_of_thought|>");
        let r4 = parser.parse_streaming("visible content");

        let final_result = r4.or(r1).unwrap();
        assert!(final_result.thinking.is_some());
        assert!(final_result.content.contains("visible"));
    }
}

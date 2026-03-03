//! DeepSeek-R1 reasoning parser.
//!
//! Handles `<think>...</think>` blocks with DeepSeek-specific edge cases:
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

/// Parser for DeepSeek-R1 style reasoning.
pub struct DeepSeekR1Parser {
    in_think: bool,
    think_buffer: String,
    content_buffer: String,
    raw_buffer: String,
    /// Whether we've seen any content yet (to detect implicit think start).
    started: bool,
}

impl DeepSeekR1Parser {
    pub fn new() -> Self {
        Self {
            in_think: false,
            think_buffer: String::new(),
            content_buffer: String::new(),
            raw_buffer: String::new(),
            started: false,
        }
    }
}

impl Default for DeepSeekR1Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for DeepSeekR1Parser {
    fn parse(&self, text: &str) -> ReasoningParseResult {
        // DeepSeek-R1 edge case: response starts with content before </think>
        // without an opening <think> tag. Treat the initial content as thinking.
        let text_trimmed = text.trim_start();

        // Check if the text starts with </think> (implicit think start)
        if let Some(after_close) = text_trimmed.strip_prefix("</think>") {
            // Everything before the implicit close was thinking (but there's nothing
            // before it in this case)
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
        // If there's an unclosed <think>, treat everything after it as thinking
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

        // DeepSeek-R1 edge case: if first content and no <think> tag,
        // check if starts with thinking implicitly
        if !self.started {
            self.started = true;
            // If the first token is <think>, enter think mode
            if self.raw_buffer.starts_with("<think>") {
                self.in_think = true;
                self.raw_buffer = self.raw_buffer["<think>".len()..].to_string();
            }
        }

        loop {
            if self.in_think {
                if let Some(end_pos) = self.raw_buffer.find("</think>") {
                    let think_part = &self.raw_buffer[..end_pos];
                    self.think_buffer.push_str(think_part);
                    self.in_think = false;
                    self.raw_buffer = self.raw_buffer[end_pos + "</think>".len()..].to_string();
                } else {
                    // Check for partial tag
                    let potential_tag = "</think>";
                    let mut partial = false;
                    for i in 1..potential_tag.len() {
                        if self.raw_buffer.ends_with(&potential_tag[..i]) {
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
            } else if let Some(start_pos) = self.raw_buffer.find("<think>") {
                let content_part = &self.raw_buffer[..start_pos];
                self.content_buffer.push_str(content_part);
                self.in_think = true;
                self.raw_buffer = self.raw_buffer[start_pos + "<think>".len()..].to_string();
            } else {
                // Check for partial opening tag
                let potential_tag = "<think>";
                let mut partial_len = 0;
                for i in 1..potential_tag.len() {
                    if self.raw_buffer.ends_with(&potential_tag[..i]) {
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
}

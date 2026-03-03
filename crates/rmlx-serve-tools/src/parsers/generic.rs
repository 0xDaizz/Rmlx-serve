//! Generic / simpler tool call parser implementations.
//!
//! These parsers handle less common tool-calling formats. They share common
//! patterns and delegate to shared utilities where possible.

use regex::Regex;
use std::sync::LazyLock;
use tracing::debug;

use crate::parsers::utils::{
    extract_json_objects, generate_tool_call_id, parse_json_tool_call, strip_think_tags,
};
use crate::tool_parser::ToolCallParser;
use crate::types::{
    DeltaToolCall, ParsedToolCall, StreamingParseResult, ToolCallParseResult,
};

// ---------------------------------------------------------------------------
// Shared streaming helper
// ---------------------------------------------------------------------------

/// Minimal streaming state shared by the generic parsers.
struct GenericStreamState {
    buffer: String,
    current_tool_count: usize,
    in_tool_call: bool,
}

impl GenericStreamState {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            current_tool_count: 0,
            in_tool_call: false,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.current_tool_count = 0;
        self.in_tool_call = false;
    }

    /// Generic streaming parse: detect marker, accumulate, emit completed calls.
    fn parse_streaming(
        &mut self,
        delta: &str,
        curr: &str,
        marker: &str,
        parse_fn: &dyn Fn(&str) -> Vec<ParsedToolCall>,
    ) -> StreamingParseResult {
        self.buffer.push_str(delta);
        let text = strip_think_tags(curr);

        if text.contains(marker) {
            self.in_tool_call = true;
        }

        if !self.in_tool_call {
            return StreamingParseResult {
                content: Some(delta.to_string()),
                tool_calls: vec![],
                finished: false,
            };
        }

        let all_calls = parse_fn(&text);
        let mut new_deltas = Vec::new();

        for (i, tc) in all_calls.iter().enumerate() {
            if i >= self.current_tool_count {
                new_deltas.push(DeltaToolCall {
                    index: i,
                    id: Some(tc.id.clone()),
                    name: Some(tc.name.clone()),
                    arguments: Some(tc.arguments.clone()),
                });
                self.current_tool_count = i + 1;
            }
        }

        let finished = !all_calls.is_empty();

        StreamingParseResult {
            content: None,
            tool_calls: new_deltas,
            finished,
        }
    }
}

// ===========================================================================
// GLM47ToolParser
// ===========================================================================

static GLM_OBSERVATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)(.*?)<\|observation\|>").unwrap()
});

/// Parser for GLM-4-7B style tool calls.
///
/// Format: tool call JSON followed by `<|observation|>` delimiter.
pub struct GLM47ToolParser {
    state: GenericStreamState,
}

impl GLM47ToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        // Content before <|observation|> may contain tool call JSON
        if let Some(cap) = GLM_OBSERVATION_RE.captures(&cleaned) {
            let before = cap.get(1).unwrap().as_str();
            let json_objs = extract_json_objects(before);
            let mut calls = Vec::new();
            for obj_str in json_objs {
                if let Some(tc) = parse_json_tool_call(&obj_str) {
                    calls.push(tc);
                }
            }
            calls
        } else {
            vec![]
        }
    }
}

impl Default for GLM47ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for GLM47ToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed GLM-4-7B tool calls");
        }

        let content = if tool_calls.is_empty() {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            None
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "<|observation|>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// GraniteToolParser
// ===========================================================================

static GRANITE_TOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap()
});

/// Parser for Granite-style tool calls.
///
/// Uses `<tool_call>` blocks with function-style JSON content.
pub struct GraniteToolParser {
    state: GenericStreamState,
}

impl GraniteToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let mut calls = Vec::new();
        for cap in GRANITE_TOOL_RE.captures_iter(&cleaned) {
            let json_str = cap.get(1).unwrap().as_str();
            // Granite may use {"name": ..., "arguments": ...} or
            // {"function": {"name": ..., "arguments": ...}}
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(obj) = v.as_object() {
                    // Check for nested "function" key
                    if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
                        let name = func
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let args = func
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let arguments = if args.is_string() {
                            args.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(&args).unwrap_or_default()
                        };
                        if !name.is_empty() {
                            calls.push(ParsedToolCall {
                                id: generate_tool_call_id(),
                                name,
                                arguments,
                            });
                        }
                    } else if let Some(tc) = parse_json_tool_call(json_str) {
                        calls.push(tc);
                    }
                }
            }
        }
        calls
    }
}

impl Default for GraniteToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for GraniteToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Granite tool calls");
        }

        // Extract content outside <tool_call> blocks
        let content_text = GRANITE_TOOL_RE.replace_all(&cleaned, "").to_string();
        let content = {
            let trimmed = content_text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "<tool_call>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// XLamToolParser
// ===========================================================================

/// Parser for xLAM-style tool calls.
///
/// Format: `{"tool_calls": [{"name": ..., "arguments": ...}, ...]}`.
pub struct XLamToolParser {
    state: GenericStreamState,
}

impl XLamToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let json_objs = extract_json_objects(&cleaned);
        let mut calls = Vec::new();

        for obj_str in json_objs {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&obj_str) {
                if let Some(obj) = v.as_object() {
                    if let Some(arr) = obj.get("tool_calls").and_then(|v| v.as_array()) {
                        for item in arr {
                            let item_str = serde_json::to_string(item).unwrap_or_default();
                            if let Some(tc) = parse_json_tool_call(&item_str) {
                                calls.push(tc);
                            }
                        }
                    }
                }
            }
        }

        calls
    }
}

impl Default for XLamToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for XLamToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed xLAM tool calls");
        }

        let content = if tool_calls.is_empty() {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            None
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "\"tool_calls\"", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// KimiToolParser
// ===========================================================================

static KIMI_TOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap()
});

/// Parser for Kimi-style tool calls.
///
/// Uses `<tool_call>` blocks with Kimi-specific JSON format.
pub struct KimiToolParser {
    state: GenericStreamState,
}

impl KimiToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let mut calls = Vec::new();

        for cap in KIMI_TOOL_RE.captures_iter(&cleaned) {
            let json_str = cap.get(1).unwrap().as_str();
            if let Some(tc) = parse_json_tool_call(json_str) {
                calls.push(tc);
            }
        }

        calls
    }
}

impl Default for KimiToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for KimiToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Kimi tool calls");
        }

        let content_text = KIMI_TOOL_RE.replace_all(&cleaned, "").to_string();
        let content = {
            let trimmed = content_text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "<tool_call>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// NemotronToolParser
// ===========================================================================

static NEMOTRON_TOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<toolcall>\s*(.*?)\s*</toolcall>").unwrap()
});

/// Parser for Nemotron-style tool calls.
///
/// Uses `<toolcall>` blocks (note: no underscore).
pub struct NemotronToolParser {
    state: GenericStreamState,
}

impl NemotronToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let mut calls = Vec::new();

        for cap in NEMOTRON_TOOL_RE.captures_iter(&cleaned) {
            let json_str = cap.get(1).unwrap().as_str();
            if let Some(tc) = parse_json_tool_call(json_str) {
                calls.push(tc);
            }
        }

        calls
    }
}

impl Default for NemotronToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for NemotronToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Nemotron tool calls");
        }

        let content_text = NEMOTRON_TOOL_RE.replace_all(&cleaned, "").to_string();
        let content = {
            let trimmed = content_text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "<toolcall>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// FunctionaryToolParser
// ===========================================================================

/// Parser for Functionary v3-style tool calls.
///
/// Format:
/// ```text
/// >>> function_name
/// {"arg1": "val1"}
/// >>> another_function
/// {"arg2": "val2"}
/// ```
pub struct FunctionaryToolParser {
    state: GenericStreamState,
}

impl FunctionaryToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let mut calls = Vec::new();

        // Split on ">>>" and process each section
        let sections: Vec<&str> = cleaned.split(">>>").collect();
        // First section (before any >>>) is content, skip it
        for section in sections.iter().skip(1) {
            let trimmed = section.trim();
            if trimmed.is_empty() {
                continue;
            }

            // First line is the function name, rest is arguments
            let (name, body) = if let Some(newline_pos) = trimmed.find('\n') {
                let n = trimmed[..newline_pos].trim().to_string();
                let b = trimmed[newline_pos + 1..].trim();
                (n, b)
            } else {
                (trimmed.to_string(), "")
            };

            // Skip "all" or content-only markers
            if name == "all" || name == "content" || name.is_empty() {
                continue;
            }

            let arguments = if body.is_empty() {
                "{}".to_string()
            } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
                serde_json::to_string(&v).unwrap_or_else(|_| body.to_string())
            } else {
                body.to_string()
            };

            calls.push(ParsedToolCall {
                id: generate_tool_call_id(),
                name,
                arguments,
            });
        }

        calls
    }
}

impl Default for FunctionaryToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for FunctionaryToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Functionary tool calls");
        }

        // Content is text before the first >>>
        let content = if let Some(idx) = cleaned.find(">>>") {
            let before = cleaned[..idx].trim();
            if before.is_empty() {
                None
            } else {
                Some(before.to_string())
            }
        } else {
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, ">>>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// HarmonyToolParser
// ===========================================================================

static HARMONY_TOOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap()
});

/// Generic harmony-style tool call parser.
///
/// Uses `<tool_call>` blocks similar to Hermes but with simpler handling.
pub struct HarmonyToolParser {
    state: GenericStreamState,
}

impl HarmonyToolParser {
    pub fn new() -> Self {
        Self {
            state: GenericStreamState::new(),
        }
    }

    fn extract_calls(text: &str) -> Vec<ParsedToolCall> {
        let cleaned = strip_think_tags(text);
        let mut calls = Vec::new();

        for cap in HARMONY_TOOL_RE.captures_iter(&cleaned) {
            let json_str = cap.get(1).unwrap().as_str();
            if let Some(tc) = parse_json_tool_call(json_str) {
                calls.push(tc);
            }
        }

        calls
    }
}

impl Default for HarmonyToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser for HarmonyToolParser {
    fn parse(&self, text: &str) -> ToolCallParseResult {
        let cleaned = strip_think_tags(text);
        let tool_calls = Self::extract_calls(&cleaned);

        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "parsed Harmony tool calls");
        }

        let content_text = HARMONY_TOOL_RE.replace_all(&cleaned, "").to_string();
        let content = {
            let trimmed = content_text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };

        ToolCallParseResult {
            tool_calls,
            content,
        }
    }

    fn parse_streaming(&mut self, _prev: &str, curr: &str, delta: &str) -> StreamingParseResult {
        self.state
            .parse_streaming(delta, curr, "<tool_call>", &Self::extract_calls)
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm47_observation() {
        let parser = GLM47ToolParser::new();
        let text =
            r#"{"name": "get_weather", "arguments": {"city": "London"}}<|observation|>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_granite_function() {
        let parser = GraniteToolParser::new();
        let text = r#"<tool_call>{"function": {"name": "search", "arguments": {"q": "rust"}}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_granite_direct() {
        let parser = GraniteToolParser::new();
        let text = r#"<tool_call>{"name": "search", "arguments": {"q": "rust"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_xlam() {
        let parser = XLamToolParser::new();
        let text = r#"{"tool_calls": [{"name": "get_weather", "arguments": {"city": "London"}}]}"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn test_kimi() {
        let parser = KimiToolParser::new();
        let text = r#"<tool_call>{"name": "search", "arguments": {"q": "hello"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_nemotron() {
        let parser = NemotronToolParser::new();
        let text = r#"<toolcall>{"name": "search", "arguments": {"q": "hello"}}</toolcall>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
    }

    #[test]
    fn test_functionary() {
        let parser = FunctionaryToolParser::new();
        let text = ">>> get_weather\n{\"city\": \"London\"}\n>>> get_time\n{\"tz\": \"UTC\"}\n";
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[1].name, "get_time");
    }

    #[test]
    fn test_harmony() {
        let parser = HarmonyToolParser::new();
        let text = r#"<tool_call>{"name": "search", "arguments": {"q": "hello"}}</tool_call>"#;
        let result = parser.parse(text);
        assert_eq!(result.tool_calls.len(), 1);
    }

    #[test]
    fn test_no_tool_calls_glm() {
        let parser = GLM47ToolParser::new();
        let text = "Just a regular response.";
        let result = parser.parse(text);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }
}

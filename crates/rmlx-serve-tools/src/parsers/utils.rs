//! Shared utility functions for tool-call parsers.

use regex::Regex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use crate::types::ParsedToolCall;

/// Monotonically increasing counter for generating unique tool call ids.
static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a tool-call id in the format `call_<unique_hex>`.
///
/// Uses a combination of a monotonic counter and the current time to produce
/// ids that are unique within a process lifetime, without requiring the `uuid`
/// crate.
pub fn generate_tool_call_id() -> String {
    let count = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    // Mix counter and time for uniqueness
    let mixed = count
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(nanos);
    format!("call_{:016x}{:08x}", nanos, mixed as u32)
}

/// Strip `<think>...</think>` blocks from text, returning the cleaned text.
///
/// Also handles `<|begin_of_thought|>...<|end_of_thought|>` blocks used by
/// DeepSeek R1 and similar models.
pub fn strip_think_tags(text: &str) -> String {
    static THINK_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    static THOUGHT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<\|begin_of_thought\|>.*?<\|end_of_thought\|>").unwrap());
    let result = THINK_RE.replace_all(text, "");
    THOUGHT_RE.replace_all(&result, "").to_string()
}

/// Try to parse a JSON object as a tool call.
///
/// Accepts objects with `"name"` and either `"arguments"` or `"parameters"`.
/// Returns `None` if the JSON is not a valid tool call shape.
pub fn parse_json_tool_call(json_str: &str) -> Option<ParsedToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = v.as_object()?;

    let name = obj.get("name")?.as_str()?.to_string();

    // Accept "arguments" or "parameters"
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    let arguments = if args.is_string() {
        args.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(&args).unwrap_or_default()
    };

    Some(ParsedToolCall {
        id: generate_tool_call_id(),
        name,
        arguments,
    })
}

/// Try to extract all JSON objects from text.
///
/// Uses brace-counting to find top-level `{...}` blocks.
pub fn extract_json_objects(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut depth = 0i32;
    let mut start = None;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        results.push(text[s..=i].to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }

    results
}

/// Try to extract a JSON array from text.
///
/// Uses bracket-counting to find the first top-level `[...]` block.
pub fn extract_json_array(text: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut start = None;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '[' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return Some(text[s..=i].to_string());
                    }
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

    #[test]
    fn test_strip_think_tags() {
        let input = "<think>reasoning here</think>Hello world";
        assert_eq!(strip_think_tags(input), "Hello world");
    }

    #[test]
    fn test_strip_think_tags_multiline() {
        let input = "<think>\nstep 1\nstep 2\n</think>\nAnswer: 42";
        assert_eq!(strip_think_tags(input), "\nAnswer: 42");
    }

    #[test]
    fn test_strip_think_tags_thought_format() {
        let input = "<|begin_of_thought|>deep reasoning here<|end_of_thought|>Answer: 42";
        assert_eq!(strip_think_tags(input), "Answer: 42");
    }

    #[test]
    fn test_strip_think_tags_mixed_formats() {
        let input = "<think>think1</think>middle<|begin_of_thought|>think2<|end_of_thought|>end";
        assert_eq!(strip_think_tags(input), "middleend");
    }

    #[test]
    fn test_extract_json_objects() {
        let text = r#"Some text {"name": "foo", "arguments": {}} more text {"name": "bar"}"#;
        let objs = extract_json_objects(text);
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn test_parse_json_tool_call() {
        let json = r#"{"name": "get_weather", "arguments": {"city": "London"}}"#;
        let tc = parse_json_tool_call(json).unwrap();
        assert_eq!(tc.name, "get_weather");
        assert!(tc.arguments.contains("London"));
    }

    #[test]
    fn test_parse_json_tool_call_with_parameters() {
        let json = r#"{"name": "get_weather", "parameters": {"city": "London"}}"#;
        let tc = parse_json_tool_call(json).unwrap();
        assert_eq!(tc.name, "get_weather");
        assert!(tc.arguments.contains("London"));
    }

    #[test]
    fn test_extract_json_array() {
        let text = r#"[{"name": "a"}, {"name": "b"}]"#;
        let arr = extract_json_array(text).unwrap();
        assert!(arr.starts_with('['));
        assert!(arr.ends_with(']'));
    }
}

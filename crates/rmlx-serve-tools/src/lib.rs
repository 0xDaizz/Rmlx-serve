//! rmlx-serve-tools: Tool/function calling and reasoning parsers.
//!
//! This crate provides parsers for extracting structured tool calls and
//! reasoning blocks from model-generated text. It supports multiple model
//! families including Hermes, Llama, Mistral, Qwen, DeepSeek, and more.
//!
//! # Architecture
//!
//! - **Tool call parsers** implement [`ToolCallParser`] and detect model-specific
//!   patterns for function/tool invocations (e.g. `<tool_call>`, `[TOOL_CALLS]`,
//!   `<|python_tag|>`).
//!
//! - **Reasoning parsers** implement [`ReasoningParser`] and separate
//!   thinking/reasoning blocks (e.g. `<think>`, `<reasoning>`) from visible content.
//!
//! - Both parser types support full-text parsing and incremental (streaming) parsing.
//!
//! - The [`ToolParserRegistry`] and [`ReasoningParserRegistry`] provide
//!   factory-based lookup by parser name.
//!
//! # Example
//!
//! ```rust
//! use rmlx_serve_tools::{ToolParserRegistry, ToolCallParser};
//!
//! let registry = ToolParserRegistry::new();
//! let parser = registry.get("hermes").unwrap();
//!
//! let text = r#"<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
//! let result = parser.parse(text);
//! assert_eq!(result.tool_calls.len(), 1);
//! assert_eq!(result.tool_calls[0].name, "get_weather");
//! ```
//!
//! This crate has NO rmlx dependencies -- only rmlx-serve-types, serde,
//! serde_json, regex, thiserror, and tracing.

pub mod parsers;
pub mod reasoning;
pub mod reasoning_parser;
pub mod tool_parser;
pub mod types;

// Re-export the primary public API at the crate root.
pub use reasoning_parser::{ReasoningParser, ReasoningParserRegistry};
pub use tool_parser::{ToolCallParser, ToolParserRegistry};
pub use types::{
    DeltaToolCall, ParsedToolCall, ReasoningParseResult, StreamingParseResult,
    ToolCallParseResult,
};

//! Tool-call parser implementations.
//!
//! Each sub-module implements [`ToolCallParser`](crate::tool_parser::ToolCallParser)
//! for a specific model family or tool-calling convention.

pub mod auto;
pub mod deepseek;
pub mod generic;
pub mod hermes;
pub mod llama;
pub mod mistral;
pub mod qwen;
pub(crate) mod utils;

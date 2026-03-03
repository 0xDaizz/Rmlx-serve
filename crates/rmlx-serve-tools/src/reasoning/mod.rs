//! Reasoning/thinking block parser implementations.
//!
//! Each sub-module implements [`ReasoningParser`](crate::reasoning_parser::ReasoningParser)
//! for a specific reasoning format.

pub mod deepseek_r1;
pub mod gpt_oss;
pub mod harmony;
pub mod qwen3;
pub mod think;

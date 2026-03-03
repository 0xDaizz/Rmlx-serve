//! HTTP request handlers.

pub mod anthropic;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod mcp;
pub mod models;

pub use anthropic::anthropic_messages;
pub use chat::chat_completions;
pub use completions::completions;
pub use embeddings::embeddings;
pub use health::{health_check, metrics};
pub use mcp::{mcp_execute_tool, mcp_list_servers, mcp_list_tools};
pub use models::list_models;

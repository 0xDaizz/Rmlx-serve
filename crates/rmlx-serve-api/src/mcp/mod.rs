//! MCP (Model Context Protocol) runtime integration.
//!
//! This module provides the client-side implementation for connecting to MCP
//! tool servers, discovering their tools, and executing tool calls. It
//! supports the stdio transport and manages multiple concurrent server
//! connections.

pub mod client;
pub mod config;
pub mod manager;
pub mod security;

//! MCP (Model Context Protocol) endpoints (placeholder).

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use rmlx_serve_types::mcp::{MCPServerStatus, MCPTool};

use crate::state::AppState;

/// Handle `GET /v1/mcp/tools`.
///
/// Returns the list of MCP tools available across all connected servers.
/// Currently returns an empty list as MCP integration is not yet implemented.
pub async fn mcp_list_tools(State(_state): State<Arc<AppState>>) -> Json<Vec<MCPTool>> {
    Json(vec![])
}

/// Handle `GET /v1/mcp/servers`.
///
/// Returns the list of connected MCP servers and their status.
/// Currently returns an empty list as MCP integration is not yet implemented.
pub async fn mcp_list_servers(State(_state): State<Arc<AppState>>) -> Json<Vec<MCPServerStatus>> {
    Json(vec![])
}

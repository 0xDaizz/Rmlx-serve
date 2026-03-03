//! MCP (Model Context Protocol) endpoints.
//!
//! Provides HTTP handlers for listing MCP tools and executing tool calls
//! against connected MCP servers.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use rmlx_serve_types::mcp::{MCPServerStatus, MCPTool};

use crate::error::ApiError;
use crate::state::AppState;

/// Request body for `POST /v1/mcp/execute`.
#[derive(Debug, Deserialize)]
pub struct McpExecuteRequest {
    /// The name of the MCP server to route the request to.
    pub server: String,

    /// The name of the tool to execute.
    pub tool: String,

    /// Arguments to pass to the tool.
    #[serde(default = "default_args")]
    pub arguments: serde_json::Value,
}

fn default_args() -> serde_json::Value {
    serde_json::json!({})
}

/// Response body for `POST /v1/mcp/execute`.
#[derive(Debug, Serialize)]
pub struct McpExecuteResponse {
    /// The tool execution result.
    pub result: serde_json::Value,
}

/// Handle `GET /v1/mcp/tools`.
///
/// Returns the list of MCP tools available across all connected servers.
pub async fn mcp_list_tools(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<MCPTool>>, ApiError> {
    if let Some(ref manager) = state.mcp_manager {
        let tools = manager.all_tools().await;
        Ok(Json(tools))
    } else {
        Ok(Json(vec![]))
    }
}

/// Handle `GET /v1/mcp/servers`.
///
/// Returns the list of connected MCP servers and their status.
pub async fn mcp_list_servers(State(_state): State<Arc<AppState>>) -> Json<Vec<MCPServerStatus>> {
    Json(vec![])
}

/// Handle `POST /v1/mcp/execute`.
///
/// Execute a tool on a specific MCP server.
pub async fn mcp_execute_tool(
    State(state): State<Arc<AppState>>,
    Json(request): Json<McpExecuteRequest>,
) -> Result<Json<McpExecuteResponse>, ApiError> {
    let manager = state
        .mcp_manager
        .as_ref()
        .ok_or_else(|| ApiError::InvalidRequest("MCP is not configured".to_string()))?;

    let result = manager
        .execute(&request.server, &request.tool, request.arguments)
        .await
        .map_err(|e| ApiError::InternalError(format!("MCP tool execution failed: {}", e)))?;

    Ok(Json(McpExecuteResponse { result }))
}

//! MCP configuration loading.
//!
//! Loads MCP server definitions from a JSON configuration file. The loader
//! searches multiple well-known locations in priority order so that users can
//! override settings per-project or globally.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for a single MCP server (stdio transport).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// The command to launch the MCP server process.
    pub command: String,

    /// Command-line arguments passed to the server process.
    #[serde(default)]
    pub args: Vec<String>,

    /// Extra environment variables injected into the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Top-level MCP configuration containing all server definitions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct McpConfig {
    /// Map of server name -> server configuration.
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

impl McpConfig {
    /// Load MCP configuration by searching multiple locations in order:
    ///
    /// 1. Explicit path from the `--mcp-config` CLI flag (passed as `cli_path`).
    /// 2. The `MCP_CONFIG` environment variable.
    /// 3. `./mcp.json` in the current working directory.
    /// 4. `~/.config/rmlx-serve/mcp.json` in the user's home directory.
    ///
    /// Returns `Ok(None)` if no configuration file is found at any location.
    /// Returns an error if a file is found but cannot be read or parsed.
    pub fn load(cli_path: Option<&str>) -> Result<Option<Self>, McpConfigError> {
        // Build the list of candidate paths in priority order.
        let candidates = Self::candidate_paths(cli_path);

        for path in &candidates {
            if path.exists() {
                tracing::info!("Loading MCP config from {}", path.display());
                let contents = std::fs::read_to_string(path).map_err(|e| {
                    McpConfigError::Io(format!("{}: {}", path.display(), e))
                })?;
                let config: McpConfig = serde_json::from_str(&contents).map_err(|e| {
                    McpConfigError::Parse(format!("{}: {}", path.display(), e))
                })?;
                return Ok(Some(config));
            }
        }

        tracing::debug!("No MCP configuration file found");
        Ok(None)
    }

    /// Return the ordered list of candidate configuration file paths.
    fn candidate_paths(cli_path: Option<&str>) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. Explicit CLI flag.
        if let Some(p) = cli_path {
            paths.push(PathBuf::from(p));
        }

        // 2. MCP_CONFIG environment variable.
        if let Ok(env_path) = std::env::var("MCP_CONFIG") {
            paths.push(PathBuf::from(env_path));
        }

        // 3. Current directory.
        paths.push(PathBuf::from("./mcp.json"));

        // 4. User config directory.
        if let Some(home) = dirs_home() {
            paths.push(home.join(".config").join("rmlx-serve").join("mcp.json"));
        }

        paths
    }
}

/// Resolve the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or({
            #[cfg(target_os = "windows")]
            {
                std::env::var("USERPROFILE").ok().map(PathBuf::from)
            }
            #[cfg(not(target_os = "windows"))]
            {
                None
            }
        })
}

/// Errors that can occur while loading MCP configuration.
#[derive(Debug, thiserror::Error)]
pub enum McpConfigError {
    /// I/O error reading the configuration file.
    #[error("failed to read MCP config: {0}")]
    Io(String),

    /// JSON parse error in the configuration file.
    #[error("failed to parse MCP config: {0}")]
    Parse(String),
}

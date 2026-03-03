//! MCP security layer.
//!
//! Provides command whitelisting, dangerous-pattern detection for shell
//! injection prevention, and audit logging for all MCP tool executions.

use std::collections::HashSet;

/// Security guard for MCP server commands and tool executions.
#[derive(Clone, Debug)]
pub struct McpSecurity {
    /// Set of allowed command names (basenames). When non-empty, only commands
    /// in this set are permitted to be launched. An empty set means all
    /// commands are allowed (no whitelist).
    command_whitelist: HashSet<String>,
}

/// Patterns that indicate potential shell injection.
const DANGEROUS_PATTERNS: &[&str] = &[";", "|", "&&", "||", "`", "$(", "$((", ">{", "<(", ">>"];

impl McpSecurity {
    /// Create a new `McpSecurity` instance with the given command whitelist.
    ///
    /// Pass an empty set to allow all commands.
    pub fn new(whitelist: HashSet<String>) -> Self {
        Self {
            command_whitelist: whitelist,
        }
    }

    /// Create a permissive security instance with no whitelist restrictions.
    pub fn permissive() -> Self {
        Self {
            command_whitelist: HashSet::new(),
        }
    }

    /// Check whether the given command is allowed by the whitelist.
    ///
    /// Returns `true` if the whitelist is empty (all commands allowed) or if
    /// the command's basename is present in the whitelist.
    pub fn is_command_allowed(&self, cmd: &str) -> bool {
        if self.command_whitelist.is_empty() {
            return true;
        }

        // Extract the basename of the command (e.g. "/usr/bin/node" -> "node").
        let basename = cmd
            .rsplit('/')
            .next()
            .unwrap_or(cmd)
            .rsplit('\\')
            .next()
            .unwrap_or(cmd);

        self.command_whitelist.contains(basename)
    }

    /// Check whether any of the arguments contain dangerous shell patterns
    /// that could indicate a shell injection attack.
    ///
    /// Returns `true` if any argument contains a dangerous pattern.
    pub fn has_dangerous_patterns(args: &[String]) -> bool {
        for arg in args {
            for pattern in DANGEROUS_PATTERNS {
                if arg.contains(pattern) {
                    tracing::warn!(
                        "Dangerous pattern {:?} detected in MCP argument: {:?}",
                        pattern,
                        arg
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Log an MCP tool execution for audit purposes.
    pub fn audit_log(server_name: &str, tool_name: &str, success: bool) {
        let status = if success { "OK" } else { "FAILED" };
        tracing::info!(
            target: "mcp_audit",
            server = server_name,
            tool = tool_name,
            status = status,
            "MCP tool execution: server={} tool={} status={}",
            server_name,
            tool_name,
            status,
        );
    }
}

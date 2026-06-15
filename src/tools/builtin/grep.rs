use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;

use crate::core::AgentError;
use crate::tools::Tool;

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using a regex pattern. Uses ripgrep if available, falls back to grep."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path to search in"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'pattern' argument".into()))?;
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'path' argument".into()))?;

        let output = Command::new("grep")
            .arg("-rn")
            .arg("--color=never")
            .arg(pattern)
            .arg(path)
            .output()
            .await
            .map_err(|e| AgentError::ToolError(format!("Failed to run grep: {}", e)))?;

        if !output.status.success() {
            return Ok("No matches found.".to_string());
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.len() > 10000 {
            return Ok(format!("{} ... (truncated to 10000 chars)", &stdout[..10000]));
        }
        Ok(stdout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_grep_find_pattern() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "hello world\nfoo bar\n").unwrap();

        let tool = GrepTool;
        let args = serde_json::json!({
            "pattern": "hello",
            "path": dir.path().join("test.txt").to_string_lossy().to_string()
        });
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("hello"));
        assert!(!result.contains("foo"));
    }

    #[tokio::test]
    async fn test_grep_no_match() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "hello world\n").unwrap();

        let tool = GrepTool;
        let args = serde_json::json!({
            "pattern": "nonexistent",
            "path": dir.path().join("test.txt").to_string_lossy().to_string()
        });
        let result = tool.call(args).await.unwrap();
        assert_eq!(result, "No matches found.");
    }

    #[tokio::test]
    async fn test_grep_missing_pattern() {
        let tool = GrepTool;
        let args = serde_json::json!({"path": "/tmp"});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_grep_missing_path() {
        let tool = GrepTool;
        let args = serde_json::json!({"pattern": "hello"});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }
}

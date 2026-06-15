use async_trait::async_trait;
use serde_json::Value;
use tokio::task::spawn_blocking;

use crate::core::AgentError;
use crate::tools::Tool;

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Search for files matching a glob pattern (e.g. '**/*.rs')"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to search for"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'pattern' argument".into()))?;
        let pattern = pattern.to_string();

        let results = spawn_blocking(move || {
            let entries: Vec<String> = glob::glob(&pattern)
                .map_err(|e| format!("Invalid glob pattern: {}", e))?
                .filter_map(|r| r.ok())
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            Ok::<Vec<String>, String>(entries)
        })
        .await
        .map_err(|e| AgentError::ToolError(e.to_string()))?
        .map_err(AgentError::ToolError)?;

        if results.is_empty() {
            Ok("No files found matching the pattern.".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_glob_find_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();

        let pattern = format!("{}/*.rs", dir.path().to_string_lossy());
        let tool = GlobTool;
        let args = serde_json::json!({"pattern": pattern});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("a.rs"));
        assert!(result.contains("b.rs"));
        assert!(!result.contains("c.txt"));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let dir = TempDir::new().unwrap();
        let pattern = format!("{}/*.xyz", dir.path().to_string_lossy());
        let tool = GlobTool;
        let args = serde_json::json!({"pattern": pattern});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result, "No files found matching the pattern.");
    }

    #[tokio::test]
    async fn test_glob_missing_pattern() {
        let tool = GlobTool;
        let args = serde_json::json!({});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_glob_recursive() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/x.rs"), "").unwrap();

        let pattern = format!("{}/**/*.rs", dir.path().to_string_lossy());
        let tool = GlobTool;
        let args = serde_json::json!({"pattern": pattern});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("x.rs"));
    }
}

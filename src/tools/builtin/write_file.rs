use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::core::AgentError;
use crate::tools::Tool;

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path. Creates parent directories if needed."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'path' argument".into()))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'content' argument".into()))?;

        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(path, content).await?;
        Ok(format!("Successfully wrote {} bytes to {}", content.len(), path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_file_success() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let path_str = path.to_string_lossy().to_string();

        let tool = WriteFileTool;
        let args = serde_json::json!({"path": path_str, "content": "hello world"});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_write_file_missing_path() {
        let tool = WriteFileTool;
        let args = serde_json::json!({"content": "data"});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_write_file_missing_content() {
        let tool = WriteFileTool;
        let args = serde_json::json!({"path": "/tmp/test.txt"});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a/b/c/d/test.txt");
        let path_str = path.to_string_lossy().to_string();

        let tool = WriteFileTool;
        let args = serde_json::json!({"path": path_str, "content": "nested"});
        tool.call(args).await.unwrap();
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }
}

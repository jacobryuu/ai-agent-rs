use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;

use crate::core::AgentError;
use crate::tools::Tool;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'path' argument".into()))?;
        let content = fs::read_to_string(path).await?;
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_read_file_success() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "test content").unwrap();
        let path = tmp.path().to_string_lossy().to_string();

        let tool = ReadFileTool;
        let args = serde_json::json!({"path": path});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result, "test content");
    }

    #[tokio::test]
    async fn test_read_file_missing_path() {
        let tool = ReadFileTool;
        let args = serde_json::json!({});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let tool = ReadFileTool;
        let args = serde_json::json!({"path": "/tmp/nonexistent_file_xyz_123"});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::IOError(_)));
    }
}

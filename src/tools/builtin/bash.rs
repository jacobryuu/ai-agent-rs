use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;

use crate::core::AgentError;
use crate::tools::Tool;

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return the output"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'command' argument".into()))?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .map_err(|e| AgentError::ToolError(format!("Failed to execute command: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&format!("STDERR:\n{}", stderr));
        }
        if !output.status.success() && result.is_empty() {
            result = format!("Command exited with code: {:?}", output.status.code());
        }

        if result.len() > 10000 {
            result = format!("{} ... (truncated to 10000 chars)", &result[..10000]);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo hello"});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[tokio::test]
    async fn test_bash_pipe() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo 'hello world' | wc -w"});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "2");
    }

    #[tokio::test]
    async fn test_bash_empty_command() {
        let tool = BashTool;
        let args = serde_json::json!({"command": ""});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "");
    }

    #[tokio::test]
    async fn test_bash_missing_command() {
        let tool = BashTool;
        let args = serde_json::json!({});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "exit 42"});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("42"));
    }

    #[tokio::test]
    async fn test_bash_stderr() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo output && echo error >&2"});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("output"));
        assert!(result.contains("error"));
    }
}

pub mod builtin;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::{AgentError, ToolCall, ToolDef, ToolResult};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn call(&self, args: Value) -> Result<String, AgentError>;
}

#[derive(Clone)]
pub struct ToolEngine {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolEngine {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get_defs(&self) -> Vec<ToolDef> {
        self.tools
            .values()
            .map(|tool| ToolDef {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            })
            .collect()
    }

    pub async fn execute(&self, tool_call: &ToolCall) -> Result<ToolResult, AgentError> {
        let tool = self
            .tools
            .get(&tool_call.function.name)
            .ok_or_else(|| AgentError::ToolNotFound(tool_call.function.name.clone()))?;

        let args_value = match &tool_call.function.arguments {
            Value::String(s) => {
                serde_json::from_str(s).unwrap_or(tool_call.function.arguments.clone())
            }
            other => other.clone(),
        };

        let content = tool.call(args_value).await?;

        Ok(ToolResult { tool_call_id: tool_call.id.clone(), content })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ToolCallFunction;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {"msg": {"type": "string"}}})
        }
        async fn call(&self, args: Value) -> Result<String, AgentError> {
            Ok(args.get("msg").and_then(|v| v.as_str()).unwrap_or("").to_string())
        }
    }

    #[tokio::test]
    async fn test_tool_engine_register_and_execute() {
        let mut engine = ToolEngine::new();
        engine.register(Arc::new(EchoTool));

        let call = ToolCall {
            id: "call_1".into(),
            function: ToolCallFunction {
                name: "echo".into(),
                arguments: serde_json::json!({"msg": "hello"}),
            },
        };
        let result = engine.execute(&call).await.unwrap();
        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, "hello");
    }

    #[tokio::test]
    async fn test_tool_engine_tool_not_found() {
        let engine = ToolEngine::new();
        let call = ToolCall {
            id: "call_1".into(),
            function: ToolCallFunction {
                name: "nonexistent".into(),
                arguments: serde_json::json!({}),
            },
        };
        let err = engine.execute(&call).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn test_tool_engine_get_defs() {
        let mut engine = ToolEngine::new();
        engine.register(Arc::new(EchoTool));
        let defs = engine.get_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }

    #[tokio::test]
    async fn test_tool_engine_string_arguments() {
        let mut engine = ToolEngine::new();
        engine.register(Arc::new(EchoTool));

        let call = ToolCall {
            id: "call_1".into(),
            function: ToolCallFunction {
                name: "echo".into(),
                arguments: Value::String(r#"{"msg": "hi"}"#.into()),
            },
        };
        let result = engine.execute(&call).await.unwrap();
        assert_eq!(result.content, "hi");
    }
}

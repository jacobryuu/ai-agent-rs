use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serde_roundtrip() {
        let msg = Message {
            role: Role::User,
            content: "hello".into(),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hello"}"#);
    }

    #[test]
    fn test_message_with_tool_calls() {
        let msg = Message {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                function: ToolCallFunction {
                    name: "glob".into(),
                    arguments: serde_json::json!({"pattern": "*.rs"}),
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"glob\""));
        assert!(json.contains("\"*.rs\""));
    }

    #[test]
    fn test_tool_result() {
        let tr = ToolResult { tool_call_id: "call_1".into(), content: "result text".into() };
        let json = serde_json::to_string(&tr).unwrap();
        assert!(json.contains("\"tool_call_id\":\"call_1\""));
        assert!(json.contains("\"result text\""));
    }

    #[test]
    fn test_tool_def() {
        let td = ToolDef {
            name: "test_tool".into(),
            description: "A test tool".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        };
        let json = serde_json::to_string(&td).unwrap();
        assert!(json.contains("\"test_tool\""));
    }

    #[test]
    fn test_llm_response() {
        let resp = LLMResponse { content: Some("answer".into()), tool_calls: None };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"answer\""));

        let resp = LLMResponse {
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                function: ToolCallFunction {
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": "ls"}),
                },
            }]),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"bash\""));
    }

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), r#""system""#);
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), r#""user""#);
        assert_eq!(serde_json::to_string(&Role::Assistant).unwrap(), r#""assistant""#);
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), r#""tool""#);
    }
}

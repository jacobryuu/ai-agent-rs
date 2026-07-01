use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::agent::graph::{Node, State};
use crate::core::{AgentError, LLMProvider, Message, Role, ToolCall};
use crate::tools::ToolEngine;

pub struct LLMNode {
    provider: Box<dyn LLMProvider>,
    tools: ToolEngine,
    system_prompt: String,
}

impl LLMNode {
    pub fn new(provider: Box<dyn LLMProvider>, tools: ToolEngine, system_prompt: String) -> Self {
        Self { provider, tools, system_prompt }
    }
}

#[async_trait]
impl Node for LLMNode {
    async fn process(&self, state: &mut State) -> Result<(), AgentError> {
        debug!("Processing LLMNode");

        let iteration: usize =
            state.context.get("iteration").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if iteration >= 25 {
            return Err(AgentError::LLMError("Max iterations (25) reached".into()));
        }
        state.context.insert("iteration".into(), serde_json::json!(iteration + 1));

        let tool_defs = self.tools.get_defs();

        let mut messages = Vec::new();
        messages.push(Message {
            role: Role::System,
            content: self.system_prompt.clone(),
            tool_calls: None,
            tool_call_id: None,
        });
        messages.extend(state.messages.clone());

        let response = self.provider.chat(&messages, &tool_defs).await?;

        let valid_calls: Vec<ToolCall> = response
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .filter(|c| !c.function.name.is_empty())
            .collect();

        if !valid_calls.is_empty() {
            state.context.insert("last_tool_count".into(), serde_json::json!(valid_calls.len()));
            state.context.insert(
                "last_tool_names".into(),
                serde_json::json!(valid_calls.iter().map(|c| &c.function.name).collect::<Vec<_>>()),
            );

            state.messages.push(Message {
                role: Role::Assistant,
                content: response.content.unwrap_or_default(),
                tool_calls: Some(valid_calls.clone()),
                tool_call_id: None,
            });
            state.next_node = Some("tools".to_string());
            info!("LLM requested {} tool calls", valid_calls.len());
        } else if let Some(content) = response.content {
            state.context.insert("has_final_response".into(), serde_json::json!(true));
            state.messages.push(Message {
                role: Role::Assistant,
                content: content.clone(),
                tool_calls: None,
                tool_call_id: None,
            });
            state.next_node = Some("__end__".to_string());
            info!("LLM returned final response");
        } else {
            return Err(AgentError::LLMError("Model returned no response".into()));
        }

        Ok(())
    }
}

pub struct ToolNode {
    tools: ToolEngine,
}

impl ToolNode {
    pub fn new(tools: ToolEngine) -> Self {
        Self { tools }
    }
}

#[async_trait]
impl Node for ToolNode {
    async fn process(&self, state: &mut State) -> Result<(), AgentError> {
        debug!("Processing ToolNode");

        let tool_calls = state.messages.last().and_then(|m| m.tool_calls.clone());

        if let Some(tool_calls) = tool_calls {
            info!("Executing {} tool calls", tool_calls.len());
            state.context.insert("tool_results".into(), serde_json::json!([]));
            let mut results = Vec::new();
            for tool_call in &tool_calls {
                info!("Executing tool: {}", tool_call.function.name);
                let result = self.tools.execute(tool_call).await?;
                results.push(serde_json::json!({
                    "tool": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                }));
                state.messages.push(Message {
                    role: Role::Tool,
                    content: result.content,
                    tool_calls: None,
                    tool_call_id: Some(result.tool_call_id),
                });
            }
            state.context.insert("tool_results".into(), serde_json::json!(results));
            state.next_node = Some("llm".to_string());
        } else {
            warn!("ToolNode called but last message has no tool calls");
            state.next_node = Some("llm".to_string());
        }

        Ok(())
    }
}

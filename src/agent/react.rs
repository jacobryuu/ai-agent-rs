use tracing::warn;

use crate::core::{AgentError, LLMProvider, Message, Role, ToolCall};
use crate::memory::SlidingWindowMemory;
use crate::tools::ToolEngine;

pub struct ReActAgent {
    provider: Box<dyn LLMProvider>,
    tools: ToolEngine,
    memory: SlidingWindowMemory,
    system_prompt: String,
    max_iterations: usize,
}

impl ReActAgent {
    pub fn new(
        provider: Box<dyn LLMProvider>,
        tools: ToolEngine,
        system_prompt: String,
        max_iterations: usize,
    ) -> Self {
        Self {
            provider,
            tools,
            memory: SlidingWindowMemory::new(20),
            system_prompt,
            max_iterations,
        }
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        self.memory.add(Message {
            role: Role::User,
            content: user_input.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });

        let tool_defs = self.tools.get_defs();

        for _iteration in 0..self.max_iterations {
            let mut messages = Vec::new();

            messages.push(Message {
                role: Role::System,
                content: self.system_prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
            });

            messages.extend(self.memory.get_history());

            let response = self.provider.chat(&messages, &tool_defs).await?;

            let valid_calls: Vec<ToolCall> = response
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .filter(|c| !c.function.name.is_empty())
                .collect();

            if !valid_calls.is_empty() {
                self.memory.add(Message {
                    role: Role::Assistant,
                    content: response.content.unwrap_or_default(),
                    tool_calls: Some(valid_calls.clone()),
                    tool_call_id: None,
                });

                for tool_call in &valid_calls {
                    let result = self.tools.execute(tool_call).await?;
                    self.memory.add(Message {
                        role: Role::Tool,
                        content: result.content,
                        tool_calls: None,
                        tool_call_id: Some(result.tool_call_id),
                    });
                }
            } else if let Some(content) = response.content {
                if !content.trim().is_empty() {
                    self.memory.add(Message {
                        role: Role::Assistant,
                        content: content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                    return Ok(content);
                }
                warn!("LLM returned empty content and no valid tool calls");
                return Err(AgentError::LLMError("Model returned empty response".into()));
            } else {
                warn!("LLM returned no content and no valid tool calls");
                return Err(AgentError::LLMError("Model returned no response".into()));
            }
        }

        Err(AgentError::LLMError("Max iterations reached without a final response".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LLMResponse;
    use crate::tools::builtin::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockProvider;

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[crate::core::ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            Ok(LLMResponse { content: Some("mock response".into()), tool_calls: None })
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    struct MockToolProvider;

    #[async_trait]
    impl LLMProvider for MockToolProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[crate::core::ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            Ok(LLMResponse {
                content: Some(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    function: crate::core::ToolCallFunction {
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "echo 'tool called'"}),
                    },
                }]),
            })
        }

        fn model_name(&self) -> &str {
            "mock_tool"
        }
    }

    struct MockChainProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl LLMProvider for MockChainProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[crate::core::ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count == 0 {
                Ok(LLMResponse {
                    content: Some(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".into(),
                        function: crate::core::ToolCallFunction {
                            name: "bash".into(),
                            arguments: serde_json::json!({"command": "echo tool_call"}),
                        },
                    }]),
                })
            } else {
                Ok(LLMResponse { content: Some("final answer".into()), tool_calls: None })
            }
        }

        fn model_name(&self) -> &str {
            "mock_chain"
        }
    }

    #[tokio::test]
    async fn test_direct_response() {
        let provider = Box::new(MockProvider);
        let tools = ToolEngine::new();
        let mut agent = ReActAgent::new(provider, tools, "prompt".into(), 10);

        let result = agent.run("hello").await.unwrap();
        assert_eq!(result, "mock response");
    }

    #[tokio::test]
    async fn test_tool_execution_then_response() {
        let provider =
            Box::new(MockChainProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut tools = ToolEngine::new();
        tools.register(Arc::new(BashTool));
        let mut agent = ReActAgent::new(provider, tools, "prompt".into(), 10);

        let result = agent.run("run a command").await.unwrap();
        assert_eq!(result, "final answer");
    }

    #[tokio::test]
    async fn test_max_iterations() {
        let provider = Box::new(MockToolProvider);
        let mut tools = ToolEngine::new();
        tools.register(Arc::new(BashTool));
        let mut agent = ReActAgent::new(provider, tools, "prompt".into(), 3);

        let result = agent.run("loop").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_tool_call_filtered() {
        struct EmptyNameProvider;
        #[async_trait]
        impl LLMProvider for EmptyNameProvider {
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: &[crate::core::ToolDef],
            ) -> Result<LLMResponse, AgentError> {
                Ok(LLMResponse {
                    content: Some("fallback response".into()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".into(),
                        function: crate::core::ToolCallFunction {
                            name: String::new(),
                            arguments: serde_json::json!({}),
                        },
                    }]),
                })
            }
            fn model_name(&self) -> &str {
                "empty"
            }
        }

        let provider = Box::new(EmptyNameProvider);
        let tools = ToolEngine::new();
        let mut agent = ReActAgent::new(provider, tools, "prompt".into(), 10);

        let result = agent.run("test").await.unwrap();
        assert_eq!(result, "fallback response");
    }
}

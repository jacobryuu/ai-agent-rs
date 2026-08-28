use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::core::error::AgentError;
use crate::core::message::{LLMResponse, Message, ToolDef};

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError>;

    async fn chat_stream<'a>(
        &'a self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<BoxStream<'a, Result<StreamingChunk, AgentError>>, AgentError> {
        let response = self.chat(messages, tools).await?;
        let content = response.content.unwrap_or_default();
        let chunk =
            StreamingChunk { content: Some(content), tool_calls: response.tool_calls, done: true };
        let stream = futures::stream::once(async { Ok(chunk) });
        Ok(Box::pin(stream))
    }

    fn model_name(&self) -> &str;

    fn supports_streaming(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct StreamingChunk {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<crate::core::message::ToolCall>>,
    pub done: bool,
}

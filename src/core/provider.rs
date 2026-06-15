use async_trait::async_trait;

use crate::core::error::AgentError;
use crate::core::message::{LLMResponse, Message, ToolDef};

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError>;

    fn model_name(&self) -> &str;
}

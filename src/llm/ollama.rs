use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::core::{AgentError, LLMProvider, LLMResponse, Message, ToolCall, ToolDef};

pub struct OllamaProvider {
    client: Client,
    host: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(host: String, model: String) -> Self {
        Self { client: Client::new(), host: host.trim_end_matches('/').to_string(), model }
    }
}

#[derive(Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    type_: String,
    function: OllamaToolFunction,
}

#[derive(Serialize)]
struct OllamaToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

impl From<ToolDef> for OllamaTool {
    fn from(t: ToolDef) -> Self {
        OllamaTool {
            type_: "function".to_string(),
            function: OllamaToolFunction {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
            },
        }
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaResponseMessage,
    done: bool,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError> {
        let url = format!("{}/api/chat", self.host);

        debug!(
            "Ollama request: model={}, messages={}, tools={}",
            self.model,
            messages.len(),
            tools.len()
        );

        let ollama_tools: Vec<OllamaTool> = tools.iter().cloned().map(OllamaTool::from).collect();

        let request = OllamaRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: ollama_tools,
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::LLMError(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AgentError::LLMError(format!("API error ({}): {}", status, body)));
        }

        let resp_text = resp
            .text()
            .await
            .map_err(|e| AgentError::LLMError(format!("Read body failed: {}", e)))?;

        debug!("Raw Ollama response: {}", resp_text);

        let ollama_resp: OllamaResponse = serde_json::from_str(&resp_text).map_err(|e| {
            AgentError::LLMError(format!(
                "Parse failed: {} - body: {}",
                e,
                &resp_text[..resp_text.len().min(500)]
            ))
        })?;

        debug!(
            "Ollama response: done={}, has_tool_calls={}",
            ollama_resp.done,
            ollama_resp.message.tool_calls.is_some()
        );

        if let Some(ref calls) = ollama_resp.message.tool_calls {
            for call in calls {
                debug!(
                    "Tool call: id={}, name='{}', args={}",
                    call.id, call.function.name, call.function.arguments
                );
            }
        }

        Ok(LLMResponse {
            content: ollama_resp.message.content,
            tool_calls: ollama_resp.message.tool_calls,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

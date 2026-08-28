use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::core::provider::StreamingChunk;
use crate::core::{AgentError, LLMProvider, LLMResponse, Message, ToolCall, ToolDef};

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self { client: Client::new(), api_key, model, base_url }
    }
}

#[derive(Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    type_: String,
    function: OpenAIToolFunction,
}

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

impl From<ToolDef> for OpenAITool {
    fn from(t: ToolDef) -> Self {
        OpenAITool {
            type_: "function".to_string(),
            function: OpenAIToolFunction {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
}

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    function: OpenAIToolCallFunction,
}

#[derive(Serialize, Deserialize)]
struct OpenAIToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Deserialize)]
struct OpenAIStreamingChunk {
    choices: Vec<OpenAIStreamingChoice>,
}

#[derive(Deserialize)]
struct OpenAIStreamingChoice {
    delta: OpenAIStreamingDelta,
}

#[derive(Deserialize)]
struct OpenAIStreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIStreamingToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIStreamingToolCall {
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIStreamingFunction>,
}

#[derive(Deserialize)]
struct OpenAIStreamingFunction {
    name: Option<String>,
    arguments: Option<String>,
}

impl From<&Message> for OpenAIMessage {
    fn from(m: &Message) -> Self {
        let role = match m.role {
            crate::core::Role::System => "system",
            crate::core::Role::User => "user",
            crate::core::Role::Assistant => "assistant",
            crate::core::Role::Tool => "tool",
        };

        let tool_calls = m.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|c| OpenAIToolCall {
                    id: c.id.clone(),
                    type_: "function".to_string(),
                    function: OpenAIToolCallFunction {
                        name: c.function.name.clone(),
                        arguments: serde_json::to_string(&c.function.arguments).unwrap_or_default(),
                    },
                })
                .collect()
        });

        OpenAIMessage {
            role: role.to_string(),
            content: Some(m.content.clone()),
            tool_calls,
            tool_call_id: m.tool_call_id.clone(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);

        debug!(
            "OpenAI request: model={}, messages={}, tools={}",
            self.model,
            messages.len(),
            tools.len()
        );

        let openai_tools: Vec<OpenAITool> = tools.iter().cloned().map(OpenAITool::from).collect();

        let openai_messages: Vec<OpenAIMessage> =
            messages.iter().map(OpenAIMessage::from).collect();

        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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

        debug!("Raw OpenAI response: {}", resp_text);

        let openai_resp: OpenAIResponse = serde_json::from_str(&resp_text).map_err(|e| {
            AgentError::LLMError(format!(
                "Parse failed: {} - body: {}",
                e,
                &resp_text[..resp_text.len().min(500)]
            ))
        })?;

        let choice = openai_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AgentError::LLMError("No choices in response".into()))?;

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|c| {
                    let args: Value = serde_json::from_str(&c.function.arguments)
                        .unwrap_or(Value::String(c.function.arguments));
                    ToolCall {
                        id: c.id,
                        function: crate::core::ToolCallFunction {
                            name: c.function.name,
                            arguments: args,
                        },
                    }
                })
                .collect()
        });

        Ok(LLMResponse { content: choice.message.content, tool_calls })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_stream<'a>(
        &'a self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<BoxStream<'a, Result<StreamingChunk, AgentError>>, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);

        debug!(
            "OpenAI streaming request: model={}, messages={}, tools={}",
            self.model,
            messages.len(),
            tools.len()
        );

        let openai_tools: Vec<OpenAITool> = tools.iter().cloned().map(OpenAITool::from).collect();

        let openai_messages: Vec<OpenAIMessage> =
            messages.iter().map(OpenAIMessage::from).collect();

        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            tools: openai_tools,
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": request.model,
                "messages": request.messages,
                "tools": request.tools,
                "stream": true,
            }))
            .send()
            .await
            .map_err(|e| AgentError::LLMError(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AgentError::LLMError(format!("API error ({}): {}", status, body)));
        }

        let byte_stream = resp.bytes_stream();
        let accumulated_content = String::new();
        let accumulated_tool_calls: Vec<ToolCall> = Vec::new();

        let stream = futures::stream::unfold(
            (byte_stream, accumulated_content, accumulated_tool_calls),
            |(mut byte_stream, mut acc_content, mut acc_tools)| async {
                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => {
                            return Some((
                                Err(AgentError::LLMError(format!("Stream error: {}", e))),
                                (byte_stream, acc_content, acc_tools),
                            ));
                        }
                    };

                    let text = String::from_utf8_lossy(&chunk);
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line == "data: [DONE]" {
                            continue;
                        }

                        if let Some(data) = line.strip_prefix("data: ") {
                            match serde_json::from_str::<OpenAIStreamingChunk>(data) {
                                Ok(stream_chunk) => {
                                    if let Some(choice) = stream_chunk.choices.into_iter().next() {
                                        if let Some(content) = choice.delta.content {
                                            acc_content.push_str(&content);
                                        }

                                        if let Some(ref tool_calls) = choice.delta.tool_calls {
                                            for tc in tool_calls {
                                                if let Some(ref func) = tc.function {
                                                    let tc_id = tc.id.clone().unwrap_or_default();
                                                    let name =
                                                        func.name.clone().unwrap_or_default();
                                                    let args =
                                                        func.arguments.clone().unwrap_or_default();

                                                    let idx = acc_tools
                                                        .iter()
                                                        .position(|t| {
                                                            !tc_id.is_empty() && t.id == tc_id
                                                        })
                                                        .or_else(|| {
                                                            if tc_id.is_empty()
                                                                && !acc_tools.is_empty()
                                                            {
                                                                Some(acc_tools.len() - 1)
                                                            } else {
                                                                None
                                                            }
                                                        });

                                                    match idx {
                                                        Some(idx) => {
                                                            if let Value::String(ref mut a) =
                                                                acc_tools[idx].function.arguments
                                                            {
                                                                a.push_str(&args);
                                                            }
                                                            if !name.is_empty() {
                                                                acc_tools[idx].function.name = name;
                                                            }
                                                        }
                                                        None => acc_tools.push(ToolCall {
                                                            id: tc_id,
                                                            function:
                                                                crate::core::ToolCallFunction {
                                                                    name,
                                                                    arguments: Value::String(args),
                                                                },
                                                        }),
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    debug!(
                                        "Failed to parse streaming line: {} - line: {}",
                                        e, line
                                    );
                                }
                            }
                        }
                    }
                }

                let tool_calls = if acc_tools.is_empty() { None } else { Some(acc_tools) };

                Some((
                    Ok(StreamingChunk { content: Some(acc_content), tool_calls, done: true }),
                    (byte_stream, String::new(), Vec::new()),
                ))
            },
        );

        Ok(Box::pin(stream))
    }
}

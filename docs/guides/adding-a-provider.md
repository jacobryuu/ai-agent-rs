# Adding an LLM Provider

Providers implement the `LLMProvider` trait from `src/core/provider.rs`.

## Step 1: Create the module

```rust
// src/llm/openai.rs
use async_trait::async_trait;
use reqwest::Client;
use crate::core::{AgentError, LLMProvider, LLMResponse, Message, ToolDef};

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { client: Client::new(), api_key, model }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError> {
        // POST to OpenAI /v1/chat/completions
        // Parse tool_calls from response
        // Return LLMResponse
        todo!()
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
```

## Step 2: Register the module

```rust
// src/llm/mod.rs
pub mod ollama;
pub mod openai;
```

## Step 3: Use in main.rs

```rust
use crate::llm::openai::OpenAIProvider;

let provider: Box<dyn LLMProvider> = if use_openai {
    Box::new(OpenAIProvider::new(api_key, "gpt-4".into()))
} else {
    Box::new(OllamaProvider::new(host, model))
};
```

## Key Points

- The `chat()` method receives messages (including system prompt) and tool definitions
- Return `LLMResponse` with either `content` (final answer) or `tool_calls` (function calling)
- Errors must be wrapped in `AgentError::LLMError(String)`
- Providers are `Send + Sync` — no interior mutability unless behind a lock
- Test with mock providers (see `src/agent/react.rs` for examples)

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::core::AgentError;
use crate::rag::{EmbeddingProvider, VectorStore};
use crate::tools::Tool;

pub struct SearchDocsTool {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_store: Arc<dyn VectorStore>,
}

impl SearchDocsTool {
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self { embedding_provider, vector_store }
    }
}

#[async_trait]
impl Tool for SearchDocsTool {
    fn name(&self) -> &str {
        "search_docs"
    }

    fn description(&self) -> &str {
        "Search for relevant information in the indexed documentation using semantic search"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information"
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of top results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'query' argument".into()))?;

        let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        let query_vector = self.embedding_provider.embed(query).await?;
        let results = self.vector_store.search(query_vector, top_k).await?;

        if results.is_empty() {
            return Ok("No relevant documentation found.".to_string());
        }

        let mut output = String::from("Found relevant documentation:\n\n");
        for (i, res) in results.iter().enumerate() {
            output.push_str(&format!("{}. [Score: {:.4}] {}\n", i + 1, res.score, res.id));
            if let Some(text) = res.metadata.get("text").and_then(|v| v.as_str()) {
                output.push_str(&format!("   Content: {}\n\n", text));
            } else {
                output.push_str(&format!("   Metadata: {}\n\n", res.metadata));
            }
        }

        Ok(output)
    }
}

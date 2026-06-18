use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("LLM request failed: {0}")]
    LLMError(String),

    #[error("Tool execution failed: {0}")]
    ToolError(String),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("RAG error: {0}")]
    RAGError(String),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("Vector store error: {0}")]
    VectorStoreError(String),

    #[allow(dead_code)]
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = AgentError::LLMError("timeout".into());
        assert_eq!(err.to_string(), "LLM request failed: timeout");

        let err = AgentError::ToolError("permission denied".into());
        assert_eq!(err.to_string(), "Tool execution failed: permission denied");

        let err = AgentError::ToolNotFound("glob".into());
        assert_eq!(err.to_string(), "Tool not found: glob");
    }

    #[test]
    fn test_serde_error_from() {
        let result: Result<(), AgentError> =
            Err(serde_json::from_str::<()>("invalid").unwrap_err().into());
        assert!(matches!(result, Err(AgentError::SerdeError(_))));
    }

    #[test]
    fn test_io_error_from() {
        let result: Result<(), AgentError> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into());
        assert!(matches!(result, Err(AgentError::IOError(_))));
    }
}

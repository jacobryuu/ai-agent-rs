pub struct Config {
    pub ollama_host: String,
    pub ollama_model: String,
    pub vector_store_uri: String,
    pub vector_table_name: String,
    pub openai_api_key: Option<String>,
    pub openai_model: String,
    pub openai_base_url: String,
    pub llm_provider: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string()),
            vector_store_uri: std::env::var("VECTOR_STORE_URI")
                .unwrap_or_else(|_| "data/lancedb".to_string()),
            vector_table_name: std::env::var("VECTOR_TABLE_NAME")
                .unwrap_or_else(|_| "documents".to_string()),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            openai_model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()),
            openai_base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            llm_provider: std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "ollama".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = Config::from_env();
        assert_eq!(config.ollama_host, "http://localhost:11434");
        assert_eq!(config.ollama_model, "qwen3:8b");
    }

    #[test]
    fn test_custom_host() {
        temp_env::with_var("OLLAMA_HOST", Some("http://custom:8080"), || {
            let config = Config::from_env();
            assert_eq!(config.ollama_host, "http://custom:8080");
        });
    }

    #[test]
    fn test_custom_model() {
        temp_env::with_var("OLLAMA_MODEL", Some("llama3:latest"), || {
            let config = Config::from_env();
            assert_eq!(config.ollama_model, "llama3:latest");
        });
    }
}

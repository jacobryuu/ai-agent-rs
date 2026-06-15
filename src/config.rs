pub struct Config {
    pub ollama_host: String,
    pub ollama_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string()),
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

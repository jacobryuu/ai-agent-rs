mod agent;
mod config;
mod core;
mod llm;
mod memory;
mod rag;
mod tools;

use std::sync::Arc;

use tracing::{info, warn};

use crate::agent::GraphAgent;
use crate::config::Config;
use crate::core::LLMProvider;
use crate::llm::ollama::OllamaProvider;
use crate::rag::{CandleEmbeddingProvider, EmbeddingProvider, LanceVectorStore};
use crate::tools::ToolEngine;
use crate::tools::builtin::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();

    let provider =
        Box::new(OllamaProvider::new(config.ollama_host.clone(), config.ollama_model.clone()));

    // Initialize RAG
    let embedding_provider = match CandleEmbeddingProvider::new() {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            warn!("Failed to initialize embedding provider: {}. RAG features will be disabled.", e);
            None
        }
    };

    let vector_store = if let Some(ref ep) = embedding_provider {
        match LanceVectorStore::new(
            &config.vector_store_uri,
            &config.vector_table_name,
            ep.dimension(),
        )
        .await
        {
            Ok(vs) => Some(Arc::new(vs)),
            Err(e) => {
                warn!("Failed to initialize vector store: {}. RAG features will be disabled.", e);
                None
            }
        }
    } else {
        None
    };

    let mut tool_engine = ToolEngine::new();
    tool_engine.register(Arc::new(ReadFileTool));
    tool_engine.register(Arc::new(WriteFileTool));
    tool_engine.register(Arc::new(GlobTool));
    tool_engine.register(Arc::new(GrepTool));
    tool_engine.register(Arc::new(BashTool));

    if let (Some(ep), Some(vs)) = (embedding_provider, vector_store) {
        tool_engine.register(Arc::new(SearchDocsTool::new(ep, vs)));
        info!("RAG features enabled.");
    }

    let system_prompt = r#"You are an AI assistant that helps users by using tools.

Available tools (use these by their exact name):
- read_file(path: string): Read file contents
- write_file(path: string, content: string): Write content to a file
- glob(pattern: string): Search for files matching a glob pattern
- grep(pattern: string, path: string): Search file contents with regex
- bash(command: string): Execute shell commands
- search_docs(query: string, top_k: integer): Search for relevant information in the documentation

When you need to use a tool, the system will handle the function call for you.
Just reason about what tool to use and the system will execute it.
After receiving tool results, continue helping the user with the information.
If you can answer without tools, just respond directly."#
        .to_string();

    info!("AI Agent started with model: {}", provider.model_name());
    println!("AI Agent (Ollama: {})", config.ollama_model);

    let mut agent = GraphAgent::new(provider, tool_engine, system_prompt);
    println!("Type 'exit' to quit");
    println!("{}", "-".repeat(40));

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match agent.run(&input).await {
            Ok(response) => println!("{}", response),
            Err(e) => eprintln!("Error: {}", e),
        }
        println!();
    }

    Ok(())
}

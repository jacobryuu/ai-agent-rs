mod agent;
mod config;
mod core;
mod llm;
mod memory;
mod tools;

use std::sync::Arc;

use tracing::info;

use crate::agent::ReActAgent;
use crate::config::Config;
use crate::core::LLMProvider;
use crate::llm::ollama::OllamaProvider;
use crate::tools::ToolEngine;
use crate::tools::builtin::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();

    let provider =
        Box::new(OllamaProvider::new(config.ollama_host.clone(), config.ollama_model.clone()));

    let mut tool_engine = ToolEngine::new();
    tool_engine.register(Arc::new(ReadFileTool));
    tool_engine.register(Arc::new(WriteFileTool));
    tool_engine.register(Arc::new(GlobTool));
    tool_engine.register(Arc::new(GrepTool));
    tool_engine.register(Arc::new(BashTool));

    let system_prompt = r#"You are an AI assistant that helps users by using tools.

Available tools (use these by their exact name):
- read_file(path: string): Read file contents
- write_file(path: string, content: string): Write content to a file
- glob(pattern: string): Search for files matching a glob pattern
- grep(pattern: string, path: string): Search file contents with regex
- bash(command: string): Execute shell commands

When you need to use a tool, the system will handle the function call for you.
Just reason about what tool to use and the system will execute it.
After receiving tool results, continue helping the user with the information.
If you can answer without tools, just respond directly."#
        .to_string();

    info!("AI Agent started with model: {}", provider.model_name());
    println!("AI Agent (Ollama: {})", config.ollama_model);

    let mut agent = ReActAgent::new(provider, tool_engine, system_prompt, 10);
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

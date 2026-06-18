# AI Agent in Rust (ai-agent-rs)

This project is a high-performance AI agent implemented in Rust, designed to interact with local LLMs via Ollama. It follows the **ReAct (Reasoning and Acting)** pattern, enabling the agent to use tools for tasks like file manipulation and shell command execution.

## Project Overview

- **Core Technology:** Rust (Edition 2024), Tokio (Async runtime), Reqwest (HTTP client).
- **LLM Backend:** Ollama (default model: `qwen3:8b`).
- **Architecture:**
    - `core/`: Foundation traits (`LLMProvider`) and data structures (`Message`, `ToolCall`, `ToolDef`).
    - `llm/`: Provider implementations (currently `OllamaProvider`).
    - `agent/`: **Graph-based** orchestration (`GraphRunner`) with nodes like `LLMNode` and `ToolNode`.
    - `rag/`: Vector search layer using **LanceDB** (embedded) and **candle** (local embeddings).
    - `tools/`: Extensible tool engine with built-in support for `read_file`, `write_file`, `glob`, `grep`, `bash`, and **`search_docs`**.
    - `memory/`: Conversation history management using a `SlidingWindowMemory`.

## Building and Running

### Prerequisites

1.  **Rust:** Install via [rustup](https://rustup.rs/).
2.  **Ollama:** Install from [ollama.com](https://ollama.com/).
3.  **Model:** Pull the default model:
    ```bash
    ollama pull qwen3:8b
    ```

### Key Commands

- **Run the CLI:**
  ```bash
  cargo run
  ```
  The agent will start an interactive session. Type `exit` to quit.

- **Run Tests:**
  ```bash
  cargo test
  ```

- **Configuration:**
  Environment variables can be used to override defaults:
  - `OLLAMA_HOST`: Default is `http://localhost:11434`.
  - `OLLAMA_MODEL`: Default is `qwen3:8b`.

## Development Conventions

- **Async First:** The project uses `tokio` for all I/O and `async-trait` for polymorphic interfaces.
- **Error Handling:** Centralized in `src/core/error.rs` using the `thiserror` crate.
- **Logging:** Instrumented with the `tracing` crate. Logs can be configured via `RUST_LOG`.
- **Testing:** New features should include unit tests within the same file (in a `mod tests` block). Mock providers are used to test the agent loop without requiring a live Ollama server.
- **Tools:** To add a new tool, implement the `Tool` trait in `src/tools/` and register it in `src/main.rs`.

## Future Roadmap (Planned Phases)

- **Phase 2:** Streaming responses and additional tools (e.g., `web_search`).
- **Phase 3:** Long-term memory using vector databases or summarization.
- **Phase 4:** Cloud LLM support (OpenAI, Anthropic).
- **Phase 5:** Multi-agent orchestration.
- **Note:** If adding RAG, LangChain, or LangGraph-like features, review `design.md` for architectural alignment.

# Getting Started

## Prerequisites

- **Rust** (latest stable) — install via [rustup](https://rustup.rs/)
- **Ollama** — install from [ollama.com](https://ollama.com/)
- A compatible model (default: `qwen3:8b`)

```bash
ollama pull qwen3:8b
```

## Build & Run

```bash
# Build
cargo build

# Run CLI
cargo run

# Run tests
cargo test
```

The agent starts an interactive REPL. Type your message at the `>` prompt. Enter `exit` or `quit` to stop.

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Model name for inference |
| `VECTOR_STORE_URI` | `data/lancedb` | LanceDB storage path |
| `VECTOR_TABLE_NAME` | `documents` | Vector table name |

Example:

```bash
OLLAMA_MODEL=llama3:latest cargo run
```

## What Happens on Startup

1. `Config::from_env()` reads environment variables
2. `OllamaProvider` connects to the configured host/model
3. `CandleEmbeddingProvider` loads a BERT model (`all-MiniLM-L6-v2`) from HuggingFace
4. `LanceVectorStore` opens or creates the vector database
5. `ToolEngine` registers all 6 built-in tools
6. `GraphAgent` wires LLM + tools + memory together
7. The REPL loop begins

If RAG initialization fails (no HuggingFace access, missing model), the agent **still starts** with all other features — only `search_docs` is unavailable.

## Indexing Documents for RAG

The vector store is **empty** by default. To populate it, use the `index` subcommand:

```bash
cargo run -- index ./docs
```

This scans the directory, chunks `.md` and `.txt` files into 512-char segments, embeds them with BERT, and stores the vectors in LanceDB.

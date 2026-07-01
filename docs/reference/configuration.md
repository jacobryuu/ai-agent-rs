# Configuration Reference

All configuration is through environment variables.

## General

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | (none) | Tracing/log level: `info`, `debug`, `warn`, `error` |

## Ollama

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Model name for chat completions |

## Vector Store

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_URI` | `data/lancedb` | Filesystem path for LanceDB storage |
| `VECTOR_TABLE_NAME` | `documents` | LanceDB table name for document vectors |

## Example

```bash
RUST_LOG=debug \
  OLLAMA_HOST=http://192.168.1.100:11434 \
  OLLAMA_MODEL=llama3:70b \
  VECTOR_STORE_URI=/data/agent/lancedb \
  VECTOR_TABLE_NAME=my_docs \
  cargo run
```

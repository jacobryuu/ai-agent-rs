# Architecture

## Overview

```
┌──────────────────────────────────────────────────┐
│                  User (CLI REPL)                  │
└────────────────────┬─────────────────────────────┘
                     │  text input / commands
                     ▼
┌──────────────────────────────────────────────────┐
│                   GraphAgent                      │
│  ┌─────────────────────────────────────────────┐  │
│  │              GraphRunner                     │  │
│  │  ┌──────────┐   loop    ┌──────────┐        │  │
│  │  │ LLMNode  │◄─────────│ ToolNode │        │  │
│  │  └────┬─────┘           └──────────┘        │  │
│  │       │                                      │  │
│  │  ┌────▼─────┐                                │  │
│  │  │ __end__  │  (stop)                        │  │
│  │  └──────────┘                                │  │
│  └─────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────┐  │
│  │        SlidingWindowMemory                  │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌──────────────┐ ┌────────┐ ┌──────────────┐
│ LLMProvider  │ │ Tools  │ │ RAG Layer    │
│ (Ollama)     │ │  x6    │ │ c+ndle+DB    │
└──────────────┘ └────────┘ └──────────────┘
```

## Module Layout

### `src/core/` — Foundation

| File | Purpose |
|------|---------|
| `error.rs` | `AgentError` enum with 9 variants, `From` impls for serde/io/candle |
| `message.rs` | `Message`, `Role`, `ToolCall`, `ToolDef`, `LLMResponse`, `ToolResult` |
| `provider.rs` | `LLMProvider` trait — the interface every LLM backend implements |

### `src/llm/` — LLM Backends

| File | Purpose |
|------|---------|
| `ollama.rs` | `OllamaProvider` — calls Ollama `/api/chat` with tool support |

Adding a new provider (e.g., OpenAI) means implementing `LLMProvider`.

### `src/agent/` — Agent Orchestration

| File | Purpose |
|------|---------|
| `mod.rs` | `GraphAgent` — public API, wires graph + memory |
| `react.rs` | `ReActAgent` — legacy loop agent (superseded by graph) |
| `graph/mod.rs` | `GraphRunner` — hash-map-based node registry, loop until `__end__` |
| `graph/state.rs` | `State` — holds `messages`, `context`, `next_node` |
| `graph/nodes/react.rs` | `LLMNode`, `ToolNode` — the two concrete node types |

### `src/tools/` — Tool System

| File | Purpose |
|------|---------|
| `mod.rs` | `Tool` trait, `ToolEngine` registry |
| `builtin/read_file.rs` | Read file contents |
| `builtin/write_file.rs` | Write file with parent dir creation |
| `builtin/glob.rs` | Glob pattern matching |
| `builtin/grep.rs` | Regex search via `grep -rn` |
| `builtin/bash.rs` | Shell command execution (10K truncation) |
| `builtin/rag.rs` | `SearchDocsTool` — semantic search over indexed docs |

### `src/rag/` — Retrieval-Augmented Generation

| File | Purpose |
|------|---------|
| `embeddings.rs` | `EmbeddingProvider` trait, `CandleEmbeddingProvider` (BERT `all-MiniLM-L6-v2`) |
| `vector_store.rs` | `VectorStore` trait, `LanceVectorStore` (embedded LanceDB) |

### `src/memory/` — Conversation Memory

| File | Purpose |
|------|---------|
| `sliding.rs` | `SlidingWindowMemory` — `VecDeque<Message>` with max limit, preserves `System` messages |

### `src/config.rs` — Configuration

Reads `OLLAMA_HOST`, `OLLAMA_MODEL`, `VECTOR_STORE_URI`, `VECTOR_TABLE_NAME` from environment.

## Agent Execution Flow

```
User Input ("list all .rs files")
    │
    ▼
GraphAgent::run(input)
    │
    ├─ push User message to State
    ├─ call GraphRunner::run(state)
    │   │
    │   ├─ LLMNode::process()
    │   │   ├─ prepend System prompt
    │   │   ├─ call OllamaProvider.chat(messages + tool_defs)
    │   │   ├─ if tool_calls → state.next_node = "tools"
    │   │   └─ if content → state.next_node = "__end__"
    │   │
    │   ├─ ToolNode::process()
    │   │   ├─ execute each tool call
    │   │   ├─ push Tool result messages
    │   │   └─ state.next_node = "llm"  (loop back)
    │   │
    │   └─ repeat until "__end__"
    │
    └─ update SlidingWindowMemory
    └─ return last assistant message
```

## Error Handling

All errors use `AgentError` (thiserror). Key `From` impls allow `?` to convert:
- `serde_json::Error` → `AgentError::SerdeError`
- `std::io::Error` → `AgentError::IOError`
- `candle_core::Error` → `AgentError::CandleError`

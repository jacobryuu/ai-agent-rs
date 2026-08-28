# ai-agent-rs Documentation

A high-performance, local-first AI agent in Rust. Runs against Ollama, uses a graph-based ReAct orchestrator, has 6 built-in tools, local RAG via candle + LanceDB, and sliding-window conversation memory.

## Contents

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Setup, configuration, and running the CLI |
| [Architecture](architecture.md) | System design, module layout, data flow |
| [Guides](guides/) | How-to guides for extending the agent |
| [Configuration Reference](reference/configuration.md) | All env vars and their defaults |
| [API Reference](reference/api.md) | Key traits, structs, and types |

## Quick Links

- **Source Layout**: `src/` with `core/`, `llm/`, `agent/`, `tools/`, `rag/`, `memory/`
- **Entry Point**: `src/main.rs`
- **Tests**: `cargo test` (47 tests, no external services required)
- **CI**: GitHub Actions — fmt, check, clippy, nextest

## Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core ReAct + Ollama + Basic Tools | ✅ Complete |
| 2 | RAG Integration (LanceDB + candle) | ✅ Complete |
| 2 (roadmap) | Document ingestion pipeline (`index` subcommand) | ✅ Complete |
| 3 | Graph Orchestrator (LangGraph-style) | ✅ Complete |
| 3 (roadmap) | State.context utilization + iteration limit | ✅ Complete |
| Cleanup | Dead code removal (`ReActAgent`) | ✅ Complete |
| 2 (roadmap) | Streaming responses | ✅ Complete |
| 2 (roadmap) | Index deduplication (hash-based) | ✅ Complete |
| 3 (roadmap) | Parallel tool execution (futures::join_all) | ✅ Complete |
| 3 (roadmap) | Long-term memory (SummaryMemory) | ✅ Complete |
| 4 | Cloud providers (OpenAI) | ✅ Complete |
| 5 | Web UI + persistence | ❌ Planned |
| 4 (roadmap) | Anthropic provider | ❌ Planned |
| 3 (roadmap) | Streaming UI integration | ❌ Planned |

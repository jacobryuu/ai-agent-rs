# AI Agent in Rust - 設計書

## 1. 目的と背景

### 目的
Rustの強力な型システムとパフォーマンスを活かし、ローカルLLM（Ollama）をバックエンドとするAIエージェントをゼロから自作する。
さらに、RAG（検索拡張生成）やGraph型オーケストレーション（LangGraph相当）を取り込み、複雑な推論タスクに対応可能にする。

### 背景
- 既存のAIエージェントフレームワーク（LangChain, Vercel AI SDK等）はPython/TypeScriptが主流
- Rustで自作することで、メモリ安全性・低レイテンシ・軽量バイナリを享受
- OllamaをLLMバックエンドとすることで、GPUなしのローカル環境でも動作可能
- RAGの導入により、外部知識（ドキュメント、コードベース）を参照可能にする
- Graph型オーケストレーションにより、状態管理とループを含む複雑なワークフローを実現する

### 採用ライブラリ・技術
| コンポーネント | 採用技術 | 選定理由 |
|-----------|------|---------|
| **LLM Backend** | Ollama | ローカル実行の容易さとAPIの標準性 |
| **Vector DB** | **LanceDB** | Embedded（サーバレス）で動作し、Rustネイティブ |
| **Embeddings** | **candle** | Pure Rustでモデル推論が可能。単一バイナリ化に貢献 |
| **Graph Logic** | 自作（Simple Graph） | LangGraphの概念を参考に、状態遷移とエッジを管理 |

## 2. アーキテクチャ全体像

```
┌──────────────────────────────────────────────────┐
│                    User (CLI)                     │
└────────────────────┬─────────────────────────────┘
                     │ 対話入力 / コマンド
                     ▼
┌──────────────────────────────────────────────────┐
│             Orchestrator (Graph / Loop)           │
│  ┌─────────────────────────────────────────────┐  │
│  │             State Management                │  │
│  │  Nodes: Planner, Agent, RAG, Evaluator      │  │
│  │  Edges: Success, Failure, Retry, Loop       │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
         │                         ▲
         ▼                         │
┌──────────────────────────────────────────────────┐
│                  Agent Core (ReAct)               │
│  1. Prompt Construction (State + Context)       │
│  2. Tool Call Selection                          │
│  3. Execution & Observation                      │
└──────────────────────────────────────────────────┘
         │                         ▲
         ▼                         │
┌──────────────────────┬───────────────────────────┐
│       Tools          │          RAG Layer        │
│  ┌──────────────┐    │   ┌──────────────────┐    │
│  │ Read/Write   │    │   │  Embeddings      │    │
│  │ Bash / Grep  │    │   │  (candle)        │    │
│  └──────────────┘    │   └────────┬─────────┘    │
│  ┌──────────────┐    │            ▼              │
│  │  RAG Tool    │◄───┼───┐ ┌──────────────────┐  │
│  └──────────────┘    │   └─┤  LanceDB (Local) │  │
└──────────────────────┴─────┴──────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│              LLM Provider Abstraction             │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Ollama (Local)   │  │  OpenAI (Future)     │  │
│  └──────────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## 3. 追加コンポーネント設計

### 3.1 `rag/` — RAGレイヤー

#### EmbeddingProvider Trait
```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, AgentError>;
}
```

#### VectorStore Trait
```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add(&self, id: &str, vector: Vec<f32>, metadata: Value) -> Result<(), AgentError>;
    async fn search(&self, vector: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>, AgentError>;
}
```

### 3.2 `graph/` — オーケストレーション（LangGraph向）

#### GraphNode & State
```rust
pub struct State {
    pub messages: Vec<Message>,
    pub context: HashMap<String, Value>,
    pub next_node: Option<String>,
}

#[async_trait]
pub trait Node: Send + Sync {
    async fn process(&self, state: &mut State) -> Result<(), AgentError>;
}
```

#### Graph Runner
- 状態を保持し、エッジの条件に従ってノード間を遷移する。
- 循環参照（ループ）を許容し、終了条件（`__end__`）に到達するまで実行。

## 4. コアモジュール設計 (既存からの変更)

### 3.1 `core/`
- `State` 型の追加（Graphオーケストレーション用）
- `Tool` トレイトに `RAGTool` を追加

### 5. フェーズ計画 (更新)

| フェーズ | 内容 | 目安工数 |
|---------|------|---------|
| Phase1 | 基礎 (ReAct + Ollama + Basic Tools) | 完了 |
| **Phase2** | **RAG Integration (LanceDB + candle)** | 3日 |
| **Phase3** | **Graph Orchestrator (LangGraph相当)** | 3日 |
| Phase4 | Cloud Providers (OpenAI/Anthropic) | 2日 |
| Phase5 | Web UI & Persistence | 4日 |

## 11. リスクと対策 (追加)

| リスク | 確度 | 影響 | 対策 |
|-------|------|------|------|
| LanceDBのRustバインディングの不安定さ | 中 | 中 | バージョンを固定し、抽象化レイヤーでラップする |
| candleのモデル読み込み速度 | 中 | 低 | 初回起動時にキャッシュし、シングルトンで保持する |
| Graphの複雑化によるデバッグ困難 | 高 | 中 | 実行ログ（トレース）を視覚化、または詳細に出力する |

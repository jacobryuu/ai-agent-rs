# AI Agent in Rust - 設計書

## 1. 目的と背景

### 目的
Rustの強力な型システムとパフォーマンスを活かし、ローカルLLM（Ollama）をバックエンドとするAIエージェントをゼロから自作する。

### 背景
- 既存のAIエージェントフレームワーク（LangChain, Vercel AI SDK等）はPython/TypeScriptが主流
- Rustで自作することで、メモリ安全性・低レイテンシ・軽量バイナリを享受
- OllamaをLLMバックエンドとすることで、GPUなしのローカル環境でも動作可能
- 内部構造を完全に理解・制御できる

### 調査済みRust AI OSS
| ライブラリ | 特徴 | 採用判断 |
|-----------|------|---------|
| **rig** | 高機能Rust Agent Framework。Trait設計が洗練 | 参考にするが、学習目的のため今回は採用しない |
| **kalosm** | LLM推論＋埋め込み対応。candleベース | ローカルLLM向けだが、Ollamaバックエンド設計に合わない |
| **llama-cpp-rs** | llama.cppのRustバインディング | Ollamaが内部で使っている。直接扱うとビルドが複雑になるため不採用 |
| **candle** | HuggingFace製のPure Rust ML | Ollama API経由の方が実用的。後方の選択肢 |

## 2. アーキテクチャ全体像

```
┌──────────────────────────────────────────────────┐
│                    User (CLI)                     │
└────────────────────┬─────────────────────────────┘
                     │ 対話入力
                     ▼
┌──────────────────────────────────────────────────┐
│                  Agent Loop                       │
│  ┌─────────────────────────────────────────────┐  │
│  │             ReAct (思考→行動→観察)           │  │
│  │                                              │  │
│  │  1. System Prompt + 会話履歴 + Tools定義     │  │
│  │      ──────────────────────────────→ LLM     │  │
│  │  2. LLM応答にToolCallが含まれるか判定         │  │
│  │  3. 含まれれば → 実行 → 結果を履歴に追加     │  │
│  │  4. LLMに再送 → 2に戻る                     │  │
│  │  5. 最終応答をUserに返す                     │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
         │                         ▲
         ▼                         │
┌──────────────────────────────────────────────────┐
│              LLM Provider Abstraction             │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Ollama (Phase1)  │  │  OpenAI (Phase4)     │  │
│  │  POST /api/chat   │  │  POST /v1/chat/...  │  │
│  │  tools パラメータ  │  │  Streaming 対応      │  │
│  └──────────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│                   Tool Engine                     │
│  ┌──────────┐ ┌───────────┐ ┌─────────────────┐  │
│  │read_file │ │write_file │ │ bash (Phase1.5)  │  │
│  └──────────┘ └───────────┘ └─────────────────┘  │
│  ┌──────────┐ ┌───────────┐ ┌─────────────────┐  │
│  │glob_file │ │grep_file  │ │ web_search(Ph2)  │  │
│  └──────────┘ └───────────┘ └─────────────────┘  │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│           Memory (会話履歴管理)                    │
│  スライディングウィンドウ方式                      │
│  max_messages=20 を超えたら古いものから削除        │
│  将来的には要約＋長期記憶へ拡張                     │
└──────────────────────────────────────────────────┘
```

## 3. コアモジュール設計

### 3.1 `core/` — 共通型・トレイト

```
core/
├── mod.rs        # 再公開
├── message.rs    # Message, Role, ToolCall, ToolResult
├── provider.rs   # LLMProvider trait
└── error.rs      # AgentError
```

#### Message型

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value, // JSON object
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}
```

#### LLMProvider Trait

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// 会話履歴とツール定義を送り、LLM応答を得る
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError>;

    /// モデル名を返す
    fn model_name(&self) -> &str;
}
```

#### ToolDef型

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}
```

### 3.2 `llm/ollama.rs` — Ollama Provider

Ollama `/api/chat` エンドポイントを利用する。

**リクエスト**:

```json
{
  "model": "qwen3:8b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "...",
        "parameters": {...}
      }
    }
  ],
  "stream": false
}
```

**レスポンス**:

```json
{
  "model": "qwen3:8b",
  "message": {
    "role": "assistant",
    "content": "...",
    "tool_calls": [
      {
        "id": "call_xxx",
        "function": {
          "name": "read_file",
          "arguments": "{\"path\": \"...\"}"
        }
      }
    ]
  },
  "done": true
}
```

**設計上の注意**:
- `arguments` はJSON文字列として来るため、`serde_json::from_str` でパースする
- Qwen3:8bはFunction Calling（toolsパラメータ）に対応している
- エラー時はリトライ機構を入れない（ユーザーにエラーを伝える）
- Ollamaサーバは `OLLAMA_HOST` 環境変数で指定可能に

### 3.3 `tools/` — ツールシステム

**Toolトレイト**:

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value; // JSON Schema
    async fn call(&self, args: serde_json::Value) -> Result<String, AgentError>;
}
```

**Phase1で実装する組み込みツール**:

| ツール名 | 説明 | JSON Schema |
|---------|------|------------|
| `read_file` | ファイルを読み込む | `{path: string}` |
| `write_file` | ファイルに書き込む | `{path: string, content: string}` |
| `glob` | ファイルを検索する | `{pattern: string}` |
| `grep` | ファイル内容を検索する | `{pattern: string, path: string}` |
| `bash` | シェルコマンドを実行する | `{command: string}` |

```rust
pub struct ReadFileTool;
pub struct WriteFileTool;
pub struct GlobTool;
pub struct GrepTool;
pub struct BashTool;
```

**ツール実行エンジン**:

```rust
pub struct ToolEngine {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolEngine {
    pub fn new() -> Self { ... }
    pub fn register(&mut self, tool: Arc<dyn Tool>) { ... }
    pub fn get_defs(&self) -> Vec<ToolDef> { ... }
    pub async fn execute(&self, tool_call: &ToolCall) -> Result<ToolResult, AgentError> { ... }
}
```

### 3.4 `memory/` — 会話履歴管理

**SlidingWindowMemory**:
- `max_messages: usize`（デフォルト20）
- `add(message)` → 最大数を超えたら古いSystem以外のメッセージから削除
- `get_history() -> Vec<Message>` → 現在の履歴を返す
- Systemメッセージは常に先頭に残す

```rust
pub struct SlidingWindowMemory {
    messages: VecDeque<Message>,
    max_messages: usize,
}
```

### 3.5 `agent/react.rs` — ReActエージェントループ

```rust
pub struct ReActAgent {
    provider: Box<dyn LLMProvider>,
    tools: ToolEngine,
    memory: SlidingWindowMemory,
    system_prompt: String,
    max_iterations: usize, // デフォルト 10
}

impl ReActAgent {
    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        // 1. ユーザー入力をmemoryに追加
        // 2. ループ開始 (max_iterationsまで)
        //    a. system + memory + tools定義をLLMに送信
        //    b. LLM応答をパース
        //    c. tool_callsがある場合 → 実行 → ToolResultをmemoryに追加 → 次のイテレーション
        //    d. tool_callsがない場合 → contentが最終応答
        // 3. 最終応答を返す
    }
}
```

**ReActループ詳細**:

```
[User Input] → Agent.run()
  │
  ├── memory.add(User(input))
  │
  ├── [Loop i=0..max_iterations]
  │     │
  │     ├── messages = [SystemPrompt] + memory.get_history()
  │     ├── response = provider.chat(messages, tool_defs)
  │     │
  │     ├── if response.tool_calls is Some:
  │     │     memory.add(Assistant(content, tool_calls))
  │     │     for tool_call in tool_calls:
  │     │         result = tool_engine.execute(tool_call).await
  │     │         memory.add(Tool(tool_call_id, result))
  │     │     continue
  │     │
  │     └── else:
  │           memory.add(Assistant(content))
  │           return content
  │
  └── [max_iterations exceeded] → return "Agent stopped due to iteration limit"
```

## 4. エラーハンドリング

```rust
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("LLM request failed: {0}")]
    LLMError(String),

    #[error("Tool execution failed: {0}")]
    ToolError(String),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
}
```

## 5. Phase1 スコープ

### 含む
- Ollama API Client（`/api/chat`、Streamingなし）
- 基本Message型定義（Role, ToolCall, ToolResult）
- Toolトレイト・ToolEngine
- 組み込みツール: `read_file`, `write_file`, `glob`, `grep`, `bash`
- ReAct Agent Loop
- SlidingWindowMemory
- 対話型CLI（対話ループ＋Exit条件）
- 環境変数によるOllama設定（`OLLAMA_HOST`, `OLLAMA_MODEL`）

### 含まない（Phase2以降）
- Streamingレスポンス
- OpenAI / Anthropic Provider
- Web検索ツール
- 長期記憶（永続化・ベクトルDB）
- マルチエージェント・Orchestrator
- Web UI
- ヒストリのファイル保存

## 6. プロジェクト構造

```
src/
├── main.rs              # CLIエントリポイント
├── config.rs            # 環境変数読み込み・設定構造体
├── core/
│   ├── mod.rs
│   ├── message.rs       # Message, Role, ToolCall, ToolResult
│   ├── provider.rs      # LLMProvider trait
│   └── error.rs         # AgentError
├── llm/
│   ├── mod.rs
│   └── ollama.rs        # OllamaProvider
├── tools/
│   ├── mod.rs           # Tool trait, ToolEngine
│   └── builtin/
│       ├── mod.rs
│       ├── read_file.rs
│       ├── write_file.rs
│       ├── glob.rs
│       ├── grep.rs
│       └── bash.rs
├── memory/
│   ├── mod.rs
│   └── sliding.rs       # SlidingWindowMemory
└── agent/
    ├── mod.rs
    └── react.rs          # ReActAgent
```

## 7. 依存関係（Cargo.toml）

```toml
[package]
name = "ai-agent-rs"
version = "0.1.0"
edition = "2024"
license = "MIT"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"
tracing = "0.1"
tracing-subscriber = "0.3"
tracing-appender = "0.2"
thiserror = "2"
```

## 8. Ollamaセットアップ

```bash
# Ollamaインストール
curl -fsSL https://ollama.com/install.sh | sh

# Qwen3:8b モデルダウンロード
ollama pull qwen3:8b

# サーバ起動（デフォルト: localhost:11434）
ollama serve
```

## 9. CLIインターフェース設計

```bash
# 環境変数設定（省略時はデフォルト値）
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen3:8b

# 実行
cargo run
```

対話例:
```
> こんにちは
Agent: こんにちは！何かお手伝いできますか？

> src/main.rs を読んで
Agent: [read_file(path="src/main.rs")]
src/main.rs の内容:
fn main() {
    println!("Hello, world!");
}

> "Hello, Rust Agent!" と出力するよう修正して
Agent: [write_file(path="src/main.rs", content="fn main() {
    println!("Hello, Rust Agent!");
}")]
ファイルを書き込みました。
```

## 10. スケジュール

| フェーズ | 内容 | 目安工数 |
|---------|------|---------|
| Phase1 Step1 | コア型定義（message, provider, error） | 0.5日 |
| Phase1 Step2 | Ollama Provider実装 | 1.5日 |
| Phase1 Step3 | Toolトレイト + 組み込みツール(5つ) | 1.5日 |
| Phase1 Step4 | ReAct Agent Loop実装 | 2日 |
| Phase1 Step5 | SlidingWindowMemory実装 | 0.5日 |
| Phase1 Step6 | CLIインターフェース + config | 1日 |
| Phase1 Step7 | 結合テスト・動作確認 | 1日 |
| **Phase1 計** | | **約8日** |
| Phase2 | Streaming + ツール拡充（web_search等） | 未定 |
| Phase3 | 長期記憶 + 会話の要約 | 未定 |
| Phase4 | OpenAI/Anthropic Provider追加 | 未定 |
| Phase5 | マルチエージェントシステム | 未定 |

## 11. リスクと対策

| リスク | 確度 | 影響 | 対策 |
|-------|------|------|------|
| Qwen3:8bのToolCall精度が低い | 中 | 中 | プロンプトエンジニアリングで補完。非対応ならプロンプトベースのツール呼出にフォールバック |
| Ollama APIのバージョン差異 | 低 | 高 | `0.1.32+` を前提にドキュメントに明記。Dockerfileでバージョン固定も検討 |
| Rustの所有権・非同期の複雑さ | 中 | 中 | `Arc<dyn>` + `Send + Sync` で統一。`async-trait` を活用 |
| ツール実行のセキュリティ（bash等） | 高 | 高 | 実行前に確認プロンプトを入れる。sandbox実行はPhase2以降の課題として明記 |
| ReActループの無限ループ | 中 | 中 | `max_iterations` で制限。デフォルト10回 |
| コンテキストウィンドウ超過 | 中 | 低 | SlidingWindowでメッセージ数制限。将来的には要約で対応 |

## 12. 不明点・要確認事項

- [ ] Qwen3:8bのOllamaでのToolCall（Function Calling）動作確認（未検証）
- [ ] Ollama `/api/chat` の `tools` パラメータが正しくQwen3に渡るか
- [ ] ToolCallの `arguments` がJSON文字列 or JSONオブジェクトのどちらで来るか（Ollamaバージョン依存）
- [ ] System Promptの最適な内容（日本語指示の必要性）
- [ ] `bash` ツールの安全性（ホワイトリスト制限をPhase1で入れるべきか）

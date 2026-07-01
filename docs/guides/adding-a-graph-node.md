# Adding a Graph Node

Graph nodes implement the `Node` trait from `src/agent/graph/mod.rs`.

## Step 1: Define the node

```rust
// src/agent/graph/nodes/validator.rs
use async_trait::async_trait;
use crate::agent::graph::{Node, State};
use crate::core::AgentError;

pub struct ValidatorNode;

#[async_trait]
impl Node for ValidatorNode {
    async fn process(&self, state: &mut State) -> Result<(), AgentError> {
        // Inspect the last assistant message
        if let Some(last) = state.messages.last() {
            if last.content.len() < 10 {
                // Reject — route back to LLM
                state.context.insert("retry_reason".into(), serde_json::json!("too_short"));
                state.next_node = Some("llm".into());
                return Ok(());
            }
        }
        // Accept — continue to end
        state.next_node = Some("__end__".into());
        Ok(())
    }
}
```

## Step 2: Register in the module

```rust
// src/agent/graph/nodes/mod.rs
pub mod react;
pub mod validator;

pub use react::{LLMNode, ToolNode};
pub use validator::ValidatorNode;
```

## Step 3: Wire into GraphRunner

```rust
// In GraphAgent::new():
let validator = Arc::new(ValidatorNode);
runner.add_node("validate", validator);

// Optionally change the entry point:
// let mut runner = GraphRunner::new("llm");  // stays the same
// LLMNode routes to "validate" instead of "__end__" after getting a response
```

## Node Contract

| Field | Purpose |
|-------|---------|
| `state.messages` | Read/write conversation messages |
| `state.context` | Key-value store for cross-node state (e.g., retry counts, flags) |
| `state.next_node` | Set to the next node name, or `"__end__"` to stop; return `None` to stop |

## Built-in Routing Values

| Value | Meaning |
|-------|---------|
| `"__end__"` | Stop graph execution |
| `"llm"` | Route to the LLM node |
| `"tools"` | Route to the tool execution node |

Any custom node name (e.g., `"validate"`, `"planner"`, `"evaluator"`) must be registered with `add_node()` before `run()` is called.

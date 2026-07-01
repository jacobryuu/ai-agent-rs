# Adding a New Tool

Tools implement the `Tool` trait from `src/tools/mod.rs`.

## Step 1: Create the file

```rust
// src/tools/builtin/my_tool.rs
use async_trait::async_trait;
use serde_json::Value;
use crate::core::AgentError;
use crate::tools::Tool;

pub struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str {
        "my_tool"
    }

    fn description(&self) -> &str {
        "What this tool does"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "string",
                    "description": "First argument"
                }
            },
            "required": ["arg1"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let arg1 = args.get("arg1")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'arg1'".into()))?;
        Ok(format!("processed: {}", arg1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_my_tool() {
        let tool = MyTool;
        let args = serde_json::json!({"arg1": "hello"});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result, "processed: hello");
    }
}
```

## Step 2: Register the module

Add to `src/tools/builtin/mod.rs`:

```rust
mod my_tool;
pub use my_tool::MyTool;
```

## Step 3: Register with ToolEngine

Add to `src/main.rs`:

```rust
tool_engine.register(Arc::new(MyTool));
```

Also add the tool's description to the `system_prompt` string so the LLM knows it exists.

## Step 4: Run tests

```bash
cargo test test_my_tool
```

## Requirements

- Tool names must be **unique** and use `snake_case`
- Parameters should follow [JSON Schema](https://json-schema.org/) format
- Keep responses under 10K chars (enforced by `BashTool` pattern; not required but recommended)
- Add tests using `tempfile` if the tool touches the filesystem

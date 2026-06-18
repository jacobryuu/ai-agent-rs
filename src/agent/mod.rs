pub mod graph;
pub mod react;

use self::graph::{
    GraphRunner, State,
    nodes::{LLMNode, ToolNode},
};
use crate::core::{AgentError, LLMProvider, Message, Role};
use crate::memory::SlidingWindowMemory;
use crate::tools::ToolEngine;
use std::sync::Arc;

pub struct GraphAgent {
    runner: GraphRunner,
    memory: SlidingWindowMemory,
}

impl GraphAgent {
    pub fn new(provider: Box<dyn LLMProvider>, tools: ToolEngine, system_prompt: String) -> Self {
        let mut runner = GraphRunner::new("llm");

        let llm_node = Arc::new(LLMNode::new(provider, tools.clone(), system_prompt));
        let tool_node = Arc::new(ToolNode::new(tools));

        runner.add_node("llm", llm_node);
        runner.add_node("tools", tool_node);

        Self { runner, memory: SlidingWindowMemory::new(20) }
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        let mut state = State::new();
        state.messages.extend(self.memory.get_history());
        state.messages.push(Message {
            role: Role::User,
            content: user_input.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });

        let final_state = self.runner.run(state).await?;

        // Update memory with the new messages (except the ones already in history)
        let history_len = self.memory.get_history().len();
        for msg in final_state.messages.iter().skip(history_len) {
            self.memory.add(msg.clone());
        }

        let last_response =
            final_state.messages.last().map(|m| m.content.clone()).unwrap_or_default();

        Ok(last_response)
    }
}

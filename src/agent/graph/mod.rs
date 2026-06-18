use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::AgentError;
pub mod nodes;
pub mod state;
pub use state::State;

#[async_trait]
pub trait Node: Send + Sync {
    async fn process(&self, state: &mut State) -> Result<(), AgentError>;
}

pub struct GraphRunner {
    nodes: HashMap<String, Arc<dyn Node>>,
    entry_point: String,
}

impl GraphRunner {
    pub fn new(entry_point: &str) -> Self {
        Self { nodes: HashMap::new(), entry_point: entry_point.to_string() }
    }

    pub fn add_node(&mut self, name: &str, node: Arc<dyn Node>) {
        self.nodes.insert(name.to_string(), node);
    }

    pub async fn run(&self, mut state: State) -> Result<State, AgentError> {
        let mut current_node_name = self.entry_point.clone();

        loop {
            let node = self.nodes.get(&current_node_name).ok_or_else(|| {
                AgentError::RAGError(format!("Node {} not found in graph", current_node_name))
            })?;

            node.process(&mut state).await?;

            if let Some(next) = state.next_node.take() {
                if next == "__end__" {
                    break;
                }
                current_node_name = next;
            } else {
                // If no next_node is set, we stop (default behavior)
                break;
            }
        }

        Ok(state)
    }
}

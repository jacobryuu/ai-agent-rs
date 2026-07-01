use serde_json::Value;
use std::collections::HashMap;

use crate::core::Message;

pub struct State {
    pub messages: Vec<Message>,
    pub context: HashMap<String, Value>,
    pub next_node: Option<String>,
}

impl State {
    pub fn new() -> Self {
        Self { messages: Vec::new(), context: HashMap::new(), next_node: None }
    }
}

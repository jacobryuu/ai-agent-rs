use std::collections::VecDeque;
use std::sync::Arc;

use tracing::debug;

use crate::core::{LLMProvider, Message, Role};

/// LLM-based conversation memory: summarizes old messages once the window
/// fills, retaining key context as a synthetic System message.
///
/// Not yet wired into `GraphAgent` (which uses `SlidingWindowMemory`); kept as
/// a ready-to-integrate strategy. Remove this attribute when integrating.
#[allow(dead_code)]
pub struct SummaryMemory {
    messages: VecDeque<Message>,
    max_messages: usize,
    summary_threshold: usize,
    provider: Arc<dyn LLMProvider>,
    summary: Option<String>,
}

#[allow(dead_code)]
impl SummaryMemory {
    pub fn new(
        max_messages: usize,
        summary_threshold: usize,
        provider: Arc<dyn LLMProvider>,
    ) -> Self {
        Self { messages: VecDeque::new(), max_messages, summary_threshold, provider, summary: None }
    }

    pub async fn add(&mut self, message: Message) {
        self.messages.push_back(message);
        self.enforce_limit().await;
    }

    pub fn get_history(&self) -> Vec<Message> {
        let mut result = Vec::new();

        if let Some(ref summary) = self.summary {
            result.push(Message {
                role: Role::System,
                content: format!("Previous conversation summary:\n{}", summary),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        result.extend(self.messages.iter().cloned());
        result
    }

    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    async fn enforce_limit(&mut self) {
        while self.messages.len() > self.max_messages {
            if self.messages.len() > self.summary_threshold {
                self.summarize_old_messages().await;
            } else {
                let has_system = self.messages.iter().any(|m| matches!(m.role, Role::System));
                if has_system
                    && self.messages.front().is_some_and(|m| !matches!(m.role, Role::System))
                {
                    self.messages.pop_front();
                } else if let Some(pos) =
                    self.messages.iter().position(|m| !matches!(m.role, Role::System))
                {
                    self.messages.remove(pos);
                } else {
                    break;
                }
            }
        }
    }

    async fn summarize_old_messages(&mut self) {
        let messages_to_summarize: Vec<Message> =
            self.messages.iter().take(self.messages.len() / 2).cloned().collect();

        if messages_to_summarize.is_empty() {
            return;
        }

        let conversation_text = messages_to_summarize
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::System => "System",
                    Role::Tool => "Tool",
                };
                format!("{}: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Summarize the following conversation concisely, preserving key information, decisions, and context:\n\n{}",
            conversation_text
        );

        let summary_messages = vec![Message {
            role: Role::User,
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.provider.chat(&summary_messages, &[]).await {
            Ok(response) => {
                if let Some(new_summary) = response.content {
                    let combined = match &self.summary {
                        Some(existing) => format!("{}\n\n{}", existing, new_summary),
                        None => new_summary,
                    };
                    self.summary = Some(combined);

                    let keep_count = self.messages.len() / 2;
                    let total = self.messages.len();
                    for _ in 0..(total - keep_count) {
                        self.messages.pop_front();
                    }

                    debug!("Summarized {} messages, keeping {}", total - keep_count, keep_count);
                }
            }
            Err(e) => {
                debug!("Failed to summarize messages: {}, falling back to eviction", e);
                let keep_count = self.messages.len() / 2;
                let total = self.messages.len();
                for _ in 0..(total - keep_count) {
                    self.messages.pop_front();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AgentError, LLMResponse, ToolDef};
    use async_trait::async_trait;

    struct MockSummaryProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl LLMProvider for MockSummaryProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(LLMResponse { content: Some("Summary of conversation".into()), tool_calls: None })
        }

        fn model_name(&self) -> &str {
            "mock_summary"
        }
    }

    fn msg(role: Role, content: &str) -> Message {
        Message { role, content: content.into(), tool_calls: None, tool_call_id: None }
    }

    #[tokio::test]
    async fn test_summary_memory_basic() {
        let provider =
            Arc::new(MockSummaryProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut mem = SummaryMemory::new(5, 3, provider);

        mem.add(msg(Role::User, "hello")).await;
        mem.add(msg(Role::Assistant, "hi")).await;

        let history = mem.get_history();
        assert_eq!(history.len(), 2);
        assert!(mem.summary().is_none());
    }

    #[tokio::test]
    async fn test_summary_memory_triggers_summary() {
        let provider =
            Arc::new(MockSummaryProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut mem = SummaryMemory::new(6, 3, provider);

        for i in 0..5 {
            mem.add(msg(Role::User, &format!("question {}", i))).await;
            mem.add(msg(Role::Assistant, &format!("answer {}", i))).await;
        }

        let history = mem.get_history();
        assert!(!history.is_empty());
        assert!(mem.summary().is_some());
        assert!(mem.summary().unwrap().contains("Summary"));
    }
}

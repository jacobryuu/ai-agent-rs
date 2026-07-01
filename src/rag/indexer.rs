use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::fs;
use tracing::{info, warn};

use crate::core::AgentError;
use crate::rag::{EmbeddingProvider, VectorStore};

pub struct Indexer {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_store: Arc<dyn VectorStore>,
    chunk_size: usize,
}

impl Indexer {
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self { embedding_provider, vector_store, chunk_size: 512 }
    }

    pub async fn index_directory(&self, dir_path: &str) -> Result<usize, AgentError> {
        let root = PathBuf::from(dir_path);
        if !root.is_dir() {
            return Err(AgentError::RAGError(format!("Not a directory: {}", dir_path)));
        }

        let mut total_chunks = 0usize;
        let mut total_files = 0usize;
        let mut error_count = 0usize;
        let mut pending = VecDeque::new();
        pending.push_back(root);

        while let Some(dir) = pending.pop_front() {
            let mut entries = match fs::read_dir(&dir).await {
                Ok(e) => e,
                Err(e) => {
                    warn!("Skipping directory '{}': {}", dir.display(), e);
                    error_count += 1;
                    continue;
                }
            };

            while let Some(entry) = entries.next_entry().await? {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    pending.push_back(entry_path);
                    continue;
                }

                let ext =
                    entry_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if !matches!(ext.as_str(), "md" | "txt" | "rs" | "toml" | "json" | "yaml" | "yml") {
                    continue;
                }

                let content = match fs::read_to_string(&entry_path).await {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("Skipping '{}': {}", entry_path.display(), e);
                        error_count += 1;
                        continue;
                    }
                };

                let chunks = Self::chunk_text(&content, self.chunk_size);
                let file_name = entry_path.to_string_lossy().to_string();
                let mut file_chunks = 0usize;

                for (i, chunk) in chunks.iter().enumerate() {
                    let doc_id = format!("{}#chunk{}", file_name, i);
                    match self.embedding_provider.embed(chunk).await {
                        Ok(vector) => {
                            let metadata = serde_json::json!({
                                "source": file_name,
                                "chunk_index": i,
                                "text": chunk,
                            });
                            match self.vector_store.add(&doc_id, vector, metadata).await {
                                Ok(()) => {
                                    total_chunks += 1;
                                    file_chunks += 1;
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to store chunk {} from '{}': {}",
                                        i, file_name, e
                                    );
                                    error_count += 1;
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to embed chunk {} from '{}': {}", i, file_name, e);
                            error_count += 1;
                        }
                    }
                }

                total_files += 1;
                info!(
                    "Indexed {} — {} chunks ({} total, {} errors)",
                    file_name, file_chunks, total_chunks, error_count
                );
            }
        }

        if error_count > 0 {
            warn!(
                "Indexing completed with {} errors. {} chunks indexed from {} files.",
                error_count, total_chunks, total_files
            );
        }

        Ok(total_chunks)
    }

    pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let bytes = text.as_bytes();
        let len = bytes.len();

        while start < len {
            let end = (start + chunk_size).min(len);
            if end < len {
                let search_start = end.saturating_sub(50);
                if let Some(newline_pos) =
                    bytes[search_start..end].iter().rposition(|&b| b == b'\n')
                {
                    let actual_end = search_start + newline_pos + 1;
                    chunks.push(text[start..actual_end].to_string());
                    start = actual_end;
                    continue;
                }
            }
            chunks.push(text[start..end].to_string());
            start = end;
        }

        if chunks.is_empty() && !text.is_empty() {
            chunks.push(text.to_string());
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_small() {
        let chunks = Indexer::chunk_text("hello world", 512);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_text_splits_at_newline() {
        let text = "aaaaa\nbbbbb\nccccc\nddddd\neeeee\nfffff\n";
        let chunks = Indexer::chunk_text(text, 20);
        assert!(chunks.len() >= 2, "should split into multiple chunks: {:?}", chunks);
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = Indexer::chunk_text("", 512);
        assert!(chunks.is_empty());
    }
}

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use async_trait::async_trait;
use futures::StreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Table, connect};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::AgentError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: Value,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    #[allow(dead_code)]
    async fn add(&self, id: &str, vector: Vec<f32>, metadata: Value) -> Result<(), AgentError>;
    async fn search(&self, vector: Vec<f32>, top_k: usize)
    -> Result<Vec<SearchResult>, AgentError>;
}

pub struct LanceVectorStore {
    uri: String,
    table_name: String,
    embedding_dim: usize,
}

impl LanceVectorStore {
    pub async fn new(
        uri: &str,
        table_name: &str,
        embedding_dim: usize,
    ) -> Result<Self, AgentError> {
        let store =
            Self { uri: uri.to_string(), table_name: table_name.to_string(), embedding_dim };
        store.ensure_table().await?;
        Ok(store)
    }

    async fn ensure_table(&self) -> Result<(), AgentError> {
        debug!("Connecting to LanceDB at {}", self.uri);
        let conn = connect(&self.uri).execute().await.map_err(|e| {
            AgentError::VectorStoreError(format!("Failed to connect to LanceDB: {}", e))
        })?;

        let tables =
            conn.table_names().execute().await.map_err(|e| {
                AgentError::VectorStoreError(format!("Failed to list tables: {}", e))
            })?;

        if !tables.contains(&self.table_name) {
            info!("Creating new table '{}' with dimension {}", self.table_name, self.embedding_dim);
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        FieldRef::new(Field::new("item", DataType::Float32, true)),
                        self.embedding_dim as i32,
                    ),
                    false,
                ),
                Field::new("metadata", DataType::Utf8, true),
            ]));

            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

            conn.create_table(&self.table_name, Box::new(reader)).execute().await.map_err(|e| {
                AgentError::VectorStoreError(format!("Failed to create table: {}", e))
            })?;
        }

        Ok(())
    }

    async fn get_table(&self) -> Result<Table, AgentError> {
        let conn = connect(&self.uri).execute().await.map_err(|e| {
            AgentError::VectorStoreError(format!("Failed to connect to LanceDB: {}", e))
        })?;

        conn.open_table(&self.table_name).execute().await.map_err(|e| {
            AgentError::VectorStoreError(format!(
                "Failed to open table '{}': {}",
                self.table_name, e
            ))
        })
    }
}

#[async_trait]
impl VectorStore for LanceVectorStore {
    async fn add(&self, id: &str, vector: Vec<f32>, metadata: Value) -> Result<(), AgentError> {
        debug!("Adding document '{}' to vector store", id);
        let table = self.get_table().await?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    FieldRef::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim as i32,
                ),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        let id_array = StringArray::from(vec![id]);
        let metadata_str = serde_json::to_string(&metadata)?;
        let metadata_array = StringArray::from(vec![Some(metadata_str)]);

        let vector_data = Float32Array::from(vector);
        let vector_array =
            FixedSizeListArray::try_new_from_values(vector_data, self.embedding_dim as i32)
                .map_err(|e| {
                    AgentError::VectorStoreError(format!("Failed to create vector array: {}", e))
                })?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(vector_array), Arc::new(metadata_array)],
        )
        .map_err(|e| {
            AgentError::VectorStoreError(format!("Failed to create record batch: {}", e))
        })?;

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table.add(Box::new(reader)).execute().await.map_err(|e| {
            AgentError::VectorStoreError(format!("Failed to add data to table: {}", e))
        })?;

        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, AgentError> {
        debug!("Searching for top {} nearest neighbors", top_k);
        let table = self.get_table().await?;

        let mut results = table
            .query()
            .nearest_to(vector.as_slice())
            .map_err(|e| {
                AgentError::VectorStoreError(format!("Query initialization failed: {}", e))
            })?
            .limit(top_k)
            .execute()
            .await
            .map_err(|e| AgentError::VectorStoreError(format!("Search execution failed: {}", e)))?;

        let mut final_results = Vec::new();

        while let Some(batch_result) = results.next().await {
            let batch = batch_result.map_err(|e| {
                AgentError::VectorStoreError(format!("Failed to read result batch: {}", e))
            })?;

            let id_col = batch
                .column(batch.schema().column_with_name("id").unwrap().0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let metadata_col = batch
                .column(batch.schema().column_with_name("metadata").unwrap().0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let score_col = batch
                .column(batch.schema().column_with_name("_distance").unwrap().0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let id = id_col.value(i).to_string();
                let metadata_str = metadata_col.value(i);
                let metadata: Value = serde_json::from_str(metadata_str).unwrap_or(Value::Null);
                let score = score_col.value(i);

                final_results.push(SearchResult { id, score, metadata });
            }
        }

        debug!("Search completed with {} results", final_results.len());
        Ok(final_results)
    }
}

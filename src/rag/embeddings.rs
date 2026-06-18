use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
use tracing::debug;

use crate::core::AgentError;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, AgentError>;
    fn dimension(&self) -> usize;
}

pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

impl CandleEmbeddingProvider {
    pub fn new() -> Result<Self, AgentError> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        debug!("Loading embedding model on device: {:?}", device);

        let api = Api::new()
            .map_err(|e| AgentError::RAGError(format!("Failed to create HF API: {}", e)))?;
        let repo = api.repo(Repo::with_revision(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let config_filename = repo
            .get("config.json")
            .map_err(|e| AgentError::RAGError(format!("Failed to get config: {}", e)))?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .map_err(|e| AgentError::RAGError(format!("Failed to get tokenizer: {}", e)))?;
        let weights_filename = repo
            .get("model.safetensors")
            .map_err(|e| AgentError::RAGError(format!("Failed to get weights: {}", e)))?;

        let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| AgentError::RAGError(format!("Failed to load tokenizer: {}", e)))?;

        // Safety: We are mmapping the weights from a local file provided by hf-hub.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_core::DType::F32,
                &device,
            )?
        };
        let model = BertModel::load(vb, &config)?;
        let dimension = config.hidden_size;

        debug!("Embedding model loaded with dimension: {}", dimension);

        Ok(Self { model, tokenizer, device, dimension })
    }
}

#[async_trait]
impl EmbeddingProvider for CandleEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, AgentError> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AgentError::RAGError(format!("Tokenizer error: {}", e)))?;
        let token_ids = tokens.get_ids();
        let token_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.get(0)?;

        // Normalize
        let norm = embeddings.sqr()?.sum_all()?.sqrt()?;
        let embeddings = (embeddings / norm)?;

        Ok(embeddings.to_vec1::<f32>()?)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

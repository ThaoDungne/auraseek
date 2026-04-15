use anyhow::Result;
use crate::core::models::{SearchQuery, SearchResult};

/// Orchestrates a search query across embeddings + filters.
#[allow(async_fn_in_trait)]
pub trait SearchEngine {
    async fn search(&mut self, query: &SearchQuery, source_dir: &str) -> Result<Vec<SearchResult>>;
}

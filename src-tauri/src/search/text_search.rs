/// Text-to-embedding search – SurrealDB edition
use anyhow::Result;
use crate::processor::AuraSeekEngine;
use crate::db::{SurrealDb, DbOperations};

pub fn encode_text_query(
    engine: &mut AuraSeekEngine,
    text: &str,
) -> Result<Vec<f32>> {
    let max_len = 64;
    let (input_ids, attention_mask) = engine.text_proc.encode(text, max_len);
    let seq_len = input_ids.iter().position(|&id| id == 0).unwrap_or(max_len);
    let seq_len = seq_len.max(1);
    engine.aura.encode_text(input_ids, attention_mask, seq_len)
}

pub async fn search_by_text_embedding(
    db: &SurrealDb,
    embedding: &[f32],
    threshold: f32,
    limit: usize,
) -> Result<Vec<(String, f32)>> {
    DbOperations::vector_search(db, embedding, threshold, limit).await
}

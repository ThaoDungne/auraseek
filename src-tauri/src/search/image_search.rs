/// Image-to-embedding search – SurrealDB edition
use anyhow::Result;
use crate::processor::AuraSeekEngine;
use crate::processor::vision::preprocess_aura;
use crate::db::{SurrealDb, DbOperations};

pub fn encode_image_query(
    engine: &mut AuraSeekEngine,
    image_path: &str,
) -> Result<Vec<f32>> {
    let blob = preprocess_aura(image_path)?;
    engine.aura.encode_image(blob, 256, 256)
}

pub async fn search_by_image_embedding(
    db: &SurrealDb,
    embedding: &[f32],
    threshold: f32,
    limit: usize,
) -> Result<Vec<(String, f32)>> {
    DbOperations::vector_search(db, embedding, threshold, limit).await
}

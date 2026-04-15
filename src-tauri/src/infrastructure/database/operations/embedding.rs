use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use surrealdb::types::{RecordId, SurrealValue};
use super::{DbOperations, record_id_to_string};

impl DbOperations {
    pub async fn insert_embedding(
        db: &SurrealDb, media_id: &str, source: &str,
        frame_ts: Option<f64>, frame_idx: Option<u32>, embedding: Vec<f32>,
    ) -> Result<()> {
        let src = source.to_string();
        let query = format!(
            "CREATE embedding SET media_id = {}, source = $src, frame_ts = $fts, frame_idx = $fidx, vec = $vec",
            media_id
        );
        db.db.query(&query)
            .bind(("src", src)).bind(("fts", frame_ts)).bind(("fidx", frame_idx)).bind(("vec", embedding))
            .await?.check().map_err(|e| anyhow::anyhow!("insert_embedding failed: {}", e))?;
        Ok(())
    }

    pub async fn vector_search(
        db: &SurrealDb, query_vec: &[f32], threshold: f32, limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut res = db.db.query(
            "SELECT media_id, vector::similarity::cosine(vec, $qvec) AS score FROM embedding WHERE vector::similarity::cosine(vec, $qvec) >= $thresh ORDER BY score DESC LIMIT $lim"
        ).bind(("qvec", query_vec.to_vec())).bind(("thresh", threshold)).bind(("lim", limit)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Hit { media_id: RecordId, score: f32 }
        let hits: Vec<Hit> = res.take(0)?;
        Ok(hits.into_iter().map(|h| (record_id_to_string(&h.media_id), h.score)).collect())
    }

    pub async fn embedding_count(db: &SurrealDb) -> Result<u64> {
        let mut res = db.db.query("SELECT count() as cnt FROM embedding GROUP ALL").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct C { cnt: u64 }
        let rows: Vec<C> = res.take(0)?;
        Ok(rows.first().map(|r| r.cnt).unwrap_or(0))
    }
}

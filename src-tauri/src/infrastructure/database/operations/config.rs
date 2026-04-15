use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use surrealdb::types::SurrealValue;
use super::DbOperations;

impl DbOperations {
    pub async fn get_source_dir(db: &SurrealDb) -> Result<Option<String>> {
        let mut res = db.db.query("SELECT source_dir FROM config_auraseek:main").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { source_dir: Option<String> }
        let rows: Vec<Row> = res.take(0)?;
        Ok(rows.into_iter().next().and_then(|r| r.source_dir))
    }

    pub async fn set_source_dir(db: &SurrealDb, source_dir: &str) -> Result<()> {
        let dir = source_dir.to_string();
        db.db.query("UPSERT config_auraseek:main SET source_dir = $dir, updated_at = time::now()")
            .bind(("dir", dir)).await?.check()
            .map_err(|e| anyhow::anyhow!("set_source_dir failed: {}", e))?;
        Ok(())
    }
}

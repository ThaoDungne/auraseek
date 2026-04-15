use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::infrastructure::database::models::{MediaRow, MediaDoc, IdOnly, ObjectEntry, FaceEntry};
use crate::core::models::TimelineGroup;
use surrealdb::types::{RecordId, SurrealValue};
use super::{DbOperations, record_id_to_string};

impl DbOperations {
    #[allow(dead_code)]
    pub async fn check_file_status(db: &SurrealDb, sha256: &str) -> Result<Option<(String, bool)>> {
        let sha = sha256.to_string();
        let mut res = db.db.query("SELECT id, processed FROM media WHERE file.sha256 = $sha LIMIT 1")
            .bind(("sha", sha)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct StatusRow { id: RecordId, processed: bool }
        let rows: Vec<StatusRow> = res.take(0)?;
        Ok(rows.first().map(|r| (record_id_to_string(&r.id), r.processed)))
    }

    pub async fn check_exact_file(db: &SurrealDb, name: &str, _sha256: &str) -> Result<Option<(String, bool)>> {
        let mut res = db.db.query("SELECT id, processed FROM media WHERE file.name = $name LIMIT 1")
            .bind(("name", name.to_string())).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct StatusRow { id: RecordId, processed: bool }
        let rows: Vec<StatusRow> = res.take(0)?;
        Ok(rows.first().map(|r| (record_id_to_string(&r.id), r.processed)))
    }

    pub async fn insert_media(db: &SurrealDb, doc: MediaDoc) -> Result<String> {
        let created: Option<IdOnly> = db.db.create("media").content(doc).await?;
        let id = created.ok_or_else(|| anyhow::anyhow!("Failed to create media record"))?.id;
        Ok(record_id_to_string(&id))
    }

    pub async fn update_media_ai(
        db: &SurrealDb, media_id: &str, objects: Vec<ObjectEntry>, faces: Vec<FaceEntry>, thumbnail: Option<String>,
    ) -> Result<()> {
        let objs_json = serde_json::to_string(&objects)?;
        let faces_json = serde_json::to_string(&faces)?;
        let query = if thumbnail.is_some() {
            format!("UPDATE {} SET objects = $objs, faces = $faces, processed = true, thumbnail = $thumb", media_id)
        } else {
            format!("UPDATE {} SET objects = $objs, faces = $faces, processed = true", media_id)
        };
        let mut rq = db.db.query(&query)
            .bind(("objs", serde_json::from_str::<serde_json::Value>(&objs_json)?))
            .bind(("faces", serde_json::from_str::<serde_json::Value>(&faces_json)?));
        if let Some(t) = thumbnail { rq = rq.bind(("thumb", t)); }
        rq.await?.check().map_err(|e| anyhow::anyhow!("update_media_ai failed: {}", e))?;
        Ok(())
    }

    pub async fn toggle_favorite(db: &SurrealDb, media_id: &str) -> Result<bool> {
        let query = format!("UPDATE {} SET favorite = !favorite RETURN AFTER", media_id);
        let mut res = db.db.query(&query).await?.check()
            .map_err(|e| anyhow::anyhow!("toggle_favorite failed: {}", e))?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct FavRow { favorite: bool }
        let rows: Vec<FavRow> = res.take(0)?;
        Ok(rows.first().map(|r| r.favorite).unwrap_or(false))
    }

    pub async fn prune_missing_media(db: &SurrealDb, source_dir: &str) -> Result<usize> {
        let mut res = db.db.query("SELECT id, file.name AS name FROM media").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct IdNameRow { id: RecordId, name: Option<String> }
        let rows: Vec<IdNameRow> = res.take(0)?;
        let mut count = 0;
        let base = std::path::Path::new(source_dir);
        for r in rows {
            if let Some(name) = r.name {
                if !base.join(&name).exists() {
                    let id_str = record_id_to_string(&r.id);
                    crate::log_info!("🗑️ Pruning missing file: {}", name);
                    db.db.query(format!("DELETE embedding WHERE media_id = {}", id_str)).await?.check()?;
                    db.db.query(format!("DELETE {}", id_str)).await?.check()?;
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    pub async fn clear_database(db: &SurrealDb) -> Result<()> {
        db.db.query("DELETE media").await?.check()?;
        db.db.query("DELETE embedding").await?.check()?;
        db.db.query("DELETE person").await?.check()?;
        db.db.query("DELETE search_history").await?.check()?;
        db.db.query("DELETE custom_album").await?.check()?;
        db.db.query("DELETE config_auraseek").await?.check()?;
        Ok(())
    }

    pub async fn get_timeline(db: &SurrealDb, limit: usize, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let mut res = db.db.query(
            "SELECT * FROM media WHERE deleted_at = NONE AND is_hidden = false AND processed = true ORDER BY metadata.created_at DESC LIMIT $lim"
        ).bind(("lim", limit)).await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }

    pub async fn get_distinct_objects(db: &SurrealDb) -> Result<Vec<String>> {
        let mut res = db.db.query(
            "SELECT array::distinct(objects.*.class_name) AS names FROM media WHERE array::len(objects) > 0 AND deleted_at = NONE AND is_hidden = false"
        ).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { names: Vec<String> }
        let rows: Vec<Row> = res.take(0)?;
        let mut all: Vec<String> = rows.into_iter().flat_map(|r| r.names).collect();
        all.sort(); all.dedup();
        Ok(all)
    }
}

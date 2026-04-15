use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::infrastructure::database::models::MediaRow;
use crate::core::models::TimelineGroup;
use surrealdb::types::SurrealValue;
use super::DbOperations;

impl DbOperations {
    pub async fn move_to_trash(db: &SurrealDb, source_dir: &str, media_id: &str) -> Result<()> {
        let mut res = db.db.query(format!("SELECT file.name AS name FROM {}", media_id)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: String }
        let row: Option<NameRow> = res.take(0)?;
        if let Some(r) = row {
            let src_path = std::path::Path::new(source_dir).join(&r.name);
            let trash_dir = std::path::Path::new(source_dir).join(".trash");
            if !trash_dir.exists() { let _ = std::fs::create_dir_all(&trash_dir); }
            let dst_path = trash_dir.join(&r.name);
            if src_path.exists() { let _ = std::fs::rename(&src_path, &dst_path); }
        }
        let query = format!("UPDATE {} SET deleted_at = time::now()", media_id);
        db.db.query(&query).await?.check().map_err(|e| anyhow::anyhow!("move_to_trash failed: {}", e))?;
        Ok(())
    }

    pub async fn restore_from_trash(db: &SurrealDb, source_dir: &str, media_id: &str) -> Result<()> {
        let mut res = db.db.query(format!("SELECT file.name AS name FROM {}", media_id)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: String }
        let row: Option<NameRow> = res.take(0)?;
        if let Some(r) = row {
            let trash_path = std::path::Path::new(source_dir).join(".trash").join(&r.name);
            let dst_path = std::path::Path::new(source_dir).join(&r.name);
            if trash_path.exists() { let _ = std::fs::rename(&trash_path, &dst_path); }
        }
        let query = format!("UPDATE {} SET deleted_at = NONE", media_id);
        db.db.query(&query).await?.check().map_err(|e| anyhow::anyhow!("restore_from_trash failed: {}", e))?;
        Ok(())
    }

    pub async fn get_trash(db: &SurrealDb, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let mut res = db.db.query(
            "SELECT * FROM media WHERE type::is_none(deleted_at) = false ORDER BY deleted_at ASC"
        ).await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }

    pub async fn empty_trash(db: &SurrealDb, source_dir: &str) -> Result<()> {
        let mut res = db.db.query("SELECT file.name AS name FROM media WHERE type::is_none(deleted_at) = false").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: Option<String> }
        let rows: Vec<NameRow> = res.take(0)?;
        for r in rows.into_iter().filter_map(|r| r.name) {
            let path = std::path::Path::new(source_dir).join(".trash").join(&r);
            if path.exists() { let _ = std::fs::remove_file(&path); }
        }
        db.db.query("DELETE media WHERE type::is_none(deleted_at) = false").await?.check()?;
        Ok(())
    }

    pub async fn auto_purge_trash(db: &SurrealDb, source_dir: &str) -> Result<()> {
        let mut res = db.db.query("SELECT file.name AS name FROM media WHERE type::is_none(deleted_at) = false AND deleted_at < time::now() - 30d").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: Option<String> }
        let rows: Vec<NameRow> = res.take(0)?;
        for r in rows.into_iter().filter_map(|r| r.name) {
            let path = std::path::Path::new(source_dir).join(".trash").join(&r);
            if path.exists() { let _ = std::fs::remove_file(&path); }
        }
        db.db.query("DELETE media WHERE type::is_none(deleted_at) = false AND deleted_at < time::now() - 30d").await?.check()?;
        Ok(())
    }

    pub async fn hard_delete_trash_item(db: &SurrealDb, source_dir: &str, media_id: &str) -> Result<()> {
        let query = format!("SELECT file.name AS name FROM media WHERE id = {} AND type::is_none(deleted_at) = false", media_id);
        let mut res = db.db.query(&query).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: Option<String> }
        let rows: Vec<NameRow> = res.take(0)?;
        if let Some(r) = rows.into_iter().next() {
            if let Some(name) = r.name {
                let path = std::path::Path::new(source_dir).join(".trash").join(&name);
                if path.exists() { let _ = std::fs::remove_file(&path); }
            }
        }
        db.db.query(format!("DELETE media WHERE id = {}", media_id)).await?.check()?;
        Ok(())
    }

    pub async fn hide_photo(db: &SurrealDb, media_id: &str) -> Result<()> {
        db.db.query(&format!("UPDATE {} SET is_hidden = true", media_id)).await?.check()
            .map_err(|e| anyhow::anyhow!("hide_photo failed: {}", e))?;
        Ok(())
    }

    pub async fn unhide_photo(db: &SurrealDb, media_id: &str) -> Result<()> {
        db.db.query(&format!("UPDATE {} SET is_hidden = false", media_id)).await?.check()
            .map_err(|e| anyhow::anyhow!("unhide_photo failed: {}", e))?;
        Ok(())
    }

    pub async fn get_hidden_photos(db: &SurrealDb, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let mut res = db.db.query(
            "SELECT * FROM media WHERE is_hidden = true AND deleted_at = NONE ORDER BY metadata.created_at DESC"
        ).await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }
}

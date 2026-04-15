use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::infrastructure::database::models::MediaRow;
use crate::core::models::{TimelineGroup, CustomAlbum};
use surrealdb::types::{RecordId, SurrealValue};
use super::{DbOperations, record_id_to_string};

impl DbOperations {
    pub fn parse_record_id(id_str: &str) -> Result<RecordId> {
        let (tb, id) = id_str.split_once(':')
            .ok_or_else(|| anyhow::anyhow!("Invalid record ID format: {}", id_str))?;
        Ok(RecordId::new(tb, id))
    }

    pub async fn create_album(db: &SurrealDb, title: String) -> Result<String> {
        crate::log_info!("🔨 [DB] Creating album: {}", title);
        let mut res = db.db.query("CREATE custom_album SET title = $title, media_ids = [], created_at = time::now()")
            .bind(("title", title)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct AlbumRow { id: RecordId }
        let rows: Vec<AlbumRow> = res.take(0)?;
        if let Some(r) = rows.into_iter().next() {
            let id = record_id_to_string(&r.id);
            crate::log_info!("✅ [DB] Album created successfully: {}", id);
            Ok(id)
        } else {
            crate::log_error!("❌ [DB] Failed to create album row");
            Err(anyhow::anyhow!("Không tạo được bản ghi album trong cơ sở dữ liệu"))
        }
    }

    pub async fn get_albums(db: &SurrealDb, source_dir: &str) -> Result<Vec<CustomAlbum>> {
        crate::log_info!("🔍 [DB] Fetching all albums...");
        let mut res = db.db.query(
            "SELECT id, title, created_at,
             array::len(media_ids) as count,
             (SELECT VALUE file.name FROM media WHERE id = $parent.media_ids[0] LIMIT 1)[0] as cover_name,
             (SELECT VALUE thumbnail FROM media WHERE id = $parent.media_ids[0] LIMIT 1)[0] as cover_thumb
             FROM custom_album ORDER BY created_at DESC"
        ).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { id: RecordId, title: String, count: Option<usize>, cover_name: Option<String>, cover_thumb: Option<String> }
        let rows: Vec<Row> = res.take(0)?;
        crate::log_info!("📂 [DB] Found {} manual albums", rows.len());
        let base = source_dir.trim_end_matches('/');
        let mut albums = Vec::new();
        for r in rows {
            let cover_url = if let Some(t) = r.cover_thumb {
                Some(if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) })
            } else if let Some(n) = r.cover_name {
                Some(format!("{}/{}", base, n))
            } else { None };
            crate::log_info!("  📷 Album '{}' count={} cover={:?}", r.title, r.count.unwrap_or(0), cover_url);
            albums.push(CustomAlbum { id: record_id_to_string(&r.id), title: r.title, count: r.count.unwrap_or(0) as u32, cover_url });
        }
        Ok(albums)
    }

    pub async fn add_to_album(db: &SurrealDb, album_id: &str, media_ids: Vec<String>) -> Result<()> {
        crate::log_info!("➕ [DB] Adding {} files to album: {}", media_ids.len(), album_id);
        let album_rid = Self::parse_record_id(album_id)?;
        let mut mids = Vec::new();
        for mid in media_ids {
            if let Ok(rid) = Self::parse_record_id(&mid) { mids.push(rid); }
        }
        let mut resp = db.db.query("UPDATE $album SET media_ids = array::distinct(array::add(media_ids, $mids)) RETURN count(media_ids) as total")
            .bind(("album", album_rid)).bind(("mids", mids)).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct CountRow { total: Option<u32> }
        let res: Vec<CountRow> = resp.take(0)?;
        if let Some(r) = res.get(0) { crate::log_info!("📋 [DB] Verified Album now has {} items", r.total.unwrap_or(0)); }
        crate::log_info!("✅ [DB] Successfully updated media_ids array");
        Ok(())
    }

    pub async fn remove_from_album(db: &SurrealDb, album_id: &str, media_ids: Vec<String>) -> Result<()> {
        let album_rid = Self::parse_record_id(album_id)?;
        let mut mids = Vec::new();
        for mid in media_ids { mids.push(Self::parse_record_id(&mid)?); }
        db.db.query("UPDATE $album SET media_ids = array::filter(media_ids, |$id| !$mids CONTAINS $id)")
            .bind(("album", album_rid)).bind(("mids", mids)).await?.check()?;
        Ok(())
    }

    pub async fn delete_album(db: &SurrealDb, album_id: &str) -> Result<()> {
        let album_rid = Self::parse_record_id(album_id)?;
        db.db.query("DELETE $album").bind(("album", album_rid)).await?.check()?;
        Ok(())
    }

    pub async fn get_album_photos(db: &SurrealDb, album_id: &str, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let album_rid = Self::parse_record_id(album_id)?;
        let mut res = db.db.query(
            "SELECT * FROM media WHERE id INSIDE (SELECT VALUE media_ids FROM $album LIMIT 1)[0] AND deleted_at = NONE AND is_hidden = false ORDER BY metadata.created_at DESC"
        ).bind(("album", album_rid)).await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }
}

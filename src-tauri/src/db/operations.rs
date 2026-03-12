/// Database operations – SurrealDB v3 edition
/// All vector search uses SurrealDB's built-in vector::similarity::cosine
use anyhow::Result;
use crate::db::surreal::SurrealDb;
use crate::db::models::*;
use std::collections::HashMap;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

const DUPLICATE_THRESHOLD: f32 = 0.92;

pub fn record_id_to_string(id: &RecordId) -> String {
    let key_str = match &id.key {
        RecordIdKey::String(s) => s.clone(),
        RecordIdKey::Number(n) => n.to_string(),
        _ => "unknown".to_string(),
    };
    format!("{}:{}", id.table, key_str)
}

#[allow(dead_code)]
fn strip_table_prefix(id: &str) -> &str {
    id.find(':').map(|i| &id[i+1..]).unwrap_or(id)
}

/// Build a `SearchResult` from a `MediaRow`, deriving the file path from source_dir.
pub fn row_to_search_result(row: &MediaRow, score: f32, source_dir: &str) -> SearchResult {
    let id_str  = record_id_to_string(&row.id);
    let base    = source_dir.trim_end_matches('/');
    SearchResult {
        media_id:         id_str,
        similarity_score: score,
        file_path:        format!("{}/{}", base, row.file.name),
        media_type:       row.media_type.clone(),
        width:            row.metadata.width,
        height:           row.metadata.height,
        detected_objects: row.objects.iter().map(|o| DetectedObject {
            class_name: o.class_name.clone(),
            conf:       o.conf,
            bbox:       BboxInfo { x: o.bbox.x, y: o.bbox.y, w: o.bbox.w, h: o.bbox.h },
            mask_rle:   o.mask_rle.clone(),
        }).collect(),
        detected_faces: row.faces.iter().map(|f| DetectedFace {
            face_id: f.face_id.clone(),
            name:    f.name.clone(),
            conf:    f.conf,
            bbox:    BboxInfo { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
        }).collect(),
        metadata: SearchResultMeta {
            width:      row.metadata.width,
            height:     row.metadata.height,
            created_at: row.metadata.created_at.as_ref().map(|dt| dt.to_string()),
            objects:    row.objects.iter().map(|o| o.class_name.clone()).collect(),
            faces:      row.faces.iter().filter_map(|f| f.name.clone()).collect(),
        },
        thumbnail_path: row.thumbnail.clone(),
    }
}

pub struct DbOperations;

impl DbOperations {
    // ─── Media CRUD ──────────────────────────────────────────────────

    /// Check duplicate by SHA-256 and return (media_id, processed_status)
    pub async fn check_file_status(db: &SurrealDb, sha256: &str) -> Result<Option<(String, bool)>> {
        let sha = sha256.to_string();
        let mut res = db.db.query(
            "SELECT id, processed FROM media WHERE file.sha256 = $sha LIMIT 1"
        )
        .bind(("sha", sha))
        .await?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct StatusRow { id: RecordId, processed: bool }

        let rows: Vec<StatusRow> = res.take(0)?;
        if let Some(row) = rows.first() {
            Ok(Some((record_id_to_string(&row.id), row.processed)))
        } else {
            Ok(None)
        }
    }

    /// Check if a file with the same name already exists in the database.
    /// We dedupe per filename so that:
    /// - mỗi file vật lý (1 đường dẫn) chỉ được ingest 1 lần
    /// - nhưng bạn vẫn có thể copy thành tên khác và được coi là media mới
    pub async fn check_exact_file(db: &SurrealDb, name: &str, sha256: &str) -> Result<Option<(String, bool)>> {
        let mut res = db.db.query(
            "SELECT id, processed FROM media WHERE file.name = $name LIMIT 1"
        )
        .bind(("name", name.to_string()))
        .await?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct StatusRow { id: RecordId, processed: bool }

        let rows: Vec<StatusRow> = res.take(0)?;
        if let Some(row) = rows.first() {
            Ok(Some((record_id_to_string(&row.id), row.processed)))
        } else {
            Ok(None)
        }
    }

    /// Insert a new media document, returns the record id as string
    pub async fn insert_media(db: &SurrealDb, doc: MediaDoc) -> Result<String> {
        let created: Option<IdOnly> = db.db
            .create("media")
            .content(doc)
            .await?;
        let id = created
            .ok_or_else(|| anyhow::anyhow!("Failed to create media record"))?
            .id;
        Ok(record_id_to_string(&id))
    }

    /// Update AI results on a media record
    pub async fn update_media_ai(
        db: &SurrealDb,
        media_id: &str,
        objects: Vec<ObjectEntry>,
        faces: Vec<FaceEntry>,
        thumbnail: Option<String>,
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
            
        if let Some(t) = thumbnail {
            rq = rq.bind(("thumb", t));
        }
            
        rq.await?
        .check()
        .map_err(|e| anyhow::anyhow!("update_media_ai failed: {}", e))?;
        Ok(())
    }

    // ─── Embeddings (vector stored in SurrealDB) ─────────────────────

    /// Insert embedding vector
    pub async fn insert_embedding(
        db: &SurrealDb,
        media_id: &str,
        source: &str,
        frame_ts: Option<f64>,
        frame_idx: Option<u32>,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let src = source.to_string();
        let query = format!(
            "CREATE embedding SET
                media_id = {},
                source   = $src,
                frame_ts = $fts,
                frame_idx = $fidx,
                vec      = $vec",
            media_id
        );
        db.db.query(&query)
        .bind(("src", src))
        .bind(("fts", frame_ts))
        .bind(("fidx", frame_idx))
        .bind(("vec", embedding))
        .await?
        .check()
        .map_err(|e| anyhow::anyhow!("insert_embedding failed: {}", e))?;
        Ok(())
    }

    /// Vector search using cosine similarity (SurrealDB built-in)
    pub async fn vector_search(
        db: &SurrealDb,
        query_vec: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut res = db.db.query(
            "SELECT
                media_id,
                vector::similarity::cosine(vec, $qvec) AS score
            FROM embedding
            WHERE vector::similarity::cosine(vec, $qvec) >= $thresh
            ORDER BY score DESC
            LIMIT $lim"
        )
        .bind(("qvec", query_vec.to_vec()))
        .bind(("thresh", threshold))
        .bind(("lim", limit))
        .await?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct Hit {
            media_id: RecordId,
            score: f32,
        }
        let hits: Vec<Hit> = res.take(0)?;
        Ok(hits.into_iter().map(|h| (record_id_to_string(&h.media_id), h.score)).collect())
    }

    /// Get embedding count
    pub async fn embedding_count(db: &SurrealDb) -> Result<u64> {
        let mut res = db.db.query("SELECT count() as cnt FROM embedding GROUP ALL").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct C { cnt: u64 }
        let rows: Vec<C> = res.take(0)?;
        Ok(rows.first().map(|r| r.cnt).unwrap_or(0))
    }

    // ─── Person / Face ───────────────────────────────────────────────

    /// Upsert a person (face cluster)
    pub async fn upsert_person(db: &SurrealDb, person: PersonDoc) -> Result<()> {
        let fid = person.face_id.clone();
        let name = person.name.clone();
        let thumb = person.thumbnail.clone();
        let conf = person.conf;
        let bbox = person.face_bbox.clone();
        db.db.query(
            "INSERT INTO person { face_id: $fid, name: $name, thumbnail: $thumb, conf: $conf, face_bbox: $bbox }
             ON DUPLICATE KEY UPDATE
                name = $input.name ?? name,
                conf = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.conf ELSE conf END,
                thumbnail = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.thumbnail ELSE thumbnail END,
                face_bbox = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.face_bbox ELSE face_bbox END"
        )
        .bind(("fid", fid))
        .bind(("name", name))
        .bind(("thumb", thumb))
        .bind(("conf", conf))
        .bind(("bbox", bbox))
        .await?
        .check()
        .map_err(|e| anyhow::anyhow!("upsert_person failed: {}", e))?;
        Ok(())
    }

    /// Name a face cluster
    pub async fn name_person(db: &SurrealDb, face_id: &str, name: &str) -> Result<()> {
        let fid = face_id.to_string();
        let n = name.to_string();
        // Update person table
        db.db.query("UPDATE person SET name = $name WHERE face_id = $fid")
            .bind(("name", n.clone()))
            .bind(("fid", fid.clone()))
            .await?;
        // Update face entries embedded in media docs
        db.db.query(
            "UPDATE media SET faces = faces.map(|$f| IF $f.face_id = $fid THEN $f.{*, name: $name} ELSE $f END) WHERE faces.*.face_id CONTAINS $fid"
        )
        .bind(("fid", fid))
        .bind(("name", n))
        .await?;
        Ok(())
    }

    // ─── Favorites ─────────────────────────────────────────────────────

    pub async fn toggle_favorite(db: &SurrealDb, media_id: &str) -> Result<bool> {
        let query = format!(
            "UPDATE {} SET favorite = !favorite RETURN AFTER",
            media_id
        );
        let mut res = db.db.query(&query).await?
            .check()
            .map_err(|e| anyhow::anyhow!("toggle_favorite failed: {}", e))?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct FavRow { favorite: bool }
        let rows: Vec<FavRow> = res.take(0)?;
        Ok(rows.first().map(|r| r.favorite).unwrap_or(false))
    }

    // ─── Config (source_dir) ─────────────────────────────────────────

    /// Get the configured source directory (always stored as `config_auraseek:main`).
    pub async fn get_source_dir(db: &SurrealDb) -> Result<Option<String>> {
        let mut res = db.db.query("SELECT source_dir FROM config_auraseek:main").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { source_dir: Option<String> }
        let rows: Vec<Row> = res.take(0)?;
        Ok(rows.into_iter().next().and_then(|r| r.source_dir))
    }

    /// Upsert the source directory into a single config record `config_auraseek:main`.
    pub async fn set_source_dir(db: &SurrealDb, source_dir: &str) -> Result<()> {
        let dir = source_dir.to_string();
        db.db.query(
            "UPSERT config_auraseek:main SET source_dir = $dir, updated_at = time::now()"
        )
        .bind(("dir", dir))
        .await?
        .check()
        .map_err(|e| anyhow::anyhow!("set_source_dir failed: {}", e))?;
        Ok(())
    }

    /// Delete all media, embeddings, and persons for a fresh start.
    pub async fn clear_database(db: &SurrealDb) -> Result<()> {
        db.db.query("DELETE media").await?.check()?;
        db.db.query("DELETE embedding").await?.check()?;
        db.db.query("DELETE person").await?.check()?;
        db.db.query("DELETE search_history").await?.check()?;
        Ok(())
    }

    /// Prune any media records whose files no longer exist on disk.
    pub async fn prune_missing_media(db: &SurrealDb, source_dir: &str) -> Result<usize> {
        let mut res = db.db.query("SELECT id, file.name AS name FROM media").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct IdNameRow { id: RecordId, name: Option<String> }
        let rows: Vec<IdNameRow> = res.take(0)?;
        
        let mut count = 0;
        let base = std::path::Path::new(source_dir);
        
        for r in rows {
            if let Some(name) = r.name {
                let path = base.join(&name);
                if !path.exists() {
                    let id_str = record_id_to_string(&r.id);
                    crate::log_info!("🗑️ Pruning missing file: {}", name);
                    
                    // Delete embedding first
                    db.db.query(format!("DELETE embedding WHERE media_id = {}", id_str)).await?.check()?;
                    // Delete the media record
                    db.db.query(format!("DELETE {}", id_str)).await?.check()?;
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    // ─── Trash & Hidden ──────────────────────────────────────────────
    
    pub async fn move_to_trash(db: &SurrealDb, media_id: &str) -> Result<()> {
        let query = format!("UPDATE {} SET deleted_at = time::now()", media_id);
        db.db.query(&query).await?.check()
            .map_err(|e| anyhow::anyhow!("move_to_trash failed: {}", e))?;
        Ok(())
    }

    pub async fn restore_from_trash(db: &SurrealDb, media_id: &str) -> Result<()> {
        let query = format!("UPDATE {} SET deleted_at = NONE", media_id);
        db.db.query(&query).await?.check()
            .map_err(|e| anyhow::anyhow!("restore_from_trash failed: {}", e))?;
        Ok(())
    }

    pub async fn get_trash(db: &SurrealDb, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let mut res = db.db.query(
            "SELECT * FROM media WHERE type::is_none(deleted_at) = false ORDER BY deleted_at ASC"
        ).await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }

    pub async fn empty_trash(db: &SurrealDb) -> Result<()> {
        // Fetch paths first to delete from disk
        let mut res = db.db.query("SELECT file.path FROM media WHERE type::is_none(deleted_at) = false").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct PathRow { path: Option<String> }
        let rows: Vec<PathRow> = res.take(0)?;
        for r in rows.into_iter().filter_map(|r| r.path) {
            let _ = std::fs::remove_file(&r); // best effort
        }
        db.db.query("DELETE media WHERE type::is_none(deleted_at) = false").await?.check()?;
        Ok(())
    }
    
    pub async fn auto_purge_trash(db: &SurrealDb) -> Result<()> {
        let mut res = db.db.query("SELECT file.path FROM media WHERE type::is_none(deleted_at) = false AND deleted_at < time::now() - 30d").await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct PathRow { path: Option<String> }
        let rows: Vec<PathRow> = res.take(0)?;
        for r in rows.into_iter().filter_map(|r| r.path) {
            let _ = std::fs::remove_file(&r); // best effort
        }
        db.db.query("DELETE media WHERE type::is_none(deleted_at) = false AND deleted_at < time::now() - 30d").await?.check()?;
        Ok(())
    }

    pub async fn hide_photo(db: &SurrealDb, media_id: &str) -> Result<()> {
        let query = format!("UPDATE {} SET is_hidden = true", media_id);
        db.db.query(&query).await?.check()
            .map_err(|e| anyhow::anyhow!("hide_photo failed: {}", e))?;
        Ok(())
    }

    pub async fn unhide_photo(db: &SurrealDb, media_id: &str) -> Result<()> {
        let query = format!("UPDATE {} SET is_hidden = false", media_id);
        db.db.query(&query).await?.check()
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

    // ─── Timeline ────────────────────────────────────────────────────

    pub async fn get_timeline(db: &SurrealDb, limit: usize, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        // Chỉ hiển thị những media đã xử lý xong pipeline (processed = true)
        let mut res = db.db.query(
            "SELECT * FROM media WHERE deleted_at = NONE AND is_hidden = false AND processed = true ORDER BY metadata.created_at DESC LIMIT $lim"
        )
        .bind(("lim", limit))
        .await?;
        let rows: Vec<MediaRow> = res.take(0)?;
        Self::group_rows_into_timeline(rows, source_dir)
    }

    fn group_rows_into_timeline(rows: Vec<MediaRow>, source_dir: &str) -> Result<Vec<TimelineGroup>> {

        let mut groups: HashMap<(i32, u32), TimelineGroup> = HashMap::new();

        for row in rows {
            let (year, month) = parse_ym(&row.metadata.created_at);
            let label = format_month_label(year, month);
            let file_path = format!("{}/{}", source_dir.trim_end_matches('/'), row.file.name);
            let thumbnail_path = row.thumbnail.clone();
            let item = TimelineItem {
                media_id:   record_id_to_string(&row.id),
                file_path,
                media_type: row.media_type.clone(),
                width:      row.metadata.width,
                height:     row.metadata.height,
                created_at: row.metadata.created_at.as_ref().map(|dt| dt.to_string()),
                objects:    row.objects.iter().map(|o| o.class_name.clone()).collect(),
                faces:      row.faces.iter().filter_map(|f| f.name.clone()).collect(),
                face_ids:   row.faces.iter().map(|f| f.face_id.clone()).collect(),
                favorite:   row.favorite,
                deleted_at: row.deleted_at.as_ref().map(|dt| dt.to_string()),
                is_hidden:  row.is_hidden,
                thumbnail_path,
                detected_objects: row.objects.iter().map(|o| DetectedObject {
                    class_name: o.class_name.clone(),
                    conf: o.conf,
                    bbox: BboxInfo { x: o.bbox.x, y: o.bbox.y, w: o.bbox.w, h: o.bbox.h },
                    mask_rle: o.mask_rle.clone(),
                }).collect(),
                detected_faces: row.faces.iter().map(|f| DetectedFace {
                    face_id: f.face_id.clone(),
                    name: f.name.clone(),
                    conf: f.conf,
                    bbox: BboxInfo { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
                }).collect(),
            };
            groups.entry((year, month)).or_insert_with(|| TimelineGroup {
                label, year, month, day: None, items: vec![],
            }).items.push(item);
        }

        let mut result: Vec<TimelineGroup> = groups.into_values().collect();
        result.sort_by(|a, b| b.year.cmp(&a.year).then(b.month.cmp(&a.month)));
        Ok(result)
    }

    // ─── Search result resolution ────────────────────────────────────

    pub async fn resolve_search_results(
        db: &SurrealDb,
        hits: Vec<(String, f32)>,
        source_dir: &str,
    ) -> Result<Vec<SearchResult>> {
        if hits.is_empty() { return Ok(vec![]); }

        let mut score_map: HashMap<String, f32> = HashMap::new();
        for (mid, score) in &hits {
            let raw = mid.strip_prefix("media:").unwrap_or(mid);
            let entry = score_map.entry(raw.to_string()).or_insert(0.0);
            if *score > *entry { *entry = *score; }
        }

        let ids: Vec<String> = score_map.keys().cloned().collect();
        let ids_str = ids.iter().map(|id| format!("media:{}", id)).collect::<Vec<_>>().join(", ");
        let query = format!("SELECT * FROM media WHERE id IN [{}] AND deleted_at = NONE AND is_hidden = false", ids_str);

        let mut res = db.db.query(&query).await?;
        let rows: Vec<MediaRow> = res.take(0)?;

        let mut results: Vec<SearchResult> = rows.into_iter().filter_map(|row| {
            let id_str = record_id_to_string(&row.id);
            let raw    = id_str.strip_prefix("media:").unwrap_or(&id_str);
            let score  = *score_map.get(raw)?;
            Some(row_to_search_result(&row, score, source_dir))
        }).collect();

        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Apply post-search filters
    pub async fn apply_filters(
        _db: &SurrealDb,
        mut results: Vec<SearchResult>,
        object: Option<&str>,
        face: Option<&str>,
        month: Option<u32>,
        year: Option<i32>,
        media_type: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if let Some(obj) = object {
            results.retain(|r| r.metadata.objects.iter().any(|o| o.to_lowercase().contains(&obj.to_lowercase())));
        }
        if let Some(f) = face {
            results.retain(|r| r.metadata.faces.iter().any(|n| n.to_lowercase().contains(&f.to_lowercase())));
        }
        if let Some(t) = media_type {
            results.retain(|r| r.media_type == t);
        }
        if month.is_some() || year.is_some() {
            results.retain(|r| {
                if let Some(ref dt_str) = r.metadata.created_at {
                    // Try multiple date formats — SurrealDB Datetime can output various formats
                    let parsed_year_month = parse_year_month_from_str(dt_str);
                    if let Some((y, m)) = parsed_year_month {
                        if let Some(fy) = year {
                            if y != fy { return false; }
                        }
                        if let Some(fm) = month {
                            if m != fm { return false; }
                        }
                        return true;
                    }
                }
                false
            });
        }
        Ok(results)
    }

    // ─── Helpers ──────────────────────────────────────────────────────
}

/// Extract (year, month) from a date string — handles all common formats
fn parse_year_month_from_str(s: &str) -> Option<(i32, u32)> {
    use chrono::Datelike;
    // RFC3339: "2026-03-05T12:45:09+00:00" or "2026-03-05T12:45:09Z"
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some((dt.year(), dt.month()));
    }
    // ISO without tz offset: "2026-03-05T12:45:09"
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some((dt.year(), dt.month()));
    }
    // With fractional seconds: "2026-03-05T12:45:09.123"
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some((dt.year(), dt.month()));
    }
    // Space-separated: "2026-03-05 12:45:09"
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some((dt.year(), dt.month()));
    }
    // Just a date: "2026-03-05"
    if let Ok(dt) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some((dt.year(), dt.month()));
    }
    // Last resort: extract YYYY-MM from the string directly
    if s.len() >= 7 {
        let parts: Vec<&str> = s.split(|c: char| c == '-' || c == '/').collect();
        if parts.len() >= 2 {
            if let (Ok(y), Ok(m)) = (parts[0].parse::<i32>(), parts[1].parse::<u32>()) {
                if (1900..=2100).contains(&y) && (1..=12).contains(&m) {
                    return Some((y, m));
                }
            }
        }
    }
    None
}

impl DbOperations {
    // ─── Distinct objects (for filter panel) ───────────────────────────

    pub async fn get_distinct_objects(db: &SurrealDb) -> Result<Vec<String>> {
        let mut res = db.db.query(
            "SELECT array::distinct(objects.*.class_name) AS names FROM media WHERE array::len(objects) > 0 AND deleted_at = NONE AND is_hidden = false"
        ).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct Row { names: Vec<String> }
        let rows: Vec<Row> = res.take(0)?;
        let mut all: Vec<String> = rows.into_iter().flat_map(|r| r.names).collect();
        all.sort();
        all.dedup();
        Ok(all)
    }

    // ─── People ──────────────────────────────────────────────────────

    pub async fn get_people(db: &SurrealDb, source_dir: &str) -> Result<Vec<PersonGroup>> {
        let mut res = db.db.query(
            "SELECT
                face_id,
                name,
                thumbnail,
                conf,
                face_bbox,
                (SELECT count() FROM media WHERE faces.*.face_id CONTAINS $parent.face_id AND deleted_at = NONE AND is_hidden = false GROUP ALL)[0].count AS photo_count,
                (SELECT file.name FROM media WHERE faces.*.face_id CONTAINS $parent.face_id AND deleted_at = NONE AND is_hidden = false LIMIT 1)[0].file.name AS cover_name
            FROM person
            ORDER BY photo_count DESC"
        ).await?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct PRow {
            face_id: String,
            name: Option<String>,
            thumbnail: Option<String>,
            conf: Option<f32>,
            face_bbox: Option<crate::db::models::Bbox>,
            photo_count: Option<u64>,
            cover_name: Option<String>,
        }
        let rows: Vec<PRow> = res.take(0)?;
        let base = source_dir.trim_end_matches('/');
        Ok(rows.into_iter().map(|r| {
            let cover_path = r.cover_name.as_ref().map(|n| format!("{}/{}", base, n));
            // thumbnail may be stored as full path (legacy) or just filename; try to normalise
            let thumbnail = r.thumbnail.map(|t| {
                if std::path::Path::new(&t).is_absolute() { t }
                else { format!("{}/{}", base, t) }
            });
            PersonGroup {
                face_id: r.face_id,
                name: r.name,
                photo_count: r.photo_count.unwrap_or(0) as u32,
                cover_path,
                thumbnail,
                conf: r.conf,
                face_bbox: r.face_bbox.map(|b| crate::db::models::BboxInfo { x: b.x, y: b.y, w: b.w, h: b.h }),
            }
        }).collect())
    }

    // ─── Duplicates ──────────────────────────────────────────────────

    pub async fn get_duplicates(
        db: &SurrealDb,
        source_dir: &str,
        media_type: Option<&str>,
        thumb_cache_dir: Option<&std::path::Path>,
    ) -> Result<Vec<DuplicateGroup>> {
        let mut groups: Vec<DuplicateGroup> = vec![];
        let base = source_dir.trim_end_matches('/');
        let type_filter = match media_type {
            Some(t) => format!(" AND media_type = '{}'", t),
            None => String::new(),
        };

        // ── helpers ──────────────────────────────────────────────────────────
        fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
            let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
            for (x, y) in a.iter().zip(b.iter()) { dot += x*y; na += x*x; nb += y*y; }
            if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
        }
        fn hamming(a: u64, b: u64) -> u32 { (a ^ b).count_ones() }
        fn resolve_video_thumb_fallback(
            base_dir: &str,
            file_name: &str,
            thumb_cache_dir: Option<&std::path::Path>,
        ) -> Option<String> {
            let video_abs = std::path::Path::new(base_dir).join(file_name);
            let stem = video_abs.file_stem()?.to_string_lossy();
            let thumb_file = format!("{}.thumb.jpg", stem);

            // Prefer cache dir if file exists there
            if let Some(cache_dir) = thumb_cache_dir {
                let p = cache_dir.join(&thumb_file);
                if p.exists() {
                    return Some(p.to_string_lossy().to_string());
                }
            }

            // Legacy: next to the video
            let parent = video_abs.parent().unwrap_or(std::path::Path::new(base_dir));
            let p = parent.join(&thumb_file);
            if p.exists() {
                return Some(p.to_string_lossy().to_string());
            }
            None
        }

        // Track which media-id sets are already covered so later stages skip them
        let mut covered: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ── 1. Exact SHA-256 duplicates ───────────────────────────────────────
        // SurrealDB 3.x không hỗ trợ HAVING như SQL truyền thống, nên sử dụng
        // subquery rồi lọc ở ngoài.
        let query = format!(
            "SELECT sha256, ids FROM (
                 SELECT file.sha256 AS sha256, array::group(id) AS ids, count() AS cnt
                 FROM media
                 WHERE deleted_at = NONE AND is_hidden = false{}
                 GROUP BY file.sha256
             ) WHERE cnt > 1",
            type_filter
        );
        let mut res = db.db.query(&query).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct DupRow { sha256: String, ids: Vec<RecordId> }
        let dup_rows: Vec<DupRow> = res.take(0)?;

        for row in dup_rows {
            let ids_str = row.ids.iter().map(|id| record_id_to_string(id)).collect::<Vec<_>>().join(", ");
            let q = format!("SELECT id, file.name AS name, file.size AS size, thumbnail FROM media WHERE id IN [{}]", ids_str);
            let mut r2 = db.db.query(&q).await?;
            #[derive(serde::Deserialize, SurrealValue)]
            struct DI { id: RecordId, name: Option<String>, size: Option<u64>, thumbnail: Option<String> }
            let items: Vec<DI> = r2.take(0)?;
            if items.len() < 2 { continue; }

            let key: Vec<String> = items.iter().map(|i| record_id_to_string(&i.id)).collect();
            for k in &key { covered.insert(k.clone()); }
            groups.push(DuplicateGroup {
                group_id: row.sha256.clone(),
                reason: "Trùng Hash — giống nhau 100%".into(),
                items: items.into_iter().map(|i| {
                    let name = i.name.clone();
                    let file_path = name.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default();
                    let thumbnail_path = i.thumbnail
                        .map(|t| {
                            if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) }
                        })
                        .or_else(|| {
                            if media_type != Some("video") { return None; }
                            let n = name.as_deref()?;
                            resolve_video_thumb_fallback(base, n, thumb_cache_dir)
                        });
                    DuplicateItem {
                        media_id: record_id_to_string(&i.id),
                        file_path,
                        size: i.size.unwrap_or(0),
                        thumbnail_path,
                    }
                }).collect(),
            });
        }

        // ── 2. pHash near-duplicate (Hamming ≤ 8) ─────────────────────────────
        let phash_q = format!(
            "SELECT id, file.name AS name, file.size AS size, file.phash AS phash, thumbnail
             FROM media
             WHERE deleted_at = NONE AND is_hidden = false AND file.phash IS NOT NONE{}",
            type_filter
        );
        let mut pr = db.db.query(&phash_q).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct PRow { id: RecordId, name: Option<String>, size: Option<u64>, phash: Option<String>, thumbnail: Option<String> }
        let prows: Vec<PRow> = pr.take(0)?;

        // parse hex u64 pHash
        let phash_items: Vec<(RecordId, Option<String>, u64, u64, Option<String>)> = prows.into_iter()
            .filter_map(|r| {
                let h = u64::from_str_radix(r.phash.as_deref()?, 16).ok()?;
                Some((r.id, r.name, r.size.unwrap_or(0), h, r.thumbnail))
            })
            .collect();

        // DSU on phash
        let n = phash_items.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, i: usize) -> usize {
            if p[i] == i { i } else { let r = find(p, p[i]); p[i] = r; r }
        }
        for i in 0..n {
            for j in (i+1)..n {
                if hamming(phash_items[i].3, phash_items[j].3) <= 8 {
                    let ri = find(&mut parent, i);
                    let rj = find(&mut parent, j);
                    if ri != rj { parent[ri] = rj; }
                }
            }
        }
        let mut ph_clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { ph_clusters.entry(find(&mut parent, i)).or_default().push(i); }
        for (_root, idxs) in ph_clusters {
            if idxs.len() < 2 { continue; }
            let ids: Vec<String> = idxs.iter().map(|&i| record_id_to_string(&phash_items[i].0)).collect();
            if ids.iter().all(|id| covered.contains(id)) { continue; }
            for id in &ids { covered.insert(id.clone()); }
            groups.push(DuplicateGroup {
                group_id: format!("phash_{}", ids[0]),
                reason: "Ảnh gần giống nhau (pHash — cùng nội dung nhưng khác kích thước/nén)".into(),
                items: idxs.iter().map(|&i| DuplicateItem {
                    media_id: record_id_to_string(&phash_items[i].0),
                    file_path: phash_items[i].1.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default(),
                    size: phash_items[i].2,
                    thumbnail_path: phash_items[i].4.as_ref()
                        .map(|t| {
                            if std::path::Path::new(t).is_absolute() { t.clone() } else { format!("{}/{}", base, t) }
                        })
                        .or_else(|| {
                            if media_type != Some("video") { return None; }
                            let name = phash_items[i].1.as_deref()?;
                            resolve_video_thumb_fallback(base, name, thumb_cache_dir)
                        }),
                }).collect(),
            });
        }

        // ── 3. Vision vector similarity ≥ 0.97 ────────────────────────────────
        // For videos we only use the embedding of the earliest extracted frame.
        let emb_src_filter = match media_type {
            Some("video") => " AND source = 'video_frame'",
            _             => " AND source = 'image'",
        };
        // Fetch one representative embedding per media_id
        let emb_q = format!(
            "SELECT media_id, vec, source, frame_idx
             FROM embedding
             WHERE media_id.deleted_at = NONE AND media_id.is_hidden = false{}{}",
            type_filter.replace("media_type", "media_id.media_type"),
            emb_src_filter
        );
        let mut er = db.db.query(&emb_q).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct ERow { media_id: RecordId, vec: Vec<f32>, frame_idx: Option<u32> }
        let all_embs: Vec<ERow> = er.take(0)?;

        // For videos: keep only the earliest frame per media_id
        let mut rep_embs: std::collections::HashMap<String, (&ERow, u32)> = std::collections::HashMap::new();
        for e in &all_embs {
            let mid = record_id_to_string(&e.media_id);
            let fi = e.frame_idx.unwrap_or(u32::MAX);
            rep_embs.entry(mid)
                .and_modify(|(prev, prev_fi)| { if fi < *prev_fi { *prev = e; *prev_fi = fi; } })
                .or_insert((e, fi));
        }
        let rep: Vec<(&ERow, String)> = rep_embs.into_iter()
            .map(|(mid, (e, _))| (e, mid))
            .collect();

        if rep.len() > 1 {
            use rayon::prelude::*;
            let pairs: Vec<(usize, usize)> = (0..rep.len()).into_par_iter().flat_map(|i| {
                let mut local = vec![];
                for j in (i+1)..rep.len() {
                    if rep[i].0.media_id == rep[j].0.media_id { continue; }
                    if cosine_sim(&rep[i].0.vec, &rep[j].0.vec) >= DUPLICATE_THRESHOLD {
                        local.push((i, j));
                    }
                }
                local
            }).collect();

            if !pairs.is_empty() {
                let mut par: Vec<usize> = (0..rep.len()).collect();
                for (i, j) in pairs {
                    let ri = find(&mut par, i); let rj = find(&mut par, j);
                    if ri != rj { par[ri] = rj; }
                }
                let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
                for i in 0..rep.len() { clusters.entry(find(&mut par, i)).or_default().push(i); }

                let mut ci = 0usize;
                for (_root, idxs) in clusters {
                    if idxs.len() < 2 { continue; }
                    let ids: Vec<String> = idxs.iter().map(|&i| rep[i].1.clone()).collect();
                    if ids.iter().all(|id| covered.contains(id)) { continue; }
                    for id in &ids { covered.insert(id.clone()); }

                    let ids_str = ids.iter().map(|id| {
                        if id.contains(':') { id.clone() } else { format!("media:{}", id) }
                    }).collect::<Vec<_>>().join(", ");
                    let q = format!("SELECT id, file.name AS name, file.size AS size, thumbnail FROM media WHERE id IN [{}]", ids_str);
                    let mut r3 = db.db.query(&q).await?;
                    #[derive(serde::Deserialize, SurrealValue)]
                    struct DI { id: RecordId, name: Option<String>, size: Option<u64>, thumbnail: Option<String> }
                    let items: Vec<DI> = r3.take(0)?;
                    if items.len() < 2 { continue; }

                    let reason = match media_type {
                        Some("video") => "Video gần giống nhau (AI phát hiện đoạn đầu tương tự ≥ 97%)",
                        _             => "Ảnh gần giống nhau (AI phát hiện ≥ 97%)",
                    };
                    groups.push(DuplicateGroup {
                        group_id: format!("vec_{}_{}", ci, record_id_to_string(&items[0].id)),
                        reason: reason.into(),
                        items: items.into_iter().map(|i| {
                            let name = i.name.clone();
                            let file_path = name.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default();
                            let thumbnail_path = i.thumbnail
                                .map(|t| {
                                    if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) }
                                })
                                .or_else(|| {
                                    if media_type != Some("video") { return None; }
                                    let n = name.as_deref()?;
                                    resolve_video_thumb_fallback(base, n, thumb_cache_dir)
                                });
                            DuplicateItem {
                                media_id: record_id_to_string(&i.id),
                                file_path,
                                size: i.size.unwrap_or(0),
                                thumbnail_path,
                            }
                        }).collect(),
                    });
                    ci += 1;
                }
            }
        }

        Ok(groups)
    }

    // ─── Search History ──────────────────────────────────────────────

    pub async fn save_search_history(
        db: &SurrealDb,
        query: Option<String>,
        image_path: Option<String>,
        filters: Option<SearchFilters>,
    ) -> Result<()> {
        let _: Option<IdOnly> = db.db
            .create("search_history")
            .content(SearchHistoryDoc { query, image_path, filters })
            .await?;
        Ok(())
    }

    pub async fn get_search_history(db: &SurrealDb, limit: usize) -> Result<Vec<SearchHistoryRow>> {
        let mut res = db.db.query(
            "SELECT * FROM search_history ORDER BY created_at DESC LIMIT $lim"
        )
        .bind(("lim", limit))
        .await?;
        Ok(res.take(0)?)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

fn parse_ym(dt: &Option<surrealdb::types::Datetime>) -> (i32, u32) {
    if let Some(dt_val) = dt {
        use chrono::Datelike;
        return (dt_val.year(), dt_val.month() as u32);
    }
    (1970, 1)
}

fn format_month_label(year: i32, month: u32) -> String {
    let months = ["Tháng 1", "Tháng 2", "Tháng 3", "Tháng 4", "Tháng 5", "Tháng 6",
                  "Tháng 7", "Tháng 8", "Tháng 9", "Tháng 10", "Tháng 11", "Tháng 12"];
    let m = months.get((month.saturating_sub(1)) as usize).unwrap_or(&"");
    format!("{} {}", m, year)
}

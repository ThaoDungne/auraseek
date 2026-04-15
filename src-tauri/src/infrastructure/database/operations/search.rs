use anyhow::Result;
use std::collections::HashMap;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::infrastructure::database::models::{MediaRow, SearchFilters, SearchHistoryDoc, SearchHistoryRow, IdOnly};
use crate::core::models::SearchResult;
use super::{DbOperations, record_id_to_string, row_to_search_result, parse_year_month_from_str};

impl DbOperations {
    pub async fn resolve_search_results(
        db: &SurrealDb, hits: Vec<(String, f32)>, source_dir: &str,
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
            let raw = id_str.strip_prefix("media:").unwrap_or(&id_str);
            let score = *score_map.get(raw)?;
            Some(row_to_search_result(&row, score, source_dir))
        }).collect();
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    pub async fn apply_filters(
        _db: &SurrealDb, mut results: Vec<SearchResult>,
        object: Option<&str>, face: Option<&str>,
        month: Option<u32>, year: Option<i32>, media_type: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if let Some(obj) = object {
            results.retain(|r| r.metadata.objects.iter().any(|o| o.to_lowercase().contains(&obj.to_lowercase())));
        }
        if let Some(f) = face {
            results.retain(|r| r.metadata.faces.iter().any(|n| n.to_lowercase().contains(&f.to_lowercase())));
        }
        if let Some(t) = media_type { results.retain(|r| r.media_type == t); }
        if month.is_some() || year.is_some() {
            results.retain(|r| {
                if let Some(ref dt_str) = r.metadata.created_at {
                    if let Some((y, m)) = parse_year_month_from_str(dt_str) {
                        if let Some(fy) = year { if y != fy { return false; } }
                        if let Some(fm) = month { if m != fm { return false; } }
                        return true;
                    }
                }
                false
            });
        }
        Ok(results)
    }

    pub async fn save_search_history(
        db: &SurrealDb, query: Option<String>, image_path: Option<String>, filters: Option<SearchFilters>,
    ) -> Result<()> {
        let _: Option<IdOnly> = db.db.create("search_history")
            .content(SearchHistoryDoc { query, image_path, filters }).await?;
        Ok(())
    }

    pub async fn get_search_history(db: &SurrealDb, limit: usize) -> Result<Vec<SearchHistoryRow>> {
        let mut res = db.db.query("SELECT * FROM search_history ORDER BY created_at DESC LIMIT $lim")
            .bind(("lim", limit)).await?;
        Ok(res.take(0)?)
    }
}

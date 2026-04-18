/// Search pipeline orchestration – SurrealDB edition
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::processor::AuraSeekEngine;
use crate::db::SurrealDb;
use crate::db::operations::{DbOperations, row_to_search_result};
use crate::db::models::SearchResult;
use crate::search::text_search::{encode_text_query, search_by_text_embedding};
use crate::search::image_search::{encode_image_query, search_by_image_embedding};

const DEFAULT_THRESHOLD: f32 = 0.6;
const DEFAULT_LIMIT: usize = 10000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    Text,
    Image,
    Combined,
    ObjectFilter,
    FaceFilter,
    FilterOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchQueryFilters {
    pub object:     Option<String>,
    pub face:       Option<String>,
    pub month:      Option<u32>,
    pub year:       Option<i32>,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub mode:       SearchMode,
    pub text:       Option<String>,
    pub image_path: Option<String>,
    pub filters:    SearchQueryFilters,
}

pub struct SearchPipeline;

impl SearchPipeline {
    pub async fn run(
        query: &SearchQuery,
        engine: &mut AuraSeekEngine,
        db: &SurrealDb,
        source_dir: &str,
    ) -> Result<Vec<SearchResult>> {
        let raw_hits = match query.mode {
            SearchMode::Text => {
                let text = query.text.as_deref().unwrap_or("");
                crate::log_info!("🔍 [SearchPipeline::run] mode=Text text='{}'", text);
                let embedding = encode_text_query(engine, text)?;
                search_by_text_embedding(db, &embedding, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?
            }

            SearchMode::Image => {
                let path = query.image_path.as_deref().unwrap_or("");
                crate::log_info!("🔍 [SearchPipeline::run] mode=Image path='{}' threshold={}", path, DEFAULT_THRESHOLD);
                let embedding = encode_image_query(engine, path)?;
                search_by_image_embedding(db, &embedding, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?
            }

            SearchMode::Combined => {
                let text = query.text.as_deref().unwrap_or("");
                let path = query.image_path.as_deref().unwrap_or("");
                crate::log_info!("🔍 [SearchPipeline::run] mode=Combined text='{}' path='{}' threshold={}", text, path, DEFAULT_THRESHOLD);

                let text_emb  = encode_text_query(engine, text)?;
                let img_emb   = encode_image_query(engine, path)?;
                let text_hits = search_by_text_embedding(db, &text_emb, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?;
                let img_hits  = search_by_image_embedding(db, &img_emb, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?;

                crate::log_info!(
                    "🔍 [SearchPipeline::run] combined text_hits={} img_hits={}",
                    text_hits.len(),
                    img_hits.len()
                );

                let text_map: HashMap<String, f32> = text_hits.into_iter().collect();
                let mut combined = vec![];
                for (mid, img_score) in img_hits {
                    if let Some(text_score) = text_map.get(&mid) {
                        combined.push((mid, (img_score + text_score) / 2.0));
                    }
                }
                crate::log_info!(
                    "🔍 [SearchPipeline::run] combined_intersection={}",
                    combined.len()
                );
                combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                combined
            }

            SearchMode::ObjectFilter => {
                let class = query.filters.object.clone().unwrap_or_default();
                crate::log_info!("🔍 ObjectFilter: class_name='{}'", class);
                let mut res = db.db.query(
                    "SELECT * FROM media WHERE objects.*.class_name CONTAINS $cls AND deleted_at = NONE AND is_hidden = false ORDER BY metadata.created_at DESC LIMIT 100"
                )
                .bind(("cls", class))
                .await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.iter()
                    .map(|row| row_to_search_result(row, 1.0, source_dir))
                    .collect();
                return DbOperations::apply_filters(
                    db, results, None, None,
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }

            SearchMode::FaceFilter => {
                let name = query.filters.face.clone().unwrap_or_default();
                crate::log_info!("🔍 FaceFilter: name='{}'", name);
                let mut res = db.db.query(
                    "SELECT * FROM media WHERE faces.*.name CONTAINS $name AND deleted_at = NONE AND is_hidden = false ORDER BY metadata.created_at DESC LIMIT 100"
                )
                .bind(("name", name))
                .await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.iter()
                    .map(|row| row_to_search_result(row, 1.0, source_dir))
                    .collect();
                return DbOperations::apply_filters(
                    db, results, None, None,
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }

            SearchMode::FilterOnly => {
                crate::log_info!("🔍 FilterOnly: {:?}", query.filters);
                let mut res = db.db.query(
                    "SELECT * FROM media WHERE deleted_at = NONE AND is_hidden = false ORDER BY metadata.created_at DESC LIMIT 200"
                ).await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.iter()
                    .map(|row| row_to_search_result(row, 1.0, source_dir))
                    .collect();
                return DbOperations::apply_filters(
                    db, results,
                    query.filters.object.as_deref(),
                    query.filters.face.as_deref(),
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }
        };

        let mut results = DbOperations::resolve_search_results(db, raw_hits, source_dir).await?;

        results = DbOperations::apply_filters(
            db, results,
            query.filters.object.as_deref(),
            query.filters.face.as_deref(),
            query.filters.month,
            query.filters.year,
            query.filters.media_type.as_deref(),
        ).await?;

        Ok(results)
    }
}

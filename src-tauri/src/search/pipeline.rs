/// Search pipeline orchestration – SurrealDB edition
/// Runs different search modes and returns unified SearchResult
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::processor::AuraSeekEngine;
use crate::db::SurrealDb;
use crate::db::operations::{DbOperations, record_id_to_string};
use crate::db::models::SearchResult;
use crate::search::text_search::{encode_text_query, search_by_text_embedding};
use crate::search::image_search::{encode_image_query, search_by_image_embedding};

const DEFAULT_THRESHOLD: f32 = 0.15;
const DEFAULT_LIMIT: usize = 100;

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
    /// Run the search pipeline (no more in-memory vector store needed).
    pub async fn run(
        query: &SearchQuery,
        engine: &mut AuraSeekEngine,
        db: &SurrealDb,
    ) -> Result<Vec<SearchResult>> {
        let raw_hits = match query.mode {
            SearchMode::Text => {
                let text = query.text.as_deref().unwrap_or("");
                let embedding = encode_text_query(engine, text)?;
                search_by_text_embedding(db, &embedding, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?
            }

            SearchMode::Image => {
                let path = query.image_path.as_deref().unwrap_or("");
                let embedding = encode_image_query(engine, path)?;
                search_by_image_embedding(db, &embedding, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?
            }

            SearchMode::Combined => {
                let text = query.text.as_deref().unwrap_or("");
                let path = query.image_path.as_deref().unwrap_or("");

                let text_emb = encode_text_query(engine, text)?;
                let img_emb  = encode_image_query(engine, path)?;

                let text_hits = search_by_text_embedding(db, &text_emb, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?;
                let img_hits  = search_by_image_embedding(db, &img_emb, DEFAULT_THRESHOLD, DEFAULT_LIMIT).await?;

                // Intersect and average scores
                let text_map: HashMap<String, f32> = text_hits.into_iter().collect();
                let mut combined = vec![];
                for (mid, img_score) in img_hits {
                    if let Some(text_score) = text_map.get(&mid) {
                        combined.push((mid, (img_score + text_score) / 2.0));
                    }
                }
                combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                combined
            }

            SearchMode::ObjectFilter => {
                let class = query.filters.object.clone().unwrap_or_default();
                let mut res = db.db.query(
                    "SELECT * FROM media WHERE objects.*.class_name CONTAINS $cls ORDER BY metadata.created_at DESC LIMIT 100"
                )
                .bind(("cls", class))
                .await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.into_iter().map(|row| SearchResult {
                    media_id: record_id_to_string(&row.id),
                    similarity_score: 1.0,
                    file_path: row.file.path,
                    media_type: row.media_type,
                    metadata: crate::db::models::SearchResultMeta {
                        width: row.metadata.width,
                        height: row.metadata.height,
                        created_at: row.metadata.created_at.as_ref().map(|dt| dt.to_string()),
                        objects: row.objects.iter().map(|o| o.class_name.clone()).collect(),
                        faces: row.faces.iter().filter_map(|f| f.name.clone()).collect(),
                    },
                }).collect();
                // Still apply remaining filters (year, month, media_type)
                return DbOperations::apply_filters(
                    db, results, None, None,
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }

            SearchMode::FaceFilter => {
                let name = query.filters.face.clone().unwrap_or_default();
                let mut res = db.db.query(
                    "SELECT * FROM media WHERE faces.*.name CONTAINS $name ORDER BY metadata.created_at DESC LIMIT 100"
                )
                .bind(("name", name))
                .await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.into_iter().map(|row| SearchResult {
                    media_id: record_id_to_string(&row.id),
                    similarity_score: 1.0,
                    file_path: row.file.path,
                    media_type: row.media_type,
                    metadata: crate::db::models::SearchResultMeta {
                        width: row.metadata.width,
                        height: row.metadata.height,
                        created_at: row.metadata.created_at.as_ref().map(|dt| dt.to_string()),
                        objects: row.objects.iter().map(|o| o.class_name.clone()).collect(),
                        faces: row.faces.iter().filter_map(|f| f.name.clone()).collect(),
                    },
                }).collect();
                return DbOperations::apply_filters(
                    db, results, None, None,
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }

            SearchMode::FilterOnly => {
                let mut res = db.db.query(
                    "SELECT * FROM media ORDER BY metadata.created_at DESC LIMIT 200"
                ).await?;
                let rows: Vec<crate::db::models::MediaRow> = res.take(0)?;
                let results: Vec<SearchResult> = rows.into_iter().map(|row| SearchResult {
                    media_id: record_id_to_string(&row.id),
                    similarity_score: 1.0,
                    file_path: row.file.path,
                    media_type: row.media_type,
                    metadata: crate::db::models::SearchResultMeta {
                        width: row.metadata.width,
                        height: row.metadata.height,
                        created_at: row.metadata.created_at.as_ref().map(|dt| dt.to_string()),
                        objects: row.objects.iter().map(|o| o.class_name.clone()).collect(),
                        faces: row.faces.iter().filter_map(|f| f.name.clone()).collect(),
                    },
                }).collect();
                return DbOperations::apply_filters(
                    db, results,
                    query.filters.object.as_deref(),
                    query.filters.face.as_deref(),
                    query.filters.month, query.filters.year,
                    query.filters.media_type.as_deref(),
                ).await;
            }
        };

        // Resolve vector hits to full SearchResult
        let mut results = DbOperations::resolve_search_results(db, raw_hits).await?;

        // Apply post-filters
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

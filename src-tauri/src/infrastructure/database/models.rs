/// Database models – SurrealDB v3 edition
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue, Datetime as SurrealDatetime};

// ────────────────────── Core document types ──────────────────────

/// App-level config stored in `config_auraseek` table (singleton record).
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct AppConfig {
    pub source_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct FileInfo {
    /// Filename only (no directory). Full path = config_auraseek.source_dir + "/" + name
    pub name:   String,
    pub size:   u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phash:  Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MediaMetadata {
    pub width:       Option<u32>,
    pub height:      Option<u32>,
    pub duration:    Option<f64>,
    pub fps:         Option<f64>,
    pub created_at:  Option<SurrealDatetime>,
    pub modified_at: Option<SurrealDatetime>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// RLE format: each [offset, length] means pixels [offset..offset+length) are 1 (row-major index).
/// Decode with image width/height: total_pixels = w * h.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct ObjectEntry {
    pub class_name: String,
    pub conf:       f32,
    pub bbox:       Bbox,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_area:  Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path:  Option<String>,
    /// Run-length encoded mask: array of [offset, length] for 1-pixels. Use media width/height to decode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_rle:   Option<Vec<[u32; 2]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct FaceEntry {
    pub face_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name:    Option<String>,
    pub conf:    f32,
    pub bbox:    Bbox,
}

/// Document stored in `media` table (for .content()).
/// `source` is NOT stored here — it lives in `config_auraseek`.
/// Full file path is derived as: source_dir + "/" + file.name
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MediaDoc {
    pub media_type: String,
    pub file:       FileInfo,
    pub metadata:   MediaMetadata,
    pub objects:    Vec<ObjectEntry>,
    pub faces:      Vec<FaceEntry>,
    pub processed:  bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail:  Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<SurrealDatetime>,
    #[serde(default)]
    pub is_hidden:  bool,
}

/// Row returned from SurrealDB with an `id` field (for .take())
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MediaRow {
    pub id:         RecordId,
    pub media_type: String,
    pub file:       FileInfo,
    pub metadata:   MediaMetadata,
    pub objects:    Vec<ObjectEntry>,
    pub faces:      Vec<FaceEntry>,
    pub processed:  bool,
    #[serde(default)]
    pub favorite:   bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail:  Option<String>,
    pub deleted_at: Option<SurrealDatetime>,
    #[serde(default)]
    pub is_hidden:  bool,
}

/// Embedding document (vector stored in SurrealDB)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct EmbeddingDoc {
    pub media_id:  RecordId,
    pub source:    String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_ts:  Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_idx: Option<u32>,
    pub vec:       Vec<f32>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct EmbeddingRow {
    pub id:        RecordId,
    pub media_id:  RecordId,
    pub source:    String,
    pub vec:       Vec<f32>,
}

/// Person / face cluster
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct PersonDoc {
    pub face_id:   String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name:      Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conf:      Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_bbox: Option<Bbox>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct PersonRow {
    pub id:        RecordId,
    pub face_id:   String,
    pub name:      Option<String>,
    pub thumbnail: Option<String>,
    pub conf:      Option<f32>,
    pub face_bbox: Option<Bbox>,
}

/// Search history
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SearchHistoryDoc {
    pub query:      Option<String>,
    pub image_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters:    Option<SearchFilters>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SearchHistoryRow {
    pub id:         RecordId,
    pub query:      Option<String>,
    pub image_path: Option<String>,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SearchFilters {
    pub object:     Option<String>,
    pub face:       Option<String>,
    pub month:      Option<u32>,
    pub year:       Option<i32>,
    pub media_type: Option<String>,
}

// API response types live in crate::core::models.
// Re-export for backward compatibility with existing `use crate::infrastructure::database::models::*` paths.
#[allow(unused_imports)]
pub use crate::core::models::{
    BboxInfo, DetectedObject, DetectedFace, SearchResult, SearchResultMeta,
    TimelineGroup, TimelineItem, PersonGroup, DuplicateGroup, DuplicateItem, CustomAlbum,
};

/// Generic record ID helper (for .take())
#[derive(Debug, Deserialize, SurrealValue)]
pub struct IdOnly {
    pub id: RecordId,
}

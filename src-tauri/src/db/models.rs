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

// ────────────────────── API response types ──────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BboxInfo {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub class_name: String,
    pub conf:       f32,
    pub bbox:       BboxInfo,
    /// RLE mask: each [offset, length] for pixels set to 1 (row-major, width × height grid).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_rle:   Option<Vec<[u32; 2]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedFace {
    pub face_id: String,
    pub name:    Option<String>,
    pub conf:    f32,
    pub bbox:    BboxInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub media_id:          String,
    pub similarity_score:  f32,
    pub file_path:         String,
    pub media_type:        String,
    pub metadata:          SearchResultMeta,
    /// Full detection data (bbox + mask_rle) for hover overlays
    pub detected_objects:  Vec<DetectedObject>,
    pub detected_faces:    Vec<DetectedFace>,
    pub width:             Option<u32>,
    pub height:            Option<u32>,
    pub thumbnail_path:    Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMeta {
    pub width:      Option<u32>,
    pub height:     Option<u32>,
    pub created_at: Option<String>,
    pub objects:    Vec<String>,
    pub faces:      Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineGroup {
    pub label:  String,
    pub year:   i32,
    pub month:  u32,
    pub day:    Option<u32>,
    pub items:  Vec<TimelineItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineItem {
    pub media_id:          String,
    /// Full absolute path derived from source_dir + "/" + file.name at query time
    pub file_path:         String,
    pub media_type:        String,
    pub width:             Option<u32>,
    pub height:            Option<u32>,
    pub created_at:        Option<String>,
    pub objects:           Vec<String>,
    pub faces:             Vec<String>,
    pub face_ids:          Vec<String>,
    pub favorite:          bool,
    pub deleted_at:        Option<String>,
    pub is_hidden:         bool,
    pub detected_objects:  Vec<DetectedObject>,
    pub detected_faces:    Vec<DetectedFace>,
    /// Absolute path to video thumbnail JPEG (None for images)
    pub thumbnail_path:    Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonGroup {
    pub face_id:     String,
    pub name:        Option<String>,
    pub photo_count: u32,
    pub cover_path:  Option<String>,
    pub thumbnail:   Option<String>,
    pub conf:        Option<f32>,
    pub face_bbox:   Option<BboxInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub group_id: String,
    pub reason:   String,
    pub items:    Vec<DuplicateItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateItem {
    pub media_id:  String,
    pub file_path: String,
    pub size:      u64,
    pub thumbnail_path: Option<String>,
}

/// Generic record ID helper (for .take())
#[derive(Debug, Deserialize, SurrealValue)]
pub struct IdOnly {
    pub id: RecordId,
}

use anyhow::Result;
use crate::core::models::{
    SearchResult, TimelineGroup, PersonGroup, DuplicateGroup, CustomAlbum,
};

/// Media document persistence (CRUD + queries).
#[allow(async_fn_in_trait)]
pub trait MediaRepository {
    async fn check_exact_file(&self, name: &str, sha256: &str) -> Result<Option<(String, bool)>>;
    async fn insert_media(&self, id: &str, doc: &serde_json::Value) -> Result<String>;
    async fn update_media_ai(&self, media_id: &str, objects: serde_json::Value, faces: serde_json::Value, thumbnail: Option<String>) -> Result<()>;
    async fn toggle_favorite(&self, media_id: &str) -> Result<bool>;
    async fn prune_missing_media(&self, source_dir: &str) -> Result<usize>;
    async fn embedding_count(&self) -> Result<usize>;
    async fn get_timeline(&self, limit: usize, source_dir: &str) -> Result<Vec<TimelineGroup>>;
    async fn get_duplicates(&self, source_dir: &str, media_type: Option<&str>, thumb_cache: Option<&std::path::Path>) -> Result<Vec<DuplicateGroup>>;
}

/// Embedding vector persistence + similarity search.
#[allow(async_fn_in_trait)]
pub trait EmbeddingRepository {
    async fn insert_embedding(&self, media_id: &str, source: &str, frame_ts: Option<f64>, frame_idx: Option<u32>, vec: Vec<f32>) -> Result<()>;
    async fn search_by_vector(&self, vec: &[f32], threshold: f32, limit: usize) -> Result<Vec<(String, f32)>>;
}

/// Person / face-cluster persistence.
#[allow(async_fn_in_trait)]
pub trait PersonRepository {
    async fn upsert_person(&self, doc: &serde_json::Value) -> Result<()>;
    async fn get_people(&self, source_dir: &str) -> Result<Vec<PersonGroup>>;
    async fn name_person(&self, face_id: &str, name: &str) -> Result<()>;
    async fn merge_people(&self, target: &str, source: &str) -> Result<()>;
    async fn delete_person(&self, face_id: &str) -> Result<()>;
}

/// Search history persistence.
#[allow(async_fn_in_trait)]
pub trait SearchHistoryRepository {
    async fn save_search_history(&self, text: Option<String>, image_path: Option<String>, filters: Option<serde_json::Value>) -> Result<()>;
    async fn get_search_history(&self, limit: usize) -> Result<Vec<serde_json::Value>>;
}

/// Album persistence.
#[allow(async_fn_in_trait)]
pub trait AlbumRepository {
    async fn create_album(&self, title: String) -> Result<String>;
    async fn get_albums(&self, source_dir: &str) -> Result<Vec<CustomAlbum>>;
    async fn add_to_album(&self, album_id: &str, media_ids: Vec<String>) -> Result<()>;
    async fn remove_from_album(&self, album_id: &str, media_ids: Vec<String>) -> Result<()>;
    async fn delete_album(&self, album_id: &str) -> Result<()>;
    async fn get_album_photos(&self, album_id: &str, source_dir: &str) -> Result<Vec<TimelineGroup>>;
}

/// App configuration persistence.
#[allow(async_fn_in_trait)]
pub trait ConfigRepository {
    async fn get_source_dir(&self) -> Result<Option<String>>;
    async fn set_source_dir(&self, dir: &str) -> Result<()>;
}

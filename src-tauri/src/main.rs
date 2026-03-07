mod utils;
mod model;
mod processor;
mod db;
mod ingest;
mod search;
mod debug_cli;

use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;
use tauri::State;

use processor::AuraSeekEngine;
use db::{SurrealDb, DbOperations};
use db::models::{SearchResult, TimelineGroup, PersonGroup, DuplicateGroup};
use ingest::image_ingest::IngestSummary;
use search::pipeline::{SearchPipeline, SearchQuery};
use utils::logger::Logger;

// ─── App State ───────────────────────────────────────────────────────────────

pub struct AppState {
    pub engine:     Arc<Mutex<Option<AuraSeekEngine>>>,
    pub db:         Arc<Mutex<Option<SurrealDb>>>,
    /// SurrealDB address, e.g. "127.0.0.1:8000"
    pub surreal_addr: Mutex<String>,
    pub surreal_user: Mutex<String>,
    pub surreal_pass: Mutex<String>,
}

impl AppState {
    fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            db: Arc::new(Mutex::new(None)),
            surreal_addr: Mutex::new("127.0.0.1:8000".to_string()),
            surreal_user: Mutex::new("root".to_string()),
            surreal_pass: Mutex::new("root".to_string()),
        }
    }
}

// ─── Tauri Commands ──────────────────────────────────────────────────────────

/// Initialize AI engine and SurrealDB connection.
#[tauri::command]
async fn cmd_init(state: State<'_, AppState>) -> Result<String, String> {
    // Init engine
    {
        let mut engine_guard = state.engine.lock().await;
        if engine_guard.is_none() {
            match AuraSeekEngine::new_default() {
                Ok(e) => { *engine_guard = Some(e); }
                Err(e) => return Err(format!("Engine init failed: {}", e)),
            }
        }
    }

    // Init DB
    {
        let addr = state.surreal_addr.lock().await.clone();
        let user = state.surreal_user.lock().await.clone();
        let pass = state.surreal_pass.lock().await.clone();
        let mut db_guard = state.db.lock().await;
        if db_guard.is_none() {
            match SurrealDb::connect(&addr, &user, &pass).await {
                Ok(sdb) => {
                    *db_guard = Some(sdb);
                }
                Err(e) => return Err(format!("SurrealDB connection failed: {}", e)),
            }
        }
    }

    // Get embedding count
    let count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            DbOperations::embedding_count(sdb).await.unwrap_or(0)
        } else { 0 }
    };

    Ok(format!("Ready. Embeddings: {}", count))
}

/// Scan a folder for images/videos and run AI pipeline.
#[tauri::command]
async fn cmd_scan_folder(
    source_path: String,
    state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    let engine_arc = state.engine.clone();
    let db_arc = state.db.clone();

    ingest::image_ingest::ingest_folder(source_path, db_arc, engine_arc, None)
        .await
        .map_err(|e| e.to_string())
}

/// Search by text query.
#[tauri::command]
async fn cmd_search_text(
    query: String,
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::Text,
        text: Some(query.clone()),
        image_path: None,
        filters: filters.unwrap_or_default(),
    };
    run_search(search_query, Some(query), None, &state).await
}

/// Search by uploaded image.
#[tauri::command]
async fn cmd_search_image(
    image_path: String,
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::Image,
        text: None,
        image_path: Some(image_path.clone()),
        filters: filters.unwrap_or_default(),
    };
    run_search(search_query, None, Some(image_path), &state).await
}

/// Combined text + image search.
#[tauri::command]
async fn cmd_search_combined(
    text: String,
    image_path: String,
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::Combined,
        text: Some(text.clone()),
        image_path: Some(image_path.clone()),
        filters: filters.unwrap_or_default(),
    };
    run_search(search_query, Some(text), Some(image_path), &state).await
}

/// Search by COCO object class name.
#[tauri::command]
async fn cmd_search_object(
    class_name: String,
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let mut f = filters.unwrap_or_default();
    f.object = Some(class_name);
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::ObjectFilter,
        text: None,
        image_path: None,
        filters: f,
    };
    run_search(search_query, None, None, &state).await
}

/// Search by face name.
#[tauri::command]
async fn cmd_search_face(
    name: String,
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let mut f = filters.unwrap_or_default();
    f.face = Some(name);
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::FaceFilter,
        text: None,
        image_path: None,
        filters: f,
    };
    run_search(search_query, None, None, &state).await
}

/// Toggle favorite status for a media item.
#[tauri::command]
async fn cmd_toggle_favorite(
    media_id: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::toggle_favorite(db, &media_id).await.map_err(|e| e.to_string())
}

/// Get timeline.
#[tauri::command]
async fn cmd_get_timeline(
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<Vec<TimelineGroup>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_timeline(db, limit.unwrap_or(5000))
        .await.map_err(|e| e.to_string())
}

/// Get all people / face clusters.
#[tauri::command]
async fn cmd_get_people(
    state: State<'_, AppState>,
) -> Result<Vec<PersonGroup>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_people(db).await.map_err(|e| e.to_string())
}

/// Name a face cluster.
#[tauri::command]
async fn cmd_name_person(
    face_id: String,
    name: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::name_person(db, &face_id, &name).await.map_err(|e| e.to_string())
}

/// Find duplicate images.
#[tauri::command]
async fn cmd_get_duplicates(
    state: State<'_, AppState>,
) -> Result<Vec<DuplicateGroup>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_duplicates(db).await.map_err(|e| e.to_string())
}

/// Get search history.
#[tauri::command]
async fn cmd_get_search_history(
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    let history = DbOperations::get_search_history(db, limit.unwrap_or(20))
        .await.map_err(|e| e.to_string())?;
    Ok(serde_json::to_value(history).unwrap_or_default())
}

/// Set SurrealDB connection info.
#[tauri::command]
async fn cmd_set_db_config(
    addr: String,
    user: String,
    pass: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    *state.surreal_addr.lock().await = addr;
    *state.surreal_user.lock().await = user;
    *state.surreal_pass.lock().await = pass;
    // Reset connection so next cmd_init will reconnect
    *state.db.lock().await = None;
    Ok(())
}

/// Filter-only search (no text/image, just year/month/media_type filters).
#[tauri::command]
async fn cmd_search_filter_only(
    filters: Option<search::pipeline::SearchQueryFilters>,
    state: State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let search_query = SearchQuery {
        mode: search::pipeline::SearchMode::FilterOnly,
        text: None,
        image_path: None,
        filters: filters.unwrap_or_default(),
    };
    run_search(search_query, None, None, &state).await
}

/// Get distinct detected object class names from DB.
#[tauri::command]
async fn cmd_get_distinct_objects(
    state: State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_distinct_objects(db).await.map_err(|e| e.to_string())
}

/// Get engine + DB status info.
#[tauri::command]
async fn cmd_get_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let engine_ready = state.engine.lock().await.is_some();
    let db_ready = state.db.lock().await.is_some();
    let vector_count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            DbOperations::embedding_count(sdb).await.unwrap_or(0)
        } else { 0 }
    };
    Ok(serde_json::json!({
        "engine_ready": engine_ready,
        "db_ready": db_ready,
        "vector_count": vector_count,
    }))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn run_search(
    query: SearchQuery,
    text: Option<String>,
    image_path: Option<String>,
    state: &State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized. Call cmd_init first.")?;

    let mut engine_guard = state.engine.lock().await;
    let engine = engine_guard.as_mut().ok_or("Engine not initialized. Call cmd_init first.")?;

    let results = SearchPipeline::run(&query, engine, db)
        .await
        .map_err(|e| e.to_string())?;

    // Save search history
    let _ = DbOperations::save_search_history(db, text, image_path, None).await;

    Ok(results)
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            cmd_init,
            cmd_scan_folder,
            cmd_search_text,
            cmd_search_image,
            cmd_search_combined,
            cmd_search_object,
            cmd_search_face,
            cmd_search_filter_only,
            cmd_toggle_favorite,
            cmd_get_timeline,
            cmd_get_people,
            cmd_name_person,
            cmd_get_duplicates,
            cmd_get_distinct_objects,
            cmd_get_search_history,
            cmd_set_db_config,
            cmd_get_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() -> Result<()> {
    Logger::init("log/auraseek.log");

    // Nếu muốn test việc sinh ra ảnh vẽ bbbox (vẽ khung nhận diện lỗi), log, json vector, cropped faces 
    // Thì đổi giá trị này thành true:
    let run_cli_debug_ingest = false; 

    if run_cli_debug_ingest {
        // Chạy thư mục input được cấu hình sẵn rồi tự sinh output, tắt app React
        debug_cli::run_debug_ingest("input", "output")?;
        return Ok(());
    }

    // Nếu fasle, chạy React Tauri App bình thường
    run();
    Ok(())
}

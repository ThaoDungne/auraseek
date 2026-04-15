use tauri::State;
use crate::app::state::AppState;
use crate::core::models::{TimelineGroup, DuplicateGroup, CustomAlbum};
use crate::infrastructure::database::DbOperations;

#[tauri::command]
pub async fn cmd_create_album(title: String, state: State<'_, AppState>) -> Result<String, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::create_album(db, title).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_albums(state: State<'_, AppState>) -> Result<Vec<CustomAlbum>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_albums(db, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_add_to_album(album_id: String, media_ids: Vec<String>, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::add_to_album(db, &album_id, media_ids).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_remove_from_album(album_id: String, media_ids: Vec<String>, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::remove_from_album(db, &album_id, media_ids).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_delete_album(album_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::delete_album(db, &album_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_album_photos(album_id: String, state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_album_photos(db, &album_id, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_duplicates(state: State<'_, AppState>, media_type: Option<String>) -> Result<Vec<DuplicateGroup>, String> {
    let db_guard   = state.db.lock().await;
    let db         = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    let thumb_cache_dir = state.data_dir.lock().unwrap().join("thumbnails");
    DbOperations::get_duplicates(db, &source_dir, media_type.as_deref(), Some(&thumb_cache_dir))
        .await.map_err(|e| e.to_string())
}

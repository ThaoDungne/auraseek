use tauri::State;
use crate::app::state::AppState;
use crate::core::models::TimelineGroup;
use crate::infrastructure::database::DbOperations;

#[tauri::command]
pub async fn cmd_toggle_favorite(media_id: String, state: State<'_, AppState>) -> Result<bool, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::toggle_favorite(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_timeline(limit: Option<usize>, state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let db_guard   = state.db.lock().await;
    let db         = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    DbOperations::get_timeline(db, limit.unwrap_or(5000), &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_move_to_trash(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::move_to_trash(db, &source_dir, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_restore_from_trash(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::restore_from_trash(db, &source_dir, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_trash(state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard   = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_trash(db, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_empty_trash(state: State<'_, AppState>) -> Result<(), String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::empty_trash(db, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_hard_delete_trash_item(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::hard_delete_trash_item(db, &source_dir, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_hide_photo(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::hide_photo(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_unhide_photo(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::unhide_photo(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_get_hidden_photos(state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard   = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_hidden_photos(db, &source_dir).await.map_err(|e| e.to_string())
}

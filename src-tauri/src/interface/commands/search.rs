use tauri::State;
use crate::app::state::AppState;
use crate::core::models::SearchResult;
use crate::infrastructure::database::DbOperations;
use crate::app::search::SearchPipeline;
use crate::core::models::{SearchMode, SearchQuery, SearchQueryFilters};

async fn run_search(
    query: SearchQuery, text: Option<String>, image_path: Option<String>, state: &State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized. Call cmd_init first.")?;
    let mut engine_guard = state.engine.lock().await;
    let engine = engine_guard.as_mut().ok_or("Engine not initialized. Call cmd_init first.")?;

    let results = match SearchPipeline::run(&query, engine, db, &source_dir).await {
        Ok(r) => { crate::log_info!("🔍 [run_search] mode={:?} text={:?} image_path={:?} results={}", query.mode, text, image_path, r.len()); r }
        Err(e) => { crate::log_error!("❌ [run_search] failed mode={:?} text={:?} image_path={:?} error={}", query.mode, text, image_path, e); return Err(e.to_string()); }
    };
    let _ = DbOperations::save_search_history(db, text, image_path, None).await;
    Ok(results)
}

#[tauri::command]
pub async fn cmd_search_text(query: String, filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    let sq = SearchQuery { mode: SearchMode::Text, text: Some(query.clone()), image_path: None, filters: filters.unwrap_or_default() };
    run_search(sq, Some(query), None, &state).await
}

#[tauri::command]
pub async fn cmd_search_image(image_path: String, filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    crate::log_info!("🔍 [cmd_search_image] raw_image_path='{}'", image_path);
    let resolved = if std::path::Path::new(&image_path).is_absolute() { image_path.clone() } else {
        let base = state.source_dir.lock().await.clone();
        if base.is_empty() { image_path.clone() } else { format!("{}/{}", base.trim_end_matches('/'), image_path) }
    };
    crate::log_info!("🔍 [cmd_search_image] resolved_image_path='{}'", resolved);
    let sq = SearchQuery { mode: SearchMode::Image, text: None, image_path: Some(resolved.clone()), filters: filters.unwrap_or_default() };
    run_search(sq, None, Some(resolved), &state).await
}

#[tauri::command]
pub async fn cmd_search_combined(text: String, image_path: String, filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    crate::log_info!("🔍 [cmd_search_combined] raw_text='{}' raw_image_path='{}'", text, image_path);
    let resolved = if std::path::Path::new(&image_path).is_absolute() { image_path.clone() } else {
        let base = state.source_dir.lock().await.clone();
        if base.is_empty() { image_path.clone() } else { format!("{}/{}", base.trim_end_matches('/'), image_path) }
    };
    crate::log_info!("🔍 [cmd_search_combined] resolved_image_path='{}'", resolved);
    let sq = SearchQuery { mode: SearchMode::Combined, text: Some(text.clone()), image_path: Some(resolved.clone()), filters: filters.unwrap_or_default() };
    run_search(sq, Some(text), Some(resolved), &state).await
}

#[tauri::command]
pub async fn cmd_search_object(class_name: String, filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    let mut f = filters.unwrap_or_default(); f.object = Some(class_name);
    let sq = SearchQuery { mode: SearchMode::ObjectFilter, text: None, image_path: None, filters: f };
    run_search(sq, None, None, &state).await
}

#[tauri::command]
pub async fn cmd_search_face(name: String, filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    let mut f = filters.unwrap_or_default(); f.face = Some(name);
    let sq = SearchQuery { mode: SearchMode::FaceFilter, text: None, image_path: None, filters: f };
    run_search(sq, None, None, &state).await
}

#[tauri::command]
pub async fn cmd_search_filter_only(filters: Option<SearchQueryFilters>, state: State<'_, AppState>) -> Result<Vec<SearchResult>, String> {
    let sq = SearchQuery { mode: SearchMode::FilterOnly, text: None, image_path: None, filters: filters.unwrap_or_default() };
    run_search(sq, None, None, &state).await
}

#[tauri::command]
pub async fn cmd_get_search_history(limit: Option<usize>, state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    let history = DbOperations::get_search_history(db, limit.unwrap_or(20)).await.map_err(|e| e.to_string())?;
    Ok(serde_json::to_value(history).unwrap_or_default())
}

#[tauri::command]
pub async fn cmd_get_distinct_objects(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_distinct_objects(db).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_save_search_image(data: Vec<u8>, ext: String, app: tauri::AppHandle) -> Result<String, String> {
    use tauri::Manager;
    let app_data_dir = app.path().app_data_dir().map_err(|e| format!("app_data_dir error: {}", e))?;
    let tmp_dir = app_data_dir.join("search_tmp");
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) { crate::log_error!("❌ Failed to create search_tmp dir: {}", e); }
    let filename = format!("search_{}.{}", chrono::Utc::now().timestamp_millis(), ext.trim_start_matches('.'));
    let path = tmp_dir.join(filename);
    if let Err(e) = std::fs::write(&path, data) { crate::log_error!("❌ Failed to write temp search image: {}", e); return Err(format!("write temp file failed: {}", e)); }
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn cmd_delete_file(path: String) -> Result<(), String> {
    match std::fs::remove_file(&path) {
        Ok(_) => Ok(()),
        Err(e) => { crate::log_error!("❌ Failed to delete temp file {}: {}", path, e); Err(format!("delete file failed: {}", e)) }
    }
}

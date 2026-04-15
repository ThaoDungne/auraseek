use tauri::State;
use crate::app::state::AppState;
use crate::app::helpers::{available_ram_percent, restart_fs_watcher};
use crate::core::models::{SyncStatus, IngestSummary};
use crate::infrastructure::database::DbOperations;

#[tauri::command]
pub async fn cmd_get_source_dir(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.source_dir.lock().await.clone())
}

#[tauri::command]
pub async fn cmd_set_source_dir(dir: String, state: State<'_, AppState>) -> Result<(), String> {
    {
        let db_guard = state.db.lock().await;
        let db = db_guard.as_ref().ok_or("DB not initialized")?;
        DbOperations::set_source_dir(db, &dir).await.map_err(|e| e.to_string())?;
    }
    *state.source_dir.lock().await = dir.clone();
    crate::log_info!("📂 source_dir updated to: {}", dir);
    restart_fs_watcher(&state, &dir);
    Ok(())
}

#[tauri::command]
pub async fn cmd_get_sync_status(state: State<'_, AppState>) -> Result<SyncStatus, String> {
    Ok(state.sync_status.lock().await.clone())
}

#[tauri::command]
pub async fn cmd_auto_scan(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() { return Err("No source directory configured".into()); }

    let ram_pct = available_ram_percent();
    crate::log_info!("🖥️  Available RAM: {:.1}%", ram_pct);
    if ram_pct < 40.0 {
        let msg = format!("Không đủ RAM ({:.1}% trống, cần >40%). Đóng bớt ứng dụng và thử lại.", ram_pct);
        crate::log_warn!("⚠️ {}", msg);
        return Err(msg);
    }

    { let st = state.sync_status.lock().await; if st.state == "syncing" { return Ok("Already syncing".into()); } }

    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    let sync_arc   = state.sync_status.clone();
    let dir        = source_dir.clone();

    crate::log_info!("🔄 Auto-scan starting for: {}", dir);
    { let mut st = sync_arc.lock().await; *st = SyncStatus { state: "syncing".into(), processed: 0, total: 0, message: "Đang đồng bộ dữ liệu...".into() }; }

    restart_fs_watcher(&state, &source_dir);

    let thumb_cache_dir = state.data_dir.lock().unwrap().join("thumbnails");
    let thumb_cache = Some(thumb_cache_dir);
    let app_handle = app.clone();
    let abort_flag = state.abort_sync.clone();
    tokio::spawn(async move {
        if let Some(ref sdb) = *db_arc.lock().await {
            let _ = DbOperations::prune_missing_media(sdb, &dir).await;
        }
        let result = crate::app::ingest::ingest_folder(
            dir.clone(), db_arc, engine_arc, Some(app_handle), thumb_cache, abort_flag
        ).await;

        let mut st = sync_arc.lock().await;
        match result {
            Ok(summary) => {
                crate::log_info!("✅ Auto-scan done: new={} skip={} err={}", summary.newly_added, summary.skipped_dup, summary.errors);
                *st = SyncStatus { state: "done".into(), processed: summary.newly_added, total: summary.total_found, message: format!("Đã đồng bộ ({} ảnh mới)", summary.newly_added) };
            }
            Err(e) => {
                crate::log_error!("❌ Auto-scan failed: {}", e);
                *st = SyncStatus { state: "error".into(), processed: 0, total: 0, message: format!("Lỗi đồng bộ: {}", e) };
            }
        }
    });
    Ok("Sync started".into())
}

#[tauri::command]
pub async fn cmd_scan_folder(
    source_path: String, app: tauri::AppHandle, state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    let thumb_cache_dir = state.data_dir.lock().unwrap().join("thumbnails");
    let abort_flag = state.abort_sync.clone();
    crate::app::ingest::ingest_folder(source_path, db_arc, engine_arc, Some(app), Some(thumb_cache_dir), abort_flag)
        .await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_ingest_files(
    file_paths: Vec<String>, state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() { return Err("Chưa chọn thư mục nguồn ảnh".into()); }
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    let thumb_cache_dir = Some(state.data_dir.lock().unwrap().join("thumbnails"));
    crate::app::ingest::ingest_files(file_paths, source_dir, db_arc, engine_arc, thumb_cache_dir)
        .await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_ingest_image_data(
    data: String, ext: String, state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    use base64::Engine as _;
    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() { return Err("Chưa chọn thư mục nguồn ảnh".into()); }
    let ext = ext.trim_start_matches('.').to_lowercase();
    let allowed = ["jpg", "jpeg", "png", "webp"];
    if !allowed.contains(&ext.as_str()) { return Err(format!("Định dạng ảnh '{}' không được hỗ trợ", ext)); }
    let bytes = base64::engine::general_purpose::STANDARD.decode(&data).map_err(|e| format!("Lỗi giải mã base64: {}", e))?;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
    let filename = format!("paste_{}.{}", ts, ext);
    let dest = std::path::Path::new(&source_dir).join(&filename);
    std::fs::write(&dest, &bytes).map_err(|e| format!("Không thể lưu ảnh: {}", e))?;
    crate::log_info!("📋 Clipboard image saved: {}", dest.display());
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    let thumb_cache_dir = Some(state.data_dir.lock().unwrap().join("thumbnails"));
    crate::app::ingest::ingest_files(vec![dest.to_string_lossy().to_string()], source_dir, db_arc, engine_arc, thumb_cache_dir)
        .await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cmd_cleanup_database(state: State<'_, AppState>) -> Result<usize, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard = state.db.lock().await;
    if let Some(ref sdb) = *db_guard {
        let _ = DbOperations::auto_purge_trash(sdb, &source_dir).await;
        let count = DbOperations::prune_missing_media(sdb, &source_dir).await.map_err(|e| e.to_string())?;
        Ok(count)
    } else { Err("DB not initialized".into()) }
}

#[tauri::command]
pub async fn cmd_reset_database(state: State<'_, AppState>) -> Result<(), String> {
    state.abort_sync.store(true, std::sync::atomic::Ordering::SeqCst);
    { let mut st = state.sync_status.lock().await; *st = SyncStatus { state: "idle".into(), processed: 0, total: 0, message: "".into() }; }
    if let Ok(mut guard) = state.watcher_handle.lock() {
        if let Some(handle) = guard.take() { handle.stop(); crate::log_info!("👁️  FS watcher stopped due to database reset"); }
    }
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::clear_database(db).await.map_err(|e| e.to_string())?;
    *state.source_dir.lock().await = String::new();
    crate::log_info!("🧹 Database and configuration reset completed.");
    Ok(())
}

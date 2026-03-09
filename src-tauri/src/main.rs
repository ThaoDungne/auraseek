mod utils;
mod model;
mod processor;
mod db;
mod ingest;
mod search;
mod debug_cli;
mod surreal_sidecar;
mod downloader;
mod fs_watcher;

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

use notify::{RecommendedWatcher};
use notify_debouncer_full::{Debouncer, FileIdMap};

type AppWatcher = Debouncer<RecommendedWatcher, FileIdMap>;

// ─── Sync status ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct SyncStatus {
    pub state:     String, // "idle" | "syncing" | "done" | "error"
    pub processed: usize,
    pub total:     usize,
    pub message:   String,
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self { state: "idle".into(), processed: 0, total: 0, message: "Chưa đồng bộ".into() }
    }
}

// ─── App State ───────────────────────────────────────────────────────────────

pub struct AppState {
    pub engine:        Arc<Mutex<Option<AuraSeekEngine>>>,
    pub db:            Arc<Mutex<Option<SurrealDb>>>,
    /// SurrealDB connection address – set by the sidecar launcher before any
    /// command runs.  Uses `std::sync::Mutex` so it can be updated synchronously
    /// from the Tauri `setup` callback (which is not async).
    pub surreal_addr:  std::sync::Mutex<String>,
    pub surreal_user:  std::sync::Mutex<String>,
    pub surreal_pass:  std::sync::Mutex<String>,
    /// Loaded from config_auraseek on init; kept in memory to avoid repeated DB queries
    pub source_dir:    Mutex<String>,
    pub sync_status:   Arc<Mutex<SyncStatus>>,
    /// Handle to the SurrealDB child process (None if we reused an external instance).
    /// Killed when the last window closes.
    pub surreal_child: std::sync::Mutex<Option<std::process::Child>>,
    /// Base app data directory (set in setup callback).
    /// Models: <data_dir>/models/   Tokenizer: <data_dir>/tokenizer/
    /// SurrealDB:  <data_dir>/db/   Logs: <data_dir>/auraseek.log
    pub data_dir:      std::sync::Mutex<std::path::PathBuf>,
    /// Handle for the file system watcher. Kept alive as long as a source_dir
    /// is configured. Replaced when source_dir changes.
    pub watcher_handle: std::sync::Mutex<Option<fs_watcher::FsWatcherHandle>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            engine:        Arc::new(Mutex::new(None)),
            db:            Arc::new(Mutex::new(None)),
            // Will be set by the sidecar launcher in `setup()`. Start empty so we
            // never accidentally fall back to a fixed port – the sidecar always
            // chooses a free port in [8000, 9000].
            surreal_addr:  std::sync::Mutex::new(String::new()),
            surreal_user:  std::sync::Mutex::new("root".to_string()),
            surreal_pass:  std::sync::Mutex::new("root".to_string()),
            source_dir:    Mutex::new(String::new()),
            sync_status:   Arc::new(Mutex::new(SyncStatus::default())),
            surreal_child: std::sync::Mutex::new(None),
            data_dir:       std::sync::Mutex::new(std::path::PathBuf::from(".")),
            watcher_handle: std::sync::Mutex::new(None),
        }
    }
}

// ─── RAM helper ──────────────────────────────────────────────────────────────

/// Returns available RAM as a percentage of total.
/// Uses the `sysinfo` crate — works on Linux, Windows, and macOS.
fn available_ram_percent() -> f64 {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total = sys.total_memory();
    let avail = sys.available_memory();
    if total > 0 {
        (avail as f64 / total as f64) * 100.0
    } else {
        50.0 // assume OK if we can't read
    }
}

/// Stop the previous FS watcher (if any) and start a new one for `source_dir`.
fn restart_fs_watcher(state: &AppState, source_dir: &str) {
    // Drop the old watcher first
    if let Ok(mut guard) = state.watcher_handle.lock() {
        if let Some(old) = guard.take() {
            old.stop();
            crate::log_info!("👁️  Previous FS watcher stopped");
        }
    }

    if source_dir.is_empty() {
        return;
    }

    match fs_watcher::start_watching(
        source_dir.to_string(),
        state.db.clone(),
        state.engine.clone(),
        state.sync_status.clone(),
    ) {
        Ok(handle) => {
            if let Ok(mut guard) = state.watcher_handle.lock() {
                *guard = Some(handle);
            }
        }
        Err(e) => {
            crate::log_warn!("⚠️ Failed to start FS watcher: {}", e);
        }
    }
}

// ─── Tauri Commands ──────────────────────────────────────────────────────────

/// Check whether all required AI model files exist in the app data directory.
/// The frontend calls this before `cmd_init` to decide whether to show the
/// download screen.
#[tauri::command]
async fn cmd_check_models(state: State<'_, AppState>) -> Result<bool, String> {
    let data_dir = state.data_dir.lock().unwrap().clone();
    Ok(downloader::all_present(&data_dir))
}

/// Start downloading missing AI model assets in the background.
/// Progress is reported via `"model-download-progress"` Tauri events.
/// The frontend listens for the `done: true` event before calling `cmd_init`.
#[tauri::command]
async fn cmd_download_models(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let data_dir = state.data_dir.lock().unwrap().clone();
    tokio::spawn(async move {
        if let Err(e) = downloader::download_models_if_missing(&app, &data_dir).await {
            crate::log_error!("❌ Model download failed: {:#}", e);
            use tauri::Emitter;
            let _ = app.emit("model-download-progress", downloader::DownloadProgress {
                file: String::new(), progress: 0.0,
                message: format!("Lỗi tải model: {}", e),
                done: false, error: e.to_string(),
                file_index: 0, file_total: 0,
                bytes_done: 0, bytes_total: 0,
            });
        }
    });
    Ok(())
}

/// Initialize AI engine and SurrealDB connection, load source_dir from config.
/// Assumes model files are already present (call cmd_download_models first if needed).
#[tauri::command]
async fn cmd_init(state: State<'_, AppState>) -> Result<String, String> {
    let data_dir = state.data_dir.lock().unwrap().clone();

    // Init engine – load models from the app data directory
    {
        let mut engine_guard = state.engine.lock().await;
        if engine_guard.is_none() {
            crate::log_info!("🚀 Initializing AI engine from {}", data_dir.display());

            let _ = std::fs::create_dir_all(data_dir.join("face_db"));
            let config = processor::pipeline::EngineConfig::new_with_dir(&data_dir);
            match AuraSeekEngine::new(config) {
                Ok(e) => {
                    crate::log_info!("✅ AI engine ready");
                    *engine_guard = Some(e);
                }
                Err(e) => return Err(format!("Engine init failed: {}. Download models first.", e)),
            }
        }
    }

    // Init DB
    {
        let addr = state.surreal_addr.lock().unwrap().clone();
        let user = state.surreal_user.lock().unwrap().clone();
        let pass = state.surreal_pass.lock().unwrap().clone();
        let mut db_guard = state.db.lock().await;
        if db_guard.is_none() {
            match SurrealDb::connect(&addr, &user, &pass).await {
                Ok(sdb) => {
                    // Start auto-purge trash worker in background
                    let sdb_clone = sdb.clone();
                    tokio::spawn(async move {
                        if let Err(e) = DbOperations::auto_purge_trash(&sdb_clone).await {
                            crate::log_warn!("Failed to auto-purge trash: {}", e);
                        }
                    });

                    *db_guard = Some(sdb);
                }
                Err(e) => return Err(format!("SurrealDB connection failed: {}", e)),
            }
        }
    }

    // Load source_dir from config_auraseek
    {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            match DbOperations::get_source_dir(sdb).await {
                Ok(Some(dir)) => {
                    crate::log_info!("📂 source_dir loaded from config: {}", dir);
                    *state.source_dir.lock().await = dir;
                    // Filesystem watcher for source_dir is managed by restart_fs_watcher.
                }
                Ok(None) => {
                    crate::log_info!("📂 No source_dir configured yet (first run)");
                }
                Err(e) => {
                    crate::log_warn!("⚠️ Failed to load source_dir: {}", e);
                }
            }
        }
    }

    let count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            DbOperations::embedding_count(sdb).await.unwrap_or(0)
        } else { 0 }
    };

    let source_dir = state.source_dir.lock().await.clone();
    crate::log_info!("✅ Init complete | embeddings={} source_dir='{}'", count, source_dir);
    Ok(format!("Ready. Embeddings: {}", count))
}

/// Get the configured source directory.
#[tauri::command]
async fn cmd_get_source_dir(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.source_dir.lock().await.clone())
}

/// Set source directory and persist to config_auraseek. Then trigger auto-scan.
/// Also (re)starts the FS watcher on the new directory.
#[tauri::command]
async fn cmd_set_source_dir(
    dir: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    {
        let db_guard = state.db.lock().await;
        let db = db_guard.as_ref().ok_or("DB not initialized")?;
        DbOperations::set_source_dir(db, &dir).await.map_err(|e| e.to_string())?;
    }
    *state.source_dir.lock().await = dir.clone();
    crate::log_info!("📂 source_dir updated to: {}", dir);

    // (Re)start FS watcher on the new source directory
    restart_fs_watcher(&state, &dir);

    Ok(())
}

/// Get current sync status.
#[tauri::command]
async fn cmd_get_sync_status(state: State<'_, AppState>) -> Result<SyncStatus, String> {
    Ok(state.sync_status.lock().await.clone())
}

/// Start background auto-scan (called by frontend on app load if source_dir is set).
/// Checks RAM: requires >40% free before starting.
#[tauri::command]
async fn cmd_auto_scan(state: State<'_, AppState>) -> Result<String, String> {
    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() {
        return Err("No source directory configured".into());
    }

    // RAM check
    let ram_pct = available_ram_percent();
    crate::log_info!("🖥️  Available RAM: {:.1}%", ram_pct);
    if ram_pct < 40.0 {
        let msg = format!("Không đủ RAM ({:.1}% trống, cần >40%). Đóng bớt ứng dụng và thử lại.", ram_pct);
        crate::log_warn!("⚠️ {}", msg);
        return Err(msg);
    }

    // Check if already syncing
    {
        let st = state.sync_status.lock().await;
        if st.state == "syncing" {
            return Ok("Already syncing".into());
        }
    }

    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    let sync_arc   = state.sync_status.clone();
    let dir        = source_dir.clone();

    crate::log_info!("🔄 Auto-scan starting for: {}", dir);
    {
        let mut st = sync_arc.lock().await;
        *st = SyncStatus { state: "syncing".into(), processed: 0, total: 0, message: "Đang đồng bộ dữ liệu...".into() };
    }

    // Start FS watcher so new files added during or after the scan are picked up
    restart_fs_watcher(&state, &source_dir);

    tokio::spawn(async move {
        // Prune first
        if let Some(ref sdb) = *db_arc.lock().await {
            let _ = DbOperations::prune_missing_media(sdb, &dir).await;
        }

        let result = ingest::image_ingest::ingest_folder(
            dir.clone(), db_arc, engine_arc, None
        ).await;

        let mut st = sync_arc.lock().await;
        match result {
            Ok(summary) => {
                crate::log_info!("✅ Auto-scan done: new={} skip={} err={}", summary.newly_added, summary.skipped_dup, summary.errors);
                *st = SyncStatus {
                    state: "done".into(),
                    processed: summary.newly_added,
                    total: summary.total_found,
                    message: format!("Đã đồng bộ ({} ảnh mới)", summary.newly_added),
                };
            }
            Err(e) => {
                crate::log_error!("❌ Auto-scan failed: {}", e);
                *st = SyncStatus {
                    state: "error".into(),
                    processed: 0,
                    total: 0,
                    message: format!("Lỗi đồng bộ: {}", e),
                };
            }
        }
    });

    Ok("Sync started".into())
}

#[tauri::command]
async fn cmd_cleanup_database(state: State<'_, AppState>) -> Result<usize, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    DbOperations::prune_missing_media(db, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_reset_database(state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::clear_database(db).await.map_err(|e| e.to_string())?;
    *state.source_dir.lock().await = String::new();
    Ok(())
}

/// Scan a folder for images/videos and run AI pipeline (manual trigger).
#[tauri::command]
async fn cmd_scan_folder(
    source_path: String,
    state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    ingest::image_ingest::ingest_folder(source_path, db_arc, engine_arc, None)
        .await
        .map_err(|e| e.to_string())
}

/// Ingest specific image files (drag-drop from outside the source folder).
/// Files are copied to source_dir then processed.
#[tauri::command]
async fn cmd_ingest_files(
    file_paths: Vec<String>,
    state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() {
        return Err("Chưa chọn thư mục nguồn ảnh".into());
    }
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    ingest::image_ingest::ingest_files(file_paths, source_dir, db_arc, engine_arc)
        .await
        .map_err(|e| e.to_string())
}

/// Ingest a raw image from clipboard (paste of screenshot / browser image).
/// `data` is base64-encoded image bytes; `ext` is the format (e.g. "png", "jpg").
#[tauri::command]
async fn cmd_ingest_image_data(
    data: String,
    ext: String,
    state: State<'_, AppState>,
) -> Result<IngestSummary, String> {
    use base64::Engine as _;

    let source_dir = state.source_dir.lock().await.clone();
    if source_dir.is_empty() {
        return Err("Chưa chọn thư mục nguồn ảnh".into());
    }

    let ext = ext.trim_start_matches('.').to_lowercase();
    let allowed = ["jpg", "jpeg", "png", "webp"];
    if !allowed.contains(&ext.as_str()) {
        return Err(format!("Định dạng ảnh '{}' không được hỗ trợ", ext));
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&data)
        .map_err(|e| format!("Lỗi giải mã base64: {}", e))?;

    let ts       = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let filename = format!("paste_{}.{}", ts, ext);
    let dest     = std::path::Path::new(&source_dir).join(&filename);

    std::fs::write(&dest, &bytes)
        .map_err(|e| format!("Không thể lưu ảnh: {}", e))?;

    crate::log_info!("📋 Clipboard image saved: {}", dest.display());

    // Ingest the saved file (already in source_dir — ingest_files skips copy if same dir)
    let engine_arc = state.engine.clone();
    let db_arc     = state.db.clone();
    ingest::image_ingest::ingest_files(
        vec![dest.to_string_lossy().to_string()],
        source_dir,
        db_arc,
        engine_arc,
    )
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
    let db_guard   = state.db.lock().await;
    let db         = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    DbOperations::get_timeline(db, limit.unwrap_or(5000), &source_dir)
        .await.map_err(|e| e.to_string())
}

/// Get all people / face clusters.
#[tauri::command]
async fn cmd_get_people(
    state: State<'_, AppState>,
) -> Result<Vec<PersonGroup>, String> {
    let db_guard   = state.db.lock().await;
    let db         = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    DbOperations::get_people(db, &source_dir).await.map_err(|e| e.to_string())
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

// ─── Trash & Hidden ──────────────────────────────────────────────────────────

#[tauri::command]
async fn cmd_move_to_trash(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::move_to_trash(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_restore_from_trash(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::restore_from_trash(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_get_trash(state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard   = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_trash(db, &source_dir).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_empty_trash(state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::empty_trash(db).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_hide_photo(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::hide_photo(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_unhide_photo(media_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::unhide_photo(db, &media_id).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn cmd_get_hidden_photos(state: State<'_, AppState>) -> Result<Vec<TimelineGroup>, String> {
    let source_dir = state.source_dir.lock().await.clone();
    let db_guard   = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized")?;
    DbOperations::get_hidden_photos(db, &source_dir).await.map_err(|e| e.to_string())
}

/// Get a human-friendly device name for the current machine.
#[tauri::command]
fn cmd_get_device_name() -> Result<String, String> {
    let name = sysinfo::System::host_name()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "Thiết bị này".to_string());
    Ok(name)
}

/// Get file size in bytes for a given absolute path.
#[tauri::command]
fn cmd_get_file_size(path: String) -> Result<u64, String> {
    use std::fs;
    use std::path::Path;
    let meta = fs::metadata(Path::new(&path))
        .map_err(|e| format!("Không đọc được thông tin file: {}", e))?;
    Ok(meta.len())
}

#[tauri::command]
async fn cmd_authenticate_os() -> Result<bool, String> {
    // Attempt Linux OS authentication via polkit (pkexec)
    #[cfg(target_os = "linux")]
    {
        match std::process::Command::new("pkexec")
            .arg("true")
            .output()
        {
            Ok(output) => {
                return Ok(output.status.success());
            }
            Err(e) => {
                crate::log_warn!("OS Auth failed: {}", e);
                return Err("Failed to trigger OS authentication.".to_string());
            }
        }
    }
    
    // For other OS, simulate success or implement platform-specific auth
    #[cfg(not(target_os = "linux"))]
    {
        Ok(true)
    }
}

/// Find duplicate images.
#[tauri::command]
async fn cmd_get_duplicates(
    state: State<'_, AppState>,
) -> Result<Vec<DuplicateGroup>, String> {
    let db_guard   = state.db.lock().await;
    let db         = db_guard.as_ref().ok_or("DB not initialized")?;
    let source_dir = state.source_dir.lock().await.clone();
    DbOperations::get_duplicates(db, &source_dir).await.map_err(|e| e.to_string())
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

/// Set SurrealDB connection info (forces reconnect on next cmd_init call).
#[tauri::command]
async fn cmd_set_db_config(
    addr: String,
    user: String,
    pass: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    *state.surreal_addr.lock().map_err(|e| e.to_string())? = addr;
    *state.surreal_user.lock().map_err(|e| e.to_string())? = user;
    *state.surreal_pass.lock().map_err(|e| e.to_string())? = pass;
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
    let db_ready     = state.db.lock().await.is_some();
    let vector_count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            DbOperations::embedding_count(sdb).await.unwrap_or(0)
        } else { 0 }
    };
    let source_dir = state.source_dir.lock().await.clone();
    Ok(serde_json::json!({
        "engine_ready": engine_ready,
        "db_ready": db_ready,
        "vector_count": vector_count,
        "source_dir": source_dir,
    }))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn run_search(
    query: SearchQuery,
    text: Option<String>,
    image_path: Option<String>,
    state: &State<'_, AppState>,
) -> Result<Vec<SearchResult>, String> {
    let source_dir = state.source_dir.lock().await.clone();

    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("DB not initialized. Call cmd_init first.")?;

    let mut engine_guard = state.engine.lock().await;
    let engine = engine_guard.as_mut().ok_or("Engine not initialized. Call cmd_init first.")?;

    let results = SearchPipeline::run(&query, engine, db, &source_dir)
        .await
        .map_err(|e| e.to_string())?;

    let _ = DbOperations::save_search_history(db, text, image_path, None).await;

    Ok(results)
}

// ─── Path helpers ────────────────────────────────────────────────────────────

/// Return the current user's home directory.
fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new())
        // ── Start SurrealDB sidecar before any command runs ──────────────────
        .setup(|app| {
            use tauri::Manager;

            let resource_dir = app.path().resource_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));

            // Base data directory for this app (platform-aware)
            let base_data_dir = app.path().app_data_dir()
                .unwrap_or_else(|_| {
                    #[cfg(windows)]
                    { std::path::PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("auraseek") }
                    #[cfg(not(windows))]
                    { dirs_home().join(".local").join("share").join("auraseek") }
                });

            crate::log_info!("📁 App data dir: {}", base_data_dir.display());

            let state = app.state::<AppState>();

            // Store base data dir so cmd_init can locate model files
            *state.data_dir.lock().unwrap() = base_data_dir.clone();

            // SurrealDB uses a sub-directory so it doesn't conflict with model files
            let surreal_data_dir = base_data_dir.join("db");

            let user = state.surreal_user.lock().unwrap().clone();
            let pass = state.surreal_pass.lock().unwrap().clone();

            match surreal_sidecar::ensure_surreal(&resource_dir, &surreal_data_dir, &user, &pass) {
                Ok((addr, child_opt)) => {
                    crate::log_info!("🗄️  SurrealDB address: {}", addr);
                    *state.surreal_addr.lock().unwrap() = addr;
                    if let Some(child) = child_opt {
                        *state.surreal_child.lock().unwrap() = Some(child);
                    }
                }
                Err(e) => {
                    // Do NOT fall back to a fixed port. If the sidecar fails, we
                    // leave `surreal_addr` empty so `cmd_init` can report a clear
                    // error instead of trying a random hard-coded port.
                    crate::log_warn!("⚠️  SurrealDB sidecar failed: {}. DB will be unavailable until restart.", e);
                    *state.surreal_addr.lock().unwrap() = String::new();
                }
            }

            Ok(())
        })
        // ── Kill the SurrealDB child when the last window is destroyed ────────
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                use tauri::Manager;
                let app = window.app_handle();
                // Only terminate if no other windows remain
                if app.webview_windows().is_empty() {
                    let state = app.state::<AppState>();
                    // Stop FS watcher
                    if let Ok(mut guard) = state.watcher_handle.lock() {
                        if let Some(handle) = guard.take() {
                            handle.stop();
                            crate::log_info!("🛑 FS watcher stopped on exit");
                        }
                    }
                    // Kill SurrealDB sidecar
                    if let Ok(mut guard) = state.surreal_child.lock() {
                        if let Some(mut child) = guard.take() {
                            crate::log_info!("🛑 Terminating SurrealDB sidecar (pid={})...", child.id());
                            let _ = child.kill();
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            cmd_check_models,
            cmd_download_models,
            cmd_init,
            cmd_get_source_dir,
            cmd_set_source_dir,
            cmd_get_sync_status,
            cmd_auto_scan,
            cmd_scan_folder,
            cmd_ingest_files,
            cmd_ingest_image_data,
            cmd_get_device_name,
            cmd_get_file_size,
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
            cmd_move_to_trash,
            cmd_restore_from_trash,
            cmd_get_trash,
            cmd_empty_trash,
            cmd_hide_photo,
            cmd_unhide_photo,
            cmd_get_hidden_photos,
            cmd_authenticate_os,
            cmd_cleanup_database,
            cmd_reset_database,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() -> Result<()> {
    // Use an absolute log path so it works regardless of the working directory
    // (AppImage launched from file manager, installed .deb, or cargo run).
    // Log location: ~/.local/share/auraseek/auraseek.log  (Linux)
    //               %APPDATA%\auraseek\auraseek.log        (Windows)
    let log_path = {
        #[cfg(windows)]
        {
            std::env::var("APPDATA")
                .map(|p| format!("{}\\auraseek\\auraseek.log", p))
                .unwrap_or_else(|_| "auraseek.log".to_string())
        }
        #[cfg(not(windows))]
        {
            std::env::var("HOME")
                .map(|h| format!("{}/.local/share/auraseek/auraseek.log", h))
                .unwrap_or_else(|_| "/tmp/auraseek.log".to_string())
        }
    };
    Logger::init(&log_path);

    let run_cli_debug_ingest = false;

    if run_cli_debug_ingest {
        debug_cli::run_debug_ingest("input", "output")?;
        return Ok(());
    }

    run();
    Ok(())
}

use tauri::State;
use crate::app::state::AppState;
use crate::app::helpers::start_db_sidecar;
use crate::infrastructure::ai::AuraSeekEngine;
use crate::infrastructure::database::{SurrealDb, DbOperations};

#[tauri::command]
pub async fn cmd_check_models(state: State<'_, AppState>) -> Result<bool, String> {
    let data_dir = state.data_dir.lock().unwrap().clone();
    Ok(crate::infrastructure::network::all_models_present(&data_dir))
}

#[tauri::command]
pub async fn cmd_download_models(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let data_dir = state.data_dir.lock().unwrap().clone();
    tokio::spawn(async move {
        if let Err(e) = crate::infrastructure::network::ModelDownloader::download_if_missing(&app, &data_dir).await {
            crate::log_error!("❌ Model download failed: {:#}", e);
            use tauri::Emitter;
            let _ = app.emit("model-download-progress", crate::infrastructure::network::DownloadProgress {
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

#[tauri::command]
pub async fn cmd_init(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    let data_dir = state.data_dir.lock().unwrap().clone();

    {
        let mut engine_guard = state.engine.lock().await;
        if engine_guard.is_none() {
            crate::log_info!("🚀 Initializing AI engine from {}", data_dir.display());
            let _ = std::fs::create_dir_all(data_dir.join("face_db"));
            let config = crate::infrastructure::ai::engine::EngineConfig::new_with_dir(&data_dir);
            match AuraSeekEngine::new(config) {
                Ok(e) => { crate::log_info!("✅ AI engine ready"); *engine_guard = Some(e); }
                Err(e) => return Err(format!("Engine init failed: {}. Download models first.", e)),
            }
        }
    }

    {
        if state.surreal_addr.lock().unwrap().is_empty() {
            let _ = start_db_sidecar(&app);
        }
        let addr = state.surreal_addr.lock().unwrap().clone();
        let user = state.surreal_user.lock().unwrap().clone();
        let pass = state.surreal_pass.lock().unwrap().clone();
        let mut db_guard = state.db.lock().await;
        if db_guard.is_none() {
            match SurrealDb::connect(&addr, &user, &pass).await {
                Ok(sdb) => {
                    let sdb_clone = sdb.clone();
                    let source_dir_clone = state.source_dir.lock().await.clone();
                    tokio::spawn(async move {
                        if let Err(e) = DbOperations::auto_purge_trash(&sdb_clone, &source_dir_clone).await {
                            crate::log_warn!("Failed to auto-purge trash: {}", e);
                        }
                    });
                    *db_guard = Some(sdb);
                }
                Err(e) => return Err(format!("SurrealDB connection failed: {}", e)),
            }
        }
    }

    {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard {
            match DbOperations::get_source_dir(sdb).await {
                Ok(Some(dir)) => { crate::log_info!("📂 source_dir loaded from config: {}", dir); *state.source_dir.lock().await = dir; }
                Ok(None) => { crate::log_info!("📂 No source_dir configured yet (first run)"); }
                Err(e) => { crate::log_warn!("⚠️ Failed to load source_dir: {}", e); }
            }
        }
    }

    let count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard { DbOperations::embedding_count(sdb).await.unwrap_or(0) } else { 0 }
    };

    let source_dir = state.source_dir.lock().await.clone();
    crate::log_info!("✅ Init complete | embeddings={} source_dir='{}'", count, source_dir);
    Ok(format!("Ready. Embeddings: {}", count))
}

use tauri::State;
use crate::app::state::AppState;
use crate::infrastructure::database::DbOperations;

#[tauri::command]
pub fn cmd_get_device_name() -> Result<String, String> {
    let name = sysinfo::System::host_name()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "Thiết bị này".to_string());
    Ok(name)
}

#[tauri::command]
pub fn cmd_get_file_size(path: String) -> Result<u64, String> {
    let meta = std::fs::metadata(std::path::Path::new(&path))
        .map_err(|e| format!("Không đọc được thông tin file: {}", e))?;
    Ok(meta.len())
}

#[tauri::command]
pub async fn cmd_authenticate_os() -> Result<bool, String> {
    crate::platform::auth::authenticate_os()
}

#[tauri::command]
pub async fn cmd_set_db_config(addr: String, user: String, pass: String, state: State<'_, AppState>) -> Result<(), String> {
    *state.surreal_addr.lock().map_err(|e| e.to_string())? = addr;
    *state.surreal_user.lock().map_err(|e| e.to_string())? = user;
    *state.surreal_pass.lock().map_err(|e| e.to_string())? = pass;
    *state.db.lock().await = None;
    Ok(())
}

#[tauri::command]
pub async fn cmd_get_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let engine_ready = state.engine.lock().await.is_some();
    let db_ready     = state.db.lock().await.is_some();
    let vector_count = {
        let db_guard = state.db.lock().await;
        if let Some(ref sdb) = *db_guard { DbOperations::embedding_count(sdb).await.unwrap_or(0) } else { 0 }
    };
    let source_dir = state.source_dir.lock().await.clone();
    Ok(serde_json::json!({ "engine_ready": engine_ready, "db_ready": db_ready, "vector_count": vector_count, "source_dir": source_dir }))
}

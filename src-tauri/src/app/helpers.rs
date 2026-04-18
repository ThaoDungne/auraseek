use crate::app::state::AppState;

pub fn available_ram_percent() -> f64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();

    let total = sys.total_memory() as f64;
    let available = sys.available_memory() as f64;
    let free = sys.free_memory() as f64;
    let used = sys.used_memory() as f64;

    let avail = if available > 0.0 {
        available
    } else if free > 0.0 {
        free
    } else if total > 0.0 {
        (total - used).max(0.0)
    } else {
        0.0
    };

    if total > 0.0 {
        (avail / total) * 100.0
    } else {
        50.0
    }
}

pub fn restart_fs_watcher(state: &AppState, source_dir: &str) {
    if let Ok(mut guard) = state.watcher_handle.lock() {
        if let Some(old) = guard.take() {
            old.stop();
            crate::log_info!("👁️  Previous FS watcher stopped");
        }
    }
    if source_dir.is_empty() { return; }

    let thumb_cache_dir = Some(state.data_dir.lock().unwrap().join("thumbnails"));
    match crate::infrastructure::fs::FileWatcher::start(
        source_dir.to_string(),
        state.db.clone(),
        state.engine.clone(),
        state.sync_status.clone(),
        thumb_cache_dir,
    ) {
        Ok(handle) => {
            if let Ok(mut guard) = state.watcher_handle.lock() { *guard = Some(handle); }
        }
        Err(e) => { crate::log_warn!("⚠️ Failed to start FS watcher: {}", e); }
    }
}

pub fn start_db_sidecar(app: &tauri::AppHandle) -> Result<(), String> {
    use tauri::Manager;
    let state = app.state::<AppState>();

    {
        let addr = state.surreal_addr.lock().unwrap();
        if !addr.is_empty() { return Ok(()); }
    }

    let resource_dir = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let data_dir = state.data_dir.lock().unwrap().clone();
    let surreal_data_dir = data_dir.join("db");
    let user = state.surreal_user.lock().unwrap().clone();
    let pass = state.surreal_pass.lock().unwrap().clone();

    match crate::infrastructure::database::surreal_sidecar::SurrealService::ensure(&resource_dir, &surreal_data_dir, &user, &pass) {
        Ok((addr, child_opt)) => {
            crate::log_info!("🗄️  SurrealDB sidecar started: {}", addr);
            *state.surreal_addr.lock().unwrap() = addr;
            if let Some(child) = child_opt {
                *state.surreal_child.lock().unwrap() = Some(child);
            }
            Ok(())
        }
        Err(e) => {
            crate::log_warn!("⚠️  SurrealDB sidecar start failed: {}. (Expected on first launch before download)", e);
            Err(e.to_string())
        }
    }
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod core;
mod shared;
mod infrastructure;
#[cfg(feature = "debug-tools")]
mod debug;
mod platform;
mod app;
mod interface;

use anyhow::Result;
use shared::logging::Logger;

use app::state::AppState;
use app::helpers::start_db_sidecar;
use interface::streaming::spawn_stream_server;

#[macro_export]
macro_rules! sdb_log_info {
    ($($arg:tt)*) => ($crate::shared::logging::logger::Logger::info(&format!(
        "{}{}[SurrealDB]{}{} {}",
        $crate::shared::logging::logger::GREEN,
        $crate::shared::logging::logger::BOLD,
        $crate::shared::logging::logger::RESET,
        $crate::shared::logging::logger::RESET,
        format!($($arg)*)
    )));
}
#[macro_export]
macro_rules! sdb_log_warn {
    ($($arg:tt)*) => ($crate::shared::logging::logger::Logger::warn(&format!(
        "{}{}[SurrealDB]{}{} {}",
        $crate::shared::logging::logger::RED,
        $crate::shared::logging::logger::BOLD,
        $crate::shared::logging::logger::RESET,
        $crate::shared::logging::logger::RESET,
        format!($($arg)*)
    )));
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    crate::platform::setup::pre_init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new())
        .setup(|app| {
            use tauri::Manager;

            if let Ok(resource_dir) = app.path().resource_dir() {
                let _ = crate::platform::setup::ensure_native_libs(&resource_dir);
            }

            let port = spawn_stream_server();
            app.state::<AppState>().stream_port.store(port, std::sync::atomic::Ordering::Relaxed);
            crate::log_info!("🎥 Video stream server started on port {}", port);

            let base_data_dir = app.path().app_data_dir()
                .unwrap_or_else(|_| crate::platform::paths::fallback_data_dir());
            crate::log_info!("📁 App data dir: {}", base_data_dir.display());

            let state = app.state::<AppState>();
            *state.data_dir.lock().unwrap() = base_data_dir;
            let _ = start_db_sidecar(&app.handle());

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                use tauri::Manager;
                let app = window.app_handle();
                if app.webview_windows().is_empty() {
                    let state = app.state::<AppState>();
                    if let Ok(mut guard) = state.watcher_handle.lock() {
                        if let Some(handle) = guard.take() { handle.stop(); crate::log_info!("🛑 FS watcher stopped on exit"); }
                    }
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
            interface::commands::init::cmd_check_models,
            interface::commands::init::cmd_download_models,
            interface::commands::init::cmd_init,
            interface::commands::media::cmd_get_source_dir,
            interface::commands::media::cmd_set_source_dir,
            interface::commands::media::cmd_get_sync_status,
            interface::commands::media::cmd_auto_scan,
            interface::commands::media::cmd_scan_folder,
            interface::commands::media::cmd_ingest_files,
            interface::commands::media::cmd_ingest_image_data,
            interface::commands::media::cmd_cleanup_database,
            interface::commands::media::cmd_reset_database,
            interface::commands::search::cmd_search_text,
            interface::commands::search::cmd_search_image,
            interface::commands::search::cmd_search_combined,
            interface::commands::search::cmd_search_object,
            interface::commands::search::cmd_search_face,
            interface::commands::search::cmd_search_filter_only,
            interface::commands::search::cmd_get_search_history,
            interface::commands::search::cmd_get_distinct_objects,
            interface::commands::search::cmd_save_search_image,
            interface::commands::search::cmd_delete_file,
            interface::commands::people::cmd_get_people,
            interface::commands::people::cmd_name_person,
            interface::commands::people::cmd_merge_people,
            interface::commands::people::cmd_delete_person,
            interface::commands::people::cmd_remove_face_from_person,
            interface::commands::timeline::cmd_toggle_favorite,
            interface::commands::timeline::cmd_get_timeline,
            interface::commands::timeline::cmd_move_to_trash,
            interface::commands::timeline::cmd_restore_from_trash,
            interface::commands::timeline::cmd_get_trash,
            interface::commands::timeline::cmd_empty_trash,
            interface::commands::timeline::cmd_hard_delete_trash_item,
            interface::commands::timeline::cmd_hide_photo,
            interface::commands::timeline::cmd_unhide_photo,
            interface::commands::timeline::cmd_get_hidden_photos,
            interface::commands::albums::cmd_create_album,
            interface::commands::albums::cmd_get_albums,
            interface::commands::albums::cmd_add_to_album,
            interface::commands::albums::cmd_remove_from_album,
            interface::commands::albums::cmd_delete_album,
            interface::commands::albums::cmd_get_album_photos,
            interface::commands::albums::cmd_get_duplicates,
            interface::commands::system::cmd_get_device_name,
            interface::commands::system::cmd_get_file_size,
            interface::commands::system::cmd_authenticate_os,
            interface::commands::system::cmd_set_db_config,
            interface::commands::system::cmd_get_status,
            interface::streaming::cmd_get_stream_port,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let log_path = crate::platform::paths::default_log_path();
    Logger::init(&log_path);

    #[cfg(feature = "debug-tools")]
    {
        let run_cli_debug_ingest = false;
        if run_cli_debug_ingest {
            debug::cli::run_debug_ingest("input1", "output")?;
            return Ok(());
        }
    }

    run();
    Ok(())
}

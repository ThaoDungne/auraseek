//! File System Watcher for the source photo directory.
//!
//! Uses the `notify` crate to watch for new/modified files in the configured
//! source directory (recursively).  When new images or videos are detected,
//! they are collected into a batch and ingested after a short debounce period
//! to avoid processing half-written files or triggering multiple times for a
//! single drag-drop of many files.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher, Config, EventKind};
use tokio::sync::{Mutex, mpsc};

use crate::db::SurrealDb;
use crate::processor::AuraSeekEngine;
use crate::ingest::image_ingest;
use crate::SyncStatus;

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "heic", "avif"];
const VIDEO_EXTS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm", "m4v", "flv", "wmv"];
const DEBOUNCE_MS: u64 = 2000;

/// Handle returned by `start_watching`. Drop it (or call `stop()`) to stop the
/// watcher and the background ingest task.
pub struct FsWatcherHandle {
    _watcher: RecommendedWatcher,
    stop_tx: mpsc::Sender<()>,
}

impl FsWatcherHandle {
    pub fn stop(self) {
        let _ = self.stop_tx.try_send(());
    }
}

fn is_media_file(path: &PathBuf) -> bool {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if fname.ends_with(".thumb.jpg") {
        return false;
    }
    IMAGE_EXTS.contains(&ext.as_str()) || VIDEO_EXTS.contains(&ext.as_str())
}

/// Start watching `source_dir` for new media files.
/// If `thumb_cache_dir` is Some, video/face thumbnails from new files are written there.
///
/// Returns a handle that keeps the watcher alive; drop it to stop.
pub fn start_watching(
    source_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    sync_status: Arc<Mutex<SyncStatus>>,
    thumb_cache_dir: Option<PathBuf>,
) -> anyhow::Result<FsWatcherHandle> {
    let (event_tx, mut event_rx) = mpsc::channel::<PathBuf>(512);
    let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);

    // --- notify watcher (runs on a background OS thread) ---
    let tx_clone = event_tx.clone();
    let mut watcher = RecommendedWatcher::new(
        move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        for path in event.paths {
                            if path.is_file() && is_media_file(&path) {
                                let _ = tx_clone.try_send(path);
                            }
                        }
                    }
                    _ => {}
                }
            }
        },
        Config::default(),
    )?;

    watcher.watch(std::path::Path::new(&source_dir), RecursiveMode::Recursive)?;
    crate::log_info!("👁️  FS watcher started on: {}", source_dir);

    // --- Debounce + ingest task (runs on tokio) ---
    let dir_for_task = source_dir.clone();
    tokio::spawn(async move {
        let mut pending: HashSet<PathBuf> = HashSet::new();

        loop {
            tokio::select! {
                // Receive new file events
                Some(path) = event_rx.recv() => {
                    pending.insert(path);
                    // Drain any more events that arrived at the same time
                    while let Ok(p) = event_rx.try_recv() {
                        pending.insert(p);
                    }

                    // Debounce: wait a bit for more events before processing
                    tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)).await;

                    // Drain again after sleep
                    while let Ok(p) = event_rx.try_recv() {
                        pending.insert(p);
                    }

                    if pending.is_empty() {
                        continue;
                    }

                    let files: Vec<String> = pending.drain()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();

                    crate::log_info!("👁️  FS watcher detected {} new file(s), ingesting...", files.len());

                    // Update sync status
                    {
                        let mut st = sync_status.lock().await;
                        *st = SyncStatus {
                            state: "syncing".into(),
                            processed: 0,
                            total: files.len(),
                            message: format!("Đang xử lý {} tệp mới...", files.len()),
                        };
                    }

                    // RAM check
                    let ram_pct = crate::available_ram_percent();
                    if ram_pct < 10.0 {
                        crate::log_warn!("⚠️ FS watcher: not enough RAM ({:.1}%), skipping batch", ram_pct);
                        let mut st = sync_status.lock().await;
                        *st = SyncStatus {
                            state: "error".into(),
                            processed: 0,
                            total: 0,
                            message: format!("Không đủ RAM ({:.1}%)", ram_pct),
                        };
                        continue;
                    }

                    let thumb_cache = thumb_cache_dir.clone();
                    match image_ingest::ingest_files(
                        files.clone(),
                        dir_for_task.clone(),
                        db.clone(),
                        engine.clone(),
                        thumb_cache,
                    ).await {
                        Ok(summary) => {
                            crate::log_info!(
                                "✅ FS watcher ingest done: new={} skip={} err={}",
                                summary.newly_added, summary.skipped_dup, summary.errors
                            );
                            let mut st = sync_status.lock().await;
                            *st = SyncStatus {
                                state: "done".into(),
                                processed: summary.newly_added,
                                total: summary.total_found,
                                message: format!("Đã xử lý {} ảnh mới", summary.newly_added),
                            };
                        }
                        Err(e) => {
                            crate::log_error!("❌ FS watcher ingest failed: {}", e);
                            let mut st = sync_status.lock().await;
                            *st = SyncStatus {
                                state: "error".into(),
                                processed: 0,
                                total: 0,
                                message: format!("Lỗi xử lý: {}", e),
                            };
                        }
                    }
                }
                // Stop signal
                _ = stop_rx.recv() => {
                    crate::log_info!("👁️  FS watcher stopped");
                    break;
                }
            }
        }
    });

    Ok(FsWatcherHandle {
        _watcher: watcher,
        stop_tx,
    })
}

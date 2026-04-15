use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tauri::Emitter;

use crate::infrastructure::database::{SurrealDb, DbOperations};
use crate::infrastructure::ai::AuraSeekEngine;
use crate::infrastructure::ingest::image_processor::{
    self, collect_files, scan_single_file, process_image_file,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
};
use crate::infrastructure::ingest::video_processor;
use crate::core::models::{IngestSummary, IngestProgress};

pub async fn ingest_folder(
    source_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    app: Option<tauri::AppHandle>,
    thumb_cache_dir: Option<PathBuf>,
    abort_sync: Arc<std::sync::atomic::AtomicBool>,
) -> Result<IngestSummary> {
    abort_sync.store(false, std::sync::atomic::Ordering::SeqCst);

    let source_path = Path::new(&source_dir);
    if !source_path.exists() {
        crate::log_error!("Source directory not found: {}", source_dir);
        return Err(anyhow::anyhow!("Source directory not found: {}", source_dir));
    }

    {
        let db_guard = db.lock().await;
        if let Some(ref sdb) = *db_guard {
            if let Err(e) = DbOperations::set_source_dir(sdb, &source_dir).await {
                crate::log_warn!("⚠️ Failed to persist source_dir to config: {}", e);
            } else {
                crate::log_info!("📝 source_dir saved to config_auraseek: {}", source_dir);
            }
        }
    }

    let (image_files, video_files) = collect_files(source_path);
    let total = image_files.len() + video_files.len();
    crate::log_info!(
        "📂 Ingest started: {} | {} images + {} videos found",
        source_dir, image_files.len(), video_files.len()
    );

    let mut summary = IngestSummary {
        total_found: total,
        newly_added: 0,
        skipped_dup: 0,
        errors: 0,
    };

    let (tx, mut rx) = mpsc::channel::<(PathBuf, String, bool)>(64);
    let db_scan = db.clone();
    let source_dir_clone = source_dir.clone();
    let abort_scan = abort_sync.clone();

    let scan_task = tokio::spawn(async move {
        let mut newly_added = 0usize;
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for path in image_files {
            if abort_scan.load(std::sync::atomic::Ordering::SeqCst) {
                crate::log_info!("🛑 Ingest scan aborted for images");
                break;
            }
            match scan_single_file(&path, &db_scan, &source_dir_clone, "image").await {
                Ok(Some(media_id)) => {
                    newly_added += 1;
                    let _ = tx.send((path, media_id, false)).await;
                }
                Ok(None) => { skipped += 1; }
                Err(e) => {
                    crate::log_warn!("⚠️ Scan error {:?}: {}", path, e);
                    errors += 1;
                }
            }
        }

        for path in video_files {
            if abort_scan.load(std::sync::atomic::Ordering::SeqCst) {
                crate::log_info!("🛑 Ingest scan aborted for videos");
                break;
            }
            match scan_single_file(&path, &db_scan, &source_dir_clone, "video").await {
                Ok(Some(media_id)) => {
                    newly_added += 1;
                    let _ = tx.send((path, media_id, true)).await;
                }
                Ok(None) => { skipped += 1; }
                Err(e) => {
                    crate::log_warn!("⚠️ Scan error {:?}: {}", path, e);
                    errors += 1;
                }
            }
        }

        (newly_added, skipped, errors)
    });

    let mut to_process = Vec::new();
    while let Some(item) = rx.recv().await {
        to_process.push(item);
    }

    let (newly_added, skipped, errors) = scan_task.await?;
    summary.newly_added = newly_added;
    summary.skipped_dup = skipped;
    summary.errors = errors;

    let total_to_process = to_process.len();
    let app_handle = app.clone();

    let mut ai_processed = 0usize;
    for (path, media_id, is_video) in to_process {
        if abort_sync.load(std::sync::atomic::Ordering::SeqCst) {
            crate::log_info!("🛑 AI processing loop aborted");
            break;
        }

        let path_str = path.to_string_lossy().to_string();
        let file_name_only = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        crate::log_info!("🤖 [AI {}/{}] Processing: {}", ai_processed + 1, total_to_process, file_name_only);

        if is_video {
            let cache_ref = thumb_cache_dir.as_deref();
            match video_processor::process_video(&path_str, &media_id, &db, &engine, cache_ref).await {
                Ok(Some(thumb)) => crate::log_info!("🎥 Video done, thumbnail: {}", thumb),
                Ok(None)        => crate::log_info!("🎥 Video done (no thumbnail)"),
                Err(e)          => crate::log_warn!("🎥 Video pipeline error for {}: {}", file_name_only, e),
            }
        } else {
            process_image_file(&path_str, &media_id, &file_name_only, &db, &engine).await;
        }

        ai_processed += 1;

        if let Some(ref app) = app_handle {
            if let Err(e) = app.emit(
                "ingest-progress",
                &IngestProgress {
                    processed: ai_processed,
                    total: total_to_process,
                    current_file: file_name_only.clone(),
                },
            ) {
                crate::log_warn!("⚠️ Failed to emit ingest-progress: {}", e);
            }
        }
    }

    crate::log_info!("✅ Ingest complete: {} new, {} skipped, {} errors, {} AI processed",
        newly_added, skipped, errors, ai_processed);

    Ok(summary)
}

pub async fn ingest_files(
    file_paths: Vec<String>,
    dest_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    thumb_cache_dir: Option<PathBuf>,
) -> Result<IngestSummary> {
    let dest_path = Path::new(&dest_dir);
    if !dest_path.exists() {
        return Err(anyhow::anyhow!("Destination directory not found: {}", dest_dir));
    }

    let mut summary = IngestSummary {
        total_found: file_paths.len(),
        newly_added: 0,
        skipped_dup: 0,
        errors: 0,
    };

    for src_path_str in &file_paths {
        let src = Path::new(src_path_str);
        let file_name = match src.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => { summary.errors += 1; continue; }
        };

        let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let is_video = VIDEO_EXTENSIONS.contains(&ext.as_str());
        let is_image = IMAGE_EXTENSIONS.contains(&ext.as_str());
        if !is_image && !is_video {
            crate::log_warn!("⚠️ Skipping unsupported file: {}", file_name);
            summary.skipped_dup += 1;
            continue;
        }
        let media_type = if is_video { "video" } else { "image" };

        let dest = dest_path.join(&file_name);
        if src.canonicalize().ok() != dest.canonicalize().ok() {
            if let Err(e) = std::fs::copy(src, &dest) {
                crate::log_warn!("⚠️ Failed to copy {} -> {}: {}", src_path_str, dest.display(), e);
                summary.errors += 1;
                continue;
            }
        }

        match scan_single_file(&dest, &db, &dest_dir, media_type).await {
            Ok(Some(media_id)) => {
                crate::log_info!("📎 Copied+ingested: {} ({}) as {}", file_name, media_id, media_type);
                summary.newly_added += 1;

                let dest_str = dest.to_string_lossy().to_string();

                if is_video {
                    let cache_ref = thumb_cache_dir.as_deref();
                    if let Err(e) = video_processor::process_video(&dest_str, &media_id, &db, &engine, cache_ref).await {
                        crate::log_warn!("🎥 Video pipeline error for {}: {}", file_name, e);
                    }
                } else {
                    process_image_file(&dest_str, &media_id, &file_name, &db, &engine).await;
                }
            }
            Ok(None) => { summary.skipped_dup += 1; }
            Err(e) => {
                crate::log_warn!("⚠️ Error ingesting {}: {}", file_name, e);
                summary.errors += 1;
            }
        }
    }

    Ok(summary)
}

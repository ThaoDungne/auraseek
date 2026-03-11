/// Image folder ingest pipeline – SurrealDB edition
/// Thread 1: scan files → insert media stubs (with sha256 dedup)
/// Thread 2: AI processing queue → update with objects/faces/embedding
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use sha2::{Sha256, Digest};

use crate::db::{SurrealDb, DbOperations};
use crate::db::models::{MediaDoc, FileInfo, MediaMetadata, ObjectEntry, FaceEntry, Bbox, PersonDoc};
use crate::processor::AuraSeekEngine;
use tauri::Emitter;
use crate::processor::pipeline::EngineOutput;
use crate::ingest::video_ingest;

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "heic", "avif"];
pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm", "m4v", "flv", "wmv"];

/// Scan a source folder and ingest all images/videos.
/// If `thumb_cache_dir` is Some, video/face thumbnails are written there instead of next to the source files.
/// If `app` is Some, an `"ingest-progress"` Tauri event is emitted after each file is processed so
/// the frontend can refresh the timeline progressively.
pub async fn ingest_folder(
    source_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    app: Option<tauri::AppHandle>,
    thumb_cache_dir: Option<std::path::PathBuf>,
) -> Result<IngestSummary> {
    let source_path = Path::new(&source_dir);
    if !source_path.exists() {
        crate::log_error!("Source directory not found: {}", source_dir);
        return Err(anyhow::anyhow!("Source directory not found: {}", source_dir));
    }

    // Persist source_dir to config_auraseek table
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

    // Channel carries (original_path, media_id, is_video)
    let (tx, mut rx) = mpsc::channel::<(PathBuf, String, bool)>(64);

    let db_scan = db.clone();
    let source_dir_clone = source_dir.clone();

    // Thread 1: file scan + insert media stubs (images then videos)
    let scan_task = tokio::spawn(async move {
        let mut newly_added = 0usize;
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for path in image_files {
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

    // Collecting exactly what needs to be processed to accurately output the counts
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

    // Thread 2: AI processing + embedding
    let mut ai_processed = 0usize;
    for (path, media_id, is_video) in to_process {
        let path_str = path.to_string_lossy().to_string();
        let file_name_only = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        crate::log_info!("🤖 [AI {}/{}] Processing: {}", ai_processed + 1, total_to_process, file_name_only);
        let _t0 = std::time::Instant::now();

        if is_video {
            // ── Full video pipeline: scene detection → AI → embeddings ───────
            let cache_ref = thumb_cache_dir.as_deref();
            match video_ingest::process_video(&path_str, &media_id, &db, &engine, cache_ref).await {
                Ok(Some(thumb)) => crate::log_info!("🎥 Video done, thumbnail: {}", thumb),
                Ok(None)        => crate::log_info!("🎥 Video done (no thumbnail)"),
                Err(e)          => crate::log_warn!("🎥 Video pipeline error for {}: {}", file_name_only, e),
            }
        } else {
            // ── Standard image pipeline ──────────────────────────────────────
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

    crate::log_info!("✅ Ingest complete: {} new, {} skipped, {} errors, {} AI processed (images+videos)",
        newly_added, skipped, errors, ai_processed);

    Ok(summary)
}

/// Shared utility: run the full AI pipeline on a single image path.
/// Returns the raw EngineOutput without writing anything to the database.
/// Both the image pipeline and video frame pipeline call this.
pub async fn analyze_image_raw(
    path_str: &str,
    engine: &Arc<Mutex<Option<AuraSeekEngine>>>,
) -> Option<EngineOutput> {
    let mut eng_guard = engine.lock().await;
    let eng = eng_guard.as_mut()?;
    match eng.process_image(path_str) {
        Ok(output) => Some(output),
        Err(e) => {
            crate::log_warn!("🤖 AI error for {}: {}", path_str, e);
            None
        }
    }
}

pub async fn process_image_file(
    path_str: &str,
    media_id: &str,
    file_name_only: &str,
    db: &Arc<Mutex<Option<SurrealDb>>>,
    engine: &Arc<Mutex<Option<AuraSeekEngine>>>,
) {
    let t0 = std::time::Instant::now();
    let output = match analyze_image_raw(path_str, engine).await {
        Some(o) => o,
        None => return,
    };

    crate::log_info!(
        "  ✅ Done in {}ms | objects={} faces={} embed_dims={}",
        t0.elapsed().as_millis(),
        output.objects.len(), output.faces.len(), output.vision_embedding.len()
    );

    let objects: Vec<ObjectEntry> = output.objects.iter().map(|o| ObjectEntry {
        class_name: o.class_name.clone(),
        conf: o.conf,
        bbox: Bbox {
            x: o.bbox[0], y: o.bbox[1],
            w: o.bbox[2] - o.bbox[0],
            h: o.bbox[3] - o.bbox[1],
        },
        mask_area: Some(o.mask_area),
        mask_path: None,
        mask_rle: Some(o.mask_rle.iter().map(|&(a, b)| [a, b]).collect()),
    }).collect();

    let faces: Vec<FaceEntry> = output.faces.iter().map(|f| FaceEntry {
        face_id: f.face_id.clone(),
        name: f.name.clone(),
        conf: f.conf,
        bbox: Bbox {
            x: f.bbox[0], y: f.bbox[1],
            w: f.bbox[2] - f.bbox[0],
            h: f.bbox[3] - f.bbox[1],
        },
    }).collect();

    let detected_faces: Vec<(String, f32, Bbox)> = faces.iter().map(|f| (
        f.face_id.clone(), f.conf,
        Bbox { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
    )).collect();

    let db_guard = db.lock().await;
    if let Some(ref sdb) = *db_guard {
        if let Err(e) = DbOperations::update_media_ai(sdb, media_id, objects, faces, None).await {
            crate::log_warn!("⚠️ update_media_ai failed for {}: {}", media_id, e);
        }
        if !output.vision_embedding.is_empty() {
            if let Err(e) = DbOperations::insert_embedding(
                sdb, media_id, "image", None, None, output.vision_embedding
            ).await {
                crate::log_warn!("⚠️ insert_embedding failed for {}: {}", media_id, e);
            }
        }
        for (fid, conf, bbox) in &detected_faces {
            crate::log_info!("  👤 Upserting person face_id={} conf={:.3}", fid, conf);
            if let Err(e) = DbOperations::upsert_person(sdb, PersonDoc {
                face_id: fid.clone(),
                name: None,
                thumbnail: Some(file_name_only.to_string()),
                conf: Some(*conf as f32),
                face_bbox: Some(bbox.clone() as Bbox),
            }).await {
                crate::log_warn!("⚠️ upsert_person failed for {}: {}", fid, e);
            }
        }
    }
}

pub async fn scan_single_file(
    path: &Path,
    db: &Arc<Mutex<Option<SurrealDb>>>,
    _source_dir: &str,
    media_type: &str,
) -> Result<Option<String>> {
    let sha256 = compute_sha256(path)?;

    // Trước đây ở đây có logic dedup theo SHA-256:
    // - nếu file có cùng sha256 và đã processed => bỏ qua (skip)
    // - nếu có nhưng chưa processed => dùng lại media_id cũ
    // Bây giờ yêu cầu là VẪN CHO PHÉP lưu các ảnh giống nhau hoàn toàn,
    // nên bỏ hẳn bước kiểm tra này. Mỗi file tìm thấy sẽ tạo một media record mới,
    // còn chức năng "Khử trùng lặp" sẽ xử lý việc gộp/hiển thị sau.

    let meta = std::fs::metadata(path)?;
    let size = meta.len();
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    {
        let db_guard = db.lock().await;
        if let Some(ref sdb) = *db_guard {
            if let Ok(Some((media_id, processed))) = DbOperations::check_exact_file(sdb, &name, &sha256).await {
                if processed {
                    return Ok(None); // Skip, exact file already processed
                } else {
                    return Ok(Some(media_id)); // Already in queue/database but not yet processed
                }
            }
        }
    }

    let path_str = path.to_string_lossy().to_string(); // used for image dimension reading only

    let (width, height) = if media_type == "image" {
        get_image_dimensions(&path_str)
    } else {
        (None, None)
    };

    let modified_at = meta.modified().ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| surrealdb::types::Datetime::from_timestamp(d.as_secs() as i64, 0));

    let doc = MediaDoc {
        media_type: media_type.to_string(),
        file: FileInfo {
            name,
            size,
            sha256,
            phash: None,
        },
        metadata: MediaMetadata {
            width,
            height,
            duration: None,
            fps: None,
            created_at: modified_at.clone(),
            modified_at,
        },
        objects: vec![],
        faces: vec![],
        thumbnail: None,
        processed: false,
        deleted_at: None,
        is_hidden: false,
    };

    let db_guard = db.lock().await;
    let sdb = db_guard.as_ref().ok_or_else(|| anyhow::anyhow!("DB not connected"))?;
    let media_id = DbOperations::insert_media(sdb, doc).await?;
    Ok(Some(media_id))
}


fn collect_files(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut images = vec![];
    let mut videos = vec![];
    collect_files_recursive(dir, &mut images, &mut videos);
    (images, videos)
}

fn collect_files_recursive(dir: &Path, images: &mut Vec<PathBuf>, videos: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, images, videos);
        } else {
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if IMAGE_EXTENSIONS.contains(&ext.as_str()) && !fname.ends_with(".thumb.jpg") && !fname.ends_with(".debug.jpg") {
                images.push(path);
            } else if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
                videos.push(path);
            }
        }
    }
    images.sort();
    videos.sort();
}

fn compute_sha256(path: &Path) -> Result<String> {
    let data = std::fs::read(path)?;
    let hash = Sha256::digest(&data);
    Ok(hex::encode(hash))
}

fn get_image_dimensions(path: &str) -> (Option<u32>, Option<u32>) {
    use image::io::Reader as ImageReader;
    if let Ok(reader) = ImageReader::open(path) {
        if let Ok(dims) = reader.into_dimensions() {
            return (Some(dims.0), Some(dims.1));
        }
    }
    (None, None)
}

/// Ingest a specific list of image files (for copy/paste or drag-drop).
/// Files are copied to `dest_dir` (source_dir) then processed.
/// If `thumb_cache_dir` is Some, video/face thumbnails are written there instead of next to the source files.
pub async fn ingest_files(
    file_paths: Vec<String>,
    dest_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    thumb_cache_dir: Option<std::path::PathBuf>,
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

        // Copy to dest_dir — skip if the file is already there (e.g. pasted via clipboard save)
        let dest = dest_path.join(&file_name);
        if src.canonicalize().ok() != dest.canonicalize().ok() {
            if let Err(e) = std::fs::copy(src, &dest) {
                crate::log_warn!("⚠️ Failed to copy {} -> {}: {}", src_path_str, dest.display(), e);
                summary.errors += 1;
                continue;
            }
        }

        // Ingest the copied/moved file
        match scan_single_file(&dest, &db, &dest_dir, media_type).await {
            Ok(Some(media_id)) => {
                crate::log_info!("📎 Copied+ingested: {} ({}) as {}", file_name, media_id, media_type);
                summary.newly_added += 1;

                let dest_str = dest.to_string_lossy().to_string();
                let t0 = std::time::Instant::now();

                if is_video {
                    // ── Full video pipeline ─────────────────────────────────
                    let cache_ref = thumb_cache_dir.as_deref();
                    if let Err(e) = video_ingest::process_video(&dest_str, &media_id, &db, &engine, cache_ref).await {
                        crate::log_warn!("🎥 Video pipeline error for {}: {}", file_name, e);
                    }
                } else {
                    // ── Standard image pipeline ─────────────────────────────
                    let mut eng_guard = engine.lock().await;
                    if let Some(ref mut eng) = *eng_guard {
                        match eng.process_image(&dest_str) {
                            Ok(output) => {
                                crate::log_info!(
                                    "  ✅ AI done in {}ms | objects={} faces={}",
                                    t0.elapsed().as_millis(), output.objects.len(), output.faces.len()
                                );
                                let objects: Vec<ObjectEntry> = output.objects.iter().map(|o| ObjectEntry {
                                    class_name: o.class_name.clone(), conf: o.conf,
                                    bbox: Bbox { x: o.bbox[0], y: o.bbox[1], w: o.bbox[2]-o.bbox[0], h: o.bbox[3]-o.bbox[1] },
                                    mask_area: Some(o.mask_area), mask_path: None,
                                    mask_rle: Some(o.mask_rle.iter().map(|&(a,b)| [a,b]).collect()),
                                }).collect();
                                let faces: Vec<FaceEntry> = output.faces.iter().map(|f| FaceEntry {
                                    face_id: f.face_id.clone(), name: f.name.clone(), conf: f.conf,
                                    bbox: Bbox { x: f.bbox[0], y: f.bbox[1], w: f.bbox[2]-f.bbox[0], h: f.bbox[3]-f.bbox[1] },
                                }).collect();
                                let detected_faces: Vec<(String, f32, Bbox)> = faces.iter().map(|f| (
                                    f.face_id.clone(), f.conf,
                                    Bbox { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
                                )).collect();
                                drop(eng_guard);

                                let db_guard = db.lock().await;
                                if let Some(ref sdb) = *db_guard {
                                    let _ = DbOperations::update_media_ai(sdb, &media_id, objects, faces, None).await;
                                    if !output.vision_embedding.is_empty() {
                                        let _ = DbOperations::insert_embedding(sdb, &media_id, "image", None, None, output.vision_embedding).await;
                                    }
                                    for (fid, conf, bbox) in &detected_faces {
                                        let _ = DbOperations::upsert_person(sdb, PersonDoc {
                                            face_id: fid.clone(), name: None,
                                            thumbnail: Some(file_name.clone()),
                                            conf: Some(*conf), face_bbox: Some(bbox.clone()),
                                        }).await;
                                    }
                                }
                            }
                            Err(e) => {
                                drop(eng_guard);
                                crate::log_warn!("🤖 AI error for {}: {}", file_name, e);
                            }
                        }
                    } else {
                        drop(eng_guard);
                    }
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

// ─── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct IngestProgress {
    pub processed:    usize,
    pub total:        usize,
    pub current_file: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct IngestSummary {
    pub total_found: usize,
    pub newly_added: usize,
    pub skipped_dup: usize,
    pub errors:      usize,
}

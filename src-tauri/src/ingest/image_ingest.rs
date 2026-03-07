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

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "heic", "avif"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm", "m4v", "flv", "wmv"];

/// Scan a source folder and ingest all images/videos.
pub async fn ingest_folder(
    source_dir: String,
    db: Arc<Mutex<Option<SurrealDb>>>,
    engine: Arc<Mutex<Option<AuraSeekEngine>>>,
    progress_tx: Option<tauri::ipc::Channel<IngestProgress>>,
) -> Result<IngestSummary> {
    let source_path = Path::new(&source_dir);
    if !source_path.exists() {
        crate::log_error!("Source directory not found: {}", source_dir);
        return Err(anyhow::anyhow!("Source directory not found: {}", source_dir));
    }

    let (image_files, _video_files) = collect_files(source_path);
    let total = image_files.len();
    crate::log_info!("📂 Ingest started: {} | {} images found", source_dir, total);

    let mut summary = IngestSummary {
        total_found: total,
        newly_added: 0,
        skipped_dup: 0,
        errors: 0,
    };

    let (tx, mut rx) = mpsc::channel::<(PathBuf, String)>(64);

    let db_scan = db.clone();
    let source_dir_clone = source_dir.clone();

    // Thread 1: file scan + insert media stubs
    let scan_task = tokio::spawn(async move {
        let mut newly_added = 0usize;
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for path in image_files {
            match scan_single_file(&path, &db_scan, &source_dir_clone, "image").await {
                Ok(Some(media_id)) => {
                    newly_added += 1;
                    let _ = tx.send((path, media_id)).await;
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

    // Thread 2: AI processing + embedding
    let mut ai_processed = 0usize;
    while let Some((path, media_id)) = rx.recv().await {
        let path_str = path.to_string_lossy().to_string();
        let mut eng_guard = engine.lock().await;
        let eng = match eng_guard.as_mut() {
            Some(e) => e,
            None => { drop(eng_guard); continue; }
        };

        match eng.process_image(&path_str) {
            Ok(output) => {
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
                    f.face_id.clone(),
                    f.conf,
                    Bbox { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
                )).collect();

                drop(eng_guard);

                // Update AI results in DB
                {
                    let db_guard = db.lock().await;
                    if let Some(ref sdb) = *db_guard {
                        if let Err(e) = DbOperations::update_media_ai(sdb, &media_id, objects, faces).await {
                            crate::log_warn!("⚠️ update_media_ai failed for {}: {}", media_id, e);
                        }

                        if !output.vision_embedding.is_empty() {
                            if let Err(e) = DbOperations::insert_embedding(
                                sdb, &media_id, "image", None, None, output.vision_embedding.clone()
                            ).await {
                                crate::log_warn!("⚠️ insert_embedding failed for {}: {}", media_id, e);
                            }
                        }

                        for (fid, conf, bbox) in &detected_faces {
                            if let Err(e) = DbOperations::upsert_person(sdb, PersonDoc {
                                face_id: fid.clone(),
                                name: None,
                                thumbnail: Some(path_str.clone()),
                                conf: Some(*conf),
                                face_bbox: Some(bbox.clone()),
                            }).await {
                                crate::log_warn!("⚠️ upsert_person failed for {}: {}", fid, e);
                            }
                        }
                    }
                }

                ai_processed += 1;
            }
            Err(e) => {
                drop(eng_guard);
                crate::log_warn!("🤖 AI error for {}: {}", path_str, e);
            }
        }

        if let Some(ref ptx) = progress_tx {
            let _ = ptx.send(IngestProgress {
                processed: ai_processed,
                total,
                current_file: path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
            });
        }
    }

    let (newly_added, skipped, errors) = scan_task.await?;
    summary.newly_added = newly_added;
    summary.skipped_dup = skipped;
    summary.errors = errors;

    crate::log_info!("✅ Ingest complete: {} new, {} skipped, {} errors, {} AI processed",
        newly_added, skipped, errors, ai_processed);

    Ok(summary)
}

async fn scan_single_file(
    path: &Path,
    db: &Arc<Mutex<Option<SurrealDb>>>,
    source_dir: &str,
    media_type: &str,
) -> Result<Option<String>> {
    let sha256 = compute_sha256(path)?;

    // Dedup check
    {
        let db_guard = db.lock().await;
        let sdb = db_guard.as_ref().ok_or_else(|| anyhow::anyhow!("DB not connected"))?;
        if DbOperations::is_duplicate_sha256(sdb, &sha256).await? {
            return Ok(None);
        }
    }

    let meta = std::fs::metadata(path)?;
    let size = meta.len();
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let path_str = path.to_string_lossy().to_string();

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
        source: source_dir.to_string(),
        file: FileInfo {
            path: path_str,
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
        processed: false,
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
            if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
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

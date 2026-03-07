/// Video processing pipeline – SurrealDB edition (placeholder)
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::db::{SurrealDb, DbOperations};
use crate::processor::AuraSeekEngine;
use crate::processor::vision::preprocess_aura;

/// Detect scenes and process keyframes for a video.
pub async fn process_video_scenes(
    video_path: &str,
    media_id: &str,
    db: &Arc<Mutex<Option<SurrealDb>>>,
    engine: &Arc<Mutex<Option<AuraSeekEngine>>>,
) -> Result<()> {
    let frames = extract_keyframes(video_path)?;

    for (frame_index, frame_path) in frames.iter().enumerate() {
        let mut eng_guard = engine.lock().await;
        let eng = match eng_guard.as_mut() {
            Some(e) => e,
            None => { drop(eng_guard); continue; }
        };

        match preprocess_aura(frame_path) {
            Ok(blob) => {
                if let Ok(embedding) = eng.aura.encode_image(blob, 256, 256) {
                    drop(eng_guard);

                    let db_guard = db.lock().await;
                    if let Some(ref sdb) = *db_guard {
                        let _ = DbOperations::insert_embedding(
                            sdb, media_id, "video_frame",
                            None, Some(frame_index as u32),
                            embedding,
                        ).await;
                    }
                } else {
                    drop(eng_guard);
                }
            }
            Err(_) => { drop(eng_guard); }
        }

        let _ = std::fs::remove_file(frame_path);
    }

    Ok(())
}

/// Extract keyframes from video (placeholder – uses image crate for GIF only)
fn extract_keyframes(video_path: &str) -> Result<Vec<String>> {
    let temp_dir = std::env::temp_dir().join("auraseek_frames");
    std::fs::create_dir_all(&temp_dir)?;

    let path = std::path::Path::new(video_path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    let mut frame_paths = vec![];

    if ext == "gif" {
        if let Ok(data) = std::fs::read(video_path) {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("vid");
            if let Ok(img) = image::load_from_memory(&data) {
                let frame_path = temp_dir.join(format!("{}_f0.jpg", stem));
                let _ = img.save(&frame_path);
                frame_paths.push(frame_path.to_string_lossy().to_string());
            }
        }
    } else {
        crate::log_warn!("🎥 Native video decoding not yet available for: {}", video_path);
    }

    Ok(frame_paths)
}

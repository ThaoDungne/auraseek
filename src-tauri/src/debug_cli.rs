//! Debug CLI — full pipeline simulation, writing per-image/per-video artifact folders.
//!
//! For each IMAGE → output/<stem>/
//!   embeddings.json, detections.json, faces.json, face_similarity.json,
//!   similarity_vs_others.json, det_seg.jpg, det_faces.jpg,
//!   mask_*.png, yolo_crop_*.jpg, face_crop_*.jpg, face_aligned_*.jpg
//!
//! For each VIDEO → output/<stem>/
//!   frames/
//!     <scene>_<frame>/   (same structure as per-image above)
//!   scene_summary.json  – list of scenes + selected frames + timestamps

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use serde_json::json;
use opencv::{
    core::{Mat, Rect, Vector},
    imgcodecs::{imread, imwrite, IMREAD_COLOR},
    prelude::*,
};

use crate::processor::AuraSeekEngine;
use crate::processor::vision::{cosine_similarity, letterbox_640, preprocess_aura};
use crate::processor::vision::yolo_postprocess::{YoloProcessor, DetectionRecord};
use crate::ingest::video_ingest::{probe_video, detect_scenes, extract_frame, is_good_brightness};
use crate::ingest::image_ingest::{IMAGE_EXTENSIONS, VIDEO_EXTENSIONS};
use crate::utils::visualize::{
    draw_detections, draw_faces, draw_segmentation, extract_masks, load_rgb, save_rgb,
};
use crate::utils::{BOLD, CYAN, GREEN, MAGENTA, RESET};

const FONT_PATH: Option<&'static str> = Some("assets/fonts/DejaVuSans.ttf");

// ─── Public entry ─────────────────────────────────────────────────────────────

/// For every image+video in `input_dir`, run the full pipeline and write debug artifacts to `output_dir`.
pub fn run_debug_ingest(input_dir: &str, output_dir: &str) -> Result<()> {
    crate::log_info!("🛠️  Debug CLI — full pipeline simulation");
    crate::log_info!("  input : {}", input_dir);
    crate::log_info!("  output: {}", output_dir);

    std::fs::create_dir_all(input_dir)?;
    std::fs::create_dir_all(output_dir)?;

    let mut engine = AuraSeekEngine::new_default()?;

    // Collect both images and videos
    let mut image_entries: Vec<PathBuf> = Vec::new();
    let mut video_entries: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(input_dir)?.filter_map(|e| e.ok()) {
        let path = entry.path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            image_entries.push(path);
        } else if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
            video_entries.push(path);
        }
    }
    image_entries.sort();
    video_entries.sort();

    let total = image_entries.len() + video_entries.len();
    if total == 0 {
        crate::log_warn!("⚠️  No images or videos found in {}", input_dir);
        return Ok(());
    }
    crate::log_info!("📂 Found {} image(s) + {} video(s)", image_entries.len(), video_entries.len());
    let total_start = Instant::now();

    // Collect (stem, embedding) for cross-image similarity (images only)
    let mut all_embeddings: Vec<(String, Vec<f32>)> = Vec::new();
    let total_count = image_entries.len() + video_entries.len();
    let mut idx = 0usize;

    // ── Process images ────────────────────────────────────────────────────────
    for path in &image_entries {
        idx += 1;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        crate::log_info!("{BOLD}{CYAN}[{}/{}] 🖼  {}{RESET}", idx, total_count, name);
        let t = Instant::now();
        match process_one(&mut engine, path, output_dir) {
            Ok(emb) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                all_embeddings.push((stem, emb));
                crate::log_info!("  {MAGENTA}elapsed: {:?}{RESET}", t.elapsed());
            }
            Err(e) => crate::log_warn!("  ⚠️  Failed: {}", e),
        }
    }

    // ── Process videos ────────────────────────────────────────────────────────
    for path in &video_entries {
        idx += 1;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        crate::log_info!("{BOLD}{CYAN}[{}/{}] 🎥  {}{RESET}", idx, total_count, name);
        let t = Instant::now();
        if let Err(e) = process_video_debug(&mut engine, path, output_dir) {
            crate::log_warn!("  ⚠️  Video failed: {}", e);
        } else {
            crate::log_info!("  {MAGENTA}elapsed: {:?}{RESET}", t.elapsed());
        }
    }

    // ── Per-image similarity (images only) ────────────────────────────────────
    if all_embeddings.len() > 1 {
        for i in 0..all_embeddings.len() {
            let (stem_i, emb_i) = &all_embeddings[i];
            let mut comparisons: Vec<serde_json::Value> = all_embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (stem_j, emb_j))| {
                    let sim = cosine_similarity(emb_i, emb_j);
                    json!({ "vs_image": stem_j, "cosine_similarity": sim })
                })
                .collect();
            comparisons.sort_by(|a, b| {
                b["cosine_similarity"].as_f64().unwrap_or(0.0)
                    .partial_cmp(&a["cosine_similarity"].as_f64().unwrap_or(0.0))
                    .unwrap()
            });
            let out_path = format!("{}/{}/similarity_vs_others.json", output_dir, stem_i);
            let _ = std::fs::write(&out_path, serde_json::to_string_pretty(&comparisons)?);
        }
        crate::log_info!("📊 Per-image similarity files written");
    }

    crate::log_info!("{BOLD}{GREEN}✅ All done in {:?}{RESET}", total_start.elapsed());
    Ok(())
}

// ─── Per-video pipeline ───────────────────────────────────────────────────────

fn process_video_debug(engine: &mut AuraSeekEngine, path: &Path, output_base: &str) -> Result<()> {
    let video_path = path.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("vid");

    // Video folder + frames sub-folder
    let video_out = format!("{}/{}", output_base, stem);
    let frames_out = format!("{}/frames", video_out);
    std::fs::create_dir_all(&frames_out)?;

    // ── Probe + scene detection (same logic as video_ingest) ──────────────────
    let (fps, total_frames) = probe_video(video_path)?;
    crate::log_info!("  fps={:.2} total_frames={}", fps, total_frames);
    if total_frames == 0 {
        crate::log_warn!("  ⚠️  Could not determine frame count");
        return Ok(());
    }

    let cuts = detect_scenes(video_path, fps)?;
    crate::log_info!("  🎬 {} scenes detected", cuts.len() + 1);

    let mut scenes: Vec<(u64, u64)> = Vec::new();
    let mut prev = 0u64;
    for cut in &cuts {
        scenes.push((prev, cut.saturating_sub(1)));
        prev = *cut;
    }
    scenes.push((prev, total_frames.saturating_sub(1)));

    // ── Extract frames (same 20/50/80 % logic + brightness filter) ───────────
    let mut scene_summary: Vec<serde_json::Value> = Vec::new();
    let mut frame_jobs: Vec<(usize, u64, PathBuf)> = Vec::new(); // (scene_idx, frame_idx, path)

    for (s_idx, (start, end)) in scenes.iter().enumerate() {
        let len = end.saturating_sub(*start);
        let candidates = [
            start + (len as f64 * 0.2) as u64,
            start + (len as f64 * 0.5) as u64,
            start + (len as f64 * 0.8) as u64,
        ];

        let mut seen = std::collections::HashSet::new();
        let mut scene_frames = Vec::new();
        for fi in candidates {
            if seen.insert(fi) {
                let out = Path::new(&frames_out).join(format!("s{:02}_f{}.jpg", s_idx, fi));
                if extract_frame(video_path, fi, fps, &out).is_ok() {
                    let (is_good, luma) = is_good_brightness(&out);
                    scene_frames.push((fi, is_good, luma, out));
                }
            }
        }

        // Keep only good-brightness frames; fallback to closest-to-128 luma
        let mut keep: Vec<u64> = scene_frames.iter()
            .filter(|(_, good, _, _)| *good)
            .map(|(fi, _, _, _)| *fi)
            .collect();
        if keep.is_empty() && !scene_frames.is_empty() {
            let best = scene_frames.iter()
                .min_by(|a, b| (a.2 - 128.0).abs().partial_cmp(&(b.2 - 128.0).abs()).unwrap())
                .unwrap();
            keep.push(best.0);
        }
        // Delete rejected frames
        for (fi, _, _, p) in &scene_frames {
            if !keep.contains(fi) { let _ = std::fs::remove_file(p); }
        }

        let kept_timestamps: Vec<f64> = keep.iter().map(|fi| *fi as f64 / fps).collect();
        scene_summary.push(json!({
            "scene_idx": s_idx,
            "start_frame": start,
            "end_frame": end,
            "start_time": *start as f64 / fps,
            "end_time": *end as f64 / fps,
            "kept_frames": keep,
            "kept_timestamps": kept_timestamps,
        }));

        for fi in keep {
            let p = Path::new(&frames_out).join(format!("s{:02}_f{}.jpg", s_idx, fi));
            frame_jobs.push((s_idx, fi, p));
        }
    }

    // Write scene summary
    std::fs::write(
        format!("{}/scene_summary.json", video_out),
        serde_json::to_string_pretty(&scene_summary)?,
    )?;

    // ── Per-frame AI pipeline ─────────────────────────────────────────────────
    crate::log_info!("  🖼️  {} frames to process", frame_jobs.len());
    for (s_idx, fi, frame_path) in &frame_jobs {
        let frame_folder = format!("{}/s{:02}_f{}", frames_out, s_idx, fi);
        std::fs::create_dir_all(&frame_folder)?;

        // Copy/move the extracted frame into its sub-folder for process_one to read
        let dest = Path::new(&frame_folder).join(format!("s{:02}_f{}.jpg", s_idx, fi));
        if let Err(e) = std::fs::copy(frame_path, &dest) {
            crate::log_warn!("  ⚠️  copy frame: {}", e);
            continue;
        }

        crate::log_info!("  → frame {}/{} @ {:.2}s", fi, total_frames, *fi as f64 / fps);
        match process_one(engine, &dest, &frame_folder) {
            Ok(_) => {}
            Err(e) => crate::log_warn!("    ⚠️  AI error: {}", e),
        }
    }

    crate::log_info!("  ✅ Video processed: {} frames", frame_jobs.len());
    Ok(())
}

// ─── Per-image pipeline ───────────────────────────────────────────────────────

fn process_one(engine: &mut AuraSeekEngine, path: &Path, output_base: &str) -> Result<Vec<f32>> {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let out_dir = format!("{}/{}", output_base, stem);
    std::fs::create_dir_all(&out_dir)?;
    let img_str = path.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;

    // ── Step 1: Vision embedding ──────────────────────────────────────────────
    let t = Instant::now();
    let vision_emb = engine.aura.encode_image(preprocess_aura(img_str)?, 256, 256)
        .unwrap_or_default();
    crate::log_info!("  [1/4] vision embed ({} dims) {:?}", vision_emb.len(), t.elapsed());

    // ── Step 2: YOLO ──────────────────────────────────────────────────────────
    let t = Instant::now();
    let lb = letterbox_640(img_str)?;
    let raw_yolo = engine.yolo.detect(lb.blob.clone())?;
    let objects = YoloProcessor::postprocess(&raw_yolo, &lb, 0.25, 0.45);
    crate::log_info!("  [2/4] yolo ({} objects) {:?}", objects.len(), t.elapsed());

    for (idx, o) in objects.iter().enumerate() {
        crate::log_info!(
            "    obj {:02}: {CYAN}{:<14}{RESET} conf={:.2} area={}",
            idx, o.class_name, o.conf, o.mask_area
        );
    }

    // ── Step 3: Face detection + aligned crops ────────────────────────────────
    let t = Instant::now();
    let frame_cv = imread(img_str, IMREAD_COLOR)?;
    let fs = frame_cv.size()?;
    let (fw, fh) = (fs.width, fs.height);

    let person_bboxes: Vec<[f32; 4]> = objects.iter()
        .filter(|o| o.class_name == "person")
        .map(|o| o.bbox)
        .collect();

    let detect_regions: Vec<(Mat, [f32; 4])> = if person_bboxes.is_empty() {
        vec![(frame_cv.try_clone()?, [0.0, 0.0, fw as f32, fh as f32])]
    } else {
        person_bboxes.iter().filter_map(|bbox| {
            let x1 = (bbox[0].max(0.0) as i32).min(fw - 1);
            let y1 = (bbox[1].max(0.0) as i32).min(fh - 1);
            let x2 = (bbox[2].max(0.0) as i32).min(fw);
            let y2 = (bbox[3].max(0.0) as i32).min(fh);
            let cw = x2 - x1; let ch = y2 - y1;
            if cw < 30 || ch < 30 { return None; }
            let roi = Mat::roi(&frame_cv, Rect::new(x1, y1, cw, ch)).ok()?.try_clone().ok()?;
            Some((roi, [bbox[0], bbox[1], bbox[2], bbox[3]]))
        }).collect()
    };

    let mut faces = Vec::new();
    let params = Vector::<i32>::new();

    if let Some(ref mut fm) = engine.face {
        for (n, (crop, offset)) in detect_regions.iter().enumerate() {
            // Save raw person crop
            let crop_path = format!("{out_dir}/face_crop_{n:02}.jpg");
            let _ = imwrite(&crop_path, crop, &params);

            // Detect with aligned mats
            let results = fm.detect_from_mat_with_aligned(&frame_cv, &engine.face_db)?;
            for (mut fg, aligned) in results {
                // Offset bbox if detecting from person crop
                if !person_bboxes.is_empty() {
                    fg.bbox[0] += offset[0]; fg.bbox[2] += offset[0];
                    fg.bbox[1] += offset[1]; fg.bbox[3] += offset[1];
                }
                // Save aligned 112×112 crop
                let aligned_path = format!("{out_dir}/face_aligned_{:02}_{n:02}.jpg", faces.len());
                let _ = imwrite(&aligned_path, &aligned, &params);
                faces.push(fg);
            }
        }

        // Session-based unknown grouping
        for f in faces.iter_mut() {
            if f.face_id == "unknown_placeholder" {
                let mut best_score = 0.55_f32;
                let mut cached_id = None;
                for (cached_emb, id) in &engine.session_faces {
                    let score = cosine_similarity(&f.embedding, cached_emb);
                    if score > best_score { best_score = score; cached_id = Some(id.clone()); }
                }
                if let Some(id) = cached_id {
                    f.face_id = id;
                } else {
                    let new_id = uuid::Uuid::new_v4().to_string();
                    f.face_id = new_id.clone();
                    engine.session_faces.push((f.embedding.clone(), new_id));
                }
            }
        }
    }
    crate::log_info!("  [3/4] faces ({} detected) {:?}", faces.len(), t.elapsed());
    for (idx, f) in faces.iter().enumerate() {
        crate::log_info!(
            "    face {:02}: id={}… name={:?} conf={:.2}",
            idx, &f.face_id[..8.min(f.face_id.len())], f.name, f.conf
        );
    }

    // ── Step 4: Visualise + save ──────────────────────────────────────────────
    let t = Instant::now();
    let (pixels, iw, ih) = load_rgb(img_str)?;

    // Per-object mask PNGs
    extract_masks(&objects, iw, ih, &out_dir)?;

    // Per-object raw bbox crops
    save_yolo_crops(&frame_cv, &objects, &out_dir);

    // YOLO segmentation overlay
    let mut px_seg = pixels.clone();
    draw_segmentation(&mut px_seg, iw, ih, &objects, 0.35);
    draw_detections(&mut px_seg, iw, ih, &objects, FONT_PATH);
    save_rgb(px_seg, iw, ih, &format!("{out_dir}/det_seg.jpg"))?;

    // Face overlay
    if !faces.is_empty() {
        let mut px_f = pixels.clone();
        draw_faces(&mut px_f, iw, ih, &faces, FONT_PATH);
        save_rgb(px_f, iw, ih, &format!("{out_dir}/det_faces.jpg"))?;
    }
    crate::log_info!("  [4/4] visualise + save {:?}", t.elapsed());

    // ── JSON artifacts ────────────────────────────────────────────────────────
    std::fs::write(
        format!("{out_dir}/embeddings.json"),
        serde_json::to_string_pretty(&json!({
            "dims": vision_emb.len(),
            "vector": vision_emb,
        }))?,
    )?;

    std::fs::write(
        format!("{out_dir}/detections.json"),
        serde_json::to_string_pretty(&objects)?,
    )?;

    if !faces.is_empty() {
        let faces_json: Vec<serde_json::Value> = faces.iter().map(|f| json!({
            "face_id":       f.face_id,
            "name":          f.name,
            "conf":          f.conf,
            "bbox":          f.bbox,
            "embedding_dims": f.embedding.len(),
            "embedding":     f.embedding,
        })).collect();
        std::fs::write(
            format!("{out_dir}/faces.json"),
            serde_json::to_string_pretty(&faces_json)?,
        )?;

        // Face-to-face cosine similarity matrix
        let mut sim_pairs: Vec<serde_json::Value> = Vec::new();
        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let sim = cosine_similarity(&faces[i].embedding, &faces[j].embedding);
                sim_pairs.push(json!({
                    "face_a": faces[i].face_id,
                    "face_b": faces[j].face_id,
                    "cosine_similarity": sim,
                }));
            }
        }
        std::fs::write(
            format!("{out_dir}/face_similarity.json"),
            serde_json::to_string_pretty(&sim_pairs)?,
        )?;
    }

    Ok(vision_emb)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Save a raw bbox crop for each YOLO detection.
fn save_yolo_crops(frame: &Mat, objects: &[DetectionRecord], out_dir: &str) {
    let Ok(s) = frame.size() else { return };
    let (fw, fh) = (s.width, s.height);
    let params = Vector::<i32>::new();
    for (n, obj) in objects.iter().enumerate() {
        let x1 = (obj.bbox[0].max(0.0) as i32).min(fw - 1);
        let y1 = (obj.bbox[1].max(0.0) as i32).min(fh - 1);
        let x2 = (obj.bbox[2].max(0.0) as i32).min(fw);
        let y2 = (obj.bbox[3].max(0.0) as i32).min(fh);
        let cw = x2 - x1; let ch = y2 - y1;
        if cw < 2 || ch < 2 { continue; }
        if let Ok(roi) = Mat::roi(frame, Rect::new(x1, y1, cw, ch)) {
            if let Ok(crop) = roi.try_clone() {
                let path = format!("{out_dir}/yolo_crop_{n:02}_{}.jpg", obj.class_name);
                let _ = imwrite(&path, &crop, &params);
            }
        }
    }
}

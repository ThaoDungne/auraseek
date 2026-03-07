//! Debug CLI: save images, files, and vectors for inspection.
//! Used only when running with `run_cli_debug_ingest = true` in main.
//! Normal app (React) uses only `AuraSeekEngine::process_image()` and never writes these artifacts.

use std::time::Instant;
use anyhow::Result;
use serde_json::json;

use crate::processor::AuraSeekEngine;
use crate::utils::visualize::{
    draw_detections, draw_faces, draw_segmentation, extract_masks, load_rgb, save_rgb,
};
use crate::utils::{BOLD, CYAN, GREEN, MAGENTA, RED, RESET, YELLOW};

const FONT_PATH: Option<&'static str> = Some("assets/fonts/DejaVuSans.ttf");

/// Run debug ingest: for each image in `input_dir`, run the AI pipeline and write
/// to `output_dir/<stem>/`: embeddings.json, detections.json, mask_*.png,
/// det_seg.jpg, faces.json, det_faces.jpg.
pub fn run_debug_ingest(input_dir: &str, output_dir: &str) -> Result<()> {
    crate::log_info!("🛠️ Starting debug cli ingest mode...");
    crate::log_info!("Input dir: {}", input_dir);
    crate::log_info!("Output dir: {}", output_dir);

    std::fs::create_dir_all(input_dir)?;
    std::fs::create_dir_all(output_dir)?;

    let mut engine = AuraSeekEngine::new_default()?;

    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let ext = p
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            ["jpg", "jpeg", "png", "bmp", "webp", "tiff"].contains(&ext.as_str())
        })
        .collect();

    entries.sort();

    if entries.is_empty() {
        crate::log_warn!("no images found in directory: {}", input_dir);
        return Ok(());
    }

    crate::log_info!("found {} images in {}", entries.len(), input_dir);
    let total_start = Instant::now();

    for (i, path) in entries.iter().enumerate() {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let step_msg = format!(
            "{BOLD}{CYAN}step {}/{}{RESET} | processing: {BOLD}{GREEN}{}{RESET}",
            i + 1,
            entries.len(),
            name
        );
        crate::log_info!("{}", step_msg);

        let start = Instant::now();
        if let Err(e) = process_and_save_debug(&mut engine, path, output_dir) {
            crate::log_warn!("failed to process {}: {}", path.display(), e);
        }
        crate::log_info!("  - {MAGENTA}step duration: {:?}{RESET}", start.elapsed());
    }

    crate::log_info!(
        "{BOLD}{GREEN}all tasks completed successfully in {:?}{RESET}",
        total_start.elapsed()
    );
    Ok(())
}

/// Run pipeline on one image and write all debug artifacts to output subdir.
fn process_and_save_debug(
    engine: &mut AuraSeekEngine,
    path: &std::path::Path,
    output_base: &str,
) -> Result<()> {
    let out_dir = format!(
        "{}/{}",
        output_base,
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("out")
    );
    std::fs::create_dir_all(&out_dir)?;
    let img_str = path.to_str().unwrap();

    let start_vision = Instant::now();
    let output = engine.process_image(img_str)?;
    let vision_dur = start_vision.elapsed();

    // 1. Save vision embedding
    std::fs::write(
        format!("{out_dir}/embeddings.json"),
        serde_json::to_string_pretty(&json!({ "vision_embedding": output.vision_embedding }))?,
    )?;

    // 2. Save detections JSON (mask_rle is serde::skip so not in JSON)
    std::fs::write(
        format!("{out_dir}/detections.json"),
        serde_json::to_string_pretty(&output.objects)?,
    )?;

    // 3. Load image, extract masks, draw segmentation + detections, save det_seg.jpg
    let viz_start = Instant::now();
    let (pixels, w, h) = load_rgb(img_str)?;
    extract_masks(&output.objects, w, h, &out_dir)?;

    let mut px = pixels.clone();
    draw_segmentation(&mut px, w, h, &output.objects, 0.35);
    draw_detections(&mut px, w, h, &output.objects, FONT_PATH);
    save_rgb(px, w, h, &format!("{out_dir}/det_seg.jpg"))?;
    let viz_dur = viz_start.elapsed();

    // 4. Save faces JSON and draw faces overlay
    let face_count = output.faces.len();
    if !output.faces.is_empty() {
        std::fs::write(
            format!("{out_dir}/faces.json"),
            serde_json::to_string_pretty(&output.faces)?,
        )?;
        let mut px_f = pixels.clone();
        draw_faces(&mut px_f, w, h, &output.faces, FONT_PATH);
        save_rgb(px_f, w, h, &format!("{out_dir}/det_faces.jpg"))?;
    }

    let stats = format!(
        "{MAGENTA}result:{RESET} {GREEN}{} objects{RESET}, {YELLOW}{} faces{RESET}, {BOLD}{RED} face-IDs: {}{RESET}",
        output.objects.len(),
        face_count,
        engine.session_faces.len()
    );
    crate::log_info!("{}", stats);

    crate::log_info!(
        "  - {CYAN}timing: {RESET}{GREEN}pipeline: {:?}{RESET} | {MAGENTA}viz+save: {:?}{RESET}",
        vision_dur,
        viz_dur
    );

    for (idx, rec) in output.objects.iter().enumerate() {
        crate::log_info!(
            "  - {CYAN}obj {}: {RESET}{BOLD}{GREEN}{:<12}{RESET} | {YELLOW}conf: {:.2}{RESET} | {MAGENTA}area: {:<8}{RESET}",
            idx,
            rec.class_name,
            rec.conf,
            rec.mask_area
        );
    }

    Ok(())
}

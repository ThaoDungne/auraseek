/// visualization helpers for drawing and exporting images
use anyhow::Result;
use image::{io::Reader as ImageReader, RgbImage, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

use crate::processor::vision::yolo_postprocess::DetectionRecord;
use crate::model::face::FaceGroup;

// standard palette for classes
pub fn palette(class_id: usize) -> (u8, u8, u8) {
    const COLORS: &[(u8, u8, u8)] = &[
        (0, 255, 0),   (255, 80, 80),  (80, 80, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255),  (128, 255, 0), (0, 128, 255),
        (255, 128, 0), (128, 0, 255),  (0, 200, 128), (200, 0, 128),
    ];
    COLORS[class_id % COLORS.len()]
}

/// load image to flat rgb buffer
pub fn load_rgb(path: &str) -> Result<(Vec<u8>, u32, u32)> {
    let img = ImageReader::open(path)?.decode()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}

/// save rgb buffer to file
pub fn save_rgb(pixels: Vec<u8>, w: u32, h: u32, path: &str) -> Result<()> {
    let img = RgbImage::from_raw(w, h, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create rgbimage from buffer"))?;
    img.save(path)?;
    Ok(())
}

/// save rgba buffer to file
pub fn save_rgba(pixels: Vec<u8>, w: u32, h: u32, path: &str) -> Result<()> {
    let img = RgbaImage::from_raw(w, h, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create rgbaimage from buffer"))?;
    img.save(path)?;
    Ok(())
}

/// draw object detections
pub fn draw_detections(
    pixels:    &mut Vec<u8>,
    w:         u32,
    h:         u32,
    records:   &[DetectionRecord],
    font_path: Option<&str>,
) {
    let font: Option<Font<'static>> = font_path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| Font::try_from_vec(b));

    let mut img = RgbImage::from_raw(w, h, pixels.clone())
        .expect("invalid pixel buffer size");

    for (idx, rec) in records.iter().enumerate() {
        let (cr, cg, cb) = palette(idx);
        let color  = image::Rgb([cr, cg, cb]);
        let x1     = rec.bbox[0] as i32;
        let y1     = rec.bbox[1] as i32;
        let bw     = (rec.bbox[2] - rec.bbox[0]).max(1.0) as u32;
        let bh     = (rec.bbox[3] - rec.bbox[1]).max(1.0) as u32;

        for t in 0..2i32 {
            draw_hollow_rect_mut(
                &mut img,
                Rect::at(x1 - t, y1 - t)
                    .of_size((bw + 2 * t as u32).max(1), (bh + 2 * t as u32).max(1)),
                color,
            );
        }

        let label  = format!("{} {:.2}", rec.class_name, rec.conf);
        let scale  = Scale::uniform(14.0);
        let lh     = 18u32;
        let lw     = (label.len() as u32 * 8 + 6).min(w);
        let ly     = (y1 - lh as i32).max(0);

        draw_filled_rect_mut(
            &mut img,
            Rect::at(x1, ly).of_size(lw, lh),
            color,
        );

        if let Some(ref f) = font {
            draw_text_mut(
                &mut img,
                image::Rgb([0u8, 0u8, 0u8]),
                x1 + 2, ly + 2,
                scale,
                f,
                &label,
            );
        }
    }
    pixels.copy_from_slice(&img.into_raw());
}

/// draw segmentation mask overlay
pub fn draw_segmentation(
    pixels:  &mut Vec<u8>,
    _w:      u32,
    _h:      u32,
    records: &[DetectionRecord],
    alpha:   f32,
) {
    for (idx, rec) in records.iter().enumerate() {
        let (mr, mg, mb) = palette(idx);
        for &(offset, length) in &rec.mask_rle {
            for i in offset..(offset + length) {
                let base = i as usize * 3;
                if base + 2 < pixels.len() {
                    pixels[base]     = (pixels[base]     as f32 * (1.0 - alpha) + mr as f32 * alpha) as u8;
                    pixels[base + 1] = (pixels[base + 1] as f32 * (1.0 - alpha) + mg as f32 * alpha) as u8;
                    pixels[base + 2] = (pixels[base + 2] as f32 * (1.0 - alpha) + mb as f32 * alpha) as u8;
                }
            }
        }
    }
}

/// export white masks on transparent background
pub fn extract_masks(
    records: &[DetectionRecord],
    w:       u32,
    h:       u32,
    out_dir: &str,
) -> Result<()> {
    let total = (w * h) as usize;
    for (i, rec) in records.iter().enumerate() {
        let mut rgba = vec![0u8; total * 4];

        for &(offset, length) in &rec.mask_rle {
            for px in offset..(offset + length) {
                let base = px as usize * 4;
                if base + 3 < rgba.len() {
                    rgba[base]     = 255;
                    rgba[base + 1] = 255;
                    rgba[base + 2] = 255;
                    rgba[base + 3] = 255; 
                }
            }
        }

        let filename = format!("{}/mask_{}_{}.png", out_dir, i, rec.class_name);
        save_rgba(rgba, w, h, &filename)?;
    }
    Ok(())
}

/// draw face detections
pub fn draw_faces(
    pixels:    &mut Vec<u8>,
    w:         u32,
    h:         u32,
    faces:     &[FaceGroup],
    font_path: Option<&str>,
) {
    let font: Option<Font<'static>> = font_path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| Font::try_from_vec(b));

    let mut img = RgbImage::from_raw(w, h, pixels.clone())
        .expect("invalid pixel buffer size");

    for (idx, fg) in faces.iter().enumerate() {
        let (cr, cg, cb) = palette(idx + 8);
        let color = image::Rgb([cr, cg, cb]);

        let x1 = fg.bbox[0] as i32;
        let y1 = fg.bbox[1] as i32;
        let bw = (fg.bbox[2] - fg.bbox[0]).max(1.0) as u32;
        let bh = (fg.bbox[3] - fg.bbox[1]).max(1.0) as u32;

        for t in 0..2i32 {
            draw_hollow_rect_mut(
                &mut img,
                Rect::at(x1 - t, y1 - t)
                    .of_size((bw + 2 * t as u32).max(1), (bh + 2 * t as u32).max(1)),
                color,
            );
        }

        let label = if let Some(ref name) = fg.name {
            format!("{}: {}", name, &fg.face_id[..8])
        } else {
            format!("unknown: {}", &fg.face_id[..8])
        };
        
        let scale = Scale::uniform(14.0);
        let lh    = 18u32;
        let lw    = (label.len() as u32 * 8 + 6).min(w);
        let ly    = (y1 - lh as i32).max(0);

        draw_filled_rect_mut(
            &mut img,
            Rect::at(x1, ly).of_size(lw, lh),
            color,
        );

        if let Some(ref f) = font {
            draw_text_mut(
                &mut img,
                image::Rgb([0u8, 0u8, 0u8]),
                x1 + 2, ly + 2,
                scale,
                f,
                &label,
            );
        }
    }
    pixels.copy_from_slice(&img.into_raw());
}

/// image preprocessing for yolo model
use anyhow::Result;
use image::{io::Reader as ImageReader, GenericImageView};

pub struct LetterboxResult {
    /// chw layout (3x640x640), pixel values / 255
    pub blob:      Vec<f32>,
    /// scale ratio: new_size / max(orig_w, orig_h)
    pub ratio:     f32,
    /// left padding in pixels
    pub pad_left:  u32,
    /// top padding in pixels
    pub pad_top:   u32,
    /// (orig_h, orig_w)
    pub orig_size: (u32, u32),
}

/// letterbox resize image to 640x640 chw layout
pub fn letterbox_640(path: &str) -> Result<LetterboxResult> {
    let img = ImageReader::open(path)?.decode()?;
    let (orig_w, orig_h): (u32, u32) = img.dimensions();

    let ratio    = 640.0f32 / orig_w.max(orig_h) as f32;
    let new_w    = (orig_w as f32 * ratio).round() as u32;
    let new_h    = (orig_h as f32 * ratio).round() as u32;
    let pad_left = (640 - new_w) / 2;
    let pad_top  = (640 - new_h) / 2;

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let rgb     = resized.to_rgb8();

    let area     = (640 * 640) as usize;
    // fill padding with yolo-standard 114
    let mut blob = vec![114.0f32 / 255.0; 3 * area];

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let px  = x + pad_left;
        let py  = y + pad_top;
        let idx = (py as usize * 640) + px as usize;
        blob[idx]            = pixel[0] as f32 / 255.0; // r
        blob[idx + area]     = pixel[1] as f32 / 255.0; // g
        blob[idx + 2 * area] = pixel[2] as f32 / 255.0; // b
    }

    Ok(LetterboxResult { blob, ratio, pad_left, pad_top, orig_size: (orig_h, orig_w) })
}

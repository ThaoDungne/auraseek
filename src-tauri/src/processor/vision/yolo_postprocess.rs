/// nms, mask reconstruction, and rle encoding for yolo
use crate::model::yolo::{YoloDet, YoloRawResult};
use crate::processor::vision::yolo_image::LetterboxResult;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectionRecord {
    pub class_name: String,
    pub conf:       f32,
    pub bbox:       [f32; 4],
    pub mask_area:  u32,
    #[serde(skip)]
    pub mask_rle:   Vec<(u32, u32)>,
}

impl DetectionRecord {
    /// decode rle to full mask bitmap
    #[allow(dead_code)]
    pub fn decode_rle(&self, total_pixels: usize) -> Vec<u8> {
        let mut mask = vec![0u8; total_pixels];
        for &(offset, length) in &self.mask_rle {
            let end = ((offset + length) as usize).min(total_pixels);
            for px in (offset as usize)..end {
                mask[px] = 1;
            }
        }
        mask
    }
}

pub struct YoloProcessor;

impl YoloProcessor {
    pub fn postprocess(
        raw:        &YoloRawResult,
        lb:         &LetterboxResult,
        conf_thresh: f32,
        iou_thresh:  f32,
    ) -> Vec<DetectionRecord> {
        let dets = raw.raw_detections(conf_thresh);
        if dets.is_empty() { return vec![]; }

        let kept = Self::nms(&dets, iou_thresh);
        let (orig_h, orig_w) = lb.orig_size;

        kept.into_iter().map(|d| {
            let unpad = |v: f32, pad: u32, clip: u32| -> f32 {
                ((v - pad as f32) / lb.ratio).clamp(0.0, clip as f32)
            };
            let x1 = unpad(d.x1, lb.pad_left, orig_w);
            let y1 = unpad(d.y1, lb.pad_top,  orig_h);
            let x2 = unpad(d.x2, lb.pad_left, orig_w);
            let y2 = unpad(d.y2, lb.pad_top,  orig_h);

            let mask = Self::process_mask(
                &raw.protos,
                &d.mask_coeffs,
                raw.proto_c, raw.proto_h, raw.proto_w,
                orig_h, orig_w,
                lb.pad_left, lb.pad_top, lb.ratio,
                x1 as u32, y1 as u32, x2 as u32, y2 as u32,
            );

            let mask_rle  = Self::encode_rle(&mask);
            let mask_area = mask.iter().filter(|&&v| v == 1).count() as u32;

            DetectionRecord {
                class_name: d.class_name.clone(),
                conf:       d.conf,
                bbox:       [x1, y1, x2, y2],
                mask_area,
                mask_rle,
            }
        }).collect()
    }

    fn nms(dets: &[YoloDet], iou_thresh: f32) -> Vec<YoloDet> {
        let mut order: Vec<usize> = (0..dets.len()).collect();
        order.sort_by(|&a, &b| dets[b].conf.partial_cmp(&dets[a].conf).unwrap());

        let area = |d: &YoloDet| (d.x2 - d.x1).max(0.0) * (d.y2 - d.y1).max(0.0);
        let iou  = |a: &YoloDet, b: &YoloDet| -> f32 {
            let ix1   = a.x1.max(b.x1);
            let iy1   = a.y1.max(b.y1);
            let ix2   = a.x2.min(b.x2);
            let iy2   = a.y2.min(b.y2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let union = area(a) + area(b) - inter;
            if union <= 0.0 { 0.0 } else { inter / union }
        };

        let mut suppressed = vec![false; dets.len()];
        let mut result     = Vec::new();

        for &i in &order {
            if suppressed[i] { continue; }
            result.push(dets[i].clone());
            for &j in &order {
                if !suppressed[j] && i != j && iou(&dets[i], &dets[j]) > iou_thresh {
                    suppressed[j] = true;
                }
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn process_mask(
        protos:   &[f32],
        coeffs:   &[f32],
        proto_c:  usize,
        proto_h:  usize,
        proto_w:  usize,
        orig_h:   u32,
        orig_w:   u32,
        pad_left: u32,
        pad_top:  u32,
        ratio:    f32,
        bx1: u32, by1: u32, bx2: u32, by2: u32,
    ) -> Vec<u8> {
        let n_px = proto_h * proto_w;
        let mut mask_flat = vec![0.0f32; n_px];
        for c in 0..proto_c {
            let coeff = coeffs[c];
            let off   = c * n_px;
            for p in 0..n_px {
                mask_flat[p] += coeff * protos[off + p];
            }
        }

        for v in mask_flat.iter_mut() {
            *v = 1.0 / (1.0 + (-(*v).clamp(-88.0, 88.0)).exp());
        }

        let mask_640 = Self::resize_bilinear(&mask_flat, proto_h, proto_w, 640, 640);

        let img_w_in = (orig_w as f32 * ratio).round() as usize;
        let img_h_in = (orig_h as f32 * ratio).round() as usize;
        let pl = pad_left as usize;
        let pt = pad_top  as usize;

        let mut cropped = vec![0.0f32; img_h_in * img_w_in];
        for row in 0..img_h_in {
            let src_off = (pt + row) * 640 + pl;
            let dst_off = row * img_w_in;
            cropped[dst_off..dst_off + img_w_in]
                .copy_from_slice(&mask_640[src_off..src_off + img_w_in]);
        }

        let mask_orig = Self::resize_bilinear(
            &cropped, img_h_in, img_w_in,
            orig_h as usize, orig_w as usize,
        );

        let ow = orig_w as usize;
        let oh = orig_h as usize;
        let mut binary = vec![0u8; oh * ow];
        for row in 0..oh {
            for col in 0..ow {
                let in_bbox = col >= bx1 as usize && col < bx2 as usize
                           && row >= by1 as usize && row < by2 as usize;
                if in_bbox && mask_orig[row * ow + col] > 0.5 {
                    binary[row * ow + col] = 1;
                }
            }
        }
        binary
    }

    fn resize_bilinear(
        src:   &[f32],
        src_h: usize, src_w: usize,
        dst_h: usize, dst_w: usize,
    ) -> Vec<f32> {
        let mut dst    = vec![0.0f32; dst_h * dst_w];
        let scale_h    = src_h as f32 / dst_h as f32;
        let scale_w    = src_w as f32 / dst_w as f32;

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sy = ((dy as f32 + 0.5) * scale_h - 0.5).max(0.0);
                let sx = ((dx as f32 + 0.5) * scale_w - 0.5).max(0.0);
                let y0 = (sy as usize).min(src_h - 1);
                let x0 = (sx as usize).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let x1 = (x0 + 1).min(src_w - 1);
                let wy = sy - y0 as f32;
                let wx = sx - x0 as f32;
                let v  = src[y0 * src_w + x0] * (1.0 - wy) * (1.0 - wx)
                       + src[y0 * src_w + x1] * (1.0 - wy) * wx
                       + src[y1 * src_w + x0] * wy * (1.0 - wx)
                       + src[y1 * src_w + x1] * wy * wx;
                dst[dy * dst_w + dx] = v;
            }
        }
        dst
    }

    fn encode_rle(mask: &[u8]) -> Vec<(u32, u32)> {
        let mut rle = Vec::new();
        let mut i   = 0usize;
        while i < mask.len() {
            if mask[i] == 1 {
                let start = i as u32;
                while i < mask.len() && mask[i] == 1 { i += 1; }
                rle.push((start, (i as u32) - start));
            } else {
                i += 1;
            }
        }
        rle
    }
}

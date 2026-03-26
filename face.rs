/// face detection and recognition using opencv (yunet and sface)
use anyhow::Result;
use opencv::{
    core::{Mat, Ptr, Size},
    objdetect::{FaceDetectorYN, FaceRecognizerSF},
    imgcodecs::{imdecode, IMREAD_COLOR, IMREAD_IGNORE_ORIENTATION},
    prelude::*,
};
use crate::processor::vision::face_image::FaceDb;
use crate::{log_info, log_warn};

const SCORE_THRESHOLD: f32 = 0.3;
const NMS_THRESHOLD: f32 = 0.3;
const TOP_K: i32 = 500000;
/// Cosine similarity threshold for face identity matching.
/// Dùng chung cho:
/// - so khớp với face_db (FaceDb::query_id)
/// - gom nhóm các mặt "unknown" trong cùng session (session_faces)
pub const COSINE_THRESHOLD: f32 = 0.33;

#[derive(Debug, Clone, serde::Serialize)]
pub struct FaceGroup {
    pub face_id:  String,
    pub name:     Option<String>,
    pub conf:     f32,
    pub bbox:     [f32; 4],
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

struct YuNetFace {
    pub row:   Mat,
    pub bbox:  [f32; 4],
    pub score: f32,
}

pub struct FaceModel {
    detector_path: String,
    detector:      Ptr<FaceDetectorYN>,
    detector_size: Size,
    recognizer:    Ptr<FaceRecognizerSF>,
}

impl FaceModel {
    pub fn new(yunet_path: &str, sface_path: &str) -> Result<Self> {
        let size = Size::new(320, 320);
        
        let (backend, target, provider_name) = if opencv::core::get_cuda_enabled_device_count()? > 0 {
            (opencv::dnn::DNN_BACKEND_CUDA, opencv::dnn::DNN_TARGET_CUDA, "CUDA")
        } else {
            (opencv::dnn::DNN_BACKEND_OPENCV, opencv::dnn::DNN_TARGET_CPU, "CPU")
        };

        log_info!("model: {:<45} | provider: {}", yunet_path, provider_name);
        log_info!("model: {:<45} | provider: {}", sface_path, provider_name);

        let detector = FaceDetectorYN::create(
            yunet_path, "", size, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K, backend, target
        )?;
        let recognizer = FaceRecognizerSF::create(
            sface_path, "", backend, target
        )?;

        Ok(Self {
            detector_path: yunet_path.to_string(),
            detector,
            detector_size: size,
            recognizer,
        })
    }

    fn detect_faces_raw(&mut self, frame: &Mat) -> Result<Vec<YuNetFace>> {
        if frame.empty() { return Ok(vec![]); }
        let size = frame.size()?;
        let (w, h) = (size.width as f32, size.height as f32);
        if w < 20.0 || h < 20.0 { return Ok(vec![]); }

        let target_size = self.detector_size; // always 640x640
        self.detector.set_input_size(target_size)?;
        
        let mut scale = (target_size.width as f32 / w).min(target_size.height as f32 / h);
        if scale > 1.0 { scale = 1.0; } // Prevent upscaling!

        let new_w = (w * scale).round() as i32;
        let new_h = (h * scale).round() as i32;

        let mut resized = Mat::default();
        opencv::imgproc::resize(
            frame,
            &mut resized,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        let mut padded = Mat::default();
        let top = (target_size.height - new_h) / 2;
        let bottom = target_size.height - new_h - top;
        let left = (target_size.width - new_w) / 2;
        let right = target_size.width - new_w - left;

        opencv::core::copy_make_border(
            &resized,
            &mut padded,
            top, bottom, left, right,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default()
        )?;

        let mut faces_mat = Mat::default();
        crate::log_info!("  TRACE: calling self.detector.detect...");
        self.detector.detect(&padded, &mut faces_mat)?;
        crate::log_info!("  TRACE: detector.detect done, rows: {}", faces_mat.rows());
        if faces_mat.rows() <= 0 { return Ok(vec![]); }

        let mut raw_faces = Vec::new();
        let ratio = 1.0 / scale;
        let pad_left_f = left as f32;
        let pad_top_f = top as f32;

        for i in 0..faces_mat.rows() {
            let mut row_ref = faces_mat.row(i)?.try_clone()?;
            let score = *row_ref.at_2d::<f32>(0, 14)?;
            if score < SCORE_THRESHOLD { continue; }

            // Map back to FRAME space
            *row_ref.at_2d_mut::<f32>(0, 0)? = (*row_ref.at_2d::<f32>(0, 0)? - pad_left_f) * ratio;
            *row_ref.at_2d_mut::<f32>(0, 1)? = (*row_ref.at_2d::<f32>(0, 1)? - pad_top_f) * ratio;
            *row_ref.at_2d_mut::<f32>(0, 2)? *= ratio;
            *row_ref.at_2d_mut::<f32>(0, 3)? *= ratio;
            
            for col_idx in 0..5 {
                let x_col = 4 + col_idx * 2;
                let y_col = 5 + col_idx * 2;
                *row_ref.at_2d_mut::<f32>(0, x_col as i32)? = (*row_ref.at_2d::<f32>(0, x_col as i32)? - pad_left_f) * ratio;
                *row_ref.at_2d_mut::<f32>(0, y_col as i32)? = (*row_ref.at_2d::<f32>(0, y_col as i32)? - pad_top_f) * ratio;
            }

            let bbox = [
                *row_ref.at_2d::<f32>(0, 0)?,
                *row_ref.at_2d::<f32>(0, 1)?,
                *row_ref.at_2d::<f32>(0, 2)?,
                *row_ref.at_2d::<f32>(0, 3)?,
            ];

            raw_faces.push(YuNetFace { row: row_ref, bbox, score });
        }
        raw_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut kept: Vec<YuNetFace> = Vec::new();
        for f in raw_faces {
            let mut overlap = false;
            for k in &kept {
                if calc_iou(&f.bbox, &k.bbox) > 0.3 { overlap = true; break; }
            }
            if !overlap { kept.push(f); }
        }

        Ok(kept)
    }

    pub fn detect_from_mat(&mut self, frame: &Mat, db: &FaceDb) -> Result<Vec<FaceGroup>> {
        crate::log_info!("  TRACE: entering detect_from_mat");
        let kept = self.detect_faces_raw(frame)?;
        crate::log_info!("  TRACE: kept {} faces", kept.len());
        let mut groups = Vec::new();

        for face in kept {
            let mut aligned = Mat::default();
            crate::log_info!("  TRACE: align_crop");
            if let Err(e) = self.recognizer.align_crop(frame, &face.row, &mut aligned) {
                log_warn!("face align_crop failed, skip one detection: {}", e);
                continue;
            }
            
            let mut feature = Mat::default();
            crate::log_info!("  TRACE: SFace feature extraction...");
            if let Err(e) = self.recognizer.feature(&aligned, &mut feature) {
                log_warn!("face feature extraction failed, skip one detection: {}", e);
                continue;
            }
            crate::log_info!("  TRACE: SFace feature done.");
            let embedding = mat_to_vec_f32(&feature)?;

            let (name, face_id) = match db.query_id(&embedding, COSINE_THRESHOLD) {
                Some((n, id)) => (Some(n), id),
                None => (None, "unknown_placeholder".to_string()),
            };

            groups.push(FaceGroup {
                face_id,
                name,
                conf: face.score,
                bbox: [
                    face.bbox[0], 
                    face.bbox[1], 
                    face.bbox[0] + face.bbox[2], 
                    face.bbox[1] + face.bbox[3]
                ],
                embedding,
            });
        }
        Ok(groups)
    }

    pub fn detect_from_path(&mut self, path: &str, db: &FaceDb) -> Result<Vec<FaceGroup>> {
        let bytes = std::fs::read(path)?;
        let buf = opencv::core::Vector::<u8>::from_iter(bytes);
        let frame = imdecode(&buf, IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION)?;
        self.detect_from_mat(&frame, db)
    }

    pub fn detect_from_mat_with_aligned(
        &mut self,
        frame: &Mat,
        db: &FaceDb,
    ) -> Result<Vec<(FaceGroup, Mat)>> {
        crate::log_info!("  TRACE: entering detect_from_mat_with_aligned");
        let kept = self.detect_faces_raw(frame)?;
        let mut out = Vec::new();
        
        for face in kept {
            let mut aligned = Mat::default();
            if let Err(e) = self.recognizer.align_crop(frame, &face.row, &mut aligned) {
                log_warn!("face align_crop failed, skip one detection: {}", e);
                continue;
            }
            let mut feature = Mat::default();
            crate::log_info!("  TRACE: SFace (with aligned) feature...");
            if let Err(e) = self.recognizer.feature(&aligned, &mut feature) {
                log_warn!("face feature extraction failed, skip one detection: {}", e);
                continue;
            }
            let embedding = mat_to_vec_f32(&feature)?;
            let (name, face_id) = match db.query_id(&embedding, COSINE_THRESHOLD) {
                Some((n, id)) => (Some(n), id),
                None => (None, "unknown_placeholder".to_string()),
            };
            out.push((FaceGroup {
                face_id,
                name,
                conf: face.score,
                bbox: [
                    face.bbox[0],
                    face.bbox[1],
                    face.bbox[0] + face.bbox[2],
                    face.bbox[1] + face.bbox[3],
                ],
                embedding,
            }, aligned));
        }
        Ok(out)
    }

    pub fn extract_feature_for_db(&mut self, img_path: &str) -> Result<Vec<Vec<f32>>> {
        let bytes = std::fs::read(img_path)?;
        let buf = opencv::core::Vector::<u8>::from_iter(bytes);
        let frame = imdecode(&buf, IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION)?;
        if frame.empty() { return Ok(vec![]); }

        let kept = self.detect_faces_raw(&frame)?;
        let mut features = Vec::new();
        for face in kept {
            let mut aligned = Mat::default();
            if let Err(e) = self.recognizer.align_crop(&frame, &face.row, &mut aligned) {
                log_warn!("face_db align_crop failed for {}, skip one detection: {}", img_path, e);
                continue;
            }
            let mut feature = Mat::default();
            if let Err(e) = self.recognizer.feature(&aligned, &mut feature) {
                log_warn!("face_db feature extraction failed for {}, skip one detection: {}", img_path, e);
                continue;
            }
            features.push(mat_to_vec_f32(&feature)?);
        }
        Ok(features)
    }
}

pub fn calc_iou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let inter_x = b1[0].max(b2[0]);
    let inter_y = b1[1].max(b2[1]);
    let inter_w = (b1[0] + b1[2]).min(b2[0] + b2[2]) - inter_x;
    let inter_h = (b1[1] + b1[3]).min(b2[1] + b2[3]) - inter_y;

    if inter_w <= 0.0 || inter_h <= 0.0 { return 0.0; }
    let inter_area = inter_w * inter_h;
    let union_area = b1[2] * b1[3] + b2[2] * b2[3] - inter_area;
    inter_area / union_area
}

pub fn mat_to_vec_f32(m: &Mat) -> Result<Vec<f32>> {
    let mut v = Vec::with_capacity(m.cols() as usize);
    for j in 0..m.cols() {
        v.push(*m.at_2d::<f32>(0, j)?);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    Ok(v.into_iter().map(|x| x / norm).collect())
}

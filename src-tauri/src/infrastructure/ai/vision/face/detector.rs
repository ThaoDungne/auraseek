/// face detection and recognition using opencv (yunet and sface)
use anyhow::Result;
use opencv::{
    core::{Mat, Ptr, Size},
    objdetect::{FaceDetectorYN, FaceRecognizerSF},
    imgcodecs::{imread, IMREAD_COLOR},
    prelude::*,
};
use super::db::FaceDb;
use crate::log_info;

const SCORE_THRESHOLD: f32 = 0.93; 
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

    /// Detect faces from an already-loaded Mat (e.g. a person crop).
    ///
    /// To avoid OpenCV DNN shape errors on some extreme aspect ratios, we
    /// always resize to the fixed detector input size that was configured
    /// when the model was created (320x320). Dynamic resizing of the
    /// detector network has been removed for stability.
    ///
    /// Returns FaceGroups with bbox in the coordinate space of the input frame.
    pub fn detect_from_mat(&mut self, frame: &Mat, db: &FaceDb) -> Result<Vec<FaceGroup>> {
        if frame.empty() { return Ok(vec![]); }

        let size = frame.size()?;
        let (w, h) = (size.width as f32, size.height as f32);
        if w < 20.0 || h < 20.0 {
            return Ok(vec![]);
        }

        // Always resize to the detector's fixed input size.
        let target_size = self.detector_size;
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            frame,
            &mut resized,
            target_size,
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        let mut faces_mat = Mat::default();
        self.detector.detect(&resized, &mut faces_mat)?;
        if faces_mat.rows() <= 0 { return Ok(vec![]); }

        let mut raw_faces = Vec::new();
        for i in 0..faces_mat.rows() {
            let row_ref = faces_mat.row(i)?;
            let score = *row_ref.at_2d::<f32>(0, 14)?;
            if score < SCORE_THRESHOLD { continue; }
            
            raw_faces.push(YuNetFace {
                row:   row_ref.try_clone()?,
                bbox:  [
                    *row_ref.at_2d::<f32>(0, 0)?,
                    *row_ref.at_2d::<f32>(0, 1)?,
                    *row_ref.at_2d::<f32>(0, 2)?,
                    *row_ref.at_2d::<f32>(0, 3)?,
                ],
                score,
            });
        }
        
        raw_faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut kept: Vec<YuNetFace> = Vec::new();
        for f in raw_faces {
            let mut overlap = false;
            for k in &kept {
                if calc_iou(&f.bbox, &k.bbox) > 0.3 {
                    overlap = true;
                    break;
                }
            }
            if !overlap { kept.push(f); }
        }

        let mut groups = Vec::new();
        // Map bboxes from resized detector space back to original frame space.
        let target_w = self.detector_size.width as f32;
        let target_h = self.detector_size.height as f32;
        let ratio_x = if target_w > 0.0 { w / target_w } else { 1.0 };
        let ratio_y = if target_h > 0.0 { h / target_h } else { 1.0 };

        for face in kept.into_iter() {
            // Map the 14 geometry features from 320x320 space back to original `frame` resolution
            // to extract highest-quality face embeddings.
            let mut scaled_row = face.row.try_clone()?;
            *scaled_row.at_2d_mut::<f32>(0, 0)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 1)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 2)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 3)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 4)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 5)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 6)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 7)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 8)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 9)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 10)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 11)? *= ratio_y;
            *scaled_row.at_2d_mut::<f32>(0, 12)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 13)? *= ratio_y;

            let mut aligned = Mat::default();
            self.recognizer.align_crop(frame, &scaled_row, &mut aligned)?;
            
            let mut feature = Mat::default();
            self.recognizer.feature(&aligned, &mut feature)?;
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
                    face.bbox[0] * ratio_x, 
                    face.bbox[1] * ratio_y, 
                    (face.bbox[0] + face.bbox[2]) * ratio_x, 
                    (face.bbox[1] + face.bbox[3]) * ratio_y
                ],
                embedding,
            });
        }
        Ok(groups)
    }

    /// Convenience: load image from path and detect faces on the full image.
    pub fn detect_from_path(&mut self, path: &str, db: &FaceDb) -> Result<Vec<FaceGroup>> {
        let frame = imread(path, IMREAD_COLOR)?;
        self.detect_from_mat(&frame, db)
    }

    /// Same as `detect_from_mat` but also returns the aligned Mat crop (112×112)
    /// for each detected face, so callers can save it for debugging.
    pub fn detect_from_mat_with_aligned(
        &mut self,
        frame: &Mat,
        db: &FaceDb,
    ) -> Result<Vec<(FaceGroup, Mat)>> {
        if frame.empty() { return Ok(vec![]); }
        let size = frame.size()?;
        let (w, h) = (size.width as f32, size.height as f32);
        if w < 20.0 || h < 20.0 { return Ok(vec![]); }

        let target_size = self.detector_size;
        let mut resized = Mat::default();
        opencv::imgproc::resize(frame, &mut resized, target_size, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;

        let mut faces_mat = Mat::default();
        self.detector.detect(&resized, &mut faces_mat)?;
        if faces_mat.rows() <= 0 { return Ok(vec![]); }

        let mut raw_faces = Vec::new();
        for i in 0..faces_mat.rows() {
            let row_ref = faces_mat.row(i)?;
            let score = *row_ref.at_2d::<f32>(0, 14)?;
            if score < SCORE_THRESHOLD { continue; }
            raw_faces.push(YuNetFace {
                row: row_ref.try_clone()?,
                bbox: [
                    *row_ref.at_2d::<f32>(0, 0)?,
                    *row_ref.at_2d::<f32>(0, 1)?,
                    *row_ref.at_2d::<f32>(0, 2)?,
                    *row_ref.at_2d::<f32>(0, 3)?,
                ],
                score,
            });
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

        let target_w = self.detector_size.width as f32;
        let target_h = self.detector_size.height as f32;
        let ratio_x = if target_w > 0.0 { w / target_w } else { 1.0 };
        let ratio_y = if target_h > 0.0 { h / target_h } else { 1.0 };

        let mut out = Vec::new();
        for face in kept {
            let mut scaled_row = face.row.try_clone()?;
            for col in [0usize, 4, 6, 8, 10, 12] { *scaled_row.at_2d_mut::<f32>(0, col as i32)? *= ratio_x; }
            for col in [1usize, 5, 7, 9, 11, 13] { *scaled_row.at_2d_mut::<f32>(0, col as i32)? *= ratio_y; }
            *scaled_row.at_2d_mut::<f32>(0, 2)? *= ratio_x;
            *scaled_row.at_2d_mut::<f32>(0, 3)? *= ratio_y;

            let mut aligned = Mat::default();
            self.recognizer.align_crop(frame, &scaled_row, &mut aligned)?;
            let mut feature = Mat::default();
            self.recognizer.feature(&aligned, &mut feature)?;
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
                    face.bbox[0] * ratio_x,
                    face.bbox[1] * ratio_y,
                    (face.bbox[0] + face.bbox[2]) * ratio_x,
                    (face.bbox[1] + face.bbox[3]) * ratio_y,
                ],
                embedding,
            }, aligned));
        }
        Ok(out)
    }

    pub fn extract_feature_for_db(&mut self, img_path: &str) -> Result<Vec<Vec<f32>>> {
        let frame = imread(img_path, IMREAD_COLOR)?;
        if frame.empty() { return Ok(vec![]); }

        let s = frame.size()?;
        if s != self.detector_size {
            self.detector_size = s;
            let (backend, target) = if opencv::core::get_cuda_enabled_device_count()? > 0 {
                (opencv::dnn::DNN_BACKEND_CUDA, opencv::dnn::DNN_TARGET_CUDA)
            } else {
                (opencv::dnn::DNN_BACKEND_OPENCV, opencv::dnn::DNN_TARGET_CPU)
            };
            self.detector = FaceDetectorYN::create(
                &self.detector_path, "", self.detector_size, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K, backend, target
            )?;
        }

        let mut faces = Mat::default();
        self.detector.detect(&frame, &mut faces)?;
        let mut features = Vec::new();
        for i in 0..faces.rows() {
            let row = faces.row(i)?;
            let mut aligned = Mat::default();
            self.recognizer.align_crop(&frame, &row, &mut aligned)?;
            let mut feature = Mat::default();
            self.recognizer.feature(&aligned, &mut feature)?;
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

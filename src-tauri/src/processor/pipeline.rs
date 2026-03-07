use anyhow::Result;
use uuid::Uuid;

use crate::model::{AuraModel, FaceModel, YoloModel};
use crate::processor::vision::{cosine_similarity, letterbox_640, preprocess_aura, FaceDb, YoloProcessor};
use crate::processor::TextProcessor;
use crate::processor::vision::yolo_postprocess::DetectionRecord;
use crate::model::face::FaceGroup;
use crate::{log_info, log_warn};
use opencv::{
    core::{Mat, Rect},
    imgcodecs::{imread, IMREAD_COLOR},
    prelude::*,
};

/// Structured output from the AI pipeline, ready for DB storage.
#[derive(Debug, Clone)]
pub struct EngineOutput {
    pub objects:          Vec<DetectionRecord>,
    pub faces:            Vec<FaceGroup>,
    pub vision_embedding: Vec<f32>,
}

const DEFAULT_CONFIG: EngineConfig = EngineConfig {
    vision_path:  "assets/models/vision_tower_aura.onnx",
    text_path:    "assets/models/text_tower_aura.onnx",
    yolo_path:    "assets/models/yolo26n-seg.onnx",
    yunet_path:   "assets/models/face_detection_yunet_2022mar.onnx",
    sface_path:   "assets/models/face_recognition_sface_2021dec.onnx",
    vocab_path:   "assets/tokenizer/vocab.txt",
    bpe_path:     "assets/tokenizer/bpe.codes",
    face_db_path: "assets/face_db",
};
pub struct EngineConfig {
    pub vision_path: &'static str,
    pub text_path: &'static str,
    pub yolo_path: &'static str,
    pub yunet_path: &'static str,
    pub sface_path: &'static str,
    pub vocab_path: &'static str,
    pub bpe_path: &'static str,
    pub face_db_path: &'static str,
}

pub struct AuraSeekEngine {
    pub aura: AuraModel,
    #[allow(dead_code)]
    pub text_proc: TextProcessor,
    pub yolo: YoloModel,
    pub face: Option<FaceModel>,
    pub face_db: FaceDb,
    pub session_faces: Vec<(Vec<f32>, String)>,
}

impl AuraSeekEngine {
    pub fn new_default() -> Result<Self> {
        Self::new(DEFAULT_CONFIG)
    }

    pub fn new(config: EngineConfig) -> Result<Self> {
        log_info!("loading ai models");
        let aura = AuraModel::new(config.vision_path, config.text_path)?;
        let text_proc = TextProcessor::new(config.vocab_path, config.bpe_path)?;
        let yolo = YoloModel::new(config.yolo_path)?;
        
        let mut face = match FaceModel::new(config.yunet_path, config.sface_path) {
            Ok(m) => Some(m),
            Err(e) => {
                log_warn!("face model failed to load: {}", e);
                None
            }
        };

        let face_db = if let Some(ref mut fm) = face {
            FaceDb::build(config.face_db_path, fm).unwrap_or_else(|_| FaceDb::empty())
        } else {
            FaceDb::empty()
        };

        Ok(Self { aura, text_proc, yolo, face, face_db, session_faces: Vec::new() })
    }

    /// Run AI pipeline on a single image and return structured output (no disk I/O).
    pub fn process_image(&mut self, img_path: &str) -> Result<EngineOutput> {
        // 1. Vision embedding
        let vision_emb = self.aura.encode_image(preprocess_aura(img_path)?, 256, 256)
            .unwrap_or_default();

        // 2. YOLO detection + segmentation
        let lb = letterbox_640(img_path)?;
        let raw = self.yolo.detect(lb.blob.clone())?;
        let objects = YoloProcessor::postprocess(&raw, &lb, 0.25, 0.45);

        // 3. Face detection — only within person bboxes from YOLO
        let mut faces = vec![];
        if let Some(ref mut fm) = self.face {
            let person_bboxes: Vec<[f32; 4]> = objects.iter()
                .filter(|o| o.class_name == "person")
                .map(|o| o.bbox)
                .collect();

            if person_bboxes.is_empty() {
                // No person detected by YOLO — run on full image as fallback
                if let Ok(detected) = fm.detect_from_path(img_path, &self.face_db) {
                    faces = detected;
                }
            } else {
                // Crop each person bbox and run face detection on the crop
                let frame = imread(img_path, IMREAD_COLOR)?;
                if !frame.empty() {
                    let img_size = frame.size()?;
                    let (img_w, img_h) = (img_size.width, img_size.height);

                    for bbox in &person_bboxes {
                        let x1 = (bbox[0].max(0.0) as i32).min(img_w - 1);
                        let y1 = (bbox[1].max(0.0) as i32).min(img_h - 1);
                        let x2 = (bbox[2].max(0.0) as i32).min(img_w);
                        let y2 = (bbox[3].max(0.0) as i32).min(img_h);
                        let cw = x2 - x1;
                        let ch = y2 - y1;
                        if cw < 30 || ch < 30 { continue; }

                        let roi = Mat::roi(&frame, Rect::new(x1, y1, cw, ch))?;
                        let crop = roi.try_clone()?;

                        if let Ok(detected) = fm.detect_from_mat(&crop, &self.face_db) {
                            for mut f in detected {
                                // Map face bbox from crop coords back to original image coords
                                f.bbox[0] += x1 as f32;
                                f.bbox[1] += y1 as f32;
                                f.bbox[2] += x1 as f32;
                                f.bbox[3] += y1 as f32;
                                faces.push(f);
                            }
                        }
                    }
                }
            }

            // Session face matching for unknown faces
            for f in faces.iter_mut() {
                if f.face_id == "unknown_placeholder" {
                    let mut best_score = 0.55;
                    let mut cached_id = None;
                    for (cached_emb, id) in &self.session_faces {
                        let score = cosine_similarity(&f.embedding, cached_emb);
                        if score > best_score {
                            best_score = score;
                            cached_id = Some(id.clone());
                        }
                    }
                    if let Some(id) = cached_id {
                        f.face_id = id;
                    } else {
                        let new_id = Uuid::new_v4().to_string();
                        f.face_id = new_id.clone();
                        self.session_faces.push((f.embedding.clone(), new_id));
                    }
                }
            }
        }

        Ok(EngineOutput { objects, faces, vision_embedding: vision_emb })
    }
}

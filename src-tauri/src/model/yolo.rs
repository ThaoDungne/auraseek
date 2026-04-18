/// yolo model for object detection and segmentation
use anyhow::Result;
use std::collections::HashMap;
use ort::value::Value;
use ort::session::Session;
use crate::utils::build_session;
use crate::log_info;

pub struct YoloRawResult {
    pub det: Vec<f32>,
    pub protos: Vec<f32>,
    pub n_det: usize,
    pub det_dim: usize,
    pub proto_c: usize,
    pub proto_h: usize,
    pub proto_w: usize,
    pub class_names: Vec<String>,
}

impl YoloRawResult {
    pub fn raw_detections(&self, conf_thresh: f32) -> Vec<YoloDet> {
        (0..self.n_det)
            .filter_map(|i| {
                let base = i * self.det_dim;
                let conf = self.det[base + 4];
                if conf < conf_thresh { return None; }
                let class_id   = self.det[base + 5] as usize;
                let class_name = self.class_names
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("cls_{}", class_id));
                Some(YoloDet {
                    x1: self.det[base],
                    y1: self.det[base + 1],
                    x2: self.det[base + 2],
                    y2: self.det[base + 3],
                    conf,
                    class_id,
                    class_name,
                    mask_coeffs: self.det[base + 6..base + self.det_dim].to_vec(),
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct YoloDet {
    pub x1:          f32,
    pub y1:          f32,
    pub x2:          f32,
    pub y2:          f32,
    pub conf:        f32,
    #[allow(dead_code)]
    pub class_id:    usize,
    pub class_name:  String,
    pub mask_coeffs: Vec<f32>,
}

pub struct YoloModel {
    session:     Session,
    class_names: Vec<String>,
}

impl YoloModel {
    pub fn new(path: &str) -> Result<Self> {
        let class_names = Self::load_class_names(path);
        log_info!("yolo: {} classes loaded", class_names.len());
        Ok(Self {
            session: build_session(path)?,
            class_names,
        })
    }

    pub fn detect(
        &mut self,
        blob:   Vec<f32>,
    ) -> Result<YoloRawResult> {
        let img_tensor = Value::from_array((
            vec![1usize, 3, 640, 640],
            blob.into_boxed_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs!["images" => img_tensor])?;

        let (shape0, det_data)   = outputs[0].try_extract_tensor::<f32>()?;
        let (shape1, proto_data) = outputs[1].try_extract_tensor::<f32>()?;

        Ok(YoloRawResult {
            det:         det_data.to_vec() as Vec<f32>,
            protos:      proto_data.to_vec() as Vec<f32>,
            n_det:       shape0[1] as usize,
            det_dim:     shape0[2] as usize,
            proto_c:     shape1[1] as usize,
            proto_h:     shape1[2] as usize,
            proto_w:     shape1[3] as usize,
            class_names: self.class_names.clone(),
        })
    }

    fn load_class_names(path: &str) -> Vec<String> {
        let fallback = || (0..80).map(|i| format!("cls_{}", i)).collect::<Vec<_>>();
        let Ok(bytes) = std::fs::read(path) else { return fallback(); };
        let text = String::from_utf8_lossy(&bytes);

        let Some(start) = text.find("{0: '") else { return fallback(); };
        let slice = &text[start..];
        let Some(end) = slice.find('}') else { return fallback(); };
        let dict_str = &slice[..=end];

        let mut map: HashMap<usize, String> = HashMap::new();
        let mut remaining = dict_str;
        while let Some(colon_pos) = remaining.find(": '") {
            let before = remaining[..colon_pos].trim_start_matches(['{', ',', ' ']);
            if let Ok(idx) = before.trim().parse::<usize>() {
                let after = &remaining[colon_pos + 3..];
                if let Some(close) = after.find('\'') {
                    map.insert(idx, after[..close].to_string());
                    remaining = &after[close + 1..];
                    continue;
                }
            }
            break;
        }

        if map.is_empty() { return fallback(); }
        let max_id = *map.keys().max().unwrap();
        (0..=max_id)
            .map(|i| map.get(&i).cloned().unwrap_or_else(|| format!("cls_{}", i)))
            .collect()
    }
}

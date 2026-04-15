use anyhow::Result;
use std::collections::HashMap;
use uuid::Uuid;
use super::detector::FaceModel;
use crate::{log_info, log_warn};

pub struct FaceDb {
    /// face_name -> vec<embedding>
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    /// face_name -> uuid (persistent for this name)
    name_to_id: HashMap<String, String>,
}

impl FaceDb {
    pub fn empty() -> Self {
        Self { 
            embeddings: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn build(db_path: &str, face_model: &mut FaceModel) -> Result<Self> {
        let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut name_to_id: HashMap<String, String> = HashMap::new();
        
        let dir = std::path::Path::new(db_path);
        if !dir.exists() {
            log_warn!("face_db directory {} not found", db_path);
            return Ok(Self { embeddings, name_to_id });
        }

        for entry in std::fs::read_dir(dir)? {
            let entry   = entry?;
            let id_path = entry.path();
            if !id_path.is_dir() { continue; }
            let identity = id_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            // create v5 uuid based on name for persistence
            let ns_uuid = Uuid::NAMESPACE_DNS;
            let stable_id = Uuid::new_v5(&ns_uuid, identity.as_bytes()).to_string();
            name_to_id.insert(identity.clone(), stable_id);

            let mut person_embs = Vec::new();
            for img_entry in std::fs::read_dir(&id_path)? {
                let img_entry = img_entry?;
                let img_path  = img_entry.path();
                if img_path.is_dir() { continue; }

                let ext = img_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if !["jpg", "jpeg", "png"].contains(&ext.as_str()) { continue; }

                match face_model.extract_feature_for_db(img_path.to_str().unwrap()) {
                    Ok(features) => {
                        for feat in features {
                            person_embs.push(feat);
                        }
                    }
                    Err(_) => {}
                }
            }

            if !person_embs.is_empty() {
                embeddings.insert(identity.clone(), person_embs);
                log_info!("face_db: loaded {} refs for {}", embeddings[&identity].len(), identity);
            }
        }

        log_info!("face_db: total identities loaded: {}", embeddings.len());
        Ok(Self { embeddings, name_to_id })
    }

    pub fn query_id(&self, embedding: &[f32], threshold: f32) -> Option<(String, String)> {
        let mut best_score = threshold;
        let mut best_name  = None;

        for (name, embs) in &self.embeddings {
            for ref_emb in embs {
                let score = cosine_similarity(embedding, ref_emb);
                if score > best_score {
                    best_score = score;
                    best_name  = Some(name.clone());
                }
            }
        }
        
        if let Some(name) = best_name {
            let id = self.name_to_id.get(&name).cloned().unwrap_or_else(|| "unknown".to_string());
            Some((name, id))
        } else {
            None
        }
    }
}

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() { return 0.0; }
    let dot:   f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm1 * norm2 + 1e-8)
}

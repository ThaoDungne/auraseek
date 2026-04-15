/// aura model for text and vision embedding
use anyhow::Result;
use ort::value::Value;
use ort::session::Session;
use crate::infrastructure::ai::runtime::build_session;

pub struct AuraModel {
    vision_session: Session,
    #[allow(dead_code)]
    text_session:   Session,
}

impl AuraModel {
    pub fn new(vision_path: &str, text_path: &str) -> Result<Self> {
        Ok(Self {
            vision_session: build_session(vision_path)?,
            text_session:   build_session(text_path)?,
        })
    }

    /// encode text to embedding vector
    ///
    /// Accepts the full padded `input_ids` / `attention_mask` tensors (both
    /// length `max_len`). The model uses `attention_mask` to ignore padding
    /// positions, matching its training-time behaviour.
    pub fn encode_text(
        &mut self,
        input_ids:      Vec<i64>,
        attention_mask: Vec<i64>,
    ) -> Result<Vec<f32>> {
        let actual_len = input_ids.len();
        let ids_tensor  = Value::from_array((vec![1, actual_len], input_ids.into_boxed_slice()))?;
        let mask_tensor = Value::from_array((vec![1, actual_len], attention_mask.into_boxed_slice()))?;
        let outputs = self.text_session.run(ort::inputs![
            "input_ids"      => ids_tensor,
            "attention_mask" => mask_tensor,
        ])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec() as Vec<f32>)
    }

    /// encode image to embedding vector
    pub fn encode_image(
        &mut self,
        blob: Vec<f32>,
        w:    u32,
        h:    u32,
    ) -> Result<Vec<f32>> {
        let img_tensor = Value::from_array((
            vec![1, 3, h as usize, w as usize],
            blob.into_boxed_slice(),
        ))?;
        let outputs = self.vision_session.run(ort::inputs!["images" => img_tensor])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec() as Vec<f32>)
    }

    /// calculate cosine similarity between two vectors
    #[allow(dead_code)]
    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let dot:   f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm1 * norm2 + 1e-8)
    }
}

use anyhow::Result;

/// AI vision pipeline: object detection + segmentation.
pub trait ObjectDetector {
    type Detection;
    fn detect(&mut self, image_path: &str) -> Result<Vec<Self::Detection>>;
}

/// Face detection + recognition.
pub trait FaceDetector {
    type FaceResult;
    fn detect_faces(&mut self, image_path: &str) -> Result<Vec<Self::FaceResult>>;
}

/// Embedding encoder (text or image → vector).
pub trait EmbeddingEncoder {
    fn encode_image(&self, image_path: &str) -> Result<Vec<f32>>;
    fn encode_text(&self, text: &str) -> Result<Vec<f32>>;
}

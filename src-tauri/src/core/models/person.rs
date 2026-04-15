use serde::{Deserialize, Serialize};
use super::media::BboxInfo;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonGroup {
    pub face_id:     String,
    pub name:        Option<String>,
    pub photo_count: u32,
    pub cover_path:  Option<String>,
    pub thumbnail:   Option<String>,
    pub conf:        Option<f32>,
    pub face_bbox:   Option<BboxInfo>,
}

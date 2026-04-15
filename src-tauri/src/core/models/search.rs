use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    Text,
    Image,
    Combined,
    ObjectFilter,
    FaceFilter,
    FilterOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchQueryFilters {
    pub object:     Option<String>,
    pub face:       Option<String>,
    pub month:      Option<u32>,
    pub year:       Option<i32>,
    pub media_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub mode:       SearchMode,
    pub text:       Option<String>,
    pub image_path: Option<String>,
    pub filters:    SearchQueryFilters,
}

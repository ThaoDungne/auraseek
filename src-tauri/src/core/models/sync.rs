/// Current state of the background sync/ingest process.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SyncStatus {
    pub state:     String,
    pub processed: usize,
    pub total:     usize,
    pub message:   String,
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self { state: "idle".into(), processed: 0, total: 0, message: "Chưa đồng bộ".into() }
    }
}

use thiserror::Error;

/// Unified application error type.
///
/// Internal code should use `AppError` (or `AppResult<T>`) everywhere.
/// At the Tauri interface boundary, convert to `String` for the frontend.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("Database error: {0}")]
    Db(#[from] surrealdb::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("AI engine error: {0}")]
    Engine(String),

    #[error("Ingest error: {0}")]
    Ingest(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type AppResult<T> = Result<T, AppError>;

impl From<AppError> for String {
    fn from(e: AppError) -> String {
        e.to_string()
    }
}

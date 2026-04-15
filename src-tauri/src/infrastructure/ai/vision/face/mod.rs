pub mod detector;
pub mod db;

pub use detector::{FaceModel, FaceGroup, COSINE_THRESHOLD};
pub use db::{FaceDb, cosine_similarity};

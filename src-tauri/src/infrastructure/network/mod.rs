pub mod downloader;

pub use downloader::{
    ModelDownloader, DownloadProgress,
    all_models_present, get_surreal_bin_path,
};

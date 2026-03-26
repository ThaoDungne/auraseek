use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};
use tokio::fs;
use std::sync::atomic::{AtomicBool, Ordering};

const BASE_URL: &str =
    "https://github.com/ai-enthusiasm/auraseek/releases/download/v1.0.0";

/// List of (filename, relative-path-under-data-dir) for every required asset.
const ASSETS: &[(&str, &str)] = &[
    ("text_visclir.onnx",                "models/text_visclir.onnx"),
    ("vision_visclir.onnx",              "models/vision_visclir.onnx"),
    ("face_recognition_sface_2021dec.onnx", "models/face_recognition_sface_2021dec.onnx"),
    ("face_detection_yunet_2022mar.onnx",   "models/face_detection_yunet_2022mar.onnx"),
    ("yolo26n-seg.onnx",                    "models/yolo26n-seg.onnx"),
    ("bpe.codes",                           "tokenizer/bpe.codes"),
    ("vocab.txt",                           "tokenizer/vocab.txt"),
    ("DejaVuSans.ttf",                      "fonts/DejaVuSans.ttf"),
];

static IS_DOWNLOADING: AtomicBool = AtomicBool::new(false);

// ─── Progress event (sent to frontend) ───────────────────────────────────────

#[derive(Clone, serde::Serialize)]
pub struct DownloadProgress {
    /// Current file name being downloaded.
    pub file: String,
    /// Progress of the current file: 0.0 – 1.0.
    pub progress: f32,
    /// Human-readable status message.
    pub message: String,
    /// Set to true when all files are ready.
    pub done: bool,
    /// Non-empty when an error occurred.
    pub error: String,
    /// Index of current file (1-based).
    pub file_index: usize,
    /// Total number of files to download.
    pub file_total: usize,
    /// Bytes downloaded for current file.
    pub bytes_done: u64,
    /// Total bytes of current file (0 if unknown).
    pub bytes_total: u64,
}

// ─── Presence check ───────────────────────────────────────────────────────────

pub fn get_surreal_bin_path(data_dir: &Path) -> PathBuf {
    let bin_name = if cfg!(windows) { "surreal.exe" } else { "surreal" };
    data_dir.join("db").join(bin_name)
}

/// Return `true` if every required asset file exists in `data_dir` and `surreal` exists.
pub fn all_present(data_dir: &Path) -> bool {
    let models_ok = ASSETS.iter().all(|(_, rel)| data_dir.join(rel).exists());
    let surreal_ok = get_surreal_bin_path(data_dir).exists();
    models_ok && surreal_ok
}

// ─── Async download (used from cmd_init and cmd_download_models) ──────────────


/// Download any missing asset files to `data_dir`, emitting
/// `"model-download-progress"` events throughout.
///
/// Skips files that already exist.  Files are written via a `.tmp` staging
/// path and renamed atomically on completion.
pub async fn download_models_if_missing(app: &AppHandle, data_dir: &Path) -> Result<()> {
    if IS_DOWNLOADING.swap(true, Ordering::SeqCst) {
        crate::log_info!("Download already in progress, skipping duplicate request.");
        return Ok(());
    }

    let res = download_models_internal(app, data_dir).await;
    IS_DOWNLOADING.store(false, Ordering::SeqCst);
    res
}

async fn download_models_internal(app: &AppHandle, data_dir: &Path) -> Result<()> {
    if let Err(e) = download_surreal_if_missing_internal(app, data_dir).await {
        crate::log_error!("❌ Failed to download SurrealDB: {}", e);
        return Err(e);
    }

    // Collect only the files we actually need to download
    let needed: Vec<(&str, PathBuf)> = ASSETS
        .iter()
        .filter_map(|(name, rel)| {
            let dest = data_dir.join(rel);
            if dest.exists() { None } else { Some((*name, dest)) }
        })
        .collect();

    if needed.is_empty() {
        crate::log_info!("✅ All model assets already present");
        return Ok(());
    }

    let total = needed.len();
    crate::log_info!("📥 Downloading {} asset(s) to {}", total, data_dir.display());

    let client = Client::new();

    for (i, (name, dest)) in needed.iter().enumerate() {
        let file_index = i + 1;

        // Ensure parent directory exists
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).await
                .with_context(|| format!("create dir {}", parent.display()))?;
        }

        let url = format!("{}/{}", BASE_URL, name);
        crate::log_info!("📥 [{}/{}] {}", file_index, total, name);

        // Announce start
        let _ = app.emit("model-download-progress", DownloadProgress {
            file: name.to_string(),
            progress: 0.0,
            message: format!("Đang tải {} ({}/{})", name, file_index, total),
            done: false, error: String::new(),
            file_index, file_total: total,
            bytes_done: 0, bytes_total: 0,
        });

        let res = client.get(&url).send().await
            .with_context(|| format!("connect to {}", url))?;

        if !res.status().is_success() {
            let msg = format!("HTTP {} for {}", res.status(), name);
            let _ = app.emit("model-download-progress", DownloadProgress {
                file: name.to_string(), progress: 0.0,
                message: msg.clone(), done: false, error: msg.clone(),
                file_index, file_total: total, bytes_done: 0, bytes_total: 0,
            });
            anyhow::bail!("{}", msg);
        }

        let bytes_total = res.content_length().unwrap_or(0);
        let mut bytes_done: u64 = 0;
        let mut stream = res.bytes_stream();

        // Write to <dest>.tmp, rename on completion
        let tmp = dest.with_extension("tmp");
        let mut file = tokio::fs::File::create(&tmp).await
            .with_context(|| format!("create {}", tmp.display()))?;

        let mut last_emit = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read chunk")?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await
                .context("write chunk")?;
            bytes_done += chunk.len() as u64;

            // Throttle events to ~10 per second
            if last_emit.elapsed().as_millis() > 100 {
                last_emit = std::time::Instant::now();
                let progress = if bytes_total > 0 {
                    bytes_done as f32 / bytes_total as f32
                } else { 0.0 };
                let _ = app.emit("model-download-progress", DownloadProgress {
                    file: name.to_string(), progress,
                    message: format!("Đang tải {} ({}/{})", name, file_index, total),
                    done: false, error: String::new(),
                    file_index, file_total: total,
                    bytes_done, bytes_total,
                });
            }
        }

        tokio::io::AsyncWriteExt::flush(&mut file).await.context("flush")?;
        drop(file);
        fs::rename(&tmp, dest).await
            .with_context(|| format!("rename to {}", dest.display()))?;

        crate::log_info!("✅ Done: {}", name);

        let _ = app.emit("model-download-progress", DownloadProgress {
            file: name.to_string(), progress: 1.0,
            message: format!("Đã tải xong {} ({}/{})", name, file_index, total),
            done: false, error: String::new(),
            file_index, file_total: total,
            bytes_done, bytes_total,
        });
    }

    // Ensure face_db dir exists
    let face_db = data_dir.join("face_db");
    if !face_db.exists() {
        fs::create_dir_all(&face_db).await?;
    }

    // All done
    let _ = app.emit("model-download-progress", DownloadProgress {
        file: "done".into(), progress: 1.0,
        message: "Sẵn sàng khởi động AI Engine...".into(),
        done: true, error: String::new(),
        file_index: total, file_total: total,
        bytes_done: 0, bytes_total: 0,
    });

    crate::log_info!("✅ All model downloads complete");
    Ok(())
}

const SURREAL_VERSION: &str = "v3.0.2";

async fn download_surreal_if_missing_internal(app: &AppHandle, data_dir: &Path) -> Result<()> {
    let bin_dest = get_surreal_bin_path(data_dir);
    if bin_dest.exists() {
        return Ok(());
    }

    let target_os = std::env::consts::OS;
    let target_arch = std::env::consts::ARCH;

    let (asset_name, is_exe) = match (target_os, target_arch) {
        ("linux",   "x86_64")  => (format!("surreal-{}.linux-amd64.tgz",    SURREAL_VERSION), false),
        ("linux",   "aarch64") => (format!("surreal-{}.linux-arm64.tgz",    SURREAL_VERSION), false),
        ("windows", "x86_64")  => (format!("surreal-{}.windows-amd64.exe",  SURREAL_VERSION), true),
        ("macos",   "x86_64")  => (format!("surreal-{}.darwin-amd64.tgz",   SURREAL_VERSION), false),
        ("macos",   "aarch64") => (format!("surreal-{}.darwin-arm64.tgz",   SURREAL_VERSION), false),
        _ => {
            anyhow::bail!("SurrealDB: unsupported platform {}/{}", target_os, target_arch);
        }
    };

    let url = format!("https://download.surrealdb.com/{}/{}", SURREAL_VERSION, asset_name);
    crate::log_info!("📥 Downloading SurrealDB {} from {}", SURREAL_VERSION, url);

    if let Some(parent) = bin_dest.parent() {
        fs::create_dir_all(parent).await?;
    }

    let _ = app.emit("model-download-progress", DownloadProgress {
        file: "surrealdb".to_string(),
        progress: 0.0,
        message: format!("Đang tải cơ sở dữ liệu hệ thống (SurrealDB)..."),
        done: false, error: String::new(),
        file_index: 0, file_total: 1,
        bytes_done: 0, bytes_total: 0,
    });

    let client = Client::new();
    let res = client.get(&url).send().await
        .with_context(|| format!("connect to {}", url))?;

    if !res.status().is_success() {
        anyhow::bail!("HTTP {} for SurrealDB", res.status());
    }

    let bytes_total = res.content_length().unwrap_or(0);
    let mut bytes_done: u64 = 0;
    let mut stream = res.bytes_stream();

    let tmp = bin_dest.with_extension("tmp_download");
    let mut file = tokio::fs::File::create(&tmp).await
        .with_context(|| format!("create {}", tmp.display()))?;

    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read chunk")?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await
            .context("write chunk")?;
        bytes_done += chunk.len() as u64;

        if last_emit.elapsed().as_millis() > 100 {
            last_emit = std::time::Instant::now();
            let progress = if bytes_total > 0 {
                bytes_done as f32 / bytes_total as f32
            } else { 0.0 };
            let _ = app.emit("model-download-progress", DownloadProgress {
                file: "surrealdb".to_string(), progress,
                message: format!("Đang tải cơ sở dữ liệu hệ thống (SurrealDB)..."),
                done: false, error: String::new(),
                file_index: 0, file_total: 1,
                bytes_done, bytes_total,
            });
        }
    }

    tokio::io::AsyncWriteExt::flush(&mut file).await.context("flush")?;
    drop(file);

    if is_exe {
        fs::rename(&tmp, &bin_dest).await?;
    } else {
        // Unix: extract .tgz
        let _ = app.emit("model-download-progress", DownloadProgress {
            file: "surrealdb".to_string(), progress: 1.0,
            message: format!("Đang giải nén SurrealDB..."),
            done: false, error: String::new(),
            file_index: 0, file_total: 1,
            bytes_done, bytes_total,
        });

        // Use spawn_blocking for CPU-bound tar extraction
        let tmp_clone = tmp.clone();
        let parent = bin_dest.parent().unwrap().to_path_buf();
        
        tokio::task::spawn_blocking(move || -> Result<()> {
            use flate2::read::GzDecoder;
            use tar::Archive;
            let f = std::fs::File::open(&tmp_clone)?;
            let gz = GzDecoder::new(f);
            let mut archive = Archive::new(gz);
            for entry in archive.entries()? {
                let mut entry = entry?;
                let path = entry.path()?;
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name == "surreal" || name == "surreal.exe" {
                    entry.unpack(parent.join(name))?;
                    break;
                }
            }
            Ok(())
        }).await.context("spawn_blocking")?.context("tar extract")?;

        let _ = fs::remove_file(&tmp).await;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(mut perms) = std::fs::metadata(&bin_dest).map(|m| m.permissions()) {
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&bin_dest, perms);
            }
        }
    }

    crate::log_info!("✅ SurrealDB downloaded and ready at {}", bin_dest.display());
    Ok(())
}


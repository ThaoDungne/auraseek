//! Debug Model Downloader — CLI-only model downloading logic.
//!
//! This module is completely separate from production download logic.
//! It can be safely removed when building for production.

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs;

const BASE_URL: &str =
    "https://github.com/ai-enthusiasm/auraseek/releases/download/v1.0.0";

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

pub fn all_models_present(data_dir: &Path) -> bool {
    ASSETS.iter().all(|(_, rel)| data_dir.join(rel).exists())
}

pub struct DebugModelDownloader;

impl DebugModelDownloader {
    /// Download models for debug CLI usage (no AppHandle needed)
    pub async fn download_if_missing(data_dir: &Path) -> Result<()> {
        if IS_DOWNLOADING.swap(true, Ordering::SeqCst) {
            crate::log_info!("Download already in progress, skipping duplicate request.");
            return Ok(());
        }

        let res = Self::download_internal(data_dir).await;
        IS_DOWNLOADING.store(false, Ordering::SeqCst);
        res
    }

    async fn download_internal(data_dir: &Path) -> Result<()> {
        if let Err(e) = Self::download_surreal_if_missing(data_dir).await {
            crate::log_error!("❌ Failed to download SurrealDB: {}", e);
            return Err(e);
        }

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

            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).await
                    .with_context(|| format!("create dir {}", parent.display()))?;
            }

            let url = format!("{}/{}", BASE_URL, name);
            crate::log_info!("📥 [{}/{}] {}", file_index, total, name);

            let res = client.get(&url).send().await
                .with_context(|| format!("connect to {}", url))?;

            if !res.status().is_success() {
                let msg = format!("HTTP {} for {}", res.status(), name);
                crate::log_error!("❌ {}", msg);
                anyhow::bail!("{}", msg);
            }

            let bytes_total = res.content_length().unwrap_or(0);
            let mut bytes_done: u64 = 0;
            let mut stream = res.bytes_stream();

            let tmp = dest.with_extension("tmp");
            let mut file = tokio::fs::File::create(&tmp).await
                .with_context(|| format!("create {}", tmp.display()))?;

            let mut last_log = std::time::Instant::now();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.context("read chunk")?;
                tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await
                    .context("write chunk")?;
                bytes_done += chunk.len() as u64;

                if last_log.elapsed().as_millis() > 1000 {
                    last_log = std::time::Instant::now();
                    let progress = if bytes_total > 0 {
                        (bytes_done as f32 / bytes_total as f32) * 100.0
                    } else { 0.0 };
                    crate::log_info!("📥 [{}/{}] {}: {:.1}% ({:.1}MB/{:.1}MB)",
                        file_index, total, name, progress,
                        bytes_done as f32 / 1024.0 / 1024.0,
                        bytes_total as f32 / 1024.0 / 1024.0);
                }
            }

            tokio::io::AsyncWriteExt::flush(&mut file).await.context("flush")?;
            drop(file);
            fs::rename(&tmp, dest).await
                .with_context(|| format!("rename to {}", dest.display()))?;

            crate::log_info!("✅ Done: {}", name);
        }

        let face_db = data_dir.join("face_db");
        if !face_db.exists() {
            fs::create_dir_all(&face_db).await?;
        }

        crate::log_info!("✅ All model downloads complete");
        Ok(())
    }

    const SURREAL_VERSION: &str = "v3.0.2";

    async fn download_surreal_if_missing(data_dir: &Path) -> Result<()> {
        let bin_dest = data_dir.join("db").join(crate::platform::paths::surreal_binary_name());
        if bin_dest.exists() {
            return Ok(());
        }

        let target_os = std::env::consts::OS;
        let target_arch = std::env::consts::ARCH;

        let (asset_name, is_exe) = match (target_os, target_arch) {
            ("linux",   "x86_64")  => (format!("surreal-{}.linux-amd64.tgz",   Self::SURREAL_VERSION), false),
            ("linux",   "aarch64") => (format!("surreal-{}.linux-arm64.tgz",   Self::SURREAL_VERSION), false),
            ("windows", "x86_64")  => (format!("surreal-{}.windows-amd64.exe", Self::SURREAL_VERSION), true),
            ("macos",   "x86_64")  => (format!("surreal-{}.darwin-amd64.tgz",  Self::SURREAL_VERSION), false),
            ("macos",   "aarch64") => (format!("surreal-{}.darwin-arm64.tgz",  Self::SURREAL_VERSION), false),
            _ => {
                anyhow::bail!("SurrealDB: unsupported platform {}/{}", target_os, target_arch);
            }
        };

        let url = format!("https://download.surrealdb.com/{}/{}", Self::SURREAL_VERSION, asset_name);
        crate::log_info!("📥 Downloading SurrealDB {} from {}", Self::SURREAL_VERSION, url);

        if let Some(parent) = bin_dest.parent() {
            fs::create_dir_all(parent).await?;
        }

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

        let mut last_log = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read chunk")?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await
                .context("write chunk")?;
            bytes_done += chunk.len() as u64;

            if last_log.elapsed().as_millis() > 1000 {
                last_log = std::time::Instant::now();
                let progress = if bytes_total > 0 {
                    (bytes_done as f32 / bytes_total as f32) * 100.0
                } else { 0.0 };
                crate::log_info!("📥 SurrealDB: {:.1}% ({:.1}MB/{:.1}MB)",
                    progress,
                    bytes_done as f32 / 1024.0 / 1024.0,
                    bytes_total as f32 / 1024.0 / 1024.0);
            }
        }

        tokio::io::AsyncWriteExt::flush(&mut file).await.context("flush")?;
        drop(file);

        if is_exe {
            fs::rename(&tmp, &bin_dest).await?;
        } else {
            crate::log_info!("📦 Extracting SurrealDB...");

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

        crate::log_info!("✅ SurrealDB downloaded successfully");
        Ok(())
    }
}
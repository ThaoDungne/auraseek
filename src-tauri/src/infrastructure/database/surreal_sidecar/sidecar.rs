use anyhow::{Context, Result};
use std::fs;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

use crate::platform::process::hidden_command;
use std::fs::OpenOptions;

pub const DB_PORT: u16 = 39790;

pub struct SurrealService;

impl SurrealService {
    #[allow(dead_code)]
    pub fn find_free_port(start: u16, end: u16) -> Option<u16> {
        (start..=end).find(|&port| TcpListener::bind(("127.0.0.1", port)).is_ok())
    }

    pub fn find_binary(resource_dir: &Path, data_dir: &Path) -> Option<PathBuf> {
        let bin = crate::platform::paths::surreal_binary_name();

        let downloaded = data_dir.join(bin);
        if downloaded.exists() {
            crate::log_info!("🗄️  Found SurrealDB binary (downloaded): {}", downloaded.display());
            return Some(downloaded);
        }

        let candidate = resource_dir.join(bin);
        if candidate.exists() {
            crate::log_info!("🗄️  Found SurrealDB binary (resource): {}", candidate.display());
            return Some(candidate);
        }

        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let candidate = dir.join(bin);
                if candidate.exists() {
                    crate::log_info!("🗄️  Found SurrealDB binary (exe dir): {}", candidate.display());
                    return Some(candidate);
                }
            }
        }

        let dev = PathBuf::from("binaries").join(bin);
        if dev.exists() {
            crate::log_info!("🗄️  Found SurrealDB binary (dev binaries/): {}", dev.display());
            return Some(dev);
        }

        let which_cmd = crate::platform::paths::which_command();
        if let Ok(out) = hidden_command(which_cmd).arg("surreal").output() {
            let s = String::from_utf8_lossy(&out.stdout);
            let first_line = s.lines().next().unwrap_or("").trim().to_string();
            if !first_line.is_empty() {
                let p = PathBuf::from(&first_line);
                crate::log_info!("🗄️  Found SurrealDB binary (PATH): {}", p.display());
                return Some(p);
            }
        }

        crate::log_warn!("⚠️  SurrealDB binary not found. Install SurrealDB or ensure the binary is bundled.");
        None
    }

    pub fn start(
        binary:   &Path,
        data_dir: &Path,
        port:     u16,
        user:     &str,
        pass:     &str,
        db_uri:   &str,
    ) -> Result<Child> {
        std::fs::create_dir_all(data_dir)
            .context("Failed to create SurrealDB data directory")?;

        let bind_addr = format!("0.0.0.0:{}", port);

        crate::log_info!("🗄️  Starting SurrealDB | binary={} port={} uri={}",
            binary.display(), port, db_uri);

        let log_path = data_dir.join("surreal.log");
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("Failed to open SurrealDB log file {}", log_path.display()))?;
        let log_file_err = log_file.try_clone()
            .with_context(|| format!("Failed to clone SurrealDB log file handle {}", log_path.display()))?;

        let mut child = hidden_command(binary);
        let child = child
            .current_dir(data_dir)
            .args(["start", "--bind", &bind_addr, "--user", user, "--pass", pass, "--log", "warn", db_uri])
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err))
            .spawn()
            .with_context(|| format!("Failed to spawn SurrealDB from {}", binary.display()))?;

        crate::log_info!("✅ SurrealDB spawned (pid={})", child.id());
        Ok(child)
    }

    pub fn wait_ready(port: u16, max_secs: u64) -> Result<()> {
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse()?;
        let deadline = Instant::now() + Duration::from_secs(max_secs);

        while Instant::now() < deadline {
            if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok() {
                crate::log_info!("✅ SurrealDB ready on port {}", port);
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(400));
        }

        Err(anyhow::anyhow!(
            "SurrealDB did not start within {}s on port {}", max_secs, port
        ))
    }

    fn read_log_snippet(data_dir: &Path) -> Option<String> {
        let log_path = data_dir.join("surreal.log");
        let s = std::fs::read_to_string(&log_path).ok()?;
        let lines: Vec<&str> = s.lines().collect();
        let start = lines.len().saturating_sub(80);
        Some(lines[start..].join("\n"))
    }

    fn cleanup_stale_files(data_dir: &Path) -> Result<()> {
        let db_path = data_dir.join("auraseek.db");
        if db_path.exists() {
            let lock_path = db_path.join("LOCK");
            if lock_path.exists() {
                crate::log_warn!(
                    "⚠️  Stale SurrealDB lock file found, removing {}",
                    lock_path.display()
                );
                fs::remove_file(&lock_path)
                    .with_context(|| format!("Failed to remove stale lock file {}", lock_path.display()))?;

                let log_path = data_dir.join("surreal.log");
                if log_path.exists() {
                    crate::log_warn!(
                        "⚠️  Removing existing SurrealDB log file {} because stale lock was found",
                        log_path.display()
                    );
                    fs::remove_file(&log_path)
                        .with_context(|| format!("Failed to remove stale SurrealDB log file {}", log_path.display()))?;
                }

                let app_log_path = crate::core::config::AppConfig::global().log_path.clone();
                if app_log_path.exists() {
                    crate::log_warn!(
                        "⚠️  Removing stale AuraSeek app log file {}",
                        app_log_path.display()
                    );
                    fs::remove_file(&app_log_path)
                        .with_context(|| format!("Failed to remove stale AuraSeek app log file {}", app_log_path.display()))?;
                }
            }
        }
        Ok(())
    }

    pub fn ensure(
        resource_dir: &Path,
        data_dir:     &Path,
        user:         &str,
        pass:         &str,
    ) -> Result<(String, Option<Child>)> {
        let port = DB_PORT;

        let binary = Self::find_binary(resource_dir, data_dir)
            .ok_or_else(|| anyhow::anyhow!(
                "SurrealDB binary not found. It should have been downloaded on first launch."
            ))?;

        Self::cleanup_stale_files(data_dir)?;

        let kv_uri = "rocksdb://auraseek.db".to_string();
        let mut child = Self::start(&binary, data_dir, port, user, pass, &kv_uri)?;

        std::thread::sleep(Duration::from_millis(300));
        if let Ok(Some(status)) = child.try_wait() {
            let snippet = Self::read_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());

            if snippet.to_lowercase().contains("access is denied") || snippet.to_lowercase().contains("os error 5") {
                crate::log_warn!(
                    "⚠️  SurrealKV datastore access denied. Falling back to in-memory SurrealDB (mem://). Data will NOT persist. See {}/surreal.log",
                    data_dir.display()
                );
                let mut mem_child = Self::start(&binary, data_dir, port, user, pass, "mem://")?;
                std::thread::sleep(Duration::from_millis(200));
                if let Ok(Some(mem_status)) = mem_child.try_wait() {
                    let mem_snippet = Self::read_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());
                    anyhow::bail!(
                        "SurrealDB exited immediately in both SurrealKV and mem:// modes (kv_status={}, mem_status={}). See {}/surreal.log.\n{}",
                        status,
                        mem_status,
                        data_dir.display(),
                        mem_snippet
                    );
                }
                return Ok((format!("127.0.0.1:{}", port), Some(mem_child)));
            }

            anyhow::bail!(
                "SurrealDB exited immediately (status={}). See {}/surreal.log.\n{}",
                status,
                data_dir.display(),
                snippet
            );
        }

        if let Err(e) = Self::wait_ready(port, 3) {
            let snippet = Self::read_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());
            anyhow::bail!(
                "{}. SurrealDB is not accepting connections on 127.0.0.1:{} yet. See {}/surreal.log.\n{}",
                e,
                port,
                data_dir.display(),
                snippet
            );
        }

        Ok((format!("127.0.0.1:{}", port), Some(child)))
    }
}

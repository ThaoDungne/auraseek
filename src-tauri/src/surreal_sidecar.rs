//! SurrealDB sidecar management.
//!
//! Locates the bundled `surreal` binary, finds a free TCP port in [8000, 9000],
//! starts SurrealDB as a background child process, and waits for it to accept
//! connections.  The caller is responsible for storing and killing the returned
//! `Child` on app exit.

use anyhow::{Context, Result};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn hidden_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}
use std::fs::OpenOptions;

// Fixed TCP port for SurrealDB. If something else is already bound to this
// port, AuraSeek will fail fast instead of silently picking another port.
pub const DB_PORT: u16 = 39790;

// ─── Port helpers ─────────────────────────────────────────────────────────────

/// Return the first port in [start, end] on which nothing is listening (i.e.
/// `TcpListener::bind` succeeds).  Returns `None` if the whole range is occupied.
#[allow(dead_code)]
pub fn find_free_port(start: u16, end: u16) -> Option<u16> {
    (start..=end).find(|&port| TcpListener::bind(("127.0.0.1", port)).is_ok())
}

/// Return `true` if port accepts TCP (anything is listening).
#[allow(dead_code)]
fn is_port_open(port: u16) -> bool {
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// Return `true` only if the service on `port` is actually SurrealDB.
///
/// Sends a raw HTTP GET `/health` request (SurrealDB exposes this endpoint on
/// the same port as its WS interface).  Looks for an HTTP 200 response; any
/// other response or error means "not SurrealDB".
///
/// Using HTTP instead of just TCP connectivity avoids false positives where
/// another service (web server, proxy, etc.) is listening on the port but
/// would stall the WS handshake indefinitely.
#[allow(dead_code)]
pub fn is_surreal_on_port(port: u16) -> bool {
    use std::io::{Read, Write};

    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_millis(400)) {
        Ok(s) => s,
        Err(_) => return false,
    };

    stream.set_read_timeout(Some(Duration::from_millis(600))).ok();
    stream.set_write_timeout(Some(Duration::from_millis(400))).ok();

    let req = format!(
        "GET /health HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nAccept: application/json\r\n\r\n",
        port
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }

    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let resp = String::from_utf8_lossy(&buf[..n]);
            // SurrealDB returns "HTTP/1.1 200 OK" for a healthy instance
            resp.starts_with("HTTP/") && resp.contains(" 200 ")
        }
        _ => false,
    }
}

/// If a SurrealDB instance is already listening in [PORT_START, PORT_END],
/// return that port so we can reuse it (skips spawning a new process).
#[allow(dead_code)]
pub fn find_existing_surreal_port() -> Option<u16> {
    // With a fixed DB_PORT we only ever need to check that single port.
    if is_port_open(DB_PORT) && is_surreal_on_port(DB_PORT) {
        Some(DB_PORT)
    } else {
        None
    }
}

// ─── Binary location ──────────────────────────────────────────────────────────

/// Locate the `surreal` binary, trying (in order):
/// 1. `resource_dir/surreal[.exe]`     – Tauri-bundled location
/// 2. Same directory as the current executable (also covers installed builds)
/// 3. `binaries/surreal[.exe]`          – dev-mode layout (src-tauri/binaries/)
/// 4. System PATH via `which` / `where`
pub fn find_binary(resource_dir: &Path, data_dir: &Path) -> Option<PathBuf> {
    let bin = if cfg!(windows) { "surreal.exe" } else { "surreal" };

    // 0. Data dir (downloaded at runtime by downloader.rs)
    // The data_dir parameter here is already `<app_data>/db/`
    let downloaded = data_dir.join(bin);
    if downloaded.exists() {
        crate::log_info!("🗄️  Found SurrealDB binary (downloaded): {}", downloaded.display());
        return Some(downloaded);
    }

    // 1. Tauri resource dir (production bundle - legacy approach)
    let candidate = resource_dir.join(bin);
    if candidate.exists() {
        crate::log_info!("🗄️  Found SurrealDB binary (resource): {}", candidate.display());
        return Some(candidate);
    }

    // 2. Alongside the current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(bin);
            if candidate.exists() {
                crate::log_info!("🗄️  Found SurrealDB binary (exe dir): {}", candidate.display());
                return Some(candidate);
            }
        }
    }

    // 3. Dev-mode: src-tauri/binaries/surreal
    let dev = PathBuf::from("binaries").join(bin);
    if dev.exists() {
        crate::log_info!("🗄️  Found SurrealDB binary (dev binaries/): {}", dev.display());
        return Some(dev);
    }

    // 4. System PATH
    let which_cmd = if cfg!(windows) { "where" } else { "which" };
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

// ─── Process management ───────────────────────────────────────────────────────

/// Spawn a SurrealDB server process.
///
/// - `binary`:   path to the `surreal` executable
/// - `data_dir`: directory where SurrealDB stores its on-disk files
/// - `port`:     TCP port to bind (`0.0.0.0:<port>`)
/// - `user/pass`: root credentials
pub fn start_surreal(
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

    // Write sidecar logs into the data dir so we can debug startup failures
    // (permission issues, bad datastore URI, missing deps, etc.).
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

fn read_surreal_log_snippet(data_dir: &Path) -> Option<String> {
    let log_path = data_dir.join("surreal.log");
    let s = std::fs::read_to_string(&log_path).ok()?;
    // Keep last ~80 lines to avoid huge logs.
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(80);
    Some(lines[start..].join("\n"))
}

/// Poll until SurrealDB accepts a TCP connection or `max_secs` elapses.
pub fn wait_for_surreal(port: u16, max_secs: u64) -> Result<()> {
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

// ─── High-level entry point ───────────────────────────────────────────────────

/// Ensure a SurrealDB instance is available, returning its address string
/// (`"127.0.0.1:<port>"`) and an optional child handle (present only when we
/// spawned the process ourselves).
///
/// Strategy:
/// 1. If something is already listening in [PORT_START, PORT_END], assume it is
///    a previous AuraSeek-managed SurrealDB and reuse it.
/// 2. Otherwise find a free port, locate the bundled binary, and start a new
///    process.
pub fn ensure_surreal(
    resource_dir: &Path,
    data_dir:     &Path,
    user:         &str,
    pass:         &str,
) -> Result<(String, Option<Child>)> {
    // NOTE: We use a fixed TCP port (DB_PORT). If something else is already
    // bound to this port, AuraSeek will fail fast with a clear error instead
    // of silently picking another port, so that the frontend can surface a
    // useful message to the user.

    let port = DB_PORT;

    // Locate binary
    let binary = find_binary(resource_dir, data_dir)
        .ok_or_else(|| anyhow::anyhow!(
            "SurrealDB binary not found. It should have been downloaded on first launch."
        ))?;

    // Start SurrealDB.
    // Primary persistent datastore: SurrealKV (directory-backed).
    let kv_uri = "rocksdb://auraseek.db".to_string();
    let mut child = start_surreal(&binary, data_dir, port, user, pass, &kv_uri)?;

    // Many "connection refused" issues are simply the sidecar exiting immediately.
    // Check quickly and surface the SurrealDB log to the app log for debugging.
    std::thread::sleep(Duration::from_millis(300));
    if let Ok(Some(status)) = child.try_wait() {
        let snippet = read_surreal_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());

        // If the OS is denying filesystem access (common with Windows security policies),
        // fall back to an in-memory DB so the app can still run (non-persistent).
        if snippet.to_lowercase().contains("access is denied") || snippet.to_lowercase().contains("os error 5") {
            crate::log_warn!(
                "⚠️  SurrealKV datastore access denied. Falling back to in-memory SurrealDB (mem://). Data will NOT persist. See {}/surreal.log",
                data_dir.display()
            );
            let mut mem_child = start_surreal(&binary, data_dir, port, user, pass, "mem://")?;
            std::thread::sleep(Duration::from_millis(200));
            if let Ok(Some(mem_status)) = mem_child.try_wait() {
                let mem_snippet = read_surreal_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());
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

    // Wait briefly for the TCP port to open so cmd_init doesn't race it.
    if let Err(e) = wait_for_surreal(port, 3) {
        let snippet = read_surreal_log_snippet(data_dir).unwrap_or_else(|| "<no surreal.log>".into());
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

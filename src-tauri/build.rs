use std::fs;
use std::path::Path;
use anyhow::{Context, Result};

/// Pinned SurrealDB server version bundled with the app.
const SURREAL_VERSION: &str = "v3.0.2";

fn main() -> Result<()> {

    // Download SurrealDB binary BEFORE tauri_build::build() so Tauri's
    // external-binary validation can find the file.
    download_surreal_binary().ok(); // Non-fatal: app can still connect to an external instance.

    tauri_build::build();

    // rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
    
    Ok(())
}

// ─── SurrealDB binary download ────────────────────────────────────────────────

/// Download the SurrealDB server binary for the current build target and place
/// it at `binaries/surreal-{target-triple}[.exe]` so Tauri can bundle it as an
/// external binary.
fn download_surreal_binary() -> Result<()> {
    let target_os   = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target      = std::env::var("TARGET").unwrap_or_default();

    // Map Cargo target → SurrealDB release asset name
    let (asset_name, is_exe) = match (target_os.as_str(), target_arch.as_str()) {
        ("linux",   "x86_64")  => (format!("surreal-{}.linux-amd64.tgz",    SURREAL_VERSION), false),
        ("linux",   "aarch64") => (format!("surreal-{}.linux-arm64.tgz",    SURREAL_VERSION), false),
        ("windows", "x86_64")  => (format!("surreal-{}.windows-amd64.exe",  SURREAL_VERSION), true),
        ("macos",   "x86_64")  => (format!("surreal-{}.darwin-amd64.tgz",   SURREAL_VERSION), false),
        ("macos",   "aarch64") => (format!("surreal-{}.darwin-arm64.tgz",   SURREAL_VERSION), false),
        _ => {
            println!("cargo:warning=SurrealDB: unsupported platform {}/{}, skipping download", target_os, target_arch);
            return Ok(());
        }
    };

    // Tauri external-binary naming: binaries/surreal-{target-triple}[.exe]
    let bin_name = if is_exe {
        format!("binaries/surreal-{}.exe", target)
    } else {
        format!("binaries/surreal-{}", target)
    };

    if Path::new(&bin_name).exists() {
        println!("cargo:warning=SurrealDB binary already at {}, skipping download", bin_name);
        return Ok(());
    }

    fs::create_dir_all("binaries").context("Failed to create binaries/ directory")?;

    let url = format!("https://download.surrealdb.com/{}/{}", SURREAL_VERSION, asset_name);
    println!("cargo:warning=Downloading SurrealDB {} from {}", SURREAL_VERSION, url);

    if is_exe {
        // Windows: direct .exe download
        download_file(&url, &bin_name)?;
    } else {
        // Unix: download .tgz and extract the 'surreal' binary
        let tgz_tmp = "binaries/_surreal_tmp.tgz";
        download_file(&url, tgz_tmp)?;
        extract_surreal_from_tgz(tgz_tmp, &bin_name)?;
        let _ = fs::remove_file(tgz_tmp);

        // Mark executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&bin_name)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&bin_name, perms)?;
        }
    }

    println!("cargo:warning=SurrealDB binary ready: {}", bin_name);
    Ok(())
}

/// Extract the `surreal` binary from a `.tgz` archive.
fn extract_surreal_from_tgz(tgz_path: &str, output_path: &str) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let file = fs::File::open(tgz_path).context("Failed to open tgz")?;
    let gz   = GzDecoder::new(file);
    let mut archive = Archive::new(gz);

    for entry in archive.entries().context("Failed to read archive entries")? {
        let mut entry = entry.context("Bad archive entry")?;
        let path = entry.path().context("Bad entry path")?;
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if name == "surreal" || name == "surreal.exe" {
            entry.unpack(output_path).context("Failed to unpack surreal binary")?;
            return Ok(());
        }
    }

    Err(anyhow::anyhow!("'surreal' binary not found inside {}", tgz_path))
}

// ─── File downloader ──────────────────────────────────────────────────────────

fn download_file(url: &str, path: &str) -> Result<()> {
    // use ureq to download
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to download from {}", url))?;

    if response.status() != 200 {
        return Err(anyhow::anyhow!("Download failed with status: {}", response.status()));
    }

    let mut file = fs::File::create(path)
        .with_context(|| format!("Failed to create file: {}", path))?;
    
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("Failed to write to file: {}", path))?;

    Ok(())
}

use std::path::Path;

/// Platform-specific initialization that must run before the Tauri app starts.
pub fn pre_init() {
    #[cfg(target_os = "linux")]
    {
        // Workaround for WebKitGTK DRI2/hardware-acceleration crash on
        // NVIDIA proprietary drivers (DMABUF → DRI2Connect X11 errors).
        unsafe {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Future macOS-specific init (e.g. Sparkle updater, entitlements checks)
    }
}

/// Ensure native shared libraries are present next to the executable.
///
/// On **Windows** this copies bundled OpenCV / MSVC DLLs from the Tauri
/// resource directory into the exe directory so the app works on machines
/// without a global install.  On other platforms this is a no-op.
pub fn ensure_native_libs(resource_dir: &Path) -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Failed to get exe parent dir"))?;

        let dlls = [
            "opencv_world460.dll",
            "opencv_videoio_ffmpeg460_64.dll",
            "msvcp140.dll",
            "vcruntime140.dll",
            "concrt140.dll",
            "vcruntime140_1.dll",
            "msvcp140_1.dll",
        ];

        for dll in &dlls {
            let src = resource_dir.join("libs").join(dll);
            let dst = exe_dir.join(dll);

            if src.exists() && !dst.exists() {
                crate::log_info!("📦 Deploying system DLL to exe dir: {}", dll);
                if let Err(e) = std::fs::copy(&src, &dst) {
                    crate::log_warn!("⚠️ Failed to copy {}: {}", dll, e);
                }
            }
        }
    }

    let _ = resource_dir;
    Ok(())
}

use std::path::PathBuf;

/// Return the current user's home directory.
pub fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Platform-aware fallback when `app_data_dir()` fails.
pub fn fallback_data_dir() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("auraseek")
    }
    #[cfg(target_os = "macos")]
    {
        dirs_home().join("Library").join("Application Support").join("auraseek")
    }
    #[cfg(target_os = "linux")]
    {
        dirs_home().join(".local").join("share").join("auraseek")
    }
}

/// Default log file path for the current platform.
pub fn default_log_path() -> String {
    #[cfg(windows)]
    {
        std::env::var("APPDATA")
            .map(|p| format!("{}\\auraseek\\auraseek.log", p))
            .unwrap_or_else(|_| "auraseek.log".to_string())
    }
    #[cfg(target_os = "macos")]
    {
        dirs_home()
            .join("Library").join("Logs").join("auraseek.log")
            .to_string_lossy().to_string()
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("HOME")
            .map(|h| format!("{}/.local/share/auraseek/auraseek.log", h))
            .unwrap_or_else(|_| "/tmp/auraseek.log".to_string())
    }
}

/// SurrealDB binary file name for the current platform.
pub fn surreal_binary_name() -> &'static str {
    if cfg!(windows) { "surreal.exe" } else { "surreal" }
}

/// Command used to locate executables on the system PATH.
pub fn which_command() -> &'static str {
    if cfg!(windows) { "where" } else { "which" }
}

/// Trigger an OS-level authentication prompt (password / biometric).
///
/// - **Linux**: uses `pkexec` (Polkit)
/// - **Windows**: triggers a UAC elevation dialog via PowerShell
/// - **Other**: returns `Ok(true)` (no-op)
pub fn authenticate_os() -> Result<bool, String> {
    #[cfg(target_os = "linux")]
    {
        match std::process::Command::new("pkexec")
            .arg("true")
            .output()
        {
            Ok(output) => return Ok(output.status.success()),
            Err(e) => {
                crate::log_warn!("OS Auth failed: {}", e);
                return Err("Failed to trigger OS authentication.".to_string());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        match std::process::Command::new("powershell")
            .args(&[
                "-NoProfile",
                "-WindowStyle", "Hidden",
                "-Command",
                "Start-Process cmd -ArgumentList '/c exit 0' -Verb RunAs -WindowStyle Hidden -Wait",
            ])
            .output()
        {
            Ok(output) => return Ok(output.status.success()),
            Err(e) => {
                crate::log_warn!("Windows OS Auth failed: {}", e);
                return Err("Lỗi kích hoạt xác thực Windows".to_string());
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, use osascript to trigger a system authentication dialog.
        // Returns true if the user successfully authenticates.
        match std::process::Command::new("osascript")
            .args(&[
                "-e",
                "do shell script \"true\" with administrator privileges",
            ])
            .output()
        {
            Ok(output) => return Ok(output.status.success()),
            Err(e) => {
                crate::log_warn!("macOS OS Auth failed: {}", e);
                return Err("Failed to trigger macOS authentication.".to_string());
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        Ok(true)
    }
}

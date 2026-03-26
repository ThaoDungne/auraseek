use anyhow::Result;

fn main() -> Result<()> {
    tauri_build::build();
    // rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
    
    Ok(())
}

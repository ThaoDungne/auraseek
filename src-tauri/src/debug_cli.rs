use anyhow::Result;
use crate::processor::AuraSeekEngine;

pub fn run_debug_ingest(input_dir: &str, output_dir: &str) -> Result<()> {
    crate::log_info!("🛠️ Starting debug cli ingest mode...");
    crate::log_info!("Input dir: {}", input_dir);
    crate::log_info!("Output dir: {}", output_dir);

    // Create directories if they don't exist
    std::fs::create_dir_all(input_dir)?;
    std::fs::create_dir_all(output_dir)?;

    // Initialize engine
    let mut engine = AuraSeekEngine::new_default()?;
    
    // Process directory (this will extract all original debug artifacts: cropped masks, JSON, vectors, visualization PNGs/JPGs)
    engine.run_dir(input_dir, output_dir)?;

    crate::log_info!("✅ Debug ingest finished successfully.");
    Ok(())
}

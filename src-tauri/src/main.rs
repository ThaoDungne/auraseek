mod utils;
mod model;
mod processor;

use anyhow::Result;
use processor::AuraSeekEngine;
use utils::logger::Logger;
use processor::vision::preprocess_aura;
use model::AuraModel;

fn main() -> Result<()> {
    // initialize logger
    Logger::init("log/.log");
    
    // example 1: using engine
    let mut engine = AuraSeekEngine::new_default()?;
    engine.run_dir("input", "output")?;

    // ---------------------------------------------------------------------

    // example 2: text to image
    let text = "con mèo nằm trên laptop";
    let (input_ids, attention_mask) = engine.text_proc.encode(text, 64);
    let text_emb = engine.aura.encode_text(input_ids, attention_mask, 64)?;

    let image_path_1 = "input/cat-1.jpg";
    let image_blob_1 = preprocess_aura(image_path_1)?;
    let image_emb_1 = engine.aura.encode_image(image_blob_1, 256, 256)?;
    let similarity = AuraModel::cosine_similarity(&text_emb, &image_emb_1);

    println!("similarity text to image: {:?}", similarity);

    // example 3: image to image
    let image_path_2 = "input/cat-2.jpg";
    let image_blob_2 = preprocess_aura(image_path_2)?;
    let image_emb_2 = engine.aura.encode_image(image_blob_2, 256, 256)?;

    let image_path_3 = "input/cat-3.jpg";
    let image_blob_3 = preprocess_aura(image_path_3)?;
    let image_emb_3 = engine.aura.encode_image(image_blob_3, 256, 256)?;
    let similarity = AuraModel::cosine_similarity(&image_emb_2, &image_emb_3);

    println!("similarity image to image: {:?}", similarity);

    Ok(())
}

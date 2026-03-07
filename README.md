# AuraSeek

AuraSeek is an AI-powered image search and processing engine built with Rust, ONNX Runtime, and Tauri. It provides high-performance capabilities for semantic search, object detection, and face recognition.

## Features

- **Semantic Search**: Text-to-Image and Image-to-Image similarity calculation using Aura Model.
- **Object Detection & Segmentation**: Integrated YOLO26 for accurate object identification and masking.
- **Face Detection & Recognition**: Automated face detection and identification with a persistent face database.
- **High Performance**: Leverages ONNX Runtime with multiple execution providers and optimized CPU paths.
- **Automated Setup**: Automatic downloading of pre-trained models on first build.

---

## Technical Specifications

### Supported Models
- **Aura Model**: Developed by **AI Enthusiasm**, based on VinAI's `phobert-base-v2` and FacebookResearch's `dinov3_vits16`.
- **Object Detection**: `YOLO26n-seg` developed by **Ultralytics**, for real-time segmentation and detection.
- **Face Detection**: `YuNet` (2022 Mar) for fast and accurate face localization.
- **Face Recognition**: `SFace` (2021 Dec) for high-precision face identification.

### Supported Execution Providers
The engine automatically detects and prioritizes the best available hardware accelerator:
- **TensorRT**: Optimized for NVIDIA GPUs (Linux/Windows).
- **CUDA**: Standard NVIDIA GPU acceleration.
- **CoreML**: Hardware acceleration for Apple Silicon (macOS).
- **DirectML**: Optimized for Windows-based GPUs (AMD/Intel/NVIDIA).
- **OpenVINO**: Optimized for Intel CPUs and GPUs.
- **CPU**: Reliable fallback for all platforms.

### Supported Operating Systems
- **Linux** (Ubuntu/Debian/Arch/etc.)
- **Windows** (10/11)
- **macOS** (Intel & Apple Silicon)

---

## Engine Usage

The `AuraSeekEngine` provides several ways to interact with the models. Below are examples of common use cases.

### 1. Batch Processing
Process an entire directory of images to generate embeddings, detect objects, and recognize faces.

```rust
    // example 1: using engine
    let mut engine = AuraSeekEngine::new_default()?;
    engine.run_dir("input", "output")?;
```

### 2. Text-to-Image Similarity
Calculate how well a text description matches an image.

```rust
    // example 2: text to image
    let text = "con mèo nằm trên laptop";
    let (input_ids, attention_mask) = engine.text_proc.encode(text, 64);
    let text_emb = engine.aura.encode_text(input_ids, attention_mask, 64)?;

    let image_path_1 = "input/cat-1.jpg";
    let image_blob_1 = preprocess_aura(image_path_1)?;
    let image_emb_1 = engine.aura.encode_image(image_blob_1, 256, 256)?;
    let similarity = AuraModel::cosine_similarity(&text_emb, &image_emb_1);

    println!("similarity text to image: {:?}", similarity);
```

### 3. Image-to-Image Similarity
Compare two images to find visual similarity.

```rust
    // example 3: image to image
    let image_path_2 = "input/cat-2.jpg";
    let image_blob_2 = preprocess_aura(image_path_2)?;
    let image_emb_2 = engine.aura.encode_image(image_blob_2, 256, 256)?;

    let image_path_3 = "input/cat-3.jpg";
    let image_blob_3 = preprocess_aura(image_path_3)?;
    let image_emb_3 = engine.aura.encode_image(image_blob_3, 256, 256)?;
    let similarity = AuraModel::cosine_similarity(&image_emb_2, &image_emb_3);

    println!("similarity image to image: {:?}", similarity);
```

---

## Setup & Development

### Automated Model Download
The project uses a `build.rs` script that automatically downloads all required models and assets from GitHub Releases if they are missing from the `assets/` directory.

### Running the Project

To run the backend engine in development mode (CLI testing):
```bash
cd src-tauri
cargo run
```

To run the full Tauri application:
```bash
yarn tauri dev
```

## Dependencies
- **Rust**: Main programming language.
- **ONNX Runtime (`ort`)**: AI model inference.
- **OpenCV**: Image handling and face detection preprocessing.
- **Tauri**: Cross-platform GUI framework.

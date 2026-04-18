# AuraSeek

AI-powered photo and video search engine built with Tauri, React, and Rust. Combines vision-language embeddings, object detection, face recognition, and vector search to enable semantic media retrieval on the desktop. It provides high-performance capabilities leveraging ONNX Runtime with multiple execution providers.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Frontend (React + TypeScript + Tailwind/shadcn)    │
│  Timeline · People · Albums · Search · Filters      │
├─────────────────────────────────────────────────────┤
│  Tauri IPC (invoke commands)                        │
├─────────────────────────────────────────────────────┤
│  Backend (Rust)                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐│
│  │ AI Engine│ │ Search   │ │ Ingest Pipeline      ││
│  │ Aura     │ │ Text     │ │ Scan → Dedup → AI    ││
│  │ YOLO     │ │ Image    │ │ → Store embeddings   ││
│  │ YuNet    │ │ Combined │ │ → Detect faces/objs  ││
│  │ SFace    │ │ Filters  │ │ → Cluster faces      ││
│  └──────────┘ └──────────┘ └──────────────────────┘│
├─────────────────────────────────────────────────────┤
│  SurrealDB (media, embeddings, persons, history)    │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop Shell | Tauri 2 |
| Frontend | React 19, TypeScript, Vite 7, Tailwind CSS 4, shadcn/ui |
| Backend | Rust (tokio, serde, chrono, uuid) |
| AI Runtime | ONNX Runtime (ort 2.0) |
| Computer Vision | OpenCV 0.92 |
| Database | SurrealDB 3.0 (WebSocket) |
| Image Processing | image, imageproc, sha2 |

### Supported Operating Systems
- **Linux** (Ubuntu/Debian/Arch/etc.)
- **Windows** (10/11)
- **macOS** (Intel & Apple Silicon)

### Supported Execution Providers
The engine automatically detects and prioritizes the best available hardware accelerator:
- **TensorRT**: Optimized for NVIDIA GPUs (Linux/Windows).
- **CUDA**: Standard NVIDIA GPU acceleration.
- **CoreML**: Hardware acceleration for Apple Silicon (macOS).
- **DirectML**: Optimized for Windows-based GPUs (AMD/Intel/NVIDIA).
- **OpenVINO**: Optimized for Intel CPUs and GPUs.
- **CPU**: Reliable fallback for all platforms.

## AI Models

Models are downloaded automatically from GitHub releases during `cargo build`.

| Model | File | Purpose | Details |
|-------|------|---------|---------|
| Aura Vision Tower | `vision_tower_aura.onnx` | Image → embedding vector | Developed by AI Enthusiasm. Based on dinov3_vits16. |
| Aura Text Tower | `text_tower_aura.onnx` | Text → embedding vector | Developed by AI Enthusiasm. Based on phobert-base-v2. |
| YOLO26-Seg | `yolo26n-seg.onnx` | Object detection + segmentation | Developed by Ultralytics. Returns 80 COCO classes. |
| YuNet | `face_detection_yunet_2022mar.onnx` | Face detection w/ bboxes | 2022 Mar version for fast/accurate localization. |
| SFace | `face_recognition_sface_2021dec.onnx` | Face embedding | 2021 Dec version for high-precision clustering. |

## Features

### Media Management
- **Folder scanning** with SHA-256 deduplication
- **Timeline view** grouped by year/month
- **Favorites** with persistent toggle
- **Duplicate detection** by file hash
- **Albums** auto-generated from detected object tags

### AI-Powered Search
- **Text search** — natural language queries encoded via PhoBERT + Aura text tower
- **Image search** — query by image, encoded via Aura vision tower
- **Combined search** — text + image with averaged similarity scores
- **Object filter** — filter by COCO class (person, dog, car, etc.)
- **Face filter** — filter by recognized person
- **Advanced filters** — year, month, media type, combinable with any search mode
- **Vector similarity** — cosine similarity on SurrealDB-stored embeddings

### Face Recognition
- **Detection pipeline**: YOLO person bbox → crop → YuNet face detection → SFace embedding
- **Clustering**: cosine similarity matching (threshold 0.55) across sessions
- **Person management**: auto-numbered groups, user-renamable
- **Cropped avatars**: face bbox stored per person, used for circular avatar crops in People view
- **Confidence scores**: stored per person, highest-confidence detection kept as representative

### Object Detection & Segmentation
- **80 COCO classes** detected per image with confidence scores
- **Bounding boxes** stored per object/face in the `media` table
- **Hover overlays** — bbox rectangles rendered on photo hover (cyan for objects, violet for faces)
- **Distinct object list** loaded from DB for filter panel

### UI/UX
- **Vietnamese IME support** — uncontrolled input with composition event handling (Telex, VNI, etc.)
- **Dark/light theme** toggle
- **Selection mode** for batch operations
- **Responsive grid** layouts with dynamic column count
- **Full-screen photo viewer**
- **Search history** persisted in SurrealDB
- **Filter panel** with live data from DB (detected objects + recognized persons)
- **Empty state messaging** when filters return no results

## Engine Usage

The `AuraSeekEngine` provides several ways to interact with the models directly in Rust. Below are examples of common use cases.

### 1. Batch Processing
Process an entire directory of images to generate embeddings, detect objects, and recognize faces.

### 2. Text-to-Image Similarity
Calculate how well a text description matches an image.

### 3. Image-to-Image Similarity
Compare two images to find visual similarity.

## Database Schema

```sql
-- Media: photos/videos with AI annotations
media (SCHEMAFULL)
  ├── media_type, source
  ├── file { path, name, size, sha256, phash }
  ├── metadata { width, height, duration, fps, created_at, modified_at }
  ├── objects[] { class_name, conf, bbox { x, y, w, h }, mask_area, mask_path }
  ├── faces[] { face_id, name, conf, bbox { x, y, w, h } }
  ├── processed, favorite

-- Embedding: vector storage for similarity search
embedding (SCHEMAFULL)
  ├── media_id → record<media>
  ├── source, frame_ts, frame_idx
  ├── vec: array<float>

-- Person: face cluster registry
person (SCHEMAFULL)
  ├── face_id (UNIQUE), name, thumbnail
  ├── conf, face_bbox { x, y, w, h }

-- Search history
search_history (SCHEMAFULL)
  ├── query, image_path, filters
```

## Project Structure

```
auraseek/
├── src/                        # React frontend
│   ├── views/                  # Page-level components
│   │   ├── timeline/           #   Photo timeline
│   │   ├── people/             #   Face groups
│   │   ├── search/             #   Search results
│   │   ├── gallery/            #   Filtered gallery
│   │   ├── albums/             #   Auto-generated albums
│   │   └── duplicates/         #   Duplicate finder
│   ├── components/
│   │   ├── layout/             #   Sidebar, Topbar
│   │   ├── photos/             #   PhotoCard, PhotoGrid
│   │   ├── common/             #   FilterPanel, Settings
│   │   └── ui/                 #   shadcn primitives
│   ├── lib/api.ts              # Tauri command bindings
│   └── types/photo.type.ts     # Frontend type definitions
│
├── src-tauri/                  # Rust backend
│   ├── src/
│   │   ├── main.rs             # Tauri commands + app state
│   │   ├── model/              # AI model wrappers
│   │   │   ├── aura.rs         #   Vision/text embeddings
│   │   │   ├── yolo.rs         #   Object detection
│   │   │   └── face.rs         #   Face detect + recognize
│   │   ├── processor/          # AI orchestration engine
│   │   ├── db/                 # SurrealDB layer
│   │   │   ├── surreal.rs      #   Connection + schema
│   │   │   ├── models.rs       #   Rust structs
│   │   │   └── operations.rs   #   Queries + filters
│   │   ├── search/             # Search pipeline
│   │   └── ingest/             # File scanning + processing
│   ├── assets/models/          # ONNX model files
│   └── Cargo.toml
│
├── vite.config.ts
├── package.json
└── tsconfig.json
```

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- [Node.js](https://nodejs.org/) (v18+)
- [SurrealDB](https://surrealdb.com/install) (v2+)
- OpenCV 4.x system libraries
- ONNX Runtime (bundled via `ort` crate)

### Setup

```bash
# Start SurrealDB
surreal start --user root --pass root --bind 0.0.0.0:8000 surrealkv:auraseek_data

# Install frontend dependencies
yarn install
```

### Running the Project

To run the backend engine in development mode (CLI testing):
```bash
cd src-tauri
cargo run
```

To run the full Tauri application in development mode (builds Rust + starts Vite dev server):
```bash
yarn tauri dev
```

### Build for Production

```bash
yarn tauri build
```

The project uses a `build.rs` script that automatically downloads all required models and assets from GitHub Releases if they are missing from the `assets/` directory during first build.

## Tauri Commands

| Command | Description |
|---------|------------|
| `cmd_init` | Initialize AI engine + DB connection |
| `cmd_scan_folder` | Scan directory, ingest media, run AI |
| `cmd_search_text` | Semantic text search with filters |
| `cmd_search_image` | Search by image similarity |
| `cmd_search_combined` | Text + image search |
| `cmd_search_object` | Filter by COCO object class |
| `cmd_search_face` | Filter by person name |
| `cmd_search_filter_only` | Year/month/type filter without search |
| `cmd_get_timeline` | Get media grouped by date |
| `cmd_get_people` | Get face clusters with counts |
| `cmd_name_person` | Rename a face cluster |
| `cmd_toggle_favorite` | Toggle media favorite status |
| `cmd_get_distinct_objects` | List all detected object classes |
| `cmd_get_duplicates` | Get duplicate file groups |

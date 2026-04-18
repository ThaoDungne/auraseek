/// SurrealDB connection module.
///
/// Strategy: try WebSocket first (lower latency for streaming queries).
/// If WS times out after 10s (e.g. AppImage sandbox, proxy, or broken WS
/// upgrade), fall back to HTTP which is simpler and more portable.
use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::opt::auth::Root;
use std::time::Duration;

const NS: &str = "auraseek";
const DB_NAME: &str = "auraseek";
const WS_TIMEOUT_SECS: u64 = 10;
const HTTP_TIMEOUT_SECS: u64 = 10;

#[derive(Clone)]
pub struct SurrealDb {
    pub db: Surreal<Any>,
}

impl SurrealDb {
    /// Connect to SurrealDB: WS first, fallback HTTP.
    pub async fn connect(addr: &str, user: &str, pass: &str) -> Result<Self> {
        // ── 1. Try WebSocket ─────────────────────────────────────────────────
        let ws_url = format!("ws://{}", addr);
        crate::log_info!("🔌 Connecting to SurrealDB via WS: {}...", ws_url);

        match tokio::time::timeout(
            Duration::from_secs(WS_TIMEOUT_SECS),
            surrealdb::engine::any::connect(&ws_url),
        ).await {
            Ok(Ok(db)) => {
                crate::log_info!("✅ WS connection established");
                return Self::finish_connect(db, addr, user, pass, "ws").await;
            }
            Ok(Err(e)) => {
                crate::log_warn!("⚠️  WS connection failed: {}. Falling back to HTTP...", e);
            }
            Err(_) => {
                crate::log_warn!("⚠️  WS connection timed out after {}s. Falling back to HTTP...", WS_TIMEOUT_SECS);
            }
        }

        // ── 2. Fallback: HTTP ────────────────────────────────────────────────
        let http_url = format!("http://{}", addr);
        crate::log_info!("🔌 Connecting to SurrealDB via HTTP: {}...", http_url);

        let db = tokio::time::timeout(
            Duration::from_secs(HTTP_TIMEOUT_SECS),
            surrealdb::engine::any::connect(&http_url),
        )
        .await
        .map_err(|_| anyhow::anyhow!("HTTP connection also timed out after {}s to {}", HTTP_TIMEOUT_SECS, addr))?
        .context(format!("HTTP connection failed to SurrealDB at {}", http_url))?;

        crate::log_info!("✅ HTTP connection established");
        Self::finish_connect(db, addr, user, pass, "http").await
    }

    /// Authenticate, select namespace/database, run schema, return `SurrealDb`.
    async fn finish_connect(
        db: Surreal<Any>,
        addr: &str,
        user: &str,
        pass: &str,
        proto: &str,
    ) -> Result<Self> {
        crate::log_info!("🔑 Authenticating as user '{}'...", user);

        tokio::time::timeout(
            Duration::from_secs(5),
            db.signin(Root {
                username: user.to_string(),
                password: pass.to_string(),
            })
        )
        .await
        .map_err(|_| anyhow::anyhow!("Authentication timeout"))?
        .context("SurrealDB authentication failed. Check username/password.")?;

        crate::log_info!("📁 Selecting namespace='{}' database='{}'", NS, DB_NAME);
        db.use_ns(NS).use_db(DB_NAME).await
            .context("Failed to select namespace/database")?;

        crate::log_info!("✅ SurrealDB connected via {} | addr={} ns={} db={}", proto, addr, NS, DB_NAME);

        let s = Self { db };
        s.ensure_schema().await?;
        crate::log_info!("📋 Schema ready: media, embedding, person, config_auraseek, search_history");
        Ok(s)
    }

    async fn ensure_schema(&self) -> Result<()> {
        self.db.query("
            -- Migration: remove legacy fields no longer stored in media
            REMOVE FIELD IF EXISTS file.path ON media;
            REMOVE FIELD IF EXISTS source ON media;
        ").await
        .context("Failed to run schema migrations")?;

        self.db.query("
            -- App config table (source folder path, etc.)
            DEFINE TABLE IF NOT EXISTS config_auraseek SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS source_dir   ON config_auraseek TYPE string;
            DEFINE FIELD IF NOT EXISTS updated_at   ON config_auraseek TYPE datetime DEFAULT time::now();

            -- Media table (path is derived at runtime as source_dir + '/' + file.name)
            DEFINE TABLE IF NOT EXISTS media SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS media_type   ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS file         ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS file.name    ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS file.size    ON media TYPE int;
            DEFINE FIELD IF NOT EXISTS file.sha256  ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS file.phash   ON media TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS metadata     ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS metadata.width      ON media TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS metadata.height     ON media TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS metadata.duration   ON media TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS metadata.fps        ON media TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS metadata.created_at  ON media TYPE option<datetime>;
            DEFINE FIELD IF NOT EXISTS metadata.modified_at ON media TYPE option<datetime>;
            DEFINE FIELD IF NOT EXISTS objects           ON media TYPE array DEFAULT [];
            DEFINE FIELD IF NOT EXISTS objects.*         ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS objects.*.class_name ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS objects.*.conf       ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS objects.*.bbox       ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS objects.*.bbox.x    ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS objects.*.bbox.y    ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS objects.*.bbox.w    ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS objects.*.bbox.h    ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS objects.*.mask_area ON media TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS objects.*.mask_path ON media TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS objects.*.mask_rle  ON media TYPE option<array>;
            DEFINE FIELD IF NOT EXISTS faces              ON media TYPE array DEFAULT [];
            DEFINE FIELD IF NOT EXISTS faces.*            ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS faces.*.face_id    ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS faces.*.name       ON media TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS faces.*.conf       ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS faces.*.bbox       ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS faces.*.bbox.x     ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS faces.*.bbox.y     ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS faces.*.bbox.w     ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS faces.*.bbox.h     ON media TYPE float;
            DEFINE FIELD IF NOT EXISTS processed    ON media TYPE bool DEFAULT false;
            DEFINE FIELD IF NOT EXISTS favorite     ON media TYPE bool DEFAULT false;
            DEFINE FIELD IF NOT EXISTS deleted_at   ON media TYPE option<datetime>;
            DEFINE FIELD IF NOT EXISTS is_hidden    ON media TYPE bool DEFAULT false;
            DEFINE FIELD IF NOT EXISTS thumbnail    ON media TYPE option<string>;
            -- SHA-256 được dùng cho chức năng \"Khử trùng lặp\" nhưng KHÔNG còn
            -- ép uniqueness nữa, để cho phép lưu nhiều bản sao giống hệt nhau.
            DEFINE INDEX IF NOT EXISTS idx_sha256   ON media FIELDS file.sha256;
            DEFINE INDEX IF NOT EXISTS idx_created  ON media FIELDS metadata.created_at;

            -- Migration for existing records
            UPDATE media SET deleted_at = NONE, is_hidden = false WHERE is_hidden = NONE;

            -- Embedding table with vector field
            DEFINE TABLE IF NOT EXISTS embedding SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS media_id     ON embedding TYPE record<media>;
            DEFINE FIELD IF NOT EXISTS source       ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS frame_ts     ON embedding TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS frame_idx    ON embedding TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS vec          ON embedding TYPE array<float>;
            DEFINE FIELD IF NOT EXISTS created_at   ON embedding TYPE datetime DEFAULT time::now();
            DEFINE INDEX IF NOT EXISTS idx_emb_media ON embedding FIELDS media_id;

            -- Person / face cluster table
            DEFINE TABLE IF NOT EXISTS person SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS face_id      ON person TYPE string;
            DEFINE FIELD IF NOT EXISTS name         ON person TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS thumbnail    ON person TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS conf         ON person TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS face_bbox    ON person TYPE option<object>;
            DEFINE FIELD IF NOT EXISTS face_bbox.x  ON person TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS face_bbox.y  ON person TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS face_bbox.w  ON person TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS face_bbox.h  ON person TYPE option<float>;
            DEFINE FIELD IF NOT EXISTS created_at   ON person TYPE datetime DEFAULT time::now();
            DEFINE INDEX IF NOT EXISTS idx_face_id  ON person FIELDS face_id UNIQUE;

            -- Search history
            DEFINE TABLE IF NOT EXISTS search_history SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS query        ON search_history TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS image_path   ON search_history TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS filters      ON search_history TYPE option<object>;
            DEFINE FIELD IF NOT EXISTS created_at   ON search_history TYPE datetime DEFAULT time::now();
        ").await
        .context("Failed to send SurrealDB schema query")?
        .check()
        .map_err(|e| anyhow::anyhow!("Schema creation failed: {}", e))?;

        Ok(())
    }
}

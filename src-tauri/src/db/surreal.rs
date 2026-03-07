/// SurrealDB connection module
use anyhow::{Context, Result};
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::{Ws, Client};
use surrealdb::opt::auth::Root;
use std::time::Duration;

const NS: &str = "auraseek";
const DB_NAME: &str = "auraseek";
const CONNECT_TIMEOUT_SECS: u64 = 10;

pub struct SurrealDb {
    pub db: Surreal<Client>,
}

impl SurrealDb {
    /// Connect to a SurrealDB instance with timeout.
    pub async fn connect(addr: &str, user: &str, pass: &str) -> Result<Self> {
        crate::log_info!("🔌 Connecting to SurrealDB at ws://{}...", addr);

        // Connect with timeout
        let db = tokio::time::timeout(
            Duration::from_secs(CONNECT_TIMEOUT_SECS),
            Surreal::new::<Ws>(addr)
        )
        .await
        .map_err(|_| anyhow::anyhow!("Connection timeout after {}s to SurrealDB at {}", CONNECT_TIMEOUT_SECS, addr))?
        .context(format!("Failed to connect to SurrealDB at {}", addr))?;

        crate::log_info!("🔑 Authenticating as user '{}'...", user);

        // Auth with timeout
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

        crate::log_info!("✅ SurrealDB connected | ns={} db={}", NS, DB_NAME);

        let s = Self { db };
        s.ensure_schema().await?;
        crate::log_info!("📋 SurrealDB schema verified (media, embedding, person, search_history)");
        Ok(s)
    }

    /// Create tables + indexes (idempotent via DEFINE ... IF NOT EXISTS)
    async fn ensure_schema(&self) -> Result<()> {
        self.db.query("
            -- Media table
            DEFINE TABLE IF NOT EXISTS media SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS media_type   ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS source       ON media TYPE string;
            DEFINE FIELD IF NOT EXISTS file         ON media TYPE object;
            DEFINE FIELD IF NOT EXISTS file.path    ON media TYPE string;
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
            DEFINE INDEX IF NOT EXISTS idx_sha256   ON media FIELDS file.sha256 UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_created  ON media FIELDS metadata.created_at;

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

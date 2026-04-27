use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use memory_core::{MemoryLink, MemoryRecord, MemoryType};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    Row, SqlitePool,
};
use std::str::FromStr;
use uuid::Uuid;

#[derive(Clone)]
pub struct SqliteMemoryStore {
    pool: SqlitePool,
}

impl SqliteMemoryStore {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let database_url = if database_url.starts_with("sqlite:") {
            database_url.to_string()
        } else {
            format!("sqlite://{database_url}")
        };

        let options = SqliteConnectOptions::from_str(&database_url)
            .with_context(|| format!("invalid SQLite database URL: {database_url}"))?
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("failed to open SQLite database")?;

        let store = Self { pool };
        store.init().await?;
        Ok(store)
    }

    pub async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                tags_json TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding_model TEXT,
                embedding_json TEXT,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT
            );
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to initialize memories table")?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_memories_memory_type
            ON memories(memory_type);
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to create memory type index")?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to create memory_links table")?;

        Ok(())
    }

    pub async fn insert_memory(&self, memory: &MemoryRecord) -> Result<()> {
        let tags_json = serde_json::to_string(&memory.tags)?;
        let embedding_json = match &memory.embedding {
            Some(embedding) => Some(serde_json::to_string(embedding)?),
            None => None,
        };

        sqlx::query(
            r#"
            INSERT INTO memories (
                id,
                content,
                memory_type,
                importance,
                confidence,
                tags_json,
                source,
                embedding_model,
                embedding_json,
                created_at,
                last_accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            "#,
        )
        .bind(memory.id.to_string())
        .bind(&memory.content)
        .bind(memory.memory_type.to_string())
        .bind(memory.importance)
        .bind(memory.confidence)
        .bind(tags_json)
        .bind(&memory.source)
        .bind(&memory.embedding_model)
        .bind(embedding_json)
        .bind(memory.created_at.to_rfc3339())
        .bind(memory.last_accessed_at.as_ref().map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await
        .context("failed to insert memory")?;

        Ok(())
    }

    pub async fn list_memories(&self) -> Result<Vec<MemoryRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id,
                content,
                memory_type,
                importance,
                confidence,
                tags_json,
                source,
                embedding_model,
                embedding_json,
                created_at,
                last_accessed_at
            FROM memories
            ORDER BY created_at DESC;
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to list memories")?;

        rows.into_iter().map(row_to_memory).collect()
    }

    pub async fn mark_accessed(&self, id: Uuid) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE memories
            SET last_accessed_at = ?
            WHERE id = ?;
            "#,
        )
        .bind(Utc::now().to_rfc3339())
        .bind(id.to_string())
        .execute(&self.pool)
        .await
        .context("failed to update last_accessed_at")?;

        Ok(())
    }

    pub async fn delete_memory(&self, id: Uuid) -> Result<()> {
        sqlx::query(
            r#"
            DELETE FROM memories
            WHERE id = ?;
            "#,
        )
        .bind(id.to_string())
        .execute(&self.pool)
        .await
        .context("failed to delete memory")?;

        Ok(())
    }

    pub async fn link_memories(&self, source_id: Uuid, target_id: Uuid, relation_type: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO memory_links (source_id, target_id, relation_type)
            VALUES (?, ?, ?);
            "#,
        )
        .bind(source_id.to_string())
        .bind(target_id.to_string())
        .bind(relation_type)
        .execute(&self.pool)
        .await
        .context("failed to link memories")?;

        Ok(())
    }

    pub async fn get_linked_memories(&self, id: Uuid) -> Result<Vec<MemoryLink>> {
        let rows = sqlx::query(
            r#"
            SELECT source_id, target_id, relation_type
            FROM memory_links
            WHERE source_id = ? OR target_id = ?;
            "#,
        )
        .bind(id.to_string())
        .bind(id.to_string())
        .fetch_all(&self.pool)
        .await
        .context("failed to get linked memories")?;

        rows.into_iter()
            .map(|row| {
                let source_id: String = row.get("source_id");
                let target_id: String = row.get("target_id");
                let relation_type: String = row.get("relation_type");

                Ok(MemoryLink {
                    source_id: Uuid::parse_str(&source_id).context("invalid source_id")?,
                    target_id: Uuid::parse_str(&target_id).context("invalid target_id")?,
                    relation_type,
                })
            })
            .collect()
    }
}

fn row_to_memory(row: sqlx::sqlite::SqliteRow) -> Result<MemoryRecord> {
    let id: String = row.get("id");
    let memory_type: String = row.get("memory_type");
    let tags_json: String = row.get("tags_json");
    let embedding_json: Option<String> = row.get("embedding_json");
    let created_at: String = row.get("created_at");
    let last_accessed_at: Option<String> = row.get("last_accessed_at");

    Ok(MemoryRecord {
        id: Uuid::parse_str(&id).context("invalid memory id")?,
        content: row.get("content"),
        memory_type: MemoryType::from_str(&memory_type).map_err(anyhow::Error::msg)?,
        importance: row.get::<f32, _>("importance"),
        confidence: row.get::<f32, _>("confidence"),
        tags: serde_json::from_str(&tags_json).context("invalid tags JSON")?,
        source: row.get("source"),
        embedding_model: row.get("embedding_model"),
        embedding: match embedding_json {
            Some(value) => Some(serde_json::from_str(&value).context("invalid embedding JSON")?),
            None => None,
        },
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .context("invalid created_at timestamp")?
            .with_timezone(&Utc),
        last_accessed_at: match last_accessed_at {
            Some(value) => Some(
                DateTime::parse_from_rfc3339(&value)
                    .context("invalid last_accessed_at timestamp")?
                    .with_timezone(&Utc),
            ),
            None => None,
        },
    })
}

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    Working,
    Episodic,
    Semantic,
    Tool,
    Failure,
    Preference,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            MemoryType::Working => "working",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Tool => "tool",
            MemoryType::Failure => "failure",
            MemoryType::Preference => "preference",
        };
        write!(f, "{value}")
    }
}

impl FromStr for MemoryType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "working" => Ok(MemoryType::Working),
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "tool" => Ok(MemoryType::Tool),
            "failure" => Ok(MemoryType::Failure),
            "preference" => Ok(MemoryType::Preference),
            other => Err(format!(
                "unknown memory type '{other}'. Use one of: working, episodic, semantic, tool, failure, preference"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub content: String,
    pub memory_type: MemoryType,
    pub importance: f32,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub source: String,
    pub embedding_model: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: Option<DateTime<Utc>>,
}

impl MemoryRecord {
    pub fn new(
        content: impl Into<String>,
        memory_type: MemoryType,
        importance: f32,
        confidence: f32,
        tags: Vec<String>,
        source: impl Into<String>,
        embedding_model: Option<String>,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            memory_type,
            importance: importance.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            tags,
            source: source.into(),
            embedding_model,
            embedding,
            created_at: Utc::now(),
            last_accessed_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub memory: MemoryRecord,
    pub semantic_similarity: f32,
    pub importance_score: f32,
    pub recency_score: f32,
    pub confidence_score: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub relation_type: String,
}

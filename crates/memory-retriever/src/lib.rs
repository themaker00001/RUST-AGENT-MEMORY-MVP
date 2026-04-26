use chrono::Utc;
use memory_core::{MemoryRecord, RetrievedMemory};

#[derive(Debug, Clone)]
pub struct RetrievalWeights {
    pub semantic: f32,
    pub importance: f32,
    pub recency: f32,
    pub confidence: f32,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        Self {
            semantic: 0.55,
            importance: 0.20,
            recency: 0.15,
            confidence: 0.10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalOptions {
    pub limit: usize,
    pub weights: RetrievalWeights,
}

impl Default for RetrievalOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            weights: RetrievalWeights::default(),
        }
    }
}

pub fn retrieve(
    query_embedding: &[f32],
    memories: &[MemoryRecord],
    options: RetrievalOptions,
) -> Vec<RetrievedMemory> {
    let mut results: Vec<RetrievedMemory> = memories
        .iter()
        .filter_map(|memory| {
            let embedding = memory.embedding.as_ref()?;
            if embedding.len() != query_embedding.len() {
                return None;
            }

            let semantic_similarity = normalized_cosine_similarity(query_embedding, embedding);
            let importance_score = memory.importance.clamp(0.0, 1.0);
            let recency_score = recency_score(memory);
            let confidence_score = memory.confidence.clamp(0.0, 1.0);

            let final_score = options.weights.semantic * semantic_similarity
                + options.weights.importance * importance_score
                + options.weights.recency * recency_score
                + options.weights.confidence * confidence_score;

            Some(RetrievedMemory {
                memory: memory.clone(),
                semantic_similarity,
                importance_score,
                recency_score,
                confidence_score,
                final_score,
            })
        })
        .collect();

    results.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(options.limit);
    results
}

pub fn normalized_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let raw = cosine_similarity(a, b);
    ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn recency_score(memory: &MemoryRecord) -> f32 {
    let reference_time = memory.last_accessed_at.unwrap_or(memory.created_at);
    let age_days = (Utc::now() - reference_time).num_days().max(0) as f32;

    // Smooth decay: today ≈ 1.0, 30 days ≈ 0.5, 90 days ≈ 0.25
    (1.0 / (1.0 + age_days / 30.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_is_normalized() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(normalized_cosine_similarity(&a, &b), 1.0);
    }
}

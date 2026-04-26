use memory_core::RetrievedMemory;

#[derive(Debug, Clone)]
pub struct ContextBuilder {
    max_memory_chars: usize,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self {
            max_memory_chars: 1_200,
        }
    }
}

impl ContextBuilder {
    pub fn new(max_memory_chars: usize) -> Self {
        Self { max_memory_chars }
    }

    pub fn build_prompt(&self, user_request: &str, memories: &[RetrievedMemory]) -> String {
        let mut prompt = String::new();

        prompt.push_str("You are an AI agent using an external memory system.\n");
        prompt.push_str("Use memories only when they are relevant to the current request.\n");
        prompt.push_str("Do not invent memory details. If memory is missing, say what you can infer from the request only.\n\n");

        if memories.is_empty() {
            prompt.push_str("Relevant memories: none found.\n\n");
        } else {
            prompt.push_str("Relevant memories:\n");
            for (index, retrieved) in memories.iter().enumerate() {
                let memory = &retrieved.memory;
                let content = truncate(&memory.content, self.max_memory_chars);

                prompt.push_str(&format!(
                    "{}. [{} | score {:.3} | semantic {:.3} | confidence {:.2} | importance {:.2}]\n   Tags: {}\n   Memory: {}\n\n",
                    index + 1,
                    memory.memory_type,
                    retrieved.final_score,
                    retrieved.semantic_similarity,
                    memory.confidence,
                    memory.importance,
                    if memory.tags.is_empty() {
                        "none".to_string()
                    } else {
                        memory.tags.join(", ")
                    },
                    content
                ));
            }
        }

        prompt.push_str("Current user request:\n");
        prompt.push_str(user_request);
        prompt.push_str("\n\nAnswer:\n");
        prompt
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut truncated: String = value.chars().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

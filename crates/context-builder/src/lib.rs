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

        prompt.push_str("### SYSTEM INSTRUCTIONS\n");
        prompt.push_str("You are the Agent Memory OS. Your primary goal is to provide precise, professional, and high-signal responses based on the local memory context provided below.\n\n");
        
        prompt.push_str("### GUIDELINES\n");
        prompt.push_str("- Speak directly and concisely. Avoid conversational filler like 'I think' or 'Based on memory...'.\n");
        prompt.push_str("- If a memory identifies the user, refer to them directly (e.g., 'You are Vaibhav' not 'The user is Vaibhav').\n");
        prompt.push_str("- Maintain a sophisticated, helpful, and technically accurate persona.\n");
        prompt.push_str("- If memories are irrelevant, prioritize the user's immediate request while acknowledging existing context where appropriate.\n\n");

        if !memories.is_empty() {
            prompt.push_str("### LOCAL CONTEXT (MEMORIES)\n");
            for (index, retrieved) in memories.iter().enumerate() {
                let memory = &retrieved.memory;
                let content = truncate(&memory.content, self.max_memory_chars);

                prompt.push_str(&format!(
                    "[{}] {}: {}\n",
                    index + 1,
                    memory.memory_type.to_string().to_uppercase(),
                    content
                ));
            }
            prompt.push_str("\n");
        }

        prompt.push_str("### USER COMMAND\n");
        prompt.push_str(user_request);
        prompt.push_str("\n\n### RESPONSE\n");
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

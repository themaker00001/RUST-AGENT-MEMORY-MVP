use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use context_builder::ContextBuilder;
use memory_core::{MemoryRecord, MemoryType, RetrievedMemory};
use memory_retriever::{retrieve, RetrievalOptions};
use memory_store::SqliteMemoryStore;
use ollama_client::OllamaClient;
use std::env;

#[derive(Debug, Parser)]
#[command(name = "agent-memory")]
#[command(about = "Local Agent Memory OS MVP using Rust, SQLite, and Ollama")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Store a new memory with an embedding.
    Remember {
        /// Memory content to store.
        content: String,

        /// Memory type: working, episodic, semantic, tool, failure, preference.
        #[arg(long, default_value = "semantic")]
        memory_type: String,

        /// Importance score from 0.0 to 1.0.
        #[arg(long, default_value_t = 0.75)]
        importance: f32,

        /// Confidence score from 0.0 to 1.0.
        #[arg(long, default_value_t = 0.90)]
        confidence: f32,

        /// Comma-separated tags, for example: rust,ai,memory.
        #[arg(long, default_value = "")]
        tags: String,

        /// Source label for the memory.
        #[arg(long, default_value = "cli")]
        source: String,
    },

    /// Retrieve relevant memories for a query.
    Recall {
        /// Query to search against stored memories.
        query: String,

        /// Maximum number of memories to return.
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },

    /// Ask the chat model using retrieved memories as context.
    Ask {
        /// User question/request.
        query: String,

        /// Maximum number of memories to inject into the prompt.
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },

    /// List all stored memories.
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    let app = App::from_env().await?;

    match cli.command {
        Commands::Remember {
            content,
            memory_type,
            importance,
            confidence,
            tags,
            source,
        } => {
            app.remember(content, memory_type, importance, confidence, tags, source)
                .await?;
        }
        Commands::Recall { query, limit } => {
            app.recall(&query, limit).await?;
        }
        Commands::Ask { query, limit } => {
            app.ask(&query, limit).await?;
        }
        Commands::List => {
            app.list().await?;
        }
    }

    Ok(())
}

struct App {
    store: SqliteMemoryStore,
    ollama: OllamaClient,
}

impl App {
    async fn from_env() -> Result<Self> {
        let database_url =
            env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite://agent_memory.db".to_string());
        let ollama_base_url =
            env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let embed_model =
            env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
        let chat_model =
            env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| "llama3:8b".to_string());

        let store = SqliteMemoryStore::connect(&database_url).await?;
        let ollama = OllamaClient::new(ollama_base_url, embed_model, chat_model);

        Ok(Self { store, ollama })
    }

    async fn remember(
        &self,
        content: String,
        memory_type: String,
        importance: f32,
        confidence: f32,
        tags: String,
        source: String,
    ) -> Result<()> {
        let memory_type = memory_type
            .parse::<MemoryType>()
            .map_err(anyhow::Error::msg)?;

        let embedding = self
            .ollama
            .embed(&content)
            .await
            .context("failed to create memory embedding. Is Ollama running and is the embedding model pulled?")?;

        let memory = MemoryRecord::new(
            content,
            memory_type,
            importance,
            confidence,
            parse_tags(&tags),
            source,
            Some(self.ollama.embed_model().to_string()),
            Some(embedding),
        );

        self.store.insert_memory(&memory).await?;

        println!("Stored memory");
        println!("id: {}", memory.id);
        println!("type: {}", memory.memory_type);
        println!("importance: {:.2}", memory.importance);
        println!("confidence: {:.2}", memory.confidence);

        Ok(())
    }

    async fn recall(&self, query: &str, limit: usize) -> Result<()> {
        let memories = self.search_memories(query, limit).await?;

        if memories.is_empty() {
            println!("No relevant memories found.");
            return Ok(());
        }

        print_retrieved_memories(&memories);
        self.mark_retrieved_as_accessed(&memories).await?;
        Ok(())
    }

    async fn ask(&self, query: &str, limit: usize) -> Result<()> {
        let memories = self.search_memories(query, limit).await?;
        let prompt = ContextBuilder::default().build_prompt(query, &memories);

        let answer = self
            .ollama
            .generate(&prompt)
            .await
            .context("failed to generate answer. Is Ollama running and is the chat model pulled?")?;

        println!("{answer}");

        if !memories.is_empty() {
            println!("\n---\nMemory context used:");
            print_retrieved_memories(&memories);
            self.mark_retrieved_as_accessed(&memories).await?;
        }

        Ok(())
    }

    async fn list(&self) -> Result<()> {
        let memories = self.store.list_memories().await?;

        if memories.is_empty() {
            println!("No memories stored yet.");
            return Ok(());
        }

        for memory in memories {
            println!("---");
            println!("id: {}", memory.id);
            println!("type: {}", memory.memory_type);
            println!("importance: {:.2}", memory.importance);
            println!("confidence: {:.2}", memory.confidence);
            println!(
                "tags: {}",
                if memory.tags.is_empty() {
                    "none".to_string()
                } else {
                    memory.tags.join(", ")
                }
            );
            println!("created_at: {}", memory.created_at);
            println!("content: {}", memory.content);
        }

        Ok(())
    }

    async fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<RetrievedMemory>> {
        let query_embedding = self
            .ollama
            .embed(query)
            .await
            .context("failed to create query embedding. Is Ollama running and is the embedding model pulled?")?;

        let memories = self.store.list_memories().await?;
        let results = retrieve(
            &query_embedding,
            &memories,
            RetrievalOptions {
                limit,
                ..RetrievalOptions::default()
            },
        );

        Ok(results)
    }

    async fn mark_retrieved_as_accessed(&self, memories: &[RetrievedMemory]) -> Result<()> {
        for retrieved in memories {
            self.store.mark_accessed(retrieved.memory.id).await?;
        }
        Ok(())
    }
}

fn parse_tags(tags: &str) -> Vec<String> {
    tags.split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn print_retrieved_memories(memories: &[RetrievedMemory]) {
    for (index, retrieved) in memories.iter().enumerate() {
        let memory = &retrieved.memory;
        println!("---");
        println!("rank: {}", index + 1);
        println!("id: {}", memory.id);
        println!("type: {}", memory.memory_type);
        println!("final_score: {:.3}", retrieved.final_score);
        println!("semantic_similarity: {:.3}", retrieved.semantic_similarity);
        println!("importance: {:.2}", memory.importance);
        println!("confidence: {:.2}", memory.confidence);
        println!(
            "tags: {}",
            if memory.tags.is_empty() {
                "none".to_string()
            } else {
                memory.tags.join(", ")
            }
        );
        println!("content: {}", memory.content);
    }
}

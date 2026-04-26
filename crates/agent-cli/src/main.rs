use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use context_builder::ContextBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use memory_core::{MemoryRecord, MemoryType, RetrievedMemory};
use memory_retriever::{retrieve, RetrievalOptions, normalized_cosine_similarity};
use memory_store::SqliteMemoryStore;
use ollama_client::OllamaClient;
use std::env;
use std::io::{self, Write};
use tokio_stream::StreamExt;

// TUI Imports
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color as TuiColor, Modifier, Style},
    widgets::{Block, Borders, List as TuiList, ListItem, ListState, Paragraph, Wrap},
    Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb.set_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(message.to_string());
    pb
}

fn get_type_color(memory_type: &MemoryType) -> Color {
    match memory_type {
        MemoryType::Working => Color::Cyan,
        MemoryType::Episodic => Color::Blue,
        MemoryType::Semantic => Color::Green,
        MemoryType::Tool => Color::Magenta,
        MemoryType::Failure => Color::Red,
        MemoryType::Preference => Color::Yellow,
    }
}

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

    /// Compress multiple episodic memories into stable semantic facts.
    Compress {
        /// Minimum similarity threshold to group memories (0.0 to 1.0).
        #[arg(long, default_value_t = 0.7)]
        threshold: f32,
    },

    /// Open an interactive TUI dashboard.
    Dash,
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
        Commands::Compress { threshold } => {
            app.compress(threshold).await?;
        }
        Commands::Dash => {
            app.dashboard().await?;
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

        let sp = create_spinner("Creating embedding...");
        let embedding = self
            .ollama
            .embed(&content)
            .await
            .context("failed to create memory embedding. Is Ollama running and is the embedding model pulled?")?;
        sp.finish_and_clear();

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

        println!("{}", "✔ Stored memory".green().bold());
        println!("  {} {}", "id:".dimmed(), memory.id.to_string().bright_black());
        println!(
            "  {} {}",
            "type:".dimmed(),
            memory.memory_type.to_string().color(get_type_color(&memory.memory_type))
        );
        println!("  {} {:.2}", "importance:".dimmed(), memory.importance);

        Ok(())
    }

    async fn recall(&self, query: &str, limit: usize) -> Result<()> {
        let sp = create_spinner("Recalling memories...");
        let memories = self.search_memories(query, limit).await?;
        sp.finish_and_clear();

        if memories.is_empty() {
            println!("{}", "No relevant memories found.".yellow());
            return Ok(());
        }

        print_retrieved_memories(&memories);
        self.mark_retrieved_as_accessed(&memories).await?;
        Ok(())
    }

    async fn ask(&self, query: &str, limit: usize) -> Result<()> {
        let sp = create_spinner("Thinking...");
        let memories = self.search_memories(query, limit).await?;
        let prompt = ContextBuilder::default().build_prompt(query, &memories);

        let mut stream = self
            .ollama
            .generate_stream(&prompt)
            .await
            .context("failed to start generation stream")?;
        
        sp.finish_and_clear();

        let frames = [
            "(V)🦀(V)",
            "(-)🦀(-)",
            "(|)🦀(|)",
            "(-)🦀(-)",
        ];
        
        println!("\n{} {}", "Agent".green().bold(), "is thinking...".dimmed());
        
        let mut full_response = String::new();
        let mut step = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            full_response.push_str(&chunk);
            
            // Animation frames for the left border
            let ferris = frames[step % frames.len()].green();
            let border = if step % 6 < 3 { "┃ ".dimmed() } else { "│ ".dimmed() };
            
            // We print the "Ferris + Border" prefix for every newline in the chunk
            let formatted = chunk.replace("\n", &format!("\n{} {}", ferris, border));
            
            // If it's the very first chunk of the stream, we need the initial prefix
            if step == 0 {
                print!("{} {} {}", ferris, border, formatted);
            } else {
                print!("{}", formatted);
            }
            
            io::stdout().flush()?;
            
            step += 1;
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }

        println!("\n\n{} {}", "───".dimmed(), format!("{} words generated", full_response.split_whitespace().count()).dimmed().italic());

        // Re-render with markdown once finished for final polish
        // print!("\x1B[2J\x1B[1;1H"); // Optional: clear and re-render if you want it to snap to markdown
        // But for now, we'll just keep the stream.

        if !memories.is_empty() {
            println!("{}", "Context used:".dimmed().italic());
            for (index, retrieved) in memories.iter().enumerate() {
                let m = &retrieved.memory;
                println!(
                    "  {} {} ({:.2})",
                    format!("[{}]", index + 1).dimmed(),
                    m.content.chars().take(60).collect::<String>().italic(),
                    retrieved.final_score
                );
            }
            self.mark_retrieved_as_accessed(&memories).await?;
        }

        Ok(())
    }

    async fn list(&self) -> Result<()> {
        let memories = self.store.list_memories().await?;

        if memories.is_empty() {
            println!("{}", "No memories stored yet.".yellow());
            return Ok(());
        }

        for memory in memories {
            let color = get_type_color(&memory.memory_type);
            println!("\n{}", "─".repeat(20).dimmed());
            println!(
                "{} {}",
                memory.memory_type.to_string().color(color).bold(),
                format!("(id: {})", memory.id.to_string().chars().take(8).collect::<String>()).dimmed()
            );
            println!("{} {:.2} | {} {:.2}", "Imp:".dimmed(), memory.importance, "Conf:".dimmed(), memory.confidence);
            if !memory.tags.is_empty() {
                println!("{} {}", "Tags:".dimmed(), memory.tags.join(", ").cyan());
            }
            println!("{}", memory.content.white());
        }

        Ok(())
    }

    async fn compress(&self, threshold: f32) -> Result<()> {
        let sp = create_spinner("Searching for compressible memories...");
        let memories = self.store.list_memories().await?;
        
        // Filter for episodic memories that have embeddings
        let episodic: Vec<&MemoryRecord> = memories.iter()
            .filter(|m| m.memory_type == MemoryType::Episodic && m.embedding.is_some())
            .collect();

        if episodic.len() < 2 {
            sp.finish_and_clear();
            println!("{}", "Not enough episodic memories to compress.".yellow());
            return Ok(());
        }

        // Simple clustering: group memories that are similar
        let mut groups: Vec<Vec<&MemoryRecord>> = Vec::new();
        let mut processed = std::collections::HashSet::new();

        for (i, m1) in episodic.iter().enumerate() {
            if processed.contains(&i) { continue; }
            
            let mut group = vec![*m1];
            processed.insert(i);

            for (j, m2) in episodic.iter().enumerate() {
                if processed.contains(&j) { continue; }
                
                let sim = normalized_cosine_similarity(
                    m1.embedding.as_ref().unwrap(),
                    m2.embedding.as_ref().unwrap()
                );

                if sim >= threshold {
                    group.push(*m2);
                    processed.insert(j);
                }
            }

            if group.len() >= 2 {
                groups.push(group);
            }
        }

        sp.finish_and_clear();

        if groups.is_empty() {
            println!("{}", "No similar memory groups found for compression.".yellow());
            return Ok(());
        }

        println!("Found {} groups to compress.\n", groups.len().to_string().green().bold());

        for group in groups {
            let sp = create_spinner(&format!("Summarizing {} memories...", group.len()));
            
            let contents: Vec<String> = group.iter().map(|m| format!("- {}", m.content)).collect();
            let prompt = format!(
                "The following are several episodic memories (events that happened). \
                Please synthesize them into a single, concise factual statement (a semantic memory). \
                Do not include preamble. Just the fact.\n\n{}",
                contents.join("\n")
            );

            let summary = self.ollama.generate(&prompt).await?;
            sp.finish_and_clear();

            println!("{} {}", "Synthesized:".green().bold(), summary.italic());

            // Create new semantic memory
            let embedding = self.ollama.embed(&summary).await?;
            let new_memory = MemoryRecord::new(
                summary,
                MemoryType::Semantic,
                0.85, // Higher importance for synthesized facts
                0.90,
                vec!["synthesized".to_string()],
                "compression-service".to_string(),
                Some(self.ollama.embed_model().to_string()),
                Some(embedding),
            );

            self.store.insert_memory(&new_memory).await?;

            // Delete old memories
            for m in group {
                self.store.delete_memory(m.id).await?;
                println!("  {} {}", "× Deleted:".red().dimmed(), m.content.chars().take(50).collect::<String>().dimmed());
            }
            println!();
        }

        println!("{}", "✔ Compression complete!".green().bold());
        Ok(())
    }

    async fn dashboard(&self) -> Result<()> {
        let memories = self.store.list_memories().await?;
        
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // TUI state
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        let mut active_panel = 0; // 0 = list, 1 = content
        let mut content_scroll = 0;

        loop {
            let memories_ref = &memories;
            let current_index = list_state.selected().unwrap_or(0);
            
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(30), Constraint::Percentage(70)].as_ref())
                    .split(f.size());

                // Left: Memory List
                let items: Vec<ListItem> = memories_ref.iter().map(|m| {
                    let color = match m.memory_type {
                        MemoryType::Working => TuiColor::Cyan,
                        MemoryType::Episodic => TuiColor::Blue,
                        MemoryType::Semantic => TuiColor::Green,
                        MemoryType::Tool => TuiColor::Magenta,
                        MemoryType::Failure => TuiColor::Red,
                        MemoryType::Preference => TuiColor::Yellow,
                    };
                    ListItem::new(format!("{} | {}", m.memory_type.to_string(), m.content.chars().take(30).collect::<String>()))
                        .style(Style::default().fg(color))
                }).collect();

                let list_block = Block::default()
                    .borders(Borders::ALL)
                    .title(" Memories ")
                    .border_style(if active_panel == 0 { Style::default().fg(TuiColor::Yellow) } else { Style::default() });

                let list = TuiList::new(items)
                    .block(list_block)
                    .highlight_style(Style::default().add_modifier(Modifier::BOLD).bg(TuiColor::DarkGray))
                    .highlight_symbol(">> ");
                
                f.render_stateful_widget(list, chunks[0], &mut list_state);

                // Right: Details
                if let Some(m) = memories_ref.get(current_index) {
                    let details_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Length(7), Constraint::Min(0)].as_ref())
                        .split(chunks[1]);

                    let stats = format!(
                        "ID:         {}\nType:       {}\nImportance: {:.2}\nConfidence: {:.2}\nTags:       {}\nCreated:    {}",
                        m.id,
                        m.memory_type,
                        m.importance,
                        m.confidence,
                        m.tags.join(", "),
                        m.created_at
                    );

                    let details_block = Paragraph::new(stats)
                        .block(Block::default().borders(Borders::ALL).title(" Metadata "));
                    f.render_widget(details_block, details_chunks[0]);

                    let content_block = Paragraph::new(m.content.as_str())
                        .wrap(Wrap { trim: true })
                        .scroll((content_scroll, 0))
                        .block(Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" Content (Scroll: {}) ", content_scroll))
                            .border_style(if active_panel == 1 { Style::default().fg(TuiColor::Yellow) } else { Style::default() }));
                    f.render_widget(content_block, details_chunks[1]);
                }
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Left | KeyCode::BackTab => {
                            active_panel = 0;
                        }
                        KeyCode::Right | KeyCode::Tab => {
                            active_panel = 1;
                        }
                        KeyCode::Down => {
                            if active_panel == 0 {
                                let i = match list_state.selected() {
                                    Some(i) => if i >= memories.len() - 1 { 0 } else { i + 1 },
                                    None => 0,
                                };
                                list_state.select(Some(i));
                                content_scroll = 0;
                            } else {
                                content_scroll += 1;
                            }
                        }
                        KeyCode::Up => {
                            if active_panel == 0 {
                                let i = match list_state.selected() {
                                    Some(i) => if i == 0 { memories.len() - 1 } else { i - 1 },
                                    None => 0,
                                };
                                list_state.select(Some(i));
                                content_scroll = 0;
                            } else {
                                if content_scroll > 0 { content_scroll -= 1; }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

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
        let color = get_type_color(&memory.memory_type);
        println!("\n{}", "─".repeat(20).dimmed());
        println!(
            "{} {} {}",
            format!("[{}]", index + 1).bold(),
            memory.memory_type.to_string().color(color).bold(),
            format!("(Score: {:.3})", retrieved.final_score).dimmed()
        );
        println!(
            "{} {:.3} | {} {:.2}",
            "Sim:".dimmed(),
            retrieved.semantic_similarity,
            "Imp:".dimmed(),
            memory.importance
        );
        println!("{}", memory.content.white());
    }
}

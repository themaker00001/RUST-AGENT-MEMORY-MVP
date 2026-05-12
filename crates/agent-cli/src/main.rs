use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use context_builder::ContextBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use memory_core::{MemoryLink, MemoryRecord, MemoryType, RetrievedMemory};
use std::collections::HashMap;
use memory_retriever::{retrieve, RetrievalOptions, normalized_cosine_similarity};
use memory_store::SqliteMemoryStore;
use ollama_client::OllamaClient;
use std::env;
use std::io::{self, Write};
use tokio_stream::StreamExt;
use uuid::Uuid;

// TUI Imports
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color as TuiColor, Modifier, Style},
    text::{Line, Span, Text},
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

    /// Link two memories together.
    Link {
        /// Source memory UUID.
        source_id: String,

        /// Target memory UUID.
        target_id: String,

        /// Type of relationship (e.g., "related_to", "caused_by", "belongs_to").
        relation: String,
    },

    /// Open an interactive TUI dashboard.
    Dash,

    /// Execute a shell command and remember its outcome as a 'tool' memory.
    Exec {
        /// The shell command to execute.
        command: String,
    },
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
        Commands::Link {
            source_id,
            target_id,
            relation,
        } => {
            app.link(source_id, target_id, relation).await?;
        }
        Commands::Dash => {
            app.dashboard().await?;
        }
        Commands::Exec { command } => {
            app.exec(&command).await?;
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

    async fn link(&self, source_id: String, target_id: String, relation: String) -> Result<()> {
        let source_uuid = Uuid::parse_str(&source_id).context("invalid source UUID")?;
        let target_uuid = Uuid::parse_str(&target_id).context("invalid target UUID")?;

        self.store.link_memories(source_uuid, target_uuid, &relation).await?;

        println!("{}", "✔ Memories linked".green().bold());
        println!("  {} {}", "source:".dimmed(), source_id.bright_black());
        println!("  {} {}", "target:".dimmed(), target_id.bright_black());
        println!("  {} {}", "relation:".dimmed(), relation.cyan());

        Ok(())
    }

    async fn dashboard(&self) -> Result<()> {
        let memories = self.store.list_memories().await?;
        let all_links = self.store.list_all_links().await?;
        let memory_map: HashMap<Uuid, &MemoryRecord> =
            memories.iter().map(|m| (m.id, m)).collect();

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut list_state = ListState::default();
        list_state.select(Some(0));
        let mut active_panel: u8 = 0; // 0 = list, 1 = right panel
        let mut content_scroll: u16 = 0;
        let mut view_mode = ViewMode::Detail;

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
                    ListItem::new(format!(
                        "{} | {}",
                        m.memory_type,
                        m.content.chars().take(30).collect::<String>()
                    ))
                    .style(Style::default().fg(tui_type_color(&m.memory_type)))
                }).collect();

                let list_block = Block::default()
                    .borders(Borders::ALL)
                    .title(" Memories ")
                    .border_style(if active_panel == 0 {
                        Style::default().fg(TuiColor::Yellow)
                    } else {
                        Style::default()
                    });

                let list = TuiList::new(items)
                    .block(list_block)
                    .highlight_style(
                        Style::default().add_modifier(Modifier::BOLD).bg(TuiColor::DarkGray),
                    )
                    .highlight_symbol(">> ");

                f.render_stateful_widget(list, chunks[0], &mut list_state);

                // Right: Detail or Graph
                if let Some(m) = memories_ref.get(current_index) {
                    let right_border = if active_panel == 1 {
                        Style::default().fg(TuiColor::Yellow)
                    } else {
                        Style::default()
                    };

                    match view_mode {
                        ViewMode::Detail => {
                            let details_chunks = Layout::default()
                                .direction(Direction::Vertical)
                                .constraints(
                                    [Constraint::Length(7), Constraint::Min(0)].as_ref(),
                                )
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

                            f.render_widget(
                                Paragraph::new(stats).block(
                                    Block::default()
                                        .borders(Borders::ALL)
                                        .title(" Metadata  [g] Graph "),
                                ),
                                details_chunks[0],
                            );

                            f.render_widget(
                                Paragraph::new(m.content.as_str())
                                    .wrap(Wrap { trim: true })
                                    .scroll((content_scroll, 0))
                                    .block(
                                        Block::default()
                                            .borders(Borders::ALL)
                                            .title(format!(" Content (↑↓ scroll: {}) ", content_scroll))
                                            .border_style(right_border),
                                    ),
                                details_chunks[1],
                            );
                        }
                        ViewMode::Graph => {
                            let graph_text = build_graph_lines(m, &all_links, &memory_map);
                            f.render_widget(
                                Paragraph::new(graph_text)
                                    .scroll((content_scroll, 0))
                                    .block(
                                        Block::default()
                                            .borders(Borders::ALL)
                                            .title(" Memory Graph  [g] Detail  [↑↓] scroll ")
                                            .border_style(right_border),
                                    ),
                                chunks[1],
                            );
                        }
                    }
                }
            })?;

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('g') => {
                            view_mode = if view_mode == ViewMode::Detail {
                                ViewMode::Graph
                            } else {
                                ViewMode::Detail
                            };
                            content_scroll = 0;
                        }
                        KeyCode::Left | KeyCode::BackTab => {
                            active_panel = 0;
                        }
                        KeyCode::Right | KeyCode::Tab => {
                            active_panel = 1;
                        }
                        KeyCode::Down => {
                            if active_panel == 0 {
                                let i = match list_state.selected() {
                                    Some(i) => {
                                        if i >= memories.len() - 1 { 0 } else { i + 1 }
                                    }
                                    None => 0,
                                };
                                list_state.select(Some(i));
                                content_scroll = 0;
                            } else {
                                content_scroll = content_scroll.saturating_add(1);
                            }
                        }
                        KeyCode::Up => {
                            if active_panel == 0 {
                                let i = match list_state.selected() {
                                    Some(i) => {
                                        if i == 0 { memories.len() - 1 } else { i - 1 }
                                    }
                                    None => 0,
                                };
                                list_state.select(Some(i));
                                content_scroll = 0;
                            } else {
                                content_scroll = content_scroll.saturating_sub(1);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
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

    async fn exec(&self, command: &str) -> Result<()> {
        let sp = create_spinner(&format!("Executing: {}", command));
        
        let output = if cfg!(target_os = "windows") {
            std::process::Command::new("cmd")
                .args(["/C", command])
                .output()?
        } else {
            std::process::Command::new("sh")
                .args(["-c", command])
                .output()?
        };
        sp.finish_and_clear();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        let mut content = format!("$ {}\n", command);
        if !stdout.is_empty() {
            content.push_str("STDOUT:\n");
            content.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !stdout.is_empty() {
                content.push('\n');
            }
            content.push_str("STDERR:\n");
            content.push_str(&stderr);
        }

        let exit_status = output.status;
        content.push_str(&format!("\nExit Status: {}", exit_status));

        println!("{}", content.dimmed());
        
        if exit_status.success() {
            println!("{}", "✔ Command executed successfully".green().bold());
        } else {
            println!("{}", "✘ Command failed".red().bold());
        }

        let sp = create_spinner("Creating embedding for command output...");
        let embedding = self
            .ollama
            .embed(&content)
            .await
            .context("failed to create memory embedding")?;
        sp.finish_and_clear();

        let memory = MemoryRecord::new(
            content,
            MemoryType::Tool,
            0.6,
            0.9,
            vec!["shell".to_string(), "exec".to_string()],
            "cli-exec".to_string(),
            Some(self.ollama.embed_model().to_string()),
            Some(embedding),
        );

        self.store.insert_memory(&memory).await?;
        println!("{}", "✔ Stored tool memory".green().bold());
        
        Ok(())
    }
}

#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    Detail,
    Graph,
}

fn tui_type_color(memory_type: &MemoryType) -> TuiColor {
    match memory_type {
        MemoryType::Working => TuiColor::Cyan,
        MemoryType::Episodic => TuiColor::Blue,
        MemoryType::Semantic => TuiColor::Green,
        MemoryType::Tool => TuiColor::Magenta,
        MemoryType::Failure => TuiColor::Red,
        MemoryType::Preference => TuiColor::Yellow,
    }
}

fn build_graph_lines(
    memory: &MemoryRecord,
    links: &[MemoryLink],
    memory_map: &HashMap<Uuid, &MemoryRecord>,
) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.push(Line::from(vec![
        Span::styled("Node  ".to_string(), Style::default().fg(TuiColor::DarkGray)),
        Span::styled(
            format!("{}…", &memory.id.to_string()[..8]),
            Style::default().fg(TuiColor::Yellow),
        ),
        Span::styled(
            format!("  ({})", memory.memory_type),
            Style::default().fg(tui_type_color(&memory.memory_type)),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        "─".repeat(38),
        Style::default().fg(TuiColor::DarkGray),
    )));
    lines.push(Line::from(""));

    lines.push(Line::from(vec![
        Span::styled("◉ ".to_string(), Style::default().fg(TuiColor::White).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("[{}] ", memory.memory_type),
            Style::default().fg(tui_type_color(&memory.memory_type)).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("\"{}\"", memory.content.chars().take(50).collect::<String>()),
            Style::default().fg(TuiColor::White),
        ),
    ]));

    let outgoing: Vec<&MemoryLink> = links.iter().filter(|l| l.source_id == memory.id).collect();
    let incoming: Vec<&MemoryLink> = links.iter().filter(|l| l.target_id == memory.id).collect();

    if !outgoing.is_empty() {
        lines.push(Line::from(Span::styled("│".to_string(), Style::default().fg(TuiColor::DarkGray))));
        lines.push(Line::from(Span::styled(
            "Outgoing:".to_string(),
            Style::default().fg(TuiColor::DarkGray).add_modifier(Modifier::ITALIC),
        )));

        for (i, link) in outgoing.iter().enumerate() {
            let connector = if i == outgoing.len() - 1 && incoming.is_empty() { "└" } else { "├" };
            let label = memory_map
                .get(&link.target_id)
                .map(|t| format!("[{}] \"{}\"", t.memory_type, t.content.chars().take(35).collect::<String>()))
                .unwrap_or_else(|| format!("[{}]", &link.target_id.to_string()[..8]));
            let target_color = memory_map
                .get(&link.target_id)
                .map(|t| tui_type_color(&t.memory_type))
                .unwrap_or(TuiColor::Gray);

            lines.push(Line::from(vec![
                Span::styled(format!("{}──[", connector), Style::default().fg(TuiColor::DarkGray)),
                Span::styled(link.relation_type.clone(), Style::default().fg(TuiColor::Yellow)),
                Span::styled("]──► ".to_string(), Style::default().fg(TuiColor::DarkGray)),
                Span::styled(label, Style::default().fg(target_color)),
            ]));
        }
    }

    if !incoming.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Incoming:".to_string(),
            Style::default().fg(TuiColor::DarkGray).add_modifier(Modifier::ITALIC),
        )));

        for (i, link) in incoming.iter().enumerate() {
            let connector = if i == incoming.len() - 1 { "└" } else { "├" };
            let label = memory_map
                .get(&link.source_id)
                .map(|s| format!("[{}] \"{}\"", s.memory_type, s.content.chars().take(35).collect::<String>()))
                .unwrap_or_else(|| format!("[{}]", &link.source_id.to_string()[..8]));
            let source_color = memory_map
                .get(&link.source_id)
                .map(|s| tui_type_color(&s.memory_type))
                .unwrap_or(TuiColor::Gray);

            lines.push(Line::from(vec![
                Span::styled(format!("{}──[", connector), Style::default().fg(TuiColor::DarkGray)),
                Span::styled(link.relation_type.clone(), Style::default().fg(TuiColor::Cyan)),
                Span::styled("]──◄ ".to_string(), Style::default().fg(TuiColor::DarkGray)),
                Span::styled(label, Style::default().fg(source_color)),
            ]));
        }
    }

    if outgoing.is_empty() && incoming.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "No links for this memory.".to_string(),
            Style::default().fg(TuiColor::DarkGray),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Use: agent-memory link <src> <tgt> <relation>".to_string(),
            Style::default().fg(TuiColor::DarkGray).add_modifier(Modifier::ITALIC),
        )));
    }

    Text::from(lines)
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

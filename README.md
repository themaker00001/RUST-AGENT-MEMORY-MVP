# Agent Memory OS — Professional Local Memory Agent

A high-performance, local memory-augmented CLI agent built with Rust, SQLite, and Ollama.

## ✨ Features

- **Professional Persona**: A sophisticated "Memory OS" system prompt for high-signal, direct responses.
- **Claude-style Streaming**: Real-time token streaming with a word counter and a conversation border.
- **Animated Ferris**: An animated Rust mascot (`🦀`) that snips its claws and "guards" the left border while the AI thinks and streams.
- **Interactive TUI Dashboard**: A full-screen interactive interface to browse, scroll, and manage your memory store.
- **Intelligent Compression**: Automatically synthesize multiple episodic memories into stable, long-term semantic facts.
- **Hybrid Retrieval**: Scoring based on semantic similarity (55%), importance (20%), recency (15%), and confidence (10%).

## 🏗 Architecture

```text
agent-memory-os-rs/
├── crates/
    ├── agent-cli/        # CLI, TUI, and Streaming Visuals
    ├── memory-core/       # Shared Memory Models
    ├── memory-store/      # SQLite Persistence (SQLx)
    ├── memory-retriever/  # Hybrid Ranking & Vector Math
    ├── ollama-client/     # Streaming API Client
    └── context-builder/   # Professional Prompt Engineering
```

## 🚀 Getting Started

### Prerequisites

```bash
# Pull local models
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### Installation

```bash
# Clone and build
cargo build
```

## 🛠 Commands

### 1. The Interactive Dashboard
Open a full-screen view of your memories. Use **Tab** to switch between the list and content, and **Arrows** to scroll.
```bash
cargo run -p agent-memory -- dash
```

### 2. Ask (with Animation & Streaming)
Ask questions using your local memory context with the "Claude-style" UI.
```bash
cargo run -p agent-memory -- ask "Who am I and what am I working on?"
```

### 3. Memory Compression
Synthesize similar episodic memories into stable semantic facts to reduce context bloat.
```bash
cargo run -p agent-memory -- compress --threshold 0.75
```

### 4. Store a Memory
```bash
cargo run -p agent-memory -- remember "User prefers Rust over Python for performance" --memory-type preference
```

### 5. List & Recall
```bash
cargo run -p agent-memory -- list
cargo run -p agent-memory -- recall "What are my language preferences?"
```

## 🧠 Memory Types

| Type | Meaning |
|---|---|
| `working` | Current active task |
| `episodic` | Events that happened (Compressible) |
| `semantic` | Stable facts (Long-term) |
| `tool` | Tool/API/terminal results |
| `failure` | Mistakes to avoid in the future |
| `preference` | User-specific settings or habits |

## 🧪 Next Steps

- **Memory Linking**: Implement Graph relationships between memories.
- **Tool Use**: Allow the agent to execute shell commands and "remember" the outcome.
- **Web Search**: Add local search capabilities to ingest fresh data into memories.

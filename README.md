# Agent Memory OS MVP — Rust + SQLite + Ollama

A local memory-augmented CLI agent built with:

- Rust workspace crates
- SQLite for local memory storage
- Ollama for local embeddings and generation
- `nomic-embed-text` for memory embeddings
- `llama3:8b` for answering
- Hybrid retrieval scoring using semantic similarity, importance, recency, and confidence

This MVP implements the foundation:

```text
Store memories → embed them → retrieve relevant memories → inject them into a prompt → ask llama3:8b
```

## Architecture

```text
agent-memory-os-rs/
├── Cargo.toml
├── README.md
└── crates/
    ├── agent-cli/
    ├── memory-core/
    ├── memory-store/
    ├── memory-retriever/
    ├── ollama-client/
    └── context-builder/
```

### Crates

| Crate | Responsibility |
|---|---|
| `memory-core` | Shared memory types and records |
| `ollama-client` | Calls Ollama embedding and generation endpoints |
| `memory-store` | Persists memories in SQLite |
| `memory-retriever` | Ranks memories using hybrid scoring |
| `context-builder` | Builds the final memory-aware prompt |
| `agent-cli` | CLI commands: `remember`, `recall`, `ask`, `list` |

## Prerequisites

Install Rust and Ollama:

```bash
# macOS
brew install rustup
rustup-init

brew install ollama
```

Pull the local models:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
```

Start Ollama:

```bash
ollama serve
```

## Setup

```bash
cp .env.example .env
cargo build
```

## Commands

### 1. Store a memory

```bash
cargo run -p agent-memory -- remember \
  "User is building an Agent Memory OS in Rust on macOS M4 with 24GB RAM" \
  --memory-type preference \
  --importance 0.95 \
  --confidence 0.95 \
  --tags rust,ai,memory,macos
```

### 2. Recall memories

```bash
cargo run -p agent-memory -- recall \
  "What project is the user building?"
```

### 3. Ask with memory context

```bash
cargo run -p agent-memory -- ask \
  "Continue the project and tell me the next implementation step"
```

### 4. List stored memories

```bash
cargo run -p agent-memory -- list
```

## Retrieval formula

The retriever uses this hybrid score:

```text
final_score =
  0.55 * semantic_similarity
+ 0.20 * importance
+ 0.15 * recency
+ 0.10 * confidence
```

This makes retrieval more useful than plain vector similarity because the system can prefer memories that are relevant, important, recent, and trusted.

## Memory types

| Type | Meaning |
|---|---|
| `working` | Current active task |
| `episodic` | Something that happened |
| `semantic` | Stable fact |
| `tool` | Tool/API/terminal result |
| `failure` | Mistake or failed approach |
| `preference` | User preference |

## Next recommended feature

Add memory compression:

```text
Many small old memories
        ↓
Summarized into one stable semantic memory
        ↓
Original memories can be archived
```

This is the next step toward a real Agent Memory OS rather than a simple memory store.

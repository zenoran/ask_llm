# ask_llm

CLI tool for querying LLMs from your terminal. Supports OpenAI, Ollama, local GGUF models, and HuggingFace.

## Install

```bash
# Basic install
curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash

# With local GGUF support (CUDA)
curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash -s -- --with-llama

# With HuggingFace transformers
curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash -s -- --with-hf

# Everything
curl -fsSL https://raw.githubusercontent.com/zenoran/ask_llm/master/install.sh | bash -s -- --all
```

## Setup

```bash
# Create config directory
mkdir -p ~/.config/ask-llm

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." >> ~/.config/ask-llm/.env

# PostgreSQL for memory (optional)
echo "POSTGRES_HOST=localhost" >> ~/.config/ask-llm/.env
echo "POSTGRES_USER=askllm" >> ~/.config/ask-llm/.env
echo "POSTGRES_PASSWORD=yourpassword" >> ~/.config/ask-llm/.env
echo "POSTGRES_DATABASE=askllm" >> ~/.config/ask-llm/.env
```

## Usage

```bash
llm "what is the meaning of life"     # Ask a question
llm                                   # Interactive mode
llm -m gpt4 "explain quantum physics" # Use specific model
llm --local "hello"                   # Use local model (no API)
llm -b nova "help me code"            # Use a specific bot
llm --status                          # Check system status
llm --list-models                     # List available models
llm --list-bots                       # List available bots
```

## Bots

- **mira** - Conversational companion (default)
- **nova** - Technical assistant
- **spark** - Lightweight local assistant (no database)

## Features

- Multiple LLM providers (OpenAI, Ollama, GGUF, HuggingFace)
- Bot personalities with isolated memory
- PostgreSQL + pgvector for semantic memory search
- Streaming responses with rich formatting
- Conversation history with configurable duration
- Model aliases for quick switching

## Commands

| Command | Description |
|---------|-------------|
| `llm` | Main CLI for querying LLMs |
| `ask-llm` | Alias for `llm` |
| `llm-service` | Background service with OpenAI-compatible API |
| `ask-llm-service` | Alias for `llm-service` |

### Background Service

The optional background service provides:
- Async task processing (memory extraction, embeddings)
- OpenAI-compatible API at `http://localhost:8642/v1/chat/completions`
- Interactive API docs at `http://localhost:8642/docs`

```bash
# Install with service support
pip install ask-llm[service]
# or
./install.sh --with-service

# Run the service
llm-service                    # Default port 8642
llm-service --port 8080        # Custom port
llm-service --host 0.0.0.0     # Listen on all interfaces
```

## Development

```bash
git clone https://github.com/zenoran/ask_llm.git
cd ask_llm
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"     # Install with dev extras
uv pip install -e ".[service]" # Include background service
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 CLI (llm)                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Prompt    │→ │  Bot/User   │→ │   Memory    │→ │    LLM Client       │ │
│  │   Input     │  │   Context   │  │  Retrieval  │  │ (OpenAI/GGUF/Ollama)│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                         ↓                      ↓            │
│                              ┌──────────────────────────────────────────┐   │
│                              │         PostgreSQL + pgvector           │   │
│                              │  ┌─────────────┐  ┌─────────────────┐   │   │
│                              │  │  Messages   │  │    Memories     │   │   │
│                              │  │  (history)  │  │  (embeddings)   │   │   │
│                              │  └─────────────┘  └─────────────────┘   │   │
│                              └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline

1. **Input Processing** - Parse prompt, load bot personality, inject user profile
2. **Memory Retrieval** - Semantic search for relevant past context
3. **Context Assembly** - Combine system prompt + memories + recent history
4. **LLM Query** - Send to configured model with streaming
5. **Memory Extraction** - Extract facts from conversation (async background)
6. **Response Display** - Rich formatted output with bot styling

### Memory System

The memory system is designed to evolve with you, not fossilize into static facts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Memory Lifecycle                                   │
│                                                                              │
│  Conversation → Fact Extraction → Embedding → Storage → Retrieval → Decay  │
│                      (LLM)         (local)    (pgvector)  (semantic)        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Scoring Formula                                 │ │
│  │                                                                         │ │
│  │  effective_score = (base_importance × (1 - recency_weight) +           │ │
│  │                     similarity × decay_factor × recency_weight)         │ │
│  │                    × access_boost                                       │ │
│  │                                                                         │ │
│  │  where:                                                                 │ │
│  │    decay_factor = exp(-age_days × ln(2) / half_life)                   │ │
│  │    access_boost = 1 + boost_factor × log(access_count + 1)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Memory Types & Decay Rates

| Type | Half-Life Multiplier | Use Case |
|------|---------------------|----------|
| `fact` | 2.0x (180 days) | Core identity - name, traits |
| `professional` | 1.5x (135 days) | Career, skills, projects |
| `health` | 1.2x (108 days) | Health conditions, fitness |
| `relationship` | 1.0x (90 days) | People, connections |
| `preference` | 0.8x (72 days) | Likes/dislikes - these change |
| `event` | 0.5x (45 days) | Past events, milestones |
| `plan` | 0.3x (27 days) | Goals, intentions - very temporal |

#### Anti-Fossilization Features

- **Temporal Decay**: Old memories naturally fade unless reinforced
- **Access Reinforcement**: Frequently recalled memories stay relevant
- **Contradiction Detection**: New facts supersede old ones (with history preserved)
- **Diversity Sampling**: Retrieval includes varied time periods and types
- **Supersession Tracking**: Old facts marked as superseded, not deleted

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `cli.py` | Entry point | Argument parsing, query routing |
| `core.py` | `AskLLM` class | Orchestrates clients, bots, memory |
| `bots.py` | Bot manager | Personality loading, system prompts |
| `clients/` | LLM clients | OpenAI, Ollama, GGUF, HuggingFace |
| `memory/postgresql.py` | Memory backend | pgvector storage, decay, search |
| `memory/embeddings.py` | Local embeddings | sentence-transformers (MiniLM) |
| `memory/extraction/` | Fact extraction | LLM-based memory distillation |
| `service/` | Background API | FastAPI, async tasks, OpenAI-compat |

### Configuration

All settings use `ASK_LLM_` prefix in `~/.config/ask-llm/.env`:

```bash
# Core
ASK_LLM_DEFAULT_MODEL_ALIAS=gpt-5.2-chat-latest
ASK_LLM_DEFAULT_BOT=mira
ASK_LLM_DEFAULT_USER=nick

# Memory Decay (anti-fossilization)
ASK_LLM_MEMORY_DECAY_ENABLED=true
ASK_LLM_MEMORY_DECAY_HALF_LIFE_DAYS=90
ASK_LLM_MEMORY_ACCESS_BOOST_FACTOR=0.15
ASK_LLM_MEMORY_RECENCY_WEIGHT=0.3
ASK_LLM_MEMORY_DIVERSITY_ENABLED=true

# Embeddings (local, no API calls)
ASK_LLM_MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2
ASK_LLM_MEMORY_EMBEDDING_DIM=384

# PostgreSQL
ASK_LLM_POSTGRES_HOST=localhost
ASK_LLM_POSTGRES_USER=askllm
ASK_LLM_POSTGRES_PASSWORD=yourpassword
ASK_LLM_POSTGRES_DATABASE=askllm
```

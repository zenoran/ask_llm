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

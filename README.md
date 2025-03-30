# Ask LLM

## Overview
Ask LLM is a Python-based command-line tool for interacting with various language models, including OpenAI and Ollama. It provides an intuitive interface for querying these models while maintaining conversation history and context.

## Features
- Support for multiple LLM providers (OpenAI and Ollama)
- Persistent conversation history
- Interactive mode with multiline input support
- Command execution with output included in prompts
- Rich, formatted output using the Rich library

## Installation

### Requirements
- Python 3.12 or higher

### Installation as a Package
You can install Ask LLM directly as a Python package:
```bash
pip install git+https://github.com/yourusername/ask_llm.git
```

Alternatively, install from source:
```bash
# Clone the repository
git clone https://github.com/yourusername/ask_llm.git
cd ask_llm

# Install the package
pip install -e .
```

### Environment Setup
For OpenAI models, set your API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

After installation, you can use either `ask-llm` or `llm` command:

```bash
# Ask a direct question
llm "What is the capital of France?"

# Enter interactive mode
llm
```

### Interactive Mode
In interactive mode:
- Type your questions directly
- Type `exit` or `quit` to leave
- Type `>` at the beginning of a line to enter multiline input mode

### Command-Line Options
- `-m`, `--model`: Specify the model to use (default is `gpt-4o`)
- `-dh`, `--delete-history`: Wipe saved chat history and start fresh
- `-ph [N]`, `--print-history [N]`: Print conversation history (optionally specify number of recent conversations)
- `-c`, `--command`: Execute a shell command and include its output with your question
- `--verbose`: Print full JSON response
- `--plain`: Use plain text output (no formatting)

## Examples

```bash
# Ask a question using a specific model
llm -m gpt-4o "Explain quantum computing in simple terms"

# Execute a command and ask about its output
llm -c "ls -la" "What files are in my current directory?"

# View your conversation history
llm -ph

# View only the last 3 conversation pairs
llm -ph 3
```

## License
This project is licensed under the MIT License.
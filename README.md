# Ask LLM

## Overview
Ask LLM is a Python-based command-line tool for interacting with various language models, including OpenAI, Ollama, and Hugging Face. It provides an intuitive interface for querying these models while maintaining conversation history and context. This is intended to be a quick way to ask LLM questions while navigating through your shell without having to open a web browser to chat through a UI. Quick access to ask for help with something and easily copy/paste examples without having to leave your shell.

## Features
- Support for multiple LLM providers:
  - **OpenAI**: GPT-4o, GPT-4.5-preview, O1, and more
  - **Ollama**: Local models like llama3, gemma3, and more
  - **Hugging Face**: Run transformer models locally with GPU acceleration 
- Persistent conversation history with configurable duration
- Interactive mode with multiline input support
- Command execution with output included in prompts
- Rich, formatted output with Markdown support
- GPU optimization for local models (especially for RTX 4090)
- Model quantization for memory efficiency

## Installation

### Requirements
- Python 3.12 or higher
- For Hugging Face models: CUDA-compatible GPU

### Basic Installation
The basic installation includes support for OpenAI and Ollama models:

```bash
# Install from GitHub
pip install git+https://github.com/zenoran/ask_llm.git
```

### Installing with Optional Dependencies

#### Hugging Face Support
To use Hugging Face models locally, install with the huggingface extra:

```bash
# Install from GitHub
pip install "git+https://github.com/zenoran/ask_llm.git#egg=ask_llm[huggingface]"
```

The `huggingface` extra includes the following dependencies:
- `torch`: PyTorch for tensor operations and neural networks
- `transformers`: Core Hugging Face Transformers library
- `bitsandbytes`: Quantization support for memory-efficient inference
- `accelerate`: Accelerated model loading and execution
- `peft`: Parameter-Efficient Fine-Tuning for optimized inference
- `xformers`: Efficient transformer implementation with memory optimizations


### Installation from Source
```bash
# Clone the repository
git clone https://github.com/zenoran/ask_llm.git
cd ask_llm

# Basic installation
pip install -e .

# With Hugging Face support
pip install -e ".[huggingface]"

# With development tools
pip install -e ".[dev]"

# With all optional dependencies
pip install -e ".[huggingface,dev]"
```

### Environment Setup
For OpenAI models, set your API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

For Ollama models, ensure the Ollama service is running locally (default URL: http://localhost:11434).

For Hugging Face models, a CUDA-capable GPU is required with appropriate drivers installed.

## Usage

After installation, you can use either `ask-llm` or `llm` command:

### Ask a direct question
`> llm "Show me some code"`

```
╭────────────────────────────────────── chatgpt-4o-latest ─────────────────────────────────────────────────╮
│                                                                                                          │
│    Sure! Here's a simple Python example that prints "Hello, world!" and adds two numbers:                │
│                                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```python
# Print a message
print("Hello, world!")

# Add two numbers
a = 5
b = 3
sum = a + b

print("The sum is:", sum)
```

Want code in another language or something more advanced? Just let me know! 😄

### Enter interactive mode
`> llm`

### Interactive Mode
In interactive mode:
- Type your questions directly
- Type `exit` or `quit` to leave
- Type `>` at the beginning of a line to enter multiline input mode

### Command-Line Options
- `-m`, `--model`: Specify the model to use (default is `chatgpt-4o-latest`)
- `-dh`, `--delete-history`: Wipe saved chat history and start fresh
- `-ph [N]`, `--print-history [N]`: Print conversation history (optionally specify number of recent conversations)
- `-c`, `--command`: Execute a shell command and include its output with your question
- `--verbose`: Print full JSON response and detailed logs
- `--plain`: Use plain text output (no formatting)
- `--refresh-models`: Refresh the cache of available Ollama models

## Supported Models

### OpenAI
- gpt-4o
- gpt-4.5-preview
- o1
- o3-mini
- chatgpt-4o-latest

### Hugging Face
- PygmalionAI/pygmalion-3-12b
- Add other models by modifying the configuration

**Note:** Hugging Face models are only available if you install the package with the `[huggingface]` extra. If you try to use a Hugging Face model without the required dependencies, the application will automatically fall back to the default model and display instructions for installing the necessary dependencies.

### Ollama
Dynamically discovers available models from your local Ollama installation.

## Examples

```bash
# Ask a question using a specific model
llm -m gpt-4o "Explain quantum computing in simple terms"

# Use a local Hugging Face model
llm -m PygmalionAI/pygmalion-3-12b "Write a short story about a robot"

# Use a local Ollama model (supports partial matching)
llm -m llama3 "What are the benefits of open-source AI?"

# Execute a command and ask about its output
llm -c "ls -la" "What files are in my current directory?"

# View your conversation history
llm -ph

# View only the last 3 conversation pairs
llm -ph 3

# Refresh the available models list
llm --refresh-models
```

## Configuration
Ask LLM uses environment variables with the prefix `ASK_LLM_` for configuration:

- `ASK_LLM_HISTORY_FILE`: Path to the chat history file (default: ~/.ask-llm-chat-history)
- `ASK_LLM_HISTORY_DURATION`: Duration in seconds to retain messages (default: 600)
- `ASK_LLM_OLLAMA_URL`: URL of the Ollama service (default: http://localhost:11434)
- `ASK_LLM_DEFAULT_MODEL`: Default model to use (default: chatgpt-4o-latest)
- `ASK_LLM_MAX_TOKENS`: Maximum tokens to generate (default: 1024)
- `ASK_LLM_TEMPERATURE`: Temperature for generation (default: 0.8)

## Error Handling and Graceful Degradation

Ask LLM is designed to handle missing dependencies gracefully:

- **Missing Hugging Face Dependencies**: If you attempt to use a Hugging Face model without installing the required dependencies, the application will:
  - Display a warning message with instructions on how to install the dependencies
  - Automatically fall back to the default model (OpenAI or Ollama)
  - Continue operation without requiring immediate action

- **Model Validation**: When specifying a model with the `-m` flag, the application validates if the required dependencies are available and provides helpful feedback if they're not.

- **Available Models**: When listing available models (e.g., in error messages), only models with available dependencies are shown.

This approach ensures that users can always use the application even if certain optional features aren't available, while making it clear how to enable those features if desired.

## Provides LLM access to other Python Apps
Setting up the clients was also a means to providing future projects with LLM access in a way I have control over.  i.e. The SesameAI CSM TTS project I'm working on which uses this LLM to generate iterative stories from prompts and play the audio for listening or downloading from a web interface.

## License
This project is licensed under the MIT License.
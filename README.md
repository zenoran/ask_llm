# Ask LLM

## Overview
Ask LLM is a Python-based application that allows users to interact with various language models, including OpenAI and Ollama. The application provides a command-line interface for querying these models and managing conversation history.

## Features
- Supports multiple language models (OpenAI and Ollama).
- Maintains conversation history for context.
- Provides a user-friendly command-line interface.
- Allows execution of shell commands and appending their output to queries.

## Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
To use the application, run the following command in your terminal:

```
python src/main.py [your question here]
```

You can also enter interactive mode by running:

```
python src/main.py
```

In interactive mode, type your questions directly. Type 'exit' or 'quit' to leave the mode.

## Command-Line Options
- `-m`, `--model`: Specify the model to use (default is `gpt-4o`).
- `-dh`, `--delete-history`: Wipe any saved chat history and start fresh.
- `-ph`, `--print-history`: Print the saved conversation history.
- `-c`, `--command`: Execute a shell command and add its output to the question.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
# Configuration settings for the application

import os

class Config:
    HISTORY_FILE = os.path.expanduser("~/.chat-history")
    HISTORY_DURATION = 60 * 10  # retain messages for 60 minutes
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODELS = ["deepseek-r1", "llama3.2", "gemma3"]
    OPENAPI_MODELS = ["gpt-4o", "gpt-4.5-preview", "o1", "o3-mini", "chatgpt-4o-latest"]
    DEFAULT_MODEL = "gpt-4o"
    MODEL_OPTIONS = ["gpt-4o", "gpt-4.5-preview"] + OLLAMA_MODELS
    BUFFER_LINES = 3  # Number of lines to collect before printing
    PRESERVE_CODE_BLOCKS = True  # Wait for complete code blocks before printing
    TMUX_COLLECT_ALL = False  # In TMUX, collect all output before printing
    SHOW_CHAR_COUNT = True  # Show character count in spinner

    SYSTEM_MESSAGE = """
    You are a helpful assistant with expert knowledge in python, linux and coding.
    Answer questions directly and as concise as possible. If you don't know the answer,
    ask for clarification or provide a relevant resource.
    """
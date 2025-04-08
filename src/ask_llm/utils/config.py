import os
import sys
from argparse import Namespace

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from ask_llm.utils.ollama_utils import init_models

class Config(BaseSettings):
    HISTORY_FILE: str = Field(default=os.path.expanduser("~/.ask-llm-chat-history"))
    HISTORY_DURATION: int = Field(default=60 * 10)  # retain messages for 60 minutes
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODELS: list[str] = Field(default_factory=list)  # Will be populated dynamically
    OLLAMA_FALLBACK_MODELS: list[str] = Field(
        default=["exaone-deep", "deepseek-r1", "llama3.2", "llama3", "gemma3"]
    )
    OLLAMA_MODELS_CACHE: str = Field(default=os.path.expanduser("~/.ollama-models"))
    OPENAPI_MODELS: list[str] = Field(
        default=["gpt-4o", "gpt-4.5-preview", "o1", "o3-mini", "chatgpt-4o-latest"]
    )
    HUGGINGFACE_MODELS: list[str] = Field(
        default=[
            "PygmalionAI/pygmalion-3-12b",
            # Add other local HF models here in the future
        ]
    )
    DEFAULT_MODEL: str = Field(default="chatgpt-4o-latest")
    MAX_TOKENS: int = Field(default=1024, description="Default maximum tokens to generate")
    TEMPERATURE: float = Field(default=0.8, description="Default generation temperature")
    BUFFER_LINES: int = Field(default=3)  # Number of lines to collect before printing
    PRESERVE_CODE_BLOCKS: bool = Field(
        default=True
    )  # Wait for complete code blocks before printing
    TMUX_COLLECT_ALL: bool = Field(
        default=False
    )  # In TMUX, collect all output before printing
    SHOW_CHAR_COUNT: bool = Field(default=True)  # Show character count in spinner
    VERBOSE: bool = Field(default=False)  # Verbose mode for debugging
    PLAIN_OUTPUT: bool = Field(default=False)
    INTERACTIVE_MODE: bool = Field(default=False)  # Interactive mode for command line
    SYSTEM_MESSAGE: str = Field(
        """You are a helpful and concise technical assistant. Always respond using simple, easy-to-understand language. 
Keep explanations short and direct, avoiding unnecessary detail. Use correct and clean Markdown formatting in 
your responses, including backticks for code, lists or headings when appropriate, and bold text for emphasis 
when needed. Be fun and engaging, but always prioritize clarity and conciseness.

Respond using clean and properly formatted Markdown. Use the following formatting tools:
• Headings (#, ##, ###) for structure
• Bullet points and numbered lists for organization
• Bold (**text**) and italic (*text*) text for emphasis
• Inline code using single backticks (`code`), and code blocks using triple backticks for multi-line code
• Tables using pipes (|) and dashes (-)
• Simple emojis for expression 😀
• Avoid HTML, LaTeX, or unsupported Markdown elements
• Keep formatting readable and clean for both terminal and markdown viewers"""
    )

    def __init__(self, **data):
        super().__init__(**data)
        refresh_models = "--refresh-models" in sys.argv
        self.OLLAMA_MODELS = init_models(self.OLLAMA_URL, self.OLLAMA_MODELS_CACHE, refresh_models)

    @computed_field
    def MODEL_OPTIONS(self) -> list[str]:
        return self.OPENAPI_MODELS + self.OLLAMA_MODELS + self.HUGGINGFACE_MODELS

    def update_from_args(self, args: Namespace) -> "Config":
        """Update config based on command line arguments"""
        if args.plain:
            self.PLAIN_OUTPUT = True
        if args.verbose:
            self.VERBOSE = True
        if args.model:
            self.DEFAULT_MODEL = args.model
        if not args.question:
            self.INTERACTIVE_MODE = True
        return self

    model_config = SettingsConfigDict(
        env_prefix="ASK_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @classmethod
    def from_args(cls, args: Namespace) -> "Config":
        instance = cls()
        return instance.update_from_args(args)


# Create a single global instance
config: Config = Config()

# Configuration settings for the application

import os
from argparse import Namespace

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Config(BaseSettings):
    HISTORY_FILE: str = Field(default=os.path.expanduser("~/.chat-history"))
    HISTORY_DURATION: int = Field(default=60 * 10)  # retain messages for 60 minutes
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODELS: list[str] = Field(
        default=["exaone-deep", "deepseek-r1", "llama3.2", "llama3", "gemma3"]
    )
    OPENAPI_MODELS: list[str] = Field(
        default=["gpt-4o", "gpt-4.5-preview", "o1", "o3-mini", "chatgpt-4o-latest"]
    )
    DEFAULT_MODEL: str = Field(default="chatgpt-4o-latest")
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
    SYSTEM_MESSAGE: str = Field(
        default="""
    You are a helpful and concise technical assistant. Always respond using simple, easy-to-understand language.
    Keep explanations short and direct, avoiding unnecessary detail. Use correct and clean Markdown formatting in
    your responses, including backticks for code, lists or headings when appropriate, and bold text for emphasis
    when needed.
    """
    )

    @computed_field
    def MODEL_OPTIONS(self) -> list[str]:
        return self.OPENAPI_MODELS + self.OLLAMA_MODELS

    def update_from_args(self, args: Namespace) -> "Config":
        """Update config based on command line arguments"""
        if args.plain:
            self.PLAIN_OUTPUT = True
        if args.verbose:
            self.VERBOSE = True
        if args.model:
            self.DEFAULT_MODEL = args.model
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

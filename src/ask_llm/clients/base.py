"""Base LLM client class.

Simple abstract base for OpenAI-compatible API clients.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from abc import ABC, abstractmethod
from typing import Iterator, List, Any
from ..models.message import Message
from ..utils.config import Config
from ..utils.streaming import render_streaming_response


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    SUPPORTS_STREAMING = False

    def __init__(self, model: str, config: Config):
        self.model = model
        self.config = config
        self.bot_name: str | None = None  # Set by AskLLM after initialization
        force_term = not self.config.PLAIN_OUTPUT
        self.console = Console(force_terminal=force_term)

    @abstractmethod
    def query(self, messages: List[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        """Query the LLM with the given messages.

        Args:
            messages: List of Message objects for conversation history.
            plaintext_output: Whether to return plain text instead of formatted.
            **kwargs: Client-specific arguments (e.g., stream).

        Returns:
            The model's response as a string.
        """
        pass

    def format_message(self, role: str, content: str):
        """Formats and prints a message based on its role."""
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
            self._print_assistant_message(content)

    def _print_user_message(self, content: str):
        """Default implementation for printing user messages."""
        self.console.print()
        self.console.print(f"[bold blue]User:[/bold blue] {content}")

    def _print_assistant_message(self, content: str, panel_title: str | None = None, panel_border_style: str = "green", second_part: str | None = None):
        """Prints assistant message. First part in panel, optional second part below."""
        display_name = self.bot_name or self.model
        title = panel_title or f"[bold {panel_border_style}]{display_name}[/bold {panel_border_style}]"
        assistant_panel = Panel(
            Markdown(content.strip()),
            title=title,
            border_style=panel_border_style,
            padding=(1, 2),
        )
        self.console.print(Align(assistant_panel, align="left"))
        if second_part:
            self.console.print()
            self.console.print(Align(Markdown(second_part.strip()), align="left", pad=False))
            self.console.print()

    @abstractmethod
    def get_styling(self) -> tuple[str | None, str]:
        """Return the specific panel title and border style for this client."""
        pass

    def stream_raw(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        """
        Stream raw text chunks from the LLM without console formatting.
        
        This method is used by the API service for SSE streaming.
        Default implementation falls back to non-streaming query.
        
        Args:
            messages: List of Message objects for conversation history.
            **kwargs: Client-specific arguments.
            
        Yields:
            Text chunks as they're generated.
        """
        # Default fallback: yield the entire response as one chunk
        response = self.query(messages, plaintext_output=True, stream=False, **kwargs)
        yield response

    def _handle_streaming_output(
        self,
        stream_iterator: Iterator[str],
        plaintext_output: bool,
        panel_title: str | None = None,
        panel_border_style: str = "green",
    ) -> str:
        """Handles streaming output using shared streaming utilities."""
        display_name = self.bot_name or self.model
        title = panel_title or f"[bold {panel_border_style}]{display_name}[/bold {panel_border_style}]"
        
        return render_streaming_response(
            stream_iterator=stream_iterator,
            console=self.console,
            panel_title=title,
            panel_border_style=panel_border_style,
            plaintext_output=plaintext_output,
        )


class StubClient(LLMClient):
    """Stub client for history-only operations that don't need actual LLM."""
    
    SUPPORTS_STREAMING = False
    
    def __init__(self, config: Config, bot_name: str = "Assistant"):
        super().__init__(model="stub", config=config)
        self.bot_name = bot_name
    
    def query(self, messages: List[Message], plaintext_output: bool = False, **kwargs: Any) -> str:
        raise NotImplementedError("StubClient does not support querying")
    
    def get_styling(self) -> tuple[str | None, str]:
        return self.bot_name, "cyan"

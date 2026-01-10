from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.text import Text
from abc import ABC, abstractmethod
from typing import Iterator, List, Any
import sys
import time
from ..models.message import Message
from ..utils.config import Config # Import Config for type hinting

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    SUPPORTS_STREAMING = False

    def __init__(self, model: str, config: Config): # Accept config
        self.model = model
        self.config = config # Store config
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
        if second_part:
            self.console.print()

    @abstractmethod
    def get_styling(self) -> tuple[str | None, str]:
        """Return the specific panel title and border style for this client."""
        pass

    def _handle_streaming_output(
        self,
        stream_iterator: Iterator[str],
        plaintext_output: bool,
        panel_title: str | None = None,
        panel_border_style: str = "green",
    ) -> str:
        """Handles streaming. First part (pre-\n\n) in panel, rest via Rich Live."""
        total_response = ""
        start_time = time.time()
        token_count = 0

        if plaintext_output:
            print("", end='')
            sys.stdout.flush()
            try:
                for content in stream_iterator:
                    if content:
                        total_response += content
                        token_count += len(content.split()) # Basic token count
                        print(content, end='')
                        sys.stdout.flush()
                print()
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
                print()
            except Exception as e:
                self.console.print(f"\n[bold red]Error during plaintext streaming:[/bold red] {e}")
                total_response += f"\nERROR: {e}"
        else:
            split_marker = "\n\n"
            first_part_buffer = ""
            first_part_printed = False
            cursor = "▌"
            live_display: Live | None = None
            visible_text = ""
            overflow_buffer = ""
            live_frozen = False
            console_size = getattr(self.console, "size", None)
            max_live_lines = max(5, (getattr(console_size, "height", 24) or 24) - 8)

            def _split_visible(text: str) -> tuple[str, str]:
                lines = text.splitlines(keepends=True)
                if len(lines) <= max_live_lines:
                    return text, ""
                return "".join(lines[:max_live_lines]), "".join(lines[max_live_lines:])

            def _start_live(initial: str) -> None:
                nonlocal live_display, visible_text
                if live_display:
                    return
                visible_text = initial
                live_display = Live(
                    Align(Markdown(f"{visible_text}{cursor}" if visible_text else cursor), align="left", pad=False),
                    console=self.console,
                    refresh_per_second=15,
                    vertical_overflow="crop",
                    transient=False,
                    auto_refresh=False,
                )
                live_display.start(refresh=True)

            def _update_live(add_cursor: bool = True) -> None:
                if not live_display:
                    return
                suffix = cursor if add_cursor else ""
                live_display.update(
                    Align(Markdown(f"{visible_text}{suffix}"), align="left", pad=False),
                    refresh=True,
                )

            try:
                for content in stream_iterator:
                    if not content:
                        continue
                    total_response += content
                    token_count += len(content.split())
                    if not first_part_printed:
                        first_part_buffer += content
                        if split_marker in first_part_buffer:
                            first_part, remainder = first_part_buffer.split(split_marker, 1)
                            self._print_assistant_message(first_part, panel_title=panel_title, panel_border_style=panel_border_style)
                            first_part_printed = True
                            prefixed = ("\n\n" + remainder) if remainder else "\n\n"
                            visible_text, overflow_buffer = _split_visible(prefixed)
                            _start_live(visible_text)
                            if overflow_buffer:
                                live_frozen = True
                                _update_live()
                        continue

                    if live_frozen:
                        overflow_buffer += content
                        continue

                    candidate = visible_text + content
                    visible_text, extra = _split_visible(candidate)
                    if extra:
                        overflow_buffer = extra
                        live_frozen = True
                    _update_live()

                if not first_part_printed and first_part_buffer:
                    self._print_assistant_message(first_part_buffer, panel_title=panel_title, panel_border_style=panel_border_style)

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
            except Exception as e:
                self.console.print(f"\n[bold red]Error during streaming:[/bold red] {e}")
                total_response += f"\nERROR: {e}"
            finally:
                if live_display:
                    _update_live(add_cursor=False)
                    live_display.stop()
                    if overflow_buffer.strip():
                        self.console.print(Align(Markdown(overflow_buffer), align="left", pad=False))
        
        return total_response
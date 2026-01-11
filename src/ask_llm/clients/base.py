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
import platform
from ..models.message import Message
from ..utils.config import Config # Import Config for type hinting

# Detect Windows for streaming workarounds
IS_WINDOWS = platform.system() == "Windows"

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
        """Handles streaming. First part (pre-\n\n) in panel, rest below.
        
        On Windows, uses a simpler approach without Rich Live to avoid
        terminal rendering issues with ANSI escape codes.
        """
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
        elif IS_WINDOWS:
            # Windows: Simple streaming without Live display to avoid rendering issues
            total_response = self._handle_streaming_windows(
                stream_iterator, panel_title, panel_border_style
            )
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
                    vertical_overflow="visible",
                    transient=True,
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
                            # Don't add back the \n\n - it causes extra blank lines in markdown rendering
                            visible_text, overflow_buffer = _split_visible(remainder.lstrip('\n'))
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
                    live_display.stop()
                    # Print final content (transient=True clears the Live area)
                    final_content = visible_text + overflow_buffer
                    if final_content.strip():
                        self.console.print(Align(Markdown(final_content.strip()), align="left", pad=False))
        
        return total_response

    def _handle_streaming_windows(
        self,
        stream_iterator: Iterator[str],
        panel_title: str | None = None,
        panel_border_style: str = "green",
    ) -> str:
        """Windows-specific streaming handler.
        
        Uses simple print-based streaming to avoid Rich Live display issues
        on Windows terminals. Collects full response, then renders with markdown.
        """
        total_response = ""
        split_marker = "\n\n"
        first_part_buffer = ""
        first_part_printed = False
        remainder_buffer = ""
        
        try:
            # Show a simple cursor while streaming
            print("▌ ", end='', flush=True)
            
            for content in stream_iterator:
                if not content:
                    continue
                total_response += content
                
                if not first_part_printed:
                    first_part_buffer += content
                    if split_marker in first_part_buffer:
                        first_part, remainder = first_part_buffer.split(split_marker, 1)
                        # Clear the cursor line and print panel
                        print("\r\033[K", end='', flush=True)
                        self._print_assistant_message(first_part, panel_title=panel_title, panel_border_style=panel_border_style)
                        first_part_printed = True
                        remainder_buffer = remainder
                        # Show streaming indicator for remainder
                        if remainder:
                            print(remainder, end='', flush=True)
                else:
                    remainder_buffer += content
                    print(content, end='', flush=True)
            
            # Finalize output
            if not first_part_printed:
                # Never found split marker, print everything in panel
                print("\r\033[K", end='', flush=True)
                self._print_assistant_message(first_part_buffer, panel_title=panel_title, panel_border_style=panel_border_style)
            elif remainder_buffer.strip():
                # Clear streaming text, render final markdown
                print("\r\033[K", end='')
                # Move up to clear streamed lines
                line_count = remainder_buffer.count('\n') + 1
                for _ in range(line_count):
                    print("\033[A\033[K", end='')
                print(flush=True)
                self.console.print(Align(Markdown(remainder_buffer.strip()), align="left", pad=False))
            else:
                print(flush=True)
                
        except KeyboardInterrupt:
            print()
            self.console.print("[bold yellow]Interrupted![/bold yellow]")
        except Exception as e:
            print()
            self.console.print(f"[bold red]Error during streaming:[/bold red] {e}")
            total_response += f"\nERROR: {e}"
        
        return total_response
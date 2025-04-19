from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from abc import ABC, abstractmethod
from typing import Iterator, List, Any
import sys
import time
from ..models.message import Message
from ..utils.config import Config # Import Config for type hinting

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, config: Config): # Accept config
        self.model = model
        self.config = config # Store config
        # Force terminal color support unless plain output is requested
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
            # Default assistant message printing (can be overridden by subclasses)
            self._print_assistant_message(content)

    def _print_user_message(self, content: str):
        """Default implementation for printing user messages."""
        self.console.print()
        self.console.print(f"[bold blue]User:[/bold blue] {content}")

    def _print_assistant_message(self, content: str, panel_title: str | None = None, panel_border_style: str = "green", second_part: str | None = None):
        """Prints assistant message. First part in panel, optional second part below."""
        title = panel_title or f"[bold {panel_border_style}]{self.model}[/bold {panel_border_style}]"
        assistant_panel = Panel(
            Markdown(content.strip()),
            title=title,
            border_style=panel_border_style,
            padding=(1, 2),
        )
        # Print the panel aligned left
        self.console.print(Align(assistant_panel, align="left"))

        # If there's a second part, print it directly below using Markdown
        if second_part:
            # Add an extra newline for separation
            self.console.print()
            # Add the double newline that separated the parts back for rendering
            # We align it manually to roughly match the panel's content alignment
            # Padding(top, right, bottom, left)
            self.console.print(Align(Markdown(second_part.strip()), align="left", pad=False))

        # Add a final newline for spacing unless only a panel was printed
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
        first_part_buffer = ""
        first_part_printed = False
        stream_ended_during_buffering = False
        rest_after_marker = "" # Store content immediately after \n\n

        if plaintext_output:
            # Original plaintext logic remains the same
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
            # Rich streaming with Live update for second part
            iterator = iter(stream_iterator)
            live_display = None
            live_buffer = ""
            split_marker = "\n\n"
            last_refresh_time = time.time()
            min_refresh_interval = 0.05 # 20fps max

            try:
                # --- Stage 1: Buffer until \n\n or end of stream ---
                while not first_part_printed:
                    try:
                        content = next(iterator)
                        if content:
                            total_response += content
                            token_count += len(content.split())
                            first_part_buffer += content

                            if split_marker in first_part_buffer:
                                parts = first_part_buffer.split(split_marker, 1)
                                first_part = parts[0]
                                rest_after_marker = parts[1] # Store for Live seeding

                                # Print first part in panel (only first part)
                                self._print_assistant_message(first_part, panel_title=panel_title, panel_border_style=panel_border_style)
                                first_part_printed = True
                                first_part_buffer = "" # Clear buffer
                                break # Exit buffering loop

                    except StopIteration:
                        # Stream ended before finding \n\n
                        if first_part_buffer:
                             # Print whatever we buffered in the panel (entire message)
                            self._print_assistant_message(first_part_buffer, panel_title=panel_title, panel_border_style=panel_border_style)
                        first_part_printed = True
                        stream_ended_during_buffering = True
                        break
                    except (KeyboardInterrupt, Exception) as e:
                         # Ensure panel is printed if interrupted during buffering with content
                         if first_part_buffer and not first_part_printed:
                             self._print_assistant_message(first_part_buffer, panel_title=panel_title, panel_border_style=panel_border_style)
                         raise e # Re-raise

                # --- Stage 2: Live display for remaining stream (if \n\n was found) ---
                if first_part_printed and not stream_ended_during_buffering:
                    # Initialize live display seeded with content after marker
                    live_buffer = rest_after_marker
                    live_display = Live(
                        Align(Markdown(live_buffer.strip() + "▌"), align="left", pad=False), # Initial content + cursor
                        console=self.console,
                        refresh_per_second=15,
                        vertical_overflow="visible",
                        auto_refresh=False # Manual refresh control
                    )
                    live_display.start(refresh=True)
                    last_refresh_time = time.time()

                    # Process remaining items from iterator
                    for content in iterator:
                        if content:
                            total_response += content
                            token_count += len(content.split())
                            live_buffer += content

                            # Refresh Live display periodically
                            current_time = time.time()
                            if current_time - last_refresh_time > min_refresh_interval:
                                live_display.update(Align(Markdown(live_buffer.strip() + "▌"), align="left", pad=False), refresh=True)
                                last_refresh_time = current_time

                    # Final update without cursor
                    live_display.update(Align(Markdown(live_buffer.strip()), align="left", pad=False), refresh=True)

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
            except Exception as e:
                self.console.print(f"\n[bold red]Error during Rich streaming:[/bold red] {e}")
                total_response += f"\nERROR: {e}"
            finally:
                if live_display and live_display.is_started:
                    # Ensure final content is shown even if loop exited early
                    live_display.update(Align(Markdown(live_buffer.strip()), align="left", pad=False), refresh=True)
                    live_display.stop()
                # Add a final newline ONLY if Live display was used
                if live_display:
                    self.console.print()

        # Token counting and speed calculation
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.config.VERBOSE and elapsed_time > 0 and token_count > 0:
            tokens_per_second = token_count / elapsed_time
            self.console.print(f"[dim]Streamed {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.2f} tokens/sec)[/dim]")

        return total_response.strip()
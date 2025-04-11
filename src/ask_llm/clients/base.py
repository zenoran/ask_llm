from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from abc import ABC, abstractmethod
from typing import Iterator
import sys

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model):
        self.model = model
        self.console = Console(force_terminal=True)

    @abstractmethod
    def query(self, messages, prompt, plaintext_output: bool = False):
        """Query the LLM with the given messages and prompt.
        
        Args:
            messages: List of Message objects for conversation history.
            prompt: The current user prompt.
            plaintext_output: Whether to return plain text instead of formatted.
            
        Returns:
            str: The model's response.
        """
        pass

    def format_message(self, role, content):
        """Format a message based on its role using common helpers."""
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
            self._print_assistant_message(content)

    def _print_user_message(self, content):
        """Default implementation for printing user messages."""
        self.console.print()
        self.console.print("[bold blue]User:[/bold blue] ", end="")
        self.console.print(Markdown(content))

    def _print_assistant_message(self, content, panel_title=None, panel_border_style="green"):
        """Default implementation for printing assistant messages.
        Formats the assistant message with a panel for the first paragraph.
        Accepts optional title and border style.
        """
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if paragraphs:
            boxed_response = paragraphs[0]
            extra_response = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
        else:
            boxed_response = content.strip()
            extra_response = ""

        # Use provided title or default
        title = panel_title or f"[bold {panel_border_style}]{self.model}[/bold {panel_border_style}]"
        
        assistant_panel = Panel(
            Markdown(boxed_response),
            title=title,
            border_style=panel_border_style,
            padding=(1, 4),
        )
        # Align right by default
        self.console.print(Align(assistant_panel, align="right"))

        if extra_response:
            self.console.print() # Add space before extra content
            self.console.print(Markdown(extra_response))
            
    def _handle_streaming_output(
        self,
        stream_iterator: Iterator[str],
        plaintext_output: bool,
        panel_title: str | None = None,
        panel_border_style: str = "green",
        first_para_panel: bool = True
    ) -> str:
        """Handles streaming output for both plaintext and Rich display."""
        total_response = ""
        if plaintext_output:
            # --- Plaintext Streaming --- 
            print("", end='') # Ensure the line starts clear
            sys.stdout.flush()
            try:
                for content in stream_iterator:
                    if content:
                        total_response += content
                        print(content, end='')
                        sys.stdout.flush()
                print() # Add final newline
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
                print() # Ensure newline after interrupt message
            except Exception as e:
                # Log specific errors if needed, return what we have + error
                self.console.print(f"\n[bold red]Error during plaintext streaming:[/bold red] {e}")
                total_response += f"\nERROR: {e}"
            # --- End Plaintext --- 
        else:
            # --- Rich Streaming --- 
            accumulated_buffer = ""
            first_para_rendered = False
            live_display = None
            remaining_text_live = None

            try:
                # Use default title if none provided
                rich_panel_title = panel_title or f"[bold {panel_border_style}]{self.model}[/bold {panel_border_style}]"
                
                for content in stream_iterator:
                    if not content:
                        continue
                        
                    total_response += content
                    
                    if not first_para_panel:
                        # Simple Rich streaming: update a single panel
                        if live_display is None:
                            live_display = Live(
                                Panel(Markdown(content.strip()), title=rich_panel_title, border_style=panel_border_style, padding=(1, 2)),
                                refresh_per_second=15, console=self.console, vertical_overflow="visible"
                            )
                            live_display.start()
                        else:
                             live_display.update(Panel(Markdown(total_response.strip()), title=rich_panel_title, border_style=panel_border_style, padding=(1, 2)))
                        continue # Skip the first-paragraph logic

                    # First Paragraph Logic (for OpenAI/Ollama)
                    accumulated_buffer += content
                    
                    # Check if we have rendered the first paragraph and started the second display
                    if first_para_rendered and remaining_text_live and remaining_text_live.is_started:
                        remaining_text_live.update(Markdown(accumulated_buffer), refresh=True)
                        continue

                    # Check if we can render the first paragraph
                    if not first_para_rendered and "\n\n" in accumulated_buffer:
                        split_at = accumulated_buffer.find("\n\n")
                        first_para = accumulated_buffer[:split_at].strip()
                        remaining = accumulated_buffer[split_at + 2:]
                        
                        if first_para: # Only print if first para is not empty
                            # Display the first paragraph using the standard method
                            self._print_assistant_message(first_para, panel_title=rich_panel_title, panel_border_style=panel_border_style)
                            self.console.print() # Add space like original clients
                            first_para_rendered = True
                        
                        # Start a new Live display for remaining text
                        accumulated_buffer = remaining # Reset buffer for the rest
                        if remaining.strip():
                            remaining_text_live = Live(Markdown(remaining), auto_refresh=True, console=self.console)
                            remaining_text_live.start()
                        
                        # If we started the second display, continue to next chunk
                        if remaining_text_live and remaining_text_live.is_started:
                            continue
                            
                    # If we haven't started the second display yet, and we should have, start it now
                    if first_para_rendered and not (remaining_text_live and remaining_text_live.is_started) and accumulated_buffer.strip():
                        remaining_text_live = Live(Markdown(accumulated_buffer), auto_refresh=True, console=self.console)
                        remaining_text_live.start()

                # --- End of Stream Handling ---
                # Stop the live displays if they were started
                if live_display and live_display.is_started:
                    # Final update for single-panel mode
                    live_display.update(Panel(Markdown(total_response.strip()), title=rich_panel_title, border_style=panel_border_style, padding=(1, 2)))
                    live_display.stop()
                if remaining_text_live and remaining_text_live.is_started:
                    remaining_text_live.stop()
                
                # If first paragraph logic was enabled but never triggered (short response)
                if first_para_panel and not first_para_rendered and total_response.strip():
                     self._print_assistant_message(total_response, panel_title=rich_panel_title, panel_border_style=panel_border_style)
                     
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
                # Stop displays on interrupt
                if live_display and live_display.is_started:
                     live_display.stop()
                if remaining_text_live and remaining_text_live.is_started:
                    remaining_text_live.stop()
                # Don't raise, just return what we have
            except Exception as e:
                 # Log specific errors if needed, return what we have + error
                 self.console.print(f"\n[bold red]Error during Rich streaming:[/bold red] {e}")
                 total_response += f"\nERROR: {e}"
                 # Stop displays on error
                 if live_display and live_display.is_started:
                     live_display.stop()
                 if remaining_text_live and remaining_text_live.is_started:
                    remaining_text_live.stop()
            finally:
                # Ensure displays are stopped in all cases (redundant but safe)
                if live_display and live_display.is_started:
                     live_display.stop()
                if remaining_text_live and remaining_text_live.is_started:
                    remaining_text_live.stop()
                self.console.print() # Ensure a newline after Rich output finishes or is interrupted
            # --- End Rich --- 
            
        return total_response.strip()
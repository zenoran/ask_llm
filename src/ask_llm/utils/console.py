# filepath: ask_llm/src/utils/console.py

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.align import Align

console = Console(force_terminal=True)

def print_message(role, content):
    if role == "user":
        print_user_message(content)
    elif role == "assistant":
        print_assistant_message(content)

def print_user_message(content):
    console.print()
    console.print("[bold blue]User:[/bold blue]")
    console.print(Markdown(content))

def print_assistant_message(content):
    if not content.strip():
        console.print("[bold red]Error:[/bold red] Empty response received.")
        return

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    boxed_response = paragraphs[0] if paragraphs else content.strip()
    extra_response = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""

    # Dynamically calculate the width of the box
    max_line_length = max(len(line) for line in boxed_response.split("\n"))
    box_width = min(max_line_length + 8, console.width - 4)  # Add padding, limit to console width

    assistant_panel = Panel(
        Markdown(boxed_response),
        title="[bold green]Assistant[/bold green]",
        border_style="green",
        padding=(1, 4),
        width=box_width,  # Set the dynamic width
    )

    console.print(Align(assistant_panel, align="right"))

    if extra_response:
        console.print(Markdown(extra_response))

def print_debug_info(buffer_lines, preserve_blocks):
    console.print(f"[dim]BUFFER_LINES: {buffer_lines}, PRESERVE_BLOCKS: {preserve_blocks}[/dim]")

def print_response_progress(char_count):
    """Print real-time progress of collecting response"""
    # Use built-in print() instead of console.print() for progress
    print(f"\r[•] Collecting response... ({char_count} chars)", end="", flush=True)

def print_response_complete(char_count):
    """Print message when response collection is complete"""
    console.print()
    console.print(Rule(style="#444444"))
    console.print(f"[#555555]Response complete: {char_count} total characters[/#555555]")

def clear_status_line():
    """Clear the current status line."""
    print("\r" + " " * 80, end="\r", flush=True)

def clear_console():
    """Clear the console to ensure consistent rendering."""
    console.clear()
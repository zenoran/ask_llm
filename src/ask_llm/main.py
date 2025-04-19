#!/usr/bin/env python
import sys
from rich.console import Console
from ask_llm.cli import main as cli_main # Import the main function from cli

console = Console()

def main() -> None:
  
    try:
        cli_main()
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in the application:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


import sys
from rich.prompt import Prompt
from rich.console import Console

class MultilineInputHandler:
    """Handles both single-line and multiline input."""
    
    def __init__(self, console=None):
        self.console = console or Console()
        
    def get_input(self, prompt_text="Enter your question (type '>' for multiline input):"):
        """
        Get user input, supporting both single and multiline modes.
        
        Args:
            prompt_text: The text to display before starting input collection
            
        Returns:
            tuple: (input_text, is_multiline)
        """
        # Display prompt and get initial input
        self.console.print(f"[bold blue]{prompt_text}[/bold blue]")
        
        # Handle non-interactive mode
        if not sys.stdin.isatty():
            return sys.stdin.read(), False
        
        # Get initial input
        try:
            initial_input = input()
            
            # Check if multiline mode is requested
            if initial_input.strip() == ">":
                self.console.print("[bold blue]Multiline mode (type 'EOF' on a new line or press Ctrl+C to finish):[/bold blue]")
                return self._get_multiline_input(), True
            elif initial_input.strip().startswith(">"):
                # If input begins with '>' but has more content, treat the rest as first line of multiline
                self.console.print("[bold blue]Multiline mode (type 'EOF' on a new line or press Ctrl+C to finish):[/bold blue]")
                first_line = initial_input.strip()[1:].lstrip()
                return self._get_multiline_input(first_line=first_line), True
            else:
                # Regular single-line input
                return initial_input, False
                
        except KeyboardInterrupt:
            # Handle Ctrl+C in single-line mode - propagate the interrupt
            self.console.print("\n[yellow]Input cancelled.[/yellow]")
            raise
    
    def _get_multiline_input(self, first_line=None):
        """Get input in multiline mode."""
        lines = []
        if first_line:
            lines.append(first_line)
        
        try:
            while True:
                line = input()
                # Allow user to exit by typing EOF
                if line.strip() == "EOF":
                    break
                lines.append(line)
        except KeyboardInterrupt:
            # Handle Ctrl+C in multiline mode by completing input
            self.console.print("\n[italic]Input complete.[/italic]")
            return "\n".join(lines)
            
        self.console.print("[italic]Input complete.[/italic]")
        return "\n".join(lines)
    
    def preview_input(self, text):
        """Preview the input as plain text before sending."""
        if not text.strip():
            return ""
            
        self.console.print()
        self.console.print("[bold blue]Input Preview:[/bold blue]")
        # Print raw text without Rich formatting
        print(text)
        
        confirmation = Prompt.ask(
            "Send this input? ",
            choices=["y", "n", "e"],
            default="y"
        )
        
        if confirmation == "e":
            # Allow editing
            self.console.print("[blue]Edit your input (type 'EOF' on a new line or press Ctrl+C to finish):[/blue]")
            edited_text = self._get_multiline_input(first_line=text.split("\n")[0] if text else None)
            return edited_text if edited_text.strip() else text
        
        return text if confirmation == "y" else ""

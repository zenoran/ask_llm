from rich.console import Console
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model):
        self.model = model
        self.console = Console()

    @abstractmethod
    def query(self, messages, prompt):
        """Query the LLM with the given messages and prompt.
        
        Args:
            messages: List of Message objects for conversation history.
            prompt: The current user prompt.
            
        Returns:
            str: The model's response.
        """
        pass

    @abstractmethod
    def format_response(self, response_text):
        """Format the response for display."""
        pass

    @abstractmethod
    def format_message(self, role, content):
        """Format a message (either user or assistant) for display."""
        pass
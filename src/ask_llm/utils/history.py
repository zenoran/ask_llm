import os
import json
import time
from ..clients.base import LLMClient
from ..models.message import Message
from rich.rule import Rule
from .config import Config


class HistoryManager:
    messages: list[Message] = []
    client: LLMClient
    config: Config

    def __init__(self, client: LLMClient, config: Config):
        self.client = client
        self.config = config
        self.history_file = config.HISTORY_FILE
        self.messages = []

    def load_history(self):
        """Load all persisted history."""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    message_dicts = json.load(f)
                    self.messages = [Message.from_dict(msg) for msg in message_dicts]
            except UnicodeDecodeError as e:
                self.client.console.print(f"[bold red]Error loading history:[/bold red] Unable to decode file. Ensure it is saved in UTF-8 format. ({e})")
            except json.JSONDecodeError as e:
                self.client.console.print(f"[bold red]Error loading history:[/bold red] Invalid JSON format in {self.history_file}. ({e})")
            except Exception as e:
                self.client.console.print(f"[bold red]Error loading history:[/bold red] {e}")

    def save_history(self):
        """Persist message history to the history file."""
        if not self.history_file:
            return
        try:
            message_dicts = [msg.to_dict() for msg in self.messages]
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(message_dicts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.client.console.print(f"[bold red]Error saving history:[/bold red] {e}")

    def get_context_messages(self):
        """Get messages to be used as context for the LLM."""
        cutoff = time.time() - self.config.HISTORY_DURATION
        active_messages = [
            msg for msg in self.messages if msg.role == "system" or msg.timestamp >= cutoff
        ]
        if not any(msg.role == "system" for msg in active_messages):
            active_messages.insert(0, Message(role="system", content=self.config.SYSTEM_MESSAGE))
        return active_messages
    
    def get_context_messages_excluding_last(self):
        """Get messages to be used as context, excluding the most recent one."""
        context_messages = self.get_context_messages()
        if len(context_messages) > 1:
            return context_messages[:-1]
        elif context_messages and context_messages[0].role == "system":
            return context_messages
        else:
            if any(msg.role == "system" for msg in context_messages):
                return [msg for msg in context_messages if msg.role == "system"]
            else:
                return [Message(role="system", content=self.config.SYSTEM_MESSAGE)]

    def add_message(self, role, content):
        """Append a message to history and save."""
        message = Message(role, content)
        self.messages.append(message)
        self.save_history()


    def print_history(self, pairs_limit=None):
        """Print the conversation history in a formatted way.

        Args:
            pairs_limit: Number of recent conversation pairs to show (-1 for all).
        """
        if not self.messages:
            self.client.console.print("[italic]No conversation history found.[/italic]")
            return

        non_system_messages = [msg for msg in self.messages if msg.role != "system"]

        if not non_system_messages:
            self.client.console.print("[italic]No conversation messages found.[/italic]")
            return

        if pairs_limit is not None and pairs_limit != -1:
            messages_to_show = min(pairs_limit * 2, len(non_system_messages))
            non_system_messages = non_system_messages[-messages_to_show:]

        self.client.console.print()
        self.client.console.print("[bold]Conversation History:[/bold]")
        self.client.console.print(Rule(style="#555555"))

        for msg in non_system_messages:
            if msg.role == "user":
                self.client._print_user_message(msg.content)
            elif msg.role == "assistant":
                panel_title, panel_border_style = self.client.get_styling()
                parts = msg.content.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self.client._print_assistant_message(
                    first_part,
                    second_part=second_part,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style
                )
            self.client.console.print(Rule(style="#333333"))
    
    def clear_history(self):
        """Clear the conversation history from memory and disk."""
        self.messages = []
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                self.client.console.print("[bold red]History cleared.[/bold red]")
            except Exception as e:
                self.client.console.print(f"[bold red]Error clearing history file:[/bold red] {e}")
        else:
            self.client.console.print("[dim]No history file found to clear or history file path not set.[/dim]")
            
    def get_last_assistant_message(self) -> str | None:
        """Get the content of the last assistant message, or None."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def remove_last_message_if_partial(self, role: str):
        """Remove the last message if it matches the specified role (used for cleanup on error/interrupt)."""
        if self.messages and self.messages[-1].role == role:
            self.messages.pop()
            self.save_history()
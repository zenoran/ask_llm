import os
import json
import time
from ask_llm.clients.base import LLMClient
from ask_llm.models.message import Message
from rich.rule import Rule
from ask_llm.utils.config import config


HISTORY_FILE = os.path.join(
    os.getenv("USERPROFILE") or os.getenv("HOME"), config.HISTORY_FILE
)
HISTORY_DURATION = 60 * 10  # retain messages for 60 minutes


class HistoryManager:
    messages: list[Message] = []
    history_file: str | None = None
    client: LLMClient


    def __init__(self, client: LLMClient, history_file=HISTORY_FILE):
        self.client = client
        self.history_file = history_file

    def load_history(self):
        """Load all persisted history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    message_dicts = json.load(f)
                    self.messages = [Message.from_dict(msg) for msg in message_dicts]
            except UnicodeDecodeError as e:
                print(f"Error loading history: Unable to decode file. Ensure it is saved in UTF-8 format. ({e})")
            except Exception as e:
                print(f"Error loading history: {e}")

    def save_history(self):
        """Persist message history to the history file."""
        try:
            message_dicts = [msg.to_dict() for msg in self.messages]
            with open(self.history_file, "w", encoding="utf-8") as f:
                # Use ensure_ascii=False to prevent escaping of Unicode characters
                json.dump(message_dicts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")

    def get_context_messages(self):
        """Get messages to be used as context for the LLM."""
        cutoff = time.time() - HISTORY_DURATION
        messages =  [
            msg for msg in self.messages if msg.role == "system" or msg.timestamp >= cutoff
        ]
        if "system" not in messages:
            # Add a system message if none exists
            messages.insert(0, Message(role="system", content=config.SYSTEM_MESSAGE))
        return messages
    
    def get_context_messages_excluding_last(self):
        """Get messages to be used as context, excluding the most recent one."""
        context_messages = self.get_context_messages()
        # Return all but the last message if there's more than one
        if len(context_messages) > 1:
            return context_messages[:-1]
        elif context_messages and context_messages[0].role == "system":
            # If only the system message exists, return it
            return context_messages
        else:
            # If there's only one non-system message (shouldn't happen in normal flow)
            # or the list is empty, return an empty list or just system message
            # Let's refine: ensure system message is always present if history is used
            if any(msg.role == "system" for msg in context_messages):
                return [msg for msg in context_messages if msg.role == "system"]
            else:
                # Add system message if missing (edge case)
                return [Message(role="system", content=config.SYSTEM_MESSAGE)]

    def add_message(self, role, content):
        """Append a message to history."""

        message = Message(role, content)
        self.messages.append(message)
        self.save_history()


    def print_history(self, pairs_limit=None):
        """Print the conversation history in a formatted way.

        Args:
            console: Rich console for output
            pairs_limit: Number of recent conversation pairs to show.
                         Use -1 for no limit.
        """
        if not self.messages:
            self.client.console.print("[italic]No conversation history found.[/italic]")
            return

        # Filter out system messages
        non_system_messages = [msg for msg in self.messages if msg.role != "system"]

        if not non_system_messages:
            self.client.console.print("[italic]No conversation messages found.[/italic]")
            return

        # If pairs_limit == -1, show full history
        if pairs_limit is not None and pairs_limit != -1:
            # A pair is user+assistant, so need 2*limit messages
            messages_to_show = min(pairs_limit * 2, len(non_system_messages))
            non_system_messages = non_system_messages[-messages_to_show:]

        self.client.console.print()
        self.client.console.print("[bold]Conversation History:[/bold]")
        self.client.console.print(Rule(style="#555555"))

        for msg in non_system_messages:
            self.client.format_message(msg.role, msg.content)
            self.client.console.print(Rule(style="#333333"))
    
    def clear_history(self):
        """Clear the conversation history."""
        self.messages = []
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                self.client.console.print("[bold red]History cleared.[/bold red]")
            except Exception as e:
                self.client.console.print(f"[bold red]Error clearing history: {e}[/bold red]")
        else:
            self.client.console.print("[bold red]No history file found to clear.[/bold red]")
            
    def get_last_assistant_message(self):
        """Get the last assistant message content for comparison.
        
        Returns:
            str or None: The content of the last assistant message, or None if no assistant messages exist.
        """
        # Filter to assistant messages only and get the most recent one
        assistant_messages = [msg for msg in self.messages if msg.role == "assistant"]
        if assistant_messages:
            return assistant_messages[-1].content
        return None
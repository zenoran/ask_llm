import json
import os
import sys
from openai import OpenAI
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.align import Align
from rich.live import Live
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import config

class OpenAIClient(LLMClient):
    """Client for OpenAI API"""

    def __init__(self, model):
        super().__init__(model)
        self.api_key = self._get_api_key()
        self.client = OpenAI(api_key=self.api_key)

    def _get_api_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Please set your OPENAI_API_KEY environment variable.")
        return api_key

    def query(self, messages, plaintext_output: bool = False):
        """Query OpenAI API with full message history, using streaming by default.

        Args:
            messages: List of message dictionaries (including the latest user prompt).
            plaintext_output: If True, return raw text. Otherwise, format output.
        """
        # Messages are now passed directly, assuming the prompt is the last one
        api_messages = self._prepare_api_messages(messages)
        response = self._stream_response(api_messages, plaintext_output)
        return response

    def _prepare_api_messages(self, messages):
        # Convert all messages in the list to API format
        api_messages = [msg.to_api_format() if hasattr(msg, 'to_api_format') else msg for msg in messages]
        # api_messages.append({"role": "user", "content": prompt}) # REMOVED
        return api_messages

    def _stream_response(self, api_messages, plaintext_output: bool = False):
        """Stream the response using the base class handler."""
        if config.VERBOSE:
            self.console.print("[bold blue]Verbose Output:[/bold blue]")
            self.console.print_json(json.dumps(api_messages))

        try:
            api_request = self.client.chat.completions.create(
                model=self.model, messages=api_messages, stream=True, store=False
            )
        except Exception as e:
            self.console.print(f"[bold red]Error making OpenAI API request:[/bold red] {e}")
            return f"ERROR: {e}"

        def _iterate_openai_chunks(stream):
            try:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            except Exception as e:
                 # This might catch API errors or connection issues during streaming
                 self.console.print(f"\n[bold red]Error during OpenAI stream processing:[/bold red] {e}")
                 yield f"\nERROR: {e}"
                 # We might want to raise an exception here depending on desired handling

        # Pass the iterator to the base handler
        return self._handle_streaming_output(
            stream_iterator=_iterate_openai_chunks(api_request),
            plaintext_output=plaintext_output,
            # Keep OpenAI default panel style (green)
            first_para_panel=True
        )

    def get_verbose_output(self, messages):
        """Get full API response for verbose output"""
        # Convert all messages in the list to API format
        api_messages = [msg.to_api_format() if hasattr(msg, 'to_api_format') else msg for msg in messages]
        # api_messages.append({"role": "user", "content": prompt}) # REMOVED

        result = self.client.chat.completions.create(
            model=self.model, messages=api_messages, store=False
        )
        return json.dumps(result, indent=2)

    def _print_buffer(self, buffer):
        """Print buffered lines to the console."""
        for line in buffer:
            self.console.print(line)

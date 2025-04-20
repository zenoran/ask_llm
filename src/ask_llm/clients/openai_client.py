import json
import os
from openai import OpenAI, OpenAIError
from typing import List, Iterator
from rich.json import JSON
from rich.rule import Rule
from ..clients.base import LLMClient
from ..utils.config import Config # Keep for type hinting
from ..models.message import Message
import logging # Import logging

logger = logging.getLogger(__name__)

class OpenAIClient(LLMClient):
    """Client for OpenAI API"""

    def __init__(self, model: str, config: Config):
        super().__init__(model, config) # Pass config to base class
        self.api_key = self._get_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def _get_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, **kwargs) -> str:
        """Query OpenAI API with full message history, using streaming by default.

        Args:
            messages: List of Message objects.
            plaintext_output: If True, return raw text.
            stream: Whether to stream the response.
            **kwargs: Additional arguments (ignored by this client).

        Returns:
            The model's response as a string.
        """
        api_messages = self._prepare_api_messages(messages)
        should_stream = stream and not self.config.NO_STREAM
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": should_stream,
        }
        if self.config.VERBOSE:
            self.console.print(Rule("Querying OpenAI API", style="green"))
            self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={payload['max_tokens']}, temp={payload['temperature']}, top_p={payload['top_p']}, stream={payload['stream']}[/italic]")
            self.console.print(Rule("Request Payload", style="dim blue"))
            try:
                payload_str = json.dumps(payload, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                self.console.print(f"[red]Error printing payload:[/red] {e}")
                import pprint
                self.console.print(pprint.pformat(payload))
            self.console.print(Rule(style="green"))
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"OpenAI Request Payload: {json.dumps(payload)}")

        if should_stream:
            response = self._stream_response(api_messages, plaintext_output, payload)
        else:
            response = self._get_full_response(api_messages, plaintext_output, payload)

        if self.config.VERBOSE:
             self.console.print(Rule(style="green"))

        return response

    def _prepare_api_messages(self, messages: List[Message]) -> list[dict]:
        prepared = []
        has_system = False
        system_message = self.config.SYSTEM_MESSAGE
        for msg in messages:
            api_msg = msg.to_api_format()
            if api_msg['role'] == 'system':
                if not has_system:
                    prepared.insert(0, api_msg)
                    has_system = True
            else:
                prepared.append(api_msg)

        if not has_system and system_message:
            prepared.insert(0, {"role": "system", "content": system_message})

        return prepared

    def _stream_response(self, api_messages: List[dict], plaintext_output: bool, payload: dict) -> str:
        """Stream the response using the base class handler."""
        try:
            stream = self.client.chat.completions.create(**payload)
            return self._handle_streaming_output(
                stream_iterator=self._iterate_openai_chunks(stream),
                plaintext_output=plaintext_output,
            )
        except OpenAIError as e:
            logger.error(f"Error during OpenAI API request: {e}")
            return f"ERROR: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI streaming: {e}")
            return f"ERROR: Unexpected error - {e}"

    def _iterate_openai_chunks(self, stream: Iterator) -> Iterator[str]:
        """Iterates through OpenAI stream chunks and yields content."""
        try:
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except OpenAIError as e:
            logger.error(f"\nError during OpenAI stream processing: {e}")
            yield f"\nERROR: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"\nUnexpected error during OpenAI stream iteration: {e}")
            yield f"\nERROR: Unexpected error - {e}"

    def _get_full_response(self, api_messages: List[dict], plaintext_output: bool, payload: dict) -> str:
        """Gets the full response without streaming."""
        try:
            payload['stream'] = False
            completion = self.client.chat.completions.create(**payload)
            response_text = completion.choices[0].message.content or ""
            if self.config.VERBOSE:
                usage = completion.usage
                if usage:
                    self.console.print(f"[dim]OpenAI Tokens: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}[/dim]")
            elif logger.isEnabledFor(logging.DEBUG):
                usage = completion.usage
                if usage:
                    logger.debug(f"OpenAI Tokens: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")

            if not plaintext_output:
                parts = response_text.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self._print_assistant_message(first_part, second_part=second_part)

            return response_text.strip()

        except OpenAIError as e:
            logger.error(f"Error making OpenAI API request: {e}")
            return f"ERROR: OpenAI API Error - {e}"
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI request: {e}")
            return f"ERROR: Unexpected error - {e}"

    def get_styling(self) -> tuple[str | None, str]:
        """Return OpenAI specific styling."""
        panel_border_style = "green"
        panel_title = f"[bold {panel_border_style}]{self.model}[/bold {panel_border_style}]"
        return panel_title, panel_border_style

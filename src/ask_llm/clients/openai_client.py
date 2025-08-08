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
    SUPPORTS_STREAMING = True

    def __init__(self, model: str, config: Config):
        super().__init__(model, config) # Pass config to base class
        self.api_key = self._get_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def _get_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def _model_supports_temperature_top_p(self) -> bool:
        """Check if the model supports temperature and top_p parameters.
        
        Some specialized models like gpt-4o-search-preview don't support these parameters.
        """
        # Models that don't support temperature/top_p
        unsupported_models = [
            "gpt-4o-search-preview",
            "gpt-4o-audio-preview",
            # Add other models here as needed
        ]
        return self.model not in unsupported_models

    # ---- Token parameter handling helpers ----
    _TOKEN_PARAM_CANDIDATES = ("max_tokens", "max_completion_tokens", "max_output_tokens")

    def _likely_new_api_model(self) -> bool:
        """Heuristic: some newer models use 'max_completion_tokens' (or 'max_output_tokens')."""
        m = self.model.lower()
        return (
            m.startswith("o3") or m.startswith("o4") or
            m in {"gpt-4o-search-preview", "gpt-4o-audio-preview"}
        )

    def _initial_token_param_key(self) -> str:
        # Prefer legacy 'max_tokens' unless the model looks like a newer family
        return "max_completion_tokens" if self._likely_new_api_model() else "max_tokens"

    def _set_token_param(self, payload: dict, key: str, value: int) -> dict:
        # Remove any existing token keys and set the desired one
        for k in self._TOKEN_PARAM_CANDIDATES:
            payload.pop(k, None)
        payload[key] = value
        return payload

    def _print_verbose_params(self, payload: dict) -> None:
        token_key = next((k for k in self._TOKEN_PARAM_CANDIDATES if k in payload), None)
        token_info = f"{token_key}={payload[token_key]}" if token_key else "max_tokens=N/A"
        temp_info = f"temp={payload.get('temperature', 'N/A')}" if 'temperature' in payload else "temp=N/A"
        top_p_info = f"top_p={payload.get('top_p', 'N/A')}" if 'top_p' in payload else "top_p=N/A"
        self.console.print(f"[dim]Params:[/dim] [italic]{token_info}, {temp_info}, {top_p_info}, stream={payload['stream']}[/italic]")

    def _chat_create_with_fallback(self, payload: dict):
        """Call chat.completions.create with fallback across token param names.

        Tries the current token key in payload first, then falls back to the other
        known keys if OpenAI returns an unsupported parameter error.
        """
        # Build ordered list of keys to try, preferring what's already in payload
        current_key = next((k for k in self._TOKEN_PARAM_CANDIDATES if k in payload), self._initial_token_param_key())
        keys_to_try = [current_key] + [k for k in self._TOKEN_PARAM_CANDIDATES if k != current_key]

        last_err: Exception | None = None
        for key in keys_to_try:
            try_payload = dict(payload)
            self._set_token_param(try_payload, key, self.config.MAX_TOKENS)
            attempted_without_temp_top_p = False
            try:
                return self.client.chat.completions.create(**try_payload)
            except OpenAIError as e:
                msg = str(e).lower()
                # If error suggests wrong token key, continue; else re-raise
                if ("unsupported parameter" in msg and any(k in msg for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"))) or (
                    "invalid_request_error" in msg and "token" in msg
                ):
                    last_err = e
                    continue
                # Temperature/top_p unsupported or unsupported value -> retry without them once
                if ("temperature" in msg or "top_p" in msg) and ("unsupported" in msg or "does not support" in msg or "unsupported_value" in msg):
                    if not attempted_without_temp_top_p:
                        try_payload_no_temp = dict(try_payload)
                        try_payload_no_temp.pop('temperature', None)
                        try_payload_no_temp.pop('top_p', None)
                        attempted_without_temp_top_p = True
                        try:
                            return self.client.chat.completions.create(**try_payload_no_temp)
                        except OpenAIError as e2:
                            # If still token key issue, move to next token key; else record and break to next key
                            m2 = str(e2).lower()
                            if ("unsupported parameter" in m2 and any(k in m2 for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"))) or (
                                "invalid_request_error" in m2 and "token" in m2
                            ):
                                last_err = e2
                                continue
                            last_err = e2
                            break
                        except Exception as e2:
                            last_err = e2
                            break
                    last_err = e
                    break
                raise
            except Exception as e:
                last_err = e
                break
        # If we exhausted retries, raise the last error
        if last_err:
            raise last_err
        # Should not reach here
        return self.client.chat.completions.create(**payload)

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
            "stream": should_stream,
        }
        # Set initial token parameter key
        self._set_token_param(payload, self._initial_token_param_key(), self.config.MAX_TOKENS)
        
        # Only add temperature and top_p for models that support them
        if self._model_supports_temperature_top_p():
            payload["temperature"] = self.config.TEMPERATURE
            payload["top_p"] = self.config.TOP_P
        if self.config.VERBOSE:
            self.console.print(Rule("Querying OpenAI API", style="green"))
            self._print_verbose_params(payload)
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
            # Try create with fallback token parameter handling
            stream = self._chat_create_with_fallback(payload)
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
            completion = self._chat_create_with_fallback(payload)
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

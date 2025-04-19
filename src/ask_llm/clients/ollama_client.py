import json
import requests
from typing import List, Iterator, Optional
import logging
from rich.json import JSON
from rich.rule import Rule

from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import Config # Keep for type hinting
from ask_llm.models.message import Message

# Get logger instance
logger = logging.getLogger(__name__)

class OllamaClient(LLMClient):
    def __init__(self, model: str, config: Config):
        super().__init__(model, config) # Pass config to base class
        # self.config = config # Stored in base class

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, **kwargs) -> str:
        """Query Ollama API with full message history, supporting streaming.

        Args:
            messages: List of Message objects.
            plaintext_output: If True, return raw text.
            stream: Whether to stream the response (always True for Ollama in this impl).
            **kwargs: Additional arguments (ignored).

        Returns:
            The model's response as a string.
        """
        api_messages = self._prepare_api_messages(messages)
        # Use self.config for stream check
        should_stream = stream and not self.config.NO_STREAM
        if should_stream:
            response = self._stream_response(api_messages, plaintext_output)
        else:
            response = self._get_full_response(api_messages, plaintext_output)
        return response

    def _prepare_api_messages(self, messages: List[Message]) -> List[dict]:
        api_messages = []
        has_system_message = False
        # Use self.config for SYSTEM_MESSAGE
        system_message = self.config.SYSTEM_MESSAGE
        for msg in messages:
            formatted_msg = msg.to_api_format()
            if formatted_msg['role'] == 'system':
                if not has_system_message:
                    api_messages.insert(0, formatted_msg)
                    has_system_message = True
                # Skip subsequent system messages
            else:
                api_messages.append(formatted_msg)

        if not has_system_message and system_message:
            api_messages.insert(0, {"role": "system", "content": system_message})

        # Clean up consecutive roles and filter error messages
        cleaned_messages = []
        if api_messages:
            cleaned_messages.append(api_messages[0])
            for i in range(1, len(api_messages)):
                if api_messages[i]['role'] != api_messages[i-1]['role']:
                    cleaned_messages.append(api_messages[i])

        final_messages = [
            msg for msg in cleaned_messages
            if not (msg.get('role') == 'assistant' and msg.get('content', '').startswith('ERROR:'))
        ]
        return final_messages

    def _stream_response(self, api_messages: List[dict], plaintext_output: bool) -> str:
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "options": {
                "temperature": self.config.TEMPERATURE,
                "num_predict": self.config.MAX_TOKENS,
                "top_p": self.config.TOP_P
            }
        }

        # Use self.config.VERBOSE for direct console output
        if self.config.VERBOSE:
            self.console.print(Rule("Querying Ollama API (Streaming)", style="purple"))
            options = payload.get('options', {})
            self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={options.get('num_predict')}, temp={options.get('temperature')}, top_p={options.get('top_p')}, stream={payload['stream']}[/italic]")
            self.console.print(Rule("Request Payload", style="dim blue"))
            try:
                payload_str = json.dumps(payload, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                self.console.print(f"[red]Error printing payload:[/red] {e}")
                import pprint
                self.console.print(pprint.pformat(payload))
            self.console.print(Rule(style="purple"))
        elif logger.isEnabledFor(logging.DEBUG):
            # Fallback to logger if not verbose but debug is enabled
            logger.debug(f"Ollama Request Payload: {json.dumps(payload)}")

        try:
            # Use self.config for URL and generation params
            response = requests.post(
                f"{self.config.OLLAMA_URL}/api/chat",
                json=payload, # Use the constructed payload
                stream=True,
                timeout=120 # Longer timeout for potentially slow model loads
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            error_msg = f"Could not connect to Ollama server at {self.config.OLLAMA_URL}. Ensure it is running." if isinstance(e, requests.exceptions.ConnectionError) else str(e)
            self.console.print(f"[bold red]Ollama Connection Error:[/bold red] {error_msg}")
            return f"ERROR: {error_msg}"
        except Exception as e:
            self.console.print(f"[bold red]Initial Ollama Request Error:[/bold red] {str(e)}")
            return f"ERROR: {str(e)}"

        # Pass the iterator to the base handler
        result = self._handle_streaming_output(
            stream_iterator=self._iterate_ollama_chunks(response, plaintext_output),
            plaintext_output=plaintext_output,
        )

        if self.config.VERBOSE:
             self.console.print(Rule(style="purple")) # Add closing rule for verbose

        return result # Return the result from _handle_streaming_output

    def _get_full_response(self, api_messages: List[dict], plaintext_output: bool) -> str:
        payload = {
            "model": self.model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": self.config.TEMPERATURE,
                "num_predict": self.config.MAX_TOKENS,
                "top_p": self.config.TOP_P
            }
        }

        # Use self.config.VERBOSE for direct console output
        if self.config.VERBOSE:
            self.console.print(Rule("Querying Ollama API (Non-Streaming)", style="purple"))
            options = payload.get('options', {})
            self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={options.get('num_predict')}, temp={options.get('temperature')}, top_p={options.get('top_p')}, stream={payload['stream']}[/italic]")
            self.console.print(Rule("Request Payload", style="dim blue"))
            try:
                payload_str = json.dumps(payload, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                self.console.print(f"[red]Error printing payload:[/red] {e}")
                import pprint
                self.console.print(pprint.pformat(payload))
            self.console.print(Rule(style="purple"))
        elif logger.isEnabledFor(logging.DEBUG):
            # Fallback to logger if not verbose but debug is enabled
            logger.debug(f"Ollama Request Payload (non-streaming): {json.dumps(payload)}")

        try:
            # Use self.config for URL and generation params
            response = requests.post(
                f"{self.config.OLLAMA_URL}/api/chat",
                json=payload, # Use the constructed payload
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get('message', {}).get('content', '')
            if not plaintext_output:
                 # Split response at the first \n\n
                 parts = response_text.split("\n\n", 1)
                 first_part = parts[0]
                 second_part = parts[1] if len(parts) > 1 else None
                 # Use appropriate style
                 self._print_assistant_message(first_part, second_part=second_part, panel_border_style="purple")
            return response_text.strip()

        except requests.exceptions.RequestException as e:
            error_msg = f"Could not connect to Ollama server at {self.config.OLLAMA_URL}. Ensure it is running." if isinstance(e, requests.exceptions.ConnectionError) else str(e)
            self.console.print(f"[bold red]Ollama Connection Error:[/bold red] {error_msg}")
            return f"ERROR: {error_msg}"
        except Exception as e:
            self.console.print(f"[bold red]Ollama Request Error:[/bold red] {str(e)}")
            return f"ERROR: {str(e)}"

        if self.config.VERBOSE:
             self.console.print(Rule(style="purple")) # Add closing rule for verbose

        return response_text # Return the stripped text

    def _iterate_ollama_chunks(self, http_response: requests.Response, plaintext_output: bool) -> Iterator[str]:
        """Iterates through Ollama stream chunks, handles errors and thought tags."""
        total_thought = ""
        in_thought = False
        thought_buffer = ""
        error_reported = False # Prevent multiple error yields

        try:
            for line in http_response.iter_lines():
                if not line or error_reported:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    self.console.print(f"[yellow]Warning: Could not decode JSON line: {line}[/yellow]")
                    continue

                if "error" in chunk:
                    error_msg = chunk["error"]
                    self.console.print(f"[bold red]Ollama Error:[/bold red] {error_msg}")
                    if "GGML_ASSERT" in error_msg:
                        self.console.print("[yellow]This might be a model compatibility issue. Try a different model or quantization.[/yellow]")
                    yield f"ERROR: {error_msg}"
                    error_reported = True
                    continue # Stop processing lines after error

                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if not content:
                        continue

                    if not plaintext_output and ('<' in content and '>' in content): # Basic check for tags
                        current_content_part = content
                        processed_content = ""
                        while current_content_part:
                            if not in_thought:
                                if "<thought>" in current_content_part:
                                    in_thought = True
                                    parts = current_content_part.partition("<thought>")
                                    processed_content += parts[0]
                                    thought_buffer += parts[2].replace("\n", " ") # Normalize newlines in thought
                                    current_content_part = ""
                                    if parts[0].strip(): yield parts[0]
                                    if not error_reported: self.console.print("[blue i]Thinking...[/blue i]", end='\r') # Overwrite thinking message
                                else:
                                    processed_content += current_content_part
                                    current_content_part = ""
                            else: # in_thought
                                if "</thought>" in current_content_part:
                                    in_thought = False
                                    parts = current_content_part.partition("</thought>")
                                    thought_buffer += parts[0].replace("\n", " ")
                                    current_content_part = parts[2]
                                    total_thought += thought_buffer.strip()
                                    # Replace config.VERBOSE check with logger level check
                                    # if config.VERBOSE and not error_reported:
                                    if logger.isEnabledFor(logging.DEBUG) and not error_reported:
                                        # Replace console.print with logger.debug
                                        # self.console.print(f"[#555555 i]Thought: {total_thought}[/#555555 i]       ") # Clear thinking message
                                        logger.debug(f"Thought: {total_thought}")
                                        self.console.print(" "*20, end='\r') # Still clear the 'Thinking...' message
                                    total_thought = ""
                                    thought_buffer = ""
                                else:
                                    thought_buffer += current_content_part.replace("\n", " ")
                                    current_content_part = ""

                        if processed_content: yield processed_content
                    else:
                        yield content # Plaintext or no tags found
        except Exception as e:
            if not error_reported:
                 self.console.print(f"\n[bold red]Error during Ollama stream processing:[/bold red] {str(e)}")
                 yield f"\nERROR: {str(e)}"
        finally:
            if not error_reported: self.console.print(" " * 20, end='\r') # Clear thinking message if active
            http_response.close()

    def get_styling(self) -> tuple[str | None, str]:
        """Return Ollama specific styling."""
        panel_border_style = "purple"
        # Use the same title format as in _stream_response / _get_full_response if needed
        # Example: panel_title = f"[bold {panel_border_style}]{self.model}[/bold {panel_border_style}]"
        panel_title = None # Default title format uses model name and border style
        return panel_title, panel_border_style

import time
import logging
import os
import contextlib
import json
from typing import List, Dict, Any, Iterator

from rich.markdown import Markdown
from rich.rule import Rule
from rich.json import JSON

from ..clients.base import LLMClient
from ..utils.config import Config # Keep for type hinting
from ..models.message import Message

try:
    from llama_cpp import Llama
    _llama_cpp_available = True
except ImportError:
    Llama = None
    _llama_cpp_available = False

logger = logging.getLogger(__name__)

class LlamaCppClient(LLMClient):
    """Client for running GGUF models using llama-cpp-python."""

    def __init__(self, model_path: str, config: Config):
        if not _llama_cpp_available:
            raise ImportError(
                "`llama-cpp-python` not found. Install it following instructions: "
                "https://github.com/abetlen/llama-cpp-python#installation"
            )
        super().__init__(model_path, config) # Pass config to base class
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the GGUF model, suppressing C++ library stderr."""
        if self.config.VERBOSE:
            self.console.print(f"Loading GGUF model: [bold yellow]{self.model_path}[/bold yellow]...")

        # TODO: Make these parameters configurable via models.yaml
        n_gpu_layers = self.config.LLAMA_CPP_N_GPU_LAYERS # -1 means load all possible layers to GPU
        n_ctx = self.config.LLAMA_CPP_N_CTX # Context size
        chat_format = self.config.CHAT_FORMAT or "chatml" # Default to chatml based on model template

        model_load_params = {
            "model_path": self.model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "chat_format": chat_format,
            "verbose": False, # Disable llama-cpp's library-level verbose logging
            # Add other potential params like rope_freq_base, rope_freq_scale etc.
        }

        # === Add final confirmation log ===
        final_chat_format = model_load_params["chat_format"]
        if self.config.VERBOSE:
            log_params = {k: v for k, v in model_load_params.items() if k != 'model_path'}
            logger.debug(f"Final chat_format passed to Llama(): {final_chat_format!r}")
            logger.debug(f"llama.cpp model load parameters (excluding path): {log_params}")

        try:
            # Redirect C++ stderr to /dev/null to silence noisy library output ONLY if not verbose
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
                    if Llama is None: # Should have been caught in __init__, but double-check
                        raise ImportError("Llama class is not available from llama_cpp.")
                    self.model = Llama(**model_load_params)

            if self.config.VERBOSE and self.model:
                ctx_size = getattr(self.model, 'n_ctx', 'N/A')
                gpu_layers = model_load_params['n_gpu_layers']
                self.console.print(
                    f"[green]Model loaded:[/green] Context={ctx_size}, GPU Layers={gpu_layers if gpu_layers != -1 else 'All'}"
                )
        except Exception as e:
            self.console.print(f"[bold red]Error loading GGUF model {self.model_path}:[/bold red] {e}")
            self.console.print("Ensure model path is correct and `llama-cpp-python` is installed with appropriate hardware acceleration (e.g., BLAS, CUDA). Check library docs.")
            raise

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, **kwargs: Any) -> str:
        """Query the loaded GGUF model.

        Args:
            messages: List of Message objects.
            plaintext_output: If True, return raw text.
            stream: If True, stream the output token by token.
            **kwargs: Additional arguments (ignored).

        Returns:
            The model's response as a string.
        """
        if not self.model:
            error_msg = "Error: Llama.cpp model not properly initialized."
            self.console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg

        api_messages = [msg.to_api_format() for msg in messages]

        generation_params = {
            "messages": api_messages,
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            # Add other parameters like stop sequences if needed
            "stream": stream and not self.config.NO_STREAM,
        }

        if self.config.VERBOSE:
             # Log parameters (messages can be very large, log separately if needed)
             log_params = {k: v for k, v in generation_params.items() if k != 'messages'}
             logger.debug(f"Llama.cpp Request Parameters: {log_params}")

             self.console.print(Rule("Querying Llama.cpp Model", style="yellow"))
             self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={generation_params['max_tokens']}, temp={generation_params['temperature']}, top_p={generation_params['top_p']}, stream={generation_params['stream']}[/italic]")

             # Print the full request payload using Rich JSON
             self.console.print(Rule("Request Payload", style="dim blue"))
             # Use json.dumps to convert the dict (including potentially non-serializable items if any were present)
             # to a string first, then parse back with JSON for Rich rendering.
             # This handles potential edge cases better than directly passing the dict.
             try:
                 payload_str = json.dumps(generation_params, indent=2) # Pretty print for structure
                 self.console.print(JSON(payload_str))
             except TypeError as e:
                 logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                 self.console.print(f"[red]Error printing payload:[/red] {e}")
                 # Fallback to basic print if JSON fails
                 import pprint
                 self.console.print(pprint.pformat(generation_params))

             self.console.print(Rule(style="yellow"))

        response_text_final = ""
        panel_title = f"[bold yellow]Llama.cpp ({os.path.basename(self.model_path)})[/bold yellow]"
        try:
            if generation_params["stream"]:
                raw_stream = self.model.create_chat_completion(**generation_params)
                response_text_final = self._handle_streaming_output(
                    stream_iterator=self._iterate_llama_cpp_chunks(raw_stream),
                    plaintext_output=plaintext_output,
                    panel_title=panel_title,
                    panel_border_style="yellow",
                )
            else:
                # Non-streaming generation
                start_time = time.time()
                completion = self.model.create_chat_completion(**generation_params)
                end_time = time.time()

                if completion and 'choices' in completion and completion['choices']:
                    response_text_final = completion['choices'][0].get('message', {}).get('content', '').strip()

                    if self.config.VERBOSE:
                        usage = completion.get('usage', {})
                        prompt_tokens = usage.get('prompt_tokens', 'N/A')
                        completion_tokens = usage.get('completion_tokens', 'N/A')
                        elapsed_time = end_time - start_time
                        if isinstance(completion_tokens, int) and elapsed_time > 0:
                            tokens_per_sec = completion_tokens / elapsed_time
                            self.console.print(f"[dim]Generated {completion_tokens} tokens (prompt: {prompt_tokens}) in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)[/dim]")
                        else:
                            self.console.print(f"[dim]Generated {completion_tokens} tokens (prompt: {prompt_tokens})[/dim]")

                    if not plaintext_output:
                        # Split response at the first \n\n
                        parts = response_text_final.split("\n\n", 1)
                        first_part = parts[0]
                        second_part = parts[1] if len(parts) > 1 else None
                        # Use appropriate style from panel_title var
                        self._print_assistant_message(first_part, second_part=second_part, panel_title=panel_title, panel_border_style="yellow")
                else:
                    self.console.print("[bold red]Error: No response generated by llama.cpp model.[/bold red]")
                    response_text_final = "Error: Failed to get response from llama.cpp model."

        except Exception as e:
            self.console.print(f"[bold red]Error during llama.cpp generation:[/bold red] {e}")
            logger.exception("Error during llama.cpp generation")
            response_text_final = f"Error: An exception occurred during generation: {e}"
        finally:
            if self.config.VERBOSE: self.console.print(Rule(style="yellow"))

        return response_text_final.strip()

    def _iterate_llama_cpp_chunks(self, stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """Extracts content delta from llama.cpp stream chunks."""
        try:
            for chunk in stream:
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content')
                if content:
                    yield content
        except Exception as e:
            self.console.print(f"\n[bold red]Error processing llama.cpp stream:[/bold red] {e}")
            logger.exception("Error processing llama.cpp stream")
            yield f"\nERROR: {e}"

    def get_styling(self) -> tuple[str | None, str]:
        """Return Llama.cpp specific styling."""
        panel_border_style = "yellow"
        panel_title = f"[bold yellow]Llama.cpp ({os.path.basename(self.model_path)})[/bold yellow]"
        return panel_title, panel_border_style

    # Override format_message to use the specific style for non-streaming
    def format_message(self, role: str, content: str):
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
             panel_title = f"[bold yellow]Llama.cpp ({os.path.basename(self.model_path)})[/bold yellow]"
             # Use base class method but provide specific title/style
             self._print_assistant_message(content, panel_title=panel_title, panel_border_style="yellow")

    def _print_user_message(self, content):
        """Prints the user message with rich formatting."""
        self.console.print()
        self.console.print(Markdown(f"**User:** {content}"))

    # Note: The base class _handle_streaming_output creates its own panel.
    # If non-streaming, we need a way to print the final message in a panel.
    # DEPRECATED: Base class handles splitting now.
    # def _print_assistant_message(self, content, panel_title, panel_border_style):
    #     """Custom implementation for Llama.cpp assistant message (yellow panel)."""
    #     assistant_panel = Panel(
    #         Markdown(content.strip()),
    #         title=panel_title,
    #         border_style=panel_border_style,
    #         padding=(1, 2),
    #     )
    #     self.console.print(assistant_panel)
    #     self.console.print() 
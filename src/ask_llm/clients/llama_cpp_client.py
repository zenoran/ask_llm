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
        n_gpu_layers = self.config.LLAMA_CPP_N_GPU_LAYERS # -1 means load all possible layers to GPU
        n_ctx = self.config.LLAMA_CPP_N_CTX # Context size
        n_batch = getattr(self.config, 'LLAMA_CPP_N_BATCH', 2048) # Batch size for prompt processing
        flash_attn = getattr(self.config, 'LLAMA_CPP_FLASH_ATTN', True) # Flash attention for memory efficiency

        model_load_params = {
            "model_path": self.model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,  # Higher batch size = faster prompt processing
            "flash_attn": flash_attn,  # Flash attention reduces VRAM usage for long contexts
            # Let llama.cpp auto-detect chat format from GGUF metadata
            # This is better than hardcoding chatml which doesn't work for all models
            # (e.g., Mistral-based models like Cydonia use mistral-instruct format)
            "chat_format": None,
            "verbose": False, # Disable llama-cpp's library-level verbose logging
        }
        final_chat_format = model_load_params["chat_format"]
        if self.config.VERBOSE:
            log_params = {k: v for k, v in model_load_params.items() if k != 'model_path'}
            logger.debug(f"Final chat_format passed to Llama(): {final_chat_format!r}")
            logger.debug(f"llama.cpp model load parameters (excluding path): {log_params}")

        try:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
                    if Llama is None: # Should have been caught in __init__, but double-check
                        raise ImportError("Llama class is not available from llama_cpp.")
                    self.model = Llama(**model_load_params)

            if self.config.VERBOSE and self.model:
                ctx_size = getattr(self.model, 'n_ctx', 'N/A')
                gpu_layers = model_load_params['n_gpu_layers']
                self.console.print(
                    f"[green]Model loaded:[/green] Context={ctx_size}, Batch={n_batch}, FlashAttn={flash_attn}, GPU Layers={gpu_layers if gpu_layers != -1 else 'All'}"
                )
        except Exception as e:
            self.console.print(f"[bold red]Error loading GGUF model {self.model_path}:[/bold red] {e}")
            self.console.print("Ensure model path is correct and `llama-cpp-python` is installed with appropriate hardware acceleration (e.g., BLAS, CUDA). Check library docs.")
            raise

    def stream_raw(self, messages: List[Message], stop: list[str] | str | None = None, **kwargs: Any) -> Iterator[str]:
        """
        Stream raw text chunks from llama.cpp without console formatting.
        
        Used by the API service for SSE streaming.
        """
        if not self.model:
            raise RuntimeError("Llama.cpp model not properly initialized.")
        
        api_messages = [msg.to_api_format() for msg in messages]
        generation_params = {
            "messages": api_messages,
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": True,
        }
        if stop:
            generation_params["stop"] = stop
        
        raw_stream = self.model.create_chat_completion(**generation_params)
        yield from self._iterate_llama_cpp_chunks(raw_stream)

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, stop: list[str] | str | None = None, **kwargs: Any) -> str:
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
            "stream": stream and not self.config.NO_STREAM,
        }
        if stop:
            generation_params["stop"] = stop

        if self.config.VERBOSE:
             log_params = {k: v for k, v in generation_params.items() if k != 'messages'}
             logger.debug(f"Llama.cpp Request Parameters: {log_params}")

             self.console.print(Rule("Querying Llama.cpp Model", style="yellow"))
             self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={generation_params['max_tokens']}, temp={generation_params['temperature']}, top_p={generation_params['top_p']}, stream={generation_params['stream']}[/italic]")
             self.console.print(Rule("Request Payload", style="dim blue"))
             try:
                 payload_str = json.dumps(generation_params, indent=2) # Pretty print for structure
                 self.console.print(JSON(payload_str))
             except TypeError as e:
                 logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                 self.console.print(f"[red]Error printing payload:[/red] {e}")
                 import pprint
                 self.console.print(pprint.pformat(generation_params))

             self.console.print(Rule(style="yellow"))

        response_text_final = ""
        # Let base class handle panel title using bot_name
        panel_title, panel_border_style = self.get_styling()
        try:
            if generation_params["stream"]:
                raw_stream = self.model.create_chat_completion(**generation_params)
                response_text_final = self._handle_streaming_output(
                    stream_iterator=self._iterate_llama_cpp_chunks(raw_stream),
                    plaintext_output=plaintext_output,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style,
                )
            else:
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
                        parts = response_text_final.split("\n\n", 1)
                        first_part = parts[0]
                        second_part = parts[1] if len(parts) > 1 else None
                        self._print_assistant_message(first_part, second_part=second_part, panel_title=panel_title, panel_border_style=panel_border_style)
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
        # Return None for title to let base class use bot_name, only specify border style
        return None, "yellow"

    def unload(self) -> None:
        """Unload the GGUF model and free GPU/CPU memory."""
        if self.model is None:
            return
        
        logger.info(f"Unloading GGUF model: {self.model_path}")
        
        try:
            # Delete the model instance
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            if self.config.VERBOSE:
                self.console.print("[green]Model unloaded and memory freed[/green]")
                
        except Exception as e:
            logger.warning(f"Error during model unload: {e}") 

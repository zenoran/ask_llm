import torch
import time
import logging
import json
from threading import Thread
from typing import List, Iterator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from rich.panel import Panel
from rich.rule import Rule
from rich.json import JSON

from ..clients.base import LLMClient
from ..utils.config import Config
from ..models.message import Message

try:
    import bitsandbytes
    _bitsandbytes_available = True
except ImportError:
    bitsandbytes = None
    _bitsandbytes_available = False

# Set transformers log level higher to avoid excessive noise
logging.getLogger("transformers").setLevel(logging.WARNING)


class HuggingFaceClient(LLMClient):
    """Client for running Hugging Face transformer models locally."""

    def __init__(self, model_id: str, config: Config):
        if not _bitsandbytes_available:
             raise ImportError("bitsandbytes library not found, which is required for quantization. Install with: pip install bitsandbytes accelerate")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. HuggingFaceClient requires a GPU.")

        super().__init__(model_id, config)
        self.model_id = model_id
        self.tokenizer: PreTrainedTokenizer | None = None
        self.model: PreTrainedModel | None = None
        self._load_model()

    def _configure_quantization(self) -> BitsAndBytesConfig | None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.config.VERBOSE: self.console.print(f"Using compute dtype: {compute_dtype}")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _load_model(self):
        self.console.print(f"Loading model: [bold cyan]{self.model_id}[/bold cyan]... (This may take a while)")
        torch.cuda.empty_cache()

        quantization_config = self._configure_quantization()
        compute_dtype = getattr(quantization_config, 'bnb_4bit_compute_dtype', torch.float16)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.config.VERBOSE: self.console.print("[dim]Tokenizer `pad_token` set to `eos_token`.[/dim]")

            model_args = {
                "quantization_config": quantization_config,
                "torch_dtype": compute_dtype,
                "attn_implementation": "sdpa", # Use SDPA if available
                # "device_map": "auto", # Consider device_map for multi-GPU
            }

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_args)
            model.to("cuda")

            # Optional: Attempt model compilation (can improve speed but might be unstable)
            # Consider making this configurable via models.yaml
            # if self._should_compile_model():
            #     model = self._compile_model(model)

            self.model = model
            if self.config.VERBOSE and hasattr(model.config, "attn_implementation"):
                self.console.print(f"[dim]Using attention implementation: {model.config.attn_implementation}[/dim]")
            self.console.print("[green]Model loaded successfully.[/green]")

        except Exception as e:
            self.console.print(f"[bold red]Error loading model {self.model_id}:[/bold red] {e}")
            self.console.print("Ensure sufficient VRAM/RAM and required libraries are installed.")
            raise

    def _should_compile_model(self) -> bool:
        # Simple check for known safe models (could be expanded)
        safe_models = ["llama", "mistral", "phi", "gemma"]
        return any(name in self.model_id.lower() for name in safe_models) and "gemma3" not in self.model_id

    def _compile_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if not self.tokenizer:
             return model # Should not happen if called after tokenizer is loaded
        try:
            self.console.print("[yellow]Attempting model compilation...[/yellow]")
            compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            # Test compilation with a small input
            test_input = self.tokenizer("Test", return_tensors="pt").to(model.device)
            _ = compiled_model.generate(**test_input, max_new_tokens=1)
            self.console.print("[green]Model compilation successful![/green]")
            return compiled_model
        except Exception as e:
            self.console.print(f"[yellow]Compilation skipped:[/yellow] {str(e)}")
            return model # Return original model on failure

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, **kwargs) -> str:
        """Query the local Hugging Face model.

        Args:
            messages: List of Message objects.
            plaintext_output: If True, return raw text.
            stream: If True, stream the output token by token.
            **kwargs: Additional arguments (ignored).

        Returns:
            The model's response as a string.
        """
        if not self.tokenizer or not self.model:
            return "Error: Model or Tokenizer not properly initialized."

        formatted_prompt = self._prepare_prompt(messages)
        if formatted_prompt is None:
            return "Error: Could not format prompt."

        # Prepare inputs AFTER potential verbose log of formatted_prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[-1]

        generation_kwargs = {
            # "input_ids": inputs.input_ids, # Add later if logging
            # "attention_mask": inputs.attention_mask, # Add later if logging
            "max_new_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id, # Use tokenizer's EOS
            "use_cache": True,
        }

        # Complete payload for logging and execution
        full_payload_for_logging = {
            "model_id": self.model_id,
            "formatted_prompt": formatted_prompt,
            # Convert tensors to lists for JSON serialization
            "input_ids": inputs.input_ids.tolist(),
            "attention_mask": inputs.attention_mask.tolist(),
            **generation_kwargs
        }

        # Verbose output section
        if self.config.VERBOSE:
            self.console.print(Rule("Querying HuggingFace Model", style="cyan"))
            # Print key params for quick view
            self.console.print(f"[dim]Params:[/dim] [italic]max_new_tokens={generation_kwargs['max_new_tokens']}, temp={generation_kwargs['temperature']}, top_p={generation_kwargs['top_p']}[/italic]")

            # Print full payload as JSON
            self.console.print(Rule("Request Payload / Generation Kwargs", style="dim blue"))
            try:
                # Serialize payload, handling potential non-serializable items if any
                # (Tensors already converted to lists)
                payload_str = json.dumps(full_payload_for_logging, indent=2)
                self.console.print(JSON(payload_str))
            except TypeError as e:
                logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                self.console.print(f"[red]Error printing payload:[/red] {e}")
                import pprint
                # Print the dictionary directly if JSON fails
                self.console.print(pprint.pformat(full_payload_for_logging))
            self.console.print(Rule(style="cyan"))

        # Add inputs back to generation_kwargs for actual execution
        generation_kwargs["input_ids"] = inputs.input_ids
        generation_kwargs["attention_mask"] = inputs.attention_mask

        # if self.config.VERBOSE:
        #     self.console.print(f"[dim]Generation Params:[/dim] [italic]max_new_tokens={generation_kwargs['max_new_tokens']}, temp={generation_kwargs['temperature']}, top_p={generation_kwargs['top_p']}[/italic]")

        stream_flag = stream and not self.config.NO_STREAM
        response_text_final = ""
        try:
            if stream_flag:
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                generation_kwargs["streamer"] = streamer
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                response_text_final = self._handle_streaming_output(
                    stream_iterator=streamer,
                    plaintext_output=plaintext_output,
                    panel_title=f"[bold cyan]{self.model_id}[/bold cyan]",
                    panel_border_style="cyan",
                )
                thread.join() # Ensure thread finishes
            else:
                # Non-streaming generation
                start_time = time.time()
                with torch.inference_mode():
                    generation = self.model.generate(**generation_kwargs)
                end_time = time.time()

                output_ids = generation[0][input_len:]
                response_text_final = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                if self.config.VERBOSE:
                    elapsed_time = end_time - start_time
                    num_tokens = output_ids.numel()
                    if elapsed_time > 0:
                        tokens_per_sec = num_tokens / elapsed_time
                        self.console.print(f"[dim]Generated {num_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)[/dim]")
                    else:
                         self.console.print(f"[dim]Generated {num_tokens} tokens[/dim]")

                if not plaintext_output:
                     # Split response at the first \n\n
                     parts = response_text_final.split("\n\n", 1)
                     first_part = parts[0]
                     second_part = parts[1] if len(parts) > 1 else None
                     # Use appropriate style
                     panel_title = f"[bold cyan]{self.model_id}[/bold cyan]"
                     self._print_assistant_message(first_part, second_part=second_part, panel_title=panel_title, panel_border_style="cyan")

        except Exception as e:
             self.console.print(f"[bold red]Error during HuggingFace generation:[/bold red] {e}")
             logging.exception("Error during HuggingFace generation")
             response_text_final = f"Error: An exception occurred - {e}"
        finally:
             torch.cuda.empty_cache()
             if self.config.VERBOSE: self.console.print(Rule(style="cyan"))

        return response_text_final.strip()

    def _prepare_prompt(self, messages: List[Message]) -> str | None:
        """Applies the chat template or basic formatting to the messages."""
        if not self.tokenizer:
            return None

        chat_history = [msg.to_api_format() for msg in messages]

        try:
            # Prefer the tokenizer's chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True
                )
                if self.config.VERBOSE: self.console.print("[dim]Using tokenizer chat template.[/dim]")
                return formatted_prompt
            else:
                # Basic fallback formatting
                if self.config.VERBOSE: self.console.print("[dim]Tokenizer lacks chat template, using basic format.[/dim]")
                formatted_prompt = self.config.SYSTEM_MESSAGE + "\n\n"
                for msg in chat_history:
                    formatted_prompt += f"**{msg['role'].capitalize()}**: {msg['content']}\n\n"
                formatted_prompt += "**Assistant**:\n"
                return formatted_prompt
        except Exception as e:
            self.console.print(f"[bold red]Error formatting prompt:[/bold red] {e}")
            return None

    def get_styling(self) -> tuple[str | None, str]:
        """Return HuggingFace specific styling."""
        panel_border_style = "cyan"
        panel_title = f"[bold {panel_border_style}]{self.model_id}[/bold {panel_border_style}]"
        return panel_title, panel_border_style

    # Override format_message to use the specific HF style for non-streaming
    def format_message(self, role: str, content: str):
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
             # Use base class method but provide HF specific title/style
             self._print_assistant_message(content, panel_title=f"[bold cyan]{self.model_id}[/bold cyan]", panel_border_style="cyan")

    # Override to use cyan panel
    def _print_assistant_message(self, content: str, panel_title: str | None = None, panel_border_style: str = "cyan"):
        super()._print_assistant_message(content, panel_title=panel_title or f"[bold cyan]HF ({self.model_id})[/bold cyan]", panel_border_style=panel_border_style)

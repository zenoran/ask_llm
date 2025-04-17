import torch
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import config

# Attempt to import bitsandbytes - will be checked later
try:
    import bitsandbytes
except ImportError:
    bitsandbytes = None  # Set to None if not found

# Set transformers log level to WARNING
import logging

logging.getLogger("transformers").setLevel(logging.WARNING)


class HuggingFaceClient(LLMClient):
    """Client for running Hugging Face transformer models locally."""

    def __init__(self, model_id):
        super().__init__(model_id)
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        self.console.print(
            f"Loading model: [bold cyan]{self.model_id}[/bold cyan]... This may take a while."
        )

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.console.print(
                "[bold red]Error: CUDA is not available. Cannot load model on GPU.[/bold red]"
            )
            raise RuntimeError("CUDA not available")

        # Set memory efficient parameters for RTX 4090
        torch.cuda.empty_cache()

        # Quieter RTX 4090 optimizations
        if torch.cuda.get_device_properties(0).name.find("4090") >= 0:
            # Small amount of VRAM reserved to prevent fragmentation
            torch.cuda.set_per_process_memory_fraction(
                0.85
            )  # Lower memory fraction for less stress

        # Configure quantization, raising error if bitsandbytes is missing
        quantization_config = None
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        # Check if bitsandbytes was imported successfully
        if bitsandbytes is None:
            self.console.print(
                "[bold red]Error:[/bold red] bitsandbytes not found, which is required for quantization."
            )
            self.console.print("Install with: pip install bitsandbytes accelerate")
            raise RuntimeError("bitsandbytes library not found")
        else:
            self.console.print(f"Using compute dtype: {compute_dtype}")
            # bitsandbytes is available, configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Ensure pad token is set if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.console.print(
                    "Tokenizer `pad_token` was missing, set to `eos_token`."
                )

            # Optimize model loading for RTX 4090
            model_args = {
                "torch_dtype": compute_dtype,
            }

            # Add quantization_config if it was successfully created
            if quantization_config:
                model_args["quantization_config"] = quantization_config
            else:
                # This case should not be reached due to the check above, but good practice
                self.console.print(
                    "[yellow]Warning:[/yellow] Quantization config not set, loading model without quantization."
                )

            # Load model with optimized parameters
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                attn_implementation="sdpa",  # Scaled Dot Product Attention for faster generation
                **model_args,
            )

            model.to("cuda")

            # RTX 4090-specific: Try using torch.compile with a safe mode
            if torch.cuda.get_device_properties(0).name.find("4090") >= 0 and hasattr(
                torch, "compile"
            ):
                try:
                    # Only compile for known models that work well with compilation
                    safe_models = ["llama", "mistral", "phi", "gemma"]
                    if any(name in self.model_id.lower() for name in safe_models):
                        compiled_model = torch.compile(
                            model,
                            mode="default",  # Less aggressive mode that causes less coil whine
                            fullgraph=False,  # Partial compilation is safer
                        )
                        # Test the compiled model with a small input to verify it works
                        test_input = self.tokenizer(
                            "Test compilation", return_tensors="pt"
                        ).to(model.device)
                        _ = compiled_model.generate(**test_input, max_new_tokens=1)
                        model = compiled_model
                        self.console.print(
                            "[green]Model compilation successful![/green]"
                        )
                except Exception as e:
                    self.console.print(
                        f"[yellow]Compilation skipped: {str(e)}[/yellow]"
                    )

            # Store the model
            self.model = model

            # Try enabling efficient attention if available for RTX 4090
            if hasattr(model.config, "attn_implementation"):
                self.console.print(
                    f"Using attention implementation: {model.config.attn_implementation}"
                )

        except Exception as e:
            self.console.print(
                f"[bold red]Error loading model {self.model_id}:[/bold red] {e}"
            )
            self.console.print(
                "Please ensure you have enough VRAM/RAM and required libraries installed."
            )
            raise

    def query(self, messages, plaintext_output: bool = False, stream: bool = True):
        """Query the local Hugging Face model with optional streaming output.
        
        Args:
            messages: List of message dictionaries (including the latest user prompt).
            plaintext_output: If True, return raw text. Otherwise, format output.
            stream: If True, stream the output token by token.
        """
        if not self.tokenizer or not self.model:
            return "Error: Model or Tokenizer not properly initialized."

        # Prepare chat input using the full message list provided
        try:
            # Clear any existing state/cache that might cause repeated outputs
            if hasattr(self.model, "clear_cache"):
                self.model.clear_cache()

            # Use the messages list directly, assuming it includes the latest user prompt
            chat_history = [msg.to_api_format() if hasattr(msg, 'to_api_format') else msg for msg in messages]
            # chat_history.append({"role": "user", "content": prompt}) # REMOVED - Prompt is already in messages

            if hasattr(self.tokenizer, "apply_chat_template"):
                # Ensure we're not reusing a cached version
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True
                )
                if config.VERBOSE:
                    self.console.print("[dim]Using tokenizer chat template.[/dim]")
            else:
                # Fallback formatting (less likely needed if using models with templates)
                if config.VERBOSE:
                    self.console.print(
                        "[dim]Tokenizer has no chat template, using basic formatting.[/dim]"
                    )
                formatted_prompt = config.SYSTEM_MESSAGE + "\n\n"
                for msg in chat_history:
                    formatted_prompt += (
                        f"{msg['role'].capitalize()}: {msg['content']}\n\n"
                    )
        except Exception as e:
            self.console.print(f"[bold red]Error formatting prompt:[/bold red] {e}")
            return "Error: Could not format prompt for the model."

        if config.VERBOSE:
            self.console.print("\n[bold blue]Formatted Prompt:[/bold blue]")
            self.console.print(f"[dim]{formatted_prompt}[/dim]")
            self.console.print(Rule(style="#777777"))

        # Tokenize the input prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        input_len = inputs.input_ids.shape[-1]

        # Handle stop tokens safely
        try:
            stop_token = "<|im_end|>"
            stop_token_id = self.tokenizer.convert_tokens_to_ids(stop_token)
            if stop_token_id == self.tokenizer.unk_token_id:
                stop_token_id = None
        except Exception:
            stop_token_id = None

        # Base generation parameters
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": config.MAX_TOKENS or 1024,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if stop_token_id is not None:
            generation_kwargs["eos_token_id"] = stop_token_id


        if stream:
            # Setup for streaming
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs["streamer"] = streamer

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream output using the base class handler
            response_text_final = ""
            try:
                response_text_final = self._handle_streaming_output(
                    stream_iterator=streamer,
                    plaintext_output=plaintext_output,
                    panel_title=f"[bold cyan]{self.model_id}[/bold cyan]",
                    panel_border_style="cyan",
                    first_para_panel=False,  # HF client uses a single panel for the whole response
                )
            finally:
                # Ensure the generation thread finishes
                thread.join()
                # Clean up resources if needed
                torch.cuda.empty_cache()

            return response_text_final

        else:
            # Non-streaming generation
            start_time = time.time()
            with torch.inference_mode():
                generation = self.model.generate(**generation_kwargs)
            end_time = time.time()
            
            output_ids = generation[0][input_len:]
            response_text_final = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            # Calculate and print tokens/second
            elapsed_time = end_time - start_time
            num_tokens = output_ids.numel()
            if elapsed_time > 0:
                tokens_per_sec = num_tokens / elapsed_time
                self.console.print(f"[dim]Generated {num_tokens} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)[/dim]")
            else:
                 self.console.print(f"[dim]Generated {num_tokens} tokens (elapsed time too short to calculate rate)[/dim]")


            # Print formatted output if not plaintext
            if not plaintext_output:
                self._print_assistant_message(response_text_final)

            # Clean up resources
            torch.cuda.empty_cache()

            return response_text_final

    def format_message(self, role, content):
        """Format a user or assistant message for display."""
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
            self._print_assistant_message(content)

    def _print_user_message(self, content):
        """Prints the user message with rich formatting."""
        self.console.print()
        self.console.print(Markdown(f"**User:** {content}"))

    def _print_assistant_message(self, content):
        """Custom implementation for HF assistant message (cyan panel)."""
        # Basic panel, as streaming handles most formatting
        assistant_panel = Panel(
            Markdown(content.strip()),
            title=f"[bold cyan]{self.model_id}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(assistant_panel)
        self.console.print()

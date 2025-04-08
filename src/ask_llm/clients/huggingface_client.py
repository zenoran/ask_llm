import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.rule import Rule
from rich.live import Live
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import config

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
        self.console.print(f"Loading model: [bold cyan]{self.model_id}[/bold cyan]... This may take a while.")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.console.print("[bold red]Error: CUDA is not available. Cannot load model on GPU.[/bold red]")
            raise RuntimeError("CUDA not available")
            
        self.console.print(f"CUDA available: [green]Yes[/green], Device Count: [bold]{torch.cuda.device_count()}[/bold]")
        
        # Set memory efficient parameters for RTX 4090
        torch.cuda.empty_cache()
        
        # Quieter RTX 4090 optimizations
        if torch.cuda.get_device_properties(0).name.find("4090") >= 0:
            # Small amount of VRAM reserved to prevent fragmentation
            torch.cuda.set_per_process_memory_fraction(0.85)  # Lower memory fraction for less stress
            # Removed deprecated nvFuser call that generates warnings
            self.console.print("[green]Applied RTX 4090 optimizations.[/green]")
            
        # Configure quantization for memory efficiency
        quantization_config = None
        try:
            import bitsandbytes
            # RTX 4090 has tensor cores that work well with bfloat16/float16
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.console.print(f"Using compute dtype: {compute_dtype}")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.console.print("Using 4-bit quantization (bitsandbytes).")
        except ImportError:
            self.console.print("[yellow]Warning:[/yellow] bitsandbytes not found. Model will not be quantized (requires more memory/VRAM).")
            self.console.print("Install with: pip install bitsandbytes accelerate")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Ensure pad token is set if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.console.print("Tokenizer `pad_token` was missing, set to `eos_token`.")

            # Optimize model loading for RTX 4090
            model_args = {
                "torch_dtype": compute_dtype if 'compute_dtype' in locals() else torch.float16,
            }
            
            if quantization_config:
                model_args["quantization_config"] = quantization_config

            # Load model with optimized parameters
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                attn_implementation="sdpa",  # Scaled Dot Product Attention for faster generation
                **model_args
            )
            
            # Move model to GPU
            self.console.print(f"Model loaded, moving to cuda...")
            model.to('cuda')
            self.console.print(f"Model on device: {model.device}")
            
            # RTX 4090-specific: Try using torch.compile with a safe mode
            if torch.cuda.get_device_properties(0).name.find("4090") >= 0 and hasattr(torch, "compile"):
                try:
                    # Only compile for known models that work well with compilation
                    safe_models = ["llama", "mistral", "phi", "gemma"]
                    if any(name in self.model_id.lower() for name in safe_models):
                        self.console.print("[yellow]Attempting quiet RTX 4090 optimized compilation...[/yellow]")
                        compiled_model = torch.compile(
                            model, 
                            mode="default",  # Less aggressive mode that causes less coil whine
                            fullgraph=False,  # Partial compilation is safer
                        )
                        # Test the compiled model with a small input to verify it works
                        test_input = self.tokenizer("Test compilation", return_tensors="pt").to(model.device)
                        _ = compiled_model.generate(**test_input, max_new_tokens=1)
                        model = compiled_model
                        self.console.print("[green]Model compilation successful![/green]")
                except Exception as e:
                    self.console.print(f"[yellow]Compilation skipped: {str(e)}[/yellow]")
            
            # Store the model
            self.model = model
            
            # Try enabling efficient attention if available for RTX 4090
            if hasattr(model.config, "attn_implementation"):
                self.console.print(f"Using attention implementation: {model.config.attn_implementation}")
            
            self.console.print(f"Model [bold cyan]{self.model_id}[/bold cyan] setup complete on {self.model.device}.")

        except Exception as e:
            self.console.print(f"[bold red]Error loading model {self.model_id}:[/bold red] {e}")
            self.console.print("Please ensure you have enough VRAM/RAM and required libraries installed.")
            raise

    def query(self, messages, prompt):
        """Query the local Hugging Face model with streaming output."""
        if not self.tokenizer or not self.model:
            return "Error: Model or Tokenizer not properly initialized."

        # Prepare chat input
        try:
            # Clear any existing state/cache that might cause repeated outputs
            if hasattr(self.model, "clear_cache"):
                self.model.clear_cache()
                
            # Create a fresh chat history for each query
            chat_history = [msg.to_api_format() for msg in messages]
            chat_history.append({"role": "user", "content": prompt})
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Ensure we're not reusing a cached version
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat_history, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                if config.VERBOSE: 
                    self.console.print("[dim]Using tokenizer chat template.[/dim]")
            else:
                if config.VERBOSE: 
                    self.console.print("[dim]Tokenizer has no chat template, using basic formatting.[/dim]")
                formatted_prompt = config.SYSTEM_MESSAGE + "\n\n"
                for msg in chat_history:
                    formatted_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
        except Exception as e:
            self.console.print(f"[bold red]Error formatting prompt:[/bold red] {e}")
            return "Error: Could not format prompt for the model."
            
        if config.VERBOSE:
            self.console.print("\n[bold blue]Formatted Prompt:[/bold blue]")
            self.console.print(f"[dim]{formatted_prompt}[/dim]")
            self.console.print(Rule(style="#777777"))

        # Tokenize the input prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Removed dummy tensor allocation which can cause coil whine
        
        # Create streamer and setup generation
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Handle stop tokens safely
        try:
            stop_token = "<|im_end|>"
            stop_token_id = self.tokenizer.convert_tokens_to_ids(stop_token)
            if stop_token_id == self.tokenizer.unk_token_id:
                stop_token_id = None
        except:
            stop_token_id = None

        # Optimize generation parameters for speed on RTX 4090
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "streamer": streamer,
            "max_new_tokens": config.MAX_TOKENS or 1024,
            "do_sample": True,
            "temperature": config.TEMPERATURE or 0.7,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,  # Enable KV caching for faster generation
            "repetition_penalty": 1.05,  # Reduced penalty to prevent power spikes
        }
        
        # Ensure we're not getting stuck in repetition loops
        if "do_sample" in generation_kwargs and generation_kwargs["do_sample"]:
            # Ensure temperature is high enough to avoid deterministic outputs
            min_temp = 0.5
            if "temperature" in generation_kwargs and generation_kwargs["temperature"] < min_temp:
                generation_kwargs["temperature"] = min_temp
                self.console.print(f"[yellow]Set minimum temperature to {min_temp} to avoid repetition.[/yellow]")

        # Removed beam search which can cause significant coil whine
        
        # Add stop token if available
        if stop_token_id is not None:
            generation_kwargs["eos_token_id"] = stop_token_id

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output using Rich Live display
        generated_text = ""
        panel = Panel(
            "", 
            title=f"[bold cyan]{self.model_id}[/bold cyan]", 
            border_style="cyan",
            padding=(1,2)
        )
        response_text_final = ""

        try:
            with Live(panel, refresh_per_second=15, console=self.console, vertical_overflow="visible") as live:
                for new_text in streamer:
                    generated_text += new_text
                    # Update the content of the panel within the Live context
                    live.update(Panel(
                        Markdown(generated_text.strip()), 
                        title=f"[bold cyan]{self.model_id}[/bold cyan]", 
                        border_style="cyan",
                        padding=(1,2)
                    ))
            # Ensure the thread finishes
            thread.join()
            
            # Final processing after stream ends
            response_text_final = generated_text.strip()

            if config.VERBOSE:
                self.console.print("\n[bold blue]Final Generated Text:[/bold blue]")
                self.console.print(f"[dim]{response_text_final}[/dim]")
                self.console.print(Rule(style="#777777"))
                
        except Exception as e:
            self.console.print(f"[bold red]Error during streaming generation:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            response_text_final = f"Error during streaming: {e}"
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Streaming interrupted.[/bold yellow]")
            response_text_final = generated_text.strip()

        return response_text_final

    def format_response(self, response_text):
        """Format the model's response for display using Rich."""
        self._print_assistant_message(response_text)

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
        """Prints the assistant message using rich Panel."""
        self.console.print()
        assistant_panel = Panel(
            Markdown(content.strip()),
            title=f"[bold cyan]{self.model_id}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(Align(assistant_panel, align="right")) 
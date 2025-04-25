import importlib.util
import pathlib
from rich.console import Console
from .utils.config import Config, is_huggingface_available, is_llama_cpp_available
from .utils.history import HistoryManager, Message
from .clients import LLMClient
import logging
from .utils.prompts import SYSTEM_REFINE_PROMPT


console = Console()
logger = logging.getLogger(__name__)

class AskLLM:
    def __init__(self, resolved_model_alias: str, config: Config):
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        self.model_definition = self.config.defined_models.get("models", {}).get(resolved_model_alias)

        if not self.model_definition:
             raise ValueError(f"Could not find model definition for resolved alias: '{resolved_model_alias}'")

        self.client: LLMClient = self.initialize_client(config)
        self.history_manager = HistoryManager(client=self.client, config=self.config)
        self.load_history()

    def initialize_client(self, config: Config) -> LLMClient:
        model_type = self.model_definition.get("type")
        model_alias = self.resolved_model_alias

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Initializing client for model: {model_alias} (Type: {model_type})")

        if model_type == "gguf":
            # Check dependencies lazily
            if importlib.util.find_spec("huggingface_hub") is None:
                 console.print("[bold red]Error:[/bold red] `huggingface-hub` is required for GGUF models.")
                 raise ImportError("huggingface-hub not found for GGUF download.")
            if not is_llama_cpp_available(): # Check lazily
                console.print("[bold red]Error:[/bold red] `llama-cpp-python` is required for GGUF models.")
                raise ImportError("`llama-cpp-python` not installed or failed to import.")
            try:
                from .clients.llama_cpp_client import LlamaCppClient
                return self._initialize_llama_cpp_client(self.model_definition, config)
            except ImportError as e:
                 console.print(f"[bold red]Import Error for GGUF:[/bold red] {e}")
                 console.print("Ensure `llama-cpp-python` and `huggingface-hub` are installed.")
                 raise
            except Exception as e:
                console.print(f"[bold red]Error initializing Llama.cpp client:[/bold red] {e}")
                logger.exception(f"Error initializing Llama.cpp client: {e}")
                raise

        elif model_type == "huggingface":
             # Check dependencies lazily
             if not is_huggingface_available(): # Check lazily
                 console.print("[bold red]Error:[/bold red] Hugging Face dependencies are required for model type 'huggingface'. Ensure `transformers` and `torch` (or `tensorflow`/`jax`) are installed.")
                 raise ImportError("HuggingFace dependencies unavailable.")
             try:
                 from .clients.huggingface_client import HuggingFaceClient
                 model_id = self.model_definition.get("model_id")
                 if not model_id:
                     raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                 return HuggingFaceClient(model_id=model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing HuggingFace client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing HuggingFace client for {model_alias}: {e}")
                raise

        elif model_type == "ollama":
            try:
                from .clients.ollama_client import OllamaClient
                model_id = self.model_definition.get("model_id")
                if not model_id:
                    raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                if model_id not in self.config.available_ollama_models:
                     console.print(f"[yellow]Warning:[/yellow] Ollama model '{model_id}' (alias: '{model_alias}') not found on server {self.config.OLLAMA_URL}.")
                     console.print(f"  Attempting to use anyway, but may fail. Pull it with: `ollama pull {model_id}`")
                return OllamaClient(model=model_id, config=config)
            except Exception as e:
                console.print(f"[bold red]Error initializing Ollama client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing Ollama client for {model_alias}: {e}")
                raise

        elif model_type == "openai":
             try:
                 from .clients.openai_client import OpenAIClient
                 model_id = self.model_definition.get("model_id")
                 if not model_id:
                     raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                 return OpenAIClient(model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing OpenAI client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing OpenAI client for {model_alias}: {e}")
                raise
        else:
            raise ImportError(f"Unsupported model type '{model_type}' defined for alias '{model_alias}' in {self.config.MODELS_CONFIG_PATH}")

    def _initialize_llama_cpp_client(self, model_def: dict, config: Config):
        # Re-import necessary hf_hub functions if needed, as the check is now local
        # Though, the check in initialize_client should prevent reaching here if unavailable
        if importlib.util.find_spec("huggingface_hub"):
             from huggingface_hub import hf_hub_download
             from huggingface_hub.utils import HfHubHTTPError
        else:
             # This state should ideally not be reached due to the check in initialize_client
             raise ImportError("huggingface-hub is required but not found for GGUF download within _initialize_llama_cpp_client.")

        from .clients.llama_cpp_client import LlamaCppClient
        repo_id = model_def.get("repo_id")
        filename = model_def.get("filename")
        alias = self.resolved_model_alias

        if not repo_id or not filename:
            raise ImportError(f"GGUF model definition for alias '{alias}' is missing 'repo_id' or 'filename' in {config.MODELS_CONFIG_PATH}")

        cache_dir = pathlib.Path(config.MODEL_CACHE_DIR).expanduser()
        model_repo_cache_dir = cache_dir / repo_id
        local_model_path = model_repo_cache_dir / filename
        model_path_to_load = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loading GGUF model: {alias} (Source: {repo_id}/{filename})")
            logger.debug(f"Expected model cache path: {local_model_path}")

        if local_model_path.is_file():
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug("Found cached GGUF model file.")
            model_path_to_load = str(local_model_path)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug("GGUF model file not found in cache. Downloading...")
            model_repo_cache_dir.mkdir(parents=True, exist_ok=True)
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Downloading '{filename}' from repo '{repo_id}'...")
            try:
                downloaded_path_str = hf_hub_download(repo_id=repo_id,filename=filename,local_dir=str(model_repo_cache_dir),)
                if pathlib.Path(downloaded_path_str) != local_model_path:
                     console.print(f"[yellow]Warning:[/yellow] Download path {downloaded_path_str} differs from expected cache path {local_model_path}. Using downloaded path.")
                model_path_to_load = downloaded_path_str
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"GGUF download complete: {model_path_to_load}")
            except HfHubHTTPError as e:
                 console.print(f"[bold red]Error accessing or downloading from Hugging Face repository '{repo_id}':[/bold red] {e}")
                 console.print("Please ensure the repository ID and filename are correct and you have internet access.")
                 raise
            except Exception as e:
                console.print(f"[bold red]Error downloading file '{filename}':[/bold red] {e}")
                logger.exception(f"Error downloading file '{filename}': {e}")
                raise

        try:
            client = LlamaCppClient(model_path=model_path_to_load, config=config)
            return client
        except ImportError:
             console.print("[bold red]Error:[/bold red] Failed to initialize LlamaCppClient. Is `llama-cpp-python` installed correctly?")
             raise
        except Exception as e:
            console.print(f"[bold red]Error initializing LlamaCppClient with {model_path_to_load}:[/bold red] {e}")
            logger.exception(f"Error initializing LlamaCppClient with {model_path_to_load}: {e}")
            raise

    def load_history(self, since_minutes: int | None= None) -> list[dict]:
        if since_minutes is None:
            since_minutes = self.config.HISTORY_DURATION
        self.history_manager.load_history(since_minutes=since_minutes)
        return [msg.to_dict() for msg in self.history_manager.messages]

    def query(self, prompt, plaintext_output: bool = False, stream: bool = True):
        try:
            self.history_manager.add_message("user", prompt)
            complete_context_messages = self.history_manager.get_context_messages()

            query_kwargs = {"messages": complete_context_messages,"plaintext_output": plaintext_output,}

            if hasattr(self.client, 'SUPPORTS_STREAMING') and self.client.SUPPORTS_STREAMING and stream:
                 query_kwargs["stream"] = True
            elif stream:
                 logger.debug(f"Streaming requested but client {type(self.client).__name__} does not support it. Disabling streaming.")

            response = self.client.query(**query_kwargs)

            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response and not self.config.ALLOW_DUPLICATE_RESPONSE:
                console.print("[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]")
                response = self.client.query(**query_kwargs)

            self.history_manager.add_message("assistant", response)
            return response
        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""

            
    def refine_prompt(self, prompt, history=None):
        """
        Refine the user's prompt using context and history to determine intent.
        
        Args:
            prompt: The user's original prompt
            history: Optional list of previous message objects or dictionaries
            
        Returns:
            A refined prompt that can be fed back to the LLM
        """
        try:
            messages = []
            messages.append(Message(role="system", content=SYSTEM_REFINE_PROMPT))
            
            # Add history context if provided
            if history:
                history_text = []
                for msg in history:
                    # Handle both Message objects and dictionaries
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        # Message object
                        history_text.append(f"{msg.role}: {msg.content}")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # Dictionary with role and content keys
                        history_text.append(f"{msg['role']}: {msg['content']}")
                    else:
                        logger.warning(f"Skipping invalid message format in history: {msg}")
                        
                if history_text:
                    context_message = f"Previous conversation history:\n{'\n'.join(history_text)}\n\nUser's raw prompt: {prompt}"
                    messages.append(Message(role="user", content=context_message))
                else:
                    messages.append(Message(role="user", content=f"User's raw prompt: {prompt}"))
            else:
                messages.append(Message(role="user", content=f"User's raw prompt: {prompt}"))
            
            # Make the query with plaintext output and no streaming for prompt refinement
            query_kwargs = {"messages": messages, "plaintext_output": True, "stream": False}
            refined_prompt = self.client.query(**query_kwargs)
            
            if logger.isEnabledFor(logging.DEBUG):
                console.print(f"[bold green]Messages:[/bold green] {messages}")
                logger.info(f"Refined prompt: {refined_prompt}")
                
            return refined_prompt
        except Exception as e:
            console.print(f"[bold yellow]Error during prompt refinement:[/bold yellow] {e}")
            logger.warning(f"Error during prompt refinement: {e}")
            # Return original prompt on error
            return prompt 
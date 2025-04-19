import argparse
import pathlib
import traceback
from rich.console import Console
from .utils.config import Config, is_huggingface_available, is_llama_cpp_available
from .utils.history import HistoryManager
from .clients import OpenAIClient, OllamaClient, LLMClient
import logging

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    hf_hub_available = True
except ImportError:
    hf_hub_available = False
    hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(ImportError("huggingface-hub is not installed"))
    HfApi = type('DummyHfApi', (), {'list_repo_files': lambda **kwargs: (_ for _ in ()).throw(ImportError("huggingface-hub is not installed"))})
    HfHubHTTPError = type('DummyHfHubHTTPError', (Exception,), {})

try:
    from .clients import HuggingFaceClient
except ImportError:
    HuggingFaceClient = None
try:
    from .clients import LlamaCppClient
except ImportError:
    LlamaCppClient = None

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
            if not hf_hub_available:
                 console.print("[bold red]Error:[/bold red] `huggingface-hub` is required for GGUF models.")
                 raise ImportError("huggingface-hub not found for GGUF download.")
            try:
                if LlamaCppClient is None:
                    raise ImportError("`llama-cpp-python` not installed or failed to import.")
                return self._initialize_llama_cpp_client(self.model_definition, config)
            except ImportError as e:
                 console.print(f"[bold red]Import Error for GGUF:[/bold red] {e}")
                 console.print("Ensure `llama-cpp-python` and `huggingface-hub` are installed.")
                 raise
            except Exception as e:
                console.print(f"[bold red]Error initializing Llama.cpp client:[/bold red] {e}")
                if self.config.VERBOSE: traceback.print_exc()
                raise

        elif model_type == "huggingface":
             if HuggingFaceClient is None:
                 console.print("[bold red]Error:[/bold red] Hugging Face dependencies are required for model type 'huggingface'.")
                 raise ImportError("HuggingFaceClient unavailable.")
             try:
                 model_id = self.model_definition.get("model_id")
                 if not model_id:
                     raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                 return HuggingFaceClient(model_id=model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing HuggingFace client for {model_alias}:[/bold red] {e}")
                if self.config.VERBOSE: traceback.print_exc()
                raise

        elif model_type == "ollama":
            try:
                model_id = self.model_definition.get("model_id")
                if not model_id:
                    raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                if model_id not in self.config.available_ollama_models:
                     console.print(f"[yellow]Warning:[/yellow] Ollama model '{model_id}' (alias: '{model_alias}') not found on server {self.config.OLLAMA_URL}.")
                     console.print(f"  Attempting to use anyway, but may fail. Pull it with: `ollama pull {model_id}`")
                return OllamaClient(model=model_id, config=config)
            except Exception as e:
                console.print(f"[bold red]Error initializing Ollama client for {model_alias}:[/bold red] {e}")
                if self.config.VERBOSE: traceback.print_exc()
                raise

        elif model_type == "openai":
             try:
                model_id = self.model_definition.get("model_id")
                if not model_id:
                    raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                return OpenAIClient(model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing OpenAI client for {model_alias}:[/bold red] {e}")
                if self.config.VERBOSE: traceback.print_exc()
                raise
        else:
            raise ValueError(f"Unsupported model type '{model_type}' defined for alias '{model_alias}' in {self.config.MODELS_CONFIG_PATH}")

    def _initialize_llama_cpp_client(self, model_def: dict, config: Config):
        if LlamaCppClient is None:
            raise ImportError("`llama-cpp-python` not installed or failed to import.")

        repo_id = model_def.get("repo_id")
        filename = model_def.get("filename")
        alias = self.resolved_model_alias

        if not repo_id or not filename:
            raise ValueError(f"GGUF model definition for alias '{alias}' is missing 'repo_id' or 'filename' in {config.MODELS_CONFIG_PATH}")

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
                if config.VERBOSE: traceback.print_exc()
                raise

        try:
            client = LlamaCppClient(model_path=model_path_to_load, config=config)
            return client
        except ImportError:
             console.print("[bold red]Error:[/bold red] Failed to initialize LlamaCppClient. Is `llama-cpp-python` installed correctly?")
             raise
        except Exception as e:
            console.print(f"[bold red]Error initializing LlamaCppClient with {model_path_to_load}:[/bold red] {e}")
            if self.config.VERBOSE: traceback.print_exc()
            raise

    def load_history(self):
        self.history_manager.load_history()

    def query(self, prompt, plaintext_output: bool = False, stream: bool = True):
        try:
            self.history_manager.add_message("user", prompt)
            complete_context_messages = self.history_manager.get_context_messages()

            query_kwargs = {"messages": complete_context_messages,"plaintext_output": plaintext_output,}

            streaming_clients = []
            if HuggingFaceClient: streaming_clients.append(HuggingFaceClient)
            if LlamaCppClient: streaming_clients.append(LlamaCppClient)

            if streaming_clients and isinstance(self.client, tuple(streaming_clients)):
                query_kwargs["stream"] = stream

            response = self.client.query(**query_kwargs)

            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response and not self.config.allow_duplicate_response:
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
            if self.config.VERBOSE:
                 traceback.print_exc()
            self.history_manager.remove_last_message_if_partial("assistant")
            return "" 
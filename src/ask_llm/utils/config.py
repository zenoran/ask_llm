import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

# Default system message fallback - only used if bot config fails to load
DEFAULT_SYSTEM_MESSAGE = """You are Nova, a personal AI assistant running locally via ask_llm.
You have persistent memory across sessions. Keep responses concise and helpful."""

PROVIDER_OPENAI = "openai"
PROVIDER_OLLAMA = "ollama"
PROVIDER_GGUF = "gguf"
PROVIDER_HF = "huggingface"
PROVIDER_UNKNOWN = "Unknown" 

logger = logging.getLogger(__name__)
console = Console()

def is_huggingface_available() -> bool:
    """Checks if Hugging Face dependencies (torch, transformers, bitsandbytes) are available."""
    torch_spec = importlib.util.find_spec("torch")
    transformers_spec = importlib.util.find_spec("transformers")
    # bitsandbytes is often optional but useful for quantization with HF
    _ = importlib.util.find_spec("bitsandbytes") 
    # Adjust logic if bitsandbytes should be strictly required
    return all([torch_spec, transformers_spec]) # Or include bitsandbytes_spec if mandatory

def is_llama_cpp_available() -> bool:
    # Restore original or remove if too obvious
    return importlib.util.find_spec("llama_cpp") is not None

def get_default_models_yaml_path() -> Path:
    env_path = os.environ.get("ASK_LLM_MODELS_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    default_config_dir = Path(xdg_config_home) / "ask-llm"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    return default_config_dir / "models.yaml"

DEFAULT_MODELS_YAML = get_default_models_yaml_path()
DOTENV_PATH = DEFAULT_MODELS_YAML.parent / ".env"

class Config(BaseSettings):
    HISTORY_FILE: str = Field(default=os.path.expanduser("~/.cache/ask_llm/chat-history"))
    HISTORY_DURATION: int = Field(default=60 * 30)  # 30 minutes in seconds
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    MODEL_CACHE_DIR: str = Field(default=os.path.expanduser("~/.cache/ask_llm/models"), description="Directory to cache downloaded GGUF models")
    MODELS_CONFIG_PATH: str = Field(default=str(DEFAULT_MODELS_YAML), description="Path to the models YAML definition file")
    DEFAULT_MODEL_ALIAS: Optional[str] = Field(default=None, description="Alias from models.yaml to use if --model is not specified")
    DEFAULT_BOT: str = Field(default="nova", description="Default bot to use if --bot is not specified")
    DEFAULT_USER: str = Field(default="default", description="Default user profile to use if --user is not specified")

    # --- Memory Settings --- #
    MEMORY_N_RESULTS: int = Field(default=10, description="Number of relevant memories to retrieve during search (Set via ASK_LLM_MEMORY_N_RESULTS)")
    MEMORY_PROTECTED_RECENT_TURNS: int = Field(default=3, description="Number of recent conversation turns to always include regardless of token limits (Set via ASK_LLM_MEMORY_PROTECTED_RECENT_TURNS)")
    MEMORY_MIN_RELEVANCE: float = Field(default=0.01, description="Minimum relevance score (0.0-1.0) for memories to be included (Set via ASK_LLM_MEMORY_MIN_RELEVANCE)")
    MEMORY_MAX_TOKEN_PERCENT: int = Field(default=30, description="Maximum percentage of context window to use for memories (Set via ASK_LLM_MEMORY_MAX_TOKEN_PERCENT)")
    MEMORY_DEDUP_SIMILARITY: float = Field(default=0.85, description="Similarity threshold (0.0-1.0) for fuzzy deduplication of memories against history (Set via ASK_LLM_MEMORY_DEDUP_SIMILARITY)")

    # --- PostgreSQL Memory Backend Settings --- #
    POSTGRES_HOST: str = Field(default="postgres.home", description="PostgreSQL server hostname (Set via ASK_LLM_POSTGRES_HOST)")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL server port (Set via ASK_LLM_POSTGRES_PORT)")
    POSTGRES_USER: str = Field(default="askllm", description="PostgreSQL username (Set via ASK_LLM_POSTGRES_USER)")
    POSTGRES_PASSWORD: str = Field(default="", description="PostgreSQL password (Set via ASK_LLM_POSTGRES_PASSWORD)")
    POSTGRES_DATABASE: str = Field(default="askllm", description="PostgreSQL database name (Set via ASK_LLM_POSTGRES_DATABASE)")
    
    # --- Memory Extraction Settings --- #
    MEMORY_EXTRACTION_ENABLED: bool = Field(default=True, description="Enable LLM-based memory extraction from conversations")
    MEMORY_EXTRACTION_MIN_IMPORTANCE: float = Field(default=0.3, description="Minimum importance score for extracted memories to be stored")
    MEMORY_EMBEDDING_DIM: int = Field(default=384, description="Dimension of embedding vectors for semantic search (384 for all-MiniLM-L6-v2)")
    MEMORY_EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model for local embeddings")
    EXTRACTION_MODEL: Optional[str] = Field(default=None, description="Model alias to use for memory extraction (defaults to first available)")
    
    # --- Memory Decay Settings --- #
    MEMORY_DECAY_ENABLED: bool = Field(default=True, description="Enable temporal decay for memory relevance scoring")
    MEMORY_DECAY_HALF_LIFE_DAYS: float = Field(default=90.0, description="Half-life in days for memory decay (90 = memory at 50% weight after 90 days)")
    MEMORY_ACCESS_BOOST_FACTOR: float = Field(default=0.15, description="How much to boost frequently accessed memories (0.15 = +15% per log(access_count))")
    MEMORY_RECENCY_WEIGHT: float = Field(default=0.3, description="Weight given to recency vs base importance (0.3 = 30% recency, 70% base)")
    MEMORY_DIVERSITY_ENABLED: bool = Field(default=True, description="Enable diversity sampling to avoid echo chambers in retrieval")
    
    # --- Memory Consolidation Settings --- #
    MEMORY_CONSOLIDATION_THRESHOLD: float = Field(default=0.75, description="Cosine similarity threshold for clustering similar memories (0.0-1.0)")

    # --- Service Settings --- #
    SERVICE_MODEL: Optional[str] = Field(default=None, description="Default model alias for the background service API")
    SERVICE_HOST: str = Field(default="127.0.0.1", description="Host for the background service to bind to")
    SERVICE_PORT: int = Field(default=8642, description="Port for the background service to listen on")

    # --- LLM Generation Settings --- #
    MAX_TOKENS: int = Field(default=1024*4, description="Default maximum tokens to generate")
    TEMPERATURE: float = Field(default=0.8, description="Default generation temperature")
    TOP_P: float = Field(default=0.95, description="Default nucleus sampling top-p")
    LLAMA_CPP_N_CTX: int = Field(default=8192, description="Context size for Llama.cpp models")
    LLAMA_CPP_N_GPU_LAYERS: int = Field(default=-1, description="Number of layers to offload to GPU (-1 for all possible layers)")

    # --- UI/Interaction Settings --- #
    VERBOSE: bool = Field(default=False, description="Verbose mode for debugging")
    PLAIN_OUTPUT: bool = Field(default=False, description="Use plain text output without Rich formatting")
    NO_STREAM: bool = Field(default=False, description="Disable streaming output")
    INTERACTIVE_MODE: bool = Field(default=False, description="Whether the app is in interactive mode (set based on args)")

    # SYSTEM_MESSAGE is set at runtime from bot config, this is just the default
    SYSTEM_MESSAGE: str = Field(default=DEFAULT_SYSTEM_MESSAGE)

    defined_models: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    available_ollama_models: List[str] = Field(default_factory=list, exclude=True)
    ollama_checked: bool = Field(default=False, exclude=True)

    model_config = SettingsConfigDict(
        env_prefix="ASK_LLM_",
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )

    def __init__(self, **values: Any):
        if 'MODELS_CONFIG_PATH' in values:
            values['MODELS_CONFIG_PATH'] = str(Path(values['MODELS_CONFIG_PATH']).expanduser().resolve())
        else:
            values['MODELS_CONFIG_PATH'] = str(DEFAULT_MODELS_YAML)

        super().__init__(**values)
        self._load_models_config()

    def _load_models_config(self):
        config_path = Path(self.MODELS_CONFIG_PATH)
        if not config_path.is_file():
            console.print(f"[yellow]Warning:[/yellow] Models configuration file not found at {config_path}")
            console.print("  Define models in a YAML file (see docs). Using empty model definitions.")
            self.defined_models = {"models": {}}
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
                if loaded_data is None:
                    loaded_data = {} # Start with empty dict
            if "models" not in loaded_data or not isinstance(loaded_data.get("models"), dict):
                console.print(f"[bold red]Warning:[/bold red] Invalid format in {config_path}. Missing or invalid top-level 'models' dictionary. Treating as empty.")
                self.defined_models = {"models": {}}
            else:
                self.defined_models = {"models": loaded_data["models"]}

        except yaml.YAMLError as e:
            console.print(f"[bold red]Error parsing YAML file {config_path}:[/bold red] {e}")
            self.defined_models = {"models": {}}
        except Exception as e:
            console.print(f"[bold red]Error loading models config {config_path}:[/bold red] {e}")
            self.defined_models = {"models": {}}

    def _check_ollama_availability(self, force_check: bool = False) -> None:
        if not force_check and self.ollama_checked:
            return

        try:
            import requests # Import lazily
            start_time = time.time()
            response = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            available_models = {model['name'] for model in response.json().get('models', [])}
            self.available_ollama_models = list(available_models)
            self.ollama_checked = True
            logger.debug(f"Ollama API query took {time.time() - start_time:.2f} ms and found {len(available_models)} models")
        except Exception as e:
            self.available_ollama_models = []
            self.ollama_checked = True
            logger.warning(f"Ollama API check failed: {str(e)}")

    def force_ollama_check(self) -> None:
        self.ollama_checked = False
        self._check_ollama_availability(force_check=True)

    def get_model_options(self) -> List[str]:
        available_options = []
        defined = self.defined_models.get("models", {})
        for alias, model_info in defined.items():
            model_type = model_info.get("type")
            if model_type == PROVIDER_OPENAI:
                available_options.append(alias)
            elif model_type == PROVIDER_OLLAMA:
                # Do not probe the Ollama server during normal execution.
                # If the alias is defined, treat it as selectable and let actual
                # Ollama usage/install flows perform connectivity/model checks.
                model_id = model_info.get("model_id")
                if model_id:
                    available_options.append(alias)
            elif model_type == PROVIDER_GGUF:
                if is_llama_cpp_available():
                    available_options.append(alias)
            elif model_type == PROVIDER_HF:
                if is_huggingface_available():
                    available_options.append(alias)

        return sorted(list(set(available_options))) # Return sorted list of unique aliases


def set_config_value(key: str, value: str, config: Config) -> bool:
    """Sets a configuration value in the .env file.

    Args:
        key: The configuration key (e.g., 'DEFAULT_MODEL_ALIAS').
        value: The value to set.
        config: The loaded Config object to get the .env path and validate keys.

    Returns:
        True if successful, False otherwise.
    """
    dotenv_path = Path(config.model_config['env_file'])
    key_upper = key.upper()
    env_var_name = f"{config.model_config['env_prefix']}{key_upper}"
    valid_keys = {k.upper() for k in Config.model_fields.keys()}
    if key_upper not in valid_keys:
        console.print(f"[bold red]Error:[/bold red] Invalid configuration key '{key}'. Valid keys are: {', '.join(sorted(Config.model_fields.keys()))}")
        return False

    lines = []
    found = False
    try:
        if dotenv_path.is_file():
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
    except IOError as e:
        console.print(f"[bold red]Error reading {dotenv_path}:[/bold red] {e}")
        return False
    new_line = f"{env_var_name}={value}\n"
    updated_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(f"{env_var_name}="):
            updated_lines.append(new_line)
            found = True
        else:
            updated_lines.append(line) # Keep existing line

    if not found:
        updated_lines.append(new_line)
    try:
        dotenv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dotenv_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        return True
    except IOError as e:
        console.print(f"[bold red]Error writing to {dotenv_path}:[/bold red] {e}")
        return False


def has_database_credentials(cfg: Config | None = None) -> bool:
    """Check if database credentials are configured.
    
    Returns True if POSTGRES_PASSWORD is set (non-empty string).
    Other credentials have defaults so password is the key indicator.
    
    Args:
        cfg: Config instance to check. If None, uses the global config.
    """
    if cfg is None:
        cfg = config
    return bool(cfg.POSTGRES_PASSWORD and cfg.POSTGRES_PASSWORD.strip())


config = Config()
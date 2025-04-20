import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from pydantic import Field, ValidationError, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

PROVIDER_OPENAI = "openai"
PROVIDER_OLLAMA = "ollama"
PROVIDER_GGUF = "gguf"
PROVIDER_HF = "huggingface"
PROVIDER_UNKNOWN = "Unknown" 

logger = logging.getLogger(__name__)
console = Console()

_hf_available = None
_llama_cpp_available = None

def is_huggingface_available() -> bool:
    global _hf_available
    if _hf_available is None:
        torch_spec = importlib.util.find_spec("torch")
        transformers_spec = importlib.util.find_spec("transformers")
        bitsandbytes_spec = importlib.util.find_spec("bitsandbytes")
        _hf_available = all([torch_spec, transformers_spec, bitsandbytes_spec])
    return _hf_available

def is_llama_cpp_available() -> bool:
    global _llama_cpp_available
    if _llama_cpp_available is None:
        _llama_cpp_available = importlib.util.find_spec("llama_cpp") is not None
    return _llama_cpp_available

def get_default_models_yaml_path() -> Path:
    env_path = os.environ.get("ASK_LLM_MODELS_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    default_config_dir = Path(xdg_config_home) / "ask-llm"
    default_config_dir.mkdir(parents=True, exist_ok=True)
    return default_config_dir / "models.yaml"

DEFAULT_MODELS_YAML = get_default_models_yaml_path()

class Config(BaseSettings):
    HISTORY_FILE: str = Field(default=os.path.expanduser("~/.ask-llm-chat-history"))
    HISTORY_DURATION: int = Field(default=60 * 10)
    OLLAMA_URL: str = Field(default="http://localhost:11434")
    MODEL_CACHE_DIR: str = Field(default=os.path.expanduser("~/.cache/ask_llm/models"), description="Directory to cache downloaded GGUF models")
    MODELS_CONFIG_PATH: str = Field(default=str(DEFAULT_MODELS_YAML), description="Path to the models YAML definition file")
    DEFAULT_MODEL_ALIAS: Optional[str] = Field(default=None, description="Alias from models.yaml to use if --model is not specified")
    ALLOW_DUPLICATE_RESPONSE: bool = Field(default=False, description="Allow identical consecutive assistant responses without retry")

    MAX_TOKENS: int = Field(default=1024, description="Default maximum tokens to generate")
    TEMPERATURE: float = Field(default=0.8, description="Default generation temperature")
    TOP_P: float = Field(default=0.95, description="Default nucleus sampling top-p")
    CHAT_FORMAT: Optional[str] = Field(default=None, description="Default chat format for Llama.cpp (e.g., llama-2, chatml, mistral) - Can be overridden per model")

    # Llama.cpp specific settings
    LLAMA_CPP_N_CTX: int = Field(default=4096, description="Context size for Llama.cpp models")
    LLAMA_CPP_N_GPU_LAYERS: int = Field(default=-1, description="Number of layers to offload to GPU (-1 for all possible layers)")

    VERBOSE: bool = Field(default=False, description="Verbose mode for debugging")
    PLAIN_OUTPUT: bool = Field(default=False, description="Use plain text output without Rich formatting")
    NO_STREAM: bool = Field(default=False, description="Disable streaming output")
    INTERACTIVE_MODE: bool = Field(default=False, description="Whether the app is in interactive mode (set based on args)")
    SYSTEM_MESSAGE: str = Field(default="""You are a helpful and concise technical assistant. Respond using simple, easy-to-understand language. Keep explanations short and direct. Use correct and clean Markdown formatting: backticks for code, lists/headings for structure, bold for emphasis. Avoid complex elements like HTML or LaTeX. Prioritize clarity and conciseness.""")

    defined_models: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    available_ollama_models: List[str] = Field(default_factory=list, exclude=True)
    ollama_checked: bool = Field(default=False, exclude=True)

    model_config = SettingsConfigDict(env_prefix="ASK_LLM_",env_file=".env",env_file_encoding="utf-8",case_sensitive=False,extra='ignore')

    def __init__(self, **values: Any):
        if 'MODELS_CONFIG_PATH' in values:
            values['MODELS_CONFIG_PATH'] = str(Path(values['MODELS_CONFIG_PATH']).expanduser().resolve())
        else:
            values['MODELS_CONFIG_PATH'] = str(DEFAULT_MODELS_YAML)

        super().__init__(**values)
        self._load_models_config()

        if self.DEFAULT_MODEL_ALIAS is None:
            self.DEFAULT_MODEL_ALIAS = self.defined_models.get("default_model_alias")

        has_ollama_models = any(model.get('type') == 'ollama' for model in self.defined_models.get('models', {}).values())
        if has_ollama_models:
            self._check_ollama_availability()

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
                    loaded_data = {"models": {}}

            if "models" not in loaded_data or not isinstance(loaded_data.get("models"), dict):
                console.print(f"[bold red]Error:[/bold red] Invalid format in {config_path}. Missing or invalid top-level 'models' dictionary.")
                self.defined_models = {"models": {}}
            else:
                self.defined_models = loaded_data
                global_overrides = {k: v for k, v in loaded_data.items() if k != "models"}
                for key, value in global_overrides.items():
                    if hasattr(self, key):
                        try:
                            setattr(self, key, value)
                        except ValidationError as e:
                            console.print(f"[yellow]Warning:[/yellow] Invalid value for '{key}' in {config_path}: {e}. Using default/environment value.")

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

    @computed_field
    @property
    def MODEL_OPTIONS(self) -> List[str]:
        available_options = []
        defined = self.defined_models.get("models", {})
        for alias, model_info in defined.items():
            model_type = model_info.get("type")
            if model_type == PROVIDER_OPENAI:
                # Assume openai package installed if type is specified
                available_options.append(alias)
            elif model_type == PROVIDER_OLLAMA:
                # Check if model name is in the list fetched from Ollama server
                model_id = model_info.get("model_id")
                if model_id and model_id in self.available_ollama_models:
                    available_options.append(alias)
            elif model_type == PROVIDER_GGUF:
                # Check if llama-cpp-python is installed
                if is_llama_cpp_available():
                    available_options.append(alias)
            elif model_type == PROVIDER_HF:
                # Check if huggingface dependencies are installed
                if is_huggingface_available():
                    available_options.append(alias)
            # Ignore models with unknown or missing type

        return sorted(list(set(available_options))) # Return sorted list of unique aliases


global_config = Config()
import re
import pathlib
import yaml
from typing import List, Dict, Any
from rich.prompt import Prompt, Confirm
from rich.console import Console

from .utils.config import Config

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    hf_hub_available = True
except ImportError:
    hf_hub_available = False
    hf_hub_download = lambda **kwargs: (_ for _ in ()).throw(ImportError("huggingface-hub is not installed"))
    HfApi = type('DummyHfApi', (), {'list_repo_files': lambda **kwargs: (_ for _ in ()).throw(ImportError("huggingface-hub is not installed"))})
    HfHubHTTPError = type('DummyHfHubHTTPError', (Exception,), {})

console = Console()

def normalize_for_match(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())

def generate_gguf_alias(repo_id: str, filename: str, existing_aliases: List[str]) -> str:
    repo_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
    base_filename = filename.lower().replace(".gguf", "")
    base_filename = re.sub(r'[._]q\d[._]\w+', '', base_filename)
    base_filename = re.sub(r'[._]q\d+k?[sm]?', '', base_filename)
    base_filename = base_filename.replace(repo_name.lower(), '').strip(' .-_/')
    alias_base = re.sub(r'[^a-z0-9]+', '-', f"{repo_name}-{base_filename}".lower()).strip('-')
    alias_base = alias_base or repo_name.lower()

    alias = alias_base
    counter = 1
    while alias in existing_aliases:
        alias = f"{alias_base}-{counter}"
        counter += 1
    return alias

def handle_add_gguf(repo_id: str, config: Config) -> bool:
    if not hf_hub_available or not HfApi or not HfHubHTTPError or not hf_hub_download:
        console.print("[bold red]Error:[/bold red] `huggingface-hub` is required to add GGUF models.")
        console.print("Install with: `pip install huggingface-hub`")
        return False

    console.print(f"Attempting to add GGUF from repository: [cyan]{repo_id}[/cyan]")

    try:
        api = HfApi()
        console.print("Listing files in repository...")
        files_in_repo = api.list_repo_files(repo_id=repo_id)
        gguf_files = sorted([f for f in files_in_repo if f.lower().endswith(".gguf")])
    except HfHubHTTPError as e:
        console.print(f"[bold red]Error accessing HF repo '{repo_id}':[/bold red] {e}")
        return False
    except Exception as e:
        console.print(f"[bold red]Error listing files in '{repo_id}':[/bold red] {e}")
        return False

    if not gguf_files:
        console.print(f"[bold red]Error:[/bold red] No GGUF files found in '{repo_id}'.")
        return False

    selected_filename = _select_gguf_file(repo_id, gguf_files)
    if not selected_filename:
        return False

    download_success = _download_gguf_file(repo_id, selected_filename, config)
    if not download_success:
        return False

    update_success = _update_models_yaml(repo_id, selected_filename, config)
    return update_success

def _select_gguf_file(repo_id: str, gguf_files: List[str]) -> str | None:
    if len(gguf_files) == 1:
        selected_filename = gguf_files[0]
        console.print(f"Found single GGUF file: [green]{selected_filename}[/green]")
        if not Confirm.ask(f"Add this file?\n  Repo: {repo_id}\n  File: {selected_filename}", default=True):
             console.print("Add operation cancelled.")
             return None
        return selected_filename
    else:
        console.print("Multiple GGUF files found. Select one:")
        choices_map = {str(i+1): fname for i, fname in enumerate(gguf_files)}
        for i, fname in enumerate(gguf_files):
            console.print(f"  [cyan]{i+1}[/cyan]: {fname}")
        while True:
            try:
                choice_str = Prompt.ask("Enter the number of the file", choices=list(choices_map.keys()), default="1")
                return choices_map[choice_str]
            except (ValueError, KeyError):
                console.print("[yellow]Invalid selection.[/yellow]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[red]Selection cancelled.[/red]")
                return None

def _download_gguf_file(repo_id: str, filename: str, config: Config) -> bool:
    if not hf_hub_download:
        return False
    console.print(f"Verifying/Downloading '[yellow]{filename}[/yellow]' to cache...")
    cache_dir = pathlib.Path(config.MODEL_CACHE_DIR).expanduser()
    model_repo_cache_dir = cache_dir / repo_id
    local_model_path = model_repo_cache_dir / filename

    if local_model_path.is_file():
         if config.VERBOSE: console.print("[dim]File already exists in cache.[/dim]")
         return True
    else:
         model_repo_cache_dir.mkdir(parents=True, exist_ok=True)
         try:
             downloaded_path_str = hf_hub_download(repo_id=repo_id,filename=filename,local_dir=str(model_repo_cache_dir),local_dir_use_symlinks=False,)
             if config.VERBOSE: console.print(f"[dim]Download complete:[/dim] {downloaded_path_str}")
             return True
         except Exception as e:
             console.print(f"[bold red]Error downloading file '{filename}':[/bold red] {e}")
             console.print("Cannot add model to config without successful download/verification.")
             return False

def _update_models_yaml(repo_id: str, selected_filename: str, config: Config) -> bool:
    yaml_path = pathlib.Path(config.MODELS_CONFIG_PATH)
    models_data: Dict[str, Any] = {"models": {}}
    success = False

    try:
        if yaml_path.is_file():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
                if isinstance(loaded_data, dict):
                    models_data = loaded_data
                elif loaded_data is not None:
                    console.print(f"[yellow]Warning:[/yellow] Existing content in {yaml_path} is not a dictionary. Overwriting structure.")
        else:
             console.print(f"[yellow]Models file {yaml_path} not found. Creating a new one.[/yellow]")

        if "models" not in models_data or not isinstance(models_data.get("models"), dict):
            models_data["models"] = {}

        existing_aliases = list(models_data.get("models", {}).keys())
        new_alias = generate_gguf_alias(repo_id, selected_filename, existing_aliases)

        model_entry = {"type": "gguf","repo_id": repo_id,"filename": selected_filename,"description": f"{repo_id}/{selected_filename} (GGUF)"}

        for alias, definition in models_data.get("models", {}).items():
            if (definition.get("type") == "gguf" and definition.get("repo_id") == repo_id and definition.get("filename") == selected_filename):
                console.print(f"[yellow]Model definition already exists under alias '[cyan]{alias}[/cyan]'. Not adding duplicate.")
                return True

        confirmed_alias = Prompt.ask(f"Proposed alias [cyan]{new_alias}[/cyan]. Use this or enter a new one?", default=new_alias)
        while not confirmed_alias or confirmed_alias in existing_aliases:
            if not confirmed_alias:
                console.print("[yellow]Alias cannot be empty.[/yellow]")
            else:
                console.print(f"[yellow]Alias '[cyan]{confirmed_alias}[/cyan]' already exists.[/yellow]")
            confirmed_alias = Prompt.ask("Please enter a unique alias", default=generate_gguf_alias(repo_id, selected_filename, existing_aliases + [confirmed_alias]))

        models_data["models"][confirmed_alias] = model_entry

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
             yaml.dump(models_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        console.print("[green]Success![/green]")
        console.print(f"Added model to [bold]{yaml_path}[/bold]")
        console.print(f"Use with: `ask-llm --model {confirmed_alias} ...`")
        success = True

    except yaml.YAMLError as e:
        console.print(f"[bold red]Error processing YAML file {yaml_path}:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]Unexpected error updating {yaml_path}:[/bold red] {e}")
        if config.VERBOSE:
            import traceback
            traceback.print_exc()

    return success 
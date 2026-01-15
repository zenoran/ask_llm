"""CLI display functions for status, bots, and user profiles.

This module contains the display/presentation logic for the CLI,
separated from the main CLI parsing and execution logic.
"""

import argparse
import importlib.util
import os
import shutil
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ask_llm.utils.config import Config, has_database_credentials, is_huggingface_available, is_llama_cpp_available
from ask_llm.bots import BotManager
from ask_llm.user_profile import UserProfileManager, DEFAULT_USER_ID

console = Console()


def _get_service_client_func() -> Callable | None:
    """Get the service client getter function."""
    try:
        from ask_llm.cli import get_service_client
        return get_service_client
    except ImportError:
        return None


def show_status(config: Config, args: argparse.Namespace | None = None):
    """Display overall system status including dependencies, bots, memory, and configuration."""
    console.print(Panel.fit("[bold magenta]ask_llm System Status[/bold magenta]", border_style="magenta"))
    console.print()
    
    # Build status table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    # --- Background Service Section ---
    _add_service_status(table, config)
    table.add_row("", "")
    
    # --- Current Session Section (based on args) ---
    if args:
        _add_session_status(table, config, args)
        table.add_row("", "")
    
    # --- Dependencies Section ---
    _add_dependencies_status(table, config)
    table.add_row("", "")
    
    # --- Bots Section ---
    _add_bots_status(table, config)
    table.add_row("", "")
    
    # --- Memory Section ---
    db_status, long_term_count, messages_count, short_term_count = _get_memory_status(config)
    table.add_row("[bold]Memory[/bold]", "")
    table.add_row("  PostgreSQL Backend", db_status)
    table.add_row("  Messages (permanent)", f"[green]{messages_count}[/green]" if messages_count else "[dim]0[/dim]")
    table.add_row("  Memories (distilled)", f"[green]{long_term_count}[/green]" if long_term_count else "[dim]0[/dim]")
    table.add_row("  Session messages", f"[green]{short_term_count}[/green]" if short_term_count else "[dim]0[/dim]")
    table.add_row("", "")
    
    # --- Configuration Section ---
    _add_config_status(table, config)
    table.add_row("", "")
    
    # --- Pipeline Self-Checks Section ---
    _add_pipeline_checks(table, config)

    console.print(table)
    console.print()


def _add_service_status(table: Table, config: Config):
    """Add background service status to the table."""
    table.add_row("[bold]Background Service[/bold]", "")
    try:
        get_service_client = _get_service_client_func()
        if get_service_client:
            service_client = get_service_client()
            if service_client and service_client.is_available(force_check=True):
                # Get detailed status
                status = service_client.get_status()
                if status and status.available:
                    uptime_str = ""
                    if status.uptime_seconds:
                        hours, remainder = divmod(int(status.uptime_seconds), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        if hours > 0:
                            uptime_str = f"{hours}h {minutes}m"
                        elif minutes > 0:
                            uptime_str = f"{minutes}m {seconds}s"
                        else:
                            uptime_str = f"{seconds}s"
                    
                    service_status = "[green]✓ Running[/green]"
                    if uptime_str:
                        service_status += f" [dim](uptime: {uptime_str})[/dim]"
                    table.add_row("  Status", service_status)
                    table.add_row("  URL", f"[dim]{service_client.http_url}[/dim]")
                    table.add_row("  Version", f"{status.version or 'unknown'}")
                    table.add_row("  Tasks Processed", f"{status.tasks_processed}")
                    table.add_row("  Tasks Pending", f"{status.tasks_pending}")
                    if status.models_loaded:
                        table.add_row("  Models Loaded", ", ".join(status.models_loaded))
                else:
                    table.add_row("  Status", "[yellow]⚠ Unhealthy response[/yellow]")
            else:
                table.add_row("  Status", "[dim]○ Not running[/dim]")
                table.add_row("  ", "[dim]Start with: llm-service[/dim]")
        else:
            table.add_row("  Status", "[dim]○ Service module not available[/dim]")
    except Exception as e:
        table.add_row("  Status", f"[red]✗ Error checking: {e}[/red]")


def _add_session_status(table: Table, config: Config, args: argparse.Namespace):
    """Add current session status to the table."""
    table.add_row("[bold]Current Session[/bold]", "")
    
    # Determine effective bot
    bot_manager = BotManager(config)
    if args.bot:
        target_bot = bot_manager.get_bot(args.bot)
        if target_bot:
            bot_display = f"[bold cyan]{target_bot.name}[/bold cyan] ({args.bot}) [dim]--bot {args.bot}[/dim]"
        else:
            bot_display = f"[red]Unknown: {args.bot}[/red]"
            target_bot = bot_manager.get_default_bot()
    elif getattr(args, 'local', False):
        target_bot = bot_manager.get_default_bot(local_mode=True)
        bot_display = f"[bold yellow]{target_bot.name}[/bold yellow] ({target_bot.slug}) [dim]--local default[/dim]"
    else:
        target_bot = bot_manager.get_default_bot()
        bot_display = f"[bold cyan]{target_bot.name}[/bold cyan] ({target_bot.slug}) [dim]default[/dim]"
    table.add_row("  Bot", bot_display)
    
    # Determine effective model: -m flag > bot's default_model > config DEFAULT_MODEL_ALIAS
    explicit_model = getattr(args, 'model', None)
    if explicit_model:
        model_alias = explicit_model
        model_source = "[dim]-m flag[/dim]"
    elif target_bot.default_model:
        model_alias = target_bot.default_model
        model_source = "[dim]bot default[/dim]"
    else:
        model_alias = config.DEFAULT_MODEL_ALIAS
        model_source = "[dim]config default[/dim]"
    
    if model_alias:
        # Check if model exists in defined models
        defined_models = config.defined_models.get("models", {})
        if model_alias in defined_models:
            model_def = defined_models.get(model_alias, {})
            model_type = model_def.get("type", "unknown")
            model_display = f"[bold green]{model_alias}[/bold green] [dim]({model_type})[/dim] {model_source}"
        else:
            # Check for partial matches
            matches = [a for a in defined_models.keys() if model_alias.lower() in a.lower()]
            if matches:
                model_display = f"[yellow]{model_alias}[/yellow] [dim](partial match: {matches[0]})[/dim] {model_source}"
            else:
                model_display = f"[red]{model_alias}[/red] [dim](not found)[/dim] {model_source}"
    else:
        model_display = "[dim]not set[/dim]"
    table.add_row("  Model", model_display)
    
    # Determine effective user
    user_id = getattr(args, 'user', DEFAULT_USER_ID)
    if getattr(args, 'local', False):
        user_display = "[dim]N/A (--local mode)[/dim]"
    else:
        try:
            profile_manager = UserProfileManager(config)
            profile = profile_manager.get_profile(user_id)
            if profile and profile.name:
                user_display = f"[bold cyan]{profile.name}[/bold cyan] ({user_id})"
            else:
                user_display = f"{user_id} [dim](no profile)[/dim]"
        except Exception:
            user_display = f"{user_id}"
    table.add_row("  User", user_display)
    
    # Show mode
    mode_parts = []
    if getattr(args, 'local', False):
        mode_parts.append("[yellow]local[/yellow]")
    if getattr(args, 'plain', False):
        mode_parts.append("plain")
    if getattr(args, 'no_stream', False):
        mode_parts.append("no-stream")
    mode_display = ", ".join(mode_parts) if mode_parts else "[dim]default[/dim]"
    table.add_row("  Mode", mode_display)


def _add_dependencies_status(table: Table, config: Config):
    """Add dependencies status to the table."""
    table.add_row("[bold]Dependencies[/bold]", "")
    
    # Check CUDA availability
    cuda_status = "[dim]○ Not available[/dim]"
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # Try to get CUDA version
        try:
            import subprocess
            import re
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                match = re.search(r"release ([\d.]+)", result.stdout)
                version = match.group(1) if match else "unknown"
                cuda_status = f"[green]✓ Available[/green] (CUDA {version})"
            else:
                cuda_status = "[yellow]⚠ nvcc found but failed[/yellow]"
        except Exception:
            cuda_status = f"[green]✓ Available[/green] (nvcc at {nvcc_path})"
    table.add_row("  CUDA", cuda_status)
    
    # Check huggingface-hub (for GGUF downloads)
    hf_hub_available = importlib.util.find_spec("huggingface_hub") is not None
    hf_hub_status = "[green]✓ Installed[/green]" if hf_hub_available else "[red]✗ Not installed[/red] (pip install huggingface-hub)"
    table.add_row("  huggingface-hub", hf_hub_status)
    
    # Check llama-cpp-python (for GGUF inference)
    llama_cpp_available = is_llama_cpp_available()
    llama_cpp_status = "[green]✓ Installed[/green]" if llama_cpp_available else "[red]✗ Not installed[/red] (pip install llama-cpp-python)"
    # If llama-cpp is installed, check if it has CUDA support
    if llama_cpp_available:
        try:
            # Suppress the ggml_cuda_init logging during import (native C code writes to fd)
            import os as os_module
            devnull = os_module.open(os_module.devnull, os_module.O_WRONLY)
            old_stdout_fd = os_module.dup(1)
            old_stderr_fd = os_module.dup(2)
            try:
                os_module.dup2(devnull, 1)
                os_module.dup2(devnull, 2)
                from llama_cpp import llama_supports_gpu_offload
                has_gpu = llama_supports_gpu_offload()
            finally:
                os_module.dup2(old_stdout_fd, 1)
                os_module.dup2(old_stderr_fd, 2)
                os_module.close(devnull)
                os_module.close(old_stdout_fd)
                os_module.close(old_stderr_fd)
            if has_gpu:
                llama_cpp_status = "[green]✓ Installed[/green] (GPU support)"
            else:
                llama_cpp_status = "[green]✓ Installed[/green] [dim](CPU only)[/dim]"
        except (ImportError, AttributeError, OSError):
            llama_cpp_status = "[green]✓ Installed[/green]"
    table.add_row("  llama-cpp-python", llama_cpp_status)
    
    # Check HuggingFace transformers (for HF models)
    hf_available = is_huggingface_available()
    hf_status = "[green]✓ Installed[/green]" if hf_available else "[dim]○ Not installed[/dim] (pip install transformers torch)"
    table.add_row("  transformers + torch", hf_status)
    
    # Check ollama connectivity
    ollama_status = "[dim]○ Not checked[/dim]"
    if config.OLLAMA_URL:
        try:
            import httpx
            response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0)
            if response.status_code == 200:
                model_count = len(response.json().get("models", []))
                ollama_status = f"[green]✓ Connected[/green] ({model_count} models)"
            else:
                ollama_status = f"[yellow]⚠ Server responded with {response.status_code}[/yellow]"
        except Exception:
            ollama_status = f"[red]✗ Not reachable[/red] ({config.OLLAMA_URL})"
    table.add_row("  Ollama server", ollama_status)
    
    # Check OpenAI API key (from environment variable)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_status = "[green]✓ API key set[/green]" if openai_api_key else "[dim]○ No API key[/dim] (export OPENAI_API_KEY=...)"
    table.add_row("  OpenAI API", openai_status)


def _add_bots_status(table: Table, config: Config):
    """Add bots status to the table."""
    bot_manager = BotManager(config)
    default_bot = bot_manager.get_default_bot()
    local_bot = bot_manager.get_default_bot(local_mode=True)
    
    table.add_row("[bold]Bots[/bold]", "")
    table.add_row("  Default", f"[bold cyan]{default_bot.name}[/bold cyan] - {default_bot.description}")
    table.add_row("  Local (--local)", f"[bold yellow]{local_bot.name}[/bold yellow] - {local_bot.description}")
    table.add_row("  Available", ", ".join(b.slug for b in bot_manager.list_bots()))


def _get_memory_status(config: Config) -> tuple[str, int, int, int]:
    """Get memory backend status information."""
    from ask_llm.core import discover_memory_backends
    
    db_status = "[yellow]Not Configured[/yellow]"
    long_term_count = 0
    messages_count = 0
    short_term_count = 0
    
    bot_manager = BotManager(config)
    default_bot = bot_manager.get_default_bot()
    
    backends = discover_memory_backends()
    if 'postgresql' in backends:
        if not has_database_credentials(config):
            db_status = "[yellow]Not Configured[/yellow] [dim](set ASK_LLM_POSTGRES_PASSWORD)[/dim]"
        else:
            try:
                backend_class = backends['postgresql']
                backend = backend_class(config=config, bot_id=default_bot.slug)
                db_stats = backend.stats()
                long_term_count = db_stats.get('memories', {}).get('total_count', 0)
                messages_count = db_stats.get('messages', {}).get('total_count', 0)
                db_status = f"[green]Connected[/green] ({config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE})"
                
                short_term_mgr = backend_class.get_short_term_manager(config=config, bot_id=default_bot.slug)
                short_term_count = short_term_mgr.count()
            except Exception as e:
                db_status = f"[red]Error: {e}[/red]"
    else:
        db_status = "[yellow]Backend not available[/yellow]"
    
    return db_status, long_term_count, messages_count, short_term_count


def _add_config_status(table: Table, config: Config):
    """Add configuration status to the table."""
    table.add_row("[bold]Configuration[/bold]", "")
    table.add_row("  Default Model", f"{config.DEFAULT_MODEL_ALIAS or '[dim]not set[/dim]'}")
    table.add_row("  Models Config", f"{config.MODELS_CONFIG_PATH}")
    table.add_row("  Memory N Results", f"{config.MEMORY_N_RESULTS}")
    table.add_row("  Min Relevance", f"{config.MEMORY_MIN_RELEVANCE}")
    table.add_row("  History Duration", f"{config.HISTORY_DURATION}s ({config.HISTORY_DURATION // 60} min)")
    table.add_row("  History File (--local)", f"{config.HISTORY_FILE}")


def _add_pipeline_checks(table: Table, config: Config):
    """Add pipeline self-checks to the table."""
    table.add_row("[bold]Pipeline Self-Checks[/bold]", "")
    
    # Check 1: Default model is valid and accessible
    defined_models = config.defined_models.get("models", {})
    if config.DEFAULT_MODEL_ALIAS:
        if config.DEFAULT_MODEL_ALIAS in defined_models:
            model_def = defined_models[config.DEFAULT_MODEL_ALIAS]
            model_type = model_def.get("type", "unknown")
            
            # Check model-specific requirements
            model_check = "[green]✓ Valid[/green]"
            hf_hub_available = importlib.util.find_spec("huggingface_hub") is not None
            llama_cpp_available = is_llama_cpp_available()
            hf_available = is_huggingface_available()
            
            if model_type == "gguf":
                if not llama_cpp_available:
                    model_check = "[red]✗ llama-cpp-python not installed[/red]"
                elif not hf_hub_available:
                    model_check = "[yellow]⚠ huggingface-hub not installed (needed for downloads)[/yellow]"
                else:
                    # Check if model file exists
                    source = model_def.get("source", "")
                    if "/" in source:
                        parts = source.split("/")
                        if len(parts) >= 3:
                            repo_id = "/".join(parts[:2])
                            filename = "/".join(parts[2:])
                            cache_path = os.path.join(config.MODEL_CACHE_DIR, repo_id, filename)
                            if os.path.exists(cache_path):
                                model_check = f"[green]✓ Cached[/green] [dim]({os.path.getsize(cache_path) / (1024**3):.1f}GB)[/dim]"
                            else:
                                model_check = "[yellow]⚠ Not cached (will download on first use)[/yellow]"
            elif model_type == "ollama":
                # Check ollama connectivity
                ollama_ok = False
                try:
                    import httpx
                    response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2.0)
                    if response.status_code == 200:
                        ollama_ok = True
                except Exception:
                    pass
                if ollama_ok:
                    model_check = "[green]✓ Ollama available[/green]"
                else:
                    model_check = "[yellow]⚠ Ollama server not reachable[/yellow]"
            elif model_type == "openai":
                if os.getenv("OPENAI_API_KEY"):
                    model_check = "[green]✓ API key set[/green]"
                else:
                    model_check = "[red]✗ No OPENAI_API_KEY[/red]"
            elif model_type == "huggingface":
                if not hf_available:
                    model_check = "[red]✗ transformers/torch not installed[/red]"
            
            table.add_row("  Default Model", f"{model_check}")
        else:
            table.add_row("  Default Model", f"[red]✗ '{config.DEFAULT_MODEL_ALIAS}' not found in models.yaml[/red]")
    else:
        table.add_row("  Default Model", "[yellow]⚠ Not configured[/yellow]")
    
    # Check 2: Models config file exists and is valid
    models_config_path = config.MODELS_CONFIG_PATH
    if os.path.exists(models_config_path):
        model_count = len(defined_models)
        if model_count > 0:
            table.add_row("  Models Config", f"[green]✓ {model_count} models defined[/green]")
        else:
            table.add_row("  Models Config", "[yellow]⚠ File exists but no models defined[/yellow]")
    else:
        table.add_row("  Models Config", f"[red]✗ File not found: {models_config_path}[/red]")
    
    # Check 3: Memory backend connectivity
    from ask_llm.core import discover_memory_backends
    backends = discover_memory_backends()
    if 'postgresql' in backends and has_database_credentials(config):
        try:
            from pgvector.psycopg2 import register_vector  # noqa: F401
            table.add_row("  Memory Backend", "[green]✓ PostgreSQL + pgvector[/green]")
        except ImportError:
            table.add_row("  Memory Backend", "[yellow]⚠ pgvector Python package not installed[/yellow]")
    elif 'postgresql' in backends and not has_database_credentials(config):
        table.add_row("  Memory Backend", "[dim]○ Not configured (set ASK_LLM_POSTGRES_PASSWORD)[/dim]")
    elif 'postgresql' in backends:
        table.add_row("  Memory Backend", "[red]✗ PostgreSQL connection failed[/red]")
    else:
        table.add_row("  Memory Backend", "[dim]○ Not configured (--local mode only)[/dim]")
    
    # Check 4: History file location is writable
    history_dir = os.path.dirname(config.HISTORY_FILE)
    if os.path.exists(history_dir):
        if os.access(history_dir, os.W_OK):
            table.add_row("  History Storage", "[green]✓ Writable[/green]")
        else:
            table.add_row("  History Storage", f"[red]✗ Not writable: {history_dir}[/red]")
    else:
        # Check if we can create it
        try:
            os.makedirs(history_dir, exist_ok=True)
            table.add_row("  History Storage", "[green]✓ Created[/green]")
        except Exception as e:
            table.add_row("  History Storage", f"[red]✗ Cannot create: {e}[/red]")
    
    # Check 5: Service mode readiness
    get_service_client = _get_service_client_func()
    if get_service_client:
        service_client = get_service_client()
        if service_client and service_client.is_available():
            table.add_row("  Service Mode", "[green]✓ Ready (use --service)[/green]")
        else:
            table.add_row("  Service Mode", "[dim]○ Service not running[/dim]")
    else:
        table.add_row("  Service Mode", "[dim]○ Service module not available[/dim]")


def show_bots(config: Config):
    """Display available bots."""
    bot_manager = BotManager(config)
    bots = bot_manager.list_bots()
    default_bot = bot_manager.get_default_bot()
    
    console.print(Panel.fit("[bold cyan]Available Bots[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Default Model")
    table.add_column("Memory", justify="center")
    
    for bot in bots:
        is_default = " ⭐" if bot.slug == default_bot.slug else ""
        memory_icon = "[green]✓[/green]" if bot.requires_memory else "[dim]✗[/dim]"
        default_model = bot.default_model or f"[dim]{config.DEFAULT_MODEL_ALIAS or 'global'}[/dim]"
        table.add_row(
            f"{bot.slug}{is_default}",
            bot.name,
            bot.description,
            default_model,
            memory_icon
        )
    
    console.print(table)
    console.print()
    console.print("[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def show_user_profile(config: Config, user_id: str = DEFAULT_USER_ID):
    """Display user profile."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
        return
    
    try:
        manager = UserProfileManager(config)
        profile = manager.get_profile(user_id)
    except Exception as e:
        console.print(f"[red]Could not load user profile: {e}[/red]")
        return
    
    if not profile:
        console.print(f"[yellow]No user profile found for '{user_id}'.[/yellow]")
        console.print("[dim]Create one with: llm --user-profile-setup[/dim]")
        return
    
    console.print(Panel.fit(f"[bold cyan]User Profile: {user_id}[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    
    table.add_row("[bold]Identity[/bold]", "")
    table.add_row("  name", profile.name or "[dim]not set[/dim]")
    table.add_row("  preferred_name", profile.preferred_name or "[dim]not set[/dim]")
    
    # Preferences
    prefs = profile.preferences or {}
    if prefs:
        table.add_row("", "")
        table.add_row("[bold]Preferences[/bold]", "")
        for key, value in prefs.items():
            table.add_row(f"  preferences.{key}", str(value))
    
    # Context
    ctx = profile.context or {}
    if ctx:
        table.add_row("", "")
        table.add_row("[bold]Context[/bold]", "")
        for key, value in ctx.items():
            if isinstance(value, list):
                table.add_row(f"  context.{key}", ", ".join(str(v) for v in value))
            else:
                table.add_row(f"  context.{key}", str(value))
    
    console.print(table)
    console.print()
    console.print("[dim]Update with: llm --user-profile-set name=\"Your Name\"[/dim]")
    console.print("[dim]Add context: llm --user-profile-set context.occupation=\"Developer\"[/dim]")
    console.print()


def show_users(config: Config):
    """Display all user profiles."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
        return
    
    try:
        manager = UserProfileManager(config)
        profiles = manager.list_all_profiles()
    except Exception as e:
        console.print(f"[red]Could not load users: {e}[/red]")
        return
    
    if not profiles:
        console.print("[yellow]No user profiles found.[/yellow]")
        console.print("[dim]Create one with: llm --user-profile-setup[/dim]")
        return
    
    console.print(Panel.fit("[bold cyan]User Profiles[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("User ID", style="cyan")
    table.add_column("Name")
    table.add_column("Preferred Name")
    table.add_column("Occupation")
    
    for profile in profiles:
        occupation = profile.context.get("occupation", "") if profile.context else ""
        table.add_row(
            profile.user_id,
            profile.name or "[dim]-[/dim]",
            profile.preferred_name or "[dim]-[/dim]",
            occupation or "[dim]-[/dim]"
        )
    
    console.print(table)
    console.print()
    console.print("[dim]Use --user <id> to select a user profile[/dim]")
    console.print()

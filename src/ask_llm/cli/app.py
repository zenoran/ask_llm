import argparse
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from rich.prompt import Prompt
from ask_llm.utils.config import Config, has_database_credentials
from ask_llm.utils.config import set_config_value
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.core import AskLLM
from ask_llm.model_manager import list_models, update_models_interactive, delete_model, ModelManager
from ask_llm.gguf_handler import handle_add_gguf
from ask_llm.bots import BotManager
from ask_llm.profiles import ProfileManager, EntityType, AttributeCategory
from ask_llm.utils.streaming import render_streaming_response, render_complete_response
from ask_llm.shared.logging import LogConfig
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from typing import Iterator

# Backward-compatible constant - prefer config.DEFAULT_USER
DEFAULT_USER_ID = "default"

console = Console()

# Cache for service client
_service_client = None

def get_service_client():
    """Get or create the service client singleton."""
    global _service_client
    if _service_client is None:
        try:
            from ask_llm.service import ServiceClient
            _service_client = ServiceClient()
        except ImportError:
            _service_client = False  # Mark as unavailable
    return _service_client if _service_client else None


def query_via_service(
    prompt: str,
    model: str | None,
    bot_id: str | None,
    user_id: str | None,
    plaintext_output: bool,
    stream: bool,
) -> bool:
    """
    Query via the background service if available.
    
    Returns True if query was handled by service, False if service unavailable.
    """
    from rich.markdown import Markdown
    from rich.align import Align
    
    client = get_service_client()
    if not client or not client.is_available():
        return False
    
    messages = [{"role": "user", "content": prompt}]
    
    # Get bot name for display
    bot_manager = BotManager(Config())
    bot = bot_manager.get_bot(bot_id) if bot_id else bot_manager.get_default_bot()
    bot_name = bot.name if bot else (bot_id or "Assistant")
    
    # Determine panel style based on bot
    panel_styles = {
        "mira": ("magenta", "magenta"),
        "nova": ("cyan", "cyan"),
        "spark": ("yellow", "yellow"),
    }
    title_style, border_style = panel_styles.get(bot_id or "", ("green", "green"))
    panel_title = f"[bold {title_style}]{bot_name}[/bold {title_style}]"
    
    try:
        if stream:
            # Stream response using shared streaming utilities
            stream_iterator = client.chat_completion(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                stream=True,
            )
            
            render_streaming_response(
                stream_iterator=stream_iterator,
                console=console,
                panel_title=panel_title,
                panel_border_style=border_style,
                plaintext_output=plaintext_output,
            )
            return True
        else:
            response = client.chat_completion_full(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
            )
            if response:
                if plaintext_output:
                    print(response)
                else:
                    # Use shared render function for split display
                    render_complete_response(
                        response=response,
                        console=console,
                        panel_title=panel_title,
                        panel_border_style=border_style,
                    )
                return True
    except Exception as e:
        logging.debug(f"Service query failed: {e}")
    
    return False


def show_status(config: Config, args: argparse.Namespace | None = None):
    """Display overall system status including dependencies, bots, memory, and configuration."""
    import importlib.util
    import os
    import shutil
    from ask_llm.utils.config import is_huggingface_available, is_llama_cpp_available, has_database_credentials
    
    console.print(Panel.fit("[bold magenta]ask_llm System Status[/bold magenta]", border_style="magenta"))
    console.print()
    
    # Build status table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    # --- Background Service Section ---
    table.add_row("[bold]Background Service[/bold]", "")
    try:
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
                
                service_status = f"[green]✓ Running[/green]"
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
    except Exception as e:
        table.add_row("  Status", f"[red]✗ Error checking: {e}[/red]")
    
    table.add_row("", "")
    
    # --- Current Session Section (based on args) ---
    if args:
        table.add_row("[bold]Current Session[/bold]", "")
        
        # Determine effective bot
        bot_manager = BotManager(config)
        if args.bot:
            target_bot = bot_manager.get_bot(args.bot)
            if target_bot:
                bot_display = f"[bold cyan]{target_bot.name}[/bold cyan] ({args.bot}) [dim]--bot {args.bot}[/dim]"
            else:
                bot_display = f"[red]Unknown: {args.bot}[/red]"
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
            model_source = f"[dim]bot default[/dim]"
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
            user_display = f"[dim]N/A (--local mode)[/dim]"
        else:
            try:
                from ask_llm.profiles import ProfileManager, EntityType
                
                manager = ProfileManager(config)
                profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
                if profile and profile.display_name:
                    user_display = f"[bold cyan]{profile.display_name}[/bold cyan] ({user_id})"
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
        
        table.add_row("", "")
    
    # --- Dependencies Section ---
    table.add_row("[bold]Dependencies[/bold]", "")
    
    # Check CUDA availability
    cuda_status = "[dim]○ Not available[/dim]"
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # Try to get CUDA version
        try:
            import subprocess
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse version from output like "Cuda compilation tools, release 12.2, V12.2.140"
                import re
                match = re.search(r"release ([\d.]+)", result.stdout)
                version = match.group(1) if match else "unknown"
                cuda_status = f"[green]✓ Available[/green] (CUDA {version})"
            else:
                cuda_status = f"[yellow]⚠ nvcc found but failed[/yellow]"
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
            import sys
            import os as os_module
            # Save original file descriptors
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
    
    table.add_row("", "")
    
    # --- Bots Section ---
    bot_manager = BotManager(config)
    default_bot = bot_manager.get_default_bot()
    local_bot = bot_manager.get_default_bot(local_mode=True)
    
    table.add_row("[bold]Bots[/bold]", "")
    table.add_row("  Default", f"[bold cyan]{default_bot.name}[/bold cyan] - {default_bot.description}")
    table.add_row("  Local (--local)", f"[bold yellow]{local_bot.name}[/bold yellow] - {local_bot.description}")
    table.add_row("  Available", ", ".join(b.slug for b in bot_manager.list_bots()))
    table.add_row("", "")
    
    # --- Memory Section ---
    db_status = "[yellow]Not Configured[/yellow]"
    long_term_count = 0
    messages_count = 0
    
    if not has_database_credentials(config):
        db_status = "[yellow]Not Configured[/yellow] [dim](set ASK_LLM_POSTGRES_PASSWORD)[/dim]"
    else:
        try:
            from ask_llm.memory.postgresql import PostgreSQLMemoryBackend
            backend = PostgreSQLMemoryBackend(config, bot_id=default_bot.slug)
            db_stats = backend.stats()
            long_term_count = db_stats.get('memories', {}).get('total_count', 0)
            messages_count = db_stats.get('messages', {}).get('total_count', 0)
            db_status = f"[green]Connected[/green] ({config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE})"
        except Exception as e:
            db_status = f"[red]Error: {e}[/red]"
    
    table.add_row("[bold]Memory[/bold]", "")
    table.add_row("  PostgreSQL Backend", db_status)
    table.add_row("  Messages (permanent)", f"[green]{messages_count}[/green]" if messages_count else "[dim]0[/dim]")
    table.add_row("  Memories (distilled)", f"[green]{long_term_count}[/green]" if long_term_count else "[dim]0[/dim]")
    table.add_row("", "")
    
    # --- Configuration Section ---
    table.add_row("[bold]Configuration[/bold]", "")
    table.add_row("  Default Model", f"{config.DEFAULT_MODEL_ALIAS or '[dim]not set[/dim]'}")
    table.add_row("  Models Config", f"{config.MODELS_CONFIG_PATH}")
    table.add_row("  Memory N Results", f"{config.MEMORY_N_RESULTS}")
    table.add_row("  Min Relevance", f"{config.MEMORY_MIN_RELEVANCE}")
    table.add_row("  History Duration", f"{config.HISTORY_DURATION}s ({config.HISTORY_DURATION // 60} min)")
    table.add_row("  History File (--local)", f"{config.HISTORY_FILE}")
    table.add_row("", "")
    
    # --- Pipeline Self-Checks Section ---
    table.add_row("[bold]Pipeline Self-Checks[/bold]", "")
    
    # Check 1: Default model is valid and accessible
    defined_models = config.defined_models.get("models", {})
    if config.DEFAULT_MODEL_ALIAS:
        if config.DEFAULT_MODEL_ALIAS in defined_models:
            model_def = defined_models[config.DEFAULT_MODEL_ALIAS]
            model_type = model_def.get("type", "unknown")
            
            # Check model-specific requirements
            model_check = "[green]✓ Valid[/green]"
            if model_type == "gguf":
                if not is_llama_cpp_available():
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
                if ollama_status.startswith("[green]"):
                    model_check = "[green]✓ Ollama available[/green]"
                else:
                    model_check = "[yellow]⚠ Ollama server not reachable[/yellow]"
            elif model_type == "openai":
                if openai_api_key:
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
    
    # Check 3: Memory backend connectivity (already checked above)
    if db_status.startswith("[green]"):
        # Test vector extension
        try:
            from pgvector.psycopg2 import register_vector
            table.add_row("  Memory Backend", "[green]✓ PostgreSQL + pgvector[/green]")
        except ImportError:
            table.add_row("  Memory Backend", "[yellow]⚠ pgvector Python package not installed[/yellow]")
    elif not has_database_credentials(config):
        table.add_row("  Memory Backend", "[dim]○ Not configured (set ASK_LLM_POSTGRES_PASSWORD)[/dim]")
    elif db_status.startswith("[red]"):
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
    service_client = get_service_client()
    if service_client and service_client.is_available():
        table.add_row("  Service Mode", "[green]✓ Ready (use --service)[/green]")
    else:
        table.add_row("  Service Mode", "[dim]○ Service not running[/dim]")

    console.print(table)
    console.print()


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
    console.print(f"[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def show_user_profile(config: Config, user_id: str = DEFAULT_USER_ID):
    """Display user profile."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
        return
    
    try:
        from ask_llm.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
        attributes = manager.get_all_attributes(EntityType.USER, user_id)
    except Exception as e:
        console.print(f"[red]Could not load user profile: {e}[/red]")
        return
    
    console.print(Panel.fit(f"[bold cyan]User Profile: {user_id}[/bold cyan]", border_style="cyan"))
    console.print()
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    
    table.add_row("[bold]Identity[/bold]", "")
    table.add_row("  Display Name", profile.display_name or "[dim]not set[/dim]")
    table.add_row("  Description", profile.description or "[dim]not set[/dim]")
    
    # Group attributes by category
    by_category = {}
    for attr in attributes:
        if attr.category not in by_category:
            by_category[attr.category] = []
        by_category[attr.category].append(attr)
    
    # Display attributes by category
    category_names = {"preference": "Preferences", "fact": "Facts", "interest": "Interests", "communication": "Communication", "context": "Context"}
    for category, attrs in sorted(by_category.items()):
        table.add_row("", "")
        table.add_row(f"[bold]{category_names.get(category, category.title())}[/bold]", "")
        for attr in sorted(attrs, key=lambda a: a.key):
            value_str = str(attr.value) if not isinstance(attr.value, str) or len(str(attr.value)) < 60 else str(attr.value)[:57] + "..."
            conf_str = f" [dim]({attr.confidence:.0%})[/dim]" if attr.confidence < 1.0 else ""
            table.add_row(f"  {attr.key}", f"{value_str}{conf_str}")
    
    console.print(table)
    console.print()
    console.print("[dim]Add attributes: llm --user-profile-set category.key=value[/dim]")
    console.print()


def show_users(config: Config):
    """Display all user profiles."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
        return
    
    try:
        from ask_llm.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profiles = manager.list_all_profiles(EntityType.USER)
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
    table.add_column("Display Name")
    table.add_column("Description")
    
    for profile in profiles:
        table.add_row(
            profile.entity_id,
            profile.display_name or "[dim]-[/dim]",
            (profile.description[:50] + "..." if profile.description and len(profile.description) > 50 else profile.description) or "[dim]-[/dim]"
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]Use --user <id> to select a user profile[/dim]")
    console.print()


def run_user_profile_setup(config: Config, user_id: str = DEFAULT_USER_ID) -> bool:
    """Run interactive user profile setup wizard."""
    if not has_database_credentials(config):
        console.print("[yellow]User profiles require database connection.[/yellow]")
        console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
        return False
    
    console.print(Panel.fit("[bold cyan]User Profile Setup[/bold cyan]", border_style="cyan"))
    console.print()
    console.print(f"[dim]Setting up profile for user: {user_id}[/dim]")
    console.print(f"[dim]Press Enter to skip any field.[/dim]")
    console.print()
    
    try:
        from ask_llm.profiles import ProfileManager, EntityType, AttributeCategory
        
        manager = ProfileManager(config)
        profile, _ = manager.get_or_create_profile(EntityType.USER, user_id)
        
        # Get existing attributes
        existing_attrs = {attr.key: attr.value for attr in manager.get_all_attributes(EntityType.USER, user_id)}
        
        # Prompt for basic info
        name = Prompt.ask(
            "What's your name?",
            default=profile.display_name or existing_attrs.get("name", "")
        )
        
        occupation = Prompt.ask(
            "What do you do? (occupation)",
            default=existing_attrs.get("occupation", "")
        )
        
        console.print()
        
        # Save to new profile system
        if name:
            manager.update_profile(EntityType.USER, user_id, display_name=name)
            manager.set_attribute(EntityType.USER, user_id, AttributeCategory.FACT, "name", name, source="explicit")
        
        if occupation:
            manager.set_attribute(EntityType.USER, user_id, AttributeCategory.FACT, "occupation", occupation, source="explicit")
        
        console.print(f"[green]✓ User profile saved for '{user_id}'[/green]")
        console.print()
        return True
            
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Setup cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Error during setup: {e}[/red]")
        return False


def ensure_user_profile(config: Config, user_id: str = DEFAULT_USER_ID) -> bool:
    """Ensure user profile exists, prompting for setup if needed.
    
    Returns True if profile exists or setup succeeded, False if setup was cancelled.
    """
    if not has_database_credentials(config):
        logging.getLogger(__name__).debug("Database credentials not configured - skipping user profile")
        return True  # Not an error, just no profile
    
    try:
        from ask_llm.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        profile, is_new = manager.get_or_create_profile(EntityType.USER, user_id)
        
        # If profile has no name, run setup wizard
        if not profile.display_name:
            console.print()
            console.print(f"[yellow]Welcome! Let's set up your user profile.[/yellow]")
            console.print()
            return run_user_profile_setup(config, user_id)
        
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not ensure user profile: {e}")
        return True  # Don't block on profile errors


def parse_arguments(config_obj: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM models from the command line using model aliases defined in models.yaml")
    parser.add_argument("-m","--model",type=str,default=None,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: bot's default or {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="(Deprecated: use --add-model gguf) Add a GGUF model from a Hugging Face repo ID.")
    parser.add_argument("--add-model",type=str,choices=['ollama', 'openai', 'gguf'],metavar="TYPE",help="Add models: 'ollama' (refresh from server), 'openai' (query API), 'gguf' (add from HuggingFace repo)")
    parser.add_argument("--delete-model",type=str,metavar="ALIAS",help="Delete the specified model alias from the configuration file after confirmation.")
    parser.add_argument("--config-set", nargs=2, metavar=("KEY", "VALUE"), help="Set a configuration value (e.g., DEFAULT_MODEL_ALIAS) in the .env file.")
    parser.add_argument("--config-list", action="store_true", help="List the current effective configuration settings.")
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-dh", "--delete-history", action="store_true", help="Clear chat history")
    parser.add_argument("-ph", "--print-history", nargs="?", const=-1, type=int, default=None,help="Print chat history (optional: number of recent pairs)")
    parser.add_argument("-c", "--command", help="Execute command and add output to question")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--no-stream", action="store_true", default=False, help="Disable streaming output")
    parser.add_argument("--local", action="store_true", help="Use local filesystem for history instead of database")
    parser.add_argument("--service", action="store_true", help="Route queries through the background service (if running)")
    parser.add_argument("-b", "--bot", type=str, default=None, help="Bot to use (nova, spark, mira). Use --list-bots to see all.")
    parser.add_argument("--list-bots", action="store_true", help="List available bots and exit")
    parser.add_argument("--status", action="store_true", help="Show memory system status and configuration")
    parser.add_argument("--user", type=str, default=DEFAULT_USER_ID, help="User profile to use (creates if not exists)")
    parser.add_argument("--list-users", action="store_true", help="List all user profiles")
    parser.add_argument("--user-profile", action="store_true", help="Show current user profile")
    parser.add_argument("--user-profile-set", metavar="FIELD=VALUE", help="Set a user profile field (e.g., name=\"Nick\")")
    parser.add_argument("--user-profile-setup", action="store_true", help="Run user profile setup wizard")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output.")
    return parser.parse_args()

def main():
    try:
        # --- Minimal Pre-parsing for flags affecting config loading/display --- 
        # We need to parse --verbose, --plain early, but also handle config paths if specified.
        # Using a separate parser for this minimal set.
        prelim_parser = argparse.ArgumentParser(add_help=False)
        prelim_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        prelim_parser.add_argument("--plain", action="store_true", help="Use plain text output")
        prelim_args, remaining_argv = prelim_parser.parse_known_args()

        # --- Instantiate Config --- 
        config_obj = Config()

    except Exception as e:
        # Catch potential Pydantic validation errors or file issues during Config init
        console.print(f"[bold red]Error initializing configuration:[/bold red] {e}")
        sys.exit(1)
    
    # Parse full arguments now that config is loaded
    args = parse_arguments(config_obj)
    config_obj.VERBOSE = args.verbose
    config_obj.DEBUG = args.debug
    config_obj.PLAIN_OUTPUT = args.plain
    config_obj.NO_STREAM = args.no_stream
    config_obj.INTERACTIVE_MODE = not args.question and not args.command  # Set interactive mode flag

    # Configure logging via centralized LogConfig
    LogConfig.configure(verbose=args.verbose, debug=args.debug)

    # Handle setting config values, guard missing attribute in stubbed args
    if getattr(args, 'config_set', None):
        key, value = args.config_set
        success = set_config_value(key, value, config_obj)
        if success:
            console.print(f"[green]Configuration '{key}' set to '{value}' in {config_obj.model_config.get('env_file', 'unknown')}.[/green]")
            sys.exit(0)
        else:
            console.print(f"[bold red]Failed to set configuration '{key}'.[/bold red]")
            sys.exit(1)
    # Handle listing config values
    elif getattr(args, 'config_list', False):
        env_file_path = config_obj.model_config.get('env_file', 'unknown')
        console.print(f"[bold magenta]Current Configuration Settings[/bold magenta]")
        console.print(f"[dim]Config file: {env_file_path}[/dim]\n")
        
        # Define which settings are important and should be set by the user
        # Format: field_name -> (is_secret, is_required, description)
        important_settings = {
            'OPENAI_API_KEY': (True, False, "Required for OpenAI/compatible APIs"),
            'DEFAULT_MODEL_ALIAS': (False, False, "Default model to use"),
            'DEFAULT_BOT': (False, False, "Default bot personality"),
            'DEFAULT_USER': (False, False, "Default user profile"),
            'POSTGRES_PASSWORD': (True, True, "Required for memory features"),
            'POSTGRES_HOST': (False, False, "PostgreSQL server hostname"),
            'OLLAMA_URL': (False, False, "Ollama server URL"),
        }
        
        # Check for missing important settings
        missing_settings = []
        for field_name, (is_secret, is_required, desc) in important_settings.items():
            value = getattr(config_obj, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                if is_required:
                    missing_settings.append((field_name, desc))
        
        if missing_settings:
            console.print("[yellow]⚠ Missing recommended settings:[/yellow]")
            for field_name, desc in missing_settings:
                env_var = f"ASK_LLM_{field_name}"
                console.print(f"  [red]✗[/red] {env_var} - {desc}")
            console.print()
        
        # Show all settings grouped by category
        exclude_fields = {'defined_models', 'available_ollama_models', 'ollama_checked', 'model_config', 'SYSTEM_MESSAGE'}
        secret_fields = {'OPENAI_API_KEY', 'POSTGRES_PASSWORD', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'}
        
        console.print("[bold]All Settings:[/bold]")
        for field_name, field_info in sorted(config_obj.model_fields.items()):
            if field_name not in exclude_fields:
                current_value = getattr(config_obj, field_name)
                if isinstance(current_value, Path):
                    current_value_str = str(current_value)
                elif field_name in secret_fields and current_value:
                    # Mask secrets
                    current_value_str = "[green]****[/green] (set)"
                elif field_name in secret_fields:
                    current_value_str = "[dim]not set[/dim]"
                elif current_value is None or (isinstance(current_value, str) and not current_value.strip()):
                    current_value_str = "[dim]not set[/dim]"
                else:
                    current_value_str = repr(current_value)

                console.print(f"  [cyan]{field_name}[/cyan]: {current_value_str}")
        
        console.print(f"\n  [dim]SYSTEM_MESSAGE: (set by bot config in bots.yaml)[/dim]")
        console.print(f"\n[dim]To set a value: llm --config-set KEY value[/dim]")
        console.print(f"[dim]Or edit: {env_file_path}[/dim]")
        sys.exit(0)

    elif getattr(args, 'status', False):
        show_status(config_obj, args)
        sys.exit(0)

    elif getattr(args, 'list_bots', False):
        show_bots(config_obj)
        sys.exit(0)

    elif getattr(args, 'list_users', False):
        show_users(config_obj)
        sys.exit(0)

    elif getattr(args, 'user_profile', False):
        show_user_profile(config_obj, args.user)
        sys.exit(0)
    
    elif getattr(args, 'user_profile_setup', False):
        success = run_user_profile_setup(config_obj, args.user)
        sys.exit(0 if success else 1)
    
    elif getattr(args, 'user_profile_set', None):
        if not has_database_credentials(config_obj):
            console.print("[yellow]User profiles require database connection.[/yellow]")
            console.print("[dim]Set ASK_LLM_POSTGRES_PASSWORD in ~/.config/ask-llm/.env[/dim]")
            sys.exit(1)
        try:
            from ask_llm.profiles import ProfileManager, EntityType, AttributeCategory
            
            field, value = args.user_profile_set.split("=", 1)
            field = field.strip()
            value = value.strip().strip('"').strip("'")
            
            manager = ProfileManager(config_obj)
            
            # Parse field: category.key or just key
            if "." in field:
                category_str, key = field.split(".", 1)
                category_map = {"preference": AttributeCategory.PREFERENCE, "fact": AttributeCategory.FACT, "interest": AttributeCategory.INTEREST, "communication": AttributeCategory.COMMUNICATION, "context": AttributeCategory.CONTEXT}
                category = category_map.get(category_str.lower())
                if not category:
                    console.print(f"[red]Invalid category: {category_str}[/red]")
                    console.print("[dim]Valid categories: preference, fact, interest, communication, context[/dim]")
                    sys.exit(1)
            else:
                # Default to fact category
                key = field
                category = AttributeCategory.FACT
            
            manager.set_attribute(EntityType.USER, args.user, category, key, value, source="explicit")
            console.print(f"[green]Set {category}.{key} = {value}[/green]")
        except ValueError:
            console.print("[red]Use format: --user-profile-set category.key=value[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(0)

    elif args.list_models:
        list_models(config_obj)
        sys.exit(0)
        return
    if args.delete_model:
        success = False
        try:
            success = delete_model(args.delete_model, config_obj)
            if success:
                console.print(f"[green]Model alias '{args.delete_model}' deleted successfully.[/green]")
        except Exception as e:
            console.print(f"[bold red]Error during delete model operation:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    if args.add_model:
        success = False
        try:
            if args.add_model == 'openai':
                success = update_models_interactive(config_obj, provider='openai')
            elif args.add_model == 'ollama':
                success = update_models_interactive(config_obj, provider='ollama')
            elif args.add_model == 'gguf':
                # Prompt for HuggingFace repo ID
                repo_id = Prompt.ask("Enter HuggingFace repo ID (e.g., TheBloke/Llama-2-7B-GGUF)")
                if repo_id and repo_id.strip():
                    success = handle_add_gguf(repo_id.strip(), config_obj)
                else:
                    console.print("[red]No repo ID provided. Cancelled.[/red]")
                    success = False
            if success:
                console.print(f"[green]Model add for '{args.add_model}' completed.[/green]")
            else:
                console.print(f"[red]Model add for '{args.add_model}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Model add cancelled.[/bold red]")
            success = False # Ensure failure on interrupt
        except Exception as e:
            console.print(f"[bold red]Error during model add:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    
    # Check if this is a history-only operation (doesn't need model)
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    # Check if using service mode (don't validate model locally - let service handle it)
    use_service = False
    if args.local:
        use_service = False
    elif getattr(args, 'service', False):
        use_service = True
    elif config_obj.USE_SERVICE:
        use_service = True
    
    # Determine which bot will be used (needed to get bot's default model)
    bot_manager = BotManager(config_obj)
    if args.bot:
        target_bot = bot_manager.get_bot(args.bot)
        if not target_bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
    elif args.local:
        target_bot = bot_manager.get_default_bot(local_mode=True)
    else:
        target_bot = bot_manager.get_bot(config_obj.DEFAULT_BOT) or bot_manager.get_default_bot()
    
    # Determine effective model alias (without validation for service mode)
    if args.model:
        effective_model = args.model
    elif target_bot.default_model:
        effective_model = target_bot.default_model
    else:
        effective_model = config_obj.DEFAULT_MODEL_ALIAS
    
    # For history-only or service mode, skip local model validation
    if history_only:
        resolved_alias = None
    elif use_service:
        # In service mode, pass the model alias directly - let the service validate it
        resolved_alias = effective_model
    else:
        # Local mode: validate model is available locally
        model_manager = ModelManager(config_obj)
        resolved_alias = model_manager.resolve_model_alias(effective_model)
        if not resolved_alias:
            sys.exit(1)
            return
    
    try:
        run_app(args, config_obj, resolved_alias)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during application execution:[/bold red] {e}")
        if config_obj.VERBOSE:
             traceback.print_exc()
        sys.exit(1)
    else:
        sys.exit(0)

def run_app(args: argparse.Namespace, config_obj: Config, resolved_alias: str):
    # Determine which bot to use
    bot_manager = BotManager(config_obj)
    
    # Handle --local with explicit --bot (conflicting intent warning)
    if args.local and args.bot:
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
        # Only warn if the bot requires memory (e.g., nova)
        if bot.requires_memory and config_obj.VERBOSE:
            console.print(f"[dim]Note: Using --local with --bot {args.bot}. The bot will run without database memory.[/dim]")
        bot_id = bot.slug
    elif args.bot:
        # Explicit bot selection
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
        bot_id = bot.slug
    elif args.local:
        # Local mode defaults to Spark
        bot = bot_manager.get_default_bot(local_mode=True)
        bot_id = bot.slug
    else:
        # Default bot from config
        bot = bot_manager.get_bot(config_obj.DEFAULT_BOT)
        if not bot:
            bot = bot_manager.get_default_bot()
        bot_id = bot.slug
    
    # Use config default if --user not specified
    user_id = args.user if args.user != DEFAULT_USER_ID else config_obj.DEFAULT_USER
    
    # Check if we're using service mode - explicit flag or config default
    # --local takes precedence and disables service mode
    use_service = False
    if args.local:
        use_service = False
    elif getattr(args, 'service', False):
        use_service = True
    elif config_obj.USE_SERVICE:
        use_service = True
    
    ask_llm = None
    
    # For history operations, we can use a lightweight path that doesn't require model init
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    if history_only:
        # Use lightweight history manager without full model initialization
        from ask_llm.clients.base import StubClient
        from ask_llm.utils.history import HistoryManager
        from ask_llm.memory_server.client import get_memory_client
        
        stub_client = StubClient(config=config_obj, bot_name=bot.name)
        
        # Initialize memory client if database available
        memory = None
        if not args.local and has_database_credentials(config_obj):
            try:
                memory = get_memory_client(
                    config=config_obj,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=getattr(config_obj, "MEMORY_SERVER_URL", None),
                )
            except Exception as e:
                logger.debug(f"Memory client init failed: {e}")
        
        history_manager = HistoryManager(
            client=stub_client,
            config=config_obj,
            db_backend=memory.get_short_term_manager() if memory else None,
            bot_id=bot_id,
        )
        history_manager.load_history()
        
        if args.delete_history:
            history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        
        if args.print_history is not None:
            history_manager.print_history(args.print_history)
        
        console.print()
        return
    
    # For query operations, we need full AskLLM
    need_local_ask_llm = not use_service or args.delete_history or args.print_history is not None
    
    if need_local_ask_llm:
        try:
            ask_llm = AskLLM(
                resolved_model_alias=resolved_alias,
                config=config_obj,
                local_mode=args.local,
                bot_id=bot_id,
                user_id=user_id,
                verbose=args.verbose,
                debug=args.debug,
            )
        except (ImportError, FileNotFoundError, ValueError, Exception) as e:
             console.print(f"[bold red]Failed to initialize LLM client for '{resolved_alias}':[/bold red] {e}")
             if config_obj.VERBOSE:
                 traceback.print_exc()
             sys.exit(1)
             
    if args.delete_history:
        if ask_llm:
            ask_llm.history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        else:
            console.print("[yellow]History operations require a valid model configuration.[/yellow]")
    
    if args.print_history is not None:
        if ask_llm:
            ask_llm.history_manager.print_history(args.print_history)
        else:
            console.print("[yellow]History operations require a valid model configuration.[/yellow]")
        if not args.question and not args.command:
             console.print()
             return
    console.print("")
    command_output_str = ""
    if args.command:
        if config_obj.VERBOSE:
            console.print(f"Executing command: [yellow]{args.command}[/yellow]", highlight=False)
        try:
            result = subprocess.run(args.command,shell=True,capture_output=True,text=True,check=False)
            output = result.stdout.strip()
            error = result.stderr.strip()
            command_prefix = f"Command `[cyan]{args.command}[/cyan]` executed.\n"
            command_output_str += command_prefix
            if output:
                command_output_str += f"\nOutput:\n```\n{output}\n```\n"
            if error:
                command_output_str += f"\nError Output:\n```\n{error}\n```\n"
            if result.returncode != 0:
                status_msg = f"\n(Command exited with status {result.returncode})"
                console.print(f"[yellow]Warning: Command exited with status {result.returncode}[/yellow]", highlight=False)
                command_output_str += status_msg
            command_output_str += "\n---\n"
        except Exception as e:
            error_msg = f"Error executing command '{args.command}': {e}"
            console.print(f"[bold red]{error_msg}[/bold red]", highlight=False)
            command_output_str += f"{error_msg}\n\n---\n"
        console.print()
    stream_flag = not config_obj.NO_STREAM
    plaintext_flag = config_obj.PLAIN_OUTPUT
    # use_tools was already set above and factored into use_service decision
    
    def do_query(prompt: str):
        """Execute query via service or local client."""
        nonlocal ask_llm
        if use_service:
            if query_via_service(
                prompt,
                model=resolved_alias,  # Use resolved model, not raw args
                bot_id=bot_id,  # Use resolved bot_id
                user_id=user_id,  # Use resolved user_id
                plaintext_output=plaintext_flag,
                stream=stream_flag,
            ):
                return
            # Service unavailable, fall back to local - need to initialize AskLLM now
            if config_obj.VERBOSE:
                console.print("[dim]Service unavailable, using local client[/dim]")
            if ask_llm is None:
                try:
                    ask_llm = AskLLM(
                        resolved_model_alias=resolved_alias,
                        config=config_obj,
                        local_mode=args.local,
                        bot_id=bot_id,
                        user_id=user_id,
                        verbose=args.verbose,
                        debug=args.debug,
                    )
                except Exception as e:
                    console.print(f"[bold red]Failed to initialize LLM client:[/bold red] {e}")
                    return
        if ask_llm:
            # Standard mode: hardcoded memory retrieval
            ask_llm.query(prompt, plaintext_output=plaintext_flag, stream=stream_flag)

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        do_query(question_text.strip())
    elif command_output_str:
        if config_obj.VERBOSE:
            console.print("Command output captured, querying LLM with it...", highlight=False)
        do_query(command_output_str.strip())
    elif not sys.stdin.isatty():
        # Handle piped input - read once and exit
        piped_input = sys.stdin.read().strip()
        if piped_input:
            do_query(piped_input)
    else:
        console.print("[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]", highlight=False)
        console.print("[bold green]Type '>' at the beginning of a line for multiline input mode (end with Ctrl+D or Ctrl+Z).[/bold green]", highlight=False)
        handler_console = console
        if ask_llm and hasattr(ask_llm.client, 'console') and ask_llm.client.console:
            handler_console = ask_llm.client.console
        input_handler = MultilineInputHandler(console=handler_console)
        while True:
            try:
                prompt_text, is_multiline = input_handler.get_input("Enter your question:")
                if prompt_text is None or prompt_text.strip().lower() in ["exit", "quit"]:
                    console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                    console.print()
                    break
                if not prompt_text.strip():
                    console.print("[dim]Empty input received. Asking again...[/dim]")
                    continue
                console.print()
                if prompt_text.strip():
                    do_query(prompt_text)
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                console.print()
                break
    console.print()


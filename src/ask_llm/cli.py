import argparse
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from rich.prompt import Prompt
from ask_llm.utils.config import Config
from ask_llm.utils.config import set_config_value
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.core import AskLLM
from ask_llm.model_manager import list_models, update_models_interactive, delete_model, ModelManager
from ask_llm.gguf_handler import handle_add_gguf
from ask_llm.bots import BotManager
from ask_llm.user_profile import UserProfileManager, UserProfile, DEFAULT_USER_ID
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel

console = Console()


def show_memory_status(config: Config):
    """Display memory system status and configuration."""
    from ask_llm.core import discover_memory_backends
    
    console.print(Panel.fit("[bold magenta]Memory System Status[/bold magenta]", border_style="magenta"))
    console.print()
    
    # Get bot info
    bot_manager = BotManager(config)
    default_bot = bot_manager.get_default_bot()
    local_bot = bot_manager.get_default_bot(local_mode=True)
    
    # Check MariaDB connection
    mariadb_status = "[red]Not Configured[/red]"
    long_term_count = 0
    short_term_count = 0
    long_term_table = ""
    short_term_table = ""
    
    backends = discover_memory_backends()
    if 'mariadb' in backends:
        try:
            backend_class = backends['mariadb']
            backend = backend_class(config, bot_id=default_bot.slug)
            mariadb_stats = backend.stats()
            long_term_count = mariadb_stats.get('total_count', 0)
            long_term_table = mariadb_stats.get('table_name', '')
            mariadb_status = f"[green]Connected[/green] ({config.MARIADB_HOST}:{config.MARIADB_PORT}/{config.MARIADB_DATABASE})"
            
            # Check for short-term table using the new dynamic table name
            short_term_mgr = backend_class.get_short_term_manager(config, bot_id=default_bot.slug)
            short_term_stats = short_term_mgr.stats()
            short_term_count = short_term_stats.get('total_count', 0)
            short_term_table = short_term_stats.get('table_name', '')
        except Exception as e:
            mariadb_status = f"[red]Error: {e}[/red]"
    else:
        mariadb_status = "[yellow]Backend not installed[/yellow] (pip install ask_llm[memory-mariadb])"
    
    # Build status table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("[bold]Bots[/bold]", "")
    table.add_row("  Default", f"[bold cyan]{default_bot.name}[/bold cyan] - {default_bot.description}")
    table.add_row("  Local (--local)", f"[bold yellow]{local_bot.name}[/bold yellow] - {local_bot.description}")
    table.add_row("  Available", ", ".join(b.slug for b in bot_manager.list_bots()))
    table.add_row("", "")
    table.add_row("[bold]Memory Backends[/bold]", "")
    table.add_row("  MariaDB Backend", mariadb_status)
    if long_term_table:
        table.add_row(f"  Long-term ({long_term_table})", f"[green]{long_term_count}[/green]" if long_term_count else "[dim]0[/dim]")
    else:
        table.add_row(f"  Long-term", "[dim]No table[/dim]")
    if short_term_table:
        table.add_row(f"  Short-term ({short_term_table})", f"[green]{short_term_count}[/green]" if short_term_count else "[dim]0[/dim]")
    else:
        table.add_row(f"  Short-term", "[dim]No table (using file)[/dim]")
    table.add_row("", "")
    table.add_row("[bold]Configuration[/bold]", "")
    table.add_row("  Memory N Results", f"{config.MEMORY_N_RESULTS}")
    table.add_row("  Min Relevance", f"{config.MEMORY_MIN_RELEVANCE}")
    table.add_row("  Protected Recent Turns", f"{config.MEMORY_PROTECTED_RECENT_TURNS}")
    table.add_row("  Dedup Similarity", f"{config.MEMORY_DEDUP_SIMILARITY}")
    table.add_row("  History Duration", f"{config.HISTORY_DURATION}s ({config.HISTORY_DURATION // 60} min)")
    table.add_row("  History File (--local)", f"{config.HISTORY_FILE}")
    
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
    table.add_column("Memory", justify="center")
    
    for bot in bots:
        is_default = " ⭐" if bot.slug == default_bot.slug else ""
        memory_icon = "[green]✓[/green]" if bot.requires_memory else "[dim]✗[/dim]"
        table.add_row(
            f"{bot.slug}{is_default}",
            bot.name,
            bot.description,
            memory_icon
        )
    
    console.print(table)
    console.print()
    console.print(f"[dim]⭐ = default bot | ✓ = requires MariaDB | Use -b/--bot <slug> to select[/dim]")
    console.print()


def show_user_profile(config: Config, user_id: str = DEFAULT_USER_ID):
    """Display user profile."""
    try:
        manager = UserProfileManager(config)
        profile = manager.get_profile(user_id)
    except Exception as e:
        console.print(f"[red]Could not load user profile: {e}[/red]")
        return
    
    if not profile:
        console.print(f"[yellow]No user profile found for '{user_id}'.[/yellow]")
        console.print(f"[dim]Create one with: llm --user-profile-setup[/dim]")
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
    console.print(f"[dim]Use --user <id> to select a user profile[/dim]")
    console.print()


def run_user_profile_setup(config: Config, user_id: str = DEFAULT_USER_ID) -> bool:
    """Run interactive user profile setup wizard."""
    console.print(Panel.fit("[bold cyan]User Profile Setup[/bold cyan]", border_style="cyan"))
    console.print()
    console.print(f"[dim]Setting up profile for user: {user_id}[/dim]")
    console.print(f"[dim]Press Enter to skip any field.[/dim]")
    console.print()
    
    try:
        manager = UserProfileManager(config)
        existing = manager.get_profile(user_id)
        
        # Prompt for basic info
        name = Prompt.ask(
            "What's your name?",
            default=existing.name if existing and existing.name else ""
        )
        
        preferred_name = Prompt.ask(
            "What should I call you? (nickname)",
            default=existing.preferred_name if existing and existing.preferred_name else name
        )
        
        occupation = Prompt.ask(
            "What do you do? (occupation)",
            default=existing.context.get("occupation", "") if existing and existing.context else ""
        )
        
        console.print()
        
        # Build profile
        profile = UserProfile(
            user_id=user_id,
            name=name if name else None,
            preferred_name=preferred_name if preferred_name else None,
            preferences={},
            context={}
        )
        
        if occupation:
            profile.context["occupation"] = occupation
        
        # Save profile
        if manager.save_profile(profile):
            console.print(f"[green]✓ User profile saved for '{user_id}'[/green]")
            console.print()
            return True
        else:
            console.print(f"[red]Failed to save user profile.[/red]")
            return False
            
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Setup cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Error during setup: {e}[/red]")
        return False


def ensure_user_profile(config: Config, user_id: str = DEFAULT_USER_ID) -> UserProfile | None:
    """Ensure user profile exists, prompting for setup if needed.
    
    Returns the profile, or None if setup was cancelled/failed.
    """
    try:
        manager = UserProfileManager(config)
        profile, is_new = manager.get_or_create_profile(user_id)
        
        # If profile has no name set, run setup wizard
        if not profile.name:
            console.print()
            console.print(f"[yellow]Welcome! Let's set up your user profile.[/yellow]")
            console.print()
            if run_user_profile_setup(config, user_id):
                return manager.get_profile(user_id)
            else:
                # User cancelled - return empty profile (will work, just no personalization)
                return profile
        
        return profile
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not ensure user profile: {e}")
        return None


def parse_arguments(config_obj: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM models from the command line using model aliases defined in models.yaml")
    parser.add_argument("-m","--model",type=str,default=config_obj.DEFAULT_MODEL_ALIAS,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="Interactively select, download, and add a GGUF model from a Hugging Face repo ID to models.yaml.")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
    parser.add_argument("--add-model",type=str,choices=['ollama', 'openai'],metavar="TYPE",help="Refresh model list from a source: 'ollama' (check server availability), 'openai' (query API and update config)")
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
    parser.add_argument("--local", action="store_true", help="Use local filesystem for history instead of MariaDB")
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
        
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    prelim_parser.add_argument("--plain", action="store_true", help="Use plain text output")
    prelim_args, _ = prelim_parser.parse_known_args()
    args = parse_arguments(config_obj)
    config_obj.VERBOSE = args.verbose
    config_obj.PLAIN_OUTPUT = args.plain
    config_obj.NO_STREAM = args.no_stream
    config_obj.INTERACTIVE_MODE = not args.question and not args.command # Set interactive mode flag

    # Configure logging ONLY if debug is enabled
    if args.debug: # Check the new debug flag
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set higher levels for noisy libraries only when debugging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("markdown_it").setLevel(logging.WARNING)
    # else: Logging remains unconfigured

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
        console.print(f"[bold magenta]Current Configuration Settings:[/bold magenta]")
        console.print(f"(Sources: Defaults, Environment Variables, {config_obj.model_config.get('env_file', 'unknown')})\n")
        exclude_fields = {'defined_models', 'available_ollama_models', 'ollama_checked', 'model_config', 'SYSTEM_MESSAGE'}
        for field_name, field_info in sorted(config_obj.model_fields.items()):
            if field_name not in exclude_fields:
                current_value = getattr(config_obj, field_name)
                if isinstance(current_value, Path):
                    current_value_str = str(current_value)
                else:
                    current_value_str = repr(current_value)

                console.print(f"  [cyan]{field_name}[/cyan]: {current_value_str}")
        console.print(f"\n  [dim]SYSTEM_MESSAGE: (set by bot config in bots.yaml)[/dim]")
        sys.exit(0)

    elif getattr(args, 'status', False):
        show_memory_status(config_obj)
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
        try:
            field, value = args.user_profile_set.split("=", 1)
            field = field.strip()
            value = value.strip().strip('"').strip("'")
            manager = UserProfileManager(config_obj)
            if manager.update_field(field, value, args.user):
                console.print(f"[green]Updated {field} = {value}[/green]")
            else:
                console.print(f"[red]Invalid field: {field}[/red]")
                console.print("[dim]Valid fields: name, preferred_name, preferences.*, context.*[/dim]")
        except ValueError:
            console.print("[red]Use format: --user-profile-set field=value[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(0)

    elif args.list_models:
        list_models(config_obj)
        sys.exit(0)
        return
    if args.add_gguf:
        success = False
        try:
            success = handle_add_gguf(args.add_gguf, config_obj)
            if success:
                console.print(f"[green]Successfully processed GGUF model request for {args.add_gguf}.[/green]")
            else:
                 console.print(f"[red]Failed to process GGUF model request for {args.add_gguf}. Check logs above.[/red]")
        except Exception as e:
            console.print(f"[bold red]Error during GGUF add operation:[/bold red] {e}")
            if config_obj.VERBOSE:
                 traceback.print_exc()
            success = False # Ensure failure on exception
        sys.exit(0 if success else 1)
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
            if success:
                console.print(f"[green]Model list refresh for '{args.add_model}' completed.[/green]")
            else:
                console.print(f"[red]Model list refresh for '{args.add_model}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Model list refresh cancelled.[/bold red]")
            success = False # Ensure failure on interrupt
        except Exception as e:
            console.print(f"[bold red]Error during model refresh:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    model_manager = ModelManager(config_obj)
    resolved_alias = model_manager.resolve_model_alias(args.model)
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
        console.print(f"[yellow]Warning:[/yellow] Using --local with --bot {args.bot}. The bot will run without database memory.")
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
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
    
    # Ensure user profile exists (prompts for setup on first use)
    # Skip in local mode - it's transient/DB-free
    # Use config default if --user not specified
    user_id = args.user if args.user != DEFAULT_USER_ID else config_obj.DEFAULT_USER
    user_profile = None
    if not args.local:
        user_profile = ensure_user_profile(config_obj, user_id)
    
    try:
        ask_llm = AskLLM(
            resolved_model_alias=resolved_alias,
            config=config_obj,
            local_mode=args.local,
            bot_id=bot_id,
            user_id=user_id
        )
    except (ImportError, FileNotFoundError, ValueError, Exception) as e:
         console.print(f"[bold red]Failed to initialize LLM client for '{resolved_alias}':[/bold red] {e}")
         if config_obj.VERBOSE:
             traceback.print_exc()
         sys.exit(1) # Or re-raise the original error if preferred: raise e
    if args.delete_history:
        ask_llm.history_manager.clear_history()
        console.print("[green]Chat history cleared.[/green]")
    if args.print_history is not None:
        ask_llm.history_manager.print_history(args.print_history)
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

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        ask_llm.query(question_text.strip(), plaintext_output=plaintext_flag, stream=stream_flag)
    elif command_output_str:
        if config_obj.VERBOSE:
            console.print("Command output captured, querying LLM with it...", highlight=False)
        ask_llm.query(command_output_str.strip(), plaintext_output=plaintext_flag, stream=stream_flag)
    elif not sys.stdin.isatty():
        # Handle piped input - read once and exit
        piped_input = sys.stdin.read().strip()
        if piped_input:
            ask_llm.query(piped_input, plaintext_output=plaintext_flag, stream=stream_flag)
    else:
        console.print("[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]", highlight=False)
        console.print("[bold green]Type '>' at the beginning of a line for multiline input mode (end with Ctrl+D or Ctrl+Z).[/bold green]", highlight=False)
        handler_console = ask_llm.client.console if hasattr(ask_llm.client, 'console') and ask_llm.client.console else console
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
                    ask_llm.query(prompt_text, plaintext_output=plaintext_flag, stream=stream_flag)
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                console.print()
                break
    console.print()


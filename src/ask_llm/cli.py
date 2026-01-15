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
from ask_llm.user_profile import UserProfileManager, UserProfile, DEFAULT_USER_ID
from ask_llm.utils.streaming import render_streaming_response, render_complete_response
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from typing import Iterator

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



# Import display functions from dedicated module
from ask_llm.cli_display import show_status, show_bots, show_user_profile, show_users


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
    
    Returns the profile, or None if setup was cancelled/failed or database unavailable.
    """
    if not has_database_credentials(config):
        logging.getLogger(__name__).debug("Database credentials not configured - skipping user profile")
        return None
    
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
    parser.add_argument("-m","--model",type=str,default=None,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: bot's default or {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="(Deprecated: use --add-model gguf) Add a GGUF model from a Hugging Face repo ID.")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
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
        # Deprecated: redirect to the consolidated logic
        console.print("[yellow]Note:[/yellow] --add-gguf is deprecated. Use --add-model gguf instead.")
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
    
    # Determine effective model: -m flag > bot's default_model > config DEFAULT_MODEL_ALIAS
    if args.model:
        effective_model = args.model
    elif target_bot.default_model:
        effective_model = target_bot.default_model
    else:
        effective_model = config_obj.DEFAULT_MODEL_ALIAS
    
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
    memory_backend = None
    
    if not use_service:
        try:
            ask_llm = AskLLM(
                resolved_model_alias=resolved_alias,
                config=config_obj,
                local_mode=args.local,
                bot_id=bot_id,
                user_id=user_id
            )
            memory_backend = ask_llm.memory_backend
        except (ImportError, FileNotFoundError, ValueError, Exception) as e:
             console.print(f"[bold red]Failed to initialize LLM client for '{resolved_alias}':[/bold red] {e}")
             if config_obj.VERBOSE:
                 traceback.print_exc()
             sys.exit(1) # Or re-raise the original error if preferred: raise e
    if args.delete_history:
        if ask_llm:
            ask_llm.history_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
        else:
            console.print("[yellow]History operations require local mode (not --service)[/yellow]")
    
    if args.print_history is not None:
        if ask_llm:
            ask_llm.history_manager.print_history(args.print_history)
        else:
            console.print("[yellow]History operations require local mode (not --service)[/yellow]")
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
    # use_service already set above when deciding whether to init AskLLM
    
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
            console.print("[dim]Service unavailable, using local client[/dim]")
            if ask_llm is None:
                try:
                    ask_llm = AskLLM(
                        resolved_model_alias=resolved_alias,
                        config=config_obj,
                        local_mode=args.local,
                        bot_id=bot_id,
                        user_id=user_id
                    )
                except Exception as e:
                    console.print(f"[bold red]Failed to initialize LLM client:[/bold red] {e}")
                    return
        if ask_llm:
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


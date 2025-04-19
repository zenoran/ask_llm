import argparse
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from ask_llm.utils.config import Config
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.core import AskLLM
from ask_llm.model_manager import list_models, update_models_interactive, delete_model, ModelManager
from ask_llm.gguf_handler import handle_add_gguf
import logging

console = Console()

def parse_arguments(config_obj: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM models from the command line using model aliases defined in models.yaml")
    parser.add_argument("-m","--model",type=str,default=config_obj.DEFAULT_MODEL_ALIAS,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="Interactively select, download, and add a GGUF model from a Hugging Face repo ID to models.yaml.")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
    parser.add_argument("--refresh-models",type=str,choices=['ollama', 'openai'],metavar="TYPE",help="Refresh model list from a source: 'ollama' (check server availability), 'openai' (query API and update config)")
    parser.add_argument("--delete-model",type=str,metavar="ALIAS",help="Delete the specified model alias from the configuration file after confirmation.")
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-dh", "--delete-history", action="store_true", help="Clear chat history")
    parser.add_argument("-ph", "--print-history", nargs="?", const=-1, type=int, default=None,help="Print chat history (optional: number of recent pairs)")
    parser.add_argument("-c", "--command", help="Execute command and add output to question")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--no-stream", action="store_true", default=False, help="Disable streaming output")
    return parser.parse_args()

def main():
    try:
        config_obj = Config()
    except FileNotFoundError:
        console.print(f"[bold yellow]Warning:[/bold yellow] Configuration file not found at default location. Using built-in defaults.")
        config_obj = Config(load_config_files=False)
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    prelim_parser.add_argument("--plain", action="store_true", help="Use plain text output")
    prelim_args, _ = prelim_parser.parse_known_args()

    # Full parse using the potentially modified config
    args = parse_arguments(config_obj)

    # NOW set config flags based on the *final* parsed args
    config_obj.VERBOSE = args.verbose
    config_obj.PLAIN_OUTPUT = args.plain
    config_obj.NO_STREAM = args.no_stream
    config_obj.INTERACTIVE_MODE = not args.question and not args.command # Set interactive mode flag

    # Ensure logger level respects the final VERBOSE setting
    # (Add logging configuration here if not done elsewhere reliably)
    logging.basicConfig(level=logging.DEBUG if config_obj.VERBOSE else logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Suppress overly verbose logs from libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    if args.list_models:
        list_models(config_obj)
        sys.exit(0)
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
    if args.delete_model:
        success = False
        try:
            success = delete_model(args.delete_model, config_obj)
            if success:
                console.print(f"[green]Model alias '{args.delete_model}' deleted successfully.[/green]")
            # No explicit else needed, delete_model prints error on failure
        except Exception as e:
            console.print(f"[bold red]Error during delete model operation:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
    if args.refresh_models:
        success = False
        try:
            if args.refresh_models == 'openai':
                success = update_models_interactive(config_obj, provider='openai')
            elif args.refresh_models == 'ollama':
                success = update_models_interactive(config_obj, provider='ollama')
            if success:
                console.print(f"[green]Model list refresh for '{args.refresh_models}' completed.[/green]")
            else:
                console.print(f"[red]Model list refresh for '{args.refresh_models}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Model list refresh cancelled.[/bold red]")
            success = False # Ensure failure on interrupt
        except Exception as e:
            console.print(f"[bold red]Error during model refresh:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
    model_manager = ModelManager(config_obj)
    resolved_alias = model_manager.resolve_model_alias(args.model)
    if not resolved_alias:
        sys.exit(1)
    try:
        run_app(args, config_obj, resolved_alias)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during application execution:[/bold red] {e}")
        if config_obj.VERBOSE:
             traceback.print_exc()
        sys.exit(1)

def run_app(args: argparse.Namespace, config_obj: Config, resolved_alias: str):
    try:
        ask_llm = AskLLM(resolved_model_alias=resolved_alias, config=config_obj)
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

    # Use config object for flags
    stream_flag = not config_obj.NO_STREAM
    plaintext_flag = config_obj.PLAIN_OUTPUT

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        ask_llm.query(question_text.strip(), plaintext_output=plaintext_flag, stream=stream_flag)
    elif command_output_str:
        if config_obj.VERBOSE:
            console.print("Command output captured, querying LLM with it...", highlight=False)
        ask_llm.query(command_output_str.strip(), plaintext_output=plaintext_flag, stream=stream_flag)
    else:
        # Interactive mode
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


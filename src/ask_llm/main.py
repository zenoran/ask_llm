import argparse
import subprocess
from ask_llm.utils.history import HistoryManager
from ask_llm.utils.config import Config, config as global_config, is_huggingface_available
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.utils.ollama_utils import find_matching_model


# Import clients only after checking availability
from ask_llm.clients import OpenAIClient, OllamaClient


class AskLLM:
    def __init__(self):
        self.model_id = global_config.DEFAULT_MODEL
        self.client = self.initialize_client()
        self.history_manager = HistoryManager(client=self.client)
        self.load_history()

    def initialize_client(self):
        model_id = self.model_id
        
        # Define client map based on available dependencies
        client_map = {
            "ollama": OllamaClient,
            "openai": OpenAIClient,
        }
        
        # Only add HuggingFace client if dependencies are available
        hf_available = is_huggingface_available()
        if hf_available:
            # Import only when available to avoid ImportError
            from ask_llm.clients import HuggingFaceClient
            client_map["huggingface"] = HuggingFaceClient
        
        # Determine model type
        if model_id in global_config.HUGGINGFACE_MODELS:
            model_type = "huggingface"
            # Check if HuggingFace is available
            if not hf_available:
                self.client.console.print(
                    "[bold yellow]Warning: HuggingFace dependencies not installed.[/bold yellow]"
                ) if hasattr(self, 'client') else print(
                    "Warning: HuggingFace dependencies not installed."
                )
                self.client.console.print(
                    "[yellow]Install with: pip install ask_llm[huggingface][/yellow]"
                ) if hasattr(self, 'client') else print(
                    "Install with: pip install ask_llm[huggingface]"
                )
                # Fall back to default model
                model_id = global_config.DEFAULT_MODEL if model_id != global_config.DEFAULT_MODEL else "gpt-4o"
                # Re-determine model type
                if model_id in global_config.OLLAMA_MODELS:
                    model_type = "ollama"
                elif model_id in global_config.OPENAPI_MODELS:
                    model_type = "openai"
                else:
                    raise ValueError(f"Could not find a fallback model.")
        elif model_id in global_config.OLLAMA_MODELS:
            model_type = "ollama"
        elif model_id in global_config.OPENAPI_MODELS:
            model_type = "openai"
        else:
            raise ValueError(f"Unknown model specified in configuration: {model_id}")

        client_class = client_map.get(model_type)
        if client_class:
            return client_class(model_id)
        else:
            raise ValueError(f"Could not find a client for model: {model_id}")

    def load_history(self):
        self.history_manager.load_history()

    def query(self, prompt, plaintext_output: bool = False):
        try:
            self.history_manager.add_message("user", prompt)
            context_messages = (
                self.history_manager.get_context_messages_excluding_last()
            )
            response = self.client.query(
                context_messages, prompt, plaintext_output=plaintext_output
            )

            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response:
                self.client.console.print(
                    "[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]"
                )
                old_temp = global_config.TEMPERATURE
                global_config.TEMPERATURE = 0.9
                response = self.client.query(
                    context_messages, prompt, plaintext_output=plaintext_output
                )
                global_config.TEMPERATURE = old_temp

            self.history_manager.add_message("assistant", response)
            return response
        except KeyboardInterrupt:
            self.client.console.print("\n[bold red]Query interrupted.[/bold red]")


def validate_model(model_name: str, current_config: Config = global_config) -> str:
    """
    Validate model name using the provided config object.
    Returns the validated model name, or raises an ArgumentTypeError if invalid.
    """
    # Check if it's a HuggingFace model but dependencies aren't available
    if model_name in current_config.HUGGINGFACE_MODELS and not is_huggingface_available():
        print(f"Warning: Model '{model_name}' requires Hugging Face dependencies.")
        print("Install with: pip install ask_llm[huggingface]")
        print("Falling back to default model.")
        return current_config.DEFAULT_MODEL if current_config.DEFAULT_MODEL not in current_config.HUGGINGFACE_MODELS else "gpt-4o"
    
    if model_name in current_config.MODEL_OPTIONS:
        return model_name

    matched_model = find_matching_model(model_name, current_config.OLLAMA_MODELS)
    if matched_model:
        return matched_model

    valid_models_str = ", ".join(current_config.MODEL_OPTIONS)
    raise argparse.ArgumentTypeError(
        f"Invalid model: '{model_name}'. Available models: {valid_models_str}"
    )

class ModelValidator:
    def __init__(self, config_to_use: Config):
        self.config = config_to_use

    def __call__(self, model_name: str) -> str:
        return validate_model(model_name, current_config=self.config)

def parse_arguments(current_config: Config = global_config):
    parser = argparse.ArgumentParser(
        description="Query LLM models from the command line"
    )
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=current_config.DEFAULT_MODEL,
        type=ModelValidator(current_config),
        help="Choose the model to use (supports partial matching).",
    )
    parser.add_argument(
        "-dh",
        "--delete-history",
        action="store_true",
        help="Wipe any saved chat history and start fresh.",
    )
    parser.add_argument(
        "-ph",
        "--print-history",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="Print the saved conversation history. Optionally specify number of recent pairs.",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="Execute a shell command and add its output to the question.",
    )
    parser.add_argument(
        "--plain", action="store_true", help="Use plain text output (no formatting)"
    )
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        default=False,
        help="Refresh available models in cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments(current_config=global_config)

    global_config.update_from_args(args)

    ask_llm = AskLLM()

    if args.delete_history:
        ask_llm.history_manager.clear_history()

    if args.print_history is not None:
        ask_llm.history_manager.print_history(args.print_history)
        if not args.question:
            return
    ask_llm.client.console.print("")

    command_output_str = ""
    if args.command:
        ask_llm.client.console.print(
            f"Executing command: [yellow]{args.command}[/yellow]"
        )
        try:
            result = subprocess.run(
                args.command,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout.strip()
            error = result.stderr.strip()

            if output:
                command_output_str += f"Command Output:\n```\n{output}\n```\n\n"
            if error:
                command_output_str += f"Command Error:\n```\n{error}\n```\n\n"
            if result.returncode != 0:
                ask_llm.client.console.print(
                    f"[yellow]Warning: Command exited with status {result.returncode}[/yellow]"
                )
                command_output_str += (
                    f"(Command exited with status {result.returncode})\n\n"
                )

        except Exception as e:
            ask_llm.client.console.print(
                f"[bold red]Error executing command:[/bold red] {e}"
            )
            command_output_str += f"Error executing command: {e}\n\n"

        ask_llm.client.console.print()

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        ask_llm.query(question_text.strip(), plaintext_output=args.plain)
    elif command_output_str:
        ask_llm.client.console.print("Command output captured, querying LLM...")
        ask_llm.query(command_output_str.strip(), plaintext_output=args.plain)
    else:
        ask_llm.client.console.print(
            "[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]"
        )
        ask_llm.client.console.print(
            "[bold green]Type '>' at the beginning to enter multiline input mode.[/bold green]"
        )
        input_handler = MultilineInputHandler(console=ask_llm.client.console)

        while True:
            try:
                prompt_text, is_multiline = input_handler.get_input(
                    "Enter your question:"
                )

                if prompt_text.strip().lower() in ["exit", "quit"]:
                    ask_llm.client.console.print(
                        "[bold red]Exiting interactive mode.[/bold red]"
                    )
                    ask_llm.client.console.print()
                    break

                if not prompt_text.strip():
                    ask_llm.client.console.print(
                        "[dim]Empty input received. Asking again...[/dim]"
                    )
                    continue

                if is_multiline and prompt_text.strip():
                    prompt_text = input_handler.preview_input(prompt_text)

                ask_llm.client.console.print()
                if prompt_text.strip():
                    ask_llm.query(prompt_text, plaintext_output=args.plain)
                ask_llm.client.console.print()

            except (KeyboardInterrupt, EOFError):
                ask_llm.client.console.print(
                    "\n[bold red]Exiting interactive mode.[/bold red]"
                )
                ask_llm.client.console.print()
                break
    ask_llm.client.console.print()


if __name__ == "__main__":
    main()

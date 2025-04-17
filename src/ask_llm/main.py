import argparse
import subprocess
from ask_llm.utils.history import HistoryManager
from ask_llm.utils.config import Config, config as global_config
from ask_llm.clients import OpenAIClient, OllamaClient, HuggingFaceClient
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.utils.ollama_utils import find_matching_model


class AskLLM:
    def __init__(self, model_id=None):
        self.model_id = model_id or global_config.DEFAULT_MODEL
        self.client = self.initialize_client()
        self.history_manager = HistoryManager(client=self.client)
        self.load_history()

    def initialize_client(self):
        model_id = self.model_id

        client_map = {
            "huggingface": HuggingFaceClient,
            "ollama": OllamaClient,
            "openai": OpenAIClient,
        }

        if model_id in global_config.HUGGINGFACE_MODELS:
            model_type = "huggingface"
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

    def query(self, prompt, plaintext_output: bool = False, stream: bool = True):
        """Add user prompt, query client, handle retries, add assistant response."""
        try:
            # Add the current user prompt to the history first
            self.history_manager.add_message("user", prompt)
            # Get the complete context, including the message just added
            complete_context_messages = self.history_manager.get_context_messages()

            # Prepare kwargs for the client query using the complete context
            query_kwargs = {
                "messages": complete_context_messages,
                # "prompt": prompt, # REMOVED - prompt is now the last item in messages
                "plaintext_output": plaintext_output,
            }

            # Only add the stream argument if the client is HuggingFaceClient
            if isinstance(self.client, HuggingFaceClient):
                query_kwargs["stream"] = stream

            response = self.client.query(**query_kwargs)

            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response:
                self.client.console.print(
                    "[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]"
                )
                old_temp = global_config.TEMPERATURE
                global_config.TEMPERATURE = 0.9

                # Re-query with adjusted temperature (using the same complete context)
                if isinstance(self.client, HuggingFaceClient):
                     query_kwargs["stream"] = stream # Ensure stream kwarg is passed on retry
                response = self.client.query(**query_kwargs)
                global_config.TEMPERATURE = old_temp

            # Add the assistant response to history
            self.history_manager.add_message("assistant", response)
            return response
        except KeyboardInterrupt:
            self.client.console.print("\n[bold red]Query interrupted.[/bold red]")


def validate_model(model_name: str, current_config: Config = global_config) -> str:
    """
    Validate model name using the provided config object.
    (Docstring content remains the same)
    """
    if model_name in current_config.MODEL_OPTIONS:
        return model_name

    # Check OpenAI models for partial matches
    matched_model = find_matching_model(model_name, current_config.OPENAPI_MODELS)
    if matched_model:
        return matched_model

    # Check Ollama models for partial matches
    matched_model = find_matching_model(model_name, current_config.OLLAMA_MODELS)
    if matched_model:
        return matched_model
        
    # Check HuggingFace models for partial matches
    matched_model = find_matching_model(model_name, current_config.HUGGINGFACE_MODELS)
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
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output (only affects HuggingFace models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments(current_config=global_config)

    global_config.update_from_args(args)

    ask_llm = AskLLM(model_id=args.model)

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
        ask_llm.query(question_text.strip(), plaintext_output=args.plain, stream=(not args.no_stream))
    elif command_output_str:
        ask_llm.client.console.print("Command output captured, querying LLM...")
        ask_llm.query(command_output_str.strip(), plaintext_output=args.plain, stream=(not args.no_stream))
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
                    ask_llm.query(prompt_text, plaintext_output=args.plain, stream=(not args.no_stream))
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

import sys
import argparse
from rich.rule import Rule
from ask_llm.utils.history import HistoryManager
from ask_llm.utils.config import config
from ask_llm.clients import OpenAIClient, OllamaClient, HuggingFaceClient
from ask_llm.utils.input_handler import MultilineInputHandler
from ask_llm.utils.ollama_utils import get_available_models as get_models, find_matching_model

class AskLLM:
    def __init__(self):  # Removed the model parameter
        self.model_id = config.DEFAULT_MODEL  # Get model from global config
        self.client = self.initialize_client(self.model_id)
        self.history_manager = HistoryManager(client=self.client)
        self.load_history()
        

    def initialize_client(self, model_id):
        # Check HuggingFace models first
        if model_id in config.HUGGINGFACE_MODELS:
             return HuggingFaceClient(model_id)
        elif model_id in config.OLLAMA_MODELS:
            return OllamaClient(model_id)
        elif model_id in config.OPENAPI_MODELS:
            return OpenAIClient(model_id)
        else:
             # This case should ideally be caught by argparse validation, but good as a fallback
             raise ValueError(f"Unknown model specified in configuration: {model_id}")

    def load_history(self):
        self.history_manager.load_history()

    def query(self, prompt, plaintext_output: bool = False):
        try:
            self.history_manager.add_message("user", prompt)
            # Force a clean query each time
            self.client.console.print("[dim]Generating response...[/dim]")
            response = self.client.query(self.history_manager.get_context_messages(), prompt, plaintext_output=plaintext_output)
            
            # Verify we didn't get a duplicate response by checking history
            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response:
                # Try once more with increased randomness
                self.client.console.print("[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]")
                old_temp = config.TEMPERATURE
                config.TEMPERATURE = 0.9  # Temporarily increase temperature
                response = self.client.query(self.history_manager.get_context_messages(), prompt, plaintext_output=plaintext_output)
                config.TEMPERATURE = old_temp  # Reset to previous setting
            
            self.history_manager.add_message("assistant", response)
            return response
        except KeyboardInterrupt:
            self.client.console.print("\n[bold red]Query interrupted.[/bold red]")

def validate_model(model_name: str) -> str:
    """
    Validate model name and support partial matching.
    Used as a custom type function for argparse.
    
    Args:
        model_name: The model name to validate
        
    Returns:
        The matched model name if valid
        
    Raises:
        argparse.ArgumentTypeError: If the model name is invalid or ambiguous
    """
    # Check OpenAI models first (exact matches only)
    if model_name in config.OPENAPI_MODELS:
        return model_name
    
    # Check HuggingFace models (exact match for now)
    if model_name in config.HUGGINGFACE_MODELS:
         return model_name
    
    # Check Ollama models with partial matching
    matched_model = find_matching_model(model_name, config.OLLAMA_MODELS)
    if matched_model:
        return matched_model
    
    # Handle potential partial matches against OpenAI models
    openai_match = find_matching_model(model_name, config.OPENAPI_MODELS)
    if openai_match:
        return openai_match
        
    # Try partial matching for HF models if desired (similar to Ollama)
    hf_match = find_matching_model(model_name, config.HUGGINGFACE_MODELS)
    if hf_match:
         return hf_match
    
    # No matches at all
    valid_models = config.MODEL_OPTIONS
    raise argparse.ArgumentTypeError(
        f"Invalid model: '{model_name}'. Available models: {', '.join(valid_models)}"
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query LLM models from the command line")
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=config.DEFAULT_MODEL,
        type=validate_model,
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
        help="Print the saved conversation history. Optionally specify the number of recent conversation pairs to display. (Use -ph with no number for full history.)",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="Execute a shell command and add its output to the question.",
    )
    parser.add_argument("--plain", action="store_true", help="Use plain text output (no formatting)")
    parser.add_argument(
        "--refresh-models", 
        action="store_true", 
        default=False,
        help="Refresh available models in cache"
    )
    return parser.parse_args()

def main() -> None:

    args = parse_arguments()
    # Update the global config instance directly from args
    config.update_from_args(args)
    
    # Instantiate AskLLM without needing a model parameter
    ask_llm = AskLLM()
    
    if args.delete_history:
        ask_llm.history_manager.clear_history()
    
    if args.print_history is not None:
        ask_llm.history_manager.print_history(args.print_history)
        if not args.question:
            return
    ask_llm.client.console.print("")

    if args.question:
        question_text = " ".join(args.question)
        ask_llm.query(question_text, plaintext_output=args.plain)
    else:
        ask_llm.client.console.print("[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]")
        ask_llm.client.console.print("[bold green]Type '>' at the beginning to enter multiline input mode.[/bold green]")
        input_handler = MultilineInputHandler(console=ask_llm.client.console)
        
        while True:
            try:
                # Get input and whether it's multiline
                prompt_text, is_multiline = input_handler.get_input("Enter your question:")
                
                if prompt_text.strip().lower() in ["exit", "quit"]:
                    ask_llm.client.console.print("[bold red]Exiting interactive mode.[/bold red]")
                    ask_llm.client.console.print()
                    break

                if not prompt_text.strip():
                    ask_llm.client.console.print("[dim]Empty input received. Asking again...[/dim]")
                    continue

                # Only show preview for multiline input
                if is_multiline and prompt_text.strip():
                    prompt_text = input_handler.preview_input(prompt_text)
                
                # If we have text after potential preview, query the model
                ask_llm.client.console.print()
                if prompt_text.strip():
                    ask_llm.query(prompt_text, plaintext_output=args.plain)
                ask_llm.client.console.print()
                
            except (KeyboardInterrupt, EOFError):
                # This will catch Ctrl+C at any point in the loop
                ask_llm.client.console.print("\n[bold red]Exiting interactive mode.[/bold red]")
                ask_llm.client.console.print()
                break
    ask_llm.client.console.print()

if __name__ == "__main__":
    main()
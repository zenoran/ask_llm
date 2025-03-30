import argparse
from ask_llm.utils.history import HistoryManager
from ask_llm.utils.config import Config, config
from ask_llm.clients import OpenAIClient, OllamaClient
from ask_llm.utils.input_handler import MultilineInputHandler

class AskLLM:
    def __init__(self):  # Removed the model parameter
        self.model = config.DEFAULT_MODEL  # Get model from global config
        self.client = self.initialize_client(self.model)
        self.history_manager = HistoryManager(client=self.client)
        self.load_history()

    def initialize_client(self, model):
        if model in config.OLLAMA_MODELS:
            return OllamaClient(model)
        return OpenAIClient(model)

    def load_history(self):
        self.history_manager.load_history()

    def query(self, prompt):
        try:
            self.history_manager.add_message("user", prompt)
            response = self.client.query(self.history_manager.get_context_messages(), prompt)
            self.history_manager.add_message("assistant", response)
            return response
        except KeyboardInterrupt:
            self.client.console.print("\n[bold red]Query interrupted.[/bold red]")

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
        choices=config.MODEL_OPTIONS,
        help="Choose the model to use.",
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

    if args.question:
        question_text = " ".join(args.question)
        ask_llm.query(question_text)
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
                    break

                if not prompt_text.strip():
                    continue

                # Only show preview for multiline input
                if is_multiline and prompt_text.strip():
                    prompt_text = input_handler.preview_input(prompt_text)
                
                # If we have text after potential preview, query the model
                if prompt_text.strip():
                    ask_llm.query(prompt_text)
                
            except KeyboardInterrupt:
                # This will catch Ctrl+C at any point in the loop
                ask_llm.client.console.print("\n[bold red]Exiting interactive mode.[/bold red]")
                break

if __name__ == "__main__":
    main()
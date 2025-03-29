import argparse
from ask_llm.utils.history import HistoryManager
from ask_llm.utils.config import Config
from ask_llm.clients import OpenAIClient, OllamaClient

class AskLLM:

    def __init__(self, model):
        self.model = model
        self.client = self.initialize_client(model)
        self.history_manager = HistoryManager(client=self.client)
        self.load_history()

    def initialize_client(self, model):
        if model in Config.OLLAMA_MODELS:
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
        help="Print full JSON response in one-shot mode",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=Config.DEFAULT_MODEL,
        choices=Config.MODEL_OPTIONS,
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    ask_llm = AskLLM(model=args.model)

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
        while True:
            try:
                prompt_text = ask_llm.client.console.input("\n[bold blue]Question> [/bold blue]")
            except (KeyboardInterrupt, EOFError):
                ask_llm.client.console.print("\n[bold red]Exiting interactive mode.[/bold red]")
                break

            if prompt_text.strip().lower() in ["exit", "quit"]:
                ask_llm.client.console.print("[bold red]Exiting interactive mode.[/bold red]")
                break
            ask_llm.query(prompt_text)

if __name__ == "__main__":
    main()
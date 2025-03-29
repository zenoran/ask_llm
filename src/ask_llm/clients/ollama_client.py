import requests
import json
from rich.markdown import Markdown
from rich.rule import Rule
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import Config

class OllamaClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)

    def query(self, messages, prompt):
        ollama_messages = self.prepare_ollama_messages(prompt)
        response = self.send_query_to_ollama(ollama_messages)
        return self.extract_response_content(response)

    def get_system_message(self, messages):
        for message in messages:
            if message.role == "system":
                return {"role": "system", "content": message.content}
        return {"role": "system", "content": Config.SYSTEM_MESSAGE.strip()}

    def prepare_ollama_messages(self, prompt):
        return [{"role": "user", "content": prompt}]

    def send_query_to_ollama(self, ollama_messages):
        payload = {"model": self.model, "messages": ollama_messages, "stream": False}
        response = requests.post(f"{Config.OLLAMA_URL}/api/chat", json=payload)
        self.handle_response_errors(response)
        return response.json()

    def handle_response_errors(self, response):
        if response.status_code != 200:
            raise Exception(f"Error calling Ollama API: {response.status_code} - {response.text}")

    def extract_response_content(self, result):
        if "message" in result and result["message"]["role"] == "assistant":
            return result["message"]["content"]
        return "No response from Ollama"

    def get_verbose_output(self, messages, prompt):
        system_message = self.get_system_message(messages)
        ollama_messages = [system_message, {"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": ollama_messages, "stream": False}
        return json.dumps(payload, indent=2)

    def format_response(self, response_text):
        self.format_message("assistant", response_text)

    def format_message(self, role, content):
        if role == "user":
            self.print_user_message(content)
        elif role == "assistant":
            self.print_assistant_message(content)

    def print_user_message(self, content):
        self.console.print()
        self.console.print("[bold blue]User:[/bold blue]")
        self.console.print(Markdown(content))

    def print_assistant_message(self, content):
        self.console.print()
        self.console.print(Rule(style="#222222"))
        self.console.print()
        self.console.print(Markdown(content))
        self.console.print()
        self.console.print(Rule(style="#222222"))
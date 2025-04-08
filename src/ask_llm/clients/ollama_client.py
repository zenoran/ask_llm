import json
import requests
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import config
from typing import List
from ask_llm.utils.ollama_utils import get_available_models as get_models


class OllamaClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)

    def query(self, messages, prompt):
        api_messages = self._prepare_api_messages(messages, prompt)
        response = self._stream_response(api_messages)
        return response

    def _prepare_api_messages(self, messages, prompt):
        api_messages = [msg.to_api_format() for msg in messages if msg.role != "system"]
        if "system" not in api_messages:
            api_messages.insert(0, {"role": "system", "content": config.SYSTEM_MESSAGE.replace("\n", "")})

        return api_messages

    def _stream_response(self, api_messages):
        """Stream the response with live updating display."""
        total_response = ""
        accumulated_buffer = (
            ""  # Buffer to collect chunks until we have a complete first paragraph
        )
        total_thought = ""
        thought_buffer = ""
        first_para_rendered = False
        in_thought = False
        live_display = Live(auto_refresh=True, console=self.console)
        if config.VERBOSE:
            self.console.print("[bold blue]Verbose Output:[/bold blue]")
            self.console.print(api_messages)

        try:
            api_request = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={"model": self.model, "messages": api_messages, "stream": True},
                stream=True,
            )
            
            # Check for HTTP errors
            api_request.raise_for_status()
            
            # Start by collecting the thought or first paragraph completely separate from the rest
            for chunk in api_request.iter_lines():
                if not chunk:
                    continue
                try:
                    chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                    
                # Check for error in response
                if "error" in chunk:
                    error_msg = chunk["error"]
                    if "GGML_ASSERT" in error_msg:
                        self.console.print(f"[bold red]Model Compatibility Error:[/bold red] {error_msg}")
                        self.console.print("[yellow]This is likely due to a compatibility issue between your model and Ollama.[/yellow]")
                        self.console.print("Try using a different quantization format for your model.")
                        return f"ERROR: {error_msg}"
                    else:
                        self.console.print(f"[bold red]Ollama Error:[/bold red] {error_msg}")
                        return f"ERROR: {error_msg}"
                        
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if not content:
                        continue

                    accumulated_buffer += content
                    total_response += content

                    if live_display.is_started:
                        live_display.update(Markdown(accumulated_buffer), refresh=True)
                        continue

                    if first_para_rendered is False:
                        if "<thought>" in accumulated_buffer:
                            in_thought = True
                            buffer_parts = str(accumulated_buffer).partition("<thought>")
                            thought_buffer += buffer_parts[2].replace("\n", "")
                            accumulated_buffer = buffer_parts[2]
                            self.console.print("[blue]Thinking...[/blue]")
                            continue
                        elif "</thought>" in accumulated_buffer:
                            in_thought = False
                            buffer_parts = str(accumulated_buffer).partition("</thought>")
                            accumulated_buffer = total_response = buffer_parts[2]
                            total_thought += buffer_parts[0]
                            if config.VERBOSE:
                                self.console.print(
                                    f"[#555555]{total_thought.replace('\n\n', '')}[/#555555]"
                                )
                            thought_buffer =  content = ""
                        elif in_thought:
                            thought_buffer += content
                            continue

                    if first_para_rendered is False and "\n\n" in accumulated_buffer:
                        buffer_parts = str(accumulated_buffer).partition("\n\n")
                        if not buffer_parts[0].strip():
                            accumulated_buffer = buffer_parts[2]
                            continue
                        self._print_assistant_message(buffer_parts[0])
                        first_para_rendered = True
                        accumulated_buffer = content = buffer_parts[2]
                        live_display.start()
                        
        except requests.exceptions.HTTPError as http_err:
            self.console.print(f"[bold red]HTTP Error:[/bold red] {http_err}")
            return f"ERROR: {http_err}"
        except requests.exceptions.ConnectionError:
            self.console.print("[bold red]Connection Error:[/bold red] Could not connect to Ollama server.")
            self.console.print("Make sure Ollama is running and accessible at: " + config.OLLAMA_URL)
            return "ERROR: Could not connect to Ollama server"
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return f"ERROR: {str(e)}"
        finally:
            if live_display.is_started:
                live_display.stop()
                
        if not first_para_rendered:
            self._print_assistant_message(accumulated_buffer)

        return total_response

    def format_response(self, response_text):
        """Format OpenAI response for display with the first paragraph in a box"""
        self.format_message("assistant", response_text)

    def format_message(self, role, content):
        """Format a message based on its role"""
        if role == "user":
            self._print_user_message(content)
        elif role == "assistant":
            self._print_assistant_message(content)

    def _print_user_message(self, content):
        self.console.print()
        self.console.print("[bold blue]User:[/bold blue] ", end="")
        self.console.print(Markdown(content))

    def _print_assistant_message(self, content):
        """Format the assistant message with a panel for the first paragraph."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if paragraphs:
            boxed_response = paragraphs[0]
            extra_response = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
        else:
            boxed_response = content.strip()
            extra_response = ""

        assistant_panel = Panel(
            Markdown(boxed_response),
            title=f"[bold green]{self.model}[/bold green]",
            border_style="green",
            padding=(1, 4),
        )

        self.console.print(Align(assistant_panel, align="right"))
        self.console.print()

        if extra_response:
            self.console.print(Markdown(extra_response))

    def _print_buffer(self, buffer):
        """Print buffered lines to the console."""
        for line in buffer:
            self.console.print(line)

    @staticmethod
    def get_available_models(base_url: str = None) -> List[str]:
        """
        Query Ollama API for available models.
        
        Args:
            base_url: The base URL of the Ollama API (default: from config)
            
        Returns:
            List of available model names, or empty list if API unreachable
        """
        if base_url is None:
            from ask_llm.utils.config import config
            base_url = config.OLLAMA_URL
            
        return get_models(base_url)

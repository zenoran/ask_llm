import re
import json
import os
from openai import OpenAI
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.align import Align
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import Config


class OpenAIClient(LLMClient):
    """Client for OpenAI API"""

    def __init__(self, model):
        super().__init__(model)
        self.api_key = self._get_api_key()
        self.client = OpenAI()
        self.client.api_key = self.api_key

    def _get_api_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Please set your OPENAI_API_KEY environment variable.")
        return api_key

    def query(self, messages, prompt):
        """Query OpenAI API with full message history, using streaming by default"""
        api_messages = self._prepare_api_messages(messages, prompt)
        response = self._stream_response(api_messages)
        return response

    def _prepare_api_messages(self, messages, prompt):
        api_messages = [msg.to_api_format() for msg in messages]
        api_messages.append({"role": "user", "content": prompt})
        return api_messages

    def _stream_response(self, api_messages):
        """Stream the response with incremental printing, ensuring the first paragraph is boxed."""
        response = ""
        buffer = ""
        in_code_block = False
        code_fence_count = 0
        first_paragraph = None

        api_request = self.client.chat.completions.create(
            model=self.model, messages=api_messages, stream=True, store=False
        )

        try:
            for chunk in api_request:
                content = chunk.choices[0].delta.content
                if not content:
                    continue

                response += content

                if first_paragraph is None and "\n\n" in content:
                    self._print_assistant_message(response)
                    partitioned = response.partition("\n\n")
                    first_paragraph = partitioned[0]
                    response = partitioned[2]
                    buffer = ""
                    continue

                buffer += content

                for char in content:
                    if Config.PRESERVE_CODE_BLOCKS and char == "`":
                        code_fence_count += 1
                    else:
                        if code_fence_count == 3:
                            in_code_block = not in_code_block
                        code_fence_count = 0

                if in_code_block:
                    continue

                if "\n\n" in buffer:
                    list_match = re.search(
                        r"(\d+\.\s+[^\n]+(?:\n\s+[^\n]+)*(?:\n\n\d+\.\s+[^\n]+(?:\n\s+[^\n]+)*)*)",
                        buffer,
                    )

                    if list_match:
                        end_pos = list_match.end()

                        if end_pos < len(buffer) and buffer[end_pos:].strip():
                            after_list = buffer[end_pos:]
                            next_line = (
                                after_list.strip().split("\n")[0]
                                if "\n" in after_list.strip()
                                else after_list.strip()
                            )

                            if (
                                re.match(r"^\d+\.", next_line)
                                or re.match(r"^\s+-", next_line)
                                or next_line.startswith("    ")
                            ):
                                continue

                            print("\r" + " " * 80, end="\r", flush=True)
                            self.console.print(Markdown(buffer[:end_pos]))
                            buffer = buffer[end_pos:]
                    else:
                        parts = buffer.split("\n\n", 1)
                        print("\r" + " " * 80, end="\r", flush=True)
                        self.console.print(Markdown(parts[0] + "\n\n"))
                        buffer = parts[1] if len(parts) > 1 else ""

        except KeyboardInterrupt:
            self.console.print(
                "\n[bold yellow]Interrupted! Stopping response collection...[/bold yellow]"
            )
            raise

        if first_paragraph is None:
            self._print_assistant_message(response)
            return response

        if buffer.strip():
            self.console.print(Markdown(buffer))

        return first_paragraph + "\n\n" + response

    def get_verbose_output(self, messages, prompt):
        """Get full API response for verbose output"""
        api_messages = []
        for message in messages:
            api_messages.append(message.to_api_format())

        api_messages.append({"role": "user", "content": prompt})

        result = self.client.chat.completions.create(
            model=self.model, messages=api_messages, store=False
        )
        return json.dumps(result, indent=2)

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
        self.console.print("[bold blue]User:[/bold blue]")
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

        self.console.print(Rule(style="#777777"))
        self.console.print()
        self.console.print(Align(assistant_panel, align="right"))
        self.console.print()

        if extra_response:
            self.console.print(Markdown(extra_response))

    def _print_buffer(self, buffer):
        """Print buffered lines to the console."""
        for line in buffer:
            self.console.print(line)

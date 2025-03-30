import json
import os
from openai import OpenAI
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.align import Align
from rich.live import Live
from ask_llm.clients.base import LLMClient


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
        """Stream the response with live updating display."""
        total_response = ""
        accumulated_buffer = ""  # Buffer to collect chunks until we have a complete first paragraph
        first_para_rendered = False
        second_display_started = False
        
        # Add a header rule
        self.console.print()
        
        api_request = self.client.chat.completions.create(
            model=self.model, messages=api_messages, stream=True, store=False
        )
        
        # Start by collecting the first paragraph completely separate from the rest
        for chunk in api_request:
            content = chunk.choices[0].delta.content
            if not content:
                continue
                
            accumulated_buffer += content
            total_response += content
            
            # Check if we have a complete first paragraph
            if not first_para_rendered and "\n\n" in accumulated_buffer:
                # We found a paragraph boundary
                split_at = accumulated_buffer.find("\n\n")
                first_para = accumulated_buffer[:split_at].strip()
                remaining = accumulated_buffer[split_at + 2:]
                
                # Display the first paragraph in a panel
                assistant_panel = Panel(
                    Markdown(first_para),
                    title=f"[bold green]{self.model}[/bold green]",
                    border_style="green",
                    padding=(1, 4),
                )
                self.console.print(Align(assistant_panel, align="right"))
                self.console.print()
                
                first_para_rendered = True
                
                # Now start fresh with a separate Live display for everything after the first paragraph
                if remaining.strip():
                    # Start with what we already have after the first paragraph
                    second_display = Live(Markdown(remaining), auto_refresh=True, console=self.console)
                    second_display.start()
                    second_display_started = True
                break
        
        # If we found a paragraph break, continue with remaining content in a new Live context
        if first_para_rendered:
            try:
                # Create a completely new Live context for just the remaining text
                remaining_text = ""
                if second_display_started:
                    remaining_text = accumulated_buffer[split_at + 2:]
                else:
                    # We didn't have any remaining content yet, so start a fresh Live
                    second_display = Live("", auto_refresh=True, console=self.console)
                    second_display.start()
                    second_display_started = True
                
                # Continue processing the rest of the chunks
                for chunk in api_request:
                    content = chunk.choices[0].delta.content
                    if not content:
                        continue
                        
                    total_response += content
                    remaining_text += content
                    
                    if remaining_text.strip():
                        second_display.update(Markdown(remaining_text))
                        
                if second_display_started:
                    second_display.stop()
                    
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Interrupted![/bold yellow]")
                if second_display_started:
                    second_display.stop()
                raise
        else:
            # We never found a paragraph break, just display everything as a whole
            self._print_assistant_message(total_response)
        
        self.console.print()
        return total_response

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

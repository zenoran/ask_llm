import json
import requests
from ask_llm.clients.base import LLMClient
from ask_llm.utils.config import config
from typing import List
from ask_llm.utils.ollama_utils import get_available_models as get_models
import sys


class OllamaClient(LLMClient):
    def __init__(self, model):
        super().__init__(model)

    def query(self, messages, plaintext_output: bool = False):
        """Query Ollama API with full message history, using streaming by default.

        Args:
            messages: List of message dictionaries (including the latest user prompt).
            plaintext_output: If True, return raw text. Otherwise, format output.
        """
        api_messages = self._prepare_api_messages(messages)
        response = self._stream_response(api_messages, plaintext_output)
        return response

    def _prepare_api_messages(self, messages):
        # Convert messages, ensuring system message is present and user prompt is last
        api_messages = []
        has_system_message = False
        for msg in messages:
            if hasattr(msg, 'to_api_format'):
                formatted_msg = msg.to_api_format()
            else:
                # Fallback: assume msg has .role and .content attributes
                try:
                    formatted_msg = {"role": msg.role, "content": msg.content}
                except AttributeError:
                    self.console.print(f"[bold red]Error:[/bold red] Message object missing 'role' or 'content' attribute: {msg}")
                    # Skip this message or handle error appropriately
                    continue 

            if formatted_msg['role'] == 'system':
                has_system_message = True
                # Ollama expects the system message first (if provided)
                api_messages.insert(0, formatted_msg)
            else:
                api_messages.append(formatted_msg)

        # If no system message was found in the input, add the default one
        if not has_system_message:
            api_messages.insert(0, {"role": "system", "content": config.SYSTEM_MESSAGE.replace("\n", "")})
        
        # Ensure the final list doesn't contain duplicate consecutive roles (Ollama requirement)
        # And ensure the final message is from the user
        # This is a basic cleanup; more robust history management might be needed
        cleaned_messages = []
        if api_messages:
            cleaned_messages.append(api_messages[0])
            for i in range(1, len(api_messages)):
                if api_messages[i]['role'] != api_messages[i-1]['role']:
                    cleaned_messages.append(api_messages[i])
            # Ensure the very last message is 'user'
            if cleaned_messages[-1]['role'] != 'user':
                 # This might happen if the last message added was assistant or system somehow
                 # Find the last user message and append it again if necessary, or handle error
                 # For now, we assume the input `messages` had the user prompt last correctly
                 pass # Revisit if issues persist

        # Add verbose logging to see the final payload being prepared
        # Filter out previous assistant messages that were just error strings
        final_messages = [msg for msg in cleaned_messages if not (msg.get('role') == 'assistant' and msg.get('content', '').startswith('ERROR:'))]

        return final_messages

    def _stream_response(self, api_messages, plaintext_output: bool = False):
        if config.VERBOSE:
            self.console.print("[bold blue]Verbose Output:[/bold blue]")
            self.console.print_json(json.dumps(api_messages))

        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={"model": self.model, "messages": api_messages, "stream": True},
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            self.console.print(f"[bold red]HTTP Error:[/bold red] {http_err}")
            return f"ERROR: {http_err}"
        except requests.exceptions.ConnectionError:
            self.console.print("[bold red]Connection Error:[/bold red] Could not connect to Ollama server.")
            self.console.print("Make sure Ollama is running and accessible at: " + config.OLLAMA_URL)
            return "ERROR: Could not connect to Ollama server"
        except Exception as e:
            self.console.print(f"[bold red]Initial Request Error:[/bold red] {str(e)}")
            return f"ERROR: {str(e)}"

        def _iterate_ollama_chunks(http_response):
            total_thought = ""
            in_thought = False
            thought_buffer = ""
            try:
                for line in http_response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        self.console.print(f"[yellow]Warning: Could not decode JSON line: {line}[/yellow]")
                        continue

                    # Check for error in response chunk
                    if "error" in chunk:
                        error_msg = chunk["error"]
                        if "GGML_ASSERT" in error_msg:
                            self.console.print(f"[bold red]Model Compatibility Error:[/bold red] {error_msg}")
                            self.console.print("[yellow]This is likely due to a compatibility issue between your model and Ollama.[/yellow]")
                            self.console.print("Try using a different quantization format for your model.")
                        else:
                            self.console.print(f"[bold red]Ollama Error:[/bold red] {error_msg}")
                        yield f"ERROR: {error_msg}" # Yield error to be included in output
                        return # Stop iteration on error

                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if not content:
                            continue
                            
                        # Handle <thought> tags if not in plaintext mode
                        if not plaintext_output:
                            current_content_part = content
                            processed_content = ""
                            while current_content_part:
                                if not in_thought:
                                    if "<thought>" in current_content_part:
                                        in_thought = True
                                        parts = current_content_part.partition("<thought>")
                                        processed_content += parts[0] # Add text before thought
                                        thought_buffer += parts[2].replace("\n", "")
                                        current_content_part = "" # Processed this chunk part
                                        if parts[0].strip(): # If there was text before thought, yield it
                                             yield parts[0]
                                        self.console.print("[blue i]Thinking...[/blue i]") # Indicate thought started
                                    else:
                                        processed_content += current_content_part
                                        current_content_part = ""
                                else: # in_thought is True
                                    if "</thought>" in current_content_part:
                                        in_thought = False
                                        parts = current_content_part.partition("</thought>")
                                        thought_buffer += parts[0].replace("\n", "")
                                        current_content_part = parts[2] # Process text after thought
                                        total_thought += thought_buffer
                                        if config.VERBOSE:
                                            self.console.print(f"[#555555 i]Thought: {total_thought.strip()}[/#555555 i]")
                                        thought_buffer = ""
                                    else:
                                        thought_buffer += current_content_part
                                        current_content_part = ""
                            
                            # Yield any processed non-thought content from this chunk
                            if processed_content:
                                yield processed_content
                        else:
                             # Plaintext mode, yield content directly
                             yield content

            except Exception as e:
                 self.console.print(f"\n[bold red]Error during Ollama stream processing:[/bold red] {str(e)}")
                 yield f"\nERROR: {str(e)}"
            finally:
                http_response.close() # Ensure the connection is closed

        # Pass the generator to the base handler
        return self._handle_streaming_output(
            stream_iterator=_iterate_ollama_chunks(response),
            plaintext_output=plaintext_output,
             # Keep Ollama default panel style (green)
            first_para_panel=True
        )

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

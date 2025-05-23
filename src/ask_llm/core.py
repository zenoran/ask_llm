import importlib.util
import pathlib
import time
import uuid
from rich.console import Console
from .utils.config import Config, is_huggingface_available, is_llama_cpp_available
from .utils.history import HistoryManager, Message
from .clients import LLMClient
import logging
from .utils.prompts import SYSTEM_REFINE_PROMPT

try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    _tiktoken_present = False
    tiktoken = None # Placeholder

try:
    from .memory import MemoryManager
    _memory_module_present = True
except ImportError:
    _memory_module_present = False
    MemoryManager = None # Placeholder if memory.py doesn't exist or fails import


console = Console()
logger = logging.getLogger(__name__)

class AskLLM:
    def __init__(self, resolved_model_alias: str, config: Config, memory_enabled: bool = False):
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        self.model_definition = self.config.defined_models.get("models", {}).get(resolved_model_alias)
        self.memory_enabled = memory_enabled
        self.memory_manager: MemoryManager | None = None # type: ignore

        if not self.model_definition:
             raise ValueError(f"Could not find model definition for resolved alias: '{resolved_model_alias}'")

        self.client: LLMClient = self.initialize_client(config)
        self.history_manager = HistoryManager(client=self.client, config=self.config)

        # Conditionally initialize MemoryManager
        if self.memory_enabled:
            if _memory_module_present and MemoryManager:
                try:
                    logger.debug("Memory flag enabled, attempting to initialize MemoryManager...")
                    # Pass the config object to MemoryManager
                    self.memory_manager = MemoryManager(config=self.config)
                except ImportError as e:
                    logger.debug(f"Failed to initialize MemoryManager due to missing dependencies: {e}")
                    if self.config.VERBOSE:
                        console.print(f"[bold yellow]Warning:[/bold yellow] Memory enabled but failed to initialize: {e}")
                        console.print("  Memory features will be disabled. Ensure 'chromadb' and 'sentence-transformers' are installed.")
                    self.memory_manager = None # Ensure it's None on failure
                except Exception as e:
                    logger.exception(f"An unexpected error occurred during MemoryManager initialization: {e}")
                    if self.config.VERBOSE:
                        console.print(f"[bold red]Error:[/bold red] Failed to initialize MemoryManager: {e}")
                        console.print("  Memory features will be disabled.")
                    self.memory_manager = None # Ensure it's None on failure
            else:
                 logger.debug("Memory flag enabled, but MemoryManager module or class could not be imported.")
                 if self.config.VERBOSE:
                      console.print("[bold yellow]Warning:[/bold yellow] Memory enabled, but memory module failed to load. Memory features disabled.")
                 self.memory_manager = None
        else:
            logger.debug("Memory flag is disabled.")


        self.load_history()

    def initialize_client(self, config: Config) -> LLMClient:
        model_type = self.model_definition.get("type")
        model_alias = self.resolved_model_alias

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Initializing client for model: {model_alias} (Type: {model_type})")

        if model_type == "gguf":
            # Check dependencies lazily
            if importlib.util.find_spec("huggingface_hub") is None:
                 console.print("[bold red]Error:[/bold red] `huggingface-hub` is required for GGUF models.")
                 raise ImportError("huggingface-hub not found for GGUF download.")
            if not is_llama_cpp_available(): # Check lazily
                console.print("[bold red]Error:[/bold red] `llama-cpp-python` is required for GGUF models.")
                raise ImportError("`llama-cpp-python` not installed or failed to import.")
            try:
                from .clients.llama_cpp_client import LlamaCppClient
                return self._initialize_llama_cpp_client(self.model_definition, config)
            except ImportError as e:
                 console.print(f"[bold red]Import Error for GGUF:[/bold red] {e}")
                 console.print("Ensure `llama-cpp-python` and `huggingface-hub` are installed.")
                 raise
            except Exception as e:
                console.print(f"[bold red]Error initializing Llama.cpp client:[/bold red] {e}")
                logger.exception(f"Error initializing Llama.cpp client: {e}")
                raise

        elif model_type == "huggingface":
             # Check dependencies lazily
             if not is_huggingface_available(): # Check lazily
                 console.print("[bold red]Error:[/bold red] Hugging Face dependencies are required for model type 'huggingface'. Ensure `transformers` and `torch` (or `tensorflow`/`jax`) are installed.")
                 raise ImportError("HuggingFace dependencies unavailable.")
             try:
                 from .clients.huggingface_client import HuggingFaceClient
                 model_id = self.model_definition.get("model_id")
                 if not model_id:
                     raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                 return HuggingFaceClient(model_id=model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing HuggingFace client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing HuggingFace client for {model_alias}: {e}")
                raise

        elif model_type == "ollama":
            try:
                from .clients.ollama_client import OllamaClient
                model_id = self.model_definition.get("model_id")
                if not model_id:
                    raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                if model_id not in self.config.available_ollama_models:
                     console.print(f"[yellow]Warning:[/yellow] Ollama model '{model_id}' (alias: '{model_alias}') not found on server {self.config.OLLAMA_URL}.")
                     console.print(f"  Attempting to use anyway, but may fail. Pull it with: `ollama pull {model_id}`")
                return OllamaClient(model=model_id, config=config)
            except Exception as e:
                console.print(f"[bold red]Error initializing Ollama client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing Ollama client for {model_alias}: {e}")
                raise

        elif model_type == "openai":
             try:
                 from .clients.openai_client import OpenAIClient
                 model_id = self.model_definition.get("model_id")
                 if not model_id:
                     raise ValueError(f"Missing 'model_id' in definition for alias '{model_alias}'")
                 return OpenAIClient(model_id, config=config)
             except Exception as e:
                console.print(f"[bold red]Error initializing OpenAI client for {model_alias}:[/bold red] {e}")
                logger.exception(f"Error initializing OpenAI client for {model_alias}: {e}")
                raise
        else:
            raise ImportError(f"Unsupported model type '{model_type}' defined for alias '{model_alias}' in {self.config.MODELS_CONFIG_PATH}")

    def _initialize_llama_cpp_client(self, model_def: dict, config: Config):
        # Re-import necessary hf_hub functions if needed, as the check is now local
        # Though, the check in initialize_client should prevent reaching here if unavailable
        if importlib.util.find_spec("huggingface_hub"):
             from huggingface_hub import hf_hub_download
             from huggingface_hub.errors import HfHubHTTPError
        else:
             # This state should ideally not be reached due to the check in initialize_client
             raise ImportError("huggingface-hub is required but not found for GGUF download within _initialize_llama_cpp_client.")

        from .clients.llama_cpp_client import LlamaCppClient
        repo_id = model_def.get("repo_id")
        filename = model_def.get("filename")
        alias = self.resolved_model_alias

        if not repo_id or not filename:
            raise ImportError(f"GGUF model definition for alias '{alias}' is missing 'repo_id' or 'filename' in {config.MODELS_CONFIG_PATH}")

        cache_dir = pathlib.Path(config.MODEL_CACHE_DIR).expanduser()
        model_repo_cache_dir = cache_dir / repo_id
        local_model_path = model_repo_cache_dir / filename
        model_path_to_load = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loading GGUF model: {alias} (Source: {repo_id}/{filename})")
            logger.debug(f"Expected model cache path: {local_model_path}")

        if local_model_path.is_file():
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug("Found cached GGUF model file.")
            model_path_to_load = str(local_model_path)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug("GGUF model file not found in cache. Downloading...")
            model_repo_cache_dir.mkdir(parents=True, exist_ok=True)
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Downloading '{filename}' from repo '{repo_id}'...")
            try:
                downloaded_path_str = hf_hub_download(repo_id=repo_id,filename=filename,local_dir=str(model_repo_cache_dir),)
                if pathlib.Path(downloaded_path_str) != local_model_path:
                     console.print(f"[yellow]Warning:[/yellow] Download path {downloaded_path_str} differs from expected cache path {local_model_path}. Using downloaded path.")
                model_path_to_load = downloaded_path_str
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"GGUF download complete: {model_path_to_load}")
            except HfHubHTTPError as e:
                 console.print(f"[bold red]Error accessing or downloading from Hugging Face repository '{repo_id}':[/bold red] {e}")
                 console.print("Please ensure the repository ID and filename are correct and you have internet access.")
                 raise
            except Exception as e:
                console.print(f"[bold red]Error downloading file '{filename}':[/bold red] {e}")
                logger.exception(f"Error downloading file '{filename}': {e}")
                raise

        try:
            client = LlamaCppClient(model_path=model_path_to_load, config=config)
            return client
        except ImportError:
             console.print("[bold red]Error:[/bold red] Failed to initialize LlamaCppClient. Is `llama-cpp-python` installed correctly?")
             raise
        except Exception as e:
            console.print(f"[bold red]Error initializing LlamaCppClient with {model_path_to_load}:[/bold red] {e}")
            logger.exception(f"Error initializing LlamaCppClient with {model_path_to_load}: {e}")
            raise

    def load_history(self, since_minutes: int | None= None) -> list[dict]:
        if since_minutes is None:
            since_minutes = self.config.HISTORY_DURATION
        self.history_manager.load_history(since_minutes=since_minutes)
        return [msg.to_dict() for msg in self.history_manager.messages]

    def _prepare_initial_prompt_and_history(self, prompt: str) -> tuple[Message | None, list[Message]]:
        """Adds user prompt to history, gets context, and separates system prompt."""
        self.history_manager.add_message("user", prompt)
        complete_context_messages = self.history_manager.get_context_messages()

        system_prompt_message: Message | None = None
        processing_messages = list(complete_context_messages) # History turns

        if processing_messages and processing_messages[0].role == "system":
            system_prompt_message = processing_messages.pop(0)
            logger.debug(f"Extracted system prompt: '{system_prompt_message.content[:60]}...'")
        else:
            logger.warning("Could not find leading system prompt in history messages.")
        return system_prompt_message, processing_messages

    def _retrieve_and_prepare_memory(self, prompt: str, history_turns: list[Message]) -> list[Message]:
        """Retrieves relevant memories and prepares them (deduplication)."""
        if not self.memory_manager:
            return []

        retrieved_memories_raw = None
        try:
            logger.debug(f"Attempting to retrieve relevant memories for prompt: '{prompt[:50]}...'")
            n_results = self.config.MEMORY_N_RESULTS
            retrieved_memories_raw = self.memory_manager.search_relevant_memories(prompt, n_results=n_results)
            if retrieved_memories_raw:
                logger.debug(f"Retrieved {len(retrieved_memories_raw)} memories (max {n_results}):")
                for i, mem in enumerate(retrieved_memories_raw):
                    logger.debug(f"  Mem {i+1}: ID={mem.get('id')}, Role={mem.get('metadata',{}).get('role')}, Dist={mem.get('distance'):.4f}, Content='{mem.get('document', '')[:60]}...'")
            else:
                logger.debug("No relevant memories found or search failed.")
        except Exception as e:
            logger.exception(f"Error during memory retrieval: {e}")
            return []

        if not retrieved_memories_raw:
            return []

        # Convert retrieved dicts to Message objects, sorted by relevance
        sorted_memories = sorted(retrieved_memories_raw, key=lambda x: x.get('distance', 0.0))
        memory_messages = [
            Message(role=mem.get('metadata', {}).get('role', 'assistant'), content=mem.get('document', ''))
            for mem in sorted_memories if mem.get('document')
        ]

        # Deduplicate against history turns
        history_content_set = {msg.content for msg in history_turns}
        unique_memory_messages = [
            mem_msg for mem_msg in memory_messages if mem_msg.content not in history_content_set
        ]
        return unique_memory_messages

    def _combine_memory_and_history(self, unique_memory_messages: list[Message], history_turns: list[Message]) -> tuple[list[Message], bool]:
        """Combines unique memories with history turns, adding a delimiter if memories exist."""
        final_combined_turns = list(history_turns)
        memory_delimiter_added = False

        if unique_memory_messages:
            logger.debug(f"Adding {len(unique_memory_messages)} unique memories to context.")
            memory_delimiter = Message(role="system", content="--- Relevant Past Conversation Snippets (older history or related topics) ---")
            final_combined_turns = unique_memory_messages + [memory_delimiter] + final_combined_turns
            memory_delimiter_added = True
        else:
            logger.debug("No unique memories to combine after deduplication.")
        
        return final_combined_turns, memory_delimiter_added

    def _truncate_messages_if_needed(self,
                                   messages_to_truncate: list[Message],
                                   system_prompt_message: Message | None,
                                   memory_delimiter_added: bool) -> list[Message]:
        """Counts tokens and truncates the combined messages if they exceed the token limit."""
        if not (_tiktoken_present and tiktoken):
            if self.memory_manager: # Only warn if memory (which relies on this) is on
                 logger.warning("`tiktoken` library not found. Skipping context token counting and truncation. Install with `pip install tiktoken`")
            return messages_to_truncate # Return as is if no tiktoken

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            TOKENS_PER_MESSAGE_OVERHEAD = 4
            SYSTEM_PROMPT_EXTRA_OVERHEAD = 3

            system_prompt_tokens = 0
            if system_prompt_message:
                system_prompt_tokens = system_prompt_message.get_token_count(
                    encoding, TOKENS_PER_MESSAGE_OVERHEAD
                ) + SYSTEM_PROMPT_EXTRA_OVERHEAD

            max_tokens_for_turns = self.config.MAX_TOKENS - system_prompt_tokens
            if max_tokens_for_turns < 0: max_tokens_for_turns = 0

            initial_token_count_turns = sum(
                msg.get_token_count(encoding, TOKENS_PER_MESSAGE_OVERHEAD) for msg in messages_to_truncate
            )
            logger.debug(f"Combined memory/history turns: {len(messages_to_truncate)} messages, ~{initial_token_count_turns} tokens. System prompt: ~{system_prompt_tokens} tokens.")
            logger.debug(f"Max tokens for turns: {max_tokens_for_turns} (Total limit: {self.config.MAX_TOKENS})")

            if initial_token_count_turns <= max_tokens_for_turns:
                logger.debug(f"Turns context is within token limits ({initial_token_count_turns} <= {max_tokens_for_turns}). No truncation needed.")
                return messages_to_truncate

            logger.warning(f"Combined turns context ({initial_token_count_turns} tokens) exceeds limit for turns ({max_tokens_for_turns}). Truncating...")
            
            truncated_turns = list(messages_to_truncate) # Work on a copy
            current_token_count_turns = initial_token_count_turns
            
            delimiter_index = -1
            if memory_delimiter_added:
                memory_delimiter_content = "--- Relevant Past Conversation Snippets (older history or related topics) ---"
                for i, msg in enumerate(truncated_turns):
                    if msg.role == "system" and msg.content == memory_delimiter_content:
                        delimiter_index = i
                        break
            
            num_memory_messages = delimiter_index if delimiter_index != -1 else 0
            # history_start_index is the index in truncated_turns where the actual history messages begin
            history_start_index = delimiter_index + 1 if delimiter_index != -1 else 0
            
            while current_token_count_turns > max_tokens_for_turns and len(truncated_turns) > 0:
                message_removed_this_iteration = False
                # 1. Try removing least relevant memories (from the start of truncated_turns)
                if num_memory_messages > 0:
                    removed_msg = truncated_turns.pop(0)
                    num_memory_messages -= 1
                    if delimiter_index != -1: delimiter_index -= 1 # Adjust delimiter index
                    history_start_index -=1 # Adjust history start index
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed memory message (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                
                # 2. Try removing delimiter if no memories left and delimiter exists
                elif delimiter_index != -1: # memory_delimiter_added implies delimiter_index was set
                    removed_msg = truncated_turns.pop(delimiter_index)
                    history_start_index -=1 # Adjust history start index as delimiter is removed
                    delimiter_index = -1 # Mark delimiter as removed
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed memory delimiter message.")
                    
                # 3. Try removing oldest history (from history_start_index)
                # Ensure history_start_index is valid and points to an actual history message
                elif history_start_index < len(truncated_turns):
                    # msg_to_remove = truncated_turns[history_start_index] # For logging before pop
                    # logger.debug(f"Truncate Turns: Attempting to remove history msg at index {history_start_index} (Role: {msg_to_remove.role}, Content: '{msg_to_remove.content[:50]}...')")
                    removed_msg = truncated_turns.pop(history_start_index) # Remove from start of history part
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed history message (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                else:
                    logger.warning("Truncation of turns stopped: No more messages to remove or invalid state.")
                    break # Exit loop if no message can be removed
                    
                if message_removed_this_iteration:
                    current_token_count_turns = sum(
                        msg.get_token_count(encoding, TOKENS_PER_MESSAGE_OVERHEAD) for msg in truncated_turns
                    )
                else: # Should not happen if logic above is correct and loop condition is met
                    logger.warning("Truncation of turns stopped: Could not remove a message in this iteration.")
                    break
            
            logger.warning(f"Turns context truncated to {len(truncated_turns)} messages, ~{current_token_count_turns} tokens.")
            return truncated_turns

        except Exception as e:
            logger.error(f"Token counting/truncation failed: {e}. Proceeding without truncation.", exc_info=self.config.VERBOSE)
            return messages_to_truncate # Return original on error

    def _assemble_final_messages(self, system_prompt_message: Message | None, truncated_turns: list[Message]) -> list[Message]:
        """Assembles the final list of messages to send to the LLM."""
        final_messages_to_send = []
        if system_prompt_message:
            final_messages_to_send.append(system_prompt_message)
        final_messages_to_send.extend(truncated_turns)
        logger.debug(f"Final context assembly: {len(final_messages_to_send)} total messages prepared.")
        return final_messages_to_send

    def _execute_llm_query(self, final_messages_to_send: list[Message], plaintext_output: bool, stream: bool) -> str:
        """Executes the query against the LLM client and handles the response."""
        query_kwargs = {"messages": final_messages_to_send, "plaintext_output": plaintext_output}

        if hasattr(self.client, 'SUPPORTS_STREAMING') and self.client.SUPPORTS_STREAMING and stream:
            query_kwargs["stream"] = True
        elif stream:
            logger.debug(f"Streaming requested but client {type(self.client).__name__} does not support it. Disabling streaming.")

        response_content = self.client.query(**query_kwargs)

        last_response = self.history_manager.get_last_assistant_message()
        if last_response and response_content == last_response and not self.config.ALLOW_DUPLICATE_RESPONSE:
            console.print("[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]")
            # Note: This might require adjusting temperature or other params in query_kwargs for actual regeneration
            response_content = self.client.query(**query_kwargs) # Re-query

        self.history_manager.add_message("assistant", response_content)
        return response_content

    def _add_conversation_to_memory(self, user_prompt: str, assistant_response: str):
        """Adds the user prompt and assistant response to the memory manager if enabled."""
        if not self.memory_manager or not assistant_response: # Don't save empty assistant responses
            return
        
        try:
            user_msg_id = str(uuid.uuid4())
            assistant_msg_id = str(uuid.uuid4())
            current_time = time.time()

            self.memory_manager.add_memory(
                message_id=user_msg_id, role="user", content=user_prompt, timestamp=current_time
            )
            self.memory_manager.add_memory(
                message_id=assistant_msg_id, role="assistant", content=assistant_response, timestamp=current_time + 0.001
            )
            logger.debug(f"Added user prompt (ID: {user_msg_id}) and assistant response (ID: {assistant_msg_id}) to memory.")
        except Exception as e:
            logger.exception(f"Failed to add messages to memory: {e}")


    def query(self, prompt: str, plaintext_output: bool = False, stream: bool = True) -> str:
        try:
            # Stage 1: Prepare initial prompt and history
            system_prompt_message, history_turns = self._prepare_initial_prompt_and_history(prompt)

            # Stage 2: Retrieve and prepare memory
            unique_memory_messages = self._retrieve_and_prepare_memory(prompt, history_turns)

            # Stage 3: Combine memory and history
            combined_turns, memory_delimiter_added = self._combine_memory_and_history(
                unique_memory_messages, history_turns
            )

            # Stage 4: Truncate messages if needed
            truncated_turns = self._truncate_messages_if_needed(
                combined_turns, system_prompt_message, memory_delimiter_added
            )

            # Stage 5: Assemble final messages for LLM
            final_messages_to_send = self._assemble_final_messages(
                system_prompt_message, truncated_turns
            )

            # Stage 6: Execute LLM query
            assistant_response = self._execute_llm_query(
                final_messages_to_send, plaintext_output, stream
            )

            # Stage 7: Add conversation to memory (if enabled and successful)
            self._add_conversation_to_memory(prompt, assistant_response)

            return assistant_response

        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}") # Log with stack trace
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
            
    def refine_prompt(self, prompt, history=None):
        """
        Refine the user's prompt using context and history to determine intent.
        
        Args:
            prompt: The user's original prompt
            history: Optional list of previous message objects or dictionaries
            
        Returns:
            A refined prompt that can be fed back to the LLM
        """
        try:
            messages = []
            messages.append(Message(role="system", content=SYSTEM_REFINE_PROMPT))
            
            # Add history context if provided
            if history:
                history_text = []
                for msg in history:
                    # Handle both Message objects and dictionaries
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        # Message object
                        history_text.append(f"{msg.role}: {msg.content}")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # Dictionary with role and content keys
                        history_text.append(f"{msg['role']}: {msg['content']}")
                    else:
                        logger.warning(f"Skipping invalid message format in history: {msg}")
                        
                if history_text:
                    context_message = f"Previous conversation history:\n{'\n'.join(history_text)}\n\nUser's raw prompt: {prompt}"
                    messages.append(Message(role="user", content=context_message))
                else:
                    messages.append(Message(role="user", content=f"User's raw prompt: {prompt}"))
            else:
                messages.append(Message(role="user", content=f"User's raw prompt: {prompt}"))
            
            # Make the query with plaintext output and no streaming for prompt refinement
            query_kwargs = {"messages": messages, "plaintext_output": True, "stream": False}
            refined_prompt = self.client.query(**query_kwargs)
            
            if logger.isEnabledFor(logging.DEBUG):
                console.print(f"[bold green]Messages:[/bold green] {messages}")
                logger.info(f"Refined prompt: {refined_prompt}")
                
            return refined_prompt
        except Exception as e:
            console.print(f"[bold yellow]Error during prompt refinement:[/bold yellow] {e}")
            logger.warning(f"Error during prompt refinement: {e}")
            # Return original prompt on error
            return prompt 
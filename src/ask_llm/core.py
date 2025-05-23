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
             from huggingface_hub.utils import HfHubHTTPError
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

    def query(self, prompt, plaintext_output: bool = False, stream: bool = True):
        try:
            self.history_manager.add_message("user", prompt)

            # Memory Retrieval (Logging Only) ---
            retrieved_memories = None
            if self.memory_manager:
                try:
                    logger.debug(f"Attempting to retrieve relevant memories for prompt: '{prompt[:50]}...'")
                    # Use configured number of results
                    n_results = self.config.MEMORY_N_RESULTS 
                    retrieved_memories = self.memory_manager.search_relevant_memories(prompt, n_results=n_results)
                    if retrieved_memories:
                        logger.debug(f"Retrieved {len(retrieved_memories)} memories (max {n_results}):")
                        for i, mem in enumerate(retrieved_memories):
                            # Log essential details - avoid logging full document content unless needed
                            logger.debug(f"  Mem {i+1}: ID={mem.get('id')}, Role={mem.get('metadata',{}).get('role')}, Dist={mem.get('distance'):.4f}, Content='{mem.get('document', '')[:60]}...'")
                    else:
                        logger.debug("No relevant memories found or search failed.")
                except Exception as e:
                    logger.exception(f"Error during memory retrieval: {e}")

            complete_context_messages = self.history_manager.get_context_messages()
            
            # Explicitly handle the primary system prompt ---
            system_prompt_message: Message | None = None
            processing_messages = list(complete_context_messages)
            if processing_messages and processing_messages[0].role == "system":
                system_prompt_message = processing_messages.pop(0)
                logger.debug(f"Extracted system prompt: '{system_prompt_message.content[:60]}...'")
            else:
                logger.warning("Could not find leading system prompt in history messages. A default might be added later if needed.")
            # Now processing_messages contains only the history turns (user/assistant)
            
            final_combined_turns = list(processing_messages) # Start with history turns
            memory_delimiter_added = False # Flag to track if delimiter was added

            # Memory Injection & Combination ---
            if retrieved_memories:
                # Convert retrieved dicts back to Message objects, sorted by relevance (most relevant first)
                sorted_memories = sorted(retrieved_memories, key=lambda x: x.get('distance', 0.0))
                memory_messages = [
                    Message(role=mem.get('metadata', {}).get('role', 'assistant'), content=mem.get('document', ''))
                    for mem in sorted_memories
                    if mem.get('document')
                ]
                
                # Stage 4: Deduplication - Remove memories matching history TURNS
                history_content_set = {msg.content for msg in processing_messages} # Use history turns only
                unique_memory_messages = [
                    mem_msg for mem_msg in memory_messages
                    if mem_msg.content not in history_content_set
                ]

                if unique_memory_messages:
                    logger.debug(f"Adding {len(unique_memory_messages)} unique memories (out of {len(memory_messages)} retrieved) to context.")
                    memory_delimiter = Message(role="system", content="--- Relevant Past Conversation Snippets (older history or related topics) ---")
                    # Combine: Memories + Delimiter + History Turns
                    final_combined_turns = unique_memory_messages + [memory_delimiter] + final_combined_turns
                    memory_delimiter_added = True
                else:
                    logger.debug("No unique memories found after deduplication.")
                    
            #  Token Counting & Truncation (Operating on combined turns) ---
            messages_to_truncate = final_combined_turns # The list to potentially shorten
            if _tiktoken_present and tiktoken:
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens_per_message = 4 # Approximation
                    
                    # We need to account for the system prompt's tokens separately
                    system_prompt_tokens = 0
                    if system_prompt_message:
                         system_prompt_tokens = tokens_per_message + len(encoding.encode(system_prompt_message.content)) + 3 # +3 for assistant prime?

                    def count_tokens_for_turns(messages: list[Message]) -> int:
                        num_tokens = 0
                        for message in messages:
                             num_tokens += tokens_per_message
                             if message.content: # Ensure content is not None
                                 num_tokens += len(encoding.encode(message.content))
                        return num_tokens

                    # Calculate max tokens available for the TURNS (memory + history)
                    max_tokens_for_turns = self.config.MAX_TOKENS - system_prompt_tokens
                    if max_tokens_for_turns < 0: max_tokens_for_turns = 0 # Avoid negative limit

                    initial_token_count_turns = count_tokens_for_turns(messages_to_truncate)
                    logger.debug(f"Combined memory/history turns: {len(messages_to_truncate)} messages, ~{initial_token_count_turns} tokens. System prompt: ~{system_prompt_tokens} tokens.")
                    logger.debug(f"Max tokens for turns: {max_tokens_for_turns} (Total limit: {self.config.MAX_TOKENS})")

                    if initial_token_count_turns > max_tokens_for_turns:
                        logger.warning(f"Combined turns context ({initial_token_count_turns} tokens) exceeds limit for turns ({max_tokens_for_turns}). Truncating...")
                        
                        # Simplified Truncation: Operates ONLY on messages_to_truncate
                        # Structure is [mem1, mem2, ..., delimiter?, hist1, hist2, ...]
                        
                        # Find delimiter index within messages_to_truncate if it exists
                        delimiter_index = -1
                        if memory_delimiter_added:
                             memory_delimiter_content = "--- Relevant Past Conversation Snippets (older history or related topics) ---"
                             for i, msg in enumerate(messages_to_truncate):
                                 if msg.role == "system" and msg.content == memory_delimiter_content:
                                     delimiter_index = i
                                     break
                        
                        num_memory_messages = delimiter_index if delimiter_index != -1 else 0
                        history_start_index = delimiter_index + 1 if delimiter_index != -1 else 0
                        
                        truncated_turns = list(messages_to_truncate) # Work on a copy
                        current_token_count_turns = initial_token_count_turns
                        
                        while current_token_count_turns > max_tokens_for_turns and len(truncated_turns) > 0:
                            message_removed = False
                            # 1. Try removing least relevant memories (start of list)
                            if num_memory_messages > 0:
                                removed_msg = truncated_turns.pop(0)
                                num_memory_messages -= 1
                                if delimiter_index != -1: delimiter_index -= 1
                                history_start_index -= 1
                                message_removed = True
                                logger.debug(f"Truncate Turns: Removed memory message (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                            
                            # 2. Try removing delimiter if no memories left
                            elif delimiter_index != -1:
                                removed_msg = truncated_turns.pop(delimiter_index)
                                history_start_index -= 1
                                delimiter_index = -1
                                message_removed = True
                                logger.debug(f"Truncate Turns: Removed memory delimiter message.")
                                
                            # 3. Try removing oldest history (start of history section)
                            elif history_start_index < len(truncated_turns):
                                msg_to_remove = truncated_turns[history_start_index]
                                logger.debug(f"Truncate Turns: Attempting to remove history msg at index {history_start_index} (Role: {msg_to_remove.role}, Content: '{msg_to_remove.content[:50]}...')")
                                removed_msg = truncated_turns.pop(history_start_index) # Remove from start of history part
                                message_removed = True
                                
                            else:
                                logger.warning("Truncation of turns stopped: No more messages to remove?")
                                break
                                
                            if message_removed:
                                current_token_count_turns = count_tokens_for_turns(truncated_turns)
                            else:
                                logger.warning("Truncation of turns stopped: Could not remove message.")
                                break
                        
                        messages_to_truncate = truncated_turns # Update the list with the truncated version
                        logger.warning(f"Turns context truncated to {len(messages_to_truncate)} messages, ~{current_token_count_turns} tokens.")
                    else:
                        logger.debug(f"Turns context is within token limits ({initial_token_count_turns} <= {max_tokens_for_turns}). No truncation needed.")

                except Exception as e:
                    logger.error(f"Token counting/truncation failed: {e}. Proceeding without truncation.", exc_info=self.config.VERBOSE)
                    messages_to_truncate = final_combined_turns # Reset to pre-truncation state on error
            elif self.memory_manager:
                 logger.warning("`tiktoken` library not found. Skipping context token counting and truncation. Install with `pip install tiktoken`")
                 messages_to_truncate = final_combined_turns # Ensure we use the combined list if no tiktoken
            # --- End Stage 4 Truncation ---
            
            # --- Final Assembly --- 
            final_messages_to_send = []
            if system_prompt_message:
                final_messages_to_send.append(system_prompt_message)
            final_messages_to_send.extend(messages_to_truncate) # Add the (potentially truncated) turns
            logger.debug(f"Final context assembly: {len(final_messages_to_send)} total messages prepared.")
            
            query_kwargs = {"messages": final_messages_to_send, "plaintext_output": plaintext_output,}

            if hasattr(self.client, 'SUPPORTS_STREAMING') and self.client.SUPPORTS_STREAMING and stream:
                 query_kwargs["stream"] = True
            elif stream:
                 logger.debug(f"Streaming requested but client {type(self.client).__name__} does not support it. Disabling streaming.")

            response = self.client.query(**query_kwargs)

            last_response = self.history_manager.get_last_assistant_message()
            if last_response and response == last_response and not self.config.ALLOW_DUPLICATE_RESPONSE:
                console.print("[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]")
                response = self.client.query(**query_kwargs)

            self.history_manager.add_message("assistant", response)

            # Add user prompt and successful assistant response to memory if enabled
            if self.memory_manager:
                try:
                    user_msg_id = str(uuid.uuid4())
                    assistant_msg_id = str(uuid.uuid4())
                    current_time = time.time()

                    # Add user prompt
                    self.memory_manager.add_memory(
                        message_id=user_msg_id,
                        role="user",
                        content=prompt,
                        timestamp=current_time # Or maybe history message timestamp if available?
                    )
                    # Add assistant response
                    self.memory_manager.add_memory(
                        message_id=assistant_msg_id,
                        role="assistant",
                        content=response,
                        timestamp=current_time + 0.001
                    )

                except Exception as e:
                    logger.exception(f"Failed to add messages to memory: {e}")
                    # Don't crash the main query process if memory addition fails

            return response
        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}")
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
import importlib.util
import pathlib
import threading
import time
import uuid
from difflib import SequenceMatcher
from importlib.metadata import entry_points
from rich.console import Console
from .utils.config import Config, is_huggingface_available, is_llama_cpp_available
from .utils.history import HistoryManager, Message
from .clients import LLMClient
from .bots import BotManager, get_system_prompt
from .user_profile import UserProfileManager
import logging

try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    _tiktoken_present = False
    tiktoken = None # Placeholder


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings using SequenceMatcher.
    
    Returns a float between 0.0 and 1.0, where 1.0 is an exact match.
    """
    return SequenceMatcher(None, text1, text2).ratio()


def discover_memory_backends() -> dict:
    """Discover installed memory backends via entry points."""
    backends = {}
    try:
        eps = entry_points(group='ask_llm.memory')
        for ep in eps:
            try:
                backends[ep.name] = ep.load()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load memory backend '{ep.name}': {e}")
    except Exception as e:
        logging.getLogger(__name__).debug(f"No memory backends discovered: {e}")
    return backends


console = Console()
logger = logging.getLogger(__name__)

class AskLLM:
    def __init__(self, resolved_model_alias: str, config: Config, local_mode: bool = False, bot_id: str = "nova", user_id: str = "default"):
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        self.model_definition = self.config.defined_models.get("models", {}).get(resolved_model_alias)
        self.local_mode = local_mode  # When True, skip database and use local filesystem
        self.bot_id = bot_id
        self.user_id = user_id
        self.memory_backend = None  # Long-term memory backend (PostgreSQL)
        self.short_term_backend = None  # Short-term memory (PostgreSQL session history)
        self.user_profile = None  # User profile from database

        if not self.model_definition:
             raise ValueError(f"Could not find model definition for resolved alias: '{resolved_model_alias}'")

        self.client: LLMClient = self.initialize_client(config)
        
        # Load the bot for system message
        bot_manager = BotManager(config)
        self.bot = bot_manager.get_bot(bot_id)
        if not self.bot:
            logger.warning(f"Bot '{bot_id}' not found, falling back to Nova")
            self.bot = bot_manager.get_default_bot()
            self.bot_id = self.bot.slug
        
        # Set bot name on client for panel display
        self.client.bot_name = self.bot.name

        # Initialize memory backends unless in local mode
        # - Long-term memory (memory_backend): Only for bots with requires_memory=true
        #   This triggers LLM-based memory extraction which has overhead
        # - Short-term memory (short_term_backend): For all bots when DB is available
        #   This is just session history storage with no LLM overhead
        if self.local_mode:
            logger.debug("Local mode enabled - using filesystem for history, skipping database")
        else:
            available_backends = discover_memory_backends()
            if available_backends:
                # Use PostgreSQL backend
                backend_name, backend_class = next(iter(available_backends.items()))
                
                try:
                    # Initialize long-term memory only if bot requires it
                    if self.bot.requires_memory:
                        logger.debug(f"Initializing long-term memory backend: {backend_name}")
                        self.memory_backend = backend_class(config=self.config, bot_id=self.bot_id)
                        logger.debug(f"Long-term memory enabled using backend: {backend_name} for bot: {self.bot_id}")
                    else:
                        logger.debug(f"Bot '{self.bot.name}' has requires_memory=false - skipping long-term memory")
                    
                    # Initialize short-term memory for all bots (no LLM overhead)
                    if hasattr(backend_class, 'get_short_term_manager'):
                        self.short_term_backend = backend_class.get_short_term_manager(config=self.config, bot_id=self.bot_id)
                        logger.debug(f"Short-term memory initialized for bot: {self.bot_id}")
                except Exception as e:
                    logger.exception(f"Failed to initialize memory backend: {e}")
                    if self.config.VERBOSE:
                        console.print(f"[bold yellow]Warning:[/bold yellow] Memory backend failed to initialize: {e}")
                    self.memory_backend = None
                    self.short_term_backend = None
            else:
                logger.debug("No memory backends discovered, using text file for history.")

        # Set system message from the bot's configured prompt
        # Inject user profile context before the bot's system prompt if not in local mode
        if not self.local_mode:
            try:
                profile_manager = UserProfileManager(config)
                self.user_profile, _ = profile_manager.get_or_create_profile(self.user_id)
                profile_context = self.user_profile.to_context_string()
                if profile_context:
                    self.config.SYSTEM_MESSAGE = f"{profile_context}\n\n{self.bot.system_prompt}"
                    logger.debug(f"Injected user profile for '{self.user_id}' into system prompt")
                else:
                    self.config.SYSTEM_MESSAGE = self.bot.system_prompt
            except Exception as e:
                logger.warning(f"Failed to load user profile: {e}")
                self.config.SYSTEM_MESSAGE = self.bot.system_prompt
        else:
            self.config.SYSTEM_MESSAGE = self.bot.system_prompt
        logger.debug(f"Using bot '{self.bot.name}' ({self.bot_id}) with system prompt set")

        # Initialize history manager with short-term backend if available
        self.history_manager = HistoryManager(
            client=self.client,
            config=self.config,
            db_backend=self.short_term_backend,
            bot_id=self.bot_id,
        )

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
                if self.config.ollama_checked and model_id not in self.config.available_ollama_models:
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
        if not self.memory_backend:
            return []

        retrieved_memories_raw = None
        try:
            logger.debug(f"Attempting to retrieve relevant memories for prompt: '{prompt[:50]}...'")
            n_results = self.config.MEMORY_N_RESULTS
            min_relevance = self.config.MEMORY_MIN_RELEVANCE
            retrieved_memories_raw = self.memory_backend.search(prompt, n_results=n_results, min_relevance=min_relevance)
            if retrieved_memories_raw:
                logger.debug(f"Retrieved {len(retrieved_memories_raw)} memories (max {n_results}, min_relevance={min_relevance}):")
                for i, mem in enumerate(retrieved_memories_raw):
                    logger.debug(f"  Mem {i+1}: ID={mem.get('id')}, Role={mem.get('metadata',{}).get('role')}, Relevance={mem.get('relevance', 0):.4f}, Content='{mem.get('document', '')[:60]}...'")
            else:
                logger.debug("No relevant memories found or search failed.")
        except Exception as e:
            logger.exception(f"Error during memory retrieval: {e}")
            return []

        if not retrieved_memories_raw:
            return []

        # Convert retrieved dicts to Message objects
        # Sort by relevance ASCENDING (least relevant first) for proper truncation order
        # Truncation removes from index 0, so least relevant should be first
        sorted_memories = sorted(retrieved_memories_raw, key=lambda x: x.get('relevance', 0.0))
        memory_messages = [
            Message(role=mem.get('metadata', {}).get('role', 'assistant'), content=mem.get('document', ''))
            for mem in sorted_memories if mem.get('document')
        ]

        # Fuzzy deduplicate against history turns
        # A memory is considered duplicate if similarity >= threshold
        similarity_threshold = self.config.MEMORY_DEDUP_SIMILARITY
        history_contents = [msg.content for msg in history_turns]
        
        unique_memory_messages = []
        for mem_msg in memory_messages:
            is_duplicate = False
            for hist_content in history_contents:
                similarity = _text_similarity(mem_msg.content, hist_content)
                if similarity >= similarity_threshold:
                    logger.debug(f"Dedup: Memory '{mem_msg.content[:40]}...' is {similarity:.2%} similar to history, skipping")
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_memory_messages.append(mem_msg)
        
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
                                   memory_delimiter_added: bool,
                                   protected_recent_count: int = 0) -> list[Message]:
        """Counts tokens and truncates the combined messages if they exceed the token limit.
        
        Args:
            messages_to_truncate: Combined list of memories and history messages.
            system_prompt_message: The system prompt message (counted separately).
            memory_delimiter_added: Whether a memory delimiter was added.
            protected_recent_count: Number of recent messages to protect from truncation.
        """
        if not (_tiktoken_present and tiktoken):
            if self.memory_backend: # Only warn if memory (which relies on this) is on
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
            
            # Find the delimiter index
            delimiter_index = -1
            memory_delimiter_content = "--- Relevant Past Conversation Snippets (older history or related topics) ---"
            if memory_delimiter_added:
                for i, msg in enumerate(truncated_turns):
                    if msg.role == "system" and msg.content == memory_delimiter_content:
                        delimiter_index = i
                        break
            
            # Track memory and history boundaries
            # Structure: [memory_0 (least relevant), ..., memory_N (most relevant), delimiter, history_0 (oldest), ..., history_N (newest)]
            num_memory_messages = delimiter_index if delimiter_index != -1 else 0
            history_start_index = delimiter_index + 1 if delimiter_index != -1 else 0
            
            # Calculate protected zone at the end (most recent history)
            total_messages = len(truncated_turns)
            protected_start_index = max(history_start_index, total_messages - protected_recent_count) if protected_recent_count > 0 else total_messages
            
            while current_token_count_turns > max_tokens_for_turns and len(truncated_turns) > 0:
                message_removed_this_iteration = False
                
                # 1. Try removing least relevant memories first (from the start, index 0)
                # Memories are ordered: least relevant first, most relevant last (just before delimiter)
                if num_memory_messages > 0:
                    removed_msg = truncated_turns.pop(0)
                    num_memory_messages -= 1
                    if delimiter_index != -1: delimiter_index -= 1
                    history_start_index -= 1
                    protected_start_index -= 1
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed least relevant memory (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                
                # 2. Try removing oldest history (non-protected)
                elif history_start_index < protected_start_index and history_start_index < len(truncated_turns):
                    removed_msg = truncated_turns.pop(history_start_index)
                    protected_start_index -= 1
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed oldest history message (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                
                # 3. Try removing delimiter if no more history to remove
                elif delimiter_index != -1:
                    removed_msg = truncated_turns.pop(delimiter_index)
                    history_start_index -= 1
                    protected_start_index -= 1
                    delimiter_index = -1
                    message_removed_this_iteration = True
                    logger.debug(f"Truncate Turns: Removed memory delimiter message.")
                
                # 4. Last resort: remove protected history (oldest of the protected messages)
                elif protected_start_index < len(truncated_turns):
                    removed_msg = truncated_turns.pop(protected_start_index)
                    message_removed_this_iteration = True
                    logger.warning(f"Truncate Turns: Removed PROTECTED history message (Role: {removed_msg.role}, Content: '{removed_msg.content[:50]}...')")
                else:
                    logger.warning("Truncation of turns stopped: No more messages to remove or invalid state.")
                    break
                    
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

    def prepare_messages_for_query(self, prompt: str) -> list[Message]:
        """
        Prepare messages for a query, including history and memory context.
        
        This method runs stages 1-5 of the query pipeline without executing the LLM call.
        Useful for external callers (like the service API) that need the prepared messages
        to stream themselves.
        
        Args:
            prompt: The user's prompt/question
            
        Returns:
            List of Message objects ready to send to the LLM, including:
            - System prompt
            - Memory context (if available)
            - Conversation history
            - The current user prompt
        """
        # Stage 1: Prepare initial prompt and history
        system_prompt_message, history_turns = self._prepare_initial_prompt_and_history(prompt)

        # Stage 2: Retrieve and prepare memory
        unique_memory_messages = self._retrieve_and_prepare_memory(prompt, history_turns)

        # Stage 3: Combine memory and history
        combined_turns, memory_delimiter_added = self._combine_memory_and_history(
            unique_memory_messages, history_turns
        )

        # Stage 4: Truncate messages if needed
        protected_turns = self.config.MEMORY_PROTECTED_RECENT_TURNS if self.memory_backend else 0
        protected_messages = protected_turns * 2
        
        truncated_turns = self._truncate_messages_if_needed(
            combined_turns, system_prompt_message, memory_delimiter_added, protected_messages
        )

        # Stage 5: Assemble final messages for LLM
        final_messages_to_send = self._assemble_final_messages(
            system_prompt_message, truncated_turns
        )
        
        return final_messages_to_send

    def finalize_response(self, prompt: str, response: str):
        """
        Finalize a response after streaming completes.
        
        This adds the assistant response to history and triggers memory extraction.
        Should be called after prepare_messages_for_query() + streaming.
        
        Args:
            prompt: The original user prompt
            response: The complete assistant response
        """
        # Add to history
        self.history_manager.add_message("assistant", response)
        
        # Trigger memory extraction in background
        if self.memory_backend and response:
            thread = threading.Thread(
                target=self._add_conversation_to_memory,
                args=(prompt, response),
                daemon=False,
            )
            thread.start()
            thread.join(timeout=0.5)

    def _execute_llm_query(self, final_messages_to_send: list[Message], plaintext_output: bool, stream: bool) -> str:
        """Executes the query against the LLM client and handles the response."""
        query_kwargs = {"messages": final_messages_to_send, "plaintext_output": plaintext_output}

        # Debug: log what's being sent
        if final_messages_to_send:
            system_msg = next((m for m in final_messages_to_send if m.role == "system"), None)
            if system_msg:
                logger.debug(f"System prompt being sent ({len(system_msg.content)} chars): {system_msg.content[:100]}...")
            else:
                logger.warning("No system message in final messages!")
            logger.debug(f"Total messages being sent: {len(final_messages_to_send)}")

        if hasattr(self.client, 'SUPPORTS_STREAMING') and self.client.SUPPORTS_STREAMING and stream:
            query_kwargs["stream"] = True
        elif stream:
            logger.debug(f"Streaming requested but client {type(self.client).__name__} does not support it. Disabling streaming.")

        response_content = self.client.query(**query_kwargs)

        self.history_manager.add_message("assistant", response_content)
        return response_content

    def _add_conversation_to_memory(self, user_prompt: str, assistant_response: str):
        """Triggers memory extraction from a conversation exchange.
        
        Only submits to background service if available. Does NOT fall back to
        local extraction to avoid slow LLM calls blocking the CLI.
        
        Note: Messages are already saved by history_manager.add_message() which uses
        short_term_backend. This method only handles memory extraction.
        """
        if not self.memory_backend or not assistant_response:
            return
        
        try:
            # Generate message IDs for the extraction service
            user_msg_id = str(uuid.uuid4())
            assistant_msg_id = str(uuid.uuid4())
            
            # Trigger memory extraction if enabled - only via background service
            if getattr(self.config, 'MEMORY_EXTRACTION_ENABLED', True):
                # Only use background service - no local fallback to avoid slow LLM calls
                if self._submit_to_background_service(user_prompt, assistant_response, [user_msg_id, assistant_msg_id]):
                    logger.debug("Memory extraction submitted to background service")
                else:
                    logger.debug("Background service unavailable - skipping memory extraction")
        except Exception as e:
            logger.exception(f"Failed to process memory extraction: {e}")
    
    def _submit_to_background_service(
        self,
        user_prompt: str,
        assistant_response: str,
        message_ids: list[str],
    ) -> bool:
        """
        Try to submit extraction task to background service.
        
        Returns True if successfully submitted, False if service unavailable.
        """
        try:
            from .service import ServiceClient
            from .service.tasks import create_extraction_task
            
            client = ServiceClient()
            if not client.is_available():
                return False
            
            task = create_extraction_task(
                user_message=user_prompt,
                assistant_message=assistant_response,
                bot_id=self.bot_id,
                user_id=self.user_id,
                message_ids=message_ids,
            )
            
            return client.submit_task(task)
            
        except ImportError:
            logger.debug("Service module not available")
            return False
        except Exception as e:
            logger.debug(f"Failed to submit to background service: {e}")
            return False
    
    def _extract_memories_from_exchange(
        self,
        user_prompt: str,
        assistant_response: str,
        message_ids: list[str]
    ):
        """Extract important facts from a conversation exchange and store as memories.
        
        Uses the MemoryExtractionService to analyze the exchange and create
        distilled, importance-weighted memories.
        """
        try:
            from .memory.extraction import MemoryExtractionService
            
            # Check if we should use LLM or heuristics for extraction
            # For local GGUF models with small context, use heuristics to avoid overflow
            model_type = self.model_definition.get("type", "")
            use_llm_extraction = model_type not in ("gguf", "ollama")
            
            if use_llm_extraction:
                extraction_service = MemoryExtractionService(llm_client=self.client)
            else:
                # Use heuristics for local models to avoid context overflow
                extraction_service = MemoryExtractionService(llm_client=None)
                logger.debug("Using heuristic extraction for local model")
            
            messages = [
                {"id": message_ids[0], "role": "user", "content": user_prompt},
                {"id": message_ids[1], "role": "assistant", "content": assistant_response},
            ]
            
            # Extract facts (LLM or heuristics based on model type)
            facts = extraction_service.extract_from_conversation(messages, use_llm=use_llm_extraction)
            
            if not facts:
                logger.debug("No facts extracted from conversation exchange")
                return
            
            # Get existing memories for deduplication
            existing_memories = []
            if hasattr(self.memory_backend, 'list_recent'):
                recent = self.memory_backend.list_recent(n=50)
                existing_memories = [
                    {
                        "id": m.get("id"),
                        "content": m.get("document", m.get("content", "")),
                        "importance": m.get("metadata", {}).get("importance", 0.5),
                    }
                    for m in recent
                ]
            
            # Determine what actions to take
            actions = extraction_service.determine_memory_actions(facts, existing_memories)
            
            min_importance = getattr(self.config, 'MEMORY_EXTRACTION_MIN_IMPORTANCE', 0.3)
            
            # Execute memory actions
            for action in actions:
                if action.action == "ADD" and action.fact:
                    if action.fact.importance >= min_importance:
                        memory_id = str(uuid.uuid4())
                        self.memory_backend.add_memory(
                            memory_id=memory_id,
                            content=action.fact.content,
                            memory_type=action.fact.memory_type,
                            importance=action.fact.importance,
                            source_message_ids=action.fact.source_message_ids,
                        )
                        logger.debug(f"Added memory: {action.fact.content[:50]}... (importance: {action.fact.importance})")
                        
                elif action.action == "UPDATE" and action.fact and action.target_memory_id:
                    # Create new memory and supersede the old one (preserves history)
                    new_memory_id = str(uuid.uuid4())
                    self.memory_backend.add_memory(
                        memory_id=new_memory_id,
                        content=action.fact.content,
                        memory_type=action.fact.memory_type,
                        importance=action.fact.importance,
                        source_message_ids=action.fact.source_message_ids,
                    )
                    # Mark old memory as superseded instead of overwriting
                    if hasattr(self.memory_backend, 'supersede_memory'):
                        self.memory_backend.supersede_memory(action.target_memory_id, new_memory_id)
                        logger.debug(f"Memory {action.target_memory_id} superseded by {new_memory_id}")
                    else:
                        logger.debug(f"Added updated memory {new_memory_id} (old: {action.target_memory_id})")
                        
                elif action.action == "DELETE" and action.target_memory_id:
                    # For explicit deletions, still use supersede if we have a replacement
                    # Otherwise mark as superseded by a "tombstone" (null supersedes = soft delete)
                    if hasattr(self.memory_backend, 'supersede_memory'):
                        # Use special marker to indicate explicit deletion
                        self.memory_backend.supersede_memory(action.target_memory_id, "DELETED")
                        logger.debug(f"Soft-deleted memory {action.target_memory_id}")
                    elif hasattr(self.memory_backend, 'delete_memory'):
                        self.memory_backend.delete_memory(action.target_memory_id)
                        logger.debug(f"Deleted memory {action.target_memory_id}")
                        
        except ImportError as e:
            logger.debug(f"Memory extraction not available: {e}")
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")


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
            # Calculate protected recent messages (each turn = 2 messages: user + assistant)
            protected_turns = self.config.MEMORY_PROTECTED_RECENT_TURNS if self.memory_backend else 0
            protected_messages = protected_turns * 2  # Each turn has user + assistant message
            
            truncated_turns = self._truncate_messages_if_needed(
                combined_turns, system_prompt_message, memory_delimiter_added, protected_messages
            )

            # Stage 5: Assemble final messages for LLM
            final_messages_to_send = self._assemble_final_messages(
                system_prompt_message, truncated_turns
            )

            # Stage 6: Execute LLM query
            assistant_response = self._execute_llm_query(
                final_messages_to_send, plaintext_output, stream
            )

            # Stage 7: Add conversation to memory in background (non-blocking)
            if self.memory_backend and assistant_response:
                thread = threading.Thread(
                    target=self._add_conversation_to_memory,
                    args=(prompt, assistant_response),
                    daemon=False,  # Non-daemon so it can complete before exit
                )
                thread.start()
                # Give the thread a brief moment to submit to background service
                # This allows quick HTTP requests to complete while still not blocking
                thread.join(timeout=0.5)

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
            refine_prompt = get_system_prompt("refine") or "You are a prompt refinement assistant."
            messages.append(Message(role="system", content=refine_prompt))
            
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
"""Service-side AskLLM class - supports loading local models (GGUF, etc.)

This is separate from the CLI's core.py which only supports OpenAI API calls.
The service runs on the server and can load models directly.
"""

import threading
import time
import uuid
from difflib import SequenceMatcher
from rich.console import Console
from ..utils.config import Config, has_database_credentials, is_llama_cpp_available
from ..utils.history import HistoryManager, Message
from ..clients import LLMClient
from ..clients.openai_client import OpenAIClient
from ..bots import Bot, BotManager, get_system_prompt
from ..profiles import ProfileManager, EntityType
from ..memory_server.client import MemoryClient, get_memory_client
from ..tools import get_tools_prompt, query_with_tools
from ..search import get_search_client, SearchClient
from .logging import get_service_logger
import logging

try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    _tiktoken_present = False
    tiktoken = None


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


console = Console()
logger = logging.getLogger(__name__)
slog = get_service_logger(__name__)


class ServiceAskLLM:
    """Server-side LLM class that can load local models directly.
    
    Supports:
    - OpenAI-compatible API
    - GGUF models via llama-cpp-python
    - Memory augmentation when database is available
    """
    
    def __init__(
        self, 
        resolved_model_alias: str, 
        config: Config, 
        local_mode: bool = False, 
        bot_id: str = "nova", 
        user_id: str = "default"
    ):
        self.resolved_model_alias = resolved_model_alias
        self.config = config
        self.model_definition = self.config.defined_models.get("models", {}).get(resolved_model_alias)
        self.local_mode = local_mode
        self.bot_id = bot_id
        self.user_id = user_id
        self.memory: MemoryClient | None = None
        self.user_profile = None
        self.profile_manager: "ProfileManager | None" = None
        self.search_client: "SearchClient | None" = None

        if not self.model_definition:
            raise ValueError(f"Could not find model definition for resolved alias: '{resolved_model_alias}'")

        # Initialize client based on model type
        self.client: LLMClient = self._initialize_client()
        
        # Load bot configuration
        bot_manager = BotManager(config)
        resolved_bot = bot_manager.get_bot(bot_id) or bot_manager.get_default_bot()
        if not resolved_bot:
            raise ValueError(f"No bot available for '{bot_id}' and no default bot found")
        if resolved_bot.slug != bot_id:
            logger.warning(f"Bot '{bot_id}' not found, falling back to {resolved_bot.slug}")
        self.bot: Bot = resolved_bot
        self.bot_id = self.bot.slug
        
        self.client.bot_name = self.bot.name

        # Initialize memory
        # - In MCP server mode, we allow memory even if DB creds aren't present locally.
        # - In embedded mode, we require DB credentials.
        self._db_available = False
        if self.local_mode:
            logger.debug("Local mode enabled - using filesystem for history")
        else:
            try:
                memory_server_url = getattr(config, "MEMORY_SERVER_URL", None)
                if self.bot.requires_memory and (memory_server_url or has_database_credentials(config)):
                    self.memory = get_memory_client(
                        config=config,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                        server_url=memory_server_url,
                    )
                    logger.debug(
                        "Memory client initialized for bot=%s mode=%s",
                        self.bot_id,
                        "mcp" if memory_server_url else "embedded",
                    )
                    self._db_available = True
                else:
                    logger.debug(f"Bot '{self.bot.name}' has requires_memory=false or memory unavailable")
            except Exception as e:
                logger.exception(f"Failed to initialize memory client: {e}")
                if self.config.VERBOSE:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Memory client failed: {e}")
                self.memory = None

        # Initialize search client if bot uses search
        if getattr(self.bot, 'uses_search', False):
            try:
                self.search_client = get_search_client(config)
                if self.search_client and self.search_client.is_available():
                    slog.info(f"Search client initialized: {self.search_client.PROVIDER.value}")
                else:
                    slog.info("Search requested but no provider available")
                    self.search_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize search client: {e}")
                self.search_client = None
        else:
            slog.debug(f"Bot '{self.bot.name}' does not use search")

        # Set system message with optional user profile and bot personality
        system_prompt = self.bot.system_prompt
        
        if self._db_available:
            try:
                self.profile_manager = ProfileManager(config)
                
                # Get user profile summary for context injection
                user_context = self.profile_manager.get_user_profile_summary(self.user_id)
                
                # Get bot personality traits (if any have developed)
                bot_context = self.profile_manager.get_bot_profile_summary(self.bot_id)
                
                # Build system message with profile contexts
                context_parts = []
                if user_context:
                    context_parts.append(f"## About the User\n{user_context}")
                    # Print full user profile on startup for monitoring
                    console.print(f"[dim]─── User Profile ({self.user_id}) ───[/dim]")
                    console.print(f"[cyan]{user_context}[/cyan]")
                    console.print(f"[dim]{'─' * 30}[/dim]")
                if bot_context:
                    context_parts.append(f"## Your Developed Traits\n{bot_context}")
                    console.print(f"[dim]─── Bot Profile ({self.bot_id}) ───[/dim]")
                    console.print(f"[magenta]{bot_context}[/magenta]")
                    console.print(f"[dim]{'─' * 30}[/dim]")
                
                if context_parts:
                    profile_context = "\n\n".join(context_parts)
                    self.config.SYSTEM_MESSAGE = f"{profile_context}\n\n{system_prompt}"
                else:
                    self.config.SYSTEM_MESSAGE = system_prompt
                    
            except Exception as e:
                logger.warning(f"Failed to load profiles: {e}")
                self.config.SYSTEM_MESSAGE = system_prompt
        else:
            self.config.SYSTEM_MESSAGE = system_prompt

        # Initialize history manager
        self.history_manager = HistoryManager(
            client=self.client,
            config=self.config,
            db_backend=self.memory.get_short_term_manager() if self.memory else None,
            bot_id=self.bot_id,
        )
        self.load_history()

    def _initialize_client(self) -> LLMClient:
        """Initialize client based on model type - supports local models."""
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        # Use ServiceLogger for deduped model loading messages
        slog.model_loading(self.resolved_model_alias, model_type)
        start_time = time.perf_counter()
        
        if model_type == "openai":
            if not model_id:
                raise ValueError(f"Missing 'model_id' in definition for '{self.resolved_model_alias}'")
            base_url = self.model_definition.get("base_url")
            api_key = self.model_definition.get("api_key")
            client = OpenAIClient(model_id, config=self.config, base_url=base_url, api_key=api_key)
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        elif model_type == "gguf":
            if not is_llama_cpp_available():
                raise ImportError(
                    "llama-cpp-python is required for GGUF models. "
                    "Install with: pip install llama-cpp-python"
                )
            from ..clients.llama_cpp_client import LlamaCppClient
            from ..gguf_handler import get_or_download_gguf_model
            
            repo_id = self.model_definition.get("repo_id")
            filename = self.model_definition.get("filename")
            if not repo_id or not filename:
                raise ValueError(
                    f"Missing 'repo_id' or 'filename' in GGUF definition for '{self.resolved_model_alias}'"
                )
            
            # Download model if needed
            model_path = get_or_download_gguf_model(repo_id, filename, self.config)
            if not model_path:
                raise FileNotFoundError(f"Could not download GGUF model: {repo_id}/{filename}")
            
            client = LlamaCppClient(model_path, config=self.config)
            load_time_ms = (time.perf_counter() - start_time) * 1000
            slog.model_loaded(self.resolved_model_alias, model_type, load_time_ms)
            return client
        
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: openai, gguf"
            )

    def load_history(self, since_minutes: int | None = None) -> list[dict]:
        if since_minutes is None:
            since_minutes = self.config.HISTORY_DURATION
        self.history_manager.load_history(since_minutes=since_minutes)
        return [msg.to_dict() for msg in self.history_manager.messages]

    def query(self, prompt: str, plaintext_output: bool = False, stream: bool = True) -> str:
        """Send a query to the LLM and return the response."""
        try:
            self.history_manager.add_message("user", prompt)
            context_messages = self._build_context_messages(prompt)
            
            # Use tool loop if bot has tools enabled
            if self.bot.uses_tools and self.memory:
                def query_fn(msgs, do_stream):
                    return self.client.query(msgs, plaintext_output=True, stream=do_stream)
                
                assistant_response = query_with_tools(
                    messages=context_messages,
                    query_fn=query_fn,
                    memory_client=self.memory,
                    profile_manager=self.profile_manager,
                    search_client=self.search_client,
                    user_id=self.user_id,
                    bot_id=self.bot_id,
                    stream=stream,
                )
            else:
                assistant_response = self.client.query(
                    context_messages, 
                    plaintext_output=plaintext_output, 
                    stream=stream
                )
            
            if assistant_response:
                self.history_manager.add_message("assistant", assistant_response)
                if self.memory:
                    self._trigger_memory_extraction(prompt, assistant_response)
            
            return assistant_response

        except KeyboardInterrupt:
            console.print("[bold red]Query interrupted.[/bold red]")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""
        except Exception as e:
            console.print(f"[bold red]Error during query:[/bold red] {e}")
            logger.exception(f"Error during query: {e}")
            self.history_manager.remove_last_message_if_partial("assistant")
            return ""

    def _build_context_messages(self, prompt: str) -> list[Message]:
        """Build messages list with system prompt, memory context, and history."""
        messages = []
        
        # Build system prompt
        system_parts = []
        if self.config.SYSTEM_MESSAGE:
            system_parts.append(self.config.SYSTEM_MESSAGE)
        
        # Add tool instructions if bot uses tools
        if self.bot.uses_tools and self.memory:
            include_search = self.search_client is not None
            slog.debug(f"Adding tools prompt (include_search={include_search})")
            system_parts.append(get_tools_prompt(include_search_tools=include_search))
        # Retrieve relevant memories if available (only if NOT using tools - tools let LLM search itself)
        elif self.memory:
            memory_context = self._retrieve_memory_context(prompt)
            if memory_context:
                system_parts.append(memory_context)
        
        if system_parts:
            messages.append(Message(role="system", content="\n\n".join(system_parts)))
        
        # Add conversation history.
        # For tool-enabled bots, optionally skip history for search-like prompts
        # to avoid stale tool outputs polluting context.
        include_history = True
        if self.bot.uses_tools and self.memory and getattr(self.config, "TOOLS_SKIP_HISTORY", True):
            include_history = not self._should_skip_history(prompt)

        if include_history:
            history = self.history_manager.get_context_messages()
            for msg in history:
                if msg.role in ("user", "assistant"):
                    messages.append(msg)
        else:
            # Always include the current user prompt when history is skipped
            if prompt:
                messages.append(Message(role="user", content=prompt))
        
        return messages

    def _should_skip_history(self, prompt: str) -> bool:
        """Return True if history should be skipped for this prompt.

        We skip history for search-like requests to prevent stale tool results
        (and previous "can't search" replies) from polluting context.
        """
        prompt_lower = (prompt or "").lower()
        search_triggers = (
            "search",
            "web_search",
            "news",
            "current",
            "today",
            "latest",
            "now",
            "date",
            "time",
            "headline",
        )
        return any(token in prompt_lower for token in search_triggers)

    def _retrieve_memory_context(self, prompt: str) -> str:
        """Retrieve relevant memories and format as context string."""
        if not self.memory:
            return ""
        
        try:
            n_results = self.config.MEMORY_N_RESULTS
            min_relevance = self.config.MEMORY_MIN_RELEVANCE
            
            mode = "mcp" if getattr(self.memory, "server_url", None) else "embedded"
            logger.info(
                "Memory retrieval (%s): query=%r n_results=%s min_relevance=%s",
                mode,
                prompt[:120],
                n_results,
                min_relevance,
            )

            memory_results = self.memory.search(prompt, n_results=n_results, min_relevance=min_relevance)

            logger.info("Memory retrieval (%s): returned=%d", mode, len(memory_results))
            
            if not memory_results:
                return ""
            
            # Convert to dicts for context builder
            memories = [
                {
                    "content": m.content,
                    "relevance": m.relevance,
                    "tags": m.tags,
                    "importance": m.importance,
                }
                for m in memory_results
            ]
            
            # Deduplicate against recent history
            history_contents = [msg.content for msg in self.history_manager.messages[-10:]]
            unique_memories = []
            for mem in memories:
                is_dup = any(
                    _text_similarity(mem["content"], h) >= self.config.MEMORY_DEDUP_SIMILARITY
                    for h in history_contents
                )
                if not is_dup:
                    unique_memories.append(mem)
            
            if not unique_memories:
                return ""
            
            # Format as context string
            from ..memory.context_builder import build_memory_context_string
            user_name = getattr(self.user_profile, 'name', None) if self.user_profile else None
            return build_memory_context_string(unique_memories, user_name=user_name)
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""

    def _trigger_memory_extraction(self, user_prompt: str, assistant_response: str):
        """Trigger background memory extraction (non-blocking)."""
        if not self.memory:
            logger.debug("Skipping extraction - no memory client")
            return
        
        logger.info(f"[Extraction] Triggering for bot={self.bot_id} user={self.user_id}")
        
        def extract():
            try:
                from ..service import ServiceClient
                from ..service.tasks import create_extraction_task
                
                client = ServiceClient()
                if client.is_available():
                    task = create_extraction_task(
                        user_message=user_prompt,
                        assistant_message=assistant_response,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                        message_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
                        model=self.resolved_model_alias,
                    )
                    result = client.submit_task(task)
                    logger.info(f"[Extraction] Task submitted: {task.task_id}")
                else:
                    logger.warning("[Extraction] Service not available")
            except Exception as e:
                logger.exception(f"[Extraction] Failed: {e}")
        
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()
        thread.join(timeout=0.5)

    def refine_prompt(self, prompt: str, history=None) -> str:
        """Refine the user's prompt using context."""
        try:
            messages = []
            refine_prompt = get_system_prompt("refine") or "You are a prompt refinement assistant."
            messages.append(Message(role="system", content=refine_prompt))
            
            if history:
                history_text = []
                for msg in history:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        history_text.append(f"{msg.role}: {msg.content}")
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        history_text.append(f"{msg['role']}: {msg['content']}")
                        
                if history_text:
                    context = f"Previous conversation:\n{chr(10).join(history_text)}\n\nUser's prompt: {prompt}"
                    messages.append(Message(role="user", content=context))
                else:
                    messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            else:
                messages.append(Message(role="user", content=f"User's prompt: {prompt}"))
            
            return self.client.query(messages, plaintext_output=True, stream=False)
        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}")
            return prompt

    # =========================================================================
    # Methods required by the service API
    # =========================================================================

    def prepare_messages_for_query(self, prompt: str) -> list[Message]:
        """Prepare messages for query including history and memory context.
        
        Called by the service API before sending to the LLM.
        """
        # Add user message to history first
        self.history_manager.add_message("user", prompt)
        
        # Build context with system prompt, memory, and history
        return self._build_context_messages(prompt)

    def _execute_llm_query(
        self, 
        messages: list[Message], 
        plaintext_output: bool = False, 
        stream: bool = False
    ) -> str:
        """Execute the LLM query and return response.
        
        Handles tool calling loop if bot has tools enabled.
        Called by the service API to get the response.
        """
        # Use tool loop if bot has tools enabled
        if self.bot.uses_tools and self.memory:
            def query_fn(msgs, do_stream):
                return self.client.query(msgs, plaintext_output=True, stream=do_stream)
            
            return query_with_tools(
                messages=messages,
                query_fn=query_fn,
                memory_client=self.memory,
                profile_manager=self.profile_manager,
                search_client=self.search_client,
                user_id=self.user_id,
                bot_id=self.bot_id,
                stream=stream,
            )
        
        return self.client.query(messages, plaintext_output=plaintext_output, stream=stream)

    def finalize_response(self, user_prompt: str, response: str):
        """Finalize the response by saving to history and triggering extraction.
        
        Called by the service API after receiving the response.
        """
        if response:
            self.history_manager.add_message("assistant", response)
            
            # Always trigger background memory extraction - it extracts facts and profile attributes
            # Even if bot has tools, the bot may not reliably use them, so extraction is a backup
            if self.memory:
                self._trigger_memory_extraction(user_prompt, response)

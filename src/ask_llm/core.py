"""Core AskLLM class - simple OpenAI-compatible chat client with memory."""

import threading
import time
import uuid
from difflib import SequenceMatcher
from rich.console import Console
from .utils.config import Config, has_database_credentials
from .utils.history import HistoryManager, Message
from .clients import LLMClient
from .clients.openai_client import OpenAIClient
from .bots import BotManager, get_system_prompt
from .profiles import ProfileManager, EntityType
from .memory_server.client import MemoryClient, get_memory_client
from .tools import get_tools_prompt, query_with_tools
from .search import get_search_client, SearchClient
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


class AskLLM:
    """Simple LLM client with optional memory augmentation.
    
    Uses OpenAI-compatible API. Memory features work when database is available,
    otherwise falls back to filesystem-only mode.
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
        self.profile_manager: ProfileManager | None = None
        self.search_client: SearchClient | None = None

        if not self.model_definition:
            raise ValueError(f"Could not find model definition for resolved alias: '{resolved_model_alias}'")

        # Initialize OpenAI-compatible client
        self.client: LLMClient = self._initialize_client()
        
        # Load bot configuration
        bot_manager = BotManager(config)
        self.bot = bot_manager.get_bot(bot_id)
        if not self.bot:
            logger.warning(f"Bot '{bot_id}' not found, falling back to Nova")
            self.bot = bot_manager.get_default_bot()
            self.bot_id = self.bot.slug
        
        self.client.bot_name = self.bot.name

        # Initialize memory if database is available
        self._db_available = False
        if self.local_mode:
            logger.debug("Local mode enabled - using filesystem for history")
        elif not has_database_credentials(config):
            logger.debug("Database credentials not configured - using filesystem for history")
        else:
            try:
                if self.bot.requires_memory:
                    self.memory = get_memory_client(
                        config=config,
                        bot_id=self.bot_id,
                        user_id=self.user_id,
                        server_url=getattr(config, "MEMORY_SERVER_URL", None),
                    )
                    logger.debug(f"Memory client initialized for bot: {self.bot_id}")
                else:
                    logger.debug(f"Bot '{self.bot.name}' has requires_memory=false")
                self._db_available = True
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
                    logger.debug(f"Search client initialized: {self.search_client.PROVIDER.value}")
                else:
                    logger.debug("Search requested but no provider available")
                    self.search_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize search client: {e}")
                self.search_client = None

        # Set system message with optional user profile and tools
        system_prompt = self.bot.system_prompt
        
        # Add tool instructions if bot uses tools
        if self.bot.uses_tools and self.memory:
            # Include search tools only if search client is available
            include_search = self.search_client is not None
            tools_prompt = get_tools_prompt(include_search_tools=include_search)
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
            logger.debug(f"Bot '{self.bot.name}' has tool calling enabled (search={include_search})")
        
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
        """Initialize OpenAI-compatible client."""
        model_type = self.model_definition.get("type")
        model_id = self.model_definition.get("model_id")
        
        if model_type != "openai":
            raise ValueError(
                f"Only 'openai' model type is supported. Got '{model_type}'. "
                "Use an OpenAI-compatible server for local models."
            )
        
        if not model_id:
            raise ValueError(f"Missing 'model_id' in definition for '{self.resolved_model_alias}'")
        
        base_url = self.model_definition.get("base_url")
        api_key = self.model_definition.get("api_key")
        
        return OpenAIClient(model_id, config=self.config, base_url=base_url, api_key=api_key)

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
                # Memory extraction for non-tool bots (tool bots store via tools)
                if self.memory and not self.bot.uses_tools:
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
        
        # Build system prompt parts
        system_parts = []
        if self.config.SYSTEM_MESSAGE:
            system_parts.append(self.config.SYSTEM_MESSAGE)
        
        # Add tool instructions OR memory context (not both)
        if self.bot.uses_tools and self.memory:
            system_parts.append(get_tools_prompt())
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
            
            memory_results = self.memory.search(prompt, n_results=n_results, min_relevance=min_relevance)
            
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
            from .memory.context_builder import build_memory_context_string
            
            # Get user's display name from profile if available
            user_name = None
            if self.profile_manager:
                try:
                    profile, _ = self.profile_manager.get_or_create_profile(
                        EntityType.USER, self.user_id
                    )
                    user_name = profile.display_name
                except Exception:
                    pass
                    
            return build_memory_context_string(unique_memories, user_name=user_name)
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""

    def _trigger_memory_extraction(self, user_prompt: str, assistant_response: str):
        """Trigger background memory extraction (non-blocking)."""
        if not self.memory:
            return
        
        def extract():
            try:
                from .service import ServiceClient
                from .service.tasks import create_extraction_task
                
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
                    client.submit_task(task)
            except Exception as e:
                logger.debug(f"Memory extraction failed: {e}")
        
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

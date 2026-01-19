"""
FastAPI-based background service for ask_llm.

Provides:
- OpenAI-compatible chat completions API
- Background task processing (memory extraction, compaction)
- Health and status endpoints

Run with: python -m ask_llm.service.server
Or: uvicorn ask_llm.service.server:app --host 0.0.0.0 --port 8642
"""

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from queue import PriorityQueue
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..utils.config import Config
from ..bots import get_bot, strip_emotes, StreamingEmoteFilter
from .tasks import Task, TaskResult, TaskStatus, TaskType
from .logging import (
    ServiceLogger,
    RequestContext,
    setup_service_logging,
    generate_request_id,
    get_service_logger,
)

log = get_service_logger(__name__)

# Configuration
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"


def _is_tcp_listening(host: str, port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


_memory_mcp_thread: threading.Thread | None = None


def _ensure_memory_mcp_server(config: Config) -> None:
    """Ensure an MCP memory server is running and configure the service to use it.

    This makes memory retrieval happen via MCP tool calls (e.g. tools/search_memories),
    which can be logged distinctly from embedded DB access.
    """
    global _memory_mcp_thread

    # Default to local MCP memory server for llm-service if not configured.
    if not getattr(config, "MEMORY_SERVER_URL", None):
        config.MEMORY_SERVER_URL = "http://127.0.0.1:8001"

    parsed = urlparse(config.MEMORY_SERVER_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8001

    # If something is already listening, assume it's the memory MCP server.
    if _is_tcp_listening(host, port):
        log.info("Memory MCP server already listening at %s", config.MEMORY_SERVER_URL)
        return

    # Start the MCP memory server in-process (HTTP transport) on a daemon thread.
    def _run():
        try:
            from ..memory_server.server import run_server
            run_server(transport="streamable-http", host=host, port=port)
        except Exception as e:
            log.error("Failed to start MCP memory server: %s", e)

    _memory_mcp_thread = threading.Thread(target=_run, daemon=True, name="memory-mcp")
    _memory_mcp_thread.start()
    log.info("Started MCP memory server at %s", config.MEMORY_SERVER_URL)


# =============================================================================
# Pydantic Models for OpenAI-compatible API
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str | None = None  # Optional, will use service default if not specified
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = None
    # ask_llm extensions
    bot_id: str | None = Field(default=None, description="Bot personality to use")
    augment_memory: bool = Field(default=True, description="Whether to augment with memory context")
    extract_memory: bool = Field(default=True, description="Whether to extract memories from response")


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ask_llm"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: list[ModelInfo]


class TaskSubmitRequest(BaseModel):
    """Request to submit a background task."""
    task_type: str
    payload: dict[str, Any]
    bot_id: str | None = None  # Will use config DEFAULT_BOT if not specified
    user_id: str = "default"
    priority: int = 0


class TaskSubmitResponse(BaseModel):
    """Response after submitting a task."""
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    result: Any | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class ServiceStatusResponse(BaseModel):
    """Service health and status."""
    status: str = "ok"
    version: str = SERVICE_VERSION
    uptime_seconds: float
    tasks_processed: int
    tasks_pending: int
    models_loaded: list[str] = []


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str = "ok"


class HistoryMessage(BaseModel):
    """A message in the conversation history."""
    role: str
    content: str
    timestamp: float


class HistoryResponse(BaseModel):
    """Response for conversation history."""
    bot_id: str
    messages: list[HistoryMessage]
    total_count: int


class HistoryClearResponse(BaseModel):
    """Response for clearing history."""
    success: bool
    message: str


# Memory Management Models
class MemoryItem(BaseModel):
    """A memory item."""
    id: str | None = None
    content: str
    importance: float = 0.5
    relevance: float | None = None
    tags: list[str] = []
    created_at: float | None = None
    last_accessed: float | None = None
    access_count: int = 0
    source_message_ids: list[str] = []


class MemorySearchRequest(BaseModel):
    """Request for memory search."""
    query: str
    method: str = "all"  # text, embedding, high-importance, all
    limit: int = 10
    min_importance: float = 0.0
    bot_id: str | None = None


class MemorySearchResponse(BaseModel):
    """Response for memory search."""
    bot_id: str
    method: str
    query: str
    results: list[MemoryItem]
    total_count: int


class MemoryStatsResponse(BaseModel):
    """Memory statistics."""
    bot_id: str
    messages: dict
    memories: dict


class MemoryForgetRequest(BaseModel):
    """Request to forget messages."""
    count: int | None = None  # forget recent N
    minutes: int | None = None  # forget last N minutes


class MemoryForgetResponse(BaseModel):
    """Response for forget operation."""
    success: bool
    messages_ignored: int
    memories_deleted: int
    message: str


class MemoryRestoreResponse(BaseModel):
    """Response for restore operation."""
    success: bool
    messages_restored: int
    message: str


class MemoryDeleteResponse(BaseModel):
    """Response for deleting a specific memory."""
    success: bool
    memory_id: str
    message: str


class MessagePreview(BaseModel):
    """Preview of a message for confirmation."""
    id: str  # UUID or int, stored as string
    role: str
    content: str
    timestamp: float | None = None


class MessagesPreviewResponse(BaseModel):
    """Response with message previews."""
    bot_id: str
    messages: list[MessagePreview]
    total_count: int


class RegenerateEmbeddingsResponse(BaseModel):
    """Response for regenerate embeddings operation."""
    success: bool
    updated: int
    failed: int
    embedding_dim: int | None = None
    message: str


class ConsolidateRequest(BaseModel):
    """Request for memory consolidation."""
    dry_run: bool = False
    similarity_threshold: float | None = None


class ConsolidateResponse(BaseModel):
    """Response for consolidation operation."""
    success: bool
    dry_run: bool
    clusters_found: int
    clusters_merged: int
    memories_consolidated: int
    new_memories_created: int
    errors: list[str] = []
    message: str


class RawCompletionRequest(BaseModel):
    """Request for raw LLM completion without bot/memory overhead.
    
    Use this for utility tasks like memory consolidation, summarization, etc.
    """
    prompt: str
    system: str | None = None
    model: str | None = None  # Uses service default if not specified
    max_tokens: int = 500
    temperature: float = 0.7


class RawCompletionResponse(BaseModel):
    """Response from raw LLM completion."""
    content: str
    model: str
    tokens: int | None = None
    elapsed_ms: float


# =============================================================================
# Background Service
# =============================================================================

class BackgroundService:
    """
    The main background service that processes async tasks.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self.tasks_processed = 0
        self._task_queue: PriorityQueue = PriorityQueue()
        self._results: dict[str, TaskResult] = {}
        self._result_events: dict[str, asyncio.Event] = {}
        self._shutdown_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        
        # Cached AskLLM instances keyed by (model_alias, bot_id, user_id)
        self._ask_llm_cache: dict[tuple[str, str, str], Any] = {}
        self._cache_lock = asyncio.Lock()
        
        # Cached LLM clients keyed by model_alias only
        # This prevents loading the same model (especially GGUF) multiple times
        # when different bot contexts need the same underlying model
        self._client_cache: dict[str, Any] = {}
        
        # Lock to serialize LLM calls - prevents CUDA crashes from concurrent access
        # llama-cpp-python is NOT thread-safe for concurrent inference
        self._llm_lock = asyncio.Lock()
        
        # Single-threaded executor for LLM calls - ensures only one runs at a time
        from concurrent.futures import ThreadPoolExecutor
        self._llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        
        # Cancellation support: when a new request comes in, cancel the current one
        # This handles the case where UI sends partial transcriptions that build up
        self._current_generation_cancel: threading.Event | None = None
        self._generation_done: threading.Event | None = None  # Signals when generation finishes
        self._cancel_lock = threading.Lock()
        
        # Memory client cache keyed by bot_id
        self._memory_clients: dict[tuple[str, str], Any] = {}
        
        # Model configuration
        self._available_models: list[str] = []
        self._default_model: str | None = None
        self._default_bot: str = config.DEFAULT_BOT or "nova"
        self._load_available_models()
    
    def _load_available_models(self):
        """Load list of available models from config."""
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())
        log.debug(f"Loaded {len(self._available_models)} models from config")
        
        # Set default model from config or use first available
        self._default_model = self.config.SERVICE_MODEL
        if not self._default_model and self._available_models:
            self._default_model = self._available_models[0]
    
    def _start_generation(self) -> tuple[threading.Event, threading.Event]:
        """Start a new generation, cancelling and waiting for any in-progress one.
        
        Returns:
            tuple of (cancel_event, done_event):
            - cancel_event: The generation should check this periodically and abort if set
            - done_event: The generation MUST set this when complete (in finally block)
        
        This ensures only one generation runs at a time by:
        1. Signalling the previous generation to cancel
        2. Waiting for it to actually finish (up to 5 seconds)
        3. Then allowing the new generation to start
        """
        with self._cancel_lock:
            # Cancel any existing generation and wait for it to finish
            if self._current_generation_cancel is not None:
                log.info("New request received - cancelling previous generation")
                self._current_generation_cancel.set()
                
                # Wait for the previous generation to signal it's done
                if self._generation_done is not None:
                    # Release lock while waiting to avoid deadlock
                    done_event = self._generation_done
                    self._cancel_lock.release()
                    try:
                        # Wait up to 5 seconds for previous generation to stop
                        done_event.wait(timeout=5.0)
                    finally:
                        self._cancel_lock.acquire()
            
            # Create new events for this generation
            cancel_event = threading.Event()
            done_event = threading.Event()
            self._current_generation_cancel = cancel_event
            self._generation_done = done_event
            return cancel_event, done_event
    
    def _end_generation(self, cancel_event: threading.Event, done_event: threading.Event):
        """Mark a generation as complete."""
        # Signal that we're done FIRST (before acquiring lock)
        done_event.set()
        
        with self._cancel_lock:
            # Only clear if this is still the current generation
            if self._current_generation_cancel is cancel_event:
                self._current_generation_cancel = None
                self._generation_done = None
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def get_memory_client(self, bot_id: str, user_id: str | None = None):
        """Get or create memory client for a bot/user pair."""
        if user_id is None:
            user_id = getattr(self.config, "DEFAULT_USER", "default")

        cache_key = (bot_id, user_id)

        if cache_key not in self._memory_clients:
            try:
                from ..memory_server.client import get_memory_client
                self._memory_clients[cache_key] = get_memory_client(
                    config=self.config,
                    bot_id=bot_id,
                    user_id=user_id,
                    server_url=getattr(self.config, "MEMORY_SERVER_URL", None),
                )
                log.memory_operation("client_init", bot_id, details="MemoryClient created")
            except Exception as e:
                log.warning(f"Memory client unavailable for {bot_id}: {e}")
                self._memory_clients[cache_key] = None
        return self._memory_clients.get(cache_key)
    
    def _get_ask_llm(self, model_alias: str, bot_id: str, user_id: str, local_mode: bool = False):
        """Get or create an AskLLM instance with caching."""
        from .core import ServiceAskLLM
        
        cache_key = (model_alias, bot_id, user_id)
        
        if cache_key in self._ask_llm_cache:
            log.cache_hit("ask_llm", f"{model_alias}/{bot_id}/{user_id}")
            return self._ask_llm_cache[cache_key]
        
        log.cache_miss("ask_llm", f"{model_alias}/{bot_id}/{user_id}")
        
        # Check if we already have a client for this model in the client cache
        # If so, we can potentially reuse it (though AskLLM still needs its own instance)
        existing_client = self._client_cache.get(model_alias)
        
        # Get model type for logging
        model_def = self.config.defined_models.get("models", {}).get(model_alias, {})
        model_type = model_def.get("type", "unknown")
        
        # Only log model loading if we don't have the client cached
        if not existing_client:
            log.model_loading(model_alias, model_type, cached=False)
        else:
            log.model_loading(model_alias, model_type, cached=True)
        load_start = time.time()
        
        try:
            # Create a copy of config for each ServiceAskLLM instance
            # This is necessary because it modifies config.SYSTEM_MESSAGE
            # based on the bot's system prompt
            instance_config = self.config.model_copy(deep=True)
            ask_llm = ServiceAskLLM(
                resolved_model_alias=model_alias,
                config=instance_config,
                local_mode=local_mode,
                bot_id=bot_id,
                user_id=user_id,
            )
            self._ask_llm_cache[cache_key] = ask_llm
            
            # Also cache the client for future reuse by extraction tasks
            if model_alias not in self._client_cache:
                self._client_cache[model_alias] = ask_llm.client
            # Note: ServiceAskLLM already logs model_loaded, don't duplicate
            
        except Exception as e:
            log.model_error(model_alias, str(e))
            raise
        
        return self._ask_llm_cache[cache_key]
    
    def get_client(self, model_alias: str):
        """Get LLM client for a given model (for extraction tasks).
        
        Uses a dedicated client cache to avoid reloading models.
        GGUF models especially are expensive to load into VRAM,
        so we cache the client independently from AskLLM instances.
        """
        if model_alias in self._client_cache:
            log.cache_hit("llm_client", model_alias)
            return self._client_cache[model_alias]
        
        log.cache_miss("llm_client", model_alias)
        
        # Check if we already have an AskLLM instance with this model
        # and can reuse its client
        for (cached_model, _, _), ask_llm in self._ask_llm_cache.items():
            if cached_model == model_alias:
                log.debug(f"Reusing client from existing AskLLM instance for '{model_alias}'")
                self._client_cache[model_alias] = ask_llm.client
                return ask_llm.client
        
        # Need to create a new client - use spark bot (no memory overhead)
        log.debug(f"Creating new client for model '{model_alias}' (extraction context)")
        ask_llm = self._get_ask_llm(model_alias, "spark", "system", local_mode=True)
        self._client_cache[model_alias] = ask_llm.client
        return ask_llm.client
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        """
        Handle an OpenAI-compatible chat completion request.
        
        This is the main entry point for the API. It:
        1. Uses ask_llm's internal history + memory for context
        2. Augments with bot system prompt and memory context (if enabled)
        3. Runs blocking LLM calls in a thread pool
        4. Stores messages and extracts memories for future use
        """
        from ..models.message import Message
        
        # Create request context for logging
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path="/v1/chat/completions",
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=False,
        )
        
        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))
        
        # Resolve model - use default if not specified or not found
        model_alias: str | None = request.model
        if not model_alias or model_alias not in self._available_models:
            if model_alias and model_alias in ("mira", "nova", "sage") and self._default_model:
                # Bot name passed as model - use default model
                model_alias = self._default_model
            elif self._default_model:
                if model_alias:
                    log.warning(f"Model '{request.model}' not found, using default: {self._default_model}")
                model_alias = self._default_model
            else:
                log.api_error(ctx, f"Model '{request.model}' not found", 400)
                raise ValueError(f"Model '{request.model}' not found. Available: {', '.join(self._available_models)}")
        
        ctx.model = model_alias
        bot_id = request.bot_id or self._default_bot
        user_id = request.user or "default"
        local_mode = not request.augment_memory
        
        # Debug: Show memory settings
        log.debug(
            f"Memory settings: augment_memory={request.augment_memory}, "
            f"local_mode={local_mode}, bot={bot_id}, user={user_id}"
        )
        
        # Get cached AskLLM instance
        ask_llm = self._get_ask_llm(model_alias, bot_id, user_id, local_mode)
        
        # Get the user's prompt (last user message)
        user_prompt = ""
        for m in reversed(request.messages):
            if m.role == "user":
                user_prompt = m.content or ""
                break
        
        if not user_prompt:
            raise ValueError("No user message found in request")
        
        # Start new generation (cancels and waits for any previous one)
        cancel_event, done_event = self._start_generation()
        
        try:
            # Run the blocking query in single-thread executor
            loop = asyncio.get_event_loop()
            llm_start_time = time.time()
            cancelled = False
            
            def _do_query():
                nonlocal cancelled
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled = True
                    return ""
                
                # Prepare messages with history and memory context
                prepared_messages = ask_llm.prepare_messages_for_query(user_prompt)
                
                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(prepared_messages)
                
                # Execute the query with prepared messages
                response, tool_context = ask_llm._execute_llm_query(
                    prepared_messages,
                    plaintext_output=True,
                    stream=False,
                )
                
                # Check if cancelled during generation
                if cancel_event.is_set():
                    log.info("Generation cancelled - newer request received")
                    cancelled = True
                    return ""
                
                # Finalize (add to history, trigger memory extraction)
                ask_llm.finalize_response(user_prompt, response, tool_context)
                
                return response
            
            response_text = await loop.run_in_executor(self._llm_executor, _do_query)
            llm_elapsed_ms = (time.time() - llm_start_time) * 1000
            
            # If cancelled, return empty response (the new request will handle it)
            if cancelled:
                return ChatCompletionResponse(
                    model=model_alias,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=""),
                            finish_reason="cancelled",
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )
        finally:
            self._end_generation(cancel_event, done_event)
        
        # Post-process for voice_optimized bots (strip emotes for TTS)
        bot = get_bot(bot_id)
        if bot and bot.voice_optimized:
            original_len = len(response_text)
            response_text = strip_emotes(response_text)
            if len(response_text) != original_len:
                log.debug(f"Stripped emotes for TTS: {original_len} -> {len(response_text)} chars")
        
        # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        prompt_text = " ".join(m.content or "" for m in request.messages)
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(response_text) // 4
        total_tokens = prompt_tokens + completion_tokens
        
        # Log response (verbose shows content summary with tokens/sec)
        log.llm_response(response_text, tokens=completion_tokens, elapsed_ms=llm_elapsed_ms)
        log.api_response(ctx, status=200, tokens=total_tokens)
        
        # Build response
        response = ChatCompletionResponse(
            model=model_alias,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )
        
        return response

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Handle a streaming chat completion request.
        
        Uses ask_llm's internal history + memory for context.
        Yields Server-Sent Events (SSE) formatted chunks.
        """
        import json
        from ..models.message import Message
        
        # Create request context for logging
        ctx = RequestContext(
            request_id=generate_request_id(),
            method="POST",
            path="/v1/chat/completions",
            model=request.model,
            bot_id=request.bot_id,
            user_id=request.user,
            stream=True,
        )
        
        # Log incoming request (verbose mode will show the full payload)
        log.api_request(ctx, request.model_dump(exclude_none=True))
        
        # Resolve model - use default if not specified or not found
        model_alias: str | None = request.model
        if not model_alias or model_alias not in self._available_models:
            if model_alias and model_alias in ("mira", "nova", "sage") and self._default_model:
                # Bot name passed as model - use default model
                model_alias = self._default_model
            elif self._default_model:
                if model_alias:
                    log.warning(f"Model '{request.model}' not found, using default: {self._default_model}")
                model_alias = self._default_model
            else:
                log.api_error(ctx, f"Model '{request.model}' not found", 400)
                raise ValueError(f"Model '{request.model}' not found.")
        
        bot_id = request.bot_id or self._default_bot
        user_id = request.user or "default"
        local_mode = not request.augment_memory
        
        # Get cached AskLLM instance
        ask_llm = self._get_ask_llm(model_alias, bot_id, user_id, local_mode)
        
        # Get the user's prompt (last user message)
        user_prompt = ""
        for m in reversed(request.messages):
            if m.role == "user":
                user_prompt = m.content or ""
                break
        
        if not user_prompt:
            raise ValueError("No user message found in request")
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        timing_holder = [0.0, 0.0]  # [start_time, end_time]
        cancelled_holder = [False]  # Track if we were cancelled
        
        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()
        
        # Start new generation (cancels and waits for any previous one)
        cancel_event, done_event = self._start_generation()
        
        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
            try:
                # Check if already cancelled before starting
                if cancel_event.is_set():
                    cancelled_holder[0] = True
                    return
                
                # Use ask_llm.prepare_messages_for_query to get full context
                # (history from DB + memory + system prompt)
                messages = ask_llm.prepare_messages_for_query(user_prompt)
                
                # Log what we're sending to the LLM (verbose mode)
                log.llm_context(messages)
                
                # Track when first token arrives
                timing_holder[0] = time.time()
                
                # Choose streaming method based on whether bot uses tools
                if ask_llm.bot.uses_tools and ask_llm.memory:
                    from ..tools import stream_with_tools
                    log.debug("Using stream_with_tools for tool-enabled bot")
                    
                    def stream_fn(msgs):
                        return ask_llm.client.stream_raw(msgs)
                    
                    stream_iter = stream_with_tools(
                        messages=messages,
                        stream_fn=stream_fn,
                        memory_client=ask_llm.memory,
                        profile_manager=ask_llm.profile_manager,
                        search_client=ask_llm.search_client,
                        user_id=ask_llm.user_id,
                        bot_id=ask_llm.bot_id,
                    )
                else:
                    stream_iter = ask_llm.client.stream_raw(messages)
                
                # Stream chunks to queue
                for chunk in stream_iter:
                    # Check for cancellation - new request came in
                    if cancel_event.is_set():
                        log.info("Generation cancelled - newer request received")
                        cancelled_holder[0] = True
                        return
                    
                    full_response_holder[0] += chunk
                    # Put chunk in queue - use call_soon_threadsafe with captured loop
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                
                timing_holder[1] = time.time()
                
                # Finalize: add response to history and trigger memory extraction
                # Only if we weren't cancelled
                if full_response_holder[0] and not cancel_event.is_set():
                    # Calculate elapsed time and log with tokens/sec
                    elapsed_ms = (timing_holder[1] - timing_holder[0]) * 1000
                    log.llm_response(full_response_holder[0], elapsed_ms=elapsed_ms)
                    ask_llm.finalize_response(user_prompt, full_response_holder[0])
                    
            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # Sentinel
        
        try:
            # Check if this bot needs emote filtering for TTS
            bot = get_bot(bot_id)
            emote_filter = StreamingEmoteFilter() if (bot and bot.voice_optimized) else None
            
            # Start streaming in single-thread executor
            loop.run_in_executor(self._llm_executor, _stream_to_queue)
            
            # Yield SSE chunks
            while True:
                chunk = await chunk_queue.get()
                
                if chunk is None:
                    # Stream complete - flush any buffered content from emote filter
                    if emote_filter:
                        final_chunk = emote_filter.flush()
                        if final_chunk:
                            data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_alias,
                                "choices": [{"index": 0, "delta": {"content": final_chunk}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    
                    # Send final chunk
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                if isinstance(chunk, Exception):
                    # Error occurred
                    raise chunk
                
                # Apply emote filter for voice_optimized bots
                if emote_filter:
                    chunk = emote_filter.process(chunk)
                    if not chunk:
                        # Chunk was filtered out or buffered
                        continue
                
                # Normal chunk
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            # Mark generation as complete
            self._end_generation(cancel_event, done_event)

    async def process_task(self, task: Task) -> TaskResult:
        """Process a single background task."""
        start_time = time.time()
        task_type_str = task.task_type.value
        
        log.debug(f"Processing task {task.task_id[:8]} ({task_type_str})")
        
        try:
            if task.task_type == TaskType.MEMORY_EXTRACTION:
                result = await self._process_extraction(task)
            elif task.task_type == TaskType.CONTEXT_COMPACTION:
                result = await self._process_compaction(task)
            elif task.task_type == TaskType.EMBEDDING_GENERATION:
                result = await self._process_embeddings(task)
            elif task.task_type == TaskType.MEANING_UPDATE:
                result = await self._process_meaning_update(task)
            elif task.task_type == TaskType.MEMORY_MAINTENANCE:
                result = await self._process_maintenance(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            log.task_completed(task.task_id, task_type_str, elapsed_ms, result)
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                processing_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log.task_failed(task.task_id, task_type_str, str(e), elapsed_ms)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                processing_time_ms=elapsed_ms,
            )
    
    async def _process_extraction(self, task: Task) -> dict:
        """Process a memory extraction task.
        
        Uses the same model that was used for the chat to avoid loading
        multiple models into VRAM. The model is passed in the task payload.
        """
        from ..memory.extraction import MemoryExtractionService
        
        messages = task.payload.get("messages", [])
        bot_id = task.bot_id
        user_id = task.user_id
        
        log.info(f"[Extraction] Processing task for bot={bot_id} user={user_id} with {len(messages)} messages")
        
        # Get the model from task payload (passed from chat request)
        # This ensures we use the same model that handled the chat
        model_to_use = task.payload.get("model")
        
        if not model_to_use:
            # Fallback: no model specified in task, skip LLM extraction
            log.debug("No model specified in extraction task - using non-LLM extraction")
            extraction_service = MemoryExtractionService(llm_client=None)
            use_llm = False
        else:
            # Get client from cache - should already be loaded from chat
            extraction_client = None
            if model_to_use in self._client_cache:
                extraction_client = self._client_cache[model_to_use]
                log.debug(f"Reusing cached client for extraction: {model_to_use}")
            else:
                # Check if any AskLLM instance has this model loaded
                for (cached_model, _, _), ask_llm in self._ask_llm_cache.items():
                    if cached_model == model_to_use:
                        extraction_client = ask_llm.client
                        self._client_cache[model_to_use] = extraction_client
                        log.debug(f"Reusing client from AskLLM instance for extraction: {model_to_use}")
                        break
            
            if not extraction_client:
                log.warning(f"Model '{model_to_use}' not in cache - skipping LLM extraction to avoid reload")
                extraction_client = None
            
            extraction_service = MemoryExtractionService(llm_client=extraction_client)
            use_llm = extraction_client is not None
        
        # Run extraction in the SAME single-threaded executor as chat completions
        # This ensures extraction waits for any in-flight chat to complete
        # The executor has max_workers=1, so operations are serialized
        loop = asyncio.get_event_loop()
        
        try:
            facts = await loop.run_in_executor(
                self._llm_executor,  # Use the single-threaded LLM executor
                lambda: extraction_service.extract_from_conversation(messages, use_llm=use_llm)
            )
        except Exception as e:
            # Catch any llama.cpp state corruption errors
            log.warning(f"Extraction failed (will retry without LLM): {e}")
            # Fallback to non-LLM extraction
            extraction_service_fallback = MemoryExtractionService(llm_client=None)
            facts = await loop.run_in_executor(
                self._llm_executor,
                lambda: extraction_service_fallback.extract_from_conversation(messages, use_llm=False)
            )
            use_llm = False
        
        if not facts:
            log.info(f"[Extraction] No facts extracted from messages")
            return {"facts_extracted": 0, "facts_stored": 0, "llm_used": use_llm}
        
        log.info(f"[Extraction] Extracted {len(facts)} facts, storing...")
        
        memory_client = self.get_memory_client(bot_id, user_id)
        stored_count = 0
        profile_count = 0
        
        if memory_client:
            min_importance = getattr(self.config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.3)
            profile_enabled = getattr(self.config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)
            
            for fact in facts:
                log.debug(f"[Extraction] Fact: '{fact.content[:50]}...' importance={fact.importance:.2f} tags={fact.tags}")
                if fact.importance >= min_importance:
                    try:
                        memory_client.add_memory(
                            content=fact.content,
                            tags=fact.tags,
                            importance=fact.importance,
                            source_message_ids=fact.source_message_ids,
                        )
                        stored_count += 1
                        if profile_enabled:
                            from ..memory_server.extraction import extract_profile_attributes_from_fact
                            if extract_profile_attributes_from_fact(
                                fact=fact,
                                user_id=user_id,
                                config=self.config,
                            ):
                                profile_count += 1
                    except Exception as e:
                        log.warning(f"Failed to store memory: {e}")
        
        log.info(f"[Extraction] Stored {stored_count} memories, {profile_count} profile attributes")
        log.memory_operation("extraction", bot_id, count=stored_count, details=f"extracted={len(facts)}, profiles={profile_count}, llm={use_llm}")
        return {"facts_extracted": len(facts), "facts_stored": stored_count, "profile_attrs": profile_count, "llm_used": use_llm}
    
    async def _process_compaction(self, task: Task) -> dict:
        """Process a context compaction task."""
        # TODO: Implement with summarization model
        return {"compacted": False, "reason": "Not yet implemented"}
    
    async def _process_embeddings(self, task: Task) -> dict:
        """Process an embedding generation task."""
        # TODO: Implement with embedding model
        return {"embeddings_generated": 0, "reason": "Not yet implemented"}

    async def _process_meaning_update(self, task: Task) -> dict:
        """Process a meaning update task (via MCP tools)."""
        bot_id = task.bot_id
        payload = task.payload
        memory_id = payload.get("memory_id")
        if not memory_id:
            return {"updated": False, "reason": "No memory_id provided"}

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"updated": False, "reason": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: memory_client.update_memory_meaning(
                memory_id=memory_id,
                intent=payload.get("intent"),
                stakes=payload.get("stakes"),
                emotional_charge=payload.get("emotional_charge"),
                recurrence_keywords=payload.get("recurrence_keywords"),
                updated_tags=payload.get("updated_tags"),
            ),
        )
        return {"updated": bool(success), "memory_id": memory_id}

    async def _process_maintenance(self, task: Task) -> dict:
        """Process a unified memory maintenance task.
        
        Uses a cached LLM client if available, otherwise runs without LLM.
        Will NOT load a new model to avoid VRAM conflicts.
        """
        bot_id = task.bot_id
        payload = task.payload

        memory_client = self.get_memory_client(bot_id)
        if not memory_client:
            return {"error": "Memory client unavailable"}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: memory_client.run_maintenance(
                run_consolidation=payload.get("run_consolidation", True),
                run_recurrence_detection=payload.get("run_recurrence_detection", True),
                run_decay_pruning=payload.get("run_decay_pruning", False),
                run_orphan_cleanup=payload.get("run_orphan_cleanup", False),
                dry_run=payload.get("dry_run", False),
            ),
        )
        return result
    
    def submit_task(self, task: Task) -> str:
        """Submit a task to the processing queue."""
        from dataclasses import dataclass, field as dataclass_field
        
        @dataclass(order=True)
        class PrioritizedTask:
            priority: int
            timestamp: float
            task: Task = dataclass_field(compare=False)
        
        prioritized = PrioritizedTask(
            priority=-task.priority,
            timestamp=time.time(),
            task=task,
        )
        self._task_queue.put(prioritized)
        self._result_events[task.task_id] = asyncio.Event()
        return task.task_id
    
    def get_result(self, task_id: str) -> TaskResult | None:
        """Get the result of a completed task."""
        return self._results.get(task_id)
    
    async def wait_for_result(self, task_id: str, timeout: float = 30.0) -> TaskResult | None:
        """Wait for a task result with timeout."""
        event = self._result_events.get(task_id)
        if not event:
            return self._results.get(task_id)
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._results.get(task_id)
        except asyncio.TimeoutError:
            return None
    
    async def worker_loop(self):
        """Main worker loop that processes tasks from the queue."""
        log.info("Background task worker started")
        
        while not self._shutdown_event.is_set():
            try:
                try:
                    prioritized = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._task_queue.get(timeout=1.0)
                    )
                except Exception:
                    continue
                
                task = prioritized.task
                log.task_submitted(task.task_id, task.task_type.value, task.bot_id)
                
                result = await self.process_task(task)
                self._results[task.task_id] = result
                self.tasks_processed += 1
                
                if task.task_id in self._result_events:
                    self._result_events[task.task_id].set()
                
                # Cleanup old results
                if len(self._results) > 1000:
                    oldest = sorted(self._results.keys())[:100]
                    for key in oldest:
                        self._results.pop(key, None)
                        self._result_events.pop(key, None)
                
            except Exception as e:
                log.exception(f"Worker loop error: {e}")
                await asyncio.sleep(1)
        
        log.info("Background task worker stopped")
    
    def start_worker(self):
        """Start the background worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker_loop())
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        log.shutdown()
        self._shutdown_event.set()
        if self._worker_task:
            self._worker_task.cancel()
    
    def get_status(self) -> ServiceStatusResponse:
        """Get service status."""
        return ServiceStatusResponse(
            uptime_seconds=self.uptime_seconds,
            tasks_processed=self.tasks_processed,
            tasks_pending=self._task_queue.qsize(),
            models_loaded=self._available_models,
        )


# =============================================================================
# FastAPI Application
# =============================================================================

# Global service instance
_service: BackgroundService | None = None


def get_service() -> BackgroundService:
    """Get the background service instance."""
    global _service
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for startup/shutdown."""
    global _service
    
    # Startup
    config = Config()

    # Prefer MCP tool-based memory retrieval for llm-service.
    # This ensures memory retrieval happens via MCP tools and can be logged clearly.
    _ensure_memory_mcp_server(config)

    _service = BackgroundService(config)
    _service.start_worker()
    
    # Log startup with rich formatting
    log.startup(
        version=SERVICE_VERSION,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        models=_service._available_models,
        default_model=_service._default_model,
    )

    log.info(
        "Memory mode: %s (%s)",
        "mcp" if getattr(config, "MEMORY_SERVER_URL", None) else "embedded",
        getattr(config, "MEMORY_SERVER_URL", ""),
    )
    
    yield
    
    # Shutdown
    await _service.shutdown()


# Create FastAPI app
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    
    app = FastAPI(
        title="ask_llm API",
        description="OpenAI-compatible API with integrated memory system",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    
    # -------------------------------------------------------------------------
    # Health & Status Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()
    
    @app.get("/status", response_model=ServiceStatusResponse, tags=["System"])
    async def get_status():
        """Get detailed service status."""
        return get_service().get_status()
    
    # -------------------------------------------------------------------------
    # OpenAI-Compatible Endpoints
    # -------------------------------------------------------------------------
    
    @app.get("/v1/models", response_model=ModelsResponse, tags=["OpenAI Compatible"])
    async def list_models():
        """List available models (OpenAI-compatible)."""
        service = get_service()
        models = [
            ModelInfo(id=alias)
            for alias in service._available_models
        ]
        return ModelsResponse(data=models)
    
    @app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
    async def chat_completions(request: ChatCompletionRequest):
        """
        Create a chat completion (OpenAI-compatible).
        
        Supports all standard OpenAI parameters plus ask_llm extensions:
        - `bot_id`: Bot personality to use (default: nova)
        - `augment_memory`: Whether to include memory context (default: true)
        - `extract_memory`: Whether to extract memories from response (default: true)
        """
        from fastapi.responses import StreamingResponse
        
        service = get_service()
        
        if request.stream:
            # Streaming response
            try:
                return StreamingResponse(
                    service.chat_completion_stream(request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                log.exception("Streaming chat completion failed")
                raise HTTPException(status_code=500, detail=str(e))
        
        try:
            response = await service.chat_completion(request)
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            log.exception("Chat completion failed")
            raise HTTPException(status_code=500, detail=str(e))
    
    # -------------------------------------------------------------------------
    # Task Management Endpoints
    # -------------------------------------------------------------------------
    
    @app.post("/v1/tasks", response_model=TaskSubmitResponse, tags=["Tasks"])
    async def submit_task(request: TaskSubmitRequest):
        """Submit a background task for processing."""
        service = get_service()
        
        try:
            task_type = TaskType(request.task_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type: {request.task_type}"
            )
        
        task = Task(
            task_type=task_type,
            payload=request.payload,
            bot_id=request.bot_id or service._default_bot,
            user_id=request.user_id,
            priority=request.priority,
        )
        
        task_id = service.submit_task(task)
        return TaskSubmitResponse(task_id=task_id)
    
    @app.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
    async def get_task_status(
        task_id: str,
        wait: bool = Query(False, description="Wait for task completion"),
        timeout: float = Query(30.0, description="Wait timeout in seconds"),
    ):
        """Get the status of a submitted task."""
        service = get_service()
        
        if wait:
            result = await service.wait_for_result(task_id, timeout)
        else:
            result = service.get_result(task_id)
        
        if result:
            return TaskStatusResponse(
                task_id=task_id,
                status=result.status.value,
                result=result.result,
                error=result.error,
                processing_time_ms=result.processing_time_ms,
            )
        else:
            return TaskStatusResponse(task_id=task_id, status="pending")

    # -------------------------------------------------------------------------
    # History Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/history", response_model=HistoryResponse, tags=["History"])
    async def get_history(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
        user_id: str = Query("default", description="User ID"),
        limit: int = Query(50, description="Maximum number of messages to return"),
    ):
        """Get conversation history for a bot."""
        service = get_service()
        
        effective_bot_id = bot_id or service._default_bot
        
        try:
            # Get or create an AskLLM instance to access history
            # Use a dummy model since we only need history access
            model_alias = list(service._available_models)[0] if service._available_models else None
            if not model_alias:
                raise HTTPException(status_code=500, detail="No models available")
            
            ask_llm = service._get_ask_llm(model_alias, effective_bot_id, user_id)
            
            # Get messages from history manager
            messages = ask_llm.history_manager.messages
            
            # Apply limit (from most recent)
            if limit > 0 and len(messages) > limit:
                messages = messages[-limit:]
            
            history_messages = [
                HistoryMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp
                )
                for msg in messages
                if msg.role != "system"  # Don't include system messages
            ]
            
            return HistoryResponse(
                bot_id=effective_bot_id,
                messages=history_messages,
                total_count=len(history_messages)
            )
        except Exception as e:
            log.error(f"Failed to get history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/history", response_model=HistoryClearResponse, tags=["History"])
    async def clear_history(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
        user_id: str = Query("default", description="User ID"),
    ):
        """Clear conversation history for a bot."""
        service = get_service()
        
        effective_bot_id = bot_id or service._default_bot
        
        try:
            model_alias = list(service._available_models)[0] if service._available_models else None
            if not model_alias:
                raise HTTPException(status_code=500, detail="No models available")
            
            ask_llm = service._get_ask_llm(model_alias, effective_bot_id, user_id)
            ask_llm.history_manager.clear_history()
            
            # Also remove from cache to force fresh state
            cache_key = (model_alias, effective_bot_id, user_id)
            if cache_key in service._ask_llm_cache:
                del service._ask_llm_cache[cache_key]
            
            return HistoryClearResponse(
                success=True,
                message=f"History cleared for bot '{effective_bot_id}'"
            )
        except Exception as e:
            log.error(f"Failed to clear history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Memory Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/memory/stats", response_model=MemoryStatsResponse, tags=["Memory"])
    async def get_memory_stats(
        bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    ):
        """Get memory statistics for a bot."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            stats = client.stats()
            return MemoryStatsResponse(
                bot_id=effective_bot_id,
                messages=stats.get("messages", {}),
                memories=stats.get("memories", {})
            )
        except Exception as e:
            log.error(f"Failed to get memory stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/search", response_model=MemorySearchResponse, tags=["Memory"])
    async def search_memory(request: MemorySearchRequest):
        """Search memories."""
        service = get_service()
        effective_bot_id = request.bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")
            
            results = []
            if request.method in ("embedding", "all"):
                # Semantic search
                memories = client.search(
                    request.query,
                    n_results=request.limit,
                    min_relevance=request.min_importance,
                )
                for mem in memories:
                    results.append(MemoryItem(
                        id=str(getattr(mem, "id", "")),
                        content=str(getattr(mem, "content", "")),
                        importance=float(getattr(mem, "importance", 0.5)),
                        relevance=getattr(mem, "relevance", None),
                        tags=list(getattr(mem, "tags", []) or []),
                        created_at=getattr(mem, "created_at", None),
                        access_count=0,
                    ))
            elif request.method == "high-importance":
                # Get high importance memories
                memories = client.list_memories(
                    limit=request.limit,
                    min_importance=request.min_importance or 0.7,
                )
                for mem in memories:
                    results.append(MemoryItem(
                        id=str(mem.get("id", "")),
                        content=mem.get("content", ""),
                        importance=mem.get("importance", 0.5),
                        tags=mem.get("tags", []),
                        created_at=mem.get("created_at"),
                    ))
            
            return MemorySearchResponse(
                bot_id=effective_bot_id,
                method=request.method,
                query=request.query,
                results=results,
                total_count=len(results)
            )
        except Exception as e:
            log.error(f"Failed to search memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory", response_model=MemorySearchResponse, tags=["Memory"])
    async def list_memories(
        bot_id: str = Query(None, description="Bot ID"),
        limit: int = Query(20, description="Max results"),
    ):
        """List all memories for a bot (ordered by importance)."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            memories = client.list_memories(limit=limit, min_importance=0.0)
            
            results = [
                MemoryItem(
                    id=str(mem.get("id", "")),
                    content=mem.get("content", ""),
                    importance=mem.get("importance", 0.5),
                    tags=mem.get("tags", []),
                    created_at=mem.get("created_at"),
                    access_count=mem.get("access_count", 0),
                )
                for mem in memories
            ]
            
            return MemorySearchResponse(
                bot_id=effective_bot_id,
                method="list",
                query="",
                results=results,
                total_count=len(results)
            )
        except Exception as e:
            log.error(f"Failed to list memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/memory/{memory_id}", response_model=MemoryDeleteResponse, tags=["Memory"])
    async def delete_memory(
        memory_id: str,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Delete a specific memory by ID."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            # The memory_id might be a prefix - try to find the full ID
            success = client.delete_memory(memory_id)
            
            if success:
                return MemoryDeleteResponse(
                    success=True,
                    memory_id=memory_id,
                    message=f"Memory '{memory_id}' deleted"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to delete memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/forget", response_model=MemoryForgetResponse, tags=["Memory"])
    async def forget_messages(
        request: MemoryForgetRequest,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Forget recent messages (soft delete)."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            if request.count:
                result = client.forget_recent_messages(request.count)
            elif request.minutes:
                result = client.forget_messages_since_minutes(request.minutes)
            else:
                raise HTTPException(status_code=400, detail="Must specify count or minutes")

            messages_ignored = int(result.get("messages_ignored", 0))
            memories_deleted = int(result.get("memories_deleted", 0))
            
            return MemoryForgetResponse(
                success=True,
                messages_ignored=messages_ignored,
                memories_deleted=memories_deleted,
                message=f"Ignored {messages_ignored} messages, deleted {memories_deleted} memories"
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to forget messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/restore", response_model=MemoryRestoreResponse, tags=["Memory"])
    async def restore_messages(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Restore ignored messages."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            restored = client.restore_ignored_messages()
            
            return MemoryRestoreResponse(
                success=True,
                messages_restored=restored,
                message=f"Restored {restored} messages"
            )
        except Exception as e:
            log.error(f"Failed to restore messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/recent", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_recent_messages(
        count: int = Query(10, description="Number of recent messages to preview"),
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview recent messages before forgetting."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_recent_messages(count)
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/minutes", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_messages_since_minutes(
        minutes: int = Query(..., description="Number of minutes to look back"),
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview messages from last N minutes before forgetting."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_messages_since_minutes(minutes)
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/memory/preview/ignored", response_model=MessagesPreviewResponse, tags=["Memory"])
    async def preview_ignored_messages(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Preview ignored messages before restoring."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            messages = client.preview_ignored_messages()
            
            return MessagesPreviewResponse(
                bot_id=effective_bot_id,
                messages=[
                    MessagePreview(
                        id=msg["id"],
                        role=msg.get("role", "?"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                    )
                    for msg in messages
                ],
                total_count=len(messages),
            )
        except Exception as e:
            log.error(f"Failed to preview ignored messages: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/regenerate-embeddings", response_model=RegenerateEmbeddingsResponse, tags=["Memory"])
    async def regenerate_embeddings(
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Regenerate embeddings for all memories."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = client.regenerate_embeddings()
            
            if "error" in result:
                return RegenerateEmbeddingsResponse(
                    success=False,
                    updated=0,
                    failed=0,
                    message=result["error"]
                )
            
            return RegenerateEmbeddingsResponse(
                success=True,
                updated=result.get("updated", 0),
                failed=result.get("failed", 0),
                embedding_dim=result.get("embedding_dim"),
                message=f"Updated {result.get('updated', 0)} embeddings"
            )
        except Exception as e:
            log.error(f"Failed to regenerate embeddings: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/memory/consolidate", response_model=ConsolidateResponse, tags=["Memory"])
    async def consolidate_memories(
        request: ConsolidateRequest,
        bot_id: str = Query(None, description="Bot ID"),
    ):
        """Find and merge redundant memories."""
        service = get_service()
        effective_bot_id = bot_id or service._default_bot
        
        try:
            client = service.get_memory_client(effective_bot_id)
            if not client:
                raise HTTPException(status_code=503, detail="Memory service unavailable")

            result = client.consolidate_memories(
                dry_run=request.dry_run,
                similarity_threshold=request.similarity_threshold,
            )
            
            return ConsolidateResponse(
                success=True,
                dry_run=bool(result.get("dry_run", request.dry_run)),
                clusters_found=int(result.get("clusters_found", 0)),
                clusters_merged=int(result.get("clusters_merged", 0)),
                memories_consolidated=int(result.get("memories_consolidated", 0)),
                new_memories_created=int(result.get("new_memories_created", 0)),
                errors=list(result.get("errors", [])),
                message=f"{'Would merge' if request.dry_run else 'Merged'} {int(result.get('clusters_merged', 0))} clusters",
            )
        except Exception as e:
            log.error(f"Failed to consolidate memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/llm/complete", response_model=RawCompletionResponse, tags=["LLM"])
    async def raw_completion(request: RawCompletionRequest):
        """Raw LLM completion using the currently loaded model.
        
        Use this for utility tasks like:
        - Memory consolidation (merging similar memories)
        - Summarization
        - Classification
        - Any task that needs LLM but not the full chat pipeline
        
        This endpoint uses the already-loaded model in the service.
        It will NOT load a new model - if no model is loaded, it returns 503.
        Only one LLM model can be loaded at a time (embedding model is separate).
        """
        import time
        service = get_service()
        
        # Check if we have any loaded client
        if not service._client_cache:
            raise HTTPException(
                status_code=503, 
                detail="No model loaded. Make a chat request first to load a model."
            )
        
        # Use the currently loaded model (there should only be one)
        loaded_models = list(service._client_cache.keys())
        model_alias = loaded_models[0]  # Use whatever is loaded
        
        # If caller specified a model, warn if it doesn't match
        if request.model and request.model != model_alias:
            log.debug(f"Requested model '{request.model}' but using loaded model '{model_alias}'")
        
        try:
            start = time.perf_counter()
            client = service._client_cache[model_alias]
            
            # Build simple messages
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})
            
            # Query the model directly
            response = client.query(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Estimate tokens
            tokens = len(response) // 4 if response else 0
            
            return RawCompletionResponse(
                content=response,
                model=model_alias,
                tokens=tokens,
                elapsed_ms=elapsed_ms,
            )
            
        except Exception as e:
            log.error(f"Raw completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

except ImportError:
    # FastAPI not installed - create stub
    app = None
    log.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _find_service_pid(port: int) -> int | None:
    """Find the PID of a process listening on the given port."""
    import subprocess
    try:
        # Use lsof to find the process listening on the port
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs (parent/child), get the first one
            pids = result.stdout.strip().split("\n")
            return int(pids[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return None


def _is_service_running(host: str, port: int) -> bool:
    """Check if the service is already running by attempting to connect."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            # Use 127.0.0.1 for 0.0.0.0 since we can't connect to 0.0.0.0
            check_host = "127.0.0.1" if host == "0.0.0.0" else host
            result = sock.connect_ex((check_host, port))
            return result == 0
    except (OSError, socket.error):
        return False


def _kill_service(port: int) -> bool:
    """Kill the service running on the given port. Returns True if successful."""
    import signal
    import os
    
    pid = _find_service_pid(port)
    if pid is None:
        return False
    
    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait briefly for the process to terminate
        import time
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)  # Check if process still exists
            except OSError:
                return True  # Process terminated
        
        # If still running, send SIGKILL
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.1)
        return True
    except OSError:
        return False


def main():
    """Entry point for the background service."""
    import argparse
    
    # Load config for defaults
    config = Config()
    
    parser = argparse.ArgumentParser(description="ask_llm background service")
    parser.add_argument("--host", default=config.SERVICE_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT, help="Port to listen on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional detail (payloads, timing)")
    parser.add_argument("--debug", action="store_true", help="Enable low-level DEBUG messages (unformatted)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--restart", action="store_true", help="Kill existing service and start a new one")
    parser.add_argument("--stop", action="store_true", help="Stop the running service and exit")
    args = parser.parse_args()
    
    # Setup logging with the new rich-formatted logger
    # Note: --verbose enables payload logging, --debug enables low-level DEBUG
    setup_service_logging(verbose=args.verbose, debug=args.debug)
    
    # Also update config.VERBOSE so BackgroundService can use it
    config.VERBOSE = args.verbose
    
    # Handle --stop: kill the service and exit
    if args.stop:
        if _is_service_running(args.host, args.port):
            print(f"Stopping service on port {args.port}...")
            if _kill_service(args.port):
                print("Service stopped.")
                return 0
            else:
                print("Failed to stop service.")
                return 1
        else:
            print(f"No service running on port {args.port}.")
            return 0
    
    # Check if service is already running
    if _is_service_running(args.host, args.port):
        if args.restart:
            print(f"Restarting service on port {args.port}...")
            if not _kill_service(args.port):
                print("Warning: Could not kill existing service, attempting to start anyway...")
            # Brief pause to ensure port is released
            import time
            time.sleep(0.5)
        else:
            print(f"Service is already running on port {args.port}.")
            print("Use --restart to restart the service, or --stop to stop it.")
            return 0
    
    if app is None:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        import uvicorn
        
        # Configure uvicorn log level
        # When using our rich logging, set uvicorn to warning to reduce noise
        uvicorn_log_level = "debug" if args.debug else "warning"
        
        uvicorn.run(
            "ask_llm.service.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=uvicorn_log_level,
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return 1


if __name__ == "__main__":
    main()

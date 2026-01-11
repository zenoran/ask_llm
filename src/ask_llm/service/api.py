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
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from queue import PriorityQueue
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from ..utils.config import Config
from .tasks import Task, TaskResult, TaskStatus, TaskType

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"


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
    bot_id: str = "nova"
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
        
        # Memory backend cache keyed by bot_id
        self._memory_backends: dict[str, Any] = {}
        
        # Model configuration
        self._available_models: list[str] = []
        self._default_model: str | None = None
        self._extraction_model: str | None = None
        self._load_available_models()
    
    def _load_available_models(self):
        """Load list of available models from config."""
        models = self.config.defined_models.get("models", {})
        self._available_models = list(models.keys())
        logger.info(f"Loaded {len(self._available_models)} models: {', '.join(self._available_models)}")
        
        # Set default model from config or use first available
        self._default_model = self.config.SERVICE_MODEL
        if not self._default_model and self._available_models:
            self._default_model = self._available_models[0]
        
        # Set extraction model from config, or prefer a local model for privacy
        # Memory extraction should use local models to avoid sending personal data externally
        self._extraction_model = self.config.EXTRACTION_MODEL
        if not self._extraction_model:
            # Try to find a local model (gguf, ollama, huggingface) for extraction
            for model_alias, model_def in models.items():
                model_type = model_def.get("type", "")
                if model_type in ("gguf", "ollama", "huggingface"):
                    self._extraction_model = model_alias
                    logger.info(f"Using local model '{model_alias}' for memory extraction (privacy)")
                    break
            # Fall back to default if no local model found
            if not self._extraction_model:
                self._extraction_model = self._default_model
                if self._extraction_model:
                    logger.warning(
                        f"No local model found for extraction, using '{self._extraction_model}'. "
                        "Set EXTRACTION_MODEL in config to use a specific local model."
                    )
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def get_memory_backend(self, bot_id: str):
        """Get or create memory backend for a bot."""
        if bot_id not in self._memory_backends:
            try:
                from ..memory.postgresql import PostgreSQLMemoryBackend
                self._memory_backends[bot_id] = PostgreSQLMemoryBackend(
                    config=self.config,
                    bot_id=bot_id,
                )
            except Exception as e:
                logger.warning(f"Memory backend unavailable for {bot_id}: {e}")
                self._memory_backends[bot_id] = None
        return self._memory_backends.get(bot_id)
    
    def _get_ask_llm(self, model_alias: str, bot_id: str, user_id: str, local_mode: bool = False):
        """Get or create an AskLLM instance with caching."""
        from ..core import AskLLM
        
        cache_key = (model_alias, bot_id, user_id)
        
        if cache_key not in self._ask_llm_cache:
            logger.info(f"Initializing AskLLM for model={model_alias}, bot={bot_id}, user={user_id}")
            try:
                # Create a copy of config for each AskLLM instance
                # This is necessary because AskLLM modifies config.SYSTEM_MESSAGE
                # based on the bot's system prompt
                instance_config = self.config.model_copy(deep=True)
                ask_llm = AskLLM(
                    resolved_model_alias=model_alias,
                    config=instance_config,
                    local_mode=local_mode,
                    bot_id=bot_id,
                    user_id=user_id,
                )
                self._ask_llm_cache[cache_key] = ask_llm
            except Exception as e:
                logger.exception(f"Failed to initialize AskLLM: {e}")
                raise
        
        return self._ask_llm_cache[cache_key]
    
    def get_client(self, model_alias: str):
        """Get LLM client for a given model (for extraction tasks).
        
        Uses 'spark' bot context since we only need the client,
        not the bot's personality or memory. Spark is the lightweight
        bot designed for no-memory operations.
        """
        ask_llm = self._get_ask_llm(model_alias, "spark", "system", local_mode=True)
        return ask_llm.client
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        """
        Handle an OpenAI-compatible chat completion request.
        
        This is the main entry point for the API. It:
        1. Uses cached AskLLM instances for efficiency
        2. Optionally augments messages with memory context
        3. Runs blocking LLM calls in a thread pool
        4. Optionally extracts memories from the response
        """
        from ..models.message import Message
        
        # Resolve model - use default if not specified or not found
        model_alias: str | None = request.model
        if not model_alias or model_alias not in self._available_models:
            if model_alias and model_alias in ("mira", "nova", "sage") and self._default_model:
                # Bot name passed as model - use default model
                model_alias = self._default_model
            elif self._default_model:
                if model_alias:
                    logger.warning(f"Model '{request.model}' not found, using default: {self._default_model}")
                model_alias = self._default_model
            else:
                raise ValueError(f"Model '{request.model}' not found. Available: {', '.join(self._available_models)}")
        
        bot_id = request.bot_id or "nova"
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
        
        # Run the blocking query in a thread pool
        loop = asyncio.get_event_loop()
        
        def _do_query():
            return ask_llm.query(
                prompt=user_prompt,
                plaintext_output=True,
                stream=False,
            )
        
        response_text = await loop.run_in_executor(None, _do_query)
        
        # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
        prompt_text = " ".join(m.content or "" for m in request.messages)
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(response_text) // 4
        
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
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        
        return response
    
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Handle a streaming chat completion request.
        
        Yields Server-Sent Events (SSE) formatted chunks.
        """
        import json
        from ..models.message import Message
        
        # Resolve model - use default if not specified or not found
        model_alias: str | None = request.model
        if not model_alias or model_alias not in self._available_models:
            if model_alias and model_alias in ("mira", "nova", "sage") and self._default_model:
                # Bot name passed as model - use default model
                model_alias = self._default_model
            elif self._default_model:
                if model_alias:
                    logger.warning(f"Model '{request.model}' not found, using default: {self._default_model}")
                model_alias = self._default_model
            else:
                raise ValueError(f"Model '{request.model}' not found.")
        
        bot_id = request.bot_id or "nova"
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
        
        # For streaming, we use ask_llm.prepare_messages_for_query to get properly
        # contextualized messages (with history, memory, system prompt)
        chunk_queue: asyncio.Queue = asyncio.Queue()
        full_response_holder = [""]  # Use list to allow mutation in nested function
        
        # Capture the event loop before entering the thread
        loop = asyncio.get_running_loop()
        
        def _stream_to_queue():
            """Run streaming in a thread and push chunks to the async queue."""
            try:
                # Use ask_llm.prepare_messages_for_query to get full context
                messages = ask_llm.prepare_messages_for_query(user_prompt)
                logger.debug(f"Streaming: Prepared {len(messages)} messages with memory/history context")
                
                # Use the client's stream_raw method for raw text chunks
                for chunk in ask_llm.client.stream_raw(messages):
                    full_response_holder[0] += chunk
                    # Put chunk in queue - use call_soon_threadsafe with captured loop
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                
                # Finalize: add response to history and trigger memory extraction
                if full_response_holder[0]:
                    ask_llm.finalize_response(user_prompt, full_response_holder[0])
                    
            except Exception as e:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # Sentinel
        
        # Start streaming in background thread
        loop.run_in_executor(None, _stream_to_queue)
        
        # Yield SSE chunks
        while True:
            chunk = await chunk_queue.get()
            
            if chunk is None:
                # Stream complete - send final chunk
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
            
            # Normal chunk
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_alias,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process a single background task."""
        start_time = time.time()
        
        try:
            if task.task_type == TaskType.MEMORY_EXTRACTION:
                result = await self._process_extraction(task)
            elif task.task_type == TaskType.CONTEXT_COMPACTION:
                result = await self._process_compaction(task)
            elif task.task_type == TaskType.EMBEDDING_GENERATION:
                result = await self._process_embeddings(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.exception(f"Task {task.task_id} failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def _process_extraction(self, task: Task) -> dict:
        """Process a memory extraction task."""
        from ..memory.extraction import MemoryExtractionService
        
        messages = task.payload.get("messages", [])
        bot_id = task.bot_id
        
        # Try to get an extraction client if we have a configured model
        extraction_client = None
        if self._extraction_model:
            try:
                extraction_client = self.get_client(self._extraction_model)
            except Exception as e:
                logger.warning(f"Could not load extraction client: {e}")
        
        extraction_service = MemoryExtractionService(llm_client=extraction_client)
        use_llm = extraction_client is not None
        
        loop = asyncio.get_event_loop()
        facts = await loop.run_in_executor(
            None,
            lambda: extraction_service.extract_from_conversation(messages, use_llm=use_llm)
        )
        
        if not facts:
            return {"facts_extracted": 0, "facts_stored": 0, "llm_used": use_llm}
        
        memory_backend = self.get_memory_backend(bot_id)
        stored_count = 0
        
        if memory_backend:
            min_importance = getattr(self.config, 'MEMORY_EXTRACTION_MIN_IMPORTANCE', 0.3)
            
            for fact in facts:
                if fact.importance >= min_importance:
                    try:
                        memory_backend.add_memory(
                            memory_id=str(uuid.uuid4()),
                            content=fact.content,
                            memory_type=fact.memory_type,
                            importance=fact.importance,
                            source_message_ids=fact.source_message_ids,
                        )
                        stored_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to store memory: {e}")
        
        return {"facts_extracted": len(facts), "facts_stored": stored_count, "llm_used": use_llm}
    
    async def _process_compaction(self, task: Task) -> dict:
        """Process a context compaction task."""
        # TODO: Implement with summarization model
        return {"compacted": False, "reason": "Not yet implemented"}
    
    async def _process_embeddings(self, task: Task) -> dict:
        """Process an embedding generation task."""
        # TODO: Implement with embedding model
        return {"embeddings_generated": 0, "reason": "Not yet implemented"}
    
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
        logger.info("Background worker started")
        
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
                logger.debug(f"Processing task {task.task_id}")
                
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
                logger.exception(f"Worker loop error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Background worker stopped")
    
    def start_worker(self):
        """Start the background worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker_loop())
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        logger.info("Shutting down background service...")
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
    _service = BackgroundService(config)
    _service.start_worker()
    logger.info(f"ask_llm service v{SERVICE_VERSION} started")
    
    yield
    
    # Shutdown
    await _service.shutdown()
    logger.info("ask_llm service stopped")


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
                logger.exception("Streaming chat completion failed")
                raise HTTPException(status_code=500, detail=str(e))
        
        try:
            response = await service.chat_completion(request)
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Chat completion failed")
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
            bot_id=request.bot_id,
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

except ImportError:
    # FastAPI not installed - create stub
    app = None
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Entry point for the background service."""
    import argparse
    
    # Load config for defaults
    config = Config()
    
    parser = argparse.ArgumentParser(description="ask_llm background service")
    parser.add_argument("--host", default=config.SERVICE_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT, help="Port to listen on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if app is None:
        print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return 1
    
    try:
        import uvicorn
        uvicorn.run(
            "ask_llm.service.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="debug" if args.verbose else "info",
        )
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return 1


if __name__ == "__main__":
    main()

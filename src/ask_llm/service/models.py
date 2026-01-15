"""
Pydantic models for OpenAI-compatible API.

This module defines the request/response models for the background service API,
following the OpenAI API specification for compatibility.
"""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# Configuration constants
DEFAULT_HTTP_PORT = 8642
SERVICE_VERSION = "0.1.0"


# =============================================================================
# OpenAI-compatible Chat Models
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


# =============================================================================
# Model Information Models
# =============================================================================

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


# =============================================================================
# Task Management Models
# =============================================================================

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


# =============================================================================
# Service Status Models
# =============================================================================

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

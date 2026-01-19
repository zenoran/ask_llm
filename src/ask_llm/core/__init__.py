"""Core LLM processing components.

This package provides the main orchestration for ask_llm:
- AskLLM: Main client class for CLI usage
- PromptBuilder: Template-based system prompt assembly
- RequestPipeline: Modular request processing with hooks
- BaseAskLLM: Shared logic for CLI and service modes
- ModelLifecycleManager: Singleton for model loading/unloading
"""

from .prompt_builder import PromptBuilder, PromptSection, SectionPosition
from .pipeline import RequestPipeline, PipelineStage, PipelineContext
from .base import BaseAskLLM
from .client import AskLLM
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

__all__ = [
    "AskLLM",
    "PromptBuilder",
    "PromptSection",
    "SectionPosition",
    "RequestPipeline",
    "PipelineStage",
    "PipelineContext",
    "BaseAskLLM",
    "ModelLifecycleManager",
    "get_model_lifecycle",
]

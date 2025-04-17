from .base import LLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient

# Conditionally import HuggingFaceClient only if dependencies are available
try:
    import torch
    import transformers
    import bitsandbytes
    import peft
    import accelerate
    import xformers
    from .huggingface_client import HuggingFaceClient
    __all__ = ['LLMClient', 'OpenAIClient', 'OllamaClient', 'HuggingFaceClient']
except ImportError:
    __all__ = ['LLMClient', 'OpenAIClient', 'OllamaClient']
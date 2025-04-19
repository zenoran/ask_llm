from .base import LLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient

# Import HuggingFaceClient conditionally
try:
    # Basic check for HF dependencies
    import torch
    import transformers
    from .huggingface_client import HuggingFaceClient
    _hf_available = True
except ImportError:
    HuggingFaceClient = None # type: ignore # Set to None if unavailable
    _hf_available = False

# Import LlamaCppClient conditionally
try:
    from llama_cpp import Llama # Check if base library is installed
    from .llama_cpp_client import LlamaCppClient
    _llama_cpp_available = True
except ImportError:
    LlamaCppClient = None # type: ignore # Set to None if unavailable
    _llama_cpp_available = False

# Build __all__ based on availability
__all__ = ['LLMClient', 'OpenAIClient', 'OllamaClient']
if _hf_available and HuggingFaceClient:
    __all__.append('HuggingFaceClient')
if _llama_cpp_available and LlamaCppClient:
    __all__.append('LlamaCppClient')

# Clean up temporary variables
del _hf_available
del _llama_cpp_available


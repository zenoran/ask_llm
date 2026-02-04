## Plan: Model Adapter Pattern for ask_llm

### TL;DR
Add a `ModelAdapter` layer that sits between the core orchestration and LLM clients, handling model-specific input formatting, stop sequences, and output cleaning. This consolidates scattered model quirks into configurable, per-model adapters while preserving all existing functionality.

---

### Current Architecture (Simplified)

```
BaseAskLLM._initialize_client()
       ↓
   LLMClient (OpenAI/LlamaCpp)
       ↓
stream_with_tools() ←── ToolFormatHandler (ReAct/XML/Native)
       ↓                    ├── get_stop_sequences()
   Response                 ├── parse_response()
       ↓                    └── sanitize_response()
   Return to user
```

**Problems identified:**
1. Model-specific stop sequences (`[HUMAN]`, `[INST]`) are hardcoded in react.py
2. Output cleaning for model quirks (`[HUMAN]`, BBCode) is in react.py - only applies to ReAct format
3. `chat_format` is passed to LlamaCppClient but not all models work correctly
4. No centralized place for per-model quirks (system role support, special tokens, etc.)

---

### Proposed Architecture

```
BaseAskLLM._initialize_client()
       ↓
   ModelAdapter ←────────── get_adapter(model_alias)
       │                         ↓
       │                    models.yaml: adapter: "mythomax"
       ↓
   LLMClient (unchanged interface)
       ↓
stream_with_tools()
       │
       ├── stop_sequences = handler.get_stop_sequences() + adapter.get_stop_sequences()
       │
       ↓
   Raw Response
       ↓
   adapter.clean_output(response)  ←── NEW: per-model cleaning
       ↓
   handler.sanitize_response()     ←── Existing: format-specific cleaning
       ↓
   Return to user
```

---

### Step 1: Create Adapter Base & Registry

**NEW FILE: src/ask_llm/adapters/base.py**

```python
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.message import Message


class ModelAdapter(ABC):
    """Handles model-specific formatting and output quirks.
    
    Adapters are responsible for:
    - Additional stop sequences (beyond tool format handlers)
    - Output cleaning (removing model-specific artifacts)
    - Message formatting quirks (system role support, etc.)
    """
    
    name: str = "default"
    
    def get_stop_sequences(self) -> list[str]:
        """Model-specific stop sequences (combined with tool handler's)."""
        return []
    
    def clean_output(self, response: str) -> str:
        """Remove model-specific artifacts from output.
        
        Called AFTER streaming completes, BEFORE format handler sanitization.
        Default: passthrough (no cleaning).
        """
        return response
    
    def supports_system_role(self) -> bool:
        """Whether model natively supports system role messages."""
        return True
    
    def transform_messages(self, messages: list["Message"]) -> list["Message"]:
        """Apply model-specific message transformations.
        
        For models that don't support system role, this can merge
        system message into first user message.
        Default: passthrough.
        """
        return messages
```

**NEW FILE: src/ask_llm/adapters/__init__.py**

```python
from .base import ModelAdapter
from .default import DefaultAdapter
from .pygmalion import PygmalionAdapter
from .registry import get_adapter, register_adapter

__all__ = ["ModelAdapter", "DefaultAdapter", "PygmalionAdapter", "get_adapter", "register_adapter"]
```

**NEW FILE: src/ask_llm/adapters/registry.py**

```python
import logging
from typing import Type
from .base import ModelAdapter
from .default import DefaultAdapter

logger = logging.getLogger(__name__)

_ADAPTERS: dict[str, Type[ModelAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[ModelAdapter]):
    """Register an adapter class."""
    _ADAPTERS[name] = adapter_cls


def get_adapter(model_alias: str, model_def: dict | None = None) -> ModelAdapter:
    """Get adapter instance for a model.
    
    Checks model definition for explicit 'adapter' field, falls back to default.
    """
    adapter_name = None
    if model_def:
        adapter_name = model_def.get("adapter")
    
    if adapter_name and adapter_name in _ADAPTERS:
        logger.debug(f"Using adapter '{adapter_name}' for model '{model_alias}'")
        return _ADAPTERS[adapter_name]()
    
    return DefaultAdapter()


# Auto-register adapters when module loads
def _register_builtins():
    from .default import DefaultAdapter
    from .pygmalion import PygmalionAdapter
    
    register_adapter("default", DefaultAdapter)
    register_adapter("pygmalion", PygmalionAdapter)
    register_adapter("mythomax", PygmalionAdapter)  # Alias


_register_builtins()
```

---

### Step 2: Create Specific Adapters

**NEW FILE: src/ask_llm/adapters/default.py**

```python
from .base import ModelAdapter


class DefaultAdapter(ModelAdapter):
    """Default no-op adapter for well-behaved models."""
    name = "default"
```

**NEW FILE: src/ask_llm/adapters/pygmalion.py**

```python
import re
from .base import ModelAdapter


class PygmalionAdapter(ModelAdapter):
    """Adapter for Pygmalion/character models (MythoMax, etc).
    
    These models output:
    - Role markers: [HUMAN], [/HUMAN], [INST], [/INST]
    - BBCode: [FONT=Arial], [/FONT], etc.
    - Sometimes try to continue conversation with fake turns
    """
    
    name = "pygmalion"
    
    def get_stop_sequences(self) -> list[str]:
        return [
            "[HUMAN]",
            "[/HUMAN]", 
            "[INST]",
            "[/INST]",
            "### Instruction:",
            "### Human:",
            "<|im_start|>user",
        ]
    
    def clean_output(self, response: str) -> str:
        # Remove BBCode tags: [FONT=Arial], [/FONT], [B], [/B], etc.
        response = re.sub(r"\[\w+(?:=[^\]]+)?\]", "", response)
        response = re.sub(r"\[/\w+\]", "", response)
        
        # Remove role markers (in case they slipped through stop sequences)
        response = re.sub(r"\[/?HUMAN\]", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\[/?INST\]", "", response, flags=re.IGNORECASE)
        
        # Remove content inside role blocks (hallucinated turns)
        response = re.sub(r"\[HUMAN\].*?\[/HUMAN\]", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"\[INST\].*?\[/INST\]", "", response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean excessive whitespace
        response = re.sub(r"\n{3,}", "\n\n", response)
        
        return response.strip()
```

---

### Step 3: Integrate Into Core

**MODIFY: src/ask_llm/core/base.py**

```python
# Add import at top
from ..adapters import get_adapter, ModelAdapter

class BaseAskLLM(ABC):
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize adapter based on model definition
        self.adapter: ModelAdapter = get_adapter(
            self.resolved_model_alias, 
            self.model_definition
        )
        
        # Initialize LLM client
        if existing_client is not None:
            self.client = existing_client
        else:
            self.client = self._initialize_client()
```

---

### Step 4: Integrate Into Streaming

**MODIFY: src/ask_llm/tools/streaming.py**

```python
def stream_with_tools(
    messages: list["Message"],
    stream_fn: Callable[[list["Message"], list[str] | None], Iterator[str]],
    # ... existing params ...
    adapter: "ModelAdapter | None" = None,  # NEW
) -> Iterator[str]:
    """Stream LLM response with tool calling support."""
    
    from ..adapters import DefaultAdapter
    if adapter is None:
        adapter = DefaultAdapter()
    
    handler = get_format_handler(tool_format)
    
    # Combine stop sequences from both handler and adapter
    handler_stops = handler.get_stop_sequences()
    adapter_stops = adapter.get_stop_sequences()
    stop_sequences = list(set(handler_stops + adapter_stops))
    
    # ... existing streaming loop ...
    
    # After stream ends, before returning non-tool response:
    if not tool_calls:
        if not streaming_mode:
            # Apply adapter cleaning FIRST, then format sanitization
            cleaned = adapter.clean_output(full_response)
            sanitized = handler.sanitize_response(cleaned)
            yield sanitized
        return
```

---

### Step 5: Remove Hardcoded Model Quirks from ReAct Handler

**MODIFY: src/ask_llm/tools/formats/react.py**

Remove from `get_stop_sequences()`:
```python
def get_stop_sequences(self) -> list[str]:
    # Only ReAct-format stop sequences, NOT model-specific
    return [
        "\nObservation:",
        "\nObservation",
        "}\nObservation",
        # REMOVED: [HUMAN], [/HUMAN], [INST], [/INST] - now in adapters
    ]
```

Remove from `sanitize_response()`:
```python
def sanitize_response(self, response: str) -> str:
    # ... keep ReAct-specific cleaning ...
    
    # REMOVED: [HUMAN]/[INST] cleaning - now in adapters
    # REMOVED: The re.sub lines for these patterns
```

---

### Step 6: Update models.yaml Format

```yaml
models:
  mythomax:
    type: gguf
    repo_id: Gryphe/MythoMax-L2-13b
    filename: MythoMax-L2-13b-Q4_K_M.gguf
    adapter: pygmalion       # NEW: specifies which adapter
    chat_format: alpaca      # Still needed for llama.cpp
    tool_format: react
    
  dolphin-qwen:
    type: gguf
    repo_id: bartowski/Dolphin3.0-Qwen2.5-3b-GGUF
    filename: Dolphin3.0-Qwen2.5-3b-Q6_K.gguf
    # No adapter field = uses default
    tool_format: react
    
  gpt4:
    type: openai
    model_id: gpt-4-turbo
    # No adapter field = uses default
    tool_format: native
```

---

### Files Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/ask_llm/adapters/__init__.py` | NEW | ~10 |
| `src/ask_llm/adapters/base.py` | NEW | ~45 |
| `src/ask_llm/adapters/registry.py` | NEW | ~40 |
| `src/ask_llm/adapters/default.py` | NEW | ~8 |
| `src/ask_llm/adapters/pygmalion.py` | NEW | ~45 |
| `src/ask_llm/core/base.py` | MODIFY | ~8 (add adapter init) |
| `src/ask_llm/tools/streaming.py` | MODIFY | ~15 (add adapter param, combine stops) |
| `src/ask_llm/tools/formats/react.py` | MODIFY | ~-20 (remove hardcoded model quirks) |

**Total: ~150 new, ~23 modified, ~20 removed**

---

### Further Considerations

1. **Streaming chunk cleaning** - Current plan only cleans final output. If you need real-time cleaning during streaming, you'd need a stateful buffer to handle patterns split across chunks. Recommend starting with final-only cleaning (simpler, works for tool calling).

2. **Should adapter own `chat_format`?** - Currently `chat_format` is in model definition and passed to LlamaCppClient. Could add `adapter.chat_format` property if you want adapters to also control this. Recommend keeping separate for now.

3. **Testing** - Add test in `tests/test_adapters.py` that verifies:
   - `PygmalionAdapter.clean_output()` removes BBCode and role markers
   - Registry returns correct adapter for model alias
   - Default adapter is passthrough

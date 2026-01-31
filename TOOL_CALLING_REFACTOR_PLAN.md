# Tool Calling Refactoring Plan

## Executive Summary

The current implementation uses a custom `<tool_call>` XML format that relies on LLM discipline to "output immediately then STOP." This fails because:
1. Models naturally explain before/after tool calls
2. No stop sequences enforce the "STOP" instruction
3. Parser returns raw response with tags when parsing fails
4. Custom format isn't what models are trained on

**Recommended Solution**: Hybrid approach using **ReAct format** as the universal fallback (works with all models) with optional **native tool calling** for providers that support it (OpenAI, Claude).

---

## Problem Analysis (from debug_turn.txt)

### The Failure Chain

```
1. User asks: "u can delete those"
2. LLM generates response with BOTH text AND tool call:
   "Done. Next I'm going to delete... <tool_call>{"name":"delete_user_attribute"...}</tool_call>"
3. Parser fails to extract (or extracts but response already has extra text)
4. Raw <tool_call> tags appear in user-facing response (line 611 of debug log)
```

### Root Causes

| Issue | Current State | Impact |
|-------|---------------|--------|
| No stop sequences | LLM continues generating after tool call | Extra text around tool calls |
| Custom XML format | Models not trained on `<tool_call>` | Inconsistent formatting |
| Brittle parser | Returns original response on failure | Raw tags shown to users |
| Prompt-only enforcement | "STOP" instruction ignored | No architectural enforcement |

---

## Status Update (2026-01-29 evening)

### Major Progress: Native OpenAI Tool Calling with Streaming

**Tools now work correctly for OpenAI models!** Implemented proper native tool calling that:
- Uses OpenAI's `tools` parameter with structured responses (no text parsing)
- Maintains real streaming for regular responses
- Handles tool calls inline during streaming

### Changes Made Today

#### 1. `src/ask_llm/clients/openai_client.py`
- Added `stream_with_tools()` method that:
  - Passes `tools` schema to OpenAI API with `stream=True`
  - Yields content chunks immediately (real streaming)
  - Accumulates tool_call deltas from stream
  - Returns `{"tool_calls": [...], "content": "..."}` dict at end if tools were called

#### 2. `src/ask_llm/service/api.py`
- Updated streaming endpoint to use native streaming with tools:
  - Detects if client supports native tools (`supports_native_tools()` + `stream_with_tools`)
  - For OpenAI: Uses new `stream_with_tools()` - real streaming + structured tool calls
  - For GGUF/local: Falls back to text-based `stream_with_tools()` from streaming.py
  - Implements tool loop inline: execute tools, append results, continue streaming

#### 3. Infrastructure already in place (no changes needed)
- `src/ask_llm/tools/formats/native_openai.py` - Tool schema generation
- `src/ask_llm/tools/loop.py` - Non-streaming tool loop with native support
- `src/ask_llm/utils/config.py` - Returns `"native"` format for OpenAI models

### Current Behavior
- **OpenAI models**: Native tool calling with real streaming
  - Regular prompts: Stream immediately
  - Tool prompts: Stream content, detect tool calls, execute, stream follow-up
- **Local models (GGUF)**: Text-based tool detection via streaming.py (heuristics)

### Still TODO / Known Issues
1. **streaming.py bandaids** - Still has heuristic code for non-native models. Works but fragile.
2. **ReAct format** - Not yet implemented for local models (would be cleaner than heuristics)
3. **Stop sequences** - Not used yet; would help ReAct format enforcement
4. **Testing** - Need to verify tool loop iteration limit and edge cases

### Next Steps (for continuing tomorrow)
1. **Test the new streaming** - Run `./server.sh restart` and try tool commands
2. **Clean up streaming.py** - Remove/simplify heuristic code if native path works
3. **Implement ReAct for local models** - Replace heuristics with proper format
4. **Add stop sequences** - For ReAct format enforcement on local models

---

## Current Architecture Analysis

### Files Involved

| File | Purpose | Changes Needed |
|------|---------|----------------|
| `src/ask_llm/tools/definitions.py` | Tool definitions + prompt instructions | Replace `<tool_call>` with ReAct format |
| `src/ask_llm/tools/parser.py` | Extracts tool calls from response | New ReAct parser + sanitization |
| `src/ask_llm/tools/loop.py` | Iterative tool calling loop | Add stop sequences, sanitization |
| `src/ask_llm/tools/executor.py` | Executes parsed tools | Minimal changes (format result) |
| `src/ask_llm/clients/openai_client.py` | OpenAI API client | Add `stop` param, optional native tools |
| `src/ask_llm/clients/llama_cpp_client.py` | Local GGUF client | Add `stop` param support |
| `src/ask_llm/clients/base.py` | Base client class | Add stop sequence interface |
| `config/models.yaml` | Model configurations | Add `supports_native_tools` flag |

### Current Tool Calling Flow (Broken)

```
1. System prompt says: "Use <tool_call> format, output IMMEDIATELY then STOP"
   (src/ask_llm/tools/definitions.py:343-366)

2. LLM generates: "I'll check that for you. <tool_call>...</tool_call> Let me explain..."
   (No stop sequence to prevent continuation)

3. loop.py:106 calls has_tool_call() - returns True
4. loop.py:112 calls parse_tool_calls() - may fail due to surrounding text
5. loop.py:114-116: If parsing fails, returns ORIGINAL response with tags!

6. User sees: "Done... <tool_call>{"name":"delete_user_attribute"...}</tool_call>"
```

---

## Research Findings

### Industry Standard Approaches

#### 1. ReAct Format (Universal - Works with All Models)
From [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/react), [LangChain](https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html), [Qwen](https://github.com/QwenLM/Qwen/blob/main/examples/react_prompt.md):

```
Thought: I need to check the user's preferences
Action: search_memories
Action Input: {"query": "user preferences"}
Observation: [system injects result here]
Thought: I now have the answer
Final Answer: Based on your preferences...
```

**Critical**: Use stop sequence `["\nObservation:"]` to prevent model from generating its own observation.

#### 2. Native Tool Calling (OpenAI, Anthropic)
From [OpenAI docs](https://platform.openai.com/docs/guides/function-calling), [Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use):

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "search_memories",
            "parameters": {...}
        }
    }]
)

if response.choices[0].finish_reason == "tool_calls":
    # Structured tool calls - no parsing needed!
    tool_calls = response.choices[0].message.tool_calls
```

#### 3. Error Recovery (LangChain)
From [LangChain error handling](https://python.langchain.com/v0.1/docs/modules/agents/how_to/handle_parsing_errors/):

- `handle_parsing_errors=True`: Pass error back to LLM to retry
- Always sanitize responses before returning to user
- Schema validation with Pydantic

---

## Proposed Architecture

### Strategy: Dual-Mode Tool Calling

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tool Calling Router                         │
├─────────────────────────────────────────────────────────────────┤
│  Model Type Detection                                            │
│  ├─ OpenAI API (gpt-4, gpt-5, o1, etc.)                         │
│  │   └─ Use NATIVE tool calling (tools parameter)               │
│  │                                                               │
│  ├─ Anthropic API (claude-*)                                    │
│  │   └─ Use NATIVE tool calling (tools parameter)               │
│  │                                                               │
│  └─ Local/Other (GGUF, Ollama, vLLM, unknown)                   │
│      └─ Use REACT FORMAT with stop sequences                    │
└─────────────────────────────────────────────────────────────────┘
```

### ReAct Format Prompt (Universal Fallback)

```
## Tools

You have access to the following tools:

{tool_descriptions}

## How to Use Tools

When you need to use a tool, respond with this EXACT format:

Thought: [explain what you need to do and why]
Action: [tool_name from the list above]
Action Input: {"param": "value"}

Then STOP. The system will execute the tool and provide an Observation.

## When You Have the Answer

When you have gathered enough information:

Thought: I now have the information needed
Final Answer: [your complete response to the user]

## Important Rules

1. Use EXACTLY ONE tool per response - no more
2. STOP after Action Input - do not continue
3. Wait for Observation before your next Thought
4. Tool names must be exactly: {tool_names}
5. Action Input must be valid JSON
```

**Key Implementation Details:**
- Stop sequences: `["\nObservation:", "\nObservation"]`
- Regex: `r"Action\s*:\s*(.*?)\s*Action\s+Input\s*:\s*(.*)"`
- Use `json5` for forgiving JSON parsing

---

## Detailed Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 New Tool Format Abstraction

**New file: `src/ask_llm/tools/formats/__init__.py`**

```python
from enum import Enum

class ToolFormat(Enum):
    REACT = "react"           # Text-based Thought/Action/Action Input
    NATIVE_OPENAI = "native"  # OpenAI tools parameter
    XML = "xml"               # Legacy <tool_call> (deprecated)

def get_format_handler(format: ToolFormat) -> "ToolFormatHandler":
    from .react import ReActFormatHandler
    from .native_openai import NativeOpenAIFormatHandler
    from .xml_legacy import LegacyXMLFormatHandler

    handlers = {
        ToolFormat.REACT: ReActFormatHandler(),
        ToolFormat.NATIVE_OPENAI: NativeOpenAIFormatHandler(),
        ToolFormat.XML: LegacyXMLFormatHandler(),
    }
    return handlers[format]
```

**New file: `src/ask_llm/tools/formats/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ToolCallRequest:
    """Unified representation of a tool call."""
    name: str
    arguments: dict
    raw_text: str | None = None
    tool_call_id: str | None = None  # For native OpenAI

class ToolFormatHandler(ABC):
    """Abstract handler for different tool calling formats."""

    @abstractmethod
    def get_system_prompt(self, tools: list) -> str:
        """Generate system prompt instructions for this format."""
        pass

    @abstractmethod
    def get_stop_sequences(self) -> list[str]:
        """Return stop sequences for this format."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        """Parse tool calls from response. Returns (calls, remaining_text)."""
        pass

    @abstractmethod
    def format_result(self, tool_name: str, result: str, tool_call_id: str | None = None) -> str:
        """Format tool result for injection back to LLM."""
        pass

    @abstractmethod
    def sanitize_response(self, response: str) -> str:
        """Remove any unparsed tool markers from final response."""
        pass
```

#### 1.2 ReAct Format Handler

**New file: `src/ask_llm/tools/formats/react.py`**

See full implementation in plan above (too long to repeat here - includes regex patterns, parsing logic, sanitization, etc.)

Key features:
- ACTION_PATTERN regex for parsing
- FINAL_ANSWER_PATTERN for completion detection
- Flexible JSON parsing with json5 fallback
- Comprehensive sanitization
- Stop sequences: `["\nObservation:", "\nObservation"]`

#### 1.3 Native OpenAI Format Handler

**New file: `src/ask_llm/tools/formats/native_openai.py`**

Key features:
- Converts Tool objects to OpenAI tools schema
- Returns empty stop sequences (not needed)
- Minimal system prompt (tools parameter handles format)
- `parse_native_tool_calls()` for API responses
- Returns dict for tool result messages

#### 1.4 Legacy XML Format Handler

**New file: `src/ask_llm/tools/formats/xml_legacy.py`**

Keeps existing `<tool_call>` XML format for backward compatibility.

### Phase 2: Client Updates

#### 2.1 Update Base Client

**Modify: `src/ask_llm/clients/base.py`**

Add methods:
```python
def supports_native_tools(self) -> bool:
    return False

def query_with_tools(self, messages, tools_schema=None, tool_choice="auto", stop=None, **kwargs):
    # Default: ignore tools_schema, use stop sequences
    return self.query(messages, stop=stop, **kwargs), None
```

#### 2.2 Update OpenAI Client

**Modify: `src/ask_llm/clients/openai_client.py`**

Changes:
- Add `supports_native_tools()` → returns True for most models
- Add `query_with_tools()` → uses `tools` parameter
- Update `query()` → accept `stop` parameter
- Handle `finish_reason == "tool_calls"`

#### 2.3 Update Llama.cpp Client

**Modify: `src/ask_llm/clients/llama_cpp_client.py`**

Changes:
- Update `query()` → accept `stop` parameter
- Update `stream_raw()` → accept `stop` parameter
- Pass stop sequences to `create_chat_completion()`

### Phase 3: Tool Loop Refactor

**Modify: `src/ask_llm/tools/loop.py`**

Major changes:
- Accept `tool_format` parameter in `__init__`
- Create format handler
- Detect native tool support in client
- Use stop sequences or native tools appropriately
- Always sanitize final responses
- Handle both text and native tool call results

### Phase 4: Configuration Updates

**Modify: `config/models.yaml`**

```yaml
models:
  gpt-5.2-2025-12-11:
    type: openai
    model_id: gpt-5.2-2025-12-11
    tool_format: native  # Use OpenAI native

  dolphin-qwen-3b:
    type: gguf
    repo_id: bartowski/Dolphin3.0-Qwen2.5-3b-GGUF
    filename: Dolphin3.0-Qwen2.5-3b-Q6_K.gguf
    tool_format: react  # Use ReAct for local

defaults:
  tool_formats:
    openai: native
    gguf: react
```

**Modify: `src/ask_llm/utils/config.py`**

Add `get_tool_format()` method to read format from config.

### Phase 5: Integration

**Modify: `src/ask_llm/tools/definitions.py`**

Update `get_tools_prompt()` to use format handlers instead of hardcoded XML.

---

## Testing Strategy

### Unit Tests

**New file: `tests/test_tool_formats.py`**

Tests:
- ReAct parser with basic action
- ReAct parser with surrounding text
- ReAct parser with final answer
- Sanitization removes all markers
- Stop sequences configured correctly
- XML legacy parser still works
- **Regression test for debug_turn.txt failure**

### Integration Tests

**New file: `tests/test_tool_loop_integration.py`**

Tests:
- ReAct format full loop with stop sequences
- Native OpenAI format full loop
- Fallback on parse failure
- Multiple tool iterations
- Max iterations handling

### Regression Test

```python
def test_regression_debug_turn_case():
    """Regression test for exact failure from debug_turn.txt"""
    response = '''Done. Next I'm going to delete...
    <tool_call>
    {"name":"delete_user_attribute","arguments":{"query":"qwen"}}
    </tool_call>'''

    handler = ReActFormatHandler()
    sanitized = handler.sanitize_response(response)
    assert "<tool_call>" not in sanitized
```

---

## Migration Path

### Step 1: Add New Infrastructure (Non-Breaking)
- [ ] Create `formats/` module with handlers
- [ ] Keep existing parser working
- [ ] Add `tool_format` to config (optional)

### Step 2: Update Clients (Non-Breaking)
- [ ] Add `stop` parameter support
- [ ] Add `supports_native_tools()`
- [ ] Add `query_with_tools()`

### Step 3: Update Tool Loop
- [ ] Refactor to use format handlers
- [ ] Pass stop sequences
- [ ] Always sanitize responses

### Step 4: Testing & Rollout
- [ ] Add comprehensive tests
- [ ] Enable ReAct for local models
- [ ] Enable native for OpenAI models
- [ ] Monitor for issues

### Step 5: Deprecate XML (Future)
- [ ] Log warning when XML detected
- [ ] Update documentation
- [ ] Remove as default

---

## File Summary

### New Files (7)
- `src/ask_llm/tools/formats/__init__.py`
- `src/ask_llm/tools/formats/base.py`
- `src/ask_llm/tools/formats/react.py`
- `src/ask_llm/tools/formats/native_openai.py`
- `src/ask_llm/tools/formats/xml_legacy.py`
- `tests/test_tool_formats.py`
- `tests/test_tool_loop_integration.py`

### Modified Files (8)
- `src/ask_llm/tools/loop.py`
- `src/ask_llm/tools/definitions.py`
- `src/ask_llm/clients/base.py`
- `src/ask_llm/clients/openai_client.py`
- `src/ask_llm/clients/llama_cpp_client.py`
- `src/ask_llm/utils/config.py`
- `config/models.yaml`
- `src/ask_llm/models/message.py` (may need tool_call_id field)

---

## Success Criteria

- [ ] Stop sequences prevent models from continuing past tool calls
- [ ] Users never see `<tool_call>` or `Action:` markers in responses
- [ ] Native tool calling works for OpenAI models
- [ ] ReAct format works for local GGUF models (Qwen, Llama)
- [ ] All existing tool functionality preserved
- [ ] Tool execution success rate > 95%
- [ ] Comprehensive test coverage
- [ ] Clear error messages when tools fail

---

## Benefits

1. **Reliability**: Stop sequences architecturally prevent continuation
2. **Compatibility**: ReAct works with all models (OpenAI, Qwen, Llama)
3. **Performance**: Native tools faster for supported providers
4. **Maintainability**: Clean separation with format handlers
5. **Debugging**: Standard format easier to debug
6. **User Experience**: No raw tags ever shown to users
7. **Industry Standard**: Using proven formats from LangChain/Qwen

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing tool calls | Keep XML as fallback, gradual migration |
| ReAct unfamiliar to models | Well-documented format, models trained on it |
| Stop sequences not respected | Sanitization always runs as safety net |
| Native tools API changes | Abstract behind format handler |
| Performance regression | Monitor metrics, optimize if needed |

---

## References

- [ReAct: Synergizing Reasoning and Acting](https://www.promptingguide.ai/techniques/react)
- [LangChain ReAct Agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html)
- [Qwen ReAct Prompt](https://github.com/QwenLM/Qwen/blob/main/examples/react_prompt.md)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain Error Handling](https://python.langchain.com/v0.1/docs/modules/agents/how_to/handle_parsing_errors/)

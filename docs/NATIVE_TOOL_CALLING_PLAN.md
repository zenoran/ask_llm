# Native Tool Calling for GGUF Models

## Problem
The current ReAct-based tool calling for GGUF models is fragile:
- Models don't respect stop sequences consistently
- Streaming detection heuristics fail on partial matches
- Models hallucinate tool results instead of waiting for execution

## Research: llama-cpp-python Native Function Calling

llama-cpp-python has built-in support for OpenAI-compatible tool calling via the `chat_format` parameter:
- `chat_format="chatml-function-calling"` - Works with any chatml-compatible model
- Uses **grammar constraints** to force valid JSON output

### Key Findings from Testing

**Native tool calling DOES work for the first iteration:**
- Model correctly detects need for tool
- Returns properly formatted tool_calls
- Grammar constraint ensures valid JSON arguments

**BUT it fails for multi-turn conversations:**
- After executing a tool and adding the result to messages
- The second LLM call returns **empty content**
- This makes it unsuitable for our use case

### Root Cause
llama-cpp-python's `chatml-function-calling` format appears to be designed for single-shot function extraction, not multi-turn agentic workflows. When the model receives a tool result and should respond with natural language, it returns empty.

## Current Solution: ReAct Format with Stop Sequences

For GGUF/local models, we use ReAct-style text parsing:
- Format: `Thought: ... Action: ... Action Input: {...}`
- Stop sequences: `\nObservation:` to halt before hallucinating results

### Key Fix: Skip stop sequences after tool execution

The critical issue was that stop sequences were being used on ALL iterations, which prevented the model from generating a complete `Final Answer:` response after receiving tool results.

**Solution (implemented in both `loop.py` and `streaming.py`):**

```python
has_executed_tools = False

for iteration in range(1, max_iterations + 1):
    # After executing tools, don't use stop sequences
    current_stop_sequences = None if has_executed_tools else stop_sequences
    
    response = query_llm(messages, stop=current_stop_sequences)
    
    if tool_calls_found:
        execute_tools()
        has_executed_tools = True  # Next iteration won't use stop sequences
```

This allows the model to:
1. First iteration: Stop at `\nObservation:` to detect tool calls
2. Follow-up iterations: Generate complete response with `Thought: I now have... Final Answer: ...`

## Future Improvements

1. **Better stop sequence handling**: Improve streaming detection to stop immediately when partial match found
2. **Structured output with grammar**: Use llama-cpp-python's JSON schema grammar for just the tool call detection, then fall back to regular completion for the follow-up
3. **Custom chat handler**: Create a custom `LlamaChatCompletionHandler` that properly handles multi-turn tool conversations

## Code Changes Made

### Added Native Tool Support (for OpenAI)
- `LlamaCppClient.supports_native_tools()` - Returns True
- `LlamaCppClient.query_with_tools()` - Wrapper that uses native calling when available
- Updated `ToolLoop._should_use_native_tools()` - Only returns True for `native` format (OpenAI API)

### NOT Using Native for GGUF (ReAct format)
Due to the multi-turn issue, GGUF models with `react` format continue to use text-based ReAct parsing.

## Testing Notes

When testing tool calling:
```bash
llm -b proto "use get_recent_history to show last 3 messages"
```

Expected behavior:
1. Model outputs ReAct format: `Thought: ... Action: ... Action Input: ...`
2. Stops at `\nObservation:` 
3. Tool executes
4. Result added to messages
5. Model continues with `Thought: I now have... Final Answer: ...`

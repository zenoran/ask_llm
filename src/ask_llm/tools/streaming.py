"""Streaming tool loop for LLM conversations.

Handles tool calling while maintaining streaming output to the client.
"""

import logging
from typing import TYPE_CHECKING, Iterator, Callable, Any

from .parser import parse_tool_calls, has_tool_call, format_tool_result
from .executor import ToolExecutor

if TYPE_CHECKING:
    from ..models.message import Message
    from ..memory_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager

# Use service logger if available, otherwise standard logging
try:
    from ..service.logging import ServiceLogger
    log = ServiceLogger(__name__)
except ImportError:
    log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 5

def stream_with_tools(
    messages: list["Message"],
    stream_fn: Callable[[list["Message"]], Iterator[str]],
    memory_client: "MemoryClient | None" = None,
    profile_manager: "ProfileManager | None" = None,
    search_client: "SearchClient | None" = None,
    model_lifecycle: "ModelLifecycleManager | None" = None,
    user_id: str = "default",
    bot_id: str = "nova",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Iterator[str]:
    """Stream LLM response with tool calling support.
    
    Accumulates the full response, then checks for tool calls.
    If a tool call is found, executes it and continues with follow-up.
    
    Args:
        messages: Initial conversation messages.
        stream_fn: Function to stream from LLM. Signature: (messages) -> Iterator[str]
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for profile tools.
        search_client: Search client for web search tools.
        model_lifecycle: Model lifecycle manager for model switching tools.
        user_id: Current user ID.
        bot_id: Current bot ID.
        max_iterations: Max tool iterations per turn.
        
    Yields:
        Text chunks from the LLM response.
    """
    from ..models.message import Message
    
    executor = ToolExecutor(
        memory_client=memory_client,
        profile_manager=profile_manager,
        search_client=search_client,
        model_lifecycle=model_lifecycle,
        user_id=user_id,
        bot_id=bot_id,
    )
    current_messages = messages.copy()

    for iteration in range(1, max_iterations + 1):
        # Only log iteration if we're past the first one (indicates tool use)
        if iteration > 1:
            log.debug(f"Tool loop iteration {iteration}/{max_iterations}")
        
        # Stream chunks while watching for tool calls
        # Buffer text to detect <tool_call> before yielding
        buffer = ""
        tool_call_started = False
        full_response_parts = []
        
        for chunk in stream_fn(current_messages):
            buffer += chunk
            
            if tool_call_started:
                # Already in a tool call - just accumulate, don't yield
                continue
            
            # Look for tool call start
            lower_buffer = buffer.lower()
            
            if "<tool_call>" in lower_buffer:
                # Found complete tool call start
                tool_call_started = True
                idx = lower_buffer.find("<tool_call>")
                if idx > 0:
                    # Yield text before the tool call
                    pre_tool = buffer[:idx]
                    yield pre_tool
                    full_response_parts.append(pre_tool)
                # Keep tool call in buffer
                buffer = buffer[idx:]
                continue
            
            # Check if buffer ends with potential tool call start
            # Be conservative - hold back anything that could be <tool_call>
            hold_back = 0
            for i in range(min(len(buffer), 11)):  # len("<tool_call>") = 11
                suffix = buffer[-(i+1):].lower()
                if "<tool_call>"[:i+1].startswith(suffix) or suffix.startswith("<tool_call>"[:i+1]):
                    hold_back = i + 1
                    break
            
            # Yield everything except the potential tool tag start
            if hold_back > 0 and len(buffer) > hold_back:
                safe_to_yield = buffer[:-hold_back]
                yield safe_to_yield
                full_response_parts.append(safe_to_yield)
                buffer = buffer[-hold_back:]
            elif hold_back == 0 and buffer:
                # No potential tool start - yield all
                yield buffer
                full_response_parts.append(buffer)
                buffer = ""
            # else: buffer is all potential tool start, keep buffering
        
        # End of stream - handle remaining buffer
        full_response = "".join(full_response_parts) + buffer
        log.debug(f"Response received: {len(full_response)} chars")
        
        # Check if there's a complete tool call
        if not tool_call_started or "</tool_call>" not in buffer.lower():
            # No complete tool call - yield any remaining buffer (it's safe text)
            if buffer and not tool_call_started:
                yield buffer
            elif buffer and tool_call_started:
                # Incomplete tool call - this is malformed, yield it anyway
                log.warning("Incomplete tool call at end of response")
                yield buffer
            log.debug("No tool call found - done")
            return
        
        # There's a complete tool call - parse it
        tool_calls, remaining_text = parse_tool_calls(full_response)
        
        if not tool_calls:
            log.warning("Tool call markers found but parsing failed")
            yield buffer  # Yield the unparseable content
            return
        
        log.info(f"🔧 Calling tool: {tool_calls[0].name}")
        
        # Execute the tool
        tool_result = executor.execute(tool_calls[0])
        log.debug(f"Tool result: {len(tool_result)} chars")
        
        # Build continuation messages
        current_messages.append(Message(role="assistant", content=full_response))
        current_messages.append(Message(role="user", content=tool_result))
        
        # Continue to next iteration
    
    log.warning(f"Max tool iterations ({max_iterations}) reached")

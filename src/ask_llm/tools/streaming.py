"""Streaming tool loop for LLM conversations.

Handles tool calling while maintaining streaming output to the client.
"""

import logging
from typing import TYPE_CHECKING, Iterator, Callable

from .parser import parse_tool_calls, has_tool_call, format_tool_result
from .executor import ToolExecutor

if TYPE_CHECKING:
    from ..models.message import Message
    from ..memory_server.client import MemoryClient

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
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Iterator[str]:
    """Stream LLM response with tool calling support.
    
    Accumulates the full response, then checks for tool calls.
    If a tool call is found, executes it and continues with follow-up.
    
    Args:
        messages: Initial conversation messages.
        stream_fn: Function to stream from LLM. Signature: (messages) -> Iterator[str]
        memory_client: Memory client for tool execution.
        max_iterations: Max tool iterations per turn.
        
    Yields:
        Text chunks from the LLM response.
    """
    from ..models.message import Message
    
    executor = ToolExecutor(memory_client=memory_client)
    current_messages = messages.copy()
    
    for iteration in range(1, max_iterations + 1):
        # Only log iteration if we're past the first one (indicates tool use)
        if iteration > 1:
            log.debug(f"Tool loop iteration {iteration}/{max_iterations}")
        
        # First, collect the full response
        chunks = []
        for chunk in stream_fn(current_messages):
            chunks.append(chunk)
        
        full_response = "".join(chunks)
        log.debug(f"Response received: {len(full_response)} chars")
        
        # Check if there's a tool call in the response
        if "<tool_call>" not in full_response.lower() or "</tool_call>" not in full_response.lower():
            # No tool call - yield all chunks and we're done
            log.debug("No tool call found - yielding response")
            for chunk in chunks:
                yield chunk
            return
        
        # There's a tool call - parse it
        tool_calls, remaining_text = parse_tool_calls(full_response)
        
        if not tool_calls:
            log.warning("Tool call markers found but parsing failed")
            for chunk in chunks:
                yield chunk
            return
        
        log.info(f"🔧 Calling tool: {tool_calls[0].name}")
        
        # Yield any text before the tool call
        idx = full_response.lower().find("<tool_call>")
        if idx > 0:
            yield full_response[:idx]
        
        # Execute the tool
        tool_result = executor.execute(tool_calls[0])
        log.debug(f"Tool result: {len(tool_result)} chars")
        
        # Build continuation messages
        current_messages.append(Message(role="assistant", content=full_response))
        current_messages.append(Message(role="user", content=tool_result))
        
        # Continue to next iteration
    
    log.warning(f"Max tool iterations ({max_iterations}) reached")

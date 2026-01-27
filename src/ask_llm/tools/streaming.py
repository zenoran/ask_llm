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
    from ..utils.config import Config

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
    config: "Config | None" = None,
    user_id: str = "",  # Required - must be passed explicitly
    bot_id: str = "nova",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Iterator[str]:
    """Stream LLM response with tool calling support.
    
    Strategy: Stream all content immediately UNLESS we detect a definitive
    tool call pattern. For small models that output tool calls in various
    formats, we check the complete response at the end.
    
    Args:
        messages: Initial conversation messages.
        stream_fn: Function to stream from LLM. Signature: (messages) -> Iterator[str]
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for profile tools.
        search_client: Search client for web search tools.
        model_lifecycle: Model lifecycle manager for model switching tools.
        config: Application config (used for lazy search setup).
        user_id: Current user ID (required).
        bot_id: Current bot ID.
        max_iterations: Max tool iterations per turn.
        
    Yields:
        Text chunks from the LLM response.
    """
    if not user_id:
        raise ValueError("user_id is required for stream_with_tools")
    from ..models.message import Message
    
    executor = ToolExecutor(
        memory_client=memory_client,
        profile_manager=profile_manager,
        search_client=search_client,
        model_lifecycle=model_lifecycle,
        config=config,
        user_id=user_id,
        bot_id=bot_id,
    )
    current_messages = messages.copy()

    for iteration in range(1, max_iterations + 1):
        # Only log iteration if we're past the first one (indicates tool use)
        if iteration > 1:
            log.debug(f"Tool loop iteration {iteration}/{max_iterations}")
        
        # STREAMING APPROACH:
        # 1. Buffer initial tokens to check if response STARTS with a tool call
        # 2. If it looks like text, switch to immediate streaming
        # 3. If it looks like a tool call, buffer everything and process
        #
        # Tool calls from models typically start with markers like:
        # - <tool_call>
        # - ```json or ```python or ``` followed by {"name":
        # - `{"name":
        #
        # Normal responses start with regular text.
        
        initial_buffer = ""
        initial_chunks = []
        streaming_mode = False  # True = stream immediately, False = still deciding
        full_response = ""
        
        # How many chars to buffer before deciding
        DECISION_THRESHOLD = 50
        
        def looks_like_tool_start(text: str) -> bool:
            """Check if text looks like it's starting a tool call."""
            stripped = text.strip()
            lower = stripped.lower()
            
            # Definite tool markers
            if lower.startswith('<tool_call>') or lower.startswith('<function_call>'):
                return True
            if lower.startswith('<|im_start|>tool'):
                return True
            
            # Code block that might be a tool call
            if stripped.startswith('```'):
                # Check if it's followed by {"name" pattern
                rest = stripped[3:].lstrip()
                if rest.startswith('json') or rest.startswith('python'):
                    rest = rest[4:].lstrip() if rest.startswith('json') else rest[6:].lstrip()
                if rest.startswith('{"name"') or rest.startswith("{'name'"):
                    return True
                # Just ``` followed by { could be tool call
                if rest.startswith('{'):
                    return True
            
            # Inline code with tool JSON
            if stripped.startswith('`{"name"') or stripped.startswith("`{'name'"):
                return True
            
            # Raw JSON tool call (no wrapper)
            if stripped.startswith('{"name"') and '"arguments"' in stripped:
                return True
            
            return False
        
        def looks_like_regular_text(text: str) -> bool:
            """Check if text clearly looks like regular prose, not a tool call."""
            stripped = text.strip()
            if not stripped:
                return False
            
            # If it starts with a letter or common punctuation, it's likely text
            first_char = stripped[0]
            if first_char.isalpha():
                return True
            if first_char in '!?.,;:()[]"\'—–-':
                return True
            # Emoji or other unicode that's not { or < or `
            if first_char not in '{<`':
                return True
            
            return False
        
        for chunk in stream_fn(current_messages):
            full_response += chunk
            
            if streaming_mode:
                # Already decided to stream - just yield
                yield chunk
                continue
            
            # Still deciding - buffer
            initial_buffer += chunk
            initial_chunks.append(chunk)
            
            # Check if we can make a decision
            stripped = initial_buffer.strip()
            
            if looks_like_tool_start(stripped):
                # Looks like a tool call - buffer everything
                log.debug("Response looks like tool call - buffering")
                # Don't yield anything yet, continue buffering
                continue
            
            if looks_like_regular_text(stripped):
                # Looks like regular text - switch to streaming mode
                log.debug("Response looks like regular text - streaming")
                streaming_mode = True
                # Yield all buffered chunks
                for buffered_chunk in initial_chunks:
                    yield buffered_chunk
                initial_chunks = []
                continue
            
            # Haven't decided yet - check if we've buffered enough
            if len(stripped) >= DECISION_THRESHOLD:
                # Buffered enough without seeing tool markers - assume it's text
                log.debug("Decision threshold reached - streaming")
                streaming_mode = True
                for buffered_chunk in initial_chunks:
                    yield buffered_chunk
                initial_chunks = []
        
        log.debug(f"Response received: {len(full_response)} chars")
        
        # If we never switched to streaming mode, we buffered everything
        if not streaming_mode:
            # Check if it's actually a tool call
            if has_tool_call(full_response):
                log.debug("Tool call confirmed in buffered response")
                
                tool_calls, remaining_text = parse_tool_calls(full_response)
                
                if not tool_calls:
                    log.warning("Tool call markers found but parsing failed - yielding as text")
                    for chunk in initial_chunks:
                        yield chunk
                    return
                
                log.info(f"🔧 Calling tool: {tool_calls[0].name}")
                
                # Yield any text before the tool call
                if remaining_text.strip():
                    yield remaining_text
                
                # Execute the tool
                tool_result = executor.execute(tool_calls[0])
                log.debug(f"Tool result: {len(tool_result)} chars")
                
                # Build continuation messages
                current_messages.append(Message(role="assistant", content=full_response))
                current_messages.append(Message(role="user", content=tool_result))
                
                # Continue to next iteration for follow-up
                continue
            else:
                # Not a tool call after all - yield buffered content
                log.debug("Not a tool call - yielding buffered content")
                for chunk in initial_chunks:
                    yield chunk
                return
        
        # Streaming mode was active - response is complete, check for tool calls
        # (In case there's a tool call AFTER regular text, which would be unusual)
        if has_tool_call(full_response):
            log.warning("Tool call found after streaming - this is unusual")
            # Already yielded text, can't un-yield. Just log and return.
        
        return
    
    log.warning(f"Max tool iterations ({max_iterations}) reached")

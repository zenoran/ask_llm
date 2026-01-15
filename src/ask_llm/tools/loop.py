"""Tool calling loop for LLM conversations.

Handles the iterative process of:
1. Querying the LLM
2. Detecting tool calls in the response
3. Executing tools via MCP
4. Feeding results back to the LLM
5. Repeating until final response
"""

import logging
from typing import TYPE_CHECKING, Callable

from .parser import parse_tool_calls, has_tool_call, format_tool_result
from .executor import ToolExecutor

if TYPE_CHECKING:
    from ..models.message import Message
    from ..memory_server.client import MemoryClient
    from ..profiles import ProfileManager

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ITERATIONS = 5


class ToolLoop:
    """Manages tool calling iterations for a single conversation turn."""
    
    def __init__(
        self,
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        user_id: str = "default",
        bot_id: str = "nova",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """
        Args:
            memory_client: Client for memory operations.
            profile_manager: Manager for profile operations.
            user_id: Current user ID.
            bot_id: Current bot ID.
            max_iterations: Maximum tool call iterations per turn.
        """
        self.executor = ToolExecutor(
            memory_client=memory_client,
            profile_manager=profile_manager,
            user_id=user_id,
            bot_id=bot_id,
        )
        self.max_iterations = max_iterations
    
    def run(
        self,
        messages: list["Message"],
        query_fn: Callable[[list["Message"], bool], str],
        stream_final: bool = True,
    ) -> str:
        """Run the tool calling loop.
        
        Args:
            messages: Initial conversation messages.
            query_fn: Function to call LLM. Signature: (messages, stream) -> response
            stream_final: Whether to stream the final response.
            
        Returns:
            Final response text (after all tool calls resolved).
        """
        from ..models.message import Message
        
        self.executor.reset_call_count()
        current_messages = messages.copy()
        
        for iteration in range(1, self.max_iterations + 1):
            # Only log iteration if we're past the first one (indicates tool use)
            if iteration > 1:
                logger.debug(f"Tool loop iteration {iteration}/{self.max_iterations}")
            
            # Query LLM - only stream on potential final iteration
            is_last_chance = (iteration == self.max_iterations)
            response = query_fn(current_messages, stream_final and is_last_chance)
            
            if not response:
                logger.debug("Empty response from LLM")
                return ""
            
            # Check for tool calls
            if not has_tool_call(response):
                logger.debug("No tool calls found - returning final response")
                return response
            
            # Parse tool calls
            tool_calls, remaining_text = parse_tool_calls(response)
            
            if not tool_calls:
                logger.warning("Tool marker found but no valid calls parsed")
                return response
            
            # Execute tools
            tool_results = self._execute_tools(tool_calls)
            
            logger.info(
                "🔧 Tool calls: %s",
                ", ".join(tc.name for tc in tool_calls)
            )
            
            # Build continuation messages
            current_messages.append(Message(role="assistant", content=response))
            current_messages.append(Message(
                role="user", 
                content="\n\n".join(tool_results)
            ))
        
        # Max iterations reached
        logger.warning(f"Tool loop: max iterations ({self.max_iterations}) reached")
        return response if 'response' in dir() else ""
    
    def _execute_tools(self, tool_calls: list) -> list[str]:
        """Execute a list of tool calls and return formatted results."""
        results = []
        
        for tc in tool_calls:
            if not self.executor.can_execute_more():
                results.append(format_tool_result(
                    tc.name, 
                    None, 
                    error="Too many tool calls this turn"
                ))
                break
            
            result = self.executor.execute(tc)
            results.append(result)
            logger.debug(f"Tool {tc.name} executed: {result[:100]}...")
        
        return results


def query_with_tools(
    messages: list["Message"],
    query_fn: Callable[[list["Message"], bool], str],
    memory_client: "MemoryClient | None" = None,
    profile_manager: "ProfileManager | None" = None,
    user_id: str = "default",
    bot_id: str = "nova",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    stream: bool = True,
) -> str:
    """Convenience function for tool-enabled queries.
    
    Args:
        messages: Conversation messages.
        query_fn: LLM query function (messages, stream) -> response.
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for user/bot profile tools.
        user_id: Current user ID.
        bot_id: Current bot ID.
        max_iterations: Max tool iterations.
        stream: Whether to stream the final response.
        
    Returns:
        Final response after tool resolution.
    """
    loop = ToolLoop(
        memory_client=memory_client,
        profile_manager=profile_manager,
        user_id=user_id,
        bot_id=bot_id,
        max_iterations=max_iterations,
    )
    return loop.run(messages, query_fn, stream_final=stream)

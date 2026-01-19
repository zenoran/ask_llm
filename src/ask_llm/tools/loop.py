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
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ITERATIONS = 5


class ToolLoop:
    """Manages tool calling iterations for a single conversation turn."""
    
    def __init__(
        self,
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        search_client: "SearchClient | None" = None,
        model_lifecycle: "ModelLifecycleManager | None" = None,
        user_id: str = "default",
        bot_id: str = "nova",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """
        Args:
            memory_client: Client for memory operations.
            profile_manager: Manager for profile operations.
            search_client: Client for web search operations.
            model_lifecycle: Manager for model lifecycle operations.
            user_id: Current user ID.
            bot_id: Current bot ID.
            max_iterations: Maximum tool call iterations per turn.
        """
        self.executor = ToolExecutor(
            memory_client=memory_client,
            profile_manager=profile_manager,
            search_client=search_client,
            model_lifecycle=model_lifecycle,
            user_id=user_id,
            bot_id=bot_id,
        )
        self.max_iterations = max_iterations
        self.tool_context: list[dict] = []  # Track tool interactions for history
    
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
        self.tool_context = []  # Reset tool context
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
            
            # Track tool interactions for history/context
            tool_summary = "\n\n".join(tool_results)
            self.tool_context.append({
                "tools_called": [tc.name for tc in tool_calls],
                "results": tool_summary,
            })
            
            # Build continuation messages
            current_messages.append(Message(role="assistant", content=response))
            current_messages.append(Message(
                role="user", 
                content=tool_summary
            ))
        
        # Max iterations reached
        logger.warning(f"Tool loop: max iterations ({self.max_iterations}) reached")
        return response if 'response' in dir() else ""
    
    def get_tool_context_summary(self) -> str:
        """Return a summary of tool interactions for saving to history."""
        if not self.tool_context:
            return ""
        
        summaries = []
        for ctx in self.tool_context:
            tools = ", ".join(ctx["tools_called"])
            summaries.append(f"[Tools used: {tools}]\n{ctx['results']}")
        return "\n\n".join(summaries)
    
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
    search_client: "SearchClient | None" = None,
    model_lifecycle: "ModelLifecycleManager | None" = None,
    user_id: str = "default",
    bot_id: str = "nova",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    stream: bool = True,
) -> tuple[str, str]:
    """Convenience function for tool-enabled queries.
    
    Args:
        messages: Conversation messages.
        query_fn: LLM query function (messages, stream) -> response.
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for user/bot profile tools.
        search_client: Search client for web search tools.
        model_lifecycle: Model lifecycle manager for model switching tools.
        user_id: Current user ID.
        bot_id: Current bot ID.
        max_iterations: Max tool iterations.
        stream: Whether to stream the final response.
        
    Returns:
        Tuple of (final_response, tool_context_summary).
        tool_context_summary contains the tool results that should be saved to history.
    """
    loop = ToolLoop(
        memory_client=memory_client,
        profile_manager=profile_manager,
        search_client=search_client,
        model_lifecycle=model_lifecycle,
        user_id=user_id,
        bot_id=bot_id,
        max_iterations=max_iterations,
    )
    response = loop.run(messages, query_fn, stream_final=stream)
    tool_context = loop.get_tool_context_summary()
    return response, tool_context

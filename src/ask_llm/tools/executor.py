"""Tool executor for LLM tool calls.

Executes tool calls by routing them to the appropriate backend
(MCP memory server, profiles, etc.) and returning formatted results.
"""

import logging
from typing import Any, TYPE_CHECKING

from .parser import ToolCall, format_tool_result, format_memories_for_result

if TYPE_CHECKING:
    from ..memory_server.client import MemoryClient
    from ..profiles import ProfileManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls using the memory client and profile manager.
    
    Routes tool calls to the appropriate backend and formats results
    for injection back into the conversation.
    """
    
    # Maximum number of tool calls per conversation turn (prevent infinite loops)
    MAX_TOOL_CALLS_PER_TURN = 5
    
    def __init__(
        self, 
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        user_id: str = "default",
        bot_id: str = "nova",
    ):
        """Initialize the executor.
        
        Args:
            memory_client: Memory client for memory operations.
            profile_manager: Profile manager for user/bot profile operations.
            user_id: Current user ID for profile operations.
            bot_id: Current bot ID for bot personality operations.
        """
        self.memory_client = memory_client
        self.profile_manager = profile_manager
        self.user_id = user_id
        self.bot_id = bot_id
        self._call_count = 0
    
    def reset_call_count(self):
        """Reset the per-turn call counter."""
        self._call_count = 0
    
    def can_execute_more(self) -> bool:
        """Check if more tool calls are allowed this turn."""
        return self._call_count < self.MAX_TOOL_CALLS_PER_TURN
    
    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return formatted result.
        
        Args:
            tool_call: The parsed tool call to execute.
            
        Returns:
            Formatted result string for injection into conversation.
        """
        self._call_count += 1
        
        if not self.can_execute_more():
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Too many tool calls this turn (max {self.MAX_TOOL_CALLS_PER_TURN})"
            )
        
        logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        
        try:
            # Memory tools
            if tool_call.name == "search_memories":
                return self._execute_search_memories(tool_call)
            elif tool_call.name == "store_memory":
                return self._execute_store_memory(tool_call)
            elif tool_call.name == "delete_memory":
                return self._execute_delete_memory(tool_call)
            # Profile tools
            elif tool_call.name == "set_user_attribute":
                return self._execute_set_user_attribute(tool_call)
            elif tool_call.name == "get_user_profile":
                return self._execute_get_user_profile(tool_call)
            elif tool_call.name == "delete_user_attribute":
                return self._execute_delete_user_attribute(tool_call)
            elif tool_call.name == "set_my_trait":
                return self._execute_set_my_trait(tool_call)
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Unknown tool: {tool_call.name}"
                )
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_call.name}")
            return format_tool_result(tool_call.name, None, error=str(e))
    
    def _execute_search_memories(self, tool_call: ToolCall) -> str:
        """Execute search_memories tool."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 5)
        
        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )
        
        try:
            results = self.memory_client.search(query, n_results=n_results)
            memories = [
                {
                    "id": r.id,
                    "content": r.content,
                    "relevance": r.relevance,
                    "importance": r.importance,
                    "tags": r.tags,
                }
                for r in results
            ]
            
            formatted = format_memories_for_result(memories)
            return format_tool_result(tool_call.name, formatted)
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))
    
    def _execute_store_memory(self, tool_call: ToolCall) -> str:
        """Execute store_memory tool."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        content = tool_call.arguments.get("content", "")
        importance = tool_call.arguments.get("importance", 0.6)
        tags = tool_call.arguments.get("tags", ["misc"])
        
        if not content:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: content"
            )
        
        # Ensure importance is in valid range
        importance = max(0.0, min(1.0, float(importance)))
        
        # Ensure tags is a list
        if isinstance(tags, str):
            tags = [tags]
        
        try:
            result = self.memory_client.store_memory(
                content=content,
                importance=importance,
                tags=tags,
            )
            
            return format_tool_result(
                tool_call.name,
                f"Memory stored successfully with ID: {result.id[:8]}"
            )
            
        except Exception as e:
            logger.error(f"Memory store failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))
    
    def _execute_delete_memory(self, tool_call: ToolCall) -> str:
        """Execute delete_memory tool."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        memory_id = tool_call.arguments.get("memory_id", "")
        
        if not memory_id:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: memory_id"
            )
        
        try:
            success = self.memory_client.delete_memory(memory_id)
            
            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Memory {memory_id} deleted successfully"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Memory {memory_id} not found"
                )
                
        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    # =========================================================================
    # Profile Tool Execution
    # =========================================================================

    def _execute_set_user_attribute(self, tool_call: ToolCall) -> str:
        """Execute set_user_attribute tool."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )
        
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        value = tool_call.arguments.get("value")
        confidence = tool_call.arguments.get("confidence", 0.8)
        
        if not category or not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameters: category and key"
            )
        
        # Validate category
        valid_categories = ["preference", "fact", "interest", "communication", "context"]
        if category.lower() not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        try:
            from ..profiles import EntityType
            
            self.profile_manager.set_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
                value=value,
                confidence=float(confidence),
                source="inferred",
            )
            
            return format_tool_result(
                tool_call.name,
                f"Saved user {category}: {key} = {value}"
            )
            
        except Exception as e:
            logger.error(f"Set user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_get_user_profile(self, tool_call: ToolCall) -> str:
        """Execute get_user_profile tool."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )
        
        try:
            summary = self.profile_manager.get_user_profile_summary(self.user_id)
            
            if not summary:
                return format_tool_result(
                    tool_call.name,
                    "No profile attributes stored for this user yet."
                )
            
            return format_tool_result(tool_call.name, summary)
            
        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_delete_user_attribute(self, tool_call: ToolCall) -> str:
        """Execute delete_user_attribute tool."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )
        
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        
        if not category or not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameters: category and key"
            )
        
        try:
            from ..profiles import EntityType
            
            success = self.profile_manager.delete_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
            )
            
            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Deleted user attribute: {category}.{key}"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Attribute {category}.{key} not found"
                )
                
        except Exception as e:
            logger.error(f"Delete user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_set_my_trait(self, tool_call: ToolCall) -> str:
        """Execute set_my_trait tool (bot personality development)."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )
        
        key = tool_call.arguments.get("key", "")
        value = tool_call.arguments.get("value")
        
        if not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: key"
            )
        
        try:
            from ..profiles import EntityType, AttributeCategory
            
            self.profile_manager.set_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=AttributeCategory.PERSONALITY,
                key=key,
                value=value,
                confidence=1.0,
                source="self",
            )
            
            return format_tool_result(
                tool_call.name,
                f"Recorded personality trait: {key} = {value}"
            )
            
        except Exception as e:
            logger.error(f"Set bot trait failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

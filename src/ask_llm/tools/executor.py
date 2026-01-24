"""Tool executor for LLM tool calls.

Executes tool calls by routing them to the appropriate backend
(MCP memory server, profiles, web search, model management, etc.) and returning formatted results.
"""

import logging
from typing import Any, Callable, TYPE_CHECKING

from .parser import ToolCall, format_tool_result, format_memories_for_result

if TYPE_CHECKING:
    from ..memory_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager

logger = logging.getLogger(__name__)

# Try to get service logger for verbose tool result logging
try:
    from ..service.logging import get_service_logger
    slog = get_service_logger(__name__)
except ImportError:
    slog = None


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
        search_client: "SearchClient | None" = None,
        model_lifecycle: "ModelLifecycleManager | None" = None,
        user_id: str = "",  # Required - must be passed explicitly
        bot_id: str = "nova",
    ):
        """Initialize the executor.
        
        Args:
            memory_client: Memory client for memory operations.
            profile_manager: Profile manager for user/bot profile operations.
            search_client: Search client for web search operations.
            model_lifecycle: Model lifecycle manager for model switching.
            user_id: Current user ID for profile operations (required).
            bot_id: Current bot ID for bot personality operations.
        """
        if not user_id:
            raise ValueError("user_id is required for ToolExecutor")
        self.memory_client = memory_client
        self.profile_manager = profile_manager
        self.search_client = search_client
        self.model_lifecycle = model_lifecycle
        self.user_id = user_id
        self.bot_id = bot_id
        self._call_count = 0
        
        # Tool dispatch table - maps tool names to handler methods
        self._handlers: dict[str, Callable[[ToolCall], str]] = {
            # Memory tools
            "search_memories": self._execute_search_memories,
            "store_memory": self._execute_store_memory,
            "delete_memory": self._execute_delete_memory,
            "search_history": self._execute_search_history,
            "forget_history": self._execute_forget_history,
            # Profile tools
            "set_user_attribute": self._execute_set_user_attribute,
            "get_user_profile": self._execute_get_user_profile,
            "delete_user_attribute": self._execute_delete_user_attribute,
            "set_my_trait": self._execute_set_my_trait,
            # Web search tools
            "web_search": self._execute_web_search,
            "news_search": self._execute_news_search,
            # Model management tools
            "list_models": self._execute_list_models,
            "get_current_model": self._execute_get_current_model,
            "switch_model": self._execute_switch_model,
            # Utility tools
            "get_current_time": self._execute_get_current_time,
        }
    
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
        
        result: str
        try:
            handler = self._handlers.get(tool_call.name)
            if handler:
                result = handler(tool_call)
            else:
                result = format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Unknown tool: {tool_call.name}"
                )
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_call.name}")
            result = format_tool_result(tool_call.name, None, error=str(e))
        
        # Log the result in verbose mode
        if slog:
            slog.tool_result(tool_call.name, result)
        
        return result
    
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
        """Execute delete_memory tool - delete by ID or by query."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        memory_id = tool_call.arguments.get("memory_id", "")
        query = tool_call.arguments.get("query", "")
        
        if not memory_id and not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: provide either memory_id or query"
            )
        
        try:
            # If query provided, search for matching memories and delete them
            if query:
                # Search for memories matching the query
                memories = self.memory_client.search(
                    query=query,
                    n_results=50,  # Get up to 50 matches
                    min_relevance=0.3  # Reasonable threshold for deletion
                )
                
                if not memories:
                    return format_tool_result(
                        tool_call.name,
                        f"No memories found matching '{query}'"
                    )
                
                # Delete each matching memory
                deleted_count = 0
                deleted_contents = []
                for memory in memories:
                    mid = memory.id
                    if mid:
                        success = self.memory_client.delete_memory(mid)
                        if success:
                            deleted_count += 1
                            content = memory.content[:50] if memory.content else ""
                            deleted_contents.append(f"- {content}...")
                
                if deleted_count > 0:
                    result = f"Deleted {deleted_count} memories matching '{query}':\n" + "\n".join(deleted_contents[:5])
                    if deleted_count > 5:
                        result += f"\n... and {deleted_count - 5} more"
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=f"Found {len(memories)} memories but failed to delete them"
                    )
            
            # Delete by specific ID
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

    def _execute_search_history(self, tool_call: ToolCall) -> str:
        """Execute search_history tool - searches ALL conversation messages."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 10)
        role_filter = tool_call.arguments.get("role_filter")
        
        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )
        
        try:
            results = self.memory_client.search_messages(
                query=query,
                n_results=n_results,
                role_filter=role_filter,
            )
            
            if not results:
                return format_tool_result(
                    tool_call.name,
                    "No messages found matching your search."
                )
            
            # Format results for the model
            lines = [f"Found {len(results)} messages matching '{query}':"]
            for i, msg in enumerate(results, 1):
                # Truncate long messages
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                # Escape newlines for readability
                content = content.replace("\n", " ")
                lines.append(f"{i}. [{msg.role}] {content}")
            
            return format_tool_result(tool_call.name, "\n".join(lines))
            
        except Exception as e:
            logger.error(f"History search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_forget_history(self, tool_call: ToolCall) -> str:
        """Execute forget_history tool - deletes recent messages and related memories."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )
        
        count = tool_call.arguments.get("count")
        minutes = tool_call.arguments.get("minutes")
        
        # Need either count or minutes
        if count is None and minutes is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="Must specify either 'count' (number of messages) or 'minutes' (time range)"
            )
        
        try:
            if count is not None:
                result = self.memory_client.forget_recent_messages(int(count))
                msg = f"Forgot {result['messages_ignored']} messages"
            else:
                result = self.memory_client.forget_messages_since_minutes(int(minutes))
                msg = f"Forgot {result['messages_ignored']} messages from the last {minutes} minutes"
            
            if result.get('memories_deleted', 0) > 0:
                msg += f" and {result['memories_deleted']} related memories"
            
            return format_tool_result(tool_call.name, msg)
            
        except Exception as e:
            logger.error(f"Forget history failed: {e}")
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
            
            logger.info(f"Setting user attribute: {category}.{key} = {value} for user {self.user_id}")
            
            self.profile_manager.set_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
                value=value,
                confidence=float(confidence),
                source="inferred",
            )
            
            logger.info(f"Successfully saved user attribute: {category}.{key}")
            
            return format_tool_result(
                tool_call.name,
                f"Saved user {category}: {key} = {value}"
            )
            
        except Exception as e:
            logger.exception(f"Set user attribute failed: {e}")
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
        """Execute delete_user_attribute tool - delete by category+key or by query."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )
        
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        query = tool_call.arguments.get("query", "")
        
        if not query and (not category or not key):
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameters: provide either (category and key) or query"
            )
        
        try:
            from ..profiles import EntityType
            
            # If query provided, search and delete matching attributes
            if query:
                count, deleted = self.profile_manager.search_and_delete_attributes(
                    entity_type=EntityType.USER,
                    entity_id=self.user_id,
                    query=query,
                    category=category if category else None,
                )
                
                if count > 0:
                    result = f"Deleted {count} attribute(s) matching '{query}':\n" + "\n".join(f"- {d}" for d in deleted[:5])
                    if count > 5:
                        result += f"\n... and {count - 5} more"
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        f"No attributes found matching '{query}'"
                    )
            
            # Delete by exact category+key
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

    # =========================================================================
    # Web Search Tool Execution
    # =========================================================================

    def _execute_web_search(self, tool_call: ToolCall) -> str:
        """Execute web_search tool."""
        if not self.search_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Web search not available"
            )
        
        query = tool_call.arguments.get("query", "")
        max_results = tool_call.arguments.get("max_results", 5)
        
        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )
        
        try:
            results = self.search_client.search(query, max_results=max_results)
            formatted = self.search_client.format_results_for_llm(results)
            
            logger.info(f"Web search '{query}' returned {len(results)} results")
            return format_tool_result(tool_call.name, formatted)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_news_search(self, tool_call: ToolCall) -> str:
        """Execute news_search tool."""
        if not self.search_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="News search not available"
            )
        
        query = tool_call.arguments.get("query", "")
        time_range = tool_call.arguments.get("time_range", "w")
        max_results = tool_call.arguments.get("max_results", 5)
        
        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )
        
        try:
            results = self.search_client.search_news(
                query, 
                max_results=max_results,
                time_range=time_range,
            )
            formatted = self.search_client.format_results_for_llm(results)
            
            logger.info(f"News search '{query}' returned {len(results)} results")
            return format_tool_result(tool_call.name, formatted)
            
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    # =========================================================================
    # Model Management Tool Execution
    # =========================================================================

    def _execute_list_models(self, tool_call: ToolCall) -> str:
        """Execute list_models tool - returns available model shortcuts."""
        if not self.model_lifecycle:
            return format_tool_result(
                tool_call.name,
                None,
                error="Model management not available"
            )
        
        try:
            models = self.model_lifecycle.get_available_models()
            current = self.model_lifecycle.current_model
            
            if not models:
                return format_tool_result(
                    tool_call.name,
                    "No models configured."
                )
            
            # Format as a simple list with current model marked
            lines = ["Available models:"]
            for model in sorted(models):
                if model == current:
                    lines.append(f"  • {model} (current)")
                else:
                    lines.append(f"  • {model}")
            
            return format_tool_result(tool_call.name, "\n".join(lines))
            
        except Exception as e:
            logger.error(f"List models failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_get_current_model(self, tool_call: ToolCall) -> str:
        """Execute get_current_model tool - returns current model info."""
        if not self.model_lifecycle:
            return format_tool_result(
                tool_call.name,
                None,
                error="Model management not available"
            )
        
        try:
            current = self.model_lifecycle.current_model
            
            if not current:
                return format_tool_result(
                    tool_call.name,
                    "No model currently loaded."
                )
            
            # Get model info for more details
            model_info = self.model_lifecycle.get_model_info(current)
            if model_info:
                model_type = model_info.get("type", "unknown")
                model_id = model_info.get("model_id", model_info.get("repo_id", ""))
                description = model_info.get("description", "")
                
                result = f"Current model: {current}\n"
                result += f"  Type: {model_type}\n"
                if model_id:
                    result += f"  Model ID: {model_id}\n"
                if description:
                    result += f"  Description: {description}"
            else:
                result = f"Current model: {current}"
            
            return format_tool_result(tool_call.name, result)
            
        except Exception as e:
            logger.error(f"Get current model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_switch_model(self, tool_call: ToolCall) -> str:
        """Execute switch_model tool - switches to a different model."""
        if not self.model_lifecycle:
            return format_tool_result(
                tool_call.name,
                None,
                error="Model management not available"
            )
        
        model_name = tool_call.arguments.get("model_name", "")
        
        if not model_name:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: model_name"
            )
        
        try:
            success, message = self.model_lifecycle.switch_model(model_name)
            
            if success:
                logger.info(f"Model switch requested: {model_name}")
                return format_tool_result(tool_call.name, message)
            else:
                return format_tool_result(tool_call.name, None, error=message)
            
        except Exception as e:
            logger.error(f"Switch model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    # ─── Utility Tools ─────────────────────────────────────────────────────────

    def _execute_get_current_time(self, tool_call: ToolCall) -> str:
        """Execute get_current_time tool - returns current date and time."""
        from datetime import datetime
        
        now = datetime.now()
        # Format: "Thursday, January 23, 2026 at 6:45 PM"
        datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        # Also include timezone if available
        try:
            import time
            tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
            datetime_str += f" ({tz_name})"
        except Exception:
            pass
        
        return format_tool_result(tool_call.name, datetime_str)

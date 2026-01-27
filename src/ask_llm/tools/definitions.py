"""Tool definitions for LLM tool calling.

Defines the available tools that bots can use, with their descriptions
and parameters in a format suitable for prompt injection.

Tool categories:
- MEMORY_TOOLS: Search/store/delete memories
- PROFILE_TOOLS: Manage user/bot attributes
- SEARCH_TOOLS: Web search capabilities
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """Definition of a tool that the LLM can call."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    
    def to_prompt_string(self) -> str:
        """Format tool for inclusion in system prompt."""
        params_str = ""
        if self.parameters:
            param_lines = []
            for p in self.parameters:
                req = "(required)" if p.required else "(optional)"
                param_lines.append(f"    - {p.name} ({p.type}): {p.description} {req}")
            params_str = "\n" + "\n".join(param_lines)
        
        return f"- **{self.name}**: {self.description}{params_str}"


# Profile tools for managing user and bot attributes
PROFILE_TOOLS = [
    Tool(
        name="set_user_attribute",
        description="Store a persistent user fact/preference for cross-conversation recall.",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="'preference', 'fact', 'interest', or 'communication'"
            ),
            ToolParameter(
                name="key",
                type="string",
                description="Attribute name, e.g. 'occupation', 'favorite_color'"
            ),
            ToolParameter(
                name="value",
                type="any",
                description="Value to store (string, number, boolean, or list)"
            ),
            ToolParameter(
                name="confidence",
                type="float",
                description="0.0-1.0 (1.0=explicit statement, 0.6-0.8=inferred)",
                required=False,
                default=0.8
            ),
        ]
    ),
    Tool(
        name="get_user_profile",
        description="Get all stored user preferences, facts, and interests.",
        parameters=[]
    ),
    Tool(
        name="delete_user_attribute",
        description="Remove a user attribute by category+key or by query.",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="'preference', 'fact', 'interest', 'communication' (optional if using query)",
                required=False
            ),
            ToolParameter(
                name="key",
                type="string",
                description="Attribute key to delete (optional if using query)",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search term to find/delete matching attributes",
                required=False
            ),
        ]
    ),
    Tool(
        name="set_my_trait",
        description="Record your own personality trait (e.g. humor_style, topic_expertise).",
        parameters=[
            ToolParameter(
                name="key",
                type="string",
                description="Trait name"
            ),
            ToolParameter(
                name="value",
                type="any",
                description="Trait value"
            ),
        ]
    ),
]


# Memory tools available to bots
MEMORY_TOOLS = [
    Tool(
        name="search_memories",
        description="Semantic search of stored facts and learned information.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query"
            ),
            ToolParameter(
                name="n_results",
                type="integer",
                description="Max results (default 5)",
                required=False,
                default=5
            ),
        ]
    ),
    Tool(
        name="store_memory",
        description="Save an important fact for future recall.",
        parameters=[
            ToolParameter(
                name="content",
                type="string",
                description="Fact to remember"
            ),
            ToolParameter(
                name="importance",
                type="float",
                description="0.0-1.0 (0.9+=core facts, 0.5-0.7=preferences)",
                required=False,
                default=0.6
            ),
            ToolParameter(
                name="tags",
                type="list[string]",
                description="Categories, e.g. ['preference'], ['fact', 'work']",
                required=False,
                default=["misc"]
            ),
        ]
    ),
    Tool(
        name="delete_memory",
        description="Remove memories by ID or query.",
        parameters=[
            ToolParameter(
                name="memory_id",
                type="string",
                description="Specific memory ID from search results",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Delete all memories matching this query",
                required=False
            ),
        ]
    ),
    Tool(
        name="search_history",
        description="Full-text search of raw conversation messages (not extracted facts).",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Keywords to search"
            ),
            ToolParameter(
                name="n_results",
                type="integer",
                description="Max results (default 10)",
                required=False,
                default=10
            ),
            ToolParameter(
                name="role_filter",
                type="string",
                description="'user', 'assistant', or omit for all",
                required=False,
                default=None
            ),
        ]
    ),
    Tool(
        name="forget_history",
        description="Delete recent messages and their extracted memories.",
        parameters=[
            ToolParameter(
                name="count",
                type="integer",
                description="Number of messages to forget (use count OR minutes)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="minutes",
                type="integer",
                description="Forget messages from last N minutes (use count OR minutes)",
                required=False,
                default=None
            ),
        ]
    ),
]


# Model management tools for switching/listing models
MODEL_TOOLS = [
    Tool(
        name="list_models",
        description="List all available AI models.",
        parameters=[]
    ),
    Tool(
        name="get_current_model",
        description="Get currently loaded model info.",
        parameters=[]
    ),
    Tool(
        name="switch_model",
        description="Switch to a different AI model.",
        parameters=[
            ToolParameter(
                name="model_name",
                type="string",
                description="Model shortcut (e.g. 'gpt4', 'claude'). Use list_models for options."
            ),
        ]
    ),
]


# Web search tools for internet access
SEARCH_TOOLS = [
    Tool(
        name="web_search",
        description="Search internet for current info, facts, or recent events.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query"
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Max results (default 5)",
                required=False,
                default=5
            ),
        ]
    ),
    Tool(
        name="news_search",
        description="Search recent news articles.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="News topic"
            ),
            ToolParameter(
                name="time_range",
                type="string",
                description="'d' (day), 'w' (week), 'm' (month)",
                required=False,
                default="w"
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Max results (default 5)",
                required=False,
                default=5
            ),
        ]
    ),
]


# Utility tools for general assistance
UTILITY_TOOLS = [
    Tool(
        name="get_current_time",
        description="Get current date and time.",
        parameters=[]
    ),
]


# All tools combined
ALL_TOOLS = MEMORY_TOOLS + PROFILE_TOOLS + MODEL_TOOLS + UTILITY_TOOLS


TOOL_CALLING_INSTRUCTIONS = '''
## Tools

Use this EXACT format for tool calls:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Output the <tool_call> block IMMEDIATELY when needed, then STOP and wait for <tool_result>.

### Available Tools:
{tools_list}
{search_guidance}
### Tool Selection:
- **Profile** (get_user_profile, set_user_attribute): For persistent user facts - check FIRST for "what do you know about me"
- **Memory** (search_memories, store_memory): For learned facts and important information
- **History** (search_history, forget_history): For raw conversation search or deletion

### Rules:
- Only use tools when you NEED information you don't have
- Call ONE tool at a time, wait for result
- TRUST tool results exactly - never contradict them
- Before saying "I don't know" about the user: check system prompt "About the User" section, then search_memories
'''

# Guidance added when search tools are enabled
SEARCH_GUIDANCE = '''
- **Search** (web_search, news_search): For current events, facts you're unsure about, or recent information
'''

# Guidance added when model tools are enabled
MODEL_GUIDANCE = '''
- **Models** (list_models, get_current_model, switch_model): For switching AI models
'''


def get_tools_prompt(
    tools: list[Tool] | None = None,
    include_profile_tools: bool = True,
    include_search_tools: bool = False,
    include_model_tools: bool = False,
) -> str:
    """Generate the tools instruction prompt.
    
    Args:
        tools: List of tools to include. If None, auto-selects based on flags.
        include_profile_tools: Whether to include profile tools (default True).
        include_search_tools: Whether to include web search tools (default False).
        include_model_tools: Whether to include model management tools (default False).
        
    Returns:
        Formatted prompt string to inject into system message.
    """
    if tools is None:
        tools = MEMORY_TOOLS.copy()
        if include_profile_tools:
            tools.extend(PROFILE_TOOLS)
        if include_search_tools:
            tools.extend(SEARCH_TOOLS)
        if include_model_tools:
            tools.extend(MODEL_TOOLS)
    
    tools_list = "\n".join(tool.to_prompt_string() for tool in tools)
    
    # Add search guidance if search tools are included
    search_guidance = ""
    if include_search_tools or any(t.name in ("web_search", "news_search") for t in tools):
        search_guidance = SEARCH_GUIDANCE
    
    # Add model guidance if model tools are included
    if include_model_tools or any(t.name in ("list_models", "switch_model", "get_current_model") for t in tools):
        search_guidance += MODEL_GUIDANCE
    
    return TOOL_CALLING_INSTRUCTIONS.format(
        tools_list=tools_list,
        search_guidance=search_guidance,
    )


def get_tool_by_name(name: str, tools: list[Tool] | None = None) -> Tool | None:
    """Get a tool definition by name."""
    if tools is None:
        tools = ALL_TOOLS
    for tool in tools:
        if tool.name == name:
            return tool
    return None

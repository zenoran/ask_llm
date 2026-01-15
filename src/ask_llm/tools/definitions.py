"""Tool definitions for LLM tool calling.

Defines the available tools that bots can use, with their descriptions
and parameters in a format suitable for prompt injection.
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
        description="Save a learned fact or preference about the user. Use this when you learn something about the user (preferences, facts, interests) that should be remembered across all conversations.",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="Category of attribute: 'preference' (likes/dislikes), 'fact' (personal info like occupation, location), 'interest' (topics they like), 'communication' (how they prefer to interact)"
            ),
            ToolParameter(
                name="key",
                type="string",
                description="Attribute name, e.g. 'favorite_color', 'occupation', 'preferred_language', 'coding_style'"
            ),
            ToolParameter(
                name="value",
                type="any",
                description="The value to store. Can be a string, number, boolean, or list."
            ),
            ToolParameter(
                name="confidence",
                type="float",
                description="How confident you are in this attribute (0.0-1.0). Use 1.0 for explicit statements, 0.6-0.8 for inferred preferences.",
                required=False,
                default=0.8
            ),
        ]
    ),
    Tool(
        name="get_user_profile",
        description="Retrieve the user's profile including all their preferences, facts, and interests. Use this to refresh your understanding of who the user is.",
        parameters=[]
    ),
    Tool(
        name="delete_user_attribute",
        description="Remove an incorrect attribute from the user's profile. Use when the user says a stored preference or fact is wrong.",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="Category of attribute: 'preference', 'fact', 'interest', 'communication'"
            ),
            ToolParameter(
                name="key",
                type="string",
                description="The attribute name to delete"
            ),
        ]
    ),
    Tool(
        name="set_my_trait",
        description="Record a personality trait or preference that you've developed. Use this to build your own personality over time based on interactions.",
        parameters=[
            ToolParameter(
                name="key",
                type="string",
                description="Trait name, e.g. 'humor_style', 'communication_preference', 'topic_expertise'"
            ),
            ToolParameter(
                name="value",
                type="any",
                description="The trait value"
            ),
        ]
    ),
]


# Memory tools available to bots
MEMORY_TOOLS = [
    Tool(
        name="search_memories",
        description="Search your memories to recall facts about the user, past conversations, or things you've learned. Use this when asked about something you should remember.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="What to search for in your memories"
            ),
            ToolParameter(
                name="n_results",
                type="integer",
                description="Maximum number of memories to retrieve",
                required=False,
                default=5
            ),
        ]
    ),
    Tool(
        name="store_memory",
        description="Save an important fact or piece of information to remember for future conversations. Use this when the user tells you something significant about themselves.",
        parameters=[
            ToolParameter(
                name="content",
                type="string",
                description="The fact or information to remember"
            ),
            ToolParameter(
                name="importance",
                type="float",
                description="How important this memory is (0.0 to 1.0). Use 0.9+ for core facts like name, profession. Use 0.5-0.7 for preferences.",
                required=False,
                default=0.6
            ),
            ToolParameter(
                name="tags",
                type="list[string]",
                description="Categories for this memory, e.g. ['preference', 'personal'], ['fact', 'work']",
                required=False,
                default=["misc"]
            ),
        ]
    ),
    Tool(
        name="delete_memory",
        description="Remove an incorrect or outdated memory. Use when the user corrects something you remembered wrong.",
        parameters=[
            ToolParameter(
                name="memory_id",
                type="string",
                description="The ID of the memory to delete (from search results)"
            ),
        ]
    ),
]


# All tools combined
ALL_TOOLS = MEMORY_TOOLS + PROFILE_TOOLS


TOOL_CALLING_INSTRUCTIONS = '''
## Tools Available

You have access to memory and profile tools. When you need to use a tool, output ONLY a tool call in this exact format:

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

After outputting a tool call, STOP and wait for the result. You will receive the result in a <tool_result> block, then you can continue your response.

### Available Tools:
{tools_list}

### When to Use Tools:

**Memory Tools** (for conversation-specific info):
- **search_memories**: When asked "do you remember...", "what's my...", or anything about past conversations
- **store_memory**: When user shares significant info that relates to a specific conversation context
- **delete_memory**: When user says you remembered something incorrectly

**Profile Tools** (for persistent user/bot attributes):
- **set_user_attribute**: When learning something fundamental about the user (name, occupation, preferences, communication style) that should persist across all conversations
- **get_user_profile**: To refresh your understanding of the user at the start of meaningful conversations
- **delete_user_attribute**: When the user says a stored attribute is wrong
- **set_my_trait**: To record your own developing personality traits and preferences

### Memory vs Profile:
- Use **memories** for: conversation events, specific discussions, things that happened
- Use **profiles** for: who the user IS (attributes, preferences, facts about them)

### Important:
- Only call ONE tool at a time
- Wait for the result before continuing
- If a tool returns no results, say so naturally
- Don't make up information - if you don't find it in memory, admit you don't know
'''


def get_tools_prompt(tools: list[Tool] | None = None, include_profile_tools: bool = True) -> str:
    """Generate the tools instruction prompt.
    
    Args:
        tools: List of tools to include. Defaults to ALL_TOOLS.
        include_profile_tools: Whether to include profile tools (default True).
        
    Returns:
        Formatted prompt string to inject into system message.
    """
    if tools is None:
        if include_profile_tools:
            tools = ALL_TOOLS
        else:
            tools = MEMORY_TOOLS
    
    tools_list = "\n".join(tool.to_prompt_string() for tool in tools)
    return TOOL_CALLING_INSTRUCTIONS.format(tools_list=tools_list)


def get_tool_by_name(name: str, tools: list[Tool] | None = None) -> Tool | None:
    """Get a tool definition by name."""
    if tools is None:
        tools = ALL_TOOLS
    for tool in tools:
        if tool.name == name:
            return tool
    return None

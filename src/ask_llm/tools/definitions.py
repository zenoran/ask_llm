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
        description="Remove an attribute from the user's profile. Use when the user says a stored preference or fact is wrong. You can delete by exact category+key or by searching with a query.",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="Category of attribute: 'preference', 'fact', 'interest', 'communication'. Optional if using query.",
                required=False
            ),
            ToolParameter(
                name="key",
                type="string",
                description="The exact attribute key to delete. Optional if using query.",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search term to find and delete matching attributes. Searches both key and value.",
                required=False
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
        description="Remove incorrect or outdated memories. Use when the user corrects something you remembered wrong or asks to forget something specific. You can delete by memory_id (from search results) or by query (searches and deletes matching memories).",
        parameters=[
            ToolParameter(
                name="memory_id",
                type="string",
                description="The ID of a specific memory to delete (from search results)",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Search query to find and delete matching memories. All memories matching this query will be deleted.",
                required=False
            ),
        ]
    ),
    Tool(
        name="search_history",
        description="Search the ENTIRE conversation history for specific words or phrases. Use this when you need to find a specific past conversation, something the user said before, or when search_memories doesn't find what you're looking for. This searches raw messages, not extracted memories.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Keywords or phrases to search for in conversation history"
            ),
            ToolParameter(
                name="n_results",
                type="integer",
                description="Maximum number of messages to retrieve",
                required=False,
                default=10
            ),
            ToolParameter(
                name="role_filter",
                type="string",
                description="Only search messages from this role: 'user', 'assistant', or omit for all messages",
                required=False,
                default=None
            ),
        ]
    ),
    Tool(
        name="forget_history",
        description="Delete recent conversation history. Use when asked to forget recent messages or clear conversation. Also deletes any memories extracted from those messages.",
        parameters=[
            ToolParameter(
                name="count",
                type="integer",
                description="Number of recent messages to forget. Use this OR minutes, not both.",
                required=False,
                default=None
            ),
            ToolParameter(
                name="minutes",
                type="integer",
                description="Forget all messages from the last N minutes. Use this OR count, not both.",
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
        description="List all available AI models. Use this when asked about what models are available, or before switching models to see options.",
        parameters=[]
    ),
    Tool(
        name="get_current_model",
        description="Get information about the currently loaded model. Use this when asked which model is running.",
        parameters=[]
    ),
    Tool(
        name="switch_model",
        description="Switch to a different AI model. This will unload the current model and load the new one. Use this when asked to change models or use a specific model.",
        parameters=[
            ToolParameter(
                name="model_name",
                type="string",
                description="The shortcut name of the model to switch to (e.g., 'gpt4', 'cydonia', 'claude'). Use list_models to see available options."
            ),
        ]
    ),
]


# Web search tools for internet access
SEARCH_TOOLS = [
    Tool(
        name="web_search",
        description="Search the internet for current information. Use this when asked about recent events, news, facts you're unsure about, or anything that might have changed since your training.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="What to search for. Be specific and include relevant keywords."
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return (default: 5)",
                required=False,
                default=5
            ),
        ]
    ),
    Tool(
        name="news_search",
        description="Search for recent news articles. Use this for current events, breaking news, or time-sensitive topics.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="News topic to search for"
            ),
            ToolParameter(
                name="time_range",
                type="string",
                description="How recent: 'd' (past day), 'w' (past week), 'm' (past month)",
                required=False,
                default="w"
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results (default: 5)",
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
        description="Get the current date and time. Use this when the user asks what time or date it is, or when you need to know the current time for context.",
        parameters=[]
    ),
]


# All tools combined
ALL_TOOLS = MEMORY_TOOLS + PROFILE_TOOLS + MODEL_TOOLS + UTILITY_TOOLS


TOOL_CALLING_INSTRUCTIONS = '''
## Tools Available

You have access to tools for memory, profile management, and optionally web search. When you need to use a tool, output a tool call in this EXACT format:

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

**CRITICAL**: When you want to use a tool, OUTPUT THE TOOL CALL IMMEDIATELY. Do not say "let me try" or "give me a sec" - just output the <tool_call> block. After outputting, STOP and wait for the <tool_result>.

### Example - Getting User Profile:
User: "What do you know about me?"
Your response:
<tool_call>
{{"name": "get_user_profile", "arguments": {{}}}}
</tool_call>

Then wait for result. After getting result, respond naturally with what you learned.

### Example - Searching Memories:
User: "Do you remember what we talked about yesterday?"
Your response:
<tool_call>
{{"name": "search_memories", "arguments": {{"query": "yesterday conversation"}}}}
</tool_call>

### Available Tools:
{tools_list}

### When to Use Tools:

**Profile Tools** (for WHO the user IS - use these first!):
- **get_user_profile**: ALWAYS use this when asked "what do you know about me" or similar
- **set_user_attribute**: When learning persistent facts (name, occupation, preferences)
- **delete_user_attribute**: When user says a stored attribute is wrong

**Memory Tools** (for facts you've learned and stored):
- **search_memories**: Searches extracted FACTS about the user (not raw conversation). Use when looking for specific learned information.
- **store_memory**: For important facts worth remembering permanently
- **delete_memory**: When user says a memory is incorrect

**History Tools** (for conversation history):
- **search_history**: Searches ALL past messages (full-text search). Use when looking for something specific that was said, or when search_memories doesn't find what you need.
- **forget_history**: Deletes recent messages and related memories. Use when asked to forget recent conversation, clear history, or undo recent messages. Specify count (number of messages) OR minutes (time range).
{search_guidance}
### CRITICAL - Before saying "I don't know":
If asked about the user's name, occupation, age, location, family, pets, or any personal detail:
1. First check the "About the User" section in your system prompt
2. If not there, SEARCH MEMORIES with a query like "user name" or "user occupation"
3. Only say "I don't know" AFTER searching returns no results

### Important:
- Output the <tool_call> block IMMEDIATELY when you need info - don't narrate
- Only call ONE tool at a time
- Wait for <tool_result> before responding
- If a tool returns empty/no results, say "I don't have anything stored about that"
- **TRUST THE TOOL RESULT**: When you receive a <tool_result>, use the EXACT information from it in your response. Never contradict, ignore, or make up different information than what the tool returned.
'''

# Guidance added when search tools are enabled
SEARCH_GUIDANCE = '''
**Web Search Tools** (for current/external information):
- **web_search**: For facts you're unsure about, recent events, or anything that might have changed. Search first, then respond with what you found.
- **news_search**: For current events, breaking news, time-sensitive topics
'''

# Guidance added when model tools are enabled
MODEL_GUIDANCE = '''
**Model Management Tools** (for switching AI models):
- **list_models**: See all available models and their shortcuts
- **get_current_model**: Check which model is currently running
- **switch_model**: Switch to a different model (e.g., switch_model("gpt4") to use GPT-4)
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

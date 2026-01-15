"""Tool calling support for LLMs.

This module provides prompt-based tool calling that works with any model,
allowing bots to autonomously search memories, store facts, manage profiles, etc.
"""

from .definitions import MEMORY_TOOLS, PROFILE_TOOLS, ALL_TOOLS, get_tools_prompt, Tool
from .parser import parse_tool_calls, ToolCall, has_tool_call, format_tool_result
from .executor import ToolExecutor
from .loop import ToolLoop, query_with_tools
from .streaming import stream_with_tools

__all__ = [
    "MEMORY_TOOLS",
    "PROFILE_TOOLS",
    "ALL_TOOLS",
    "get_tools_prompt", 
    "Tool",
    "parse_tool_calls",
    "ToolCall",
    "has_tool_call",
    "format_tool_result",
    "ToolExecutor",
    "ToolLoop",
    "query_with_tools",
    "stream_with_tools",
]


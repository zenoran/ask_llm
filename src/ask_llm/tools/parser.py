"""Parser for tool calls in LLM output.

Extracts tool calls from model responses that use the prompt-based
tool calling format with <tool_call> markers.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _try_fix_json(json_str: str) -> str | None:
    """Try to fix common JSON errors from LLM output.
    
    Models sometimes output malformed JSON like:
    - Extra closing braces: {"name": "foo", "arguments": {}}}
    - Missing quotes around keys
    - Trailing commas
    
    Returns:
        Fixed JSON string, or None if unfixable.
    """
    original = json_str
    
    # Fix extra closing braces (common: }}} instead of }})
    # Count opening and closing braces
    open_count = json_str.count('{')
    close_count = json_str.count('}')
    if close_count > open_count:
        # Remove extra closing braces from the end
        excess = close_count - open_count
        while excess > 0 and json_str.rstrip().endswith('}'):
            json_str = json_str.rstrip()[:-1]
            excess -= 1
    
    # Fix missing closing braces
    if open_count > close_count:
        json_str = json_str + ('}' * (open_count - close_count))
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    if json_str != original:
        return json_str
    return None


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    name: str
    arguments: dict[str, Any]
    raw_text: str  # Original text that was parsed
    
    def __str__(self) -> str:
        return f"ToolCall({self.name}, {self.arguments})"


# Pattern to match tool calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL | re.IGNORECASE
)

# Alternative patterns for models that might vary the format slightly
ALT_PATTERNS = [
    # Without closing tag (model might stop after JSON)
    re.compile(r'<tool_call>\s*(\{[^<]*\})', re.DOTALL | re.IGNORECASE),
    # With different casing or spacing
    re.compile(r'<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>', re.DOTALL),
    # Function call style
    re.compile(r'<function_call>\s*(\{.*?\})\s*</function_call>', re.DOTALL | re.IGNORECASE),
    # ChatML-style: <|im_start|>tool_call {"name": ...} (no closing tag usually)
    re.compile(r'<\|im_start\|>tool_call\s*(\{[^\n]*\})', re.DOTALL | re.IGNORECASE),
    # ChatML with newline before JSON
    re.compile(r'<\|im_start\|>tool_call\s*\n\s*(\{.*?\})', re.DOTALL | re.IGNORECASE),
]


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse tool calls from model output.
    
    Args:
        text: The model's response text.
        
    Returns:
        Tuple of (list of ToolCall objects, remaining text after tool calls removed)
    """
    tool_calls = []
    remaining_text = text
    
    # Try main pattern first
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    
    # If no matches, try alternative patterns
    if not matches:
        for pattern in ALT_PATTERNS:
            matches = list(pattern.finditer(text))
            if matches:
                break
    
    for match in matches:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            
            # Support both formats:
            # {"name": "tool", "arguments": {...}}
            # {"name": "tool", "args": {...}}
            name = data.get("name")
            arguments = data.get("arguments") or data.get("args") or {}
            
            if name:
                tool_calls.append(ToolCall(
                    name=name,
                    arguments=arguments,
                    raw_text=match.group(0)
                ))
                # Remove the tool call from remaining text
                remaining_text = remaining_text.replace(match.group(0), "", 1)
            else:
                logger.warning(f"Tool call missing 'name' field: {json_str}")
                
        except json.JSONDecodeError as e:
            # Try to fix common JSON errors from models
            fixed_json = _try_fix_json(json_str)
            if fixed_json:
                try:
                    data = json.loads(fixed_json)
                    name = data.get("name")
                    arguments = data.get("arguments") or data.get("args") or {}
                    if name:
                        tool_calls.append(ToolCall(
                            name=name,
                            arguments=arguments,
                            raw_text=match.group(0)
                        ))
                        remaining_text = remaining_text.replace(match.group(0), "", 1)
                        logger.debug(f"Fixed malformed JSON: {json_str} -> {fixed_json}")
                        continue
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse tool call JSON: {e}\nText: {json_str}")
    
    # Clean up remaining text
    remaining_text = remaining_text.strip()
    
    return tool_calls, remaining_text


def has_tool_call(text: str) -> bool:
    """Quick check if text contains a tool call marker."""
    lower = text.lower()
    return (
        "<tool_call>" in lower 
        or "<function_call>" in lower
        or "<|im_start|>tool_call" in lower
    )


def format_tool_result(tool_name: str, result: Any, error: str | None = None) -> str:
    """Format a tool result for injection back into the conversation.
    
    Args:
        tool_name: Name of the tool that was called.
        result: The result from the tool execution.
        error: Optional error message if tool failed.
        
    Returns:
        Formatted string to inject as a message.
    """
    if error:
        return f"<tool_result name=\"{tool_name}\" status=\"error\">\n{error}\n</tool_result>"
    
    # Format result nicely
    if isinstance(result, (dict, list)):
        result_str = json.dumps(result, indent=2, default=str)
    else:
        result_str = str(result)
    
    return f"<tool_result name=\"{tool_name}\" status=\"success\">\n{result_str}\n</tool_result>"


def format_memories_for_result(memories: list[dict]) -> str:
    """Format memory search results for the model.
    
    Args:
        memories: List of memory dicts from search.
        
    Returns:
        Human-readable formatted string.
    """
    if not memories:
        return "No memories found matching your query."
    
    lines = [f"Found {len(memories)} relevant memories:"]
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        relevance = mem.get("relevance", 0)
        memory_id = mem.get("id", "")[:8] if mem.get("id") else ""
        lines.append(f"{i}. [id:{memory_id}] (relevance: {relevance:.2f}) {content}")
    
    return "\n".join(lines)

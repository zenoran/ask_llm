"""Shared utilities for LLM clients.

This module provides common functionality used across different LLM client
implementations to reduce code duplication and ensure consistent behavior.
"""

import logging
from typing import List

from ..models.message import Message
from ..utils.config import Config

logger = logging.getLogger(__name__)


def prepare_api_messages(
    messages: List[Message],
    config: Config,
    merge_system_messages: bool = True,
    dedupe_consecutive_roles: bool = False,
    filter_error_messages: bool = False,
) -> List[dict]:
    """Prepare messages for API consumption with consistent handling.
    
    This function provides a unified way to prepare messages for different
    LLM APIs, handling common operations like:
    - Merging multiple system messages into one
    - Ensuring a system message exists (from config or messages)
    - De-duplicating consecutive messages with the same role
    - Filtering out error messages from assistant responses
    
    Args:
        messages: List of Message objects to prepare
        config: Config object with SYSTEM_MESSAGE setting
        merge_system_messages: If True, merge all system messages into one at the start
        dedupe_consecutive_roles: If True, remove consecutive messages with same role
        filter_error_messages: If True, remove assistant messages starting with 'ERROR:'
        
    Returns:
        List of message dicts in API format (role, content)
    """
    system_contents: List[str] = []
    non_system_messages: List[dict] = []
    system_message = config.SYSTEM_MESSAGE
    
    # First pass: separate system and non-system messages
    for msg in messages:
        api_msg = msg.to_api_format()
        if api_msg['role'] == 'system':
            if merge_system_messages:
                system_contents.append(api_msg['content'])
            else:
                # Only keep first system message if not merging
                if not system_contents:
                    system_contents.append(api_msg['content'])
        else:
            non_system_messages.append(api_msg)
    
    # Build result list
    result: List[dict] = []
    
    # Add merged system message at the start
    if system_contents:
        merged_system = "\n\n".join(system_contents)
        result.append({"role": "system", "content": merged_system})
    elif system_message:
        result.append({"role": "system", "content": system_message})
    
    # Add non-system messages
    result.extend(non_system_messages)
    
    # Optionally dedupe consecutive same-role messages
    if dedupe_consecutive_roles and len(result) > 1:
        deduped = [result[0]]
        for i in range(1, len(result)):
            if result[i]['role'] != result[i-1]['role']:
                deduped.append(result[i])
        result = deduped
    
    # Optionally filter error messages
    if filter_error_messages:
        result = [
            msg for msg in result
            if not (msg.get('role') == 'assistant' and msg.get('content', '').startswith('ERROR:'))
        ]
    
    return result


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate the number of tokens in a text.
    
    This is a rough approximation that works reasonably well for most LLMs.
    For accurate counting, use the specific tokenizer for your model.
    
    Args:
        text: Text to estimate tokens for
        chars_per_token: Average characters per token (default: 4)
        
    Returns:
        Estimated token count
    """
    return len(text) // chars_per_token


def format_verbose_params(
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    **extra_params
) -> str:
    """Format parameters for verbose logging output.
    
    Args:
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Nucleus sampling top-p
        stream: Whether streaming is enabled
        **extra_params: Additional parameters to include
        
    Returns:
        Formatted string for logging
    """
    parts = []
    
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if temperature is not None:
        parts.append(f"temp={temperature}")
    if top_p is not None:
        parts.append(f"top_p={top_p}")
    if stream is not None:
        parts.append(f"stream={stream}")
    
    for key, value in extra_params.items():
        if value is not None:
            parts.append(f"{key}={value}")
    
    return ", ".join(parts)

"""
Memory extraction service for distilling important facts from conversations.
Uses LLM to extract facts and handle memory updates/conflicts.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TYPE_CHECKING

from .prompts import (
    FACT_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
    MEMORY_TYPES,
    get_fact_extraction_prompt,
    get_memory_update_prompt,
    estimate_importance,
)

if TYPE_CHECKING:
    from ...models.message import Message

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be used for extraction."""
    
    def query(self, messages: list, plaintext_output: bool = True, **kwargs) -> str:
        """Query the LLM with messages."""
        ...


@dataclass
class ExtractedFact:
    """A fact extracted from a conversation."""
    content: str
    memory_type: str
    importance: float
    source_message_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "source_message_ids": self.source_message_ids,
        }


@dataclass
class MemoryAction:
    """An action to take on memory (ADD, UPDATE, DELETE, NONE)."""
    action: str  # ADD, UPDATE, DELETE, NONE
    fact: ExtractedFact | None = None
    target_memory_id: str | None = None
    reason: str = ""


class MemoryExtractionService:
    """
    Service for extracting important facts from conversations and managing memory updates.
    
    This service uses an LLM to:
    1. Extract discrete facts from conversation exchanges
    2. Compare new facts against existing memories
    3. Determine appropriate actions (ADD, UPDATE, DELETE, NONE)
    """
    
    def __init__(self, llm_client: LLMClientProtocol | None = None):
        """
        Initialize the extraction service.
        
        Args:
            llm_client: Optional LLM client for extraction. If not provided,
                       will use heuristic-based extraction.
        """
        self.llm_client = llm_client
    
    def extract_from_conversation(
        self,
        messages: list[dict],
        use_llm: bool = True,
    ) -> list[ExtractedFact]:
        """
        Extract important facts from a conversation.
        
        Args:
            messages: List of message dicts with 'role', 'content', and optionally 'id'
            use_llm: Whether to use LLM for extraction (falls back to heuristics if False)
            
        Returns:
            List of ExtractedFact objects
        """
        if not messages:
            return []
        
        # Collect message IDs for source linking
        message_ids = [msg.get("id", str(uuid.uuid4())) for msg in messages]
        
        # Format conversation for the prompt
        conversation_text = self._format_conversation(messages)
        
        if use_llm and self.llm_client:
            try:
                return self._extract_with_llm(conversation_text, message_ids)
            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to heuristics: {e}")
                return self._extract_with_heuristics(messages, message_ids)
        else:
            return self._extract_with_heuristics(messages, message_ids)
    
    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages into a conversation string for the prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _extract_with_llm(
        self,
        conversation_text: str,
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """Use LLM to extract facts from the conversation."""
        from ...models.message import Message
        
        prompt = get_fact_extraction_prompt(conversation_text)
        
        # Use stream=False to avoid any output printing
        response = self.llm_client.query(
            messages=[Message(role="user", content=prompt)],
            plaintext_output=True,
            stream=False,
        )
        
        # Parse JSON response
        facts = self._parse_extraction_response(response, message_ids)
        return facts
    
    def _parse_extraction_response(
        self,
        response: str,
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """Parse the LLM response into ExtractedFact objects."""
        facts = []
        
        # Try to extract JSON from the response (look for code blocks first)
        # Match ```json ... ``` or ``` ... ``` blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Fall back to finding raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.debug("No JSON found in extraction response")
                return facts
            json_str = json_match.group()
        
        try:
            # Try to fix common LLM JSON issues
            # 1. Replace single quotes with double quotes (but not in strings)
            # 2. Add quotes around unquoted property names
            fixed_json = self._fix_json(json_str)
            data = json.loads(fixed_json)
            raw_facts = data.get("facts", [])
            
            for raw_fact in raw_facts:
                content = raw_fact.get("content", "").strip()
                if not content:
                    continue
                
                memory_type = raw_fact.get("memory_type", "misc")
                if memory_type not in MEMORY_TYPES:
                    memory_type = "misc"
                
                importance = float(raw_fact.get("importance", 0.5))
                importance = max(0.0, min(1.0, importance))  # Clamp to 0-1
                
                facts.append(ExtractedFact(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    source_message_ids=message_ids.copy(),
                ))
                
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse extraction JSON: {e}")
        except Exception as e:
            logger.debug(f"Error processing extraction response: {e}")
        
        return facts
    
    def _fix_json(self, json_str: str) -> str:
        """Attempt to fix common JSON issues from LLM output."""
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Try to add quotes around unquoted property names
        # Match word: at the start of a line or after { or ,
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        
        return fixed
    
    def _extract_with_heuristics(
        self,
        messages: list[dict],
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """
        Extract facts using simple heuristics (fallback when LLM unavailable).
        
        This is a simplified extraction that looks for certain patterns
        indicating important information.
        """
        facts = []
        
        # Patterns that might indicate important information
        patterns = {
            "preference": [
                r"(?:i|I)\s+(?:prefer|like|love|hate|dislike)\s+(.+)",
                r"(?:my|I)\s+favorite\s+(.+)",
            ],
            "fact": [
                r"(?:i|I)\s+(?:am|'m)\s+(?:a|an)?\s*(.+)",
                r"(?:my|I)\s+(?:name|age|location)\s+(?:is)?\s*(.+)",
            ],
            "professional": [
                r"(?:i|I)\s+work\s+(?:at|for|on|as)\s+(.+)",
                r"(?:my|I)\s+(?:job|career|profession)\s+(?:is)?\s*(.+)",
            ],
            "health": [
                r"(?:i|I)\s+(?:have|had|suffer from)\s+(.+?)(?:\.|$)",
                r"(?:my|I)\s+(?:doctor|medication|condition)\s+(.+)",
            ],
        }
        
        for msg in messages:
            if msg.get("role") != "user":
                continue
            
            content = msg.get("content", "")
            
            for memory_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        
                        fact_content = f"User {match.strip()}"
                        importance = estimate_importance(content)
                        
                        facts.append(ExtractedFact(
                            content=fact_content,
                            memory_type=memory_type,
                            importance=importance,
                            source_message_ids=message_ids.copy(),
                        ))
        
        return facts
    
    def determine_memory_actions(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """
        Determine what actions to take for new facts against existing memories.
        
        Args:
            new_facts: Facts extracted from recent conversation
            existing_memories: Current memories from the database
            
        Returns:
            List of MemoryAction objects describing what to do
        """
        if not new_facts:
            return []
        
        if not existing_memories:
            # All facts are new, just add them
            return [
                MemoryAction(action="ADD", fact=fact, reason="New memory")
                for fact in new_facts
            ]
        
        if self.llm_client:
            try:
                return self._determine_actions_with_llm(new_facts, existing_memories)
            except Exception as e:
                logger.warning(f"LLM action determination failed: {e}")
                return self._determine_actions_with_heuristics(new_facts, existing_memories)
        else:
            return self._determine_actions_with_heuristics(new_facts, existing_memories)
    
    def _determine_actions_with_llm(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """Use LLM to determine memory actions."""
        from ...models.message import Message
        
        # Format memories and facts for the prompt
        existing_json = json.dumps([
            {"id": m.get("id"), "content": m.get("content"), "importance": m.get("importance", 0.5)}
            for m in existing_memories
        ], indent=2)
        
        new_facts_json = json.dumps([f.to_dict() for f in new_facts], indent=2)
        
        prompt = get_memory_update_prompt(existing_json, new_facts_json)
        
        # Use stream=False to avoid any output printing
        response = self.llm_client.query(
            messages=[Message(role="user", content=prompt)],
            plaintext_output=True,
            stream=False,
        )
        
        return self._parse_action_response(response, new_facts)
    
    def _parse_action_response(
        self,
        response: str,
        new_facts: list[ExtractedFact],
    ) -> list[MemoryAction]:
        """Parse the LLM response into MemoryAction objects."""
        actions = []
        
        # Try to extract JSON from the response (look for code blocks first)
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.debug("No JSON found in action response")
                # Default to adding all facts
                return [MemoryAction(action="ADD", fact=f) for f in new_facts]
            json_str = json_match.group()
        
        try:
            fixed_json = self._fix_json(json_str)
            data = json.loads(fixed_json)
            raw_actions = data.get("actions", [])
            
            for raw_action in raw_actions:
                action_type = raw_action.get("action", "NONE").upper()
                
                if action_type == "ADD":
                    fact_data = raw_action.get("fact", {})
                    fact = ExtractedFact(
                        content=fact_data.get("content", ""),
                        memory_type=fact_data.get("memory_type", "misc"),
                        importance=float(fact_data.get("importance", 0.5)),
                        source_message_ids=fact_data.get("source_message_ids", []),
                    )
                    actions.append(MemoryAction(
                        action="ADD",
                        fact=fact,
                        reason=raw_action.get("reason", ""),
                    ))
                    
                elif action_type == "UPDATE":
                    fact_data = raw_action.get("fact", {})
                    fact = ExtractedFact(
                        content=fact_data.get("content", ""),
                        memory_type=fact_data.get("memory_type", "misc"),
                        importance=float(fact_data.get("importance", 0.5)),
                        source_message_ids=fact_data.get("source_message_ids", []),
                    )
                    actions.append(MemoryAction(
                        action="UPDATE",
                        fact=fact,
                        target_memory_id=raw_action.get("target_memory_id"),
                        reason=raw_action.get("reason", ""),
                    ))
                    
                elif action_type == "DELETE":
                    actions.append(MemoryAction(
                        action="DELETE",
                        target_memory_id=raw_action.get("target_memory_id"),
                        reason=raw_action.get("reason", ""),
                    ))
                # NONE actions are ignored
                
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse action JSON: {e}")
            return [MemoryAction(action="ADD", fact=f) for f in new_facts]
        
        return actions
    
    def _determine_actions_with_heuristics(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """
        Simple heuristic-based action determination.
        
        Checks for exact and near-duplicate content to avoid adding duplicates.
        """
        actions = []
        
        existing_contents = {
            m.get("id"): m.get("content", "").lower()
            for m in existing_memories
        }
        
        for fact in new_facts:
            fact_lower = fact.content.lower()
            is_duplicate = False
            
            for mem_id, mem_content in existing_contents.items():
                # Check for exact match or high similarity
                if fact_lower == mem_content:
                    is_duplicate = True
                    break
                
                # Simple similarity check (could use proper similarity metric)
                if self._simple_similarity(fact_lower, mem_content) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                actions.append(MemoryAction(
                    action="ADD",
                    fact=fact,
                    reason="New information",
                ))
        
        return actions
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

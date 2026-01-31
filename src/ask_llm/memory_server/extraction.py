"""Extraction service wrapper for MCP memory server.

Wraps the existing MemoryExtractionService for use with MCP tools.
Handles LLM client initialization and async interface.
Also extracts profile attributes from high-importance facts.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ask_llm.memory.extraction.service import MemoryExtractionService, ExtractedFact
from ask_llm.utils.config import Config

logger = logging.getLogger(__name__)

# Lazy-loaded extraction service
_service: MemoryExtractionService | None = None
_llm_client: Any = None


# Minimum importance to consider for profile attributes (fallback)
PROFILE_ATTRIBUTE_MIN_IMPORTANCE = 0.6


def extract_profile_attributes_from_fact(
    fact: ExtractedFact | dict,
    user_id: str,  # Required - must be passed explicitly
    config: Config | None = None,
) -> bool:
    """Extract profile attributes from a fact if the LLM identified it as one.
    
    The LLM extraction prompt now includes profile_attribute classification.
    This function reads that classification and stores the attribute.
    
    Args:
        fact: ExtractedFact object or dict with content, tags, importance, profile_attribute
        user_id: The user to associate the attribute with
        config: Config object for database connection
        
    Returns:
        True if an attribute was created, False otherwise
    """
    config = config or Config()
    
    # Extract fact properties
    if isinstance(fact, dict):
        content = fact.get("content", "")
        importance = fact.get("importance", 0.5)
        profile_attr = fact.get("profile_attribute")
    else:
        content = fact.content
        importance = fact.importance
        profile_attr = getattr(fact, "profile_attribute", None)
    
    # Check if profile extraction is enabled
    if not getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True):
        logger.debug("[Profile] Profile extraction disabled")
        return False
    
    # Check if LLM identified this as a profile attribute
    if not profile_attr or not isinstance(profile_attr, dict):
        logger.debug(f"[Profile] No profile_attribute from LLM for: '{content[:50]}...'")
        return False
    
    category = profile_attr.get("category")
    key = profile_attr.get("key")
    
    if not category or not key:
        logger.debug(f"[Profile] Invalid profile_attribute format: {profile_attr}")
        return False
    
    logger.info(f"[Profile] LLM identified attribute: {category}.{key} = '{content[:50]}...'")
    
    # Store as profile attribute
    try:
        from ask_llm.profiles import ProfileManager, EntityType

        manager = ProfileManager(config)
        attr = manager.set_attribute(
            entity_type=EntityType.USER,
            entity_id=user_id,
            category=category,
            key=key,
            value=content,  # Store full content as value
            confidence=importance,  # Map importance to confidence
            source="extracted",
        )

        # Determine if this was a create or update based on timestamps
        is_new = attr.created_at == attr.updated_at
        action = "Created" if is_new else "Updated"
        logger.info(f"[Profile] ✓ {action} attribute: {category}.{key} for user {user_id}")
        return is_new  # Return True only if newly created
        
    except Exception as e:
        logger.exception(f"[Profile] Failed to create profile attribute: {e}")
        return False


def _get_extraction_client(config: Config) -> Any:
    """Get or create an LLM client for extraction.
    
    Prefers a lightweight model for extraction to avoid loading
    large models just for fact extraction.
    """
    global _llm_client
    
    if _llm_client is not None:
        return _llm_client
    
    # Check if there's a configured extraction model
    extraction_model = config.EXTRACTION_MODEL
    
    if not extraction_model:
        # Try to find a suitable model (prefer smaller GGUF or OpenAI)
        models = config.defined_models.get("models", {})
        
        # Priority: extraction-specific > small GGUF > any OpenAI > any available
        for alias, info in models.items():
            if "extract" in alias.lower() or "small" in alias.lower():
                extraction_model = alias
                break
        
        if not extraction_model:
            # Fall back to any OpenAI model (fast, no VRAM)
            for alias, info in models.items():
                if info.get("type") == "openai":
                    extraction_model = alias
                    break
        
        if not extraction_model:
            # Fall back to first available
            if models:
                extraction_model = next(iter(models.keys()))
    
    if not extraction_model:
        logger.warning("No model available for extraction, using heuristics only")
        return None
    
    # Initialize the client
    try:
        from ask_llm.core import AskLLM
        
        # Create a minimal AskLLM instance for extraction
        # This will load the model if needed
        ask_llm = AskLLM(
            resolved_model_alias=extraction_model,
            bot_id="nova",  # Use nova for extraction context
            config=config,
        )
        _llm_client = ask_llm.client
        logger.debug(f"Initialized extraction client with model: {extraction_model}")
        return _llm_client
        
    except Exception as e:
        logger.warning(f"Failed to initialize extraction client: {e}")
        return None


def get_extraction_service(config: Config | None = None) -> MemoryExtractionService:
    """Get the singleton extraction service."""
    global _service
    
    if _service is None:
        config = config or Config()
        llm_client = _get_extraction_client(config)
        _service = MemoryExtractionService(llm_client=llm_client)
    
    return _service


async def extract_facts_from_messages(
    messages: list[dict],
    config: Config | None = None,
    use_llm: bool = True,
    user_id: str = "",  # Required - must be passed explicitly
    extract_profile_attributes: bool | None = None,
) -> list[dict]:
    """Extract facts from conversation messages.
    
    Args:
        messages: List of message dicts with role/content.
        config: Config object (uses default if not provided).
        use_llm: Whether to use LLM extraction (falls back to heuristics if False).
        user_id: User ID for profile attribute extraction (required).
        extract_profile_attributes: Whether to also create profile attributes from facts.
        
    Returns:
        List of extracted fact dicts.
    """
    if not user_id:
        raise ValueError("user_id is required for extract_facts_from_messages")
    config = config or Config()
    
    # Check if extraction is enabled
    if not config.MEMORY_EXTRACTION_ENABLED:
        logger.debug("Memory extraction is disabled")
        return []
    
    service = get_extraction_service(config)
    
    try:
        facts = service.extract_from_conversation(
            messages=messages,
            use_llm=use_llm and service.llm_client is not None,
        )
        
        # Filter by minimum importance
        min_importance = config.MEMORY_EXTRACTION_MIN_IMPORTANCE
        facts = [f for f in facts if f.importance >= min_importance]
        
        # Also extract profile attributes from high-importance facts
        if extract_profile_attributes is None:
            extract_profile_attributes = getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

        if extract_profile_attributes:
            for fact in facts:
                try:
                    created = extract_profile_attributes_from_fact(
                        fact=fact,
                        user_id=user_id,
                        config=config,
                    )
                    if created:
                        logger.debug(f"Created profile attribute from: {fact.content[:50]}...")
                except Exception as e:
                    logger.warning(f"Profile attribute extraction failed: {e}")
        
        return [f.to_dict() for f in facts]
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return []

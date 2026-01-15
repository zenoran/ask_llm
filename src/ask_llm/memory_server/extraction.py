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


# Tags that indicate the fact should also be stored as a profile attribute
PROFILE_ATTRIBUTE_TAGS = {
    "preference": "preference",
    "fact": "fact", 
    "professional": "fact",
    "health": "fact",
    "relationship": "fact",
}

# Minimum importance to consider for profile attributes (fallback)
PROFILE_ATTRIBUTE_MIN_IMPORTANCE = 0.6


def extract_profile_attributes_from_fact(
    fact: ExtractedFact | dict,
    user_id: str = "default",
    config: Config | None = None,
) -> bool:
    """Extract profile attributes from a fact if applicable.
    
    Checks if the fact represents something that should be stored as a
    user profile attribute (in addition to long-term memory).
    
    Args:
        fact: ExtractedFact object or dict with content, tags, importance
        user_id: The user to associate the attribute with
        config: Config object for database connection
        
    Returns:
        True if an attribute was created, False otherwise
    """
    config = config or Config()
    
    # Extract fact properties
    if isinstance(fact, dict):
        content = fact.get("content", "")
        tags = fact.get("tags", [])
        importance = fact.get("importance", 0.5)
    else:
        content = fact.content
        tags = fact.tags
        importance = fact.importance
    
    logger.info(f"[Profile] Checking fact: '{content[:60]}...' tags={tags} importance={importance:.2f}")
    
    # Check if profile extraction is enabled
    if not getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True):
        logger.debug("[Profile] Profile extraction disabled")
        return False

    # Check if importance meets threshold
    min_importance = getattr(config, "MEMORY_PROFILE_ATTRIBUTE_MIN_IMPORTANCE", PROFILE_ATTRIBUTE_MIN_IMPORTANCE)
    if importance < min_importance:
        logger.debug(f"[Profile] Importance {importance:.2f} < threshold {min_importance:.2f}")
        return False
    
    # Check if any tags indicate a profile attribute
    category = None
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in PROFILE_ATTRIBUTE_TAGS:
            category = PROFILE_ATTRIBUTE_TAGS[tag_lower]
            break
    
    if not category:
        logger.debug(f"[Profile] No matching category for tags: {tags}")
        return False
    
    # Try to extract a key from the content
    key = _extract_attribute_key(content)
    if not key:
        logger.debug(f"[Profile] Could not extract key from content")
        return False
    
    logger.info(f"[Profile] Extracted key '{key}' for category '{category}'")
    
    # Store as profile attribute
    try:
        from ask_llm.profiles import ProfileManager, EntityType
        
        manager = ProfileManager(config)
        manager.set_attribute(
            entity_type=EntityType.USER,
            entity_id=user_id,
            category=category,
            key=key,
            value=content,  # Store full content as value
            confidence=importance,  # Map importance to confidence
            source="extracted",
        )
        
        logger.info(f"[Profile] ✓ Created attribute: {category}.{key} for user {user_id}")
        return True
        
    except Exception as e:
        logger.exception(f"[Profile] Failed to create profile attribute: {e}")
        return False


def _extract_attribute_key(content: str) -> str | None:
    """Extract a meaningful key from fact content.
    
    Attempts to identify what the fact is about and create a reasonable key.
    """
    content_lower = content.lower()
    
    logger.debug(f"Extracting key from: {content_lower[:100]}...")
    
    # Pattern: "User is <age> years old" -> age
    if re.search(r"user (?:is )?(\d+)(?: years?)? old", content_lower):
        return "age"
    
    # Pattern: "User has <N> dogs/cats/pets" -> pets
    if re.search(r"user has (?:\d+ )?(?:dogs?|cats?|pets?|animals?)", content_lower):
        return "pets"
    
    # Pattern: "User's dog/cat/pet is named" or "User has a dog named" -> pet_names
    if re.search(r"(?:dog|cat|pet)(?:s)? (?:is |are )?named", content_lower):
        return "pet_names"
    
    # Pattern: "User is a <something>" -> occupation, role, etc.
    if match := re.search(r"user (?:is|works as) (?:a |an )?([^,\.]+)", content_lower):
        term = match.group(1).strip()
        if any(word in term for word in ["engineer", "developer", "manager", "programmer", "designer"]):
            return "occupation"
        if "student" in term:
            return "occupation"
        if any(word in term for word in ["years old", "year old"]):
            return "age"
        return "role"
    
    # Pattern: "User lives in <place>" -> location
    if re.search(r"user (?:lives|resides|is located|is based|is from) in", content_lower):
        return "location"
    
    # Pattern: "User prefers <something>" -> specific preference
    if match := re.search(r"user (?:prefers?|likes?|loves?|enjoys?) ([^,\.]+)", content_lower):
        term = match.group(1).strip()
        if any(word in term for word in ["python", "java", "code", "programming", "language"]):
            return "programming_preference"
        if any(word in term for word in ["dark", "light", "theme", "mode"]):
            return "ui_preference"
        if any(word in term for word in ["concise", "detailed", "verbose", "brief"]):
            return "communication_style"
        if any(word in term for word in ["dogs", "cats", "pets", "animals"]):
            return "pet_preference"
        return "likes"
    
    # Pattern: "User's name is <name>" -> name
    if re.search(r"user'?s? name is", content_lower):
        return "name"
    
    # Pattern: "User has <condition>" -> health
    if re.search(r"user has (?:chronic |a )?", content_lower):
        if any(word in content_lower for word in ["pain", "condition", "disease", "allergy", "diabetes"]):
            return "health_condition"
        # Could be pets or other things
        if any(word in content_lower for word in ["dog", "cat", "pet", "child", "kid"]):
            return "family"
    
    # Pattern: "User works on/with <something>" -> work context
    if re.search(r"user works (?:on|with)", content_lower):
        return "work_focus"
    
    # Pattern: "User uses <something>" -> tools/tech
    if re.search(r"user uses", content_lower):
        return "tools"
    
    # Pattern: "User is married/single/divorced" -> relationship_status
    if re.search(r"user is (?:married|single|divorced|engaged|widowed)", content_lower):
        return "relationship_status"
    
    # Pattern: "User has <N> children/kids" -> family
    if re.search(r"user has (?:\d+ )?(?:children|kids|child)", content_lower):
        return "children"
    
    # Fallback: use first significant noun phrase after "User"
    if match := re.search(r"user'?s? ([a-z_]+)", content_lower):
        key = match.group(1).replace(" ", "_")
        # Skip generic words
        if key not in ["is", "has", "the", "a", "an", "was", "will", "would", "can", "could"]:
            return key
    
    # Last resort: if content has "User" and some noun, try to extract it
    if "user" in content_lower:
        # Look for patterns like "User <verb> <noun>"
        if match := re.search(r"user (?:\w+ ){1,2}(\w+)", content_lower):
            word = match.group(1)
            if len(word) > 3 and word not in ["that", "this", "with", "from", "about"]:
                return word
    
    logger.debug(f"Could not extract key from: {content_lower[:60]}...")
    return None


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
    user_id: str = "default",
    extract_profile_attributes: bool | None = None,
) -> list[dict]:
    """Extract facts from conversation messages.
    
    Args:
        messages: List of message dicts with role/content.
        config: Config object (uses default if not provided).
        use_llm: Whether to use LLM extraction (falls back to heuristics if False).
        user_id: User ID for profile attribute extraction.
        extract_profile_attributes: Whether to also create profile attributes from facts.
        
    Returns:
        List of extracted fact dicts.
    """
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

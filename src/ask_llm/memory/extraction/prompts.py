"""
Memory extraction prompts adapted from Mem0's approach.
These prompts are used to distill important facts from conversations
and handle memory updates/conflicts.
"""

# Memory types for categorization
MEMORY_TYPES = [
    "preference",      # User preferences and likes/dislikes
    "fact",           # Factual information about the user
    "event",          # Past or planned events
    "plan",           # Goals, intentions, future plans
    "professional",   # Work, career, skills
    "health",         # Health-related information
    "relationship",   # Information about relationships/people
    "misc",           # Anything that doesn't fit above
]

# Note: Use get_fact_extraction_prompt(conversation) to get the formatted prompt
FACT_EXTRACTION_PROMPT_TEMPLATE = """You are a memory extraction assistant. Your task is to extract important facts, preferences, and information from conversations that would be valuable to remember for future interactions.

Analyze the conversation and extract discrete facts. Focus on:
- User preferences (likes, dislikes, habits)
- Personal details (name, location, occupation, relationships)
- Plans and goals (upcoming events, aspirations)
- Professional information (job, skills, projects)
- Health information (conditions, medications, fitness)
- Important events (past experiences, milestones)

Rules:
1. Extract ONLY information explicitly stated or strongly implied
2. Each fact should be a single, atomic piece of information
3. Write facts in third person (e.g., "User prefers dark mode" not "You prefer dark mode")
4. Be specific and include context when relevant
5. Do NOT infer or assume information not present
6. Do NOT extract trivial or temporary information (e.g., "User said hello")
7. If no important facts are present, return an empty list

For each fact, also assign:
- memory_type: One of """ + str(MEMORY_TYPES) + """
- importance: Float 0.0-1.0 where:
  - 0.0-0.3: Nice to know, low priority
  - 0.4-0.6: Moderately important, useful context
  - 0.7-0.9: Important, should be remembered
  - 1.0: Critical, essential information

Examples:

Input conversation:
User: I'm a software engineer working at Google
Assistant: That's great! What kind of projects do you work on?
User: Mostly backend stuff with Python and Go

Output:
```json
{{
  "facts": [
    {{
      "content": "User is a software engineer working at Google",
      "memory_type": "professional",
      "importance": 0.8
    }},
    {{
      "content": "User works primarily on backend projects",
      "memory_type": "professional", 
      "importance": 0.6
    }},
    {{
      "content": "User uses Python and Go programming languages",
      "memory_type": "professional",
      "importance": 0.7
    }}
  ]
}}
```

Input conversation:
User: Can you help me write a for loop?
Assistant: Sure! Here's an example...

Output:
```json
{{
  "facts": []
}}
```

Input conversation:
User: I have a doctor's appointment next Tuesday for my back pain
Assistant: I hope it goes well! Is the back pain something new?
User: No, I've had chronic lower back issues for about 2 years now

Output:
```json
{{
  "facts": [
    {{
      "content": "User has a doctor's appointment scheduled for next Tuesday",
      "memory_type": "event",
      "importance": 0.5
    }},
    {{
      "content": "User has chronic lower back pain that has persisted for approximately 2 years",
      "memory_type": "health",
      "importance": 0.8
    }}
  ]
}}
```

Now extract facts from the following conversation:

{conversation}

Output only valid JSON:"""


def get_fact_extraction_prompt(conversation: str) -> str:
    """Get the fact extraction prompt with the conversation filled in."""
    return FACT_EXTRACTION_PROMPT_TEMPLATE + "\n\n" + conversation + "\n\nOutput only valid JSON:"


MEMORY_UPDATE_PROMPT_TEMPLATE = """You are a memory management assistant. Your task is to compare newly extracted facts against existing memories and determine the appropriate action for each new fact.

Existing memories:
{existing_memories}

New facts to process:
{new_facts}

For each new fact, determine ONE of these actions:
- ADD: The fact is new information not covered by existing memories
- UPDATE: The fact updates/modifies an existing memory (include which memory ID to update)
- DELETE: The fact contradicts an existing memory, making it obsolete (include which memory ID to delete)
- NONE: The fact is already captured by existing memories, no action needed

Rules:
1. Be conservative with UPDATE - only use when the new fact clearly supersedes old info
2. Use DELETE when information is explicitly contradicted (e.g., "moved from X to Y" deletes "lives in X")
3. Similar but not identical facts should both be kept (ADD, not UPDATE)
4. When updating, preserve the higher importance score unless the update is more significant

Examples:

Existing memories:
[
  {{"id": "mem_001", "content": "User lives in San Francisco", "importance": 0.7}},
  {{"id": "mem_002", "content": "User is a software engineer", "importance": 0.8}}
]

New facts:
[
  {{"content": "User recently moved to Seattle", "memory_type": "fact", "importance": 0.8}},
  {{"content": "User works on machine learning projects", "memory_type": "professional", "importance": 0.7}}
]

Output:
```json
{{
  "actions": [
    {{
      "action": "DELETE",
      "target_memory_id": "mem_001",
      "reason": "User moved from San Francisco to Seattle"
    }},
    {{
      "action": "ADD",
      "fact": {{
        "content": "User lives in Seattle",
        "memory_type": "fact",
        "importance": 0.8
      }},
      "reason": "New location information"
    }},
    {{
      "action": "ADD",
      "fact": {{
        "content": "User works on machine learning projects",
        "memory_type": "professional",
        "importance": 0.7
      }},
      "reason": "Adds specific detail about work, complements existing engineer memory"
    }}
  ]
}}
```

Now process the new facts against the existing memories.

Output only valid JSON:"""


def get_memory_update_prompt(existing_memories: str, new_facts: str) -> str:
    """Get the memory update prompt with memories and facts filled in."""
    return MEMORY_UPDATE_PROMPT_TEMPLATE.format(
        existing_memories=existing_memories,
        new_facts=new_facts,
    )


# Legacy aliases for backwards compatibility
FACT_EXTRACTION_PROMPT = FACT_EXTRACTION_PROMPT_TEMPLATE
MEMORY_UPDATE_PROMPT = MEMORY_UPDATE_PROMPT_TEMPLATE


IMPORTANCE_KEYWORDS = {
    # High importance (0.8-1.0)
    "high": [
        "always", "never", "hate", "love", "allergic", "diagnosed",
        "married", "divorced", "died", "born", "retired", "hired",
        "promoted", "fired", "moved", "bought", "sold", "critical",
        "emergency", "chronic", "permanent",
    ],
    # Medium importance (0.5-0.7)
    "medium": [
        "prefer", "usually", "often", "work", "live", "study",
        "hobby", "enjoy", "interested", "plan", "goal", "want",
        "need", "birthday", "anniversary", "meeting", "appointment",
    ],
    # Low importance (0.2-0.4)
    "low": [
        "sometimes", "occasionally", "might", "maybe", "tried",
        "thinking", "considering", "heard", "saw", "read",
    ],
}


def estimate_importance(text: str) -> float:
    """
    Estimate importance score based on keyword heuristics.
    This is a fallback when LLM scoring is not available.
    """
    text_lower = text.lower()
    
    # Check for high importance keywords
    for keyword in IMPORTANCE_KEYWORDS["high"]:
        if keyword in text_lower:
            return 0.85
    
    # Check for medium importance keywords
    for keyword in IMPORTANCE_KEYWORDS["medium"]:
        if keyword in text_lower:
            return 0.6
    
    # Check for low importance keywords
    for keyword in IMPORTANCE_KEYWORDS["low"]:
        if keyword in text_lower:
            return 0.35
    
    # Default to medium-low importance
    return 0.5

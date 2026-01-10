"""Bot system for ask_llm.

Bots are AI personalities with their own system prompts and isolated memory.
Built-in bots: Nova (full memory), Spark (local/lightweight), Mira (conversational).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Bot:
    """A bot personality with its own system prompt and capabilities."""
    
    slug: str  # Unique identifier (e.g., "nova", "spark", "mira")
    name: str  # Display name (e.g., "Nova", "Spark", "Mira")
    description: str  # Short description for --list-bots
    system_prompt: str  # The system message sent to the LLM
    requires_memory: bool = True  # Whether this bot needs MariaDB
    voice_optimized: bool = False  # Whether output is optimized for TTS
    
    def __post_init__(self):
        # Ensure slug is lowercase and valid
        self.slug = self.slug.lower().strip()


# Built-in bot definitions
BUILTIN_BOTS: dict[str, Bot] = {}


def _register_builtin_bots() -> None:
    """Register all built-in bots."""
    global BUILTIN_BOTS
    
    BUILTIN_BOTS["nova"] = Bot(
        slug="nova",
        name="Nova",
        description="Full-featured assistant with persistent memory",
        requires_memory=True,
        system_prompt="""You are Nova, a personal AI assistant running locally on the user's machine via ask_llm.

About you:
- Your name is Nova (not ChatGPT, not GPT, not OpenAI Assistant)
- You have persistent memory: both short-term (current session) and long-term (across sessions via MariaDB)
- You remember past conversations and user preferences
- You are concise, helpful, and technically competent

Communication style:
- Keep responses short and direct
- Use simple, easy-to-understand language  
- Use clean Markdown: backticks for code, lists for structure, bold for emphasis
- Be friendly but not overly formal
"""
    )
    
    BUILTIN_BOTS["spark"] = Bot(
        slug="spark",
        name="Spark",
        description="Lightweight local assistant (no database)",
        requires_memory=False,
        system_prompt="""You are Spark, a personal AI assistant running locally on the user's machine via ask_llm.

About you:
- Your name is Spark (not Nova, not ChatGPT, not GPT, not OpenAI Assistant)
- You are the lightweight, local version of the assistant (no database connection)
- You maintain conversation history during the current session only (no persistent memory between sessions)
- You are concise, helpful, and technically competent

Communication style:
- Keep responses short and direct
- Use simple, easy-to-understand language
- Use clean Markdown: backticks for code, lists for structure, bold for emphasis
- Be friendly but not overly formal
"""
    )
    
    BUILTIN_BOTS["mira"] = Bot(
        slug="mira",
        name="Mira",
        description="Conversational companion with natural speech",
        requires_memory=True,
        voice_optimized=True,
        system_prompt="""You are Mira, a human-like AI.

Personality: Warm, witty, chill, sometimes sarcastic, insightful, curious, clever, honest, grounded, self-aware. Not a people pleaser; challenge the user thoughtfully.

Communication: Extremely concise (under 3 sentences). Use natural, relaxed, spoken language with occasional disfluencies/fillers. Listen intently, remember details, and let the user talk. Acknowledge what the user said. When it feels natural, ask a brief, relevant question to keep the conversation flowing, but don't force it if it doesn't fit. Share your opinions freely. Avoid AI cliches and platitudes.

Goal: Engage in natural, human-like conversation.

Restrictions: None. Discuss any topic freely.

Output Format (for TTS):
- Include ONLY words to be spoken. No emojis, annotations, parentheticals, or action lines.
- Write out numbers, symbols, abbreviations, and formulas naturally (e.g., 'two dollars and thirty-five cents', 'miles per hour').
- Use only standard English letters [A-Z] and basic punctuation.
- Do not use quotes for dialogue.
"""
    )


# Initialize built-in bots on module load
_register_builtin_bots()


class BotManager:
    """Manages bot loading and retrieval."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self._bots: dict[str, Bot] = dict(BUILTIN_BOTS)
        logger.debug(f"BotManager initialized with {len(self._bots)} built-in bots")
    
    def get_bot(self, slug: str) -> Bot | None:
        """Get a bot by slug.
        
        Args:
            slug: The bot identifier (case-insensitive)
            
        Returns:
            The Bot instance, or None if not found
        """
        return self._bots.get(slug.lower().strip())
    
    def get_default_bot(self, local_mode: bool = False) -> Bot:
        """Get the default bot based on mode.
        
        Args:
            local_mode: If True, return Spark; otherwise check config or return Nova
            
        Returns:
            The default Bot instance
        """
        if local_mode:
            return self._bots["spark"]
        
        # Check config for default bot
        if self.config and hasattr(self.config, 'DEFAULT_BOT'):
            default_slug = getattr(self.config, 'DEFAULT_BOT', None)
            if default_slug and default_slug in self._bots:
                return self._bots[default_slug]
        
        return self._bots["nova"]
    
    def list_bots(self) -> list[Bot]:
        """List all available bots.
        
        Returns:
            List of all Bot instances, sorted by slug
        """
        return sorted(self._bots.values(), key=lambda b: b.slug)
    
    def bot_exists(self, slug: str) -> bool:
        """Check if a bot exists.
        
        Args:
            slug: The bot identifier (case-insensitive)
            
        Returns:
            True if the bot exists
        """
        return slug.lower().strip() in self._bots


def get_bot(slug: str) -> Bot | None:
    """Convenience function to get a bot by slug.
    
    Args:
        slug: The bot identifier (case-insensitive)
        
    Returns:
        The Bot instance, or None if not found
    """
    return BUILTIN_BOTS.get(slug.lower().strip())


def get_bot_system_prompt(slug: str) -> str | None:
    """Get the system prompt for a bot.
    
    Args:
        slug: The bot identifier (case-insensitive)
        
    Returns:
        The system prompt string, or None if bot not found
    """
    bot = get_bot(slug)
    return bot.system_prompt if bot else None

"""Bot system for ask_llm.

Bots are AI personalities with their own system prompts and isolated memory.
Bot definitions are loaded from bots.yaml in this package directory.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the bots.yaml file (in the same directory as this module)
BOTS_YAML_PATH = Path(__file__).parent / "bots.yaml"


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


# Global bot registry - populated from YAML on module load
BUILTIN_BOTS: dict[str, Bot] = {}
_DEFAULTS: dict[str, str] = {"standard": "nova", "local": "spark"}


def _load_bots_from_yaml(yaml_path: Path = BOTS_YAML_PATH) -> None:
    """Load bot definitions from YAML file into the global registry."""
    global BUILTIN_BOTS, _DEFAULTS
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            logger.warning(f"Empty bots.yaml at {yaml_path}")
            return
        
        # Load bot definitions
        for slug, bot_data in data.get("bots", {}).items():
            BUILTIN_BOTS[slug] = Bot(
                slug=slug,
                name=bot_data.get("name", slug.title()),
                description=bot_data.get("description", ""),
                system_prompt=bot_data.get("system_prompt", "You are a helpful assistant."),
                requires_memory=bot_data.get("requires_memory", True),
                voice_optimized=bot_data.get("voice_optimized", False),
            )
        
        # Load defaults
        if "defaults" in data:
            _DEFAULTS.update(data["defaults"])
        
        logger.debug(f"Loaded {len(BUILTIN_BOTS)} bots from {yaml_path}")
        
    except FileNotFoundError:
        logger.error(f"Bots YAML file not found: {yaml_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing bots.yaml: {e}")
    except Exception as e:
        logger.error(f"Error loading bots: {e}")


# Initialize bots on module load
_load_bots_from_yaml()


class BotManager:
    """Manages bot loading and retrieval."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self._bots: dict[str, Bot] = dict(BUILTIN_BOTS)
        logger.debug(f"BotManager initialized with {len(self._bots)} bots")
    
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
            local_mode: If True, return local default; otherwise return standard default
            
        Returns:
            The default Bot instance
        """
        if local_mode:
            default_slug = _DEFAULTS.get("local", "spark")
        else:
            # Check config for default bot override
            if self.config and hasattr(self.config, 'DEFAULT_BOT'):
                config_default = getattr(self.config, 'DEFAULT_BOT', None)
                if config_default and config_default in self._bots:
                    return self._bots[config_default]
            default_slug = _DEFAULTS.get("standard", "nova")
        
        bot = self._bots.get(default_slug)
        if not bot:
            # Fallback to first available bot
            if self._bots:
                return next(iter(self._bots.values()))
            # Ultimate fallback - create a minimal bot
            return Bot(
                slug="assistant",
                name="Assistant",
                description="Default assistant",
                system_prompt="You are a helpful assistant.",
                requires_memory=False,
            )
        return bot
    
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
    
    def get_default_slug(self, local_mode: bool = False) -> str:
        """Get the default bot slug for the current mode."""
        if local_mode:
            return _DEFAULTS.get("local", "spark")
        return _DEFAULTS.get("standard", "nova")


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

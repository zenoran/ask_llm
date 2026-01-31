"""Nextcloud Talk bot configuration management.

Nextcloud bot configs are stored in the bots.yaml files under the 'nextcloud' key.
User config (~/.config/ask-llm/bots.yaml) takes priority over repo config.
This module uses the centralized bot config loader from ask_llm.bots.
"""

from dataclasses import dataclass
from typing import Optional

from ask_llm.bots import (
    get_all_raw_bot_data,
    save_user_bot_config,
    remove_user_bot_section,
)


@dataclass
class NextcloudBot:
    """Configuration for a single Nextcloud Talk bot."""
    ask_llm_bot: str  # The bot ID in bots.yaml
    nextcloud_bot_id: int
    secret: str
    conversation_token: str
    enabled: bool = True

    @property
    def name(self) -> str:
        """Bot name is the ask_llm bot ID."""
        return self.ask_llm_bot


class NextcloudBotConfig:
    """Manages Nextcloud bot configuration.
    
    Uses the centralized bot config loader which merges:
    - Repo bots.yaml (defaults)
    - User bots.yaml (~/.config/ask-llm/bots.yaml) (overrides)
    
    Writes ONLY to user bots.yaml to keep secrets out of repo.
    """

    def __init__(self, bots_yaml_path=None):
        """Initialize config.
        
        Args:
            bots_yaml_path: Ignored - kept for backwards compatibility.
                           Config loading is now centralized in ask_llm.bots.
        """
        # No separate loading needed - uses centralized loader
        pass

    def _get_nextcloud_bots(self) -> dict[str, NextcloudBot]:
        """Get all bots with nextcloud config from centralized loader."""
        bots = {}
        for slug, bot_data in get_all_raw_bot_data().items():
            nc_config = bot_data.get('nextcloud')
            if nc_config:
                bots[slug] = NextcloudBot(
                    ask_llm_bot=slug,
                    nextcloud_bot_id=nc_config.get('bot_id'),
                    secret=nc_config.get('secret', ''),
                    conversation_token=nc_config.get('conversation_token'),
                    enabled=nc_config.get('enabled', True),
                )
        return bots

    @property
    def bots(self) -> dict[str, NextcloudBot]:
        """Get all configured Nextcloud bots."""
        return self._get_nextcloud_bots()

    def load(self):
        """Reload config - triggers centralized reload via property access."""
        # No-op: _get_nextcloud_bots() calls get_all_raw_bot_data() which 
        # checks for config file changes automatically
        pass

    def _check_reload(self):
        """Check for config changes - handled by centralized loader."""
        # No-op: handled automatically by get_all_raw_bot_data()
        pass

    def save(self):
        """Save is handled per-bot via add_bot() now."""
        # No-op: saves happen individually via save_user_bot_config()
        pass

    def add_bot(
        self,
        ask_llm_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ) -> NextcloudBot:
        """Add Nextcloud config to a bot.
        
        Saves to ~/.config/ask-llm/bots.yaml (user config, not repo).
        """
        nc_data = {
            'bot_id': nextcloud_bot_id,
            'secret': secret,
            'conversation_token': conversation_token,
            'enabled': True,
        }
        save_user_bot_config(ask_llm_bot, 'nextcloud', nc_data)
        
        return NextcloudBot(
            ask_llm_bot=ask_llm_bot,
            nextcloud_bot_id=nextcloud_bot_id,
            secret=secret,
            conversation_token=conversation_token,
            enabled=True,
        )

    def remove_bot(self, ask_llm_bot: str) -> bool:
        """Remove Nextcloud config from a bot.
        
        Removes from user config only (doesn't touch repo).
        """
        return remove_user_bot_section(ask_llm_bot, 'nextcloud')

    def get_bot_by_conversation(self, token: str) -> Optional[NextcloudBot]:
        """Get bot config by conversation token."""
        for bot in self.bots.values():
            if bot.enabled and bot.conversation_token == token:
                return bot
        return None

    def get_bot(self, ask_llm_bot: str) -> Optional[NextcloudBot]:
        """Get bot config by ask_llm bot ID."""
        return self.bots.get(ask_llm_bot)

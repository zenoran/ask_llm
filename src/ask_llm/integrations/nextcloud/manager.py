"""Nextcloud bot manager singleton."""

from pathlib import Path
from typing import Optional
from .config import NextcloudBot, NextcloudBotConfig


class NextcloudBotManager:
    """Singleton manager for Nextcloud bots."""

    _instance: Optional['NextcloudBotManager'] = None

    def __init__(self, bots_yaml_path: Path):
        self.config = NextcloudBotConfig(bots_yaml_path)

    @classmethod
    def get_instance(cls, bots_yaml_path: Optional[Path] = None) -> 'NextcloudBotManager':
        """Get singleton instance."""
        if cls._instance is None:
            if bots_yaml_path is None:
                # Get default bots.yaml path
                from ...utils.config import Config
                config = Config()
                from ...bots import get_bots_yaml_path
                bots_yaml_path = get_bots_yaml_path()

            cls._instance = cls(bots_yaml_path)
        return cls._instance

    def list_bots(self) -> list[NextcloudBot]:
        """List all configured Nextcloud bots."""
        return list(self.config.bots.values())

    def add_bot(
        self,
        ask_llm_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ) -> NextcloudBot:
        """Add Nextcloud config to a bot."""
        return self.config.add_bot(
            ask_llm_bot=ask_llm_bot,
            nextcloud_bot_id=nextcloud_bot_id,
            secret=secret,
            conversation_token=conversation_token,
        )

    def remove_bot(self, ask_llm_bot: str) -> bool:
        """Remove Nextcloud config from a bot."""
        return self.config.remove_bot(ask_llm_bot)

    def get_bot_by_conversation(self, token: str) -> Optional[NextcloudBot]:
        """Get bot for a conversation token."""
        return self.config.get_bot_by_conversation(token)

    def get_bot(self, ask_llm_bot: str) -> Optional[NextcloudBot]:
        """Get bot config by ask_llm bot ID."""
        return self.config.get_bot(ask_llm_bot)


# Global accessor
def get_nextcloud_manager() -> NextcloudBotManager:
    """Get the global Nextcloud bot manager."""
    return NextcloudBotManager.get_instance()

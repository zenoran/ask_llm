"""Nextcloud Talk bot configuration management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


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
    """Manages Nextcloud bot configuration from bots.yaml."""

    def __init__(self, bots_yaml_path: Path):
        self.bots_yaml_path = bots_yaml_path
        self.bots: dict[str, NextcloudBot] = {}
        self._last_mtime: float = 0
        self.load()

    def _check_reload(self):
        """Reload config if file has changed."""
        if not self.bots_yaml_path.exists():
            return
        mtime = self.bots_yaml_path.stat().st_mtime
        if mtime > self._last_mtime:
            self.load()

    def load(self):
        """Load Nextcloud bot configs from bots.yaml."""
        if not self.bots_yaml_path.exists():
            return

        self._last_mtime = self.bots_yaml_path.stat().st_mtime
        self.bots.clear()

        with open(self.bots_yaml_path) as f:
            data = yaml.safe_load(f)

        # Look for nextcloud config in each bot
        for bot_id, bot_config in data.get('bots', {}).items():
            nc_config = bot_config.get('nextcloud')
            if nc_config:
                self.bots[bot_id] = NextcloudBot(
                    ask_llm_bot=bot_id,
                    nextcloud_bot_id=nc_config.get('bot_id'),
                    secret=nc_config.get('secret'),
                    conversation_token=nc_config.get('conversation_token'),
                    enabled=nc_config.get('enabled', True),
                )

    def save(self):
        """Save Nextcloud configs back to bots.yaml."""
        with open(self.bots_yaml_path) as f:
            data = yaml.safe_load(f)

        if 'bots' not in data:
            data['bots'] = {}

        # Update nextcloud section for each bot
        for bot_id, nc_bot in self.bots.items():
            if bot_id not in data['bots']:
                continue

            data['bots'][bot_id]['nextcloud'] = {
                'bot_id': nc_bot.nextcloud_bot_id,
                'secret': nc_bot.secret,
                'conversation_token': nc_bot.conversation_token,
                'enabled': nc_bot.enabled,
            }

        with open(self.bots_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add_bot(
        self,
        ask_llm_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ):
        """Add Nextcloud config to a bot."""
        bot = NextcloudBot(
            ask_llm_bot=ask_llm_bot,
            nextcloud_bot_id=nextcloud_bot_id,
            secret=secret,
            conversation_token=conversation_token,
            enabled=True,
        )
        self.bots[ask_llm_bot] = bot
        self.save()
        return bot

    def remove_bot(self, ask_llm_bot: str) -> bool:
        """Remove Nextcloud config from a bot."""
        if ask_llm_bot in self.bots:
            del self.bots[ask_llm_bot]
            # Remove from YAML
            with open(self.bots_yaml_path) as f:
                data = yaml.safe_load(f)
            if 'bots' in data and ask_llm_bot in data['bots']:
                if 'nextcloud' in data['bots'][ask_llm_bot]:
                    del data['bots'][ask_llm_bot]['nextcloud']
            with open(self.bots_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            return True
        return False

    def get_bot_by_conversation(self, token: str) -> Optional[NextcloudBot]:
        """Get bot config by conversation token."""
        self._check_reload()
        for bot in self.bots.values():
            if bot.enabled and bot.conversation_token == token:
                return bot
        return None

    def get_bot(self, ask_llm_bot: str) -> Optional[NextcloudBot]:
        """Get bot config by ask_llm bot ID."""
        self._check_reload()
        return self.bots.get(ask_llm_bot)

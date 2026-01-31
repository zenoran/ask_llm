"""Nextcloud Talk bot configuration management.

Nextcloud bot configs (including secrets) are stored in the USER config:
    ~/.config/ask-llm/bots.yaml

This is separate from the repo's bots.yaml which only contains bot definitions.
The user config is merged with the repo config at runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


def _get_user_bots_yaml_path() -> Path:
    """Get path to user's bots.yaml config file."""
    from ask_llm.utils.config import DOTENV_PATH
    # DOTENV_PATH is ~/.config/ask-llm/.env, so parent is the config dir
    return DOTENV_PATH.parent / "bots.yaml"


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
    
    Reads nextcloud configs from BOTH:
    - Repo bots.yaml (passed in constructor) - for legacy/fallback
    - User bots.yaml (~/.config/ask-llm/bots.yaml) - preferred location for secrets
    
    Writes ONLY to user bots.yaml to keep secrets out of repo.
    """

    def __init__(self, bots_yaml_path: Path):
        """Initialize config.
        
        Args:
            bots_yaml_path: Path to repo's bots.yaml (for reading only)
        """
        self.repo_bots_yaml_path = bots_yaml_path
        self.user_bots_yaml_path = _get_user_bots_yaml_path()
        self.bots: dict[str, NextcloudBot] = {}
        self._last_mtime: float = 0
        self._last_user_mtime: float = 0
        self.load()

    def _check_reload(self):
        """Reload config if either file has changed."""
        repo_mtime = self.repo_bots_yaml_path.stat().st_mtime if self.repo_bots_yaml_path.exists() else 0
        user_mtime = self.user_bots_yaml_path.stat().st_mtime if self.user_bots_yaml_path.exists() else 0
        
        if repo_mtime > self._last_mtime or user_mtime > self._last_user_mtime:
            self.load()

    def load(self):
        """Load Nextcloud bot configs from both repo and user bots.yaml.
        
        User config takes precedence over repo config.
        """
        self.bots.clear()
        
        # Track mtimes
        if self.repo_bots_yaml_path.exists():
            self._last_mtime = self.repo_bots_yaml_path.stat().st_mtime
        if self.user_bots_yaml_path.exists():
            self._last_user_mtime = self.user_bots_yaml_path.stat().st_mtime

        # Load from repo bots.yaml first (legacy support)
        if self.repo_bots_yaml_path.exists():
            self._load_from_yaml(self.repo_bots_yaml_path)
        
        # Load from user bots.yaml (overrides repo)
        if self.user_bots_yaml_path.exists():
            self._load_from_yaml(self.user_bots_yaml_path)
    
    def _load_from_yaml(self, yaml_path: Path):
        """Load nextcloud configs from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        if not data:
            return

        # Look for nextcloud config in each bot
        for bot_id, bot_config in data.get('bots', {}).items():
            nc_config = bot_config.get('nextcloud')
            if nc_config:
                self.bots[bot_id] = NextcloudBot(
                    ask_llm_bot=bot_id,
                    nextcloud_bot_id=nc_config.get('bot_id'),
                    secret=nc_config.get('secret', ''),
                    conversation_token=nc_config.get('conversation_token'),
                    enabled=nc_config.get('enabled', True),
                )

    def save(self):
        """Save Nextcloud configs to USER bots.yaml (not repo).
        
        This keeps secrets out of the repository.
        """
        # Ensure user config directory exists
        self.user_bots_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing user config or create empty
        if self.user_bots_yaml_path.exists():
            with open(self.user_bots_yaml_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        if 'bots' not in data:
            data['bots'] = {}

        # Update nextcloud section for each bot
        for bot_id, nc_bot in self.bots.items():
            if bot_id not in data['bots']:
                data['bots'][bot_id] = {}

            data['bots'][bot_id]['nextcloud'] = {
                'bot_id': nc_bot.nextcloud_bot_id,
                'secret': nc_bot.secret,
                'conversation_token': nc_bot.conversation_token,
                'enabled': nc_bot.enabled,
            }

        with open(self.user_bots_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        self._last_user_mtime = self.user_bots_yaml_path.stat().st_mtime

    def add_bot(
        self,
        ask_llm_bot: str,
        nextcloud_bot_id: int,
        secret: str,
        conversation_token: str,
    ):
        """Add Nextcloud config to a bot.
        
        Saves to ~/.config/ask-llm/bots.yaml (user config, not repo).
        """
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
        """Remove Nextcloud config from a bot.
        
        Removes from user config only (doesn't touch repo).
        """
        if ask_llm_bot not in self.bots:
            return False
        
        del self.bots[ask_llm_bot]
        
        # Remove from user YAML
        if self.user_bots_yaml_path.exists():
            with open(self.user_bots_yaml_path) as f:
                data = yaml.safe_load(f) or {}
            if 'bots' in data and ask_llm_bot in data['bots']:
                if 'nextcloud' in data['bots'][ask_llm_bot]:
                    del data['bots'][ask_llm_bot]['nextcloud']
                    # Clean up empty bot entry
                    if not data['bots'][ask_llm_bot]:
                        del data['bots'][ask_llm_bot]
            with open(self.user_bots_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return True

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

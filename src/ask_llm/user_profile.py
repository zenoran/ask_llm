"""User profile system for ask_llm.

User profiles are stored in PostgreSQL and contain user preferences and context
that bots can reference to personalize responses.

Uses SQLModel ORM for clean, type-safe database operations.
"""

import logging
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Column, JSON
from sqlmodel import Field, Session, SQLModel, create_engine, select

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"


class UserProfile(SQLModel, table=True):
    """SQLModel for user_profiles table."""
    
    __tablename__ = "user_profiles"  # type: ignore[assignment]
    
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(max_length=50, unique=True, index=True)
    name: str | None = Field(default=None, max_length=100)
    preferred_name: str | None = Field(default=None, max_length=100)
    preferences: dict = Field(default_factory=dict, sa_column=Column(JSON))
    context: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_display_name(self) -> str:
        """Get the name to use when addressing the user."""
        return self.preferred_name or self.name or ""
    
    def to_context_string(self) -> str:
        """Format user profile as context string for system prompt."""
        lines = []
        
        display_name = self.get_display_name()
        if display_name:
            lines.append(f"User's name: {display_name}")
        
        # Add preferences (non-default values)
        prefs = self.preferences or {}
        pref_parts = []
        if prefs.get("verbosity") and prefs["verbosity"] != "concise":
            pref_parts.append(f"{prefs['verbosity']} responses")
        if prefs.get("code_language") and prefs["code_language"] != "python":
            pref_parts.append(f"prefers {prefs['code_language']} for code examples")
        if prefs.get("response_style") and prefs["response_style"] != "direct":
            pref_parts.append(f"{prefs['response_style']} communication style")
        
        if pref_parts:
            lines.append(f"User preferences: {', '.join(pref_parts)}")
        
        # Add context items
        ctx = self.context or {}
        for key, value in ctx.items():
            if value:
                if isinstance(value, list):
                    lines.append(f"User's {key.replace('_', ' ')}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"User's {key.replace('_', ' ')}: {value}")
        
        return "\n".join(lines) if lines else ""


class UserProfileManager:
    """Manages user profiles in PostgreSQL using SQLModel ORM."""
    
    def __init__(self, config: Any):
        self.config = config
        
        host = getattr(config, 'POSTGRES_HOST', 'localhost')
        port = int(getattr(config, 'POSTGRES_PORT', 5432))
        user = getattr(config, 'POSTGRES_USER', 'askllm')
        password = getattr(config, 'POSTGRES_PASSWORD', '')
        database = getattr(config, 'POSTGRES_DATABASE', 'askllm')
        
        encoded_password = quote_plus(password)
        connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(connection_url, echo=False)
        self._ensure_table_exists()
        logger.debug(f"UserProfileManager connected to {host}:{port}/{database}")
    
    def _ensure_table_exists(self) -> None:
        """Create user_profiles table if it doesn't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.debug("Ensured user_profiles table exists")
    
    def _normalize_user_id(self, user_id: str) -> str:
        """Normalize user_id to lowercase."""
        return user_id.lower().strip()
    
    def list_all_profiles(self) -> list[UserProfile]:
        """Get all user profiles."""
        with Session(self.engine) as session:
            statement = select(UserProfile)
            return list(session.exec(statement).all())
    
    def get_profile(self, user_id: str = DEFAULT_USER_ID) -> UserProfile | None:
        """Get a user profile by ID, or None if not found."""
        user_id = self._normalize_user_id(user_id)
        with Session(self.engine) as session:
            statement = select(UserProfile).where(UserProfile.user_id == user_id)
            return session.exec(statement).first()
    
    def get_or_create_profile(self, user_id: str = DEFAULT_USER_ID) -> tuple[UserProfile, bool]:
        """Get existing profile or create a new empty one.
        
        Returns:
            Tuple of (profile, is_new) where is_new indicates if profile was just created.
        """
        user_id = self._normalize_user_id(user_id)
        profile = self.get_profile(user_id)
        if profile:
            return profile, False
        
        # Create new empty profile
        profile = UserProfile(user_id=user_id)
        with Session(self.engine) as session:
            session.add(profile)
            session.commit()
            session.refresh(profile)
            logger.debug(f"Created new user profile: {user_id}")
        
        return profile, True
    
    def save_profile(self, profile: UserProfile) -> bool:
        """Save a user profile to the database."""
        # Normalize user_id before saving
        profile.user_id = self._normalize_user_id(profile.user_id)
        with Session(self.engine) as session:
            try:
                # Check if exists
                existing = session.exec(
                    select(UserProfile).where(UserProfile.user_id == profile.user_id)
                ).first()
                
                if existing:
                    # Update existing
                    existing.name = profile.name
                    existing.preferred_name = profile.preferred_name
                    existing.preferences = profile.preferences
                    existing.context = profile.context
                    existing.updated_at = datetime.utcnow()
                    session.add(existing)
                else:
                    session.add(profile)
                
                session.commit()
                logger.debug(f"Saved user profile: {profile.user_id}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save user profile: {e}")
                return False
    
    def update_field(self, field_path: str, value: Any, user_id: str = DEFAULT_USER_ID) -> bool:
        """Update a specific field in the user profile.
        
        Args:
            field_path: Field name like "name", "preferred_name", or 
                       "preferences.verbosity", "context.occupation"
            value: The new value
            user_id: User ID
        """
        user_id = self._normalize_user_id(user_id)
        with Session(self.engine) as session:
            try:
                profile = session.exec(
                    select(UserProfile).where(UserProfile.user_id == user_id)
                ).first()
                
                if not profile:
                    profile = UserProfile(user_id=user_id)
                    session.add(profile)
                
                parts = field_path.split(".", 1)
                
                if len(parts) == 1:
                    # Top-level field
                    if parts[0] in ("name", "preferred_name"):
                        setattr(profile, parts[0], str(value) if value else None)
                    else:
                        logger.warning(f"Unknown top-level field: {parts[0]}")
                        return False
                elif parts[0] == "preferences":
                    if profile.preferences is None:
                        profile.preferences = {}
                    profile.preferences[parts[1]] = value
                elif parts[0] == "context":
                    if profile.context is None:
                        profile.context = {}
                    profile.context[parts[1]] = value
                else:
                    logger.warning(f"Unknown field category: {parts[0]}")
                    return False
                
                profile.updated_at = datetime.utcnow()
                session.commit()
                logger.debug(f"Updated user profile field {field_path} for {user_id}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update user profile field: {e}")
                return False
    
    def delete_context_field(self, field_name: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """Remove a field from the context."""
        with Session(self.engine) as session:
            try:
                profile = session.exec(
                    select(UserProfile).where(UserProfile.user_id == user_id)
                ).first()
                
                if profile and profile.context and field_name in profile.context:
                    del profile.context[field_name]
                    profile.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete context field: {e}")
                return False
    
    def list_users(self) -> list[str]:
        """List all user IDs in the database."""
        with Session(self.engine) as session:
            profiles = session.exec(select(UserProfile)).all()
            return [p.user_id for p in profiles]
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        with Session(self.engine) as session:
            try:
                profile = session.exec(
                    select(UserProfile).where(UserProfile.user_id == user_id)
                ).first()
                
                if profile:
                    session.delete(profile)
                    session.commit()
                    logger.debug(f"Deleted user profile: {user_id}")
                    return True
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete user profile: {e}")
                return False


def load_user_profile(config: Any, user_id: str = DEFAULT_USER_ID) -> UserProfile | None:
    """Convenience function to load a user profile."""
    try:
        manager = UserProfileManager(config)
        return manager.get_profile(user_id)
    except Exception as e:
        logger.warning(f"Could not load user profile from DB: {e}")
        return None

"""Short-term session history manager for PostgreSQL backend.

This module provides session-scoped message history management,
separate from the long-term memory storage for cleaner separation of concerns.
"""

import time
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    pass


class PostgreSQLShortTermManager:
    """Manages short-term (session) conversation history using PostgreSQL.
    
    This stores messages permanently but provides session-scoped access
    for building context windows.
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
        # Import here to avoid circular imports
        from .postgresql import PostgreSQLMemoryBackend
        
        self.config = config
        self.bot_id = bot_id
        self._backend = PostgreSQLMemoryBackend(config, bot_id)
        self._current_session_id = str(uuid.uuid4())
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._current_session_id
    
    def new_session(self) -> str:
        """Start a new session and return its ID."""
        self._current_session_id = str(uuid.uuid4())
        return self._current_session_id
    
    def add_message(self, role: str, content: str, timestamp: float | None = None) -> str:
        """Add a message to the current session.
        
        Returns the message ID.
        """
        message_id = str(uuid.uuid4())
        ts = timestamp or datetime.utcnow().timestamp()
        
        self._backend.add_message(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=ts,
            session_id=self._current_session_id,
        )
        
        return message_id
    
    def get_session_history(self, limit: int = 50) -> list[dict]:
        """Get messages from the current session."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            # Return in chronological order
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in reversed(rows)
            ]
    
    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """Get recent messages regardless of session (for context building)."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                }
                for row in reversed(rows)
            ]
    
    def count(self) -> int:
        """Count messages in current session."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(func.count())
                .select_from(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
            )
            return session.execute(stmt).scalar() or 0

    def get_messages(self, since_minutes: int | None = None) -> list:
        """Get messages, optionally filtered by time.
        
        Args:
            since_minutes: If provided, only return messages from the last N seconds
                          (note: despite the name, this is actually in seconds for 
                          backward compatibility with HISTORY_DURATION).
        
        Returns:
            List of Message objects.
        """
        from ..models.message import Message
        
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.asc())
            )
            
            if since_minutes is not None:
                cutoff = time.time() - since_minutes
                stmt = stmt.where(self._backend.messages_table.c.timestamp >= cutoff)
            
            rows = session.execute(stmt).fetchall()
            
            return [
                Message(role=row.role, content=row.content, timestamp=row.timestamp)
                for row in rows
            ]

    def clear(self) -> bool:
        """Clear all messages for this bot."""
        return self._backend.clear()

    def remove_last_message_if_partial(self, role: str) -> bool:
        """Remove the last message if it matches the specified role.
        
        Used for cleanup on error/interrupt.
        """
        with Session(self._backend.engine) as session:
            # Find the last message
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(1)
            )
            row = session.execute(stmt).fetchone()
            
            if row and row.role == role:
                delete_stmt = (
                    self._backend.messages_table.delete()
                    .where(self._backend.messages_table.c.id == row.id)
                )
                session.execute(delete_stmt)
                session.commit()
                return True
            return False

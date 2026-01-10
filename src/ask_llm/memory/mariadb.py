"""MariaDB memory backend for ask_llm using dynamic per-bot tables.

Each bot gets its own isolated tables:
  - {bot_id}_memories: Long-term memory with fulltext search
  - {bot_id}_messages: Short-term conversation history

Uses SQLAlchemy ORM for standard CRUD operations.
Raw SQL is only used for:
  - FULLTEXT search (MATCH AGAINST) - no ORM equivalent
  - Table creation with FULLTEXT indexes - requires specific syntax
  - information_schema queries for storage stats
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

from sqlalchemy import (
    Column, String, Text, Float, DateTime, Integer, MetaData, Table,
    create_engine, select, delete, text, insert, update
)
from sqlalchemy.orm import Session

from .base import MemoryBackend

if TYPE_CHECKING:
    from ..models.message import Message as MessageModel

logger = logging.getLogger(__name__)


def _sanitize_table_name(bot_id: str) -> str:
    """Sanitize bot_id for use in table names.
    
    Only allows alphanumeric and underscore, lowercase.
    """
    sanitized = re.sub(r'[^a-z0-9_]', '', bot_id.lower())
    if not sanitized:
        sanitized = "default"
    return sanitized


# Shared metadata for table definitions
metadata = MetaData()

# Cache for table objects
_memory_table_cache: dict[str, Table] = {}
_message_table_cache: dict[str, Table] = {}


def get_memory_table(bot_id: str) -> Table:
    """Get or create a memory Table for a specific bot."""
    table_name = f"{_sanitize_table_name(bot_id)}_memories"
    
    if table_name in _memory_table_cache:
        return _memory_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("created_at", DateTime, default=datetime.utcnow),
        extend_existing=True
    )
    
    _memory_table_cache[table_name] = table
    return table


def get_message_table(bot_id: str) -> Table:
    """Get or create a message Table for a specific bot."""
    table_name = f"{_sanitize_table_name(bot_id)}_messages"
    
    if table_name in _message_table_cache:
        return _message_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("created_at", DateTime, default=datetime.utcnow),
        extend_existing=True
    )
    
    _message_table_cache[table_name] = table
    return table


class MariaDBMemoryBackend(MemoryBackend):
    """MariaDB-based memory backend with dynamic per-bot tables.
    
    Each bot gets its own table for complete isolation:
      - {bot_id}_memories: Long-term memory with fulltext search
    
    Uses SQLAlchemy ORM for CRUD operations.
    
    Configuration (via environment variables or .env):
        ASK_LLM_MARIADB_HOST: Database host (default: localhost)
        ASK_LLM_MARIADB_PORT: Database port (default: 3306)
        ASK_LLM_MARIADB_USER: Database user
        ASK_LLM_MARIADB_PASSWORD: Database password
        ASK_LLM_MARIADB_DATABASE: Database name (default: ask_llm)
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
        super().__init__(config, bot_id=bot_id)
        
        # Get MariaDB connection settings from config
        host = getattr(config, 'MARIADB_HOST', 'localhost')
        port = int(getattr(config, 'MARIADB_PORT', 3306))
        user = getattr(config, 'MARIADB_USER', 'ask_llm')
        password = getattr(config, 'MARIADB_PASSWORD', '')
        database = getattr(config, 'MARIADB_DATABASE', 'ask_llm')
        
        self.database = database
        self._table_name = f"{_sanitize_table_name(bot_id)}_memories"
        
        # Get the table for this bot
        self.table = get_memory_table(bot_id)
        
        # Build connection URL for MariaDB
        encoded_password = quote_plus(password)
        connection_url = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(connection_url, echo=False)
        self._ensure_table_exists()
        logger.debug(f"Connected to MariaDB at {host}:{port}/{database} (table: {self._table_name})")
    
    def _ensure_table_exists(self) -> None:
        """Create the bot's memory table if it doesn't exist.
        
        Uses raw SQL for FULLTEXT index which isn't supported by SQLAlchemy's DDL.
        """
        create_sql = text(f"""
            CREATE TABLE IF NOT EXISTS `{self._table_name}` (
                id VARCHAR(36) PRIMARY KEY,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                timestamp DOUBLE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_{self._table_name}_timestamp (timestamp),
                FULLTEXT INDEX ft_content (content)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        with self.engine.connect() as conn:
            try:
                conn.execute(create_sql)
                conn.commit()
                logger.debug(f"Ensured table {self._table_name} exists")
            except Exception as e:
                logger.error(f"Failed to create table {self._table_name}: {e}")
                raise
    
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to memory storage."""
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {message_id}")
            return
        
        with Session(self.engine) as session:
            try:
                # Check if exists
                stmt = select(self.table).where(self.table.c.id == message_id)
                existing = session.execute(stmt).first()
                
                if existing:
                    # Update existing
                    stmt = (
                        update(self.table)
                        .where(self.table.c.id == message_id)
                        .values(content=content, timestamp=timestamp)
                    )
                    session.execute(stmt)
                else:
                    # Insert new
                    stmt = insert(self.table).values(
                        id=message_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        created_at=datetime.utcnow()
                    )
                    session.execute(stmt)
                
                session.commit()
                logger.debug(f"Added memory ID: {message_id} to {self._table_name}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add memory {message_id}: {e}")
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for memories relevant to the query using fulltext search.
        
        Uses raw SQL for MATCH AGAINST (FULLTEXT) - no ORM equivalent.
        """
        if not query or query.isspace():
            logger.warning("Attempted to search with empty query")
            return None
        
        with self.engine.connect() as conn:
            try:
                # Use MATCH AGAINST for fulltext search
                fulltext_sql = text(f"""
                    SELECT id, role, content, timestamp,
                           MATCH(content) AGAINST(:query IN NATURAL LANGUAGE MODE) AS relevance
                    FROM `{self._table_name}`
                    WHERE MATCH(content) AGAINST(:query IN NATURAL LANGUAGE MODE)
                    ORDER BY relevance DESC
                    LIMIT :limit
                """)
                
                result = conn.execute(fulltext_sql, {"query": query, "limit": n_results})
                rows = result.fetchall()
                
                if not rows:
                    # Fallback to LIKE search if fulltext returns nothing
                    search_terms = query.split()[:5]
                    like_pattern = '%' + '%'.join(search_terms) + '%'
                    
                    like_sql = text(f"""
                        SELECT id, role, content, timestamp, 0.5 AS relevance
                        FROM `{self._table_name}`
                        WHERE content LIKE :pattern
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """)
                    
                    result = conn.execute(like_sql, {"pattern": like_pattern, "limit": n_results})
                    rows = result.fetchall()
                
                if not rows:
                    return None
                
                # Normalize relevance scores to 0.0-1.0 range
                max_relevance = max(row.relevance for row in rows) or 1.0
                
                formatted = []
                for row in rows:
                    normalized_relevance = row.relevance / max_relevance if max_relevance > 0 else 0.5
                    
                    if normalized_relevance < min_relevance:
                        continue
                    
                    formatted.append({
                        'id': row.id,
                        'document': row.content,
                        'metadata': {
                            'role': row.role,
                            'timestamp': row.timestamp
                        },
                        'relevance': normalized_relevance
                    })
                
                return formatted if formatted else None
                
            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return None
    
    def clear(self) -> bool:
        """Clear all memories from this bot's table."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.table)
                session.execute(stmt)
                session.commit()
                logger.info(f"Cleared all memories from {self._table_name}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear memories: {e}")
                return False
    
    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories."""
        with Session(self.engine) as session:
            try:
                stmt = (
                    select(self.table)
                    .order_by(self.table.c.timestamp.desc())
                    .limit(n)
                )
                rows = session.execute(stmt).fetchall()
                
                return [
                    {
                        'id': row.id,
                        'document': row.content,
                        'metadata': {
                            'role': row.role,
                            'timestamp': row.timestamp
                        }
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Failed to list recent memories: {e}")
                return []
    
    def stats(self) -> dict:
        """Get statistics about the memory storage."""
        with Session(self.engine) as session:
            try:
                stmt = select(self.table)
                rows = session.execute(stmt).fetchall()
                
                if not rows:
                    return {
                        'total_count': 0,
                        'oldest_timestamp': None,
                        'newest_timestamp': None,
                        'storage_size': 0,
                        'table_name': self._table_name
                    }
                
                timestamps = [row.timestamp for row in rows]
                
                # Storage size requires raw SQL (information_schema)
                size_sql = text("""
                    SELECT data_length + index_length as storage_size
                    FROM information_schema.tables
                    WHERE table_schema = :database AND table_name = :table_name
                """)
                with self.engine.connect() as conn:
                    size_result = conn.execute(size_sql, {
                        "database": self.database,
                        "table_name": self._table_name
                    }).first()
                
                return {
                    'total_count': len(rows),
                    'oldest_timestamp': min(timestamps),
                    'newest_timestamp': max(timestamps),
                    'storage_size': size_result.storage_size if size_result else 0,
                    'table_name': self._table_name
                }
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
                return {'total_count': 0, 'error': str(e)}
    
    def delete(self, message_id: str) -> bool:
        """Delete a specific memory by ID."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.table).where(self.table.c.id == message_id)
                result = session.execute(stmt)
                session.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted memory ID: {message_id}")
                return deleted
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete memory {message_id}: {e}")
                return False
    
    def prune_older_than(self, days: int) -> int:
        """Delete memories older than a specified number of days."""
        cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
        
        with Session(self.engine) as session:
            try:
                stmt = delete(self.table).where(self.table.c.timestamp < cutoff_timestamp)
                result = session.execute(stmt)
                session.commit()
                count = result.rowcount
                logger.info(f"Pruned {count} memories older than {days} days from {self._table_name}")
                return count
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to prune memories: {e}")
                return 0

    @classmethod
    def get_short_term_manager(cls, config: Any, bot_id: str = "nova") -> "ShortTermMemoryManager":
        """Factory method to create a ShortTermMemoryManager instance."""
        return ShortTermMemoryManager(config, bot_id=bot_id)


class ShortTermMemoryManager:
    """MariaDB-based short-term memory with dynamic per-bot tables.
    
    Each bot gets its own table: {bot_id}_messages
    
    Uses SQLAlchemy ORM for all operations.
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
        self.config = config
        self.bot_id = bot_id
        self._table_name = f"{_sanitize_table_name(bot_id)}_messages"
        
        # Get the table for this bot
        self.table = get_message_table(bot_id)
        
        # Get MariaDB connection settings from config
        host = getattr(config, 'MARIADB_HOST', 'localhost')
        port = int(getattr(config, 'MARIADB_PORT', 3306))
        user = getattr(config, 'MARIADB_USER', 'ask_llm')
        password = getattr(config, 'MARIADB_PASSWORD', '')
        database = getattr(config, 'MARIADB_DATABASE', 'ask_llm')
        
        self.database = database
        encoded_password = quote_plus(password)
        connection_url = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(connection_url, echo=False)
        self._ensure_table_exists()
        logger.debug(f"Short-term memory connected (table: {self._table_name})")
    
    def _ensure_table_exists(self) -> None:
        """Create the bot's messages table if it doesn't exist."""
        create_sql = text(f"""
            CREATE TABLE IF NOT EXISTS `{self._table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                timestamp DOUBLE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_{self._table_name}_timestamp (timestamp),
                INDEX idx_{self._table_name}_role (role)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        with self.engine.connect() as conn:
            try:
                conn.execute(create_sql)
                conn.commit()
                logger.debug(f"Ensured table {self._table_name} exists")
            except Exception as e:
                logger.error(f"Failed to create table {self._table_name}: {e}")
                raise
    
    def get_messages(self, since_minutes: int | None = None) -> list["MessageModel"]:
        """Load messages from short-term memory.
        
        Args:
            since_minutes: Only load messages from the last N seconds (legacy naming).
        """
        from ..models.message import Message as MessageModel
        
        with Session(self.engine) as session:
            try:
                if since_minutes is not None:
                    cutoff = time.time() - since_minutes
                    stmt = (
                        select(self.table)
                        .where(self.table.c.timestamp >= cutoff)
                        .order_by(self.table.c.timestamp)
                    )
                else:
                    stmt = select(self.table).order_by(self.table.c.timestamp)
                
                rows = session.execute(stmt).fetchall()
                return [
                    MessageModel(role=row.role, content=row.content, timestamp=row.timestamp)
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Failed to load short-term messages: {e}")
                return []
    
    def add_message(self, role: str, content: str, timestamp: float | None = None) -> None:
        """Add a message to short-term memory."""
        if not content:
            return
        
        with Session(self.engine) as session:
            try:
                stmt = insert(self.table).values(
                    role=role,
                    content=content,
                    timestamp=timestamp or time.time(),
                    created_at=datetime.utcnow()
                )
                session.execute(stmt)
                session.commit()
                logger.debug(f"Added short-term message: {role}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add short-term message: {e}")
    
    def clear(self) -> bool:
        """Clear all short-term memory."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.table)
                session.execute(stmt)
                session.commit()
                logger.info(f"Cleared short-term memory from {self._table_name}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear short-term memory: {e}")
                return False
    
    def get_message_count(self) -> int:
        """Get the count of messages in short-term memory."""
        with Session(self.engine) as session:
            try:
                stmt = select(self.table)
                rows = session.execute(stmt).fetchall()
                return len(rows)
            except Exception as e:
                logger.error(f"Failed to count short-term messages: {e}")
                return 0
    
    def remove_last_message_if_partial(self, role: str) -> bool:
        """Remove the last message if it matches the specified role."""
        with Session(self.engine) as session:
            try:
                # Get the last message
                stmt = (
                    select(self.table)
                    .order_by(self.table.c.timestamp.desc(), self.table.c.id.desc())
                    .limit(1)
                )
                last_row = session.execute(stmt).first()
                
                if last_row and last_row.role == role:
                    del_stmt = delete(self.table).where(self.table.c.id == last_row.id)
                    session.execute(del_stmt)
                    session.commit()
                    logger.debug(f"Removed last {role} message from {self._table_name}")
                    return True
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to remove last message: {e}")
                return False
    
    def get_last_assistant_message(self) -> str | None:
        """Get the content of the last assistant message."""
        with Session(self.engine) as session:
            try:
                stmt = (
                    select(self.table)
                    .where(self.table.c.role == "assistant")
                    .order_by(self.table.c.timestamp.desc(), self.table.c.id.desc())
                    .limit(1)
                )
                row = session.execute(stmt).first()
                return row.content if row else None
            except Exception as e:
                logger.error(f"Failed to get last assistant message: {e}")
                return None
    
    def stats(self) -> dict:
        """Get statistics about short-term memory storage."""
        with Session(self.engine) as session:
            try:
                stmt = select(self.table)
                rows = session.execute(stmt).fetchall()
                
                if not rows:
                    return {
                        'total_count': 0,
                        'oldest_timestamp': None,
                        'newest_timestamp': None,
                        'table_name': self._table_name
                    }
                
                timestamps = [row.timestamp for row in rows]
                
                return {
                    'total_count': len(rows),
                    'oldest_timestamp': min(timestamps),
                    'newest_timestamp': max(timestamps),
                    'table_name': self._table_name
                }
            except Exception as e:
                logger.error(f"Failed to get short-term memory stats: {e}")
                return {'total_count': 0, 'error': str(e)}

"""MariaDB memory backend for ask_llm using SQLModel ORM.

This backend stores conversation memories in MariaDB using fulltext search
for semantic retrieval. Uses SQLModel (Pydantic + SQLAlchemy) for type safety.
"""

import logging
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

from sqlalchemy import Column, Index, Text, text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .base import MemoryBackend

if TYPE_CHECKING:
    from ..models.message import Message as MessageModel

logger = logging.getLogger(__name__)


class Memory(SQLModel, table=True):
    """SQLModel for the long-term memories table."""
    
    __tablename__: str = "memories"  # type: ignore[assignment]
    
    id: str = Field(primary_key=True, max_length=36)
    bot_id: str = Field(default="nova", max_length=50, index=True)  # Bot isolation
    role: str = Field(max_length=20)
    content: str = Field(sa_column_kwargs={"nullable": False})
    timestamp: float = Field(index=True)
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index('ft_content', 'content', mysql_prefix='FULLTEXT'),
        Index('idx_bot_timestamp', 'bot_id', 'timestamp'),
    )


class ShortTermMemory(SQLModel, table=True):
    """SQLModel for the short-term memory table (session history)."""
    
    __tablename__: str = "short_term_memory"  # type: ignore[assignment]
    
    id: int | None = Field(default=None, primary_key=True)
    bot_id: str = Field(default="nova", max_length=50, index=True)  # Bot isolation
    role: str = Field(max_length=20)
    content: str = Field(sa_column=Column(Text, nullable=False))
    timestamp: float = Field(index=True)
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_stm_bot_timestamp', 'bot_id', 'timestamp'),
    )


class MariaDBMemoryBackend(MemoryBackend):
    """MariaDB-based memory backend using SQLModel and fulltext search.
    
    Stores messages in a MariaDB table with fulltext indexing for
    relevance-based retrieval.
    
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
        
        # Build connection URL for MariaDB (URL-encode password for special chars)
        self.database = database
        encoded_password = quote_plus(password)
        connection_url = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(connection_url, echo=False)
        self._ensure_table_exists()
        self._run_migrations()
        logger.debug(f"Connected to MariaDB at {host}:{port}/{database} (bot: {bot_id})")
    
    def _ensure_table_exists(self) -> None:
        """Create the memories table if it doesn't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.debug("Ensured memories table exists")
    
    def _run_migrations(self) -> None:
        """Run any needed database migrations."""
        with Session(self.engine) as session:
            try:
                # Check if bot_id column exists in memories table
                result = session.exec(text(  # type: ignore
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema = :db AND table_name = 'memories' AND column_name = 'bot_id'"
                ).bindparams(db=self.database)).first()
                
                if result and result[0] == 0:
                    # Add bot_id column and migrate existing data to 'nova'
                    logger.info("Migrating memories table: adding bot_id column")
                    session.exec(text("ALTER TABLE memories ADD COLUMN bot_id VARCHAR(50) DEFAULT 'nova' NOT NULL"))  # type: ignore
                    session.exec(text("CREATE INDEX idx_bot_timestamp ON memories (bot_id, timestamp)"))  # type: ignore
                    session.commit()
                    logger.info("Migration complete: bot_id column added to memories")
                
                # Check if bot_id column exists in short_term_memory table
                result = session.exec(text(  # type: ignore
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema = :db AND table_name = 'short_term_memory' AND column_name = 'bot_id'"
                ).bindparams(db=self.database)).first()
                
                if result and result[0] == 0:
                    # Add bot_id column and migrate existing data to 'nova'
                    logger.info("Migrating short_term_memory table: adding bot_id column")
                    session.exec(text("ALTER TABLE short_term_memory ADD COLUMN bot_id VARCHAR(50) DEFAULT 'nova' NOT NULL"))  # type: ignore
                    session.exec(text("CREATE INDEX idx_stm_bot_timestamp ON short_term_memory (bot_id, timestamp)"))  # type: ignore
                    session.commit()
                    logger.info("Migration complete: bot_id column added to short_term_memory")
                    
            except Exception as e:
                logger.warning(f"Migration check/run encountered an issue: {e}")
    
    def add(self, message_id: str, role: str, content: str, timestamp: float, bot_id: str | None = None) -> None:
        """Add a message to memory storage."""
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {message_id}")
            return
        
        memory = Memory(
            id=message_id, 
            bot_id=bot_id or self.bot_id,
            role=role, 
            content=content, 
            timestamp=timestamp
        )
        
        with Session(self.engine) as session:
            try:
                session.add(memory)
                session.commit()
                logger.debug(f"Added memory ID: {message_id} (bot: {memory.bot_id})")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add memory {message_id}: {e}")
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0, bot_id: str | None = None) -> list[dict] | None:
        """Search for memories relevant to the query using fulltext search.
        
        Args:
            query: The text to search for.
            n_results: Maximum number of results to return.
            min_relevance: Minimum relevance score (0.0-1.0) to include.
            bot_id: Filter by bot (defaults to instance bot_id).
        """
        if not query or query.isspace():
            logger.warning("Attempted to search with empty query")
            return None
        
        search_bot_id = bot_id or self.bot_id
        
        with Session(self.engine) as session:
            try:
                # Use raw SQL for MATCH AGAINST (not directly supported by SQLAlchemy ORM)
                fulltext_sql = text("""
                    SELECT id, role, content, timestamp,
                           MATCH(content) AGAINST(:query IN NATURAL LANGUAGE MODE) AS relevance
                    FROM memories
                    WHERE bot_id = :bot_id AND MATCH(content) AGAINST(:query IN NATURAL LANGUAGE MODE)
                    ORDER BY relevance DESC
                    LIMIT :limit
                """)
                
                result = session.exec(fulltext_sql.bindparams(query=query, bot_id=search_bot_id, limit=n_results))  # type: ignore
                rows = result.fetchall()
                
                if not rows:
                    # Fallback to LIKE search if fulltext returns nothing
                    search_terms = query.split()[:5]
                    like_pattern = '%' + '%'.join(search_terms) + '%'
                    
                    like_sql = text("""
                        SELECT id, role, content, timestamp, 0.5 AS relevance
                        FROM memories
                        WHERE bot_id = :bot_id AND content LIKE :pattern
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """)
                    
                    result = session.exec(like_sql.bindparams(bot_id=search_bot_id, pattern=like_pattern, limit=n_results))  # type: ignore
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
    
    def clear(self, bot_id: str | None = None) -> bool:
        """Clear all memories from storage for the specified bot."""
        clear_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                session.exec(text("DELETE FROM memories WHERE bot_id = :bot_id").bindparams(bot_id=clear_bot_id))  # type: ignore
                session.commit()
                logger.info(f"Cleared all memories for bot: {clear_bot_id}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear memories: {e}")
                return False
    
    def list_recent(self, n: int = 10, bot_id: str | None = None) -> list[dict]:
        """List the most recent memories for the specified bot."""
        list_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                statement = select(Memory).where(Memory.bot_id == list_bot_id).order_by(Memory.timestamp.desc()).limit(n)  # type: ignore
                memories = session.exec(statement).all()
                
                return [
                    {
                        'id': mem.id,
                        'document': mem.content,
                        'metadata': {
                            'role': mem.role,
                            'timestamp': mem.timestamp
                        }
                    }
                    for mem in memories
                ]
            except Exception as e:
                logger.error(f"Failed to list recent memories: {e}")
                return []
    
    def stats(self, bot_id: str | None = None) -> dict:
        """Get statistics about the memory storage for the specified bot."""
        stats_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                stats_sql = text("""
                    SELECT 
                        COUNT(*) as total_count,
                        MIN(timestamp) as oldest_timestamp,
                        MAX(timestamp) as newest_timestamp
                    FROM memories
                    WHERE bot_id = :bot_id
                """)
                result = session.exec(stats_sql.bindparams(bot_id=stats_bot_id)).first()  # type: ignore
                
                size_sql = text("""
                    SELECT data_length + index_length as storage_size
                    FROM information_schema.tables
                    WHERE table_schema = :database AND table_name = 'memories'
                """)
                size_result = session.exec(size_sql.bindparams(database=self.database)).first()  # type: ignore
                
                if not result:
                    return {'total_count': 0, 'oldest_timestamp': None, 'newest_timestamp': None, 'storage_size': 0, 'bot_id': stats_bot_id}
                
                return {
                    'total_count': result.total_count or 0,
                    'oldest_timestamp': result.oldest_timestamp,
                    'newest_timestamp': result.newest_timestamp,
                    'storage_size': size_result.storage_size if size_result else 0,
                    'bot_id': stats_bot_id
                }
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
                return {'total_count': 0, 'error': str(e), 'bot_id': stats_bot_id}
    
    def delete(self, message_id: str) -> bool:
        """Delete a specific memory by ID."""
        with Session(self.engine) as session:
            try:
                memory = session.get(Memory, message_id)
                if memory:
                    session.delete(memory)
                    session.commit()
                    logger.debug(f"Deleted memory ID: {message_id}")
                    return True
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete memory {message_id}: {e}")
                return False
    
    def prune_older_than(self, days: int, bot_id: str | None = None) -> int:
        """Delete memories older than a specified number of days for the specified bot."""
        cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
        prune_bot_id = bot_id or self.bot_id
        
        with Session(self.engine) as session:
            try:
                delete_sql = text("DELETE FROM memories WHERE bot_id = :bot_id AND timestamp < :cutoff")
                result = session.exec(delete_sql.bindparams(bot_id=prune_bot_id, cutoff=cutoff_timestamp))  # type: ignore
                session.commit()
                deleted_count = result.rowcount
                logger.info(f"Pruned {deleted_count} memories older than {days} days for bot: {prune_bot_id}")
                return deleted_count
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to prune memories: {e}")
                return 0

    @classmethod
    def get_short_term_manager(cls, config: Any, bot_id: str = "nova") -> "ShortTermMemoryManager":
        """Factory method to create a ShortTermMemoryManager instance.
        
        This allows the core module to get a short-term memory backend
        without needing to know about the specific implementation.
        
        Args:
            config: Application configuration object
            bot_id: Bot identifier for memory isolation
        """
        return ShortTermMemoryManager(config, bot_id=bot_id)


class ShortTermMemoryManager:
    """MariaDB-based short-term memory (session history) backend.
    
    Replaces the file-based history when MariaDB is available.
    Behaves exactly like the file-based history but persists to MariaDB.
    
    This class provides the same interface expected by HistoryManager.
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
        self.config = config
        self.bot_id = bot_id
        
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
        logger.debug(f"Short-term memory connected to MariaDB at {host}:{port}/{database} (bot: {bot_id})")
    
    def _ensure_table_exists(self) -> None:
        """Create the short_term_memory table if it doesn't exist."""
        SQLModel.metadata.create_all(self.engine)
        logger.debug("Ensured short_term_memory table exists")
    
    def get_messages(self, since_minutes: int | None = None, bot_id: str | None = None) -> list["MessageModel"]:
        """Load messages from short-term memory.
        
        Args:
            since_minutes: Only load messages from the last N minutes.
            bot_id: Filter by bot (defaults to instance bot_id).
            
        Returns:
            List of Message objects compatible with HistoryManager.
        """
        # Import here to avoid circular imports
        from ..models.message import Message as MessageModel
        
        get_bot_id = bot_id or self.bot_id
        
        with Session(self.engine) as session:
            try:
                if since_minutes is not None:
                    cutoff = time.time() - (since_minutes * 60)
                    statement = select(ShortTermMemory).where(
                        ShortTermMemory.bot_id == get_bot_id,
                        ShortTermMemory.timestamp >= cutoff
                    ).order_by(ShortTermMemory.timestamp)  # type: ignore[arg-type]
                else:
                    statement = select(ShortTermMemory).where(
                        ShortTermMemory.bot_id == get_bot_id
                    ).order_by(ShortTermMemory.timestamp)  # type: ignore[arg-type]
                
                messages = session.exec(statement).all()
                return [
                    MessageModel(role=msg.role, content=msg.content, timestamp=msg.timestamp)
                    for msg in messages
                ]
            except Exception as e:
                logger.error(f"Failed to load short-term messages: {e}")
                return []
    
    def add_message(self, role: str, content: str, timestamp: float | None = None, bot_id: str | None = None) -> None:
        """Add a message to short-term memory."""
        if not content:
            return
        
        add_bot_id = bot_id or self.bot_id
            
        msg = ShortTermMemory(
            bot_id=add_bot_id,
            role=role,
            content=content,
            timestamp=timestamp or time.time()
        )
        
        with Session(self.engine) as session:
            try:
                session.add(msg)
                session.commit()
                logger.debug(f"Added short-term message: {role} (bot: {add_bot_id})")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add short-term message: {e}")
    
    def clear(self, bot_id: str | None = None) -> bool:
        """Clear all short-term memory for the specified bot."""
        clear_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                session.exec(text("DELETE FROM short_term_memory WHERE bot_id = :bot_id").bindparams(bot_id=clear_bot_id))  # type: ignore
                session.commit()
                logger.info(f"Cleared short-term memory for bot: {clear_bot_id}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear short-term memory: {e}")
                return False
    
    def get_message_count(self, bot_id: str | None = None) -> int:
        """Get the count of messages in short-term memory for the specified bot."""
        count_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                result = session.exec(text("SELECT COUNT(*) FROM short_term_memory WHERE bot_id = :bot_id").bindparams(bot_id=count_bot_id)).first()  # type: ignore
                return result[0] if result else 0
            except Exception as e:
                logger.error(f"Failed to count short-term messages: {e}")
                return 0
    
    def remove_last_message_if_partial(self, role: str, bot_id: str | None = None) -> bool:
        """Remove the last message if it matches the specified role.
        
        Used by HistoryManager to cleanup partial messages on error/interrupt.
        """
        remove_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                # Get the last message for this bot
                last_msg = session.exec(
                    select(ShortTermMemory).where(
                        ShortTermMemory.bot_id == remove_bot_id
                    ).order_by(ShortTermMemory.timestamp.desc()).limit(1)  # type: ignore
                ).first()
                
                if last_msg and last_msg.role == role:
                    session.delete(last_msg)
                    session.commit()
                    logger.debug(f"Removed last {role} message from short-term memory (bot: {remove_bot_id})")
                    return True
                return False
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to remove last message: {e}")
                return False
    
    def get_last_assistant_message(self, bot_id: str | None = None) -> str | None:
        """Get the content of the last assistant message for the specified bot."""
        get_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                last_msg = session.exec(
                    select(ShortTermMemory).where(
                        ShortTermMemory.bot_id == get_bot_id,
                        ShortTermMemory.role == "assistant"
                    ).order_by(ShortTermMemory.timestamp.desc()).limit(1)  # type: ignore
                ).first()
                return last_msg.content if last_msg else None
            except Exception as e:
                logger.error(f"Failed to get last assistant message: {e}")
                return None
    
    def stats(self, bot_id: str | None = None) -> dict:
        """Get statistics about short-term memory storage for the specified bot."""
        stats_bot_id = bot_id or self.bot_id
        with Session(self.engine) as session:
            try:
                stats_sql = text("""
                    SELECT 
                        COUNT(*) as total_count,
                        MIN(timestamp) as oldest_timestamp,
                        MAX(timestamp) as newest_timestamp
                    FROM short_term_memory
                    WHERE bot_id = :bot_id
                """)
                result = session.exec(stats_sql.bindparams(bot_id=stats_bot_id)).first()  # type: ignore
                
                if not result:
                    return {'total_count': 0, 'oldest_timestamp': None, 'newest_timestamp': None, 'bot_id': stats_bot_id}
                
                return {
                    'total_count': result.total_count or 0,
                    'oldest_timestamp': result.oldest_timestamp,
                    'newest_timestamp': result.newest_timestamp,
                    'bot_id': stats_bot_id
                }
            except Exception as e:
                logger.error(f"Failed to get short-term memory stats: {e}")
                return {'total_count': 0, 'error': str(e), 'bot_id': stats_bot_id}

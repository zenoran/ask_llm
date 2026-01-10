"""PostgreSQL memory backend for ask_llm with pgvector support.

This backend uses PostgreSQL with the pgvector extension for semantic similarity search.
Each bot gets its own isolated tables:
  - {bot_id}_messages: Permanent message storage (all messages)
  - {bot_id}_memories: Distilled, importance-weighted memories with embeddings

The separation allows:
  - Messages: Complete conversation history, never deleted
  - Memories: Curated facts extracted from conversations with importance scores
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

from sqlalchemy import (
    Column, String, Text, Float, DateTime, Integer, Boolean, MetaData, Table,
    create_engine, select, delete, text, insert, update, JSON
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

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


def get_message_table_pg(bot_id: str) -> Table:
    """Get or create a message Table for a specific bot (PostgreSQL version)."""
    table_name = f"{_sanitize_table_name(bot_id)}_messages"
    
    if table_name in _message_table_cache:
        return _message_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("session_id", String(36), nullable=True),  # For grouping conversations
        Column("processed", Boolean, default=False),  # Whether memory extraction has run
        Column("created_at", DateTime, default=datetime.utcnow),
        extend_existing=True
    )
    
    _message_table_cache[table_name] = table
    return table


def get_memory_table_pg(bot_id: str) -> Table:
    """Get or create a memory Table for a specific bot (PostgreSQL version)."""
    table_name = f"{_sanitize_table_name(bot_id)}_memories"
    
    if table_name in _memory_table_cache:
        return _memory_table_cache[table_name]
    
    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),  # UUID
        Column("content", Text, nullable=False),
        Column("memory_type", String(50), nullable=False, default="misc"),
        Column("importance", Float, nullable=False, default=0.5),
        Column("source_message_ids", JSON, nullable=True),  # Array of message UUIDs
        Column("access_count", Integer, default=0),  # For reinforcement
        Column("last_accessed", DateTime, nullable=True),
        Column("created_at", DateTime, default=datetime.utcnow),
        Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        # embedding column added via raw SQL for pgvector
        extend_existing=True
    )
    
    _memory_table_cache[table_name] = table
    return table


class PostgreSQLMemoryBackend(MemoryBackend):
    """PostgreSQL-based memory backend with pgvector for semantic search.
    
    This is the new memory backend designed for:
    - Permanent message storage (all messages preserved)
    - Distilled memories with importance weighting
    - Semantic similarity search via pgvector
    - Source linking back to original messages
    
    Configuration (via environment variables or .env):
        ASK_LLM_POSTGRES_HOST: Database host (default: localhost)
        ASK_LLM_POSTGRES_PORT: Database port (default: 5432)
        ASK_LLM_POSTGRES_USER: Database user
        ASK_LLM_POSTGRES_PASSWORD: Database password
        ASK_LLM_POSTGRES_DATABASE: Database name (default: askllm)
    """
    
    # Embedding dimension (matches common embedding models)
    EMBEDDING_DIM = 1536  # OpenAI ada-002 dimension, can be configured
    
    def __init__(self, config: Any, bot_id: str = "nova", embedding_dim: int = 1536):
        super().__init__(config, bot_id=bot_id)
        
        # Get PostgreSQL connection settings from config
        host = getattr(config, 'POSTGRES_HOST', 'postgres.home')
        port = int(getattr(config, 'POSTGRES_PORT', 5432))
        user = getattr(config, 'POSTGRES_USER', 'askllm')
        password = getattr(config, 'POSTGRES_PASSWORD', '')
        database = getattr(config, 'POSTGRES_DATABASE', 'askllm')
        
        self.database = database
        self.bot_id_sanitized = _sanitize_table_name(bot_id)
        self._messages_table_name = f"{self.bot_id_sanitized}_messages"
        self._memories_table_name = f"{self.bot_id_sanitized}_memories"
        self.embedding_dim = embedding_dim
        
        # Get table definitions
        self.messages_table = get_message_table_pg(bot_id)
        self.memories_table = get_memory_table_pg(bot_id)
        
        # Build connection URL for PostgreSQL
        encoded_password = quote_plus(password)
        connection_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
        
        self.engine = create_engine(
            connection_url,
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
        )
        
        self._ensure_tables_exist()
        logger.debug(f"Connected to PostgreSQL at {host}:{port}/{database} (bot: {bot_id})")
    
    def _ensure_tables_exist(self) -> None:
        """Create the bot's tables if they don't exist."""
        with self.engine.connect() as conn:
            # Ensure pgvector extension is available
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            except Exception as e:
                logger.debug(f"pgvector extension check: {e}")
            
            # Create messages table
            messages_sql = text(f"""
                CREATE TABLE IF NOT EXISTS {self._messages_table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    session_id VARCHAR(36),
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create memories table with vector column
            memories_sql = text(f"""
                CREATE TABLE IF NOT EXISTS {self._memories_table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR(50) NOT NULL DEFAULT 'misc',
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_message_ids JSONB,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector({self.embedding_dim})
                )
            """)
            
            try:
                conn.execute(messages_sql)
                conn.execute(memories_sql)
                conn.commit()
                
                # Create indexes
                self._create_indexes(conn)
                
                logger.debug(f"Ensured tables exist for bot {self.bot_id}")
            except Exception as e:
                logger.error(f"Failed to create tables: {e}")
                raise
    
    def _create_indexes(self, conn) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_timestamp ON {self._messages_table_name}(timestamp)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_session ON {self._messages_table_name}(session_id)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._messages_table_name}_processed ON {self._messages_table_name}(processed)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_importance ON {self._memories_table_name}(importance)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_type ON {self._memories_table_name}(memory_type)",
            f"CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_accessed ON {self._memories_table_name}(last_accessed)",
        ]
        
        # HNSW index for vector similarity (if we have embeddings)
        # This is the fastest index type for pgvector
        hnsw_index = f"""
            CREATE INDEX IF NOT EXISTS idx_{self._memories_table_name}_embedding 
            ON {self._memories_table_name} 
            USING hnsw (embedding vector_cosine_ops)
        """
        
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception as e:
                logger.debug(f"Index creation (may already exist): {e}")
        
        try:
            conn.execute(text(hnsw_index))
        except Exception as e:
            logger.debug(f"HNSW index creation (may already exist): {e}")
        
        conn.commit()
    
    # =========================================================================
    # Message Storage (permanent conversation history)
    # =========================================================================
    
    def add_message(
        self,
        message_id: str,
        role: str,
        content: str,
        timestamp: float,
        session_id: str | None = None,
    ) -> None:
        """Add a message to permanent storage.
        
        Messages are NEVER deleted - they form the complete conversation history.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for message ID: {message_id}")
            return
        
        with Session(self.engine) as session:
            try:
                # Check if exists (upsert)
                stmt = select(self.messages_table).where(
                    self.messages_table.c.id == message_id
                )
                existing = session.execute(stmt).first()
                
                if existing:
                    stmt = (
                        update(self.messages_table)
                        .where(self.messages_table.c.id == message_id)
                        .values(content=content, timestamp=timestamp)
                    )
                    session.execute(stmt)
                else:
                    stmt = insert(self.messages_table).values(
                        id=message_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        session_id=session_id,
                        processed=False,
                        created_at=datetime.utcnow(),
                    )
                    session.execute(stmt)
                
                session.commit()
                logger.debug(f"Added message {message_id} to {self._messages_table_name}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to add message {message_id}: {e}")
    
    def get_unprocessed_messages(self, limit: int = 100) -> list[dict]:
        """Get messages that haven't been processed for memory extraction."""
        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.processed == False)
                .order_by(self.messages_table.c.timestamp.asc())
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
                for row in rows
            ]
    
    def mark_messages_processed(self, message_ids: list[str]) -> None:
        """Mark messages as processed for memory extraction."""
        if not message_ids:
            return
        
        with Session(self.engine) as session:
            try:
                stmt = (
                    update(self.messages_table)
                    .where(self.messages_table.c.id.in_(message_ids))
                    .values(processed=True)
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to mark messages processed: {e}")
    
    def get_messages_by_ids(self, message_ids: list[str]) -> list[dict]:
        """Retrieve messages by their IDs (for context retrieval)."""
        if not message_ids:
            return []
        
        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.id.in_(message_ids))
                .order_by(self.messages_table.c.timestamp.asc())
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
                for row in rows
            ]
    
    # =========================================================================
    # Memory Storage (distilled, importance-weighted)
    # =========================================================================
    
    def add_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "misc",
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Add a distilled memory to storage."""
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {memory_id}")
            return
        
        with self.engine.connect() as conn:
            try:
                # Check if exists
                check_sql = text(f"""
                    SELECT id FROM {self._memories_table_name} WHERE id = :id
                """)
                existing = conn.execute(check_sql, {"id": memory_id}).first()
                
                if existing:
                    # Update existing
                    if embedding:
                        update_sql = text(f"""
                            UPDATE {self._memories_table_name}
                            SET content = :content,
                                memory_type = :memory_type,
                                importance = :importance,
                                source_message_ids = :source_ids,
                                embedding = :embedding,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :id
                        """)
                        conn.execute(update_sql, {
                            "id": memory_id,
                            "content": content,
                            "memory_type": memory_type,
                            "importance": importance,
                            "source_ids": json.dumps(source_message_ids or []),
                            "embedding": str(embedding),
                        })
                    else:
                        update_sql = text(f"""
                            UPDATE {self._memories_table_name}
                            SET content = :content,
                                memory_type = :memory_type,
                                importance = :importance,
                                source_message_ids = :source_ids,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :id
                        """)
                        conn.execute(update_sql, {
                            "id": memory_id,
                            "content": content,
                            "memory_type": memory_type,
                            "importance": importance,
                            "source_ids": json.dumps(source_message_ids or []),
                        })
                else:
                    # Insert new
                    if embedding:
                        insert_sql = text(f"""
                            INSERT INTO {self._memories_table_name}
                            (id, content, memory_type, importance, source_message_ids, embedding, created_at)
                            VALUES (:id, :content, :memory_type, :importance, :source_ids, :embedding, CURRENT_TIMESTAMP)
                        """)
                        conn.execute(insert_sql, {
                            "id": memory_id,
                            "content": content,
                            "memory_type": memory_type,
                            "importance": importance,
                            "source_ids": json.dumps(source_message_ids or []),
                            "embedding": str(embedding),
                        })
                    else:
                        insert_sql = text(f"""
                            INSERT INTO {self._memories_table_name}
                            (id, content, memory_type, importance, source_message_ids, created_at)
                            VALUES (:id, :content, :memory_type, :importance, :source_ids, CURRENT_TIMESTAMP)
                        """)
                        conn.execute(insert_sql, {
                            "id": memory_id,
                            "content": content,
                            "memory_type": memory_type,
                            "importance": importance,
                            "source_ids": json.dumps(source_message_ids or []),
                        })
                
                conn.commit()
                logger.debug(f"Added memory {memory_id} to {self._memories_table_name}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to add memory {memory_id}: {e}")
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.memories_table).where(
                    self.memories_table.c.id == memory_id
                )
                result = session.execute(stmt)
                session.commit()
                return result.rowcount > 0
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False
    
    def update_memory_access(self, memory_id: str) -> None:
        """Update access tracking for a memory (reinforcement)."""
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    UPDATE {self._memories_table_name}
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = :id
                """)
                conn.execute(sql, {"id": memory_id})
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to update memory access: {e}")
    
    # =========================================================================
    # Search Methods
    # =========================================================================
    
    def search_memories_by_text(
        self,
        query: str,
        n_results: int = 5,
        min_importance: float = 0.0,
        memory_types: list[str] | None = None,
    ) -> list[dict]:
        """Search memories using PostgreSQL full-text search."""
        if not query or query.isspace():
            return []
        
        with self.engine.connect() as conn:
            try:
                # Build the query with optional filters
                type_filter = ""
                if memory_types:
                    type_list = ", ".join(f"'{t}'" for t in memory_types)
                    type_filter = f"AND memory_type IN ({type_list})"
                
                # Use PostgreSQL full-text search
                sql = text(f"""
                    SELECT id, content, memory_type, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) AS rank
                    FROM {self._memories_table_name}
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    AND importance >= :min_importance
                    {type_filter}
                    ORDER BY rank DESC, importance DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "query": query,
                    "min_importance": min_importance,
                    "limit": n_results,
                }).fetchall()
                
                results = []
                for row in rows:
                    # Update access tracking
                    self.update_memory_access(row.id)
                    
                    results.append({
                        "id": row.id,
                        "content": row.content,
                        "memory_type": row.memory_type,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": row.access_count,
                        "relevance": row.rank,
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return []
    
    def search_memories_by_embedding(
        self,
        embedding: list[float],
        n_results: int = 5,
        min_importance: float = 0.0,
        memory_types: list[str] | None = None,
    ) -> list[dict]:
        """Search memories using vector similarity (pgvector)."""
        if not embedding:
            return []
        
        with self.engine.connect() as conn:
            try:
                type_filter = ""
                if memory_types:
                    type_list = ", ".join(f"'{t}'" for t in memory_types)
                    type_filter = f"AND memory_type IN ({type_list})"
                
                # Use cosine distance for similarity
                sql = text(f"""
                    SELECT id, content, memory_type, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           1 - (embedding <=> :embedding) AS similarity
                    FROM {self._memories_table_name}
                    WHERE embedding IS NOT NULL
                    AND importance >= :min_importance
                    {type_filter}
                    ORDER BY embedding <=> :embedding
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "embedding": str(embedding),
                    "min_importance": min_importance,
                    "limit": n_results,
                }).fetchall()
                
                results = []
                for row in rows:
                    self.update_memory_access(row.id)
                    
                    results.append({
                        "id": row.id,
                        "content": row.content,
                        "memory_type": row.memory_type,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": row.access_count,
                        "similarity": row.similarity,
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search by embedding: {e}")
                return []
    
    # =========================================================================
    # MemoryBackend Interface Implementation
    # =========================================================================
    
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to storage (implements MemoryBackend interface).
        
        For backwards compatibility, this adds to the messages table.
        """
        self.add_message(message_id, role, content, timestamp)
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for relevant memories (implements MemoryBackend interface)."""
        results = self.search_memories_by_text(query, n_results, min_importance=min_relevance)
        
        if not results:
            return None
        
        # Format for backwards compatibility
        return [
            {
                "id": r["id"],
                "document": r["content"],
                "metadata": {
                    "memory_type": r["memory_type"],
                    "importance": r["importance"],
                    "source_message_ids": r["source_message_ids"],
                },
                "relevance": r.get("relevance", r["importance"]),
            }
            for r in results
        ]
    
    def clear(self) -> bool:
        """Clear all memories (NOT messages - those are permanent)."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.memories_table)
                session.execute(stmt)
                session.commit()
                logger.info(f"Cleared all memories from {self._memories_table_name}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear memories: {e}")
                return False
    
    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories."""
        with Session(self.engine) as session:
            stmt = (
                select(self.memories_table)
                .order_by(self.memories_table.c.created_at.desc())
                .limit(n)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "document": row.content,
                    "metadata": {
                        "memory_type": row.memory_type,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                    },
                }
                for row in rows
            ]
    
    def stats(self) -> dict:
        """Get statistics about memory storage."""
        with self.engine.connect() as conn:
            try:
                # Memory stats
                mem_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest,
                        AVG(importance) as avg_importance,
                        pg_total_relation_size('{self._memories_table_name}') as size_bytes
                    FROM {self._memories_table_name}
                """)
                mem_row = conn.execute(mem_stats_sql).first()
                
                # Message stats
                msg_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN processed THEN 1 ELSE 0 END) as processed,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM {self._messages_table_name}
                """)
                msg_row = conn.execute(msg_stats_sql).first()
                
                return {
                    "memories": {
                        "total_count": mem_row.total if mem_row else 0,
                        "oldest_timestamp": mem_row.oldest.isoformat() if mem_row and mem_row.oldest else None,
                        "newest_timestamp": mem_row.newest.isoformat() if mem_row and mem_row.newest else None,
                        "avg_importance": float(mem_row.avg_importance) if mem_row and mem_row.avg_importance else 0.0,
                        "storage_size_bytes": mem_row.size_bytes if mem_row else 0,
                    },
                    "messages": {
                        "total_count": msg_row.total if msg_row else 0,
                        "processed_count": msg_row.processed if msg_row else 0,
                        "oldest_timestamp": msg_row.oldest.isoformat() if msg_row and msg_row.oldest else None,
                        "newest_timestamp": msg_row.newest.isoformat() if msg_row and msg_row.newest else None,
                    },
                }
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {"error": str(e)}
    
    # =========================================================================
    # Short-term Memory Manager (for session history)
    # =========================================================================
    
    @classmethod
    def get_short_term_manager(cls, config: Any, bot_id: str = "nova") -> "PostgreSQLShortTermManager":
        """Factory method to get a short-term memory manager."""
        return PostgreSQLShortTermManager(config, bot_id)


class PostgreSQLShortTermManager:
    """Manages short-term (session) conversation history using PostgreSQL.
    
    This stores messages permanently
    but provides session-scoped access for building context windows.
    """
    
    def __init__(self, config: Any, bot_id: str = "nova"):
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
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
            )
            return session.execute(stmt).scalar() or 0

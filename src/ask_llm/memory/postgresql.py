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
import time
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
        Column("superseded_by", String(36), nullable=True),  # ID of memory that replaced this one
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
    
    # Embedding dimension (matches sentence-transformers all-MiniLM-L6-v2)
    EMBEDDING_DIM = 384
    
    def __init__(self, config: Any, bot_id: str = "nova", embedding_dim: int | None = None):
        super().__init__(config, bot_id=bot_id)
        
        # Get embedding settings from config
        self.embedding_model = getattr(config, 'MEMORY_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        if embedding_dim is None:
            embedding_dim = getattr(config, 'MEMORY_EMBEDDING_DIM', 384)
        
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
                    superseded_by VARCHAR(36),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding vector({self.embedding_dim})
                )
            """)
            
            # Add superseded_by column if it doesn't exist (migration)
            add_superseded_sql = text(f"""
                ALTER TABLE {self._memories_table_name} 
                ADD COLUMN IF NOT EXISTS superseded_by VARCHAR(36)
            """)
            
            try:
                conn.execute(messages_sql)
                conn.execute(memories_sql)
                # Run migration for existing tables
                try:
                    conn.execute(add_superseded_sql)
                except Exception:
                    pass  # Column may already exist
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
        """Add a distilled memory to storage.
        
        If no embedding is provided, one will be generated automatically
        using the configured local embedding model.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {memory_id}")
            return
        
        # Generate embedding if not provided
        if embedding is None:
            try:
                from .embeddings import generate_embedding
                embedding = generate_embedding(content, self.embedding_model)
                if embedding:
                    logger.debug(f"Generated embedding for memory: {content[:50]}...")
            except Exception as e:
                logger.debug(f"Could not generate embedding: {e}")
        
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
    
    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        """Mark an old memory as superseded by a new one.
        
        This preserves history while ensuring the old memory won't be retrieved.
        Use this instead of delete when a fact changes (e.g., user moved cities).
        
        Args:
            old_memory_id: ID of the memory being superseded
            new_memory_id: ID of the new memory that replaces it
            
        Returns:
            True if successful
        """
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    UPDATE {self._memories_table_name}
                    SET superseded_by = :new_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :old_id
                """)
                conn.execute(sql, {"old_id": old_memory_id, "new_id": new_memory_id})
                conn.commit()
                logger.debug(f"Memory {old_memory_id} superseded by {new_memory_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to supersede memory: {e}")
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
        """Search memories using vector similarity with temporal decay and diversity.
        
        The effective score combines:
        - Semantic similarity (cosine distance)
        - Base importance
        - Temporal decay (memories fade over time)
        - Access boost (frequently accessed memories get reinforced)
        - Diversity sampling (avoid echo chambers by sampling across time/types)
        """
        if not embedding:
            return []
        
        # Get decay settings from config
        decay_enabled = getattr(self.config, 'MEMORY_DECAY_ENABLED', True)
        half_life_days = getattr(self.config, 'MEMORY_DECAY_HALF_LIFE_DAYS', 90.0)
        access_boost_factor = getattr(self.config, 'MEMORY_ACCESS_BOOST_FACTOR', 0.15)
        recency_weight = getattr(self.config, 'MEMORY_RECENCY_WEIGHT', 0.3)
        diversity_enabled = getattr(self.config, 'MEMORY_DIVERSITY_ENABLED', True)
        
        # Different decay rates per memory type (multiplier on half_life)
        # Higher = slower decay (more persistent)
        type_decay_multipliers = {
            'fact': 2.0,          # Core facts persist longer
            'professional': 1.5,  # Career info moderately persistent
            'preference': 0.8,    # Preferences change
            'health': 1.2,        # Health info somewhat persistent
            'relationship': 1.0,  # Relationships change at normal rate
            'event': 0.5,         # Events become less relevant quickly
            'plan': 0.3,          # Plans/goals are very temporal
            'misc': 1.0,          # Default rate
        }
        
        with self.engine.connect() as conn:
            try:
                type_filter = ""
                if memory_types:
                    type_list = ", ".join(f"'{t}'" for t in memory_types)
                    type_filter = f"AND memory_type IN ({type_list})"
                
                if decay_enabled:
                    # Fetch more candidates for post-processing with decay + diversity
                    fetch_limit = n_results * 4 if diversity_enabled else n_results * 2
                    
                    sql = text(f"""
                        SELECT id, content, memory_type, importance, source_message_ids,
                               access_count, last_accessed, created_at, superseded_by,
                               1 - (embedding <=> :embedding) AS similarity,
                               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0 AS age_days
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        AND superseded_by IS NULL
                        {type_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                else:
                    # Simple similarity-based search without decay
                    sql = text(f"""
                        SELECT id, content, memory_type, importance, source_message_ids,
                               access_count, last_accessed, created_at, superseded_by,
                               1 - (embedding <=> :embedding) AS similarity,
                               0 AS age_days
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        AND superseded_by IS NULL
                        {type_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                    fetch_limit = n_results
                
                rows = conn.execute(sql, {
                    "embedding": str(embedding),
                    "min_importance": min_importance,
                    "limit": fetch_limit,
                }).fetchall()
                
                if not rows:
                    return []
                
                # Calculate effective scores with decay
                import math
                scored_results = []
                
                for row in rows:
                    # Convert Decimal types to float for calculations
                    similarity = float(row.similarity) if row.similarity else 0.0
                    importance = float(row.importance) if row.importance else 0.5
                    age_days = float(row.age_days) if row.age_days else 0.0
                    access_count = int(row.access_count) if row.access_count else 0
                    
                    # Get type-specific half-life
                    type_multiplier = type_decay_multipliers.get(row.memory_type, 1.0)
                    effective_half_life = half_life_days * type_multiplier
                    
                    # Temporal decay: exp(-age * ln(2) / half_life)
                    if decay_enabled and age_days > 0:
                        decay_factor = math.exp(-age_days * math.log(2) / effective_half_life)
                    else:
                        decay_factor = 1.0
                    
                    # Access boost: 1 + factor * log(access_count + 1)
                    access_boost = 1.0 + access_boost_factor * math.log(access_count + 1)
                    
                    # Combined score:
                    # similarity provides base relevance
                    # importance is the extracted importance
                    # decay_factor reduces old memories
                    # access_boost reinforces frequently used ones
                    # recency_weight balances recency vs base importance
                    base_score = similarity * importance
                    recency_score = similarity * decay_factor
                    effective_score = (
                        (1 - recency_weight) * base_score + 
                        recency_weight * recency_score
                    ) * access_boost
                    
                    scored_results.append({
                        "id": row.id,
                        "content": row.content,
                        "memory_type": row.memory_type,
                        "importance": importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": access_count,
                        "similarity": similarity,
                        "age_days": age_days,
                        "decay_factor": decay_factor,
                        "effective_score": effective_score,
                    })
                
                # Sort by effective score
                scored_results.sort(key=lambda x: x["effective_score"], reverse=True)
                
                # Apply diversity sampling if enabled
                if diversity_enabled and len(scored_results) > n_results:
                    results = self._diversity_sample(scored_results, n_results)
                else:
                    results = scored_results[:n_results]
                
                # Update access counts for retrieved memories
                for r in results:
                    self.update_memory_access(r["id"])
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search by embedding: {e}")
                return []
    
    def _diversity_sample(self, candidates: list[dict], n_results: int) -> list[dict]:
        """Sample diverse memories to avoid echo chambers.
        
        Strategy:
        1. Always include top-scoring result
        2. Ensure representation from different memory types
        3. Ensure representation from different time periods
        4. Fill remaining slots by score
        """
        if len(candidates) <= n_results:
            return candidates
        
        selected = []
        used_ids = set()
        
        # 1. Always take the top result
        selected.append(candidates[0])
        used_ids.add(candidates[0]["id"])
        
        # 2. Ensure type diversity - try to get one from each represented type
        types_seen = {candidates[0]["memory_type"]}
        for candidate in candidates[1:]:
            if len(selected) >= n_results:
                break
            if candidate["memory_type"] not in types_seen and candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
                types_seen.add(candidate["memory_type"])
        
        # 3. Ensure temporal diversity - split into time buckets
        if len(selected) < n_results:
            # Recent (< 7 days), Medium (7-30 days), Old (30+ days)
            buckets = {"recent": [], "medium": [], "old": []}
            for candidate in candidates:
                if candidate["id"] in used_ids:
                    continue
                age = candidate.get("age_days", 0)
                if age < 7:
                    buckets["recent"].append(candidate)
                elif age < 30:
                    buckets["medium"].append(candidate)
                else:
                    buckets["old"].append(candidate)
            
            # Try to get one from each bucket we haven't covered
            for bucket_name in ["medium", "old", "recent"]:  # Prioritize less-recent
                if len(selected) >= n_results:
                    break
                bucket = buckets[bucket_name]
                if bucket:
                    candidate = bucket[0]
                    selected.append(candidate)
                    used_ids.add(candidate["id"])
        
        # 4. Fill remaining by score
        for candidate in candidates:
            if len(selected) >= n_results:
                break
            if candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
        
        # Re-sort by effective_score for consistent ordering
        selected.sort(key=lambda x: x["effective_score"], reverse=True)
        
        return selected
    
    # =========================================================================
    # MemoryBackend Interface Implementation
    # =========================================================================
    
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to storage (implements MemoryBackend interface).
        
        For backwards compatibility, this adds to the messages table.
        """
        self.add_message(message_id, role, content, timestamp)
    
    def search_messages_by_text(
        self,
        query: str,
        n_results: int = 5,
        exclude_recent_seconds: float = 5.0,
        role_filter: str | None = "user",
    ) -> list[dict]:
        """Search raw messages using PostgreSQL full-text search.
        
        This is a fallback when no distilled memories exist yet.
        Uses OR logic so any matching word will return results.
        
        Args:
            query: Search query
            n_results: Max number of results
            exclude_recent_seconds: Exclude messages from the last N seconds to avoid
                                   finding the query message itself
            role_filter: Only include messages with this role (default: "user" to avoid
                        retrieving assistant hallucinations as facts). Set to None to 
                        include all roles.
        """
        if not query or query.isspace():
            return []
        
        cutoff_time = time.time() - exclude_recent_seconds
        
        with self.engine.connect() as conn:
            try:
                # Use websearch_to_tsquery for better handling, but we need OR logic
                # Extract words and build an OR query manually
                # Filter out common words AND conversational meta-words that don't help find content
                stop_words = {
                    # Standard English stop words
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                    'we', 'you', 'i', 'he', 'she', 'it', 'they', 'them', 'their', 'our',
                    'my', 'your', 'his', 'her', 'its', 'what', 'which', 'who', 'whom',
                    'this', 'that', 'these', 'those', 'am', 'not', 'no', 'yes', 'so',
                    'if', 'then', 'than', 'too', 'very', 'just', 'about', 'before', 'after',
                    # Conversational meta-words (common in memory queries but not content)
                    'remember', 'tell', 'told', 'said', 'say', 'know', 'think', 'thought',
                    'talk', 'talked', 'talking', 'conversation', 'conversations', 'discussed',
                    'discuss', 'discussion', 'mention', 'mentioned', 'anything', 'something',
                    'everything', 'nothing', 'past', 'previous', 'earlier', 'last', 'time',
                    'when', 'where', 'how', 'why', 'like', 'want', 'wanted', 'please',
                }
                
                # Extract meaningful words
                import re
                words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
                meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
                
                if not meaningful_words:
                    # Fall back to simple search if no meaningful words
                    meaningful_words = [w for w in words if len(w) > 2][:3]
                
                if not meaningful_words:
                    return []
                
                # Build OR query: word1 | word2 | word3
                or_query = ' | '.join(meaningful_words)
                
                # Build role filter clause
                role_clause = "AND role = :role" if role_filter else ""
                
                sql = text(f"""
                    SELECT id, role, content, timestamp,
                           ts_rank(to_tsvector('english', content), to_tsquery('english', :query)) AS rank
                    FROM {self._messages_table_name}
                    WHERE to_tsvector('english', content) @@ to_tsquery('english', :query)
                    AND timestamp < :cutoff
                    {role_clause}
                    ORDER BY rank DESC, timestamp DESC
                    LIMIT :limit
                """)
                
                params = {
                    "query": or_query,
                    "limit": n_results,
                    "cutoff": cutoff_time,
                }
                if role_filter:
                    params["role"] = role_filter
                
                rows = conn.execute(sql, params).fetchall()
                
                return [
                    {
                        "id": row.id,
                        "content": row.content,
                        "role": row.role,
                        "timestamp": row.timestamp,
                        "relevance": row.rank,
                    }
                    for row in rows
                ]
                
            except Exception as e:
                logger.error(f"Failed to search messages: {e}")
                return []
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for relevant memories (implements MemoryBackend interface).
        
        Uses text-based search for memory retrieval. Semantic embedding search
        is only available when running through the background service (--service mode).
        """
        # Use text search (fast, no model loading required)
        text_results = self.search_memories_by_text(query, n_results, min_importance=min_relevance)
        if text_results:
            logger.debug(f"Found {len(text_results)} memories via text search")
            return [
                {
                    "id": r["id"],
                    "document": r["content"],
                    "metadata": {
                        "memory_type": r["memory_type"],
                        "importance": r["importance"],
                        "source_message_ids": r["source_message_ids"],
                    },
                    "relevance": r.get("similarity", r.get("relevance", r["importance"])),
                }
                for r in text_results
            ]
        
        # Fallback: search raw messages if no memories exist
        # Exclude messages from the last 10 seconds to avoid finding the current query
        message_results = self.search_messages_by_text(query, n_results, exclude_recent_seconds=10.0)
        
        if not message_results:
            return None
        
        logger.debug(f"Falling back to {len(message_results)} message results")
        
        # Format message results like memory results
        return [
            {
                "id": r["id"],
                "document": r["content"],
                "metadata": {
                    "role": r["role"],
                    "timestamp": r["timestamp"],
                },
                "relevance": r.get("relevance", 0.5),
            }
            for r in message_results
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
    
    def regenerate_embeddings(self, batch_size: int = 50) -> dict:
        """Regenerate embeddings for all memories that don't have them.
        
        This is useful after installing sentence-transformers or when
        switching embedding models. Also handles dimension migration.
        
        Args:
            batch_size: Number of memories to process at once
            
        Returns:
            Dict with counts of updated and failed memories
        """
        try:
            from .embeddings import generate_embeddings_batch, get_embedding_dimension
        except ImportError:
            return {"error": "sentence-transformers not installed", "updated": 0, "failed": 0}
        
        # Check if we need to alter the column dimension
        actual_dim = get_embedding_dimension(self.embedding_model)
        
        updated = 0
        failed = 0
        
        with self.engine.connect() as conn:
            # First, check and alter column dimension if needed
            try:
                check_dim_sql = text(f"""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = '{self._memories_table_name}'::regclass 
                    AND attname = 'embedding'
                """)
                result = conn.execute(check_dim_sql).first()
                if result:
                    current_dim = result[0]
                    if current_dim != actual_dim and current_dim > 0:
                        logger.info(f"Migrating embedding dimension from {current_dim} to {actual_dim}")
                        # Drop index, alter column, recreate index
                        conn.execute(text(f"DROP INDEX IF EXISTS idx_{self._memories_table_name}_embedding"))
                        conn.execute(text(f"ALTER TABLE {self._memories_table_name} ALTER COLUMN embedding TYPE vector({actual_dim})"))
                        conn.execute(text(f"""
                            CREATE INDEX idx_{self._memories_table_name}_embedding 
                            ON {self._memories_table_name} 
                            USING hnsw (embedding vector_cosine_ops)
                        """))
                        # Clear existing embeddings since they're wrong dimension
                        conn.execute(text(f"UPDATE {self._memories_table_name} SET embedding = NULL"))
                        conn.commit()
            except Exception as e:
                logger.debug(f"Could not check/alter embedding dimension: {e}")
            
            # Get all memories without embeddings
            fetch_sql = text(f"""
                SELECT id, content 
                FROM {self._memories_table_name}
                WHERE embedding IS NULL
                ORDER BY importance DESC
                LIMIT :limit
            """)
            
            while True:
                rows = conn.execute(fetch_sql, {"limit": batch_size}).fetchall()
                if not rows:
                    break
                
                # Extract texts and IDs
                ids = [row.id for row in rows]
                texts = [row.content for row in rows]
                
                # Generate embeddings in batch
                embeddings = generate_embeddings_batch(texts, self.embedding_model)
                
                # Update each memory
                for mem_id, embedding in zip(ids, embeddings):
                    if embedding:
                        try:
                            update_sql = text(f"""
                                UPDATE {self._memories_table_name}
                                SET embedding = :embedding
                                WHERE id = :id
                            """)
                            conn.execute(update_sql, {
                                "id": mem_id,
                                "embedding": str(embedding),
                            })
                            updated += 1
                        except Exception as e:
                            logger.error(f"Failed to update embedding for {mem_id}: {e}")
                            failed += 1
                    else:
                        failed += 1
                
                conn.commit()
                logger.info(f"Regenerated embeddings: {updated} updated, {failed} failed")
        
        return {"updated": updated, "failed": failed, "embedding_dim": actual_dim}
    
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
            stmt = select(self._backend.messages_table).order_by(
                self._backend.messages_table.c.timestamp.asc()
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

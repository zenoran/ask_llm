"""MariaDB memory backend for ask_llm.

This backend stores conversation memories in MariaDB using fulltext search
for semantic retrieval. It's lightweight and doesn't require heavy ML dependencies.
"""

import logging
import time
from typing import Any

from .base import MemoryBackend

logger = logging.getLogger(__name__)


class MariaDBMemoryBackend(MemoryBackend):
    """MariaDB-based memory backend using fulltext search.
    
    Stores messages in a MariaDB table with fulltext indexing for
    relevance-based retrieval. Requires mysql-connector-python.
    
    Configuration (via environment variables or .env):
        ASK_LLM_MARIADB_HOST: Database host (default: localhost)
        ASK_LLM_MARIADB_PORT: Database port (default: 3306)
        ASK_LLM_MARIADB_USER: Database user
        ASK_LLM_MARIADB_PASSWORD: Database password
        ASK_LLM_MARIADB_DATABASE: Database name (default: ask_llm)
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Get MariaDB connection settings from config or environment
        self.host = getattr(config, 'MARIADB_HOST', 'localhost')
        self.port = int(getattr(config, 'MARIADB_PORT', 3306))
        self.user = getattr(config, 'MARIADB_USER', 'ask_llm')
        self.password = getattr(config, 'MARIADB_PASSWORD', '')
        self.database = getattr(config, 'MARIADB_DATABASE', 'ask_llm')
        
        self._connection = None
        self._ensure_table_exists()
    
    def _get_connection(self):
        """Get or create a database connection."""
        if self._connection is None or not self._connection.is_connected():
            try:
                import mysql.connector
                self._connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    autocommit=True
                )
                logger.debug(f"Connected to MariaDB at {self.host}:{self.port}/{self.database}")
            except Exception as e:
                logger.error(f"Failed to connect to MariaDB: {e}")
                raise
        return self._connection
    
    def _ensure_table_exists(self):
        """Create the memories table if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id VARCHAR(36) PRIMARY KEY,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FULLTEXT INDEX ft_content (content)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            logger.debug("Ensured memories table exists")
        except Exception as e:
            logger.error(f"Failed to create memories table: {e}")
            raise
        finally:
            cursor.close()
    
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to memory storage."""
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {message_id}")
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO memories (id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
                (message_id, role, content, timestamp)
            )
            logger.debug(f"Added memory ID: {message_id}")
        except Exception as e:
            logger.error(f"Failed to add memory {message_id}: {e}")
        finally:
            cursor.close()
    
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for memories relevant to the query using fulltext search.
        
        Args:
            query: The text to search for.
            n_results: Maximum number of results to return.
            min_relevance: Minimum relevance score (0.0-1.0) to include.
        """
        if not query or query.isspace():
            logger.warning("Attempted to search with empty query")
            return None
            
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # Use MATCH AGAINST for fulltext search with relevance scoring
            cursor.execute("""
                SELECT id, role, content, timestamp,
                       MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) AS relevance
                FROM memories
                WHERE MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)
                ORDER BY relevance DESC
                LIMIT %s
            """, (query, query, n_results))
            
            results = cursor.fetchall()
            
            if not results:
                # Fallback to LIKE search if fulltext returns nothing
                search_terms = query.split()[:5]  # Use first 5 words
                like_pattern = '%' + '%'.join(search_terms) + '%'
                cursor.execute("""
                    SELECT id, role, content, timestamp, 0.5 AS relevance
                    FROM memories
                    WHERE content LIKE %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (like_pattern, n_results))
                results = cursor.fetchall()
            
            if not results:
                return None
            
            # Normalize relevance scores to 0.0-1.0 range
            # MariaDB FULLTEXT relevance varies based on content, normalize using max
            max_relevance = max(row.get('relevance', 0) for row in results) or 1.0
                
            # Format results to match expected interface
            formatted = []
            for row in results:
                raw_relevance = row.get('relevance', 0)
                # Normalize to 0.0-1.0 (higher is more relevant)
                normalized_relevance = raw_relevance / max_relevance if max_relevance > 0 else 0.5
                
                # Filter by min_relevance threshold
                if normalized_relevance < min_relevance:
                    continue
                    
                formatted.append({
                    'id': row['id'],
                    'document': row['content'],
                    'metadata': {
                        'role': row['role'],
                        'timestamp': row['timestamp']
                    },
                    'relevance': normalized_relevance
                })
            
            return formatted if formatted else None
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return None
        finally:
            cursor.close()
    
    def clear(self) -> bool:
        """Clear all memories from storage."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM memories")
            logger.info("Cleared all memories")
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
        finally:
            cursor.close()
    
    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT id, role, content, timestamp
                FROM memories
                ORDER BY timestamp DESC
                LIMIT %s
            """, (n,))
            
            results = cursor.fetchall()
            formatted = []
            for row in results:
                formatted.append({
                    'id': row['id'],
                    'document': row['content'],
                    'metadata': {
                        'role': row['role'],
                        'timestamp': row['timestamp']
                    }
                })
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to list recent memories: {e}")
            return []
        finally:
            cursor.close()
    
    def stats(self) -> dict:
        """Get statistics about the memory storage."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    MIN(timestamp) as oldest_timestamp,
                    MAX(timestamp) as newest_timestamp
                FROM memories
            """)
            row = cursor.fetchone()
            
            # Get table size
            cursor.execute("""
                SELECT 
                    data_length + index_length as storage_size
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'memories'
            """, (self.database,))
            size_row = cursor.fetchone()
            
            return {
                'total_count': row['total_count'] or 0,
                'oldest_timestamp': row['oldest_timestamp'],
                'newest_timestamp': row['newest_timestamp'],
                'storage_size': size_row['storage_size'] if size_row else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'total_count': 0, 'error': str(e)}
        finally:
            cursor.close()
    
    def delete(self, message_id: str) -> bool:
        """Delete a specific memory by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM memories WHERE id = %s", (message_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted memory ID: {message_id}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete memory {message_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def prune_older_than(self, days: int) -> int:
        """Delete memories older than a specified number of days."""
        cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM memories WHERE timestamp < %s",
                (cutoff_timestamp,)
            )
            deleted_count = cursor.rowcount
            logger.info(f"Pruned {deleted_count} memories older than {days} days")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to prune memories: {e}")
            return 0
        finally:
            cursor.close()
    
    def __del__(self):
        """Close the database connection on cleanup."""
        if self._connection and self._connection.is_connected():
            self._connection.close()

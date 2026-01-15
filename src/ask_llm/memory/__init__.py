"""Memory backend interface for ask_llm.

This module defines the abstract interface for memory backends.
Memory backends provide semantic search and storage for conversation history,
allowing the LLM to recall relevant past conversations.

To implement a custom memory backend:
1. Create a class that inherits from MemoryBackend
2. Implement all abstract methods
3. Register your backend as an entry point in your package's pyproject.toml:

    [project.entry-points."ask_llm.memory"]
    your_backend_name = "your_package.module:YourBackendClass"

Included backends:
- PostgreSQLMemoryBackend: Uses PostgreSQL with pgvector for semantic search
- PostgreSQLShortTermManager: Session-scoped message history using PostgreSQL
"""

from .base import MemoryBackend

# Lazy import for PostgreSQL backend (requires psycopg2)
def get_postgresql_backend():
    from .postgresql import PostgreSQLMemoryBackend
    return PostgreSQLMemoryBackend


def get_short_term_manager():
    from .short_term import PostgreSQLShortTermManager
    return PostgreSQLShortTermManager


__all__ = ["MemoryBackend", "get_postgresql_backend", "get_short_term_manager"]

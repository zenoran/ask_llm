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

Example backend implementations:
- MariaDB with fulltext search
- PostgreSQL with pgvector
- SQLite with FTS5
- Redis with vector similarity
"""

from .base import MemoryBackend

__all__ = ["MemoryBackend"]

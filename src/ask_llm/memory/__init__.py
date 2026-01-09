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
- MariaDBMemoryBackend: Uses MariaDB with fulltext search (requires mysql-connector-python)
"""

from .base import MemoryBackend

# Lazy import for MariaDB backend (requires mysql-connector-python)
def get_mariadb_backend():
    from .mariadb import MariaDBMemoryBackend
    return MariaDBMemoryBackend

__all__ = ["MemoryBackend", "get_mariadb_backend"]

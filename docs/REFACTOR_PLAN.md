# ask_llm Refactoring Plan: MCP-Based Architecture

> **Created**: January 13, 2026  
> **Branch**: `refactor/mcp-architecture`  
> **Goal**: Transform ask_llm into a modular, MCP-based architecture with GGUF as the primary LLM backend

---

## Executive Summary

This plan outlines a ground-up refactoring of ask_llm to:
1. Convert the memory system to an **MCP server** with memory tools
2. Create a clean separation between **llm-client** (CLI), **llm-server** (backend API), and **llm-memory** (MCP server)
3. Focus on **GGUF/llama-cpp-python** as the primary LLM backend with OpenAI-compatible API patterns for future extensibility
4. Embed tools directly in the MCP server code using the Python MCP SDK (`FastMCP`)

---

## Current Architecture Analysis

### What Exists Today

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   argparse  │  │  Bot System │  │   User Profile Manager  │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AskLLM (core.py)                       │  │
│  │  - Orchestrates clients, memory, history                  │  │
│  │  - Model resolution                                       │  │
│  │  - Query execution                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│              │                    │                    │        │
│              ▼                    ▼                    ▼        │
│  ┌───────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   LLM Clients     │  │ History Manager│  │ Memory Backend │ │
│  │ - OpenAI          │  │ (utils/history)│  │ (postgresql.py)│ │
│  │ - Ollama          │  └────────────────┘  │ - Messages     │ │
│  │ - LlamaCpp (GGUF) │                      │ - Memories     │ │
│  │ - HuggingFace     │                      │ - Embeddings   │ │
│  └───────────────────┘                      │ - Extraction   │ │
│                                             └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Optional: FastAPI Service (service/)               │
│              - Background task processing                       │
│              - SSE streaming                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Pain Points Identified

1. **Tight Coupling**: Memory, history, and LLM clients are tightly coupled through `AskLLM`
2. **Complex Initialization**: `core.py` has too many responsibilities
3. **Memory Complexity**: Memory extraction is embedded within the PostgreSQL backend
4. **Multiple LLM Clients**: Supporting 4 different clients creates maintenance burden
5. **Service Layer**: Current FastAPI service is a bolted-on addition, not core architecture
6. **Configuration Sprawl**: Config scattered across multiple sources

---

## Target Architecture

### Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         llm-client (CLI)                               │
│  - argparse-based CLI                                                  │
│  - Rich console output                                                 │
│  - Connects to llm-server via HTTP or directly to GGUF models         │
│  - Bot personality selection                                           │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
     ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
     │ Direct GGUF  │    │   llm-server (API)   │    │ OpenAI-compat   │
     │ Local Mode   │    │  - FastAPI/Starlette │    │ (Future)        │
     │ (--local)    │    │  - Streamable HTTP   │    │                 │
     └──────────────┘    │  - Session mgmt      │    └──────────────────┘
                         │  - MCP client        │
                         └──────────┬───────────┘
                                    │
                                    │ MCP Protocol (Streamable HTTP)
                                    │
                         ┌──────────▼───────────┐
                         │  llm-memory (MCP)    │
                         │  - FastMCP server    │
                         │  - Memory tools      │
                         │  - PostgreSQL+pgvec  │
                         │  - Embeddings        │
                         │  - Fact extraction   │
                         └──────────────────────┘
```

### Component Responsibilities

#### 1. `llm-client` (CLI Frontend)
**Purpose**: User-facing CLI for interacting with LLMs

**Responsibilities**:
- Command-line argument parsing
- Rich terminal output formatting
- Bot/personality selection
- Direct GGUF model loading for local mode
- HTTP client to llm-server for networked mode
- Session/conversation management (client-side)

**Key Files**:
- `src/ask_llm/cli.py` - Main CLI entry point
- `src/ask_llm/client/__init__.py` - Client-side logic
- `src/ask_llm/client/local.py` - Direct GGUF execution
- `src/ask_llm/client/remote.py` - HTTP client to llm-server
- `src/ask_llm/bots.py` - Bot configuration (retained)

#### 2. `llm-server` (Backend API)
**Purpose**: Centralized LLM inference and MCP client

**Responsibilities**:
- OpenAI-compatible chat completion API
- GGUF model loading and management
- MCP client to connect to llm-memory server
- Context building (inject memory into prompts)
- Streaming response handling
- Model management (list, load, unload)

**Key Files**:
- `src/ask_llm/server/__init__.py`
- `src/ask_llm/server/app.py` - FastMCP-based server
- `src/ask_llm/server/llm.py` - LLM inference wrapper (GGUF-focused)
- `src/ask_llm/server/mcp_client.py` - MCP client to memory server
- `src/ask_llm/server/context.py` - Context building from memory

#### 3. `llm-memory` (MCP Server)
**Purpose**: Memory management via MCP protocol

**Responsibilities**:
- MCP server exposing memory tools
- PostgreSQL + pgvector storage
- Semantic search with embeddings
- Memory extraction (distillation from conversations)
- User profile management
- Message history storage

**MCP Tools** (embedded in code via FastMCP decorators):
- `store_memory` - Store a new memory/fact
- `search_memories` - Semantic search across memories
- `get_recent_context` - Get recent conversation context
- `add_message` - Add a message to history
- `extract_facts` - Extract facts from conversation (LLM-based)
- `update_memory` - Update existing memory
- `delete_memory` - Remove a memory
- `get_user_profile` - Get user profile context
- `update_user_profile` - Update user profile

**Key Files**:
- `src/ask_llm/memory_server/__init__.py`
- `src/ask_llm/memory_server/server.py` - FastMCP server with tools
- `src/ask_llm/memory_server/storage.py` - PostgreSQL backend
- `src/ask_llm/memory_server/embeddings.py` - Embedding generation
- `src/ask_llm/memory_server/extraction.py` - Fact extraction logic

---

## MCP Server Design: llm-memory

### Tool Definitions

```python
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Optional
import uuid

# Initialize the MCP server
mcp = FastMCP(
    "llm-memory",
    json_response=True,
    stateless_http=True,  # Recommended for production
)

# ============== DATA MODELS ==============

class Memory(BaseModel):
    """A distilled memory/fact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="The memory content")
    tags: list[str] = Field(default=["misc"], description="Categorization tags")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_message_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class SearchResult(BaseModel):
    """A memory search result with relevance score."""
    memory: Memory
    relevance: float = Field(description="Similarity score 0-1")

class Message(BaseModel):
    """A conversation message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(description="user, assistant, or system")
    content: str
    timestamp: float = Field(default_factory=lambda: time.time())
    bot_id: str = Field(default="default")
    session_id: Optional[str] = None

# ============== MEMORY TOOLS ==============

@mcp.tool()
async def store_memory(
    content: str,
    tags: list[str] = ["misc"],
    importance: float = 0.5,
    source_message_ids: list[str] = [],
    bot_id: str = "default",
    ctx: Context = None,
) -> Memory:
    """Store a new memory/fact in the knowledge base.
    
    Args:
        content: The memory content to store
        tags: Categorization tags (identity, preference, relationship, etc.)
        importance: Importance score from 0.0 to 1.0
        source_message_ids: IDs of messages this memory was derived from
        bot_id: Bot namespace for isolation
    
    Returns:
        The stored Memory object with generated ID
    """
    # Implementation: Generate embedding, store in PostgreSQL
    ...

@mcp.tool()
async def search_memories(
    query: str,
    bot_id: str = "default",
    n_results: int = 10,
    min_relevance: float = 0.5,
    tags: list[str] | None = None,
) -> list[SearchResult]:
    """Search memories using semantic similarity.
    
    Args:
        query: Natural language search query
        bot_id: Bot namespace for isolation
        n_results: Maximum number of results
        min_relevance: Minimum similarity threshold (0-1)
        tags: Optional filter by specific tags
    
    Returns:
        List of SearchResult objects with memories and relevance scores
    """
    # Implementation: Generate query embedding, vector search
    ...

@mcp.tool()
async def get_recent_context(
    bot_id: str = "default",
    user_id: str = "default",
    max_messages: int = 20,
    max_age_seconds: int = 3600,
) -> list[Message]:
    """Get recent conversation context from message history.
    
    Args:
        bot_id: Bot namespace
        user_id: User for profile context
        max_messages: Maximum messages to return
        max_age_seconds: Only include messages within this time window
    
    Returns:
        List of recent Message objects
    """
    ...

@mcp.tool()
async def add_message(
    role: str,
    content: str,
    bot_id: str = "default",
    session_id: str | None = None,
) -> Message:
    """Add a message to the conversation history.
    
    Args:
        role: Message role (user, assistant, system)
        content: Message content
        bot_id: Bot namespace
        session_id: Optional session grouping
    
    Returns:
        The stored Message object
    """
    ...

@mcp.tool()
async def extract_facts(
    messages: list[dict],
    bot_id: str = "default",
    store: bool = True,
) -> list[Memory]:
    """Extract important facts from conversation messages.
    
    Uses LLM-based extraction to identify discrete facts worth remembering.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        bot_id: Bot namespace
        store: Whether to store extracted facts as memories
    
    Returns:
        List of extracted Memory objects
    """
    # Implementation: LLM-based fact extraction (current extraction/service.py logic)
    ...

@mcp.tool()
async def update_memory(
    memory_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
    bot_id: str = "default",
) -> Memory:
    """Update an existing memory.
    
    Args:
        memory_id: ID of memory to update
        content: New content (optional)
        tags: New tags (optional)
        importance: New importance score (optional)
        bot_id: Bot namespace
    
    Returns:
        Updated Memory object
    """
    ...

@mcp.tool()
async def delete_memory(
    memory_id: str,
    bot_id: str = "default",
    soft_delete: bool = True,
) -> bool:
    """Delete a memory.
    
    Args:
        memory_id: ID of memory to delete
        bot_id: Bot namespace
        soft_delete: If True, mark as forgotten instead of hard delete
    
    Returns:
        True if deleted successfully
    """
    ...

# ============== USER PROFILE TOOLS ==============

@mcp.tool()
async def get_user_profile(
    user_id: str = "default",
) -> dict:
    """Get user profile information.
    
    Returns:
        User profile dict with name, preferences, context
    """
    ...

@mcp.tool()
async def update_user_profile(
    user_id: str = "default",
    name: str | None = None,
    preferred_name: str | None = None,
    preferences: dict | None = None,
    context: dict | None = None,
) -> dict:
    """Update user profile information.
    
    Args:
        user_id: User identifier
        name: Full name
        preferred_name: How user prefers to be addressed
        preferences: User preferences dict
        context: Additional context (occupation, interests, etc.)
    
    Returns:
        Updated user profile dict
    """
    ...

# ============== SERVER ENTRY POINT ==============

if __name__ == "__main__":
    # Run with streamable HTTP transport for production
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
```

### MCP Resources (Optional Enhancement)

```python
@mcp.resource("memory://stats")
async def get_memory_stats() -> str:
    """Get memory system statistics."""
    stats = await storage.get_stats()
    return json.dumps(stats, indent=2)

@mcp.resource("memory://bots/{bot_id}/recent")
async def get_bot_recent_memories(bot_id: str) -> str:
    """Get recent memories for a specific bot."""
    memories = await storage.get_recent(bot_id, limit=10)
    return json.dumps([m.dict() for m in memories], indent=2)
```

---

## Migration Strategy

### Phase 1: Extract & Isolate (Foundation)
**Goal**: Create clean module boundaries without breaking existing functionality

1. **Create new package structure**:
   ```
   src/ask_llm/
   ├── client/           # CLI client logic
   │   ├── __init__.py
   │   ├── local.py      # Direct GGUF execution
   │   └── remote.py     # HTTP client to server
   ├── server/           # LLM server (rename from service/)
   │   ├── __init__.py
   │   ├── app.py        # FastAPI app
   │   └── llm.py        # GGUF inference
   ├── memory_server/    # MCP memory server
   │   ├── __init__.py
   │   ├── server.py     # FastMCP tools
   │   ├── storage.py    # PostgreSQL backend
   │   ├── embeddings.py
   │   └── extraction.py
   ├── shared/           # Shared utilities
   │   ├── __init__.py
   │   ├── config.py
   │   ├── models.py     # Pydantic models
   │   └── logging.py    # NEW: Event-based logging framework
   ├── cli.py            # Thin CLI entry point
   └── bots.py           # Bot definitions
   ```

2. **Implement logging framework first** (`shared/logging.py`):
   - Create `EventCollector`, `EventRenderer`, `LogConfig`
   - Define `EventCategory` and `EventLevel` enums
   - Add `@log_operation` decorator
   - This becomes the foundation for all other components

3. **Extract memory functionality** into `memory_server/`:
   - Move `memory/postgresql.py` → `memory_server/storage.py`
   - Move `memory/extraction/` → `memory_server/extraction.py`
   - Move `memory/embeddings.py` → `memory_server/embeddings.py`

4. **Simplify LLM client** to GGUF-only in first pass:
   - Retain `clients/llama_cpp_client.py` as primary
   - Create adapter interface for future backends

### Phase 2: Build MCP Memory Server
**Goal**: Fully functional MCP server with all memory tools

1. **Create FastMCP server** with tools:
   - Implement all 9 tools listed above
   - Use existing PostgreSQL backend code
   - Test with MCP Inspector

2. **Add transport options**:
   - STDIO for local development
   - Streamable HTTP for production

3. **Write integration tests**:
   - Tool functionality tests
   - MCP protocol compliance tests

### Phase 3: Build LLM Server
**Goal**: Central inference server with MCP client

1. **Create FastAPI/Starlette server**:
   - OpenAI-compatible chat completion endpoint
   - GGUF model management
   - MCP client to connect to memory server

2. **Implement context building**:
   - Query memory server for relevant context
   - Inject into prompts before inference
   - Handle user profiles

3. **Add streaming support**:
   - SSE for chat completions
   - Proper chunk handling

### Phase 4: Refactor CLI
**Goal**: Thin CLI that uses server or local mode

1. **Simplify CLI**:
   - Remove embedded memory/LLM logic
   - Add remote/local mode selection
   - Retain bot selection and output formatting

2. **Update entry points**:
   ```toml
   [project.scripts]
   llm = "ask_llm.cli:main"
   llm-server = "ask_llm.server:main"
   llm-memory = "ask_llm.memory_server:main"
   ```

### Phase 5: Polish & Documentation
**Goal**: Production-ready release

1. **Update installation**:
   - Modify `install.sh` for new structure
   - Update pipx install with optional components

2. **Write documentation**:
   - Architecture overview
   - MCP server API reference
   - Deployment guide

3. **Add OpenAI backend** (future):
   - Create OpenAI adapter following same interface
   - Plug into server without changes

---

## File Changes Summary

### Files to Create

| Path | Purpose |
|------|---------|
| `src/ask_llm/client/__init__.py` | Client package init |
| `src/ask_llm/client/local.py` | Direct GGUF execution |
| `src/ask_llm/client/remote.py` | HTTP client to server |
| `src/ask_llm/server/app.py` | FastAPI LLM server |
| `src/ask_llm/server/llm.py` | GGUF inference wrapper |
| `src/ask_llm/server/mcp_client.py` | MCP client for memory |
| `src/ask_llm/server/context.py` | Context building |
| `src/ask_llm/memory_server/__init__.py` | Memory server package |
| `src/ask_llm/memory_server/server.py` | FastMCP tools server |
| `src/ask_llm/memory_server/storage.py` | PostgreSQL backend |
| `src/ask_llm/memory_server/embeddings.py` | Embedding generation |
| `src/ask_llm/memory_server/extraction.py` | Fact extraction |
| `src/ask_llm/shared/__init__.py` | Shared utilities |
| `src/ask_llm/shared/config.py` | Unified config |
| `src/ask_llm/shared/models.py` | Pydantic models |

### Files to Modify

| Path | Changes |
|------|---------|
| `src/ask_llm/cli.py` | Simplify to thin CLI |
| `src/ask_llm/bots.py` | Keep as-is, shared |
| `pyproject.toml` | New entry points, dependencies |
| `install.sh` | Updated installation |

### Files to Delete/Archive

| Path | Reason |
|------|--------|
| `src/ask_llm/core.py` | Replaced by server/client split |
| `src/ask_llm/clients/openai_client.py` | Future iteration |
| `src/ask_llm/clients/ollama_client.py` | Future iteration |
| `src/ask_llm/clients/huggingface_client.py` | Future iteration |
| `src/ask_llm/memory/` | Replaced by memory_server/ |
| `src/ask_llm/service/` | Replaced by server/ |
| `build/` | Outdated build artifacts |

---

## Dependencies

### Core Dependencies (Required)

```toml
[project]
dependencies = [
    "rich>=13.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
server = [
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "mcp[cli]>=1.20.0",  # MCP Python SDK
]
memory = [
    "mcp[cli]>=1.20.0",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "pgvector>=0.2.0",
    "sentence-transformers>=2.2",
]
gguf = [
    "llama-cpp-python>=0.3.0",
    "huggingface-hub>=0.20.0",
]
all = [
    "ask_llm[server,memory,gguf]",
]
```

### Entry Points

```toml
[project.scripts]
llm = "ask_llm.cli:main"
llm-server = "ask_llm.server:main"
llm-memory = "ask_llm.memory_server:main"
```

---

## Key Design Decisions

### 1. MCP over Custom Protocol
**Rationale**: MCP is becoming the standard for LLM-tool communication. Using it:
- Makes memory server usable by any MCP-compatible client
- Provides well-defined protocol with lifecycle management
- Enables future composability with other MCP servers

### 2. GGUF as Primary Backend
**Rationale**: Local LLM execution provides:
- Privacy (no data sent to third parties)
- Cost savings (no API fees)
- Latency control (local inference)
- llama-cpp-python is mature and actively maintained

### 3. FastMCP for Server Implementation
**Rationale**: FastMCP simplifies MCP server creation:
- Decorator-based tool definition
- Automatic JSON schema generation
- Built-in transport handling (STDIO, HTTP)
- Type-safe with Pydantic

### 4. Separate Memory Server
**Rationale**: Isolation enables:
- Independent scaling
- Shared memory across multiple LLM instances
- Clear API contract via MCP tools
- Easier testing and maintenance

### 5. Stateless HTTP Transport
**Rationale**: For production deployments:
- Better scalability across nodes
- JSON responses for easier debugging
- Compatible with load balancers

---

## Testing Strategy

### Unit Tests
- Memory storage operations
- Embedding generation
- Tool input validation
- Context building logic

### Integration Tests
- MCP tool execution
- LLM server ↔ memory server communication
- End-to-end query flow

### MCP Compliance Tests
- Use MCP Inspector for manual testing
- Validate tool schemas
- Test lifecycle (initialize, tool list, call, shutdown)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MCP SDK breaking changes | Medium | Pin version, monitor releases |
| PostgreSQL performance at scale | Low | Already tested with pgvector |
| llama-cpp-python compatibility | Medium | Test with specific GGUF models |
| Migration breaks existing users | High | Maintain backward-compat CLI flags |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Extract & Isolate | 3-4 days | None |
| Phase 2: MCP Memory Server | 4-5 days | Phase 1 |
| Phase 3: LLM Server | 3-4 days | Phase 2 |
| Phase 4: CLI Refactor | 2-3 days | Phase 3 |
| Phase 5: Polish | 2-3 days | Phase 4 |

**Total**: ~2-3 weeks for full implementation

---

## Development Conventions

### Python Command Usage
- **Never use `python` or `python3` directly** in scripts or terminals.
- Always use **`uv run python`** (or `uv run <script>`) for development and testing.
- `uv` manages the project virtual environment automatically and ensures correct dependencies.

### Installation & Distribution
- The project is installed globally via **`pipx`** using `install.sh`.
- `install.sh` must continue to work with `pipx install` for CLI commands (`llm`, `llm-server`, `llm-memory`).
- Entry points are defined in `pyproject.toml` under `[project.scripts]`.
- After changes to entry points or dependencies, reinstall with:
  ```bash
  ./install.sh --local .
  ```

### Testing After Changes
- Validate CLI works: `llm --status`
- Validate verbose mode: `llm --verbose "test" --local`
- Validate debug mode: `llm --debug "test" --local`

---

## Next Steps

1. **Create feature branch**: `refactor/mcp-architecture`
2. **Set up new package structure** (Phase 1.1)
3. **Extract memory code** into `memory_server/` (Phase 1.2)
4. **Implement first MCP tool** (`search_memories`) as proof of concept
5. **Iterate on remaining tools**

---

---

## Logging Framework Design

### Current Problems

1. **Inconsistent output**: Mix of `print()`, `console.print()`, and `logging.debug()` statements
2. **Duplicate messages**: Same event logged multiple times at different points
3. **No structured events**: Hard to trace flow through the system
4. **Verbose vs Debug conflated**: `--verbose` and `--debug` do similar things
5. **Library noise**: Third-party libraries (httpx, sqlalchemy) pollute debug output

### Logging Philosophy

| Mode | Purpose | Output Style |
|------|---------|--------------|
| **Normal** | User-facing results only | Clean Rich panels, minimal status |
| **Verbose** (`--verbose`) | Flow visibility for power users | Structured Rich output showing query flow, tool calls, memory operations |
| **Debug** (`--debug`) | Developer troubleshooting | Raw Python logging at DEBUG level, includes library output |

### Architecture: Event-Based Logging

Instead of scattered log statements, use a centralized **event system** that:
1. Emits structured events for significant operations
2. Renders events appropriately based on verbosity level
3. Aggregates related events into single output blocks
4. Provides consistent formatting across all components

```python
# src/ask_llm/shared/logging.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.rule import Rule
import logging
import time
from contextlib import contextmanager

# ============== EVENT TYPES ==============

class EventCategory(Enum):
    """Categories of events for filtering and grouping."""
    QUERY = auto()       # User query processing
    LLM = auto()         # LLM inference operations
    MEMORY = auto()      # Memory storage/retrieval
    MCP = auto()         # MCP protocol operations
    MODEL = auto()       # Model loading/management
    CONFIG = auto()      # Configuration operations
    SYSTEM = auto()      # System-level events

class EventLevel(Enum):
    """Event importance levels."""
    INFO = auto()        # Always shown in verbose mode
    DETAIL = auto()      # Additional details for verbose
    TRACE = auto()       # Fine-grained trace (debug only)

@dataclass
class Event:
    """A structured logging event."""
    category: EventCategory
    level: EventLevel
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float | None = None
    parent_id: str | None = None  # For nested operations
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

# ============== EVENT COLLECTOR ==============

class EventCollector:
    """
    Collects and aggregates events for batched rendering.
    
    Instead of printing immediately, events are collected and rendered
    together to avoid duplicate/scattered output.
    """
    
    def __init__(self):
        self.events: list[Event] = []
        self._operation_stack: list[str] = []
        self._start_times: dict[str, float] = {}
    
    def emit(
        self,
        category: EventCategory,
        message: str,
        level: EventLevel = EventLevel.INFO,
        **data
    ) -> str:
        """Emit an event and return its ID."""
        parent_id = self._operation_stack[-1] if self._operation_stack else None
        event = Event(
            category=category,
            level=level,
            message=message,
            data=data,
            parent_id=parent_id,
        )
        self.events.append(event)
        return event.event_id
    
    @contextmanager
    def operation(self, category: EventCategory, name: str, **data):
        """
        Context manager for timed operations.
        
        Usage:
            with collector.operation(EventCategory.LLM, "inference", model="llama"):
                # do inference
                pass
            # Automatically logs duration
        """
        event_id = self.emit(category, f"Starting: {name}", EventLevel.DETAIL, **data)
        self._operation_stack.append(event_id)
        self._start_times[event_id] = time.perf_counter()
        
        try:
            yield event_id
        finally:
            self._operation_stack.pop()
            duration_ms = (time.perf_counter() - self._start_times.pop(event_id)) * 1000
            self.emit(
                category,
                f"Completed: {name}",
                EventLevel.INFO,
                duration_ms=duration_ms,
                **data
            )
    
    def clear(self):
        """Clear collected events after rendering."""
        self.events.clear()

# ============== EVENT RENDERER ==============

class EventRenderer:
    """
    Renders collected events based on verbosity settings.
    
    Provides clean, integrated output that groups related events
    and avoids duplicate messages.
    """
    
    def __init__(self, console: Console, verbose: bool = False, debug: bool = False):
        self.console = console
        self.verbose = verbose
        self.debug = debug
        self._logger = logging.getLogger("ask_llm")
    
    def render_query_flow(self, collector: EventCollector):
        """
        Render a complete query flow as a single integrated block.
        
        Groups events by category and shows timing information.
        """
        if not self.verbose and not self.debug:
            return  # Normal mode: no flow output
        
        events = collector.events
        if not events:
            return
        
        if self.verbose:
            self._render_verbose_flow(events)
        
        if self.debug:
            self._render_debug_flow(events)
    
    def _render_verbose_flow(self, events: list[Event]):
        """Rich formatted output for --verbose mode."""
        
        # Group events by category
        by_category: dict[EventCategory, list[Event]] = {}
        for event in events:
            if event.level == EventLevel.TRACE:
                continue  # Skip trace events in verbose mode
            by_category.setdefault(event.category, []).append(event)
        
        # Build a tree view of the query flow
        tree = Tree("[bold cyan]Query Flow[/bold cyan]")
        
        category_icons = {
            EventCategory.QUERY: "💬",
            EventCategory.LLM: "🤖",
            EventCategory.MEMORY: "🧠",
            EventCategory.MCP: "🔌",
            EventCategory.MODEL: "📦",
            EventCategory.CONFIG: "⚙️",
            EventCategory.SYSTEM: "🖥️",
        }
        
        for category, cat_events in by_category.items():
            icon = category_icons.get(category, "•")
            branch = tree.add(f"{icon} [bold]{category.name}[/bold]")
            
            for event in cat_events:
                # Format event message with timing if available
                msg = event.message
                if event.duration_ms is not None:
                    msg += f" [dim]({event.duration_ms:.1f}ms)[/dim]"
                
                # Add data as sub-items if present
                if event.data:
                    event_branch = branch.add(msg)
                    for key, value in event.data.items():
                        if key != "duration_ms":
                            event_branch.add(f"[dim]{key}:[/dim] {value}")
                else:
                    branch.add(msg)
        
        self.console.print()
        self.console.print(Panel(tree, title="[dim]Verbose Output[/dim]", border_style="dim"))
    
    def _render_debug_flow(self, events: list[Event]):
        """Raw logging output for --debug mode."""
        for event in events:
            extra = ""
            if event.data:
                extra = f" | {event.data}"
            if event.duration_ms:
                extra += f" | {event.duration_ms:.2f}ms"
            
            self._logger.debug(
                f"[{event.category.name}] {event.message}{extra}"
            )
    
    def render_status(
        self,
        message: str,
        category: EventCategory = EventCategory.SYSTEM,
        style: str = "dim",
    ):
        """Render a simple status message (verbose mode only)."""
        if self.verbose:
            icon = {
                EventCategory.LLM: "🤖",
                EventCategory.MEMORY: "🧠",
                EventCategory.MCP: "🔌",
                EventCategory.MODEL: "📦",
            }.get(category, "•")
            self.console.print(f"[{style}]{icon} {message}[/{style}]")
    
    def render_error(self, message: str, exception: Exception | None = None):
        """Render an error (always shown)."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")
        if exception and self.debug:
            self._logger.exception(exception)

# ============== GLOBAL LOGGER SETUP ==============

class LogConfig:
    """
    Centralized logging configuration.
    
    Configures Python logging based on verbosity flags and suppresses
    noisy third-party loggers.
    """
    
    # Libraries that are noisy at DEBUG level
    NOISY_LOGGERS = [
        "httpx",
        "httpcore", 
        "openai",
        "requests",
        "urllib3",
        "markdown_it",
        "sqlalchemy.engine",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "asyncio",
    ]
    
    @classmethod
    def configure(cls, verbose: bool = False, debug: bool = False):
        """
        Configure logging based on verbosity settings.
        
        Args:
            verbose: Enable verbose Rich output (INFO level for ask_llm)
            debug: Enable raw debug logging (DEBUG level for all)
        """
        if debug:
            # Full debug mode: DEBUG level, show everything
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S',
            )
            # Still suppress the noisiest libraries unless explicitly debugging them
            for logger_name in cls.NOISY_LOGGERS:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        elif verbose:
            # Verbose mode: INFO level for ask_llm only
            logging.basicConfig(
                level=logging.WARNING,  # Default to WARNING
                format='%(message)s',
            )
            # Our loggers get INFO
            logging.getLogger("ask_llm").setLevel(logging.INFO)
        
        else:
            # Normal mode: WARNING and above only
            logging.basicConfig(
                level=logging.WARNING,
                format='%(message)s',
            )
    
    @classmethod
    def get_renderer(cls, console: Console, verbose: bool, debug: bool) -> EventRenderer:
        """Get an EventRenderer configured for current verbosity."""
        return EventRenderer(console, verbose=verbose, debug=debug)
    
    @classmethod
    def get_collector(cls) -> EventCollector:
        """Get a new EventCollector for a query."""
        return EventCollector()


# ============== CONVENIENCE DECORATORS ==============

def log_operation(category: EventCategory, name: str | None = None):
    """
    Decorator to automatically log function execution.
    
    Usage:
        @log_operation(EventCategory.MEMORY, "search")
        def search_memories(query: str):
            ...
    """
    def decorator(func):
        op_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = _get_current_collector()
            if collector:
                with collector.operation(category, op_name):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = _get_current_collector()
            if collector:
                with collector.operation(category, op_name):
                    return await func(*args, **kwargs)
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator

# Thread-local storage for current collector
import threading
import functools
import asyncio
import uuid

_collector_context = threading.local()

def _get_current_collector() -> EventCollector | None:
    return getattr(_collector_context, 'collector', None)

def set_current_collector(collector: EventCollector):
    _collector_context.collector = collector

@contextmanager
def query_context(collector: EventCollector):
    """Context manager to set the current collector for a query."""
    old = _get_current_collector()
    set_current_collector(collector)
    try:
        yield collector
    finally:
        set_current_collector(old)
```

### Usage Example: Query Flow

```python
# In cli.py or wherever queries are executed

from ask_llm.shared.logging import (
    LogConfig, EventCategory, query_context
)

def execute_query(prompt: str, config: Config, verbose: bool, debug: bool):
    """Execute a query with proper logging."""
    
    # Configure logging once at startup
    LogConfig.configure(verbose=verbose, debug=debug)
    
    # Get renderer and collector
    renderer = LogConfig.get_renderer(console, verbose, debug)
    collector = LogConfig.get_collector()
    
    with query_context(collector):
        # Emit query start event
        collector.emit(EventCategory.QUERY, "Processing user query", prompt=prompt[:50])
        
        # Memory retrieval (will add its own events)
        with collector.operation(EventCategory.MEMORY, "retrieve_context"):
            memories = memory_client.search_memories(prompt)
            collector.emit(
                EventCategory.MEMORY, 
                f"Found {len(memories)} relevant memories",
                EventLevel.INFO
            )
        
        # LLM inference
        with collector.operation(EventCategory.LLM, "generate_response"):
            response = llm.generate(prompt, context=memories)
        
        # Render the complete flow
        renderer.render_query_flow(collector)
    
    return response
```

### Output Examples

#### Normal Mode (no flags)
```
╭─ Nova ──────────────────────────────────────────────╮
│ Here's the answer to your question...              │
╰─────────────────────────────────────────────────────╯
```

#### Verbose Mode (`--verbose`)
```
╭─ Verbose Output ────────────────────────────────────╮
│ Query Flow                                          │
│ ├── 💬 QUERY                                        │
│ │   └── Processing user query                       │
│ │       prompt: What is the weather...              │
│ ├── 🧠 MEMORY                                       │
│ │   ├── Starting: retrieve_context                  │
│ │   ├── Found 3 relevant memories (23.4ms)          │
│ │   └── Completed: retrieve_context (45.2ms)        │
│ ├── 🤖 LLM                                          │
│ │   ├── Starting: generate_response                 │
│ │   │   model: qwen2.5-7b                           │
│ │   └── Completed: generate_response (1523.4ms)     │
╰─────────────────────────────────────────────────────╯

╭─ Nova ──────────────────────────────────────────────╮
│ Here's the answer to your question...              │
╰─────────────────────────────────────────────────────╯
```

#### Debug Mode (`--debug`)
```
14:32:01 [ask_llm.cli] DEBUG: [QUERY] Processing user query | {'prompt': 'What is...'}
14:32:01 [ask_llm.memory] DEBUG: [MEMORY] Starting: retrieve_context
14:32:01 [ask_llm.memory] DEBUG: [MEMORY] Generating embedding for query
14:32:01 [ask_llm.memory] DEBUG: [MEMORY] Vector search: 384 dimensions, top_k=10
14:32:01 [ask_llm.memory] DEBUG: [MEMORY] Found 3 relevant memories | 45.23ms
14:32:01 [ask_llm.memory] DEBUG: [MEMORY] Completed: retrieve_context | 45.23ms
14:32:01 [ask_llm.llm] DEBUG: [LLM] Starting: generate_response | {'model': 'qwen2.5-7b'}
14:32:01 [ask_llm.llm] DEBUG: [LLM] Context tokens: 1234, max_tokens: 2048
14:32:03 [ask_llm.llm] DEBUG: [LLM] Completed: generate_response | 1523.45ms
14:32:03 [ask_llm.llm] DEBUG: [LLM] Generated 156 tokens (102.3 tok/s)

╭─ Nova ──────────────────────────────────────────────╮
│ Here's the answer to your question...              │
╰─────────────────────────────────────────────────────╯
```

### Integration Points

| Component | Events Emitted |
|-----------|----------------|
| **CLI** | Query start, bot selection, mode flags |
| **LLM Client** | Model load, inference start/end, token counts, errors |
| **Memory Server** | Search, store, extract operations with timing |
| **MCP Client** | Tool calls, connection status, responses |
| **Context Builder** | Memory injection, prompt assembly |

### Migration from Current Logging

1. **Remove scattered `print()`/`console.print()` for status**
   - Replace with `collector.emit()` or `renderer.render_status()`

2. **Remove verbose-gated `if config.VERBOSE: console.print(...)`**
   - Replace with events at appropriate `EventLevel`

3. **Keep `logging.debug()` for truly debug-level info**
   - These only appear with `--debug` flag

4. **Consolidate duplicate messages**
   - E.g., "Loading model..." and "Model loaded" become a single `operation()` block

### Configuration Changes

```python
# In Config class (utils/config.py)

class Config(BaseSettings):
    # ... existing fields ...
    
    # Logging (set at runtime, not from env)
    VERBOSE: bool = False
    DEBUG: bool = False
    
    # These move from runtime flags to Config
    # So they can be accessed by any component
```

```python
# In cli.py - early initialization

def main():
    # Parse args
    args = parse_arguments(config_obj)
    
    # Configure logging FIRST before any other operations
    LogConfig.configure(
        verbose=args.verbose,
        debug=args.debug,
    )
    
    # Store in config for component access
    config_obj.VERBOSE = args.verbose
    config_obj.DEBUG = args.debug
    
    # Now proceed with normal flow...
```

---

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/latest)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Memory Server (Reference)](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [FastMCP Guide](https://pypi.org/project/mcp/)
- [Rich Console Logging](https://rich.readthedocs.io/en/latest/logging.html)

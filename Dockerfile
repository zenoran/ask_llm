# Multi-stage build for ask_llm
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY install.sh server.sh ./
RUN chmod +x install.sh server.sh

# Create virtual environment and install dependencies
# Use uv sync for all extras needed for service deployment
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync --extra mcp --extra service --extra search --extra memory

# Runtime stage
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy virtual environment and source from base
COPY --from=base /app/.venv /app/.venv
COPY --from=base /app/src /app/src
COPY --from=base /app/pyproject.toml /app/pyproject.toml
COPY --from=base /app/server.sh /app/server.sh

# Create necessary directories
RUN mkdir -p /app/.run /app/.logs /root/.config/ask-llm

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ASK_LLM_MEMORY_SERVER_HOST=0.0.0.0 \
    ASK_LLM_MEMORY_SERVER_PORT=8001 \
    ASK_LLM_SERVICE_HOST=0.0.0.0 \
    ASK_LLM_SERVICE_PORT=8642

# Expose ports
EXPOSE 8001 8642

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8642/health || exit 1

# Default command starts both MCP server and LLM service
CMD ["/app/server.sh", "start", "--stdout"]

"""Allow running as: uv run python -m ask_llm.memory_server"""

from .server import run_server

if __name__ == "__main__":
    run_server()

"""
Rich-formatted logging for the llm-service.

Provides intelligent logging with three verbosity levels:
- Normal: Clean, rich-formatted logs for key events (API calls, model loading, errors)
- Verbose (--verbose/-v): Additional detail including payloads and timing
- Debug (--debug): Low-level DEBUG messages, unformatted for debugging

Usage:
    from .logging import ServiceLogger, setup_service_logging

    # At startup
    setup_service_logging(verbose=args.verbose, debug=args.debug)

    # Create logger for a module
    log = ServiceLogger(__name__)

    # Log events
    log.api_request("POST", "/v1/chat/completions", request_data)
    log.model_loading("gpt4", "openai")
    log.info("Something happened")
"""

import json
import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Context variable to track request IDs across async operations
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Custom theme for service logs
SERVICE_THEME = Theme({
    "api.method": "bold cyan",
    "api.path": "green",
    "api.status.ok": "bold green",
    "api.status.error": "bold red",
    "model.name": "bold magenta",
    "model.type": "dim magenta",
    "timing": "dim cyan",
    "request_id": "dim yellow",
    "bot.name": "bold blue",
    "user.id": "dim blue",
    "memory": "yellow",
    "task": "cyan",
})

# Global state
_verbose = False
_debug = False
_console: Console | None = None


def get_console() -> Console:
    """Get the shared console instance."""
    global _console
    if _console is None:
        _console = Console(theme=SERVICE_THEME, stderr=True)
    return _console


def setup_service_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for the service.

    Args:
        verbose: Enable verbose logging (shows payloads, timing details)
        debug: Enable debug logging (low-level DEBUG messages, unformatted)
    """
    global _verbose, _debug

    _verbose = verbose
    _debug = debug

    # Determine log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.INFO

    console = get_console()

    # Configure root logger for the service
    if debug:
        # Debug mode: simple format, no rich
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
    else:
        # Normal/verbose mode: rich formatted output
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
                markup=True,
            )],
            force=True,
        )

    # Reduce noise from third-party libraries
    # These libraries produce a lot of INFO/DEBUG messages that clutter the output
    noisy_loggers = [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "asyncio",
        "urllib3",
        "urllib3.connectionpool",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "transformers",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "torch",
        "filelock",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "tqdm",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING if not debug else logging.INFO)
    
    # Suppress tqdm progress bars in non-debug mode
    # This prevents the "Batches: 100%..." output from sentence-transformers
    if not debug:
        os.environ["TQDM_DISABLE"] = "1"


def generate_request_id() -> str:
    """Generate a short request ID for tracing."""
    return uuid4().hex[:8]


def set_request_id(request_id: str) -> None:
    """Set the current request ID (for async context)."""
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return _request_id.get()


@dataclass
class RequestContext:
    """Context for tracking a single request."""
    request_id: str
    method: str
    path: str
    start_time: float = field(default_factory=time.time)
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    stream: bool = False

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class ServiceLogger:
    """
    Rich-formatted logger for the llm-service.

    Provides structured logging methods for common service events.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._console = get_console()

    # -------------------------------------------------------------------------
    # Standard logging methods (delegate to underlying logger)
    # -------------------------------------------------------------------------

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Structured service logging methods
    # -------------------------------------------------------------------------

    def startup(self, version: str, host: str, port: int, models: list[str], default_model: str | None) -> None:
        """Log service startup with configuration summary."""
        if _debug:
            self._logger.info(f"Service v{version} starting on {host}:{port}")
            self._logger.info(f"Models: {', '.join(models)}")
            self._logger.info(f"Default model: {default_model}")
            return

        table = Table(title="ask_llm Service Started", show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")
        table.add_row("Version", f"[bold]{version}[/bold]")
        table.add_row("Endpoint", f"[bold green]http://{host}:{port}[/bold green]")
        table.add_row("Models", f"[model.name]{len(models)} available[/model.name]")
        if default_model:
            table.add_row("Default", f"[model.name]{default_model}[/model.name]")

        self._console.print()
        self._console.print(Panel(table, border_style="green"))
        self._console.print()

    def shutdown(self) -> None:
        """Log service shutdown."""
        if _debug:
            self._logger.info("Service shutting down")
            return

        self._console.print("[dim]Service shutting down...[/dim]")

    def api_request(self, ctx: RequestContext, payload: dict[str, Any] | None = None) -> None:
        """Log an incoming API request."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        method = f"[api.method]{ctx.method}[/api.method]"
        path = f"[api.path]{ctx.path}[/api.path]"

        parts = [req_id, method, path]

        if ctx.model:
            parts.append(f"model=[model.name]{ctx.model}[/model.name]")
        if ctx.bot_id:
            parts.append(f"bot=[bot.name]{ctx.bot_id}[/bot.name]")
        if ctx.stream:
            parts.append("[dim]streaming[/dim]")

        self._logger.info(" ".join(parts))

        # Verbose: show compact incoming summary
        if _verbose and payload:
            messages = payload.get("messages", [])
            if messages:
                # Get the user's prompt (last user message)
                user_prompt = ""
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_prompt = m.get("content", "")[:100]
                        if len(m.get("content", "")) > 100:
                            user_prompt += "..."
                        break
                
                self._console.print(f"  [bold green]← Client[/bold green] | {len(messages)} msg(s)")
                if user_prompt:
                    self._console.print(f"  [green]prompt:[/green] {user_prompt}")

    def api_response(self, ctx: RequestContext, status: int = 200, tokens: int | None = None) -> None:
        """Log an API response."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        elapsed = f"[timing]{ctx.elapsed_ms:.0f}ms[/timing]"

        if status < 400:
            status_str = f"[api.status.ok]{status}[/api.status.ok]"
        else:
            status_str = f"[api.status.error]{status}[/api.status.error]"

        parts = [req_id, status_str, elapsed]

        if tokens:
            parts.append(f"[dim]{tokens} tokens[/dim]")

        self._logger.info(" ".join(parts))

    def api_error(self, ctx: RequestContext, error: str, status: int = 500) -> None:
        """Log an API error."""
        req_id = f"[request_id]\\[{ctx.request_id}][/request_id]"
        status_str = f"[api.status.error]{status}[/api.status.error]"
        elapsed = f"[timing]{ctx.elapsed_ms:.0f}ms[/timing]"

        self._logger.error(f"{req_id} {status_str} {elapsed} {error}")

    def model_loading(self, model_alias: str, model_type: str, cached: bool = False) -> None:
        """Log model loading event."""
        if cached:
            self._logger.info(
                f"[model.name]{model_alias}[/model.name] [model.type]({model_type})[/model.type] [dim]using cached instance[/dim]"
            )
        else:
            self._logger.info(
                f"[model.name]{model_alias}[/model.name] [model.type]({model_type})[/model.type] [bold yellow]loading...[/bold yellow]"
            )

    def model_loaded(self, model_alias: str, model_type: str, load_time_ms: float) -> None:
        """Log model loaded successfully."""
        self._logger.info(
            f"[model.name]{model_alias}[/model.name] [model.type]({model_type})[/model.type] "
            f"[bold green]loaded[/bold green] [timing]{load_time_ms:.0f}ms[/timing]"
        )

    def model_error(self, model_alias: str, error: str) -> None:
        """Log model loading error."""
        self._logger.error(f"[model.name]{model_alias}[/model.name] [bold red]failed to load:[/bold red] {error}")

    def memory_operation(self, operation: str, bot_id: str, count: int | None = None, details: str | None = None) -> None:
        """Log memory operations (retrieval, storage, extraction)."""
        parts = [f"[memory]{operation}[/memory]", f"bot=[bot.name]{bot_id}[/bot.name]"]
        if count is not None:
            parts.append(f"[dim]{count} items[/dim]")
        if details:
            parts.append(f"[dim]{details}[/dim]")

        self._logger.info(" ".join(parts))

    def task_submitted(self, task_id: str, task_type: str, bot_id: str) -> None:
        """Log background task submission."""
        self._logger.info(
            f"[task]task[/task] [dim]{task_id[:8]}[/dim] "
            f"[bold]{task_type}[/bold] bot=[bot.name]{bot_id}[/bot.name] [dim]queued[/dim]"
        )

    def task_completed(self, task_id: str, task_type: str, elapsed_ms: float, result: dict | None = None) -> None:
        """Log background task completion."""
        self._logger.info(
            f"[task]task[/task] [dim]{task_id[:8]}[/dim] "
            f"[bold]{task_type}[/bold] [bold green]completed[/bold green] [timing]{elapsed_ms:.0f}ms[/timing]"
        )

        if _verbose and result:
            self._log_payload("Result", result)

    def task_failed(self, task_id: str, task_type: str, error: str, elapsed_ms: float) -> None:
        """Log background task failure."""
        self._logger.error(
            f"[task]task[/task] [dim]{task_id[:8]}[/dim] "
            f"[bold]{task_type}[/bold] [bold red]failed[/bold red] [timing]{elapsed_ms:.0f}ms[/timing]: {error}"
        )

    def cache_hit(self, cache_type: str, key: str) -> None:
        """Log cache hit (verbose only)."""
        if _verbose:
            self._logger.debug(f"[dim]cache hit: {cache_type} [{key}][/dim]")

    def cache_miss(self, cache_type: str, key: str) -> None:
        """Log cache miss (verbose only)."""
        if _verbose:
            self._logger.debug(f"[dim]cache miss: {cache_type} [{key}][/dim]")

    def llm_context(self, messages: list, label: str = "LLM Context") -> None:
        """
        Log a summary of messages being sent to the LLM (verbose only).
        
        Shows a compact summary: message counts by type, memory content, and the user prompt.
        """
        if not _verbose:
            return
        
        # Count messages by role
        role_counts = {"system": 0, "user": 0, "assistant": 0}
        user_prompt = ""
        memory_content = ""
        
        for msg in messages:
            # Handle both Message objects and dicts
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content or ""
            else:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
            
            role_counts[role] = role_counts.get(role, 0) + 1
            
            # Check for memory context (second system message typically)
            if role == "system" and "What You Remember" in content:
                memory_content = content
            
            # Capture the last user message as the prompt
            if role == "user":
                user_prompt = content
        
        # Calculate history turns (user+assistant pairs, excluding current prompt)
        history_turns = min(role_counts["user"] - 1, role_counts["assistant"])
        
        # Build compact summary line
        parts = [f"[bold cyan]→ LLM[/bold cyan]"]
        
        if history_turns > 0:
            parts.append(f"[dim]history:[/dim] {history_turns} turns")
        
        memory_facts = memory_content.count("• ") if memory_content else 0
        if memory_facts > 0:
            parts.append(f"[memory]memory:[/memory] {memory_facts} facts")
        else:
            parts.append("[dim]memory: none[/dim]")
        
        parts.append(f"[dim]total:[/dim] {len(messages)} msgs")
        
        self._console.print(" | ".join(parts))
        
        # Show the user prompt (truncated)
        if user_prompt:
            display = user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
            self._console.print(f"  [green]prompt:[/green] {display}")
        
        # Show memory content (this is the key troubleshooting info)
        if memory_content:
            self._console.print(f"  [yellow]memories:[/yellow]")
            # Show the full memory content, nicely indented
            for line in memory_content.split("\n"):
                if line.strip():
                    # Indent and truncate very long lines
                    display_line = line[:120] + "..." if len(line) > 120 else line
                    self._console.print(f"    {display_line}")

    def llm_response(self, response: str, tokens: int | None = None, elapsed_ms: float | None = None) -> None:
        """Log a summary of the LLM response (verbose only)."""
        if not _verbose:
            return
        
        parts = [f"[bold blue]← LLM[/bold blue]"]
        
        # Show tokens or chars
        if tokens:
            parts.append(f"{tokens} tok")
        else:
            # Estimate tokens (~4 chars per token)
            tokens = len(response) // 4
            parts.append(f"~{tokens} tok")
        
        # Show timing and tokens/sec if available
        if elapsed_ms and elapsed_ms > 0 and tokens:
            secs = elapsed_ms / 1000
            tok_per_sec = tokens / secs
            # Color code: green = fast (>30), yellow = ok (10-30), red = slow (<10)
            if tok_per_sec >= 30:
                speed_color = "green"
            elif tok_per_sec >= 10:
                speed_color = "yellow"
            else:
                speed_color = "red"
            parts.append(f"[{speed_color}]{tok_per_sec:.1f} tok/s[/{speed_color}]")
            parts.append(f"[dim]{elapsed_ms:.0f}ms[/dim]")
        
        self._console.print(" | ".join(parts))
        
        # Show truncated response
        display = response[:100] + "..." if len(response) > 100 else response
        self._console.print(f"  [blue]response:[/blue] {display}")

    def _log_payload(self, label: str, data: dict[str, Any], max_content_len: int = 500) -> None:
        """Log a payload (for verbose mode)."""
        # Deep copy and truncate long content
        def truncate(obj: Any, depth: int = 0) -> Any:
            if depth > 5:
                return "..."
            if isinstance(obj, dict):
                return {k: truncate(v, depth + 1) for k, v in obj.items()}
            if isinstance(obj, list):
                return [truncate(item, depth + 1) for item in obj[:10]]  # Max 10 items
            if isinstance(obj, str) and len(obj) > max_content_len:
                return obj[:max_content_len] + "..."
            return obj

        truncated = truncate(data)
        try:
            formatted = json.dumps(truncated, indent=2, default=str)
            self._console.print(f"[dim]{label}:[/dim]")
            self._console.print(formatted, highlight=True)
        except Exception:
            self._logger.debug(f"{label}: {truncated}")


# Module-level convenience function
def get_service_logger(name: str) -> ServiceLogger:
    """Get a ServiceLogger instance for the given module name."""
    return ServiceLogger(name)

#!/usr/bin/env python3
"""
Memory management CLI tool for ask_llm.
Thin client that proxies all operations through the llm-service.

Usage:
    llm-memory --stats                           # Show memory statistics
    llm-memory "what do you know about me"       # Search all methods
    llm-memory -m embedding "nick"               # Search via embeddings only
    llm-memory -m high-importance                # Show high-importance memories
    llm-memory --list-all -n 20                  # List top 20 memories
    llm-memory --delete <ID>                     # Delete a memory by UUID prefix
    llm-memory --list-attrs                      # List user profile attributes with IDs
    llm-memory --list-attrs nick                 # List attrs for specific user
    llm-memory --delete-attr 42                  # Delete a profile attribute by integer ID
    llm-memory --history                         # Show recent conversation history
    llm-memory --search-history "topic"          # Search conversation history
    llm-memory --forget-recent 5                 # Soft-delete last 5 messages (recoverable)
    llm-memory --forget-minutes 30               # Soft-delete messages from last 30 min
    llm-memory --restore                         # Restore soft-deleted messages
    llm-memory --regenerate-embeddings           # Regenerate all embeddings
    llm-memory --consolidate                     # Merge redundant memories
    llm-memory --consolidate-dry-run             # Preview consolidation
"""

import argparse
import sys
from datetime import datetime

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ask_llm.utils.config import config

console = Console()

# Timeout settings
DEFAULT_TIMEOUT = 30.0
LONG_TIMEOUT = 120.0  # For operations like consolidation


def get_service_url() -> str:
    """Get the service base URL from config."""
    host = config.SERVICE_HOST or "localhost"
    port = config.SERVICE_PORT  # Default is 8642 from config
    return f"http://{host}:{port}"


def check_service_available() -> bool:
    """Check if the llm-service is running."""
    try:
        response = httpx.get(f"{get_service_url()}/health", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def api_get(endpoint: str, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a GET request to the service API."""
    try:
        response = httpx.get(
            f"{get_service_url()}{endpoint}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def api_post(endpoint: str, json_data: dict | None = None, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a POST request to the service API."""
    try:
        response = httpx.post(
            f"{get_service_url()}{endpoint}",
            json=json_data or {},
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def api_delete(endpoint: str, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a DELETE request to the service API."""
    try:
        response = httpx.delete(
            f"{get_service_url()}{endpoint}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def display_messages_preview(messages: list[dict], max_content_length: int = 80):
    """Display a preview of messages for confirmation prompts."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Time", style="dim", width=19)
    table.add_column("Role", width=10)
    table.add_column("Content", overflow="ellipsis")
    
    for msg in messages:
        ts = msg.get("timestamp")
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "yellow"
        
        content = msg.get("content", "")
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        content = content.replace("\n", " ").replace("\r", "")
        
        table.add_row(time_str, f"[{role_style}]{role}[/{role_style}]", content)
    
    console.print(table)


def display_results(results: list, method: str, show_ids: bool = True):
    """Display memory results in a table."""
    table = Table()
    if show_ids:
        table.add_column("ID", style="dim", width=8)
    table.add_column("Rel", justify="right", style="cyan", width=5)
    table.add_column("Imp", justify="right", style="yellow", width=4)
    table.add_column("Tags", style="magenta", width=20)
    table.add_column("Content", style="white", max_width=70)
    
    for r in results:
        content = r.get("content", "")
        content = content[:100] + "..." if len(content) > 100 else content
        relevance = r.get("relevance") or r.get("similarity", 0) or 0
        importance = r.get("importance", 0.5)
        tags = r.get("tags", ["misc"])
        if isinstance(tags, str):
            import json
            tags = json.loads(tags) if tags else ["misc"]
        
        memory_id = r.get("id", "")[:8] if r.get("id") else "?"
        
        row = []
        if show_ids:
            row.append(memory_id)
        row.extend([
            f"{relevance:.3f}",
            f"{importance:.2f}",
            ", ".join(tags[:3]) if tags else "misc",
            content,
        ])
        table.add_row(*row)
    
    console.print(table)
    console.print(f"[dim]Found {len(results)} results via {method}[/dim]")
    if show_ids:
        console.print(f"[dim]Use --delete <ID> to remove a memory[/dim]\n")


def show_stats(bot_id: str):
    """Show memory statistics."""
    data = api_get("/v1/memory/stats", {"bot_id": bot_id})
    if not data:
        return

    messages = data.get("messages", {})
    memories = data.get("memories", {})
    
    mem_total = memories.get("total_count", memories.get("total", 0))
    mem_with_emb = memories.get("with_embeddings", mem_total)  # Fallback to total if not reported
    msg_total = messages.get("total_count", messages.get("total", 0))
    msg_forgotten = messages.get("forgotten_count", messages.get("ignored", 0))
    
    console.print(Panel(f"""
[bold]Memories:[/bold] {mem_total} total
[bold]Messages:[/bold] {msg_total} total ({msg_forgotten} forgotten/recoverable)
""", title=f"Memory Statistics - {data.get('bot_id', bot_id)}"))

    # Tag distribution (if available)
    tag_counts = memories.get("tag_counts", {})
    if tag_counts:
        table = Table(title="Tag Distribution")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", justify="right")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            table.add_row(tag, str(count))
        console.print(table)
    
    # Importance distribution (if available)
    importance_dist = memories.get("importance_distribution", {})
    if importance_dist:
        table = Table(title="Importance Distribution")
        table.add_column("Range", style="cyan")
        table.add_column("Count", justify="right")
        for range_name, count in sorted(importance_dist.items(), reverse=True):
            table.add_row(range_name, str(count))
        console.print(table)


def list_all_memories(bot_id: str, limit: int):
    """List all memories ordered by importance."""
    data = api_get("/v1/memory", {"bot_id": bot_id, "limit": limit})
    if not data:
        return

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return
    
    table = Table(title=f"Top {limit} Memories by Importance")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Imp", justify="right", style="cyan", width=4)
    table.add_column("Tags", style="magenta", width=20)
    table.add_column("Content", style="white", max_width=60)
    table.add_column("Acc", justify="right", width=3)
    
    for mem in results:
        content = mem.get("content", "")
        content = content[:100] + "..." if len(content) > 100 else content
        tags = mem.get("tags", ["misc"])
        memory_id = mem.get("id", "")[:8] if mem.get("id") else "?"
        table.add_row(
            memory_id,
            f"{mem.get('importance', 0.5):.2f}",
            ", ".join(tags[:3]),
            content,
            str(mem.get("access_count", 0)),
        )
    
    console.print(table)
    console.print(f"[dim]Use --delete <ID> to remove a memory[/dim]")


def search_memories(bot_id: str, query: str, method: str, limit: int, min_importance: float):
    """Search memories using specified method."""
    console.print(f"[bold]Query:[/bold] \"{query}\"\n")
    
    if method == "all":
        # Show results from multiple methods
        for m in ["embedding", "high-importance"]:
            console.print(f"[bold cyan]{m.title()} Search[/bold cyan]")
            data = api_post("/v1/memory/search", {
                "query": query,
                "method": m,
                "limit": limit,
                "min_importance": min_importance if m == "high-importance" else 0.0,
                "bot_id": bot_id,
            })
            if data and data.get("results"):
                display_results(data["results"], m)
            else:
                console.print("[dim]No results[/dim]\n")
    else:
        data = api_post("/v1/memory/search", {
            "query": query,
            "method": method,
            "limit": limit,
            "min_importance": min_importance,
            "bot_id": bot_id,
        })
        if data and data.get("results"):
            display_results(data["results"], method)
        else:
            console.print("[dim]No results[/dim]")


def handle_forget_recent(bot_id: str, count: int, skip_confirm: bool):
    """Ignore the last N messages and delete associated long-term memories."""
    # Preview messages first
    data = api_get("/v1/memory/preview/recent", {"bot_id": bot_id, "count": count})
    if not data or not data.get("messages"):
        console.print("[yellow]No messages to ignore[/yellow]")
        return
    
    messages = data["messages"]
    console.print(f"[bold]Messages to ignore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Ignore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        result = api_post("/v1/memory/forget", {"count": count}, {"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
            console.print("[dim]Use --restore to undo[/dim]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_forget_minutes(bot_id: str, minutes: int, skip_confirm: bool):
    """Ignore messages from the last N minutes and delete associated long-term memories."""
    # Preview messages first
    data = api_get("/v1/memory/preview/minutes", {"bot_id": bot_id, "minutes": minutes})
    if not data or not data.get("messages"):
        console.print("[yellow]No messages in that time range[/yellow]")
        return
    
    messages = data["messages"]
    console.print(f"[bold]Messages from last {minutes} minutes to ignore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Ignore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        result = api_post("/v1/memory/forget", {"minutes": minutes}, {"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
            console.print("[dim]Use --restore to undo[/dim]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_restore(bot_id: str, skip_confirm: bool):
    """Restore all ignored messages."""
    # Preview ignored messages first
    data = api_get("/v1/memory/preview/ignored", {"bot_id": bot_id})
    if not data or not data.get("messages"):
        console.print("[yellow]No ignored messages to restore[/yellow]")
        return
    
    messages = data["messages"]
    console.print(f"[bold]Ignored messages to restore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Restore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        result = api_post("/v1/memory/restore", params={"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_regenerate_embeddings(bot_id: str):
    """Regenerate embeddings for all memories."""
    console.print(f"[bold]Regenerating embeddings for bot: {bot_id}[/bold]\n")
    
    with console.status("[bold green]Generating embeddings..."):
        result = api_post("/v1/memory/regenerate-embeddings", params={"bot_id": bot_id}, timeout=LONG_TIMEOUT)
    console.print()  # Newline after spinner clears
    
    if not result:
        return
    
    if result.get("success"):
        console.print(f"[green]✓ Updated: {result.get('updated', 0)}[/green]")
        if result.get("failed", 0) > 0:
            console.print(f"[yellow]! Failed: {result.get('failed', 0)}[/yellow]")
        if result.get("embedding_dim"):
            console.print(f"  Embedding dimension: {result['embedding_dim']}")
    else:
        console.print(f"[red]Error: {result.get('message', 'Unknown error')}[/red]")
        console.print("Install with: [cyan]pipx runpip ask-llm install sentence-transformers[/cyan]")


def handle_consolidate(bot_id: str, dry_run: bool = False):
    """Find and merge redundant memories."""
    mode_str = "[yellow](DRY RUN)[/yellow] " if dry_run else ""
    console.print(f"[bold]{mode_str}Memory Consolidation for bot: {bot_id}[/bold]\n")
    
    with console.status("[bold green]Finding and consolidating similar memories..."):
        result = api_post(
            "/v1/memory/consolidate",
            {"dry_run": dry_run},
            {"bot_id": bot_id},
            timeout=LONG_TIMEOUT,
        )
    console.print()  # Newline after spinner clears
    
    if not result:
        return
    
    console.print()
    if dry_run:
        console.print(f"[yellow]Would merge:[/yellow]")
    else:
        console.print(f"[green]Results:[/green]")
    
    console.print(f"  Clusters found: {result.get('clusters_found', 0)}")
    console.print(f"  Clusters merged: {result.get('clusters_merged', 0)}")
    console.print(f"  Memories consolidated: {result.get('memories_consolidated', 0)}")
    if not dry_run:
        console.print(f"  New memories created: {result.get('new_memories_created', 0)}")
    
    errors = result.get("errors", [])
    if errors:
        console.print(f"\n[yellow]Errors ({len(errors)}):[/yellow]")
        for err in errors[:5]:
            console.print(f"  - {err}")


def handle_delete_memory(bot_id: str, memory_id: str, skip_confirm: bool):
    """Delete a specific memory by ID."""
    # First, try to find the memory to show what will be deleted
    data = api_post("/v1/memory/search", {
        "query": "",
        "method": "embedding",
        "limit": 100,
        "bot_id": bot_id,
    })
    
    matching = None
    if data and data.get("results"):
        for mem in data["results"]:
            if mem.get("id", "").startswith(memory_id):
                matching = mem
                break
    
    if not matching:
        # Try listing all memories
        data = api_get("/v1/memory", {"bot_id": bot_id, "limit": 100})
        if data and data.get("results"):
            for mem in data["results"]:
                if mem.get("id", "").startswith(memory_id):
                    matching = mem
                    break
    
    if not matching:
        console.print(f"[red]Memory '{memory_id}' not found[/red]")
        return
    
    full_id = matching.get("id", memory_id)
    content = matching.get("content", "")[:100]
    tags = matching.get("tags", [])
    
    console.print(f"[bold]Memory to delete:[/bold]")
    console.print(f"  ID: [dim]{full_id}[/dim]")
    console.print(f"  Tags: [magenta]{', '.join(tags) if tags else 'none'}[/magenta]")
    console.print(f"  Content: {content}{'...' if len(matching.get('content', '')) > 100 else ''}")
    console.print()
    
    if not skip_confirm:
        confirm = console.input("[bold yellow]Delete this memory? (y/N):[/bold yellow] ").strip().lower()
        if confirm not in ('y', 'yes'):
            console.print("[dim]Cancelled[/dim]")
            return
    
    result = api_delete(f"/v1/memory/{full_id}", {"bot_id": bot_id})
    if result and result.get("success"):
        console.print(f"[green]✓ Memory deleted[/green]")
    else:
        console.print(f"[red]Failed to delete memory[/red]")


def handle_delete_attribute(attribute_id: int, skip_confirm: bool):
    """Delete a user profile attribute by ID."""
    # First get info about the attribute
    result = api_delete(f"/v1/users/attribute/{attribute_id}")
    
    if not result:
        console.print(f"[red]Failed to delete attribute {attribute_id}[/red]")
        return
    
    if result.get("success"):
        deleted = result.get("deleted", {})
        console.print(f"[green]✓ Deleted:[/green] {deleted.get('category', '?')}.{deleted.get('key', '?')} = {deleted.get('value', '?')}")
        console.print(f"[dim]  From: {deleted.get('entity_type', '?')}/{deleted.get('entity_id', '?')}[/dim]")
    else:
        console.print(f"[red]Failed to delete attribute: {result.get('detail', 'Unknown error')}[/red]")


def handle_list_attrs(user_id: str):
    """List user profile attributes with IDs."""
    data = api_get(f"/v1/users/{user_id}")
    
    if not data:
        console.print(f"[red]Failed to get profile for user '{user_id}'[/red]")
        return
    
    attributes = data.get("attributes", [])
    if not attributes:
        console.print(f"[yellow]No attributes found for user '{user_id}'[/yellow]")
        return
    
    table = Table(title=f"Profile Attributes for {user_id}")
    table.add_column("ID", style="cyan", justify="right", width=6)
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Key", style="yellow", width=25)
    table.add_column("Value", style="white", max_width=50)
    table.add_column("Conf", justify="right", width=4)
    
    for attr in attributes:
        value = str(attr.get("value", ""))
        if len(value) > 50:
            value = value[:47] + "..."
        table.add_row(
            str(attr.get("id", "?")),
            attr.get("category", "?"),
            attr.get("key", "?"),
            value,
            f"{attr.get('confidence', 1.0):.1f}",
        )
    
    console.print(table)
    console.print(f"\n[dim]Use --delete-attr <ID> to remove an attribute[/dim]")


def handle_show_history(bot_id: str, limit: int):
    """Show recent conversation history."""
    data = api_get("/v1/history", {"bot_id": bot_id, "limit": limit})
    
    if not data:
        console.print(f"[red]Failed to get history for bot '{bot_id}'[/red]")
        return
    
    messages = data.get("messages", [])
    if not messages:
        console.print(f"[yellow]No history found for bot '{bot_id}'[/yellow]")
        return
    
    table = Table(title=f"Conversation History ({len(messages)} messages)")
    table.add_column("Time", style="dim", width=19)
    table.add_column("Role", width=10)
    table.add_column("Content", style="white", max_width=80)
    
    for msg in messages:
        ts = msg.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "yellow"
        content = msg.get("content", "")
        if len(content) > 100:
            content = content[:97] + "..."
        content = content.replace("\n", " ")
        table.add_row(time_str, f"[{role_style}]{role}[/{role_style}]", content)
    
    console.print(table)


def handle_search_history(bot_id: str, query: str, limit: int):
    """Search conversation history."""
    data = api_post("/v1/history/search", params={"bot_id": bot_id, "query": query, "limit": limit})
    
    if not data:
        console.print(f"[red]Failed to search history for bot '{bot_id}'[/red]")
        return
    
    messages = data.get("messages", [])
    if not messages:
        console.print(f"[yellow]No messages found matching '{query}'[/yellow]")
        return
    
    table = Table(title=f"Search Results for '{query}' ({len(messages)} matches)")
    table.add_column("Time", style="dim", width=19)
    table.add_column("Role", width=10)
    table.add_column("Content", style="white", max_width=80)
    
    for msg in messages:
        ts = msg.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "yellow"
        content = msg.get("content", "")
        if len(content) > 100:
            content = content[:97] + "..."
        content = content.replace("\n", " ")
        table.add_row(time_str, f"[{role_style}]{role}[/{role_style}]", content)
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Memory management for ask_llm")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--bot", "-b", default="nova", help="Bot ID (default: nova)")
    
    # Search options
    search_group = parser.add_argument_group("Search options")
    search_group.add_argument("--method", "-m", choices=["text", "embedding", "high-importance", "messages", "all"], 
                        default="all", help="Search method to use")
    search_group.add_argument("--limit", "-n", type=int, default=10, help="Max results (default: 10)")
    search_group.add_argument("--min-importance", type=float, default=0.0, help="Min importance filter")
    search_group.add_argument("--list-all", action="store_true", help="List all memories (ignores query)")
    search_group.add_argument("--stats", action="store_true", help="Show memory statistics")
    
    # Memory deletion
    del_group = parser.add_argument_group("Memory deletion")
    del_group.add_argument("--delete", "-d", metavar="ID", help="Delete a memory by ID (use first 8 chars of UUID)")
    del_group.add_argument("--delete-attr", metavar="ID", type=int, help="Delete a user profile attribute by its integer ID (use --list-attrs to see IDs)")
    del_group.add_argument("--list-attrs", metavar="USER", nargs="?", const="nick", help="List user profile attributes with IDs (default user: nick)")
    
    # Message management
    msg_group = parser.add_argument_group("Message management (soft-delete)")
    msg_group.add_argument("--forget-recent", type=int, metavar="N", help="Ignore the last N messages (reversible)")
    msg_group.add_argument("--forget-minutes", type=int, metavar="N", help="Ignore messages from last N minutes (reversible)")
    msg_group.add_argument("--restore", action="store_true", help="Restore all ignored messages")
    msg_group.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    
    # History management
    hist_group = parser.add_argument_group("History management")
    hist_group.add_argument("--history", action="store_true", help="Show recent conversation history")
    hist_group.add_argument("--search-history", metavar="QUERY", help="Search conversation history")
    
    # Memory maintenance
    maint_group = parser.add_argument_group("Memory maintenance")
    maint_group.add_argument("--regenerate-embeddings", action="store_true", help="Regenerate embeddings for all memories")
    maint_group.add_argument("--consolidate", action="store_true", help="Find and merge redundant memories")
    maint_group.add_argument("--consolidate-dry-run", action="store_true", help="Show what would be consolidated without making changes")
    
    args = parser.parse_args()
    
    # Check service availability
    if not check_service_available():
        console.print("[red]Error: llm-service is not running.[/red]")
        console.print("[dim]Start it with: llm-service[/dim]")
        sys.exit(1)
    
    console.print(f"\n[bold cyan]Memory Tool[/bold cyan] - Bot: [yellow]{args.bot}[/yellow]\n")
    
    # Handle delete command first
    if args.delete:
        handle_delete_memory(args.bot, args.delete, args.yes)
        return
    
    # Handle delete attribute command
    if args.delete_attr:
        handle_delete_attribute(args.delete_attr, args.yes)
        return
    
    # Handle list attributes command
    if args.list_attrs:
        handle_list_attrs(args.list_attrs)
        return
    
    # Handle message management commands first
    if args.forget_recent:
        handle_forget_recent(args.bot, args.forget_recent, args.yes)
        return
    
    if args.forget_minutes:
        handle_forget_minutes(args.bot, args.forget_minutes, args.yes)
        return
    
    if args.restore:
        handle_restore(args.bot, args.yes)
        return
    
    # History management
    if args.history:
        handle_show_history(args.bot, args.limit)
        return
    
    if args.search_history:
        handle_search_history(args.bot, args.search_history, args.limit)
        return
    
    # Memory maintenance
    if args.regenerate_embeddings:
        handle_regenerate_embeddings(args.bot)
        return
    
    if args.consolidate or args.consolidate_dry_run:
        handle_consolidate(args.bot, dry_run=args.consolidate_dry_run)
        return
    
    # Show stats
    if args.stats:
        show_stats(args.bot)
        return
    
    # List all memories
    if args.list_all:
        list_all_memories(args.bot, args.limit)
        return
    
    # Search
    if not args.query and args.method != "high-importance":
        console.print("[red]Error:[/red] Query required for search (use --list-all to see all memories, or --stats)")
        sys.exit(1)
    
    search_memories(args.bot, args.query, args.method, args.limit, args.min_importance)


if __name__ == "__main__":
    main()

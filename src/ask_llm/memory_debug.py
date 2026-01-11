#!/usr/bin/env python3
"""
Memory management CLI tool for ask_llm.

Usage:
    llm-memory --stats                           # Show memory statistics
    llm-memory "what do you know about me"       # Search all methods
    llm-memory -m embedding "nick"               # Search via embeddings only
    llm-memory -m high-importance                # Show high-importance memories
    llm-memory --list-all -n 20                  # List top 20 memories
    llm-memory --forget-recent 5                 # Soft-delete last 5 messages
    llm-memory --forget-minutes 30               # Soft-delete messages from last 30 min
    llm-memory --restore                         # Restore soft-deleted messages
    llm-memory --regenerate-embeddings           # Regenerate all embeddings
    llm-memory --consolidate                     # Merge redundant memories
    llm-memory --consolidate-dry-run             # Preview consolidation
"""

import argparse
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Memory management for ask_llm")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--bot", "-b", default="mira", help="Bot ID (default: mira)")
    
    # Search options
    search_group = parser.add_argument_group("Search options")
    search_group.add_argument("--method", "-m", choices=["text", "embedding", "high-importance", "messages", "all"], 
                        default="all", help="Search method to use")
    search_group.add_argument("--limit", "-n", type=int, default=10, help="Max results (default: 10)")
    search_group.add_argument("--min-importance", type=float, default=0.0, help="Min importance filter")
    search_group.add_argument("--list-all", action="store_true", help="List all memories (ignores query)")
    search_group.add_argument("--stats", action="store_true", help="Show memory statistics")
    
    # Message management
    msg_group = parser.add_argument_group("Message management (soft-delete)")
    msg_group.add_argument("--forget-recent", type=int, metavar="N", help="Ignore the last N messages (reversible)")
    msg_group.add_argument("--forget-minutes", type=int, metavar="N", help="Ignore messages from last N minutes (reversible)")
    msg_group.add_argument("--restore", action="store_true", help="Restore all ignored messages")
    msg_group.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    
    # Memory maintenance
    maint_group = parser.add_argument_group("Memory maintenance")
    maint_group.add_argument("--regenerate-embeddings", action="store_true", help="Regenerate embeddings for all memories")
    maint_group.add_argument("--consolidate", action="store_true", help="Find and merge redundant memories")
    maint_group.add_argument("--consolidate-dry-run", action="store_true", help="Show what would be consolidated without making changes")
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from .memory.postgresql import PostgreSQLMemoryBackend
    from .utils.config import config
    
    console.print(f"\n[bold cyan]Memory Tool[/bold cyan] - Bot: [yellow]{args.bot}[/yellow]\n")
    
    # Initialize memory backend
    try:
        backend = PostgreSQLMemoryBackend(config=config, bot_id=args.bot)
        console.print(f"[green]✓[/green] Connected to PostgreSQL at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE}\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to connect: {e}")
        sys.exit(1)
    
    # Handle message management commands first
    if args.forget_recent:
        handle_forget_recent(backend, args.forget_recent, args.yes)
        return
    
    if args.forget_minutes:
        handle_forget_minutes(backend, args.forget_minutes, args.yes)
        return
    
    if args.restore:
        handle_restore(backend, args.yes)
        return
    
    # Memory maintenance
    if args.regenerate_embeddings:
        handle_regenerate_embeddings(backend, config)
        return
    
    if args.consolidate or args.consolidate_dry_run:
        handle_consolidate(backend, config, dry_run=args.consolidate_dry_run)
        return
    
    # Show stats
    if args.stats:
        show_stats(backend, args.bot)
        return
    
    # List all memories
    if args.list_all:
        list_all_memories(backend, args.limit)
        return
    
    # Search
    if not args.query and args.method != "high-importance":
        console.print("[red]Error:[/red] Query required for search (use --list-all to see all memories, or --stats)")
        sys.exit(1)
    
    if args.method == "all":
        search_all_methods(backend, args.query, args.limit, args.min_importance)
    elif args.method == "text":
        search_text(backend, args.query, args.limit, args.min_importance)
    elif args.method == "embedding":
        search_embedding(backend, args.query, args.limit, args.min_importance)
    elif args.method == "high-importance":
        search_high_importance(backend, args.limit, args.min_importance or 0.7)
    elif args.method == "messages":
        search_messages(backend, args.query, args.limit)


def display_messages_preview(messages: list[dict], max_content_length: int = 80):
    """Display a preview of messages for confirmation prompts."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Time", style="dim", width=19)
    table.add_column("Role", width=10)
    table.add_column("Content", overflow="ellipsis")
    
    for msg in messages:
        ts = msg.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "yellow"
        
        content = msg.get("content", "")
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        content = content.replace("\n", " ").replace("\r", "")
        
        table.add_row(time_str, f"[{role_style}]{role}[/{role_style}]", content)
    
    console.print(table)


def handle_forget_recent(backend, count: int, skip_confirm: bool):
    """Ignore the last N messages."""
    messages = backend.preview_recent_messages(count)
    if not messages:
        console.print("[yellow]No messages to ignore[/yellow]")
        return
    
    console.print(f"[bold]Messages to ignore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Ignore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        ignored = backend.ignore_recent_messages(count)
        console.print(f"[green]Ignored {ignored} messages (use --restore to undo)[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_forget_minutes(backend, minutes: int, skip_confirm: bool):
    """Ignore messages from the last N minutes."""
    messages = backend.preview_messages_since_minutes(minutes)
    if not messages:
        console.print("[yellow]No messages in that time range[/yellow]")
        return
    
    console.print(f"[bold]Messages from last {minutes} minutes to ignore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Ignore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        ignored = backend.ignore_messages_since_minutes(minutes)
        console.print(f"[green]Ignored {ignored} messages (use --restore to undo)[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_restore(backend, skip_confirm: bool):
    """Restore all ignored messages."""
    messages = backend.preview_ignored_messages()
    if not messages:
        console.print("[yellow]No ignored messages to restore[/yellow]")
        return
    
    console.print(f"[bold]Ignored messages to restore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)
    
    if skip_confirm or console.input("\n[bold yellow]Restore these messages? (y/N):[/bold yellow] ").strip().lower() in ('y', 'yes'):
        restored = backend.restore_ignored_messages()
        console.print(f"[green]Restored {restored} messages[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


def show_stats(backend, bot_id: str):
    """Show memory statistics."""
    from sqlalchemy import text
    
    with backend.engine.connect() as conn:
        # Count memories
        mem_count = conn.execute(text(f"SELECT COUNT(*) FROM {bot_id}_memories")).scalar()
        mem_with_embedding = conn.execute(text(f"SELECT COUNT(*) FROM {bot_id}_memories WHERE embedding IS NOT NULL")).scalar()
        
        # Count messages
        msg_count = conn.execute(text(f"SELECT COUNT(*) FROM {bot_id}_messages")).scalar()
        
        # Count forgotten messages
        try:
            msg_forgotten = conn.execute(text(f"SELECT COUNT(*) FROM {bot_id}_forgotten_messages")).scalar() or 0
        except Exception:
            msg_forgotten = 0  # Table may not exist yet
        
        # Memory types
        type_counts = conn.execute(text(f"""
            SELECT memory_type, COUNT(*) as count 
            FROM {bot_id}_memories 
            GROUP BY memory_type 
            ORDER BY count DESC
        """)).fetchall()
        
        # Importance distribution
        importance_dist = conn.execute(text(f"""
            SELECT 
                CASE 
                    WHEN importance >= 0.9 THEN '0.9-1.0 (Critical)'
                    WHEN importance >= 0.7 THEN '0.7-0.9 (Important)'
                    WHEN importance >= 0.5 THEN '0.5-0.7 (Moderate)'
                    WHEN importance >= 0.3 THEN '0.3-0.5 (Low)'
                    ELSE '0.0-0.3 (Trivial)'
                END as range,
                COUNT(*) as count
            FROM {bot_id}_memories
            GROUP BY range
            ORDER BY range DESC
        """)).fetchall()
    
    console.print(Panel(f"""
[bold]Memories:[/bold] {mem_count} total ({mem_with_embedding} with embeddings)
[bold]Messages:[/bold] {msg_count} total ({msg_forgotten} forgotten/recoverable)
""", title="Memory Statistics"))
    
    # Memory types table
    table = Table(title="Memory Types")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    for row in type_counts:
        table.add_row(row.memory_type, str(row.count))
    console.print(table)
    
    # Importance distribution
    table = Table(title="Importance Distribution")
    table.add_column("Range", style="cyan")
    table.add_column("Count", justify="right")
    for row in importance_dist:
        table.add_row(row.range, str(row.count))
    console.print(table)


def list_all_memories(backend, limit: int):
    """List all memories ordered by importance."""
    from sqlalchemy import text
    
    with backend.engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, content, memory_type, importance, access_count, created_at
            FROM {backend._memories_table_name}
            ORDER BY importance DESC, access_count DESC, created_at DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()
    
    table = Table(title=f"Top {limit} Memories by Importance")
    table.add_column("Imp", justify="right", style="cyan", width=4)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Content", style="white", max_width=80)
    table.add_column("Acc", justify="right", width=3)
    
    for row in rows:
        content = row.content[:100] + "..." if len(row.content) > 100 else row.content
        table.add_row(
            f"{row.importance:.2f}",
            row.memory_type,
            content,
            str(row.access_count or 0),
        )
    
    console.print(table)


def search_all_methods(backend, query: str, limit: int, min_importance: float):
    """Try all search methods and show results."""
    console.print(f"[bold]Query:[/bold] \"{query}\"\n")
    
    # 1. Text search
    console.print("[bold cyan]1. Text Search (PostgreSQL full-text)[/bold cyan]")
    text_results = backend.search_memories_by_text(query, limit, min_importance=min_importance)
    if text_results:
        display_results(text_results, "text")
    else:
        console.print("[dim]No results[/dim]\n")
    
    # 2. Embedding search
    console.print("[bold cyan]2. Embedding Search (semantic similarity)[/bold cyan]")
    from .memory.embeddings import generate_embedding, get_embedding_model
    
    model = get_embedding_model(backend.embedding_model)
    if model is not None:
        try:
            # Generate embedding for query
            query_embedding = generate_embedding(query, backend.embedding_model)
            if query_embedding:
                emb_results = backend.search_memories_by_embedding(query_embedding, limit, min_importance=min_importance)
                if emb_results:
                    display_results(emb_results, "embedding")
                else:
                    console.print("[dim]No results[/dim]\n")
            else:
                console.print("[dim]Failed to generate embedding[/dim]\n")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
    else:
        console.print("[dim]Embedding model not available[/dim]\n")
    
    # 3. High-importance fallback
    console.print("[bold cyan]3. High-Importance Fallback (importance >= 0.7)[/bold cyan]")
    high_imp_results = backend.get_high_importance_memories(limit, min_importance=0.7)
    if high_imp_results:
        display_results(high_imp_results, "high-importance")
    else:
        console.print("[dim]No results[/dim]\n")
    
    # 4. Message search (last resort)
    console.print("[bold cyan]4. Message Search (raw messages fallback)[/bold cyan]")
    msg_results = backend.search_messages_by_text(query, limit)
    if msg_results:
        display_message_results(msg_results)
    else:
        console.print("[dim]No results[/dim]\n")


def search_text(backend, query: str, limit: int, min_importance: float):
    """Text search only."""
    console.print(f"[bold]Text Search:[/bold] \"{query}\"\n")
    results = backend.search_memories_by_text(query, limit, min_importance=min_importance)
    if results:
        display_results(results, "text")
    else:
        console.print("[dim]No results[/dim]")


def search_embedding(backend, query: str, limit: int, min_importance: float):
    """Embedding search only."""
    from .memory.embeddings import generate_embedding, get_embedding_model
    
    console.print(f"[bold]Embedding Search:[/bold] \"{query}\"\n")
    
    model = get_embedding_model(backend.embedding_model)
    if model is None:
        console.print("[red]Embedding model not available[/red]")
        return
    
    try:
        query_embedding = generate_embedding(query, backend.embedding_model)
        if query_embedding:
            results = backend.search_memories_by_embedding(query_embedding, limit, min_importance=min_importance)
            if results:
                display_results(results, "embedding")
            else:
                console.print("[dim]No results[/dim]")
        else:
            console.print("[dim]Failed to generate embedding[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def search_high_importance(backend, limit: int, min_importance: float):
    """High-importance memories."""
    console.print(f"[bold]High-Importance Memories[/bold] (>= {min_importance})\n")
    results = backend.get_high_importance_memories(limit, min_importance=min_importance)
    if results:
        display_results(results, "high-importance")
    else:
        console.print("[dim]No results[/dim]")


def search_messages(backend, query: str, limit: int):
    """Message search."""
    console.print(f"[bold]Message Search:[/bold] \"{query}\"\n")
    results = backend.search_messages_by_text(query, limit)
    if results:
        display_message_results(results)
    else:
        console.print("[dim]No results[/dim]")


def display_results(results: list, method: str):
    """Display memory results in a table."""
    table = Table()
    table.add_column("Rel", justify="right", style="cyan", width=5)
    table.add_column("Imp", justify="right", style="yellow", width=4)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Content", style="white", max_width=80)
    
    for r in results:
        content = r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
        relevance = r.get("relevance", r.get("similarity", 0))
        table.add_row(
            f"{relevance:.3f}",
            f"{r['importance']:.2f}",
            r["memory_type"],
            content,
        )
    
    console.print(table)
    console.print(f"[dim]Found {len(results)} results via {method}[/dim]\n")


def display_message_results(results: list):
    """Display message results in a table."""
    table = Table()
    table.add_column("Rel", justify="right", style="cyan", width=5)
    table.add_column("Role", style="magenta", width=10)
    table.add_column("Content", style="white", max_width=80)
    
    for r in results:
        content = r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
        relevance = r.get("relevance", 0)
        table.add_row(
            f"{relevance:.3f}",
            r.get("role", "?"),
            content,
        )
    
    console.print(table)
    console.print(f"[dim]Found {len(results)} message results[/dim]\n")


def handle_regenerate_embeddings(backend, config):
    """Regenerate embeddings for all memories."""
    stats = backend.stats()
    mem_stats = stats.get("memories", {})
    total = mem_stats.get("total_count", 0)
    
    if total == 0:
        console.print("[yellow]No memories found to process.[/yellow]")
        return
    
    console.print(f"Found [bold]{total}[/bold] memories to process")
    console.print(f"Using embedding model: [cyan]{backend.embedding_model}[/cyan]")
    console.print()
    
    with console.status("[bold green]Generating embeddings..."):
        result = backend.regenerate_embeddings()
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        console.print("Install with: [cyan]pipx runpip ask-llm install sentence-transformers[/cyan]")
    else:
        console.print(f"[green]✓ Updated: {result['updated']}[/green]")
        if result.get("failed", 0) > 0:
            console.print(f"[yellow]! Failed: {result['failed']}[/yellow]")
        console.print(f"  Embedding dimension: {result.get('embedding_dim', 'unknown')}")


def handle_consolidate(backend, config, dry_run: bool = False):
    """Find and merge redundant memories."""
    from .memory.consolidation import MemoryConsolidator, get_local_llm_client
    
    mode_str = "[yellow](DRY RUN)[/yellow] " if dry_run else ""
    console.print(f"[bold]{mode_str}Memory Consolidation[/bold]\n")
    
    # Show current stats
    stats = backend.stats()
    mem_stats = stats.get("memories", {})
    total = mem_stats.get("total_count", 0)
    
    if total < 2:
        console.print("[yellow]Not enough memories to consolidate (need at least 2).[/yellow]")
        return
    
    console.print(f"Found [bold]{total}[/bold] total memories")
    console.print(f"Similarity threshold: [cyan]{config.MEMORY_CONSOLIDATION_THRESHOLD}[/cyan]")
    
    # Try to get a local LLM for intelligent merging (skip for dry_run)
    llm_client = None
    if not dry_run:
        with console.status("[bold green]Loading local LLM for merging..."):
            llm_client = get_local_llm_client(config)
    
    if llm_client:
        console.print(f"Using LLM: [green]✓ Local model loaded[/green]")
    else:
        console.print(f"Using LLM: [yellow]✗ None (will use heuristic merging)[/yellow]")
    console.print()
    
    # Create consolidator and run
    consolidator = MemoryConsolidator(
        backend=backend,
        llm_client=llm_client,
        similarity_threshold=config.MEMORY_CONSOLIDATION_THRESHOLD,
        config=config,
    )
    
    with console.status("[bold green]Finding similar memory clusters..."):
        memories = consolidator.get_all_active_memories_with_embeddings()
        clusters = consolidator.find_clusters(memories)
    
    if not clusters:
        console.print("[green]No redundant memory clusters found. Memories are already consolidated.[/green]")
        return
    
    console.print(f"Found [bold]{len(clusters)}[/bold] clusters of similar memories:\n")
    
    # Show cluster details
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("#", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Avg Similarity", justify="right")
    table.add_column("Sample Content")
    
    for i, cluster in enumerate(clusters[:20], 1):  # Show first 20
        sample = cluster.memories[0]["content"][:60] + "..." if len(cluster.memories[0]["content"]) > 60 else cluster.memories[0]["content"]
        table.add_row(
            str(i),
            str(len(cluster)),
            cluster.memories[0]["memory_type"],
            f"{cluster.avg_similarity:.2f}",
            sample,
        )
    
    console.print(table)
    
    if len(clusters) > 20:
        console.print(f"[dim]... and {len(clusters) - 20} more clusters[/dim]")
    console.print()
    
    # Run consolidation
    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made.[/yellow]\n")
    
    with console.status("[bold green]Consolidating memories..."):
        result = consolidator.consolidate(dry_run=dry_run)
    
    # Show results
    console.print()
    if dry_run:
        console.print(f"[yellow]Would merge:[/yellow]")
    else:
        console.print(f"[green]Results:[/green]")
    
    console.print(f"  Clusters found: {result.clusters_found}")
    console.print(f"  Clusters merged: {result.clusters_merged}")
    console.print(f"  Memories consolidated: {result.memories_consolidated}")
    if not dry_run:
        console.print(f"  New memories created: {result.new_memories_created}")
    
    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for err in result.errors[:5]:
            console.print(f"  - {err}")


if __name__ == "__main__":
    main()

import os
import sqlite3
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from consistencyguard.store import init_db, get_db_path
from consistencyguard.reporter import (
    print_violations,
    print_summary,
    print_trend,
    print_agent_stats,
    export_violations,
    print_hallucination_diff,
)

load_dotenv()
console = Console()


@click.group()
def cli():
    """ConsistencyGuard — LLM output consistency monitor."""
    init_db()


@cli.command()
@click.option("--limit", default=20, help="Max violations to show")
@click.option("--agent", default=None, help="Filter by agent ID")
@click.option(
    "--severity", default=None,
    type=click.Choice(["critical", "warning", "info"], case_sensitive=False),
    help="Filter by severity",
)
@click.option("--since", default=None, type=int, help="Last N hours only")
def report(limit, agent, severity, since):
    """Show recent consistency violations."""
    print_violations(limit, agent_id=agent, severity=severity, since_hours=since)
    print_summary()


@cli.command()
@click.option("--hours", default=24, help="Lookback window in hours")
def trend(hours):
    """Show hourly violation chart."""
    print_trend(hours)


@cli.command()
@click.option("--hours", default=24, help="Lookback window in hours")
def agents(hours):
    """Show per-agent violation statistics."""
    print_agent_stats(hours)


@cli.command()
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv"], case_sensitive=False),
    default="json",
    help="Output format",
)
@click.option("--output", "-o", default=None, type=click.Path(), help="File to write (default: stdout)")
@click.option("--agent", default=None, help="Filter by agent ID")
@click.option(
    "--severity", default=None,
    type=click.Choice(["critical", "warning", "info"], case_sensitive=False),
)
@click.option("--since", default=None, type=int, help="Last N hours only")
def export(fmt, output, agent, severity, since):
    """Export violations to JSON or CSV."""
    data = export_violations(
        format=fmt, agent_id=agent, severity=severity, since_hours=since
    )
    if output:
        with open(output, "w") as f:
            f.write(data)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        click.echo(data)


@cli.command()
def health():
    """Show system health: DB stats, env config, model status."""
    table = Table(title="ConsistencyGuard Health", show_lines=True)
    table.add_column("Check", width=28)
    table.add_column("Status", width=12)
    table.add_column("Detail", width=40)

    # DB file
    db_path = get_db_path()
    db_exists = os.path.exists(db_path)
    db_size = f"{os.path.getsize(db_path) / 1024:.1f} KB" if db_exists else "—"
    table.add_row(
        "SQLite DB",
        "[green]OK[/green]" if db_exists else "[yellow]NEW[/yellow]",
        f"{db_path}  ({db_size})",
    )

    # Table counts
    try:
        with sqlite3.connect(db_path) as conn:
            calls = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
            viols = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        table.add_row("LLM calls stored", "[green]OK[/green]", str(calls))
        table.add_row("Violations stored", "[green]OK[/green]", str(viols))
    except Exception as e:
        table.add_row("DB tables", "[red]ERROR[/red]", str(e))

    # Env vars
    for var, default in [
        ("PROVIDER", "anthropic"),
        ("MODEL", "claude-haiku-4-5-20251001"),
        ("SIMILARITY_THRESHOLD", "0.92"),
        ("DIVERGENCE_THRESHOLD", "0.25"),
        ("COMPARISON_WINDOW_DAYS", "unlimited"),
        ("WEBHOOK_URL", "not set"),
    ]:
        val = os.getenv(var, default)
        table.add_row(var, "[dim]env[/dim]", val)

    # API key presence
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        present = bool(os.getenv(key))
        table.add_row(
            key,
            "[green]set[/green]" if present else "[dim]not set[/dim]",
            "***" if present else "—",
        )

    # Embedding model
    try:
        from consistencyguard.embedder import get_model
        m = get_model()
        table.add_row("Embedding model", "[green]loaded[/green]", "all-MiniLM-L6-v2")
    except Exception as e:
        table.add_row("Embedding model", "[red]ERROR[/red]", str(e))

    console.print(table)


@cli.command()
@click.argument("prompt")
@click.option("--agent-id", default="cli-test", help="Agent identifier")
@click.option("--model", default=None, help="Model name override")
@click.option(
    "--provider", default=None,
    type=click.Choice(["anthropic", "openai", "gemini"], case_sensitive=False),
    help="LLM provider (default: PROVIDER env var or 'anthropic')",
)
def check(prompt, agent_id, model, provider):
    """
    Send a prompt through the guard and show result.
    Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY) in .env.
    """
    from consistencyguard.proxy import guarded_call

    console.print(f"[dim]Sending prompt: {prompt[:80]}[/dim]")
    response, violations = guarded_call(
        prompt=prompt,
        agent_id=agent_id,
        model=model,
        provider=provider,
    )

    console.print(f"\n[bold]Response:[/bold] {response}\n")

    if violations:
        console.print(
            f"[bold red]⚠ {len(violations)} violation(s) detected![/bold red]"
        )
        for v in violations:
            console.print(f"  [{v.severity.value}] {v.explanation}")
    else:
        console.print("[green]✓ No consistency violations.[/green]")


@cli.command()
@click.argument("prompt")
@click.option("--runs", default=10, show_default=True, help="Number of times to run the prompt")
@click.option("--model", default=None, help="Model name override")
@click.option(
    "--provider", default=None,
    type=click.Choice(["anthropic", "openai", "gemini"], case_sensitive=False),
    help="LLM provider",
)
@click.option("--outlier-threshold", default=0.25, show_default=True,
              help="Divergence from median above which a run is flagged as outlier")
def reliability(prompt, runs, model, provider, outlier_threshold):
    """
    Run PROMPT N times and measure output variance.

    Produces a reliability score (0.0–1.0), per-run divergence,
    pairwise matrix, and outlier detection.

    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.

    Example:

        cg reliability "What is the refund policy?" --runs 5
    """
    from consistencyguard.hallucination_diff import run_reliability_test

    console.print(f"[dim]Running prompt {runs}× — this may take a moment...[/dim]\n")

    report = run_reliability_test(
        prompt=prompt,
        runs=runs,
        model=model,
        provider=provider,
        outlier_threshold=outlier_threshold,
    )
    print_hallucination_diff(report)

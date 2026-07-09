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
    export_violations_html,
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
    type=click.Choice(["json", "csv", "html", "markdown"], case_sensitive=False),
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
@click.argument("prompt")
@click.option(
    "--providers", "-p",
    default="groq,gemini",
    show_default=True,
    help="Comma-separated provider list (e.g. groq,gemini)",
)
@click.option("--drift-threshold", default=0.25, show_default=True,
              help="Pairwise divergence above which DRIFT is declared")
def crosscheck(prompt, providers, drift_threshold):
    """
    Run PROMPT on multiple free providers in parallel and compare responses.

    Example:

        cg crosscheck "What is the capital of France?" --providers groq,gemini
    """
    from consistencyguard.cross_check import cross_provider_check
    from rich.table import Table

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    console.print(f"[dim]Running prompt on {provider_list}…[/dim]\n")

    try:
        report = cross_provider_check(
            prompt=prompt,
            providers=provider_list,
            drift_threshold=drift_threshold,
        )
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1)

    verdict_color = "green" if report.verdict == "AGREEMENT" else "bold red"

    # Provider responses table
    resp_table = Table(title="Provider Responses", show_lines=True)
    resp_table.add_column("Provider", width=14)
    resp_table.add_column("Response", width=80)
    for prov, text in report.provider_responses.items():
        resp_table.add_row(prov, text[:300] + ("…" if len(text) > 300 else ""))
    console.print(resp_table)

    # Pairwise divergence table
    div_table = Table(title="Pairwise Divergence", show_lines=False)
    div_table.add_column("Pair", width=24)
    div_table.add_column("Divergence", justify="right", width=12)
    div_table.add_column("Status", width=14)
    for pair, div in report.pairwise_divergence.items():
        status = "[red]DRIFT[/red]" if div > drift_threshold else "[green]OK[/green]"
        div_table.add_row(pair, f"{div:.4f}", status)
    console.print(div_table)

    console.print(
        f"\n[bold]Agreement Score:[/bold] {report.agreement_score:.4f}  "
        f"[bold]Verdict:[/bold] [{verdict_color}]{report.verdict}[/{verdict_color}]\n"
    )


@cli.command()
def health():
    """Show system health: DB stats, env config, model status. Exit 0=healthy, 1=unhealthy."""
    table = Table(title="ConsistencyGuard Health", show_lines=True)
    table.add_column("Check", width=28)
    table.add_column("Status", width=12)
    table.add_column("Detail", width=40)

    failures = []

    # DB file
    db_path = get_db_path()
    db_exists = os.path.exists(db_path)
    db_size = f"{os.path.getsize(db_path) / 1024:.1f} KB" if db_exists else "—"
    table.add_row(
        "SQLite DB",
        "[green]OK[/green]" if db_exists else "[yellow]NEW[/yellow]",
        f"{db_path}  ({db_size})",
    )

    # DB writable check
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS _health_probe (id INTEGER)")
            conn.execute("DELETE FROM _health_probe")
        table.add_row("DB writable", "[green]OK[/green]", "write test passed")
    except Exception as e:
        table.add_row("DB writable", "[red]FAIL[/red]", str(e))
        failures.append("db_writable")

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

    # API key presence + reachability
    _PROVIDER_URLS = {
        "ANTHROPIC_API_KEY": "https://api.anthropic.com",
        "OPENAI_API_KEY": "https://api.openai.com",
        "GEMINI_API_KEY": "https://generativelanguage.googleapis.com",
    }
    any_key_set = False
    for key, base_url in _PROVIDER_URLS.items():
        present = bool(os.getenv(key))
        if present:
            any_key_set = True
            # HEAD reachability check with 2s timeout
            try:
                import httpx
                with httpx.Client(timeout=2.0) as client:
                    client.head(base_url)
                table.add_row(key, "[green]set + reachable[/green]", "***")
            except Exception as exc:
                table.add_row(key, "[yellow]set, unreachable[/yellow]", str(exc)[:38])
                failures.append(f"{key}_unreachable")
        else:
            table.add_row(key, "[dim]not set[/dim]", "—")

    if not any_key_set:
        failures.append("no_api_key")

    # Embedding model
    try:
        from consistencyguard.embedder import get_model
        get_model()
        table.add_row("Embedding model", "[green]loaded[/green]", "all-MiniLM-L6-v2")
    except Exception as e:
        table.add_row("Embedding model", "[red]ERROR[/red]", str(e))
        failures.append("embedding_model")

    console.print(table)

    if failures:
        console.print(f"[red]Unhealthy — failed checks: {', '.join(failures)}[/red]")
        raise SystemExit(1)


@cli.command()
@click.option("--label", default="", help="Human-readable label for this baseline snapshot")
def pin(label):
    """Pin current DB snapshot to a secret GitHub Gist. Requires GITHUB_TOKEN env var."""
    from consistencyguard.baseline import pin_baseline
    try:
        gist_id = pin_baseline(label=label)
        console.print(f"[green]Baseline pinned.[/green] Gist ID: {gist_id}")
        console.print(f"[dim]https://gist.github.com/{gist_id}[/dim]")
    except EnvironmentError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[dim]Set GITHUB_TOKEN=<your PAT with 'gist' scope> and retry.[/dim]")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[red]Error pinning baseline: {exc}[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("gist_id")
def restore(gist_id):
    """Fetch a pinned baseline snapshot from GitHub Gist and print its summary."""
    from consistencyguard.baseline import restore_baseline
    try:
        snapshot = restore_baseline(gist_id=gist_id)
        label = snapshot.get("label", "(no label)")
        pinned_at = snapshot.get("pinned_at", "unknown")
        agents = snapshot.get("agents", {})
        thresholds = snapshot.get("thresholds", {})
        console.print(f"[bold]Baseline snapshot:[/bold] {label}")
        console.print(f"[dim]Pinned at: {pinned_at}[/dim]")
        console.print(f"Agents in snapshot: {len(agents)}")
        for aid in agents:
            console.print(f"  [cyan]{aid}[/cyan]")
        console.print(f"Thresholds: {thresholds}")
    except EnvironmentError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[dim]Set GITHUB_TOKEN=<your PAT with 'gist' scope> and retry.[/dim]")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[red]Error restoring baseline: {exc}[/red]")
        raise SystemExit(1)


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

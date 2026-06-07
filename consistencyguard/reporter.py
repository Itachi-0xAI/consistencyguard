import csv
import io
import json
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from consistencyguard.store import (
    get_all_violations,
    get_stats,
    get_trend_data,
    get_agent_stats,
    get_violations_filtered,
)

console = Console()

SEV_COLOR = {
    "critical": "bold red",
    "warning": "yellow",
    "info": "cyan",
}


def print_violations(
    limit: int = 20,
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    since_hours: Optional[int] = None,
) -> None:
    if agent_id or severity or since_hours:
        violations = get_violations_filtered(
            agent_id=agent_id, severity=severity, since_hours=since_hours, limit=limit
        )
    else:
        violations = get_all_violations()[:limit]

    if not violations:
        console.print(
            Panel("[green]No consistency violations detected.[/green]",
                  title="ConsistencyGuard")
        )
        return

    table = Table(
        title=f"ConsistencyGuard — Last {len(violations)} Violations",
        show_lines=True,
    )
    table.add_column("Severity", style="bold", width=10)
    table.add_column("Agent", width=14)
    table.add_column("Prompt (truncated)", width=35)
    table.add_column("Divergence", justify="right", width=10)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Timestamp", width=20)

    for v in violations:
        sev = v["severity"]
        color = SEV_COLOR.get(sev, "white")
        table.add_row(
            Text(sev.upper(), style=color),
            v["agent_id"],
            v["new_prompt"][:60] + "...",
            f"{v['response_divergence']:.2f}",
            f"{v['prompt_similarity']:.2f}",
            v["timestamp"][:19],
        )

    console.print(table)


def print_summary() -> None:
    stats = get_stats()
    rate = (
        stats["total_violations"] / stats["total_calls"] * 100
        if stats["total_calls"] > 0 else 0
    )

    content = (
        f"[bold]Total LLM Calls:[/bold]     {stats['total_calls']}\n"
        f"[bold]Total Violations:[/bold]    {stats['total_violations']}\n"
        f"[bold red]Critical:[/bold red]           {stats['critical']}\n"
        f"[bold yellow]Warning:[/bold yellow]            {stats['warning']}\n"
        f"[bold cyan]Info:[/bold cyan]               {stats['info']}\n"
        f"[bold]Violation Rate:[/bold]      {rate:.1f}%"
    )
    console.print(Panel(content, title="ConsistencyGuard Summary"))


def print_trend(hours: int = 24) -> None:
    """Render an hourly violation bar chart for the last N hours."""
    data = get_trend_data(hours=hours)

    if not data:
        console.print(Panel(
            f"[dim]No violations recorded in the last {hours} hours.[/dim]",
            title=f"Violation Trend (last {hours}h)",
        ))
        return

    max_count = max(d["count"] for d in data) or 1
    bar_width = 30

    table = Table(title=f"Violation Trend — last {hours}h", show_lines=False, box=None)
    table.add_column("Hour", width=14)
    table.add_column("Count", justify="right", width=6)
    table.add_column("", width=bar_width + 2)
    table.add_column("C/W/I", width=10)

    for bucket in data:
        n = bucket["count"]
        bar_len = max(1, int(n / max_count * bar_width)) if n > 0 else 0
        bar = ("█" * bar_len) if n > 0 else "[dim]·[/dim]"
        color = "red" if bucket["critical"] else "yellow" if bucket["warning"] else "cyan"
        table.add_row(
            bucket["bucket"],
            str(n),
            f"[{color}]{bar}[/{color}]" if n > 0 else bar,
            f"{bucket['critical']}/{bucket['warning']}/{bucket['info']}",
        )

    console.print(table)


def print_agent_stats(hours: int = 24) -> None:
    """Render a per-agent violation breakdown table."""
    stats = get_agent_stats(hours=hours)

    if not stats:
        console.print(Panel(
            f"[dim]No violation data in the last {hours} hours.[/dim]",
            title="Agent Statistics",
        ))
        return

    table = Table(title=f"Agent Statistics — last {hours}h", show_lines=True)
    table.add_column("Agent ID", width=20)
    table.add_column("Calls", justify="right", width=7)
    table.add_column("Violations", justify="right", width=10)
    table.add_column("Critical", justify="right", width=9)
    table.add_column("Warning", justify="right", width=8)
    table.add_column("Info", justify="right", width=6)
    table.add_column("Rate", justify="right", width=8)
    table.add_column("Last Violation", width=20)

    for s in stats:
        rate = s["violation_rate"]
        rate_color = "red" if rate > 20 else "yellow" if rate > 5 else "green"
        table.add_row(
            s["agent_id"],
            str(s["total_calls"]),
            str(s["total_violations"]),
            str(s["critical"]),
            str(s["warning"]),
            str(s["info"]),
            f"[{rate_color}]{rate:.1f}%[/{rate_color}]",
            (s["last_violation"] or "")[:19],
        )

    console.print(table)


def print_hallucination_diff(report) -> None:
    """Render a HallucinationDiffReport to the terminal."""
    from consistencyguard.models import ViolationSeverity

    VERDICT_COLOR = {
        "RELIABLE": "bold green",
        "UNSTABLE": "bold yellow",
        "CRITICAL": "bold red",
    }
    SEV_SYMBOL = {
        ViolationSeverity.INFO: ("·", "cyan"),
        ViolationSeverity.WARNING: ("▲", "yellow"),
        ViolationSeverity.CRITICAL: ("✖", "bold red"),
    }

    verdict_color = VERDICT_COLOR.get(report.verdict, "white")

    # ── header panel ─────────────────────────────────────────────────────────
    header = (
        f"[bold]Prompt:[/bold]  {report.prompt[:100]}\n"
        f"[bold]Model:[/bold]   {report.model}\n"
        f"[bold]Runs:[/bold]    {report.runs}\n\n"
        f"[bold]Reliability Score:[/bold]  [{verdict_color}]{report.reliability_score:.2f} / 1.00[/{verdict_color}]\n"
        f"[bold]Verdict:[/bold]            [{verdict_color}]{report.verdict}[/{verdict_color}]\n\n"
        f"[dim]Mean pairwise divergence:  {report.mean_pairwise_divergence:.4f}[/dim]\n"
        f"[dim]Max  pairwise divergence:  {report.max_pairwise_divergence:.4f}[/dim]\n"
        f"[dim]Outlier runs (≥{report.outlier_threshold:.2f} from median): {report.outlier_count} / {report.runs}[/dim]"
    )
    console.print(Panel(header, title="Hallucination Diff Report"))

    # ── per-run table ─────────────────────────────────────────────────────────
    table = Table(title="Per-Run Results", show_lines=True)
    table.add_column("Run", width=5, justify="right")
    table.add_column("Div. from Median", width=17, justify="right")
    table.add_column("Severity", width=10)
    table.add_column("Outlier", width=8, justify="center")
    table.add_column("Response (truncated)", width=60)

    for r in report.run_results:
        sym, color = SEV_SYMBOL.get(r.severity, ("?", "white"))
        is_median = r.response == report.median_response
        table.add_row(
            str(r.run_index),
            f"[{color}]{r.divergence_from_median:.4f}[/{color}]",
            f"[{color}]{sym} {r.severity.value.upper()}[/{color}]",
            "[dim]median[/dim]" if is_median else ("[red]YES[/red]" if r.is_outlier else "[green]no[/green]"),
            r.response[:120] + ("…" if len(r.response) > 120 else ""),
        )

    console.print(table)

    # ── pairwise matrix ───────────────────────────────────────────────────────
    if report.runs <= 10:
        console.print(Rule("Pairwise Divergence Matrix"))
        n = report.runs
        header_row = "     " + "".join(f"  R{j+1:02d}" for j in range(n))
        console.print(f"[dim]{header_row}[/dim]")
        for i in range(n):
            row_str = f"[dim]R{i+1:02d}[/dim] "
            for j in range(n):
                val = report.pairwise_matrix[i][j]
                if i == j:
                    row_str += " [dim] ——[/dim]"
                elif val >= 0.40:
                    row_str += f" [bold red]{val:.2f}[/bold red]"
                elif val >= 0.25:
                    row_str += f" [yellow]{val:.2f}[/yellow]"
                else:
                    row_str += f" [green]{val:.2f}[/green]"
            console.print(row_str)

    # ── median response ───────────────────────────────────────────────────────
    console.print(Rule("Median Response (most representative)"))
    console.print(Panel(report.median_response[:500], border_style="dim"))


def export_violations(
    format: str = "json",
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    since_hours: Optional[int] = None,
    limit: int = 10_000,
) -> str:
    """
    Return violations serialized as JSON or CSV string.
    Does not write to disk — caller handles output/file writing.
    """
    rows = get_violations_filtered(
        agent_id=agent_id, severity=severity, since_hours=since_hours, limit=limit
    )

    if format == "json":
        return json.dumps(rows, indent=2, default=str)

    # CSV
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue()

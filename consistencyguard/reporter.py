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


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ConsistencyGuard — Violation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1 {{ color: #f8fafc; font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 0.9rem; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e2330;
           border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.4); }}
  th {{ background: #273042; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: .05em; padding: 10px 14px; text-align: left; }}
  td {{ padding: 10px 14px; border-top: 1px solid #273042; font-size: 0.85rem;
        vertical-align: top; word-break: break-word; max-width: 300px; }}
  tr:hover td {{ background: #273042; }}
  .sev-critical {{ color: #f87171; font-weight: 700; }}
  .sev-warning  {{ color: #fbbf24; font-weight: 700; }}
  .sev-info     {{ color: #38bdf8; font-weight: 700; }}
  .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .footer {{ margin-top: 16px; color: #475569; font-size: 0.78rem; }}
</style>
</head>
<body>
<h1>ConsistencyGuard — Violation Report</h1>
<p class="subtitle">Generated {generated_at} &bull; {count} violation(s)</p>
<table>
  <thead>
    <tr>
      <th>Severity</th><th>Agent</th><th>Prompt</th>
      <th>Divergence</th><th>Similarity</th><th>Timestamp</th><th>Explanation</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
<p class="footer">ConsistencyGuard &mdash; LLM output consistency monitor</p>
</body>
</html>"""


def export_violations_html(
    violations: Optional[list] = None,
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    since_hours: Optional[int] = None,
    limit: int = 10_000,
) -> str:
    """
    Return a standalone HTML string for the given violations list.
    If *violations* is None the store is queried with the filter kwargs.
    """
    import html as _html
    from datetime import datetime, timezone

    if violations is None:
        rows_data = get_violations_filtered(
            agent_id=agent_id, severity=severity, since_hours=since_hours, limit=limit
        )
    else:
        rows_data = violations

    def _sev_class(s: str) -> str:
        return {"critical": "sev-critical", "warning": "sev-warning", "info": "sev-info"}.get(
            str(s).lower(), ""
        )

    html_rows: list[str] = []
    for v in rows_data:
        sev = str(v.get("severity", "")).lower() if isinstance(v, dict) else str(getattr(v, "severity", "")).lower()
        agent = _html.escape(str(v.get("agent_id", "") if isinstance(v, dict) else getattr(v, "agent_id", "")))
        prompt_raw = v.get("new_prompt", "") if isinstance(v, dict) else getattr(v, "new_prompt", "")
        prompt = _html.escape(str(prompt_raw)[:120])
        div_val = v.get("response_divergence", 0) if isinstance(v, dict) else getattr(v, "response_divergence", 0)
        sim_val = v.get("prompt_similarity", 0) if isinstance(v, dict) else getattr(v, "prompt_similarity", 0)
        ts = str(v.get("timestamp", "") if isinstance(v, dict) else getattr(v, "timestamp", ""))[:19]
        expl_raw = v.get("explanation", "") if isinstance(v, dict) else getattr(v, "explanation", "")
        expl = _html.escape(str(expl_raw)[:200])

        html_rows.append(
            f'    <tr>'
            f'<td class="{_sev_class(sev)}">{_html.escape(sev.upper())}</td>'
            f'<td>{agent}</td>'
            f'<td>{prompt}{"…" if len(str(prompt_raw)) > 120 else ""}</td>'
            f'<td class="num">{float(div_val):.3f}</td>'
            f'<td class="num">{float(sim_val):.3f}</td>'
            f'<td>{_html.escape(ts)}</td>'
            f'<td>{expl}</td>'
            f'</tr>'
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _HTML_TEMPLATE.format(
        generated_at=now,
        count=len(rows_data),
        rows="\n".join(html_rows) if html_rows else '    <tr><td colspan="7" style="text-align:center;color:#64748b">No violations found.</td></tr>',
    )


_SEV_EMOJI = {
    "critical": "🔴",
    "warning": "🟡",
    "info": "🔵",
}


def export_violations_markdown(rows: list[dict]) -> str:
    """
    Return a GitHub-pasteable Markdown table for the given violation rows.
    Columns: Severity | Agent | Prompt (truncated 60ch) | Divergence | Timestamp
    Includes a summary header line with totals.
    """
    from datetime import datetime, timezone

    counts = {"critical": 0, "warning": 0, "info": 0}
    for v in rows:
        sev = str(v.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = (
        f"**ConsistencyGuard Violation Report** — {now} — "
        f"{len(rows)} violation(s): "
        f"🔴 {counts['critical']} critical / "
        f"🟡 {counts['warning']} warning / "
        f"🔵 {counts['info']} info"
    )

    header = "| Severity | Agent | Prompt | Divergence | Timestamp |"
    separator = "| --- | --- | --- | --- | --- |"

    table_rows: list[str] = []
    for v in rows:
        sev = str(v.get("severity", "")).lower()
        emoji = _SEV_EMOJI.get(sev, "⚪")
        agent = str(v.get("agent_id", ""))
        prompt_raw = str(v.get("new_prompt", ""))
        prompt_trunc = prompt_raw[:60] + ("…" if len(prompt_raw) > 60 else "")
        div_val = v.get("response_divergence", 0)
        ts = str(v.get("timestamp", ""))[:19]
        table_rows.append(
            f"| {emoji} {sev.upper()} | {agent} | {prompt_trunc} | {float(div_val):.3f} | {ts} |"
        )

    parts = [summary, "", header, separator] + table_rows
    return "\n".join(parts)


def export_violations(
    format: str = "json",
    agent_id: Optional[str] = None,
    severity: Optional[str] = None,
    since_hours: Optional[int] = None,
    limit: int = 10_000,
) -> str:
    """
    Return violations serialized as JSON, CSV, HTML, or Markdown string.
    Does not write to disk — caller handles output/file writing.
    """
    rows = get_violations_filtered(
        agent_id=agent_id, severity=severity, since_hours=since_hours, limit=limit
    )

    if format == "json":
        return json.dumps(rows, indent=2, default=str)

    if format == "html":
        return export_violations_html(
            violations=rows,
        )

    if format == "markdown":
        return export_violations_markdown(rows)

    # CSV
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue()

"""
Tests for reporter.export_violations_markdown.
No real DB or HTTP calls.
"""
from __future__ import annotations

from consistencyguard.reporter import export_violations_markdown


def _make_row(severity: str, agent: str, prompt: str, divergence: float, ts: str) -> dict:
    return {
        "severity": severity,
        "agent_id": agent,
        "new_prompt": prompt,
        "response_divergence": divergence,
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_table_structure():
    """Output must contain the Markdown table header and one data row per violation."""
    rows = [
        _make_row("critical", "support-bot", "What is the refund policy?", 0.43, "2026-06-01T10:00:00"),
        _make_row("warning", "billing-bot", "How do I cancel?", 0.27, "2026-06-01T11:00:00"),
    ]
    md = export_violations_markdown(rows)

    assert "| Severity | Agent | Prompt | Divergence | Timestamp |" in md
    assert "| --- |" in md
    # Two data rows
    data_lines = [l for l in md.splitlines() if l.startswith("| ") and "Severity" not in l and "---" not in l]
    assert len(data_lines) == 2
    # Summary header
    assert "2 violation(s)" in md
    assert "support-bot" in md
    assert "billing-bot" in md


def test_empty_rows():
    """Empty input must still produce a valid (zero-row) Markdown table."""
    md = export_violations_markdown([])

    assert "| Severity | Agent | Prompt | Divergence | Timestamp |" in md
    assert "0 violation(s)" in md
    # No data rows beyond header + separator
    data_lines = [l for l in md.splitlines() if l.startswith("| ") and "Severity" not in l and "---" not in l]
    assert data_lines == []


def test_severity_emoji():
    """Each severity level must map to the correct emoji."""
    rows = [
        _make_row("critical", "a1", "prompt", 0.5, "2026-06-01T00:00:00"),
        _make_row("warning", "a2", "prompt", 0.3, "2026-06-01T00:00:00"),
        _make_row("info", "a3", "prompt", 0.1, "2026-06-01T00:00:00"),
    ]
    md = export_violations_markdown(rows)

    assert "🔴" in md
    assert "🟡" in md
    assert "🔵" in md
    # Summary line emojis
    assert "🔴 1 critical" in md
    assert "🟡 1 warning" in md
    assert "🔵 1 info" in md

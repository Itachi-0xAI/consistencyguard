"""
Baseline pinning via GitHub Gist (free, secret gists, no extra deps — uses httpx).

Functions
---------
pin_baseline(label)   -> gist_id str
restore_baseline(gist_id) -> snapshot dict  (read-only, does NOT mutate DB)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import httpx

from consistencyguard.store import get_db_path

_GIST_API = "https://api.github.com/gists"
_SNAPSHOT_FILENAME = "cg_baseline.json"


def _require_token() -> str:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. "
            "Create a free GitHub PAT with 'gist' scope at "
            "https://github.com/settings/tokens and export it as GITHUB_TOKEN."
        )
    return token


def _build_snapshot(label: str) -> dict[str, Any]:
    """
    Collect one representative embedding per agent (latest call) plus
    current threshold env vars. Returns a pure-JSON-serialisable dict.
    """
    db_path = get_db_path()
    agents: dict[str, Any] = {}

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT agent_id, prompt_embedding, timestamp
                FROM llm_calls
                ORDER BY id DESC
                """
            ).fetchall()
        seen: set[str] = set()
        for agent_id, emb_json, ts in rows:
            if agent_id not in seen:
                seen.add(agent_id)
                # Stores the prompt embedding (not response) — captures what the agent
                # was asked, which is stable across retraining. Use for prompt-space
                # drift detection, not response-drift detection.
                agents[agent_id] = {
                    "embedding": json.loads(emb_json),
                    "captured_at": ts,
                }
    except Exception:
        pass  # empty DB is fine

    thresholds = {
        "SIMILARITY_THRESHOLD": os.getenv("SIMILARITY_THRESHOLD", "0.92"),
        "DIVERGENCE_THRESHOLD": os.getenv("DIVERGENCE_THRESHOLD", "0.25"),
        "COMPARISON_WINDOW_DAYS": os.getenv("COMPARISON_WINDOW_DAYS", ""),
    }

    return {
        "label": label,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "agents": agents,
        "thresholds": thresholds,
    }


def pin_baseline(label: str = "") -> str:
    """
    Serialise the current SQLite store snapshot and POST it to GitHub Gists
    as a private (secret) gist.

    Parameters
    ----------
    label : str
        Human-readable tag, e.g. ``"v1.0-known-good"``.

    Returns
    -------
    str
        The gist ID (use ``https://gist.github.com/<id>`` to view).

    Raises
    ------
    EnvironmentError
        If ``GITHUB_TOKEN`` is not set.
    httpx.HTTPStatusError
        On a non-2xx response from the GitHub API.
    """
    token = _require_token()
    snapshot = _build_snapshot(label)
    payload = {
        "description": f"ConsistencyGuard baseline — {label or 'unlabelled'}",
        "public": False,
        "files": {
            _SNAPSHOT_FILENAME: {
                "content": json.dumps(snapshot, indent=2),
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(_GIST_API, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["id"]


def restore_baseline(gist_id: str) -> dict[str, Any]:
    """
    Fetch a previously pinned baseline snapshot from GitHub Gists.

    This function is **read-only**: it does not write anything to the local
    SQLite database.  The caller can inspect or replay the returned dict.

    Parameters
    ----------
    gist_id : str
        The gist ID returned by :func:`pin_baseline`.

    Returns
    -------
    dict
        The parsed snapshot (keys: ``label``, ``pinned_at``, ``agents``,
        ``thresholds``).

    Raises
    ------
    EnvironmentError
        If ``GITHUB_TOKEN`` is not set.
    httpx.HTTPStatusError
        On a non-2xx response from the GitHub API.
    KeyError
        If the gist does not contain the expected snapshot file.
    """
    token = _require_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{_GIST_API}/{gist_id}", headers=headers)
    resp.raise_for_status()
    gist_data = resp.json()
    raw_content = gist_data["files"][_SNAPSHOT_FILENAME]["content"]
    return json.loads(raw_content)

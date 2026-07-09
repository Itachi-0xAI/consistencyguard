"""
Tests for consistencyguard.baseline — all HTTP calls are mocked, no real network.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pin_success(monkeypatch, tmp_path):
    """pin_baseline posts to GitHub Gists and returns the gist ID."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from consistencyguard.store import init_db
    init_db()

    fake_resp = _make_response(201, {"id": "abc123def456"})

    with patch("consistencyguard.baseline.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value = fake_resp

        from consistencyguard.baseline import pin_baseline
        gist_id = pin_baseline(label="v1.0-known-good")

    assert gist_id == "abc123def456"
    post_call = instance.post.call_args
    assert post_call is not None
    payload = post_call.kwargs.get("json") or post_call.args[1]
    assert payload["public"] is False
    assert "cg_baseline.json" in payload["files"]
    snapshot = json.loads(payload["files"]["cg_baseline.json"]["content"])
    assert snapshot["label"] == "v1.0-known-good"
    assert "thresholds" in snapshot


def test_restore_success(monkeypatch, tmp_path):
    """restore_baseline fetches the gist and returns the parsed snapshot dict."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")

    snapshot = {
        "label": "v1.0-known-good",
        "pinned_at": "2026-06-01T00:00:00+00:00",
        "agents": {"support-bot": {"embedding": [0.1, 0.2], "captured_at": "2026-06-01"}},
        "thresholds": {"SIMILARITY_THRESHOLD": "0.92"},
    }
    gist_body = {
        "files": {
            "cg_baseline.json": {"content": json.dumps(snapshot)}
        }
    }
    fake_resp = _make_response(200, gist_body)

    with patch("consistencyguard.baseline.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = fake_resp

        from consistencyguard.baseline import restore_baseline
        result = restore_baseline("abc123def456")

    assert result["label"] == "v1.0-known-good"
    assert "support-bot" in result["agents"]
    assert result["thresholds"]["SIMILARITY_THRESHOLD"] == "0.92"


def test_missing_token_pin(monkeypatch):
    """pin_baseline raises EnvironmentError when GITHUB_TOKEN is absent."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    from consistencyguard.baseline import pin_baseline
    with pytest.raises(EnvironmentError, match="GITHUB_TOKEN"):
        pin_baseline(label="test")


def test_network_error_pin(monkeypatch, tmp_path):
    """pin_baseline propagates httpx errors on network failure."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from consistencyguard.store import init_db
    init_db()

    import httpx

    with patch("consistencyguard.baseline.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.side_effect = httpx.ConnectError("Connection refused")

        from consistencyguard.baseline import pin_baseline
        with pytest.raises(httpx.ConnectError):
            pin_baseline(label="test")

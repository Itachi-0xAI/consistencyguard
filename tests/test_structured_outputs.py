"""
Tests for structured output validation in guarded_call.
No real LLM calls — provider is mocked.
"""

import json
import pytest
from unittest.mock import MagicMock

from consistencyguard.proxy import guarded_call
from consistencyguard.models import ValidationIssue


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "structured_test.db"))
    import consistencyguard.proxy as _proxy
    _proxy._db_initialized = False
    yield


def _mock_provider(response_text: str) -> MagicMock:
    p = MagicMock()
    p.complete.return_value = response_text
    return p


def test_semantically_equal_json_different_key_order_zero_divergence():
    """Two calls returning the same JSON with different key order produce zero divergence."""
    schema = {"name": str, "value": int}

    resp_a = json.dumps({"name": "alpha", "value": 42})
    resp_b = json.dumps({"value": 42, "name": "alpha"})  # different key order

    provider_a = _mock_provider(resp_a)
    provider_b = _mock_provider(resp_b)

    _, violations_a = guarded_call(
        "Return a JSON with name and value.",
        provider=provider_a,
        agent_id="schema-agent",
        expected_schema=schema,
    )
    # First call — no history, no violations expected
    assert violations_a == []

    _, violations_b = guarded_call(
        "Return a JSON with name and value.",
        provider=provider_b,
        agent_id="schema-agent",
        expected_schema=schema,
    )
    # Second call matches same prompt; canonical form is identical → zero divergence
    assert violations_b == []


def test_invalid_json_sets_schema_status():
    """When the response is not valid JSON, a violation with INVALID_JSON issue is returned."""
    schema = {"name": str}
    provider = _mock_provider("this is not json at all")

    _, violations = guarded_call(
        "Give me JSON.",
        provider=provider,
        agent_id="schema-agent",
        expected_schema=schema,
    )

    assert len(violations) >= 1
    all_issues = [issue for v in violations for issue in v.validation_issues]
    assert ValidationIssue.INVALID_JSON in all_issues


def test_missing_key_produces_schema_mismatch():
    """When a required key is absent from the JSON, SCHEMA_MISMATCH issue is attached."""
    schema = {"name": str, "score": float}
    provider = _mock_provider(json.dumps({"name": "test"}))  # missing "score"

    _, violations = guarded_call(
        "Give me JSON with name and score.",
        provider=provider,
        agent_id="schema-agent",
        expected_schema=schema,
    )

    assert len(violations) >= 1
    all_issues = [issue for v in violations for issue in v.validation_issues]
    assert ValidationIssue.SCHEMA_MISMATCH in all_issues

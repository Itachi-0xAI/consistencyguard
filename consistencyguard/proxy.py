"""
Core proxy. guarded_call / aguarded_call are the drop-in replacement
for direct LLM API calls. Both return (response_text, violations).
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Union, Optional

from consistencyguard.models import LLMCall, ConsistencyViolation, ValidationIssue
from consistencyguard.embedder import embed
from consistencyguard.store import init_db, save_call, save_violation
from consistencyguard.detector import check_consistency
from consistencyguard.providers import get_provider, AnthropicProvider, OpenAIProvider
from consistencyguard.webhooks import fire_webhook, afire_webhook
from consistencyguard.tracing import span as _span

_db_initialized = False

def _validate_schema(response: str, expected_schema: dict) -> tuple[str, str | None]:
    """
    Validate response against expected_schema.
    Returns (schema_status, canonical_text_or_None).
    canonical_text is set only when status == "ok".
    Uses `type(...) is` not `isinstance` to avoid bool/int false-negatives.
    """
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return "invalid_json", None

    for key in expected_schema:
        if key not in parsed:
            return "missing_keys", None

    for key, expected_type in expected_schema.items():
        if isinstance(expected_type, type):
            if type(parsed[key]) is not expected_type:
                return "type_mismatch", None

    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return "ok", canonical


def _ensure_db() -> None:
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True


ProviderArg = Optional[Union[str, AnthropicProvider, OpenAIProvider]]


def guarded_call(
    prompt: str,
    model: str = None,
    max_tokens: int = 1024,
    agent_id: str = "default",
    api_key: str = None,
    provider: ProviderArg = None,
    expected_schema: dict | None = None,
) -> tuple[str, list[ConsistencyViolation]]:
    """
    Drop-in replacement for a direct LLM call.
    Returns (response_text, list_of_violations).

    Args:
        provider: "anthropic" | "openai" | an already-instantiated provider
                  object. Defaults to PROVIDER env var (default: anthropic).
    """
    _ensure_db()

    if model is None:
        model = os.getenv("MODEL", "claude-haiku-4-5-20251001")

    llm = provider if hasattr(provider, "complete") else get_provider(provider, api_key)

    provider_name = type(llm).__name__

    with _span("cg.guarded_call",
               **{"cg.agent_id": agent_id, "cg.model": model, "cg.provider": provider_name}
               ) as active_span:
        prompt_embedding = embed(prompt)
        call = LLMCall(
            prompt=prompt,
            response="",
            model=model,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            prompt_embedding=prompt_embedding,
        )

        response_text = llm.complete(prompt, model, max_tokens)
        call.response = response_text

        # Structured output validation
        _schema_status = None
        if expected_schema is not None and response_text:
            _schema_status, canonical = _validate_schema(response_text, expected_schema)
            if _schema_status == "ok":
                call.response = canonical  # canonical used for embedding/comparison

        call_id = save_call(call)
        call.id = call_id

        violations = check_consistency(call)

        if _schema_status is not None and _schema_status != "ok":
            _issue_map = {
                "invalid_json": ValidationIssue.INVALID_JSON,
                "missing_keys": ValidationIssue.SCHEMA_MISMATCH,
                "type_mismatch": ValidationIssue.SCHEMA_MISMATCH,
            }
            issue = _issue_map.get(_schema_status)
            for v in violations:
                if issue and issue not in v.validation_issues:
                    v.validation_issues.append(issue)
            if not violations:
                from consistencyguard.models import ViolationSeverity
                violations.append(ConsistencyViolation(
                    call_id_new=call_id,
                    call_id_ref=call_id,
                    prompt_similarity=1.0,
                    response_divergence=0.0,
                    severity=ViolationSeverity.INFO,
                    new_prompt=prompt,
                    new_response=response_text,
                    ref_response="",
                    explanation=f"Schema validation failed: {_schema_status}",
                    agent_id=agent_id,
                    validation_issues=[issue] if issue else [],
                ))

        for v in violations:
            v.call_id_new = call_id
            save_violation(v)
            fire_webhook(v)

        if active_span is not None:
            max_div = max((v.response_divergence for v in violations), default=0.0)
            active_span.set_attribute("cg.violations_count", len(violations))
            active_span.set_attribute("cg.max_divergence", max_div)

    return response_text, violations


async def aguarded_call(
    prompt: str,
    model: str = None,
    max_tokens: int = 1024,
    agent_id: str = "default",
    api_key: str = None,
    provider: ProviderArg = None,
    expected_schema: dict | None = None,
) -> tuple[str, list[ConsistencyViolation]]:
    """Async version of guarded_call."""
    _ensure_db()

    if model is None:
        model = os.getenv("MODEL", "claude-haiku-4-5-20251001")

    llm = provider if hasattr(provider, "acomplete") else get_provider(provider, api_key)

    provider_name = type(llm).__name__

    with _span("cg.aguarded_call",
               **{"cg.agent_id": agent_id, "cg.model": model, "cg.provider": provider_name}
               ) as active_span:
        # sentence-transformers is CPU-bound / not async-native — run in thread pool
        loop = asyncio.get_event_loop()
        prompt_embedding = await loop.run_in_executor(None, embed, prompt)

        call = LLMCall(
            prompt=prompt,
            response="",
            model=model,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            prompt_embedding=prompt_embedding,
        )

        response_text = await llm.acomplete(prompt, model, max_tokens)
        call.response = response_text

        # Structured output validation
        _schema_status = None
        if expected_schema is not None and response_text:
            _schema_status, canonical = _validate_schema(response_text, expected_schema)
            if _schema_status == "ok":
                call.response = canonical  # canonical used for embedding/comparison

        call_id = save_call(call)
        call.id = call_id

        violations = await loop.run_in_executor(None, check_consistency, call)

        if _schema_status is not None and _schema_status != "ok":
            _issue_map = {
                "invalid_json": ValidationIssue.INVALID_JSON,
                "missing_keys": ValidationIssue.SCHEMA_MISMATCH,
                "type_mismatch": ValidationIssue.SCHEMA_MISMATCH,
            }
            issue = _issue_map.get(_schema_status)
            for v in violations:
                if issue and issue not in v.validation_issues:
                    v.validation_issues.append(issue)
            if not violations:
                from consistencyguard.models import ViolationSeverity
                violations.append(ConsistencyViolation(
                    call_id_new=call_id,
                    call_id_ref=call_id,
                    prompt_similarity=1.0,
                    response_divergence=0.0,
                    severity=ViolationSeverity.INFO,
                    new_prompt=prompt,
                    new_response=response_text,
                    ref_response="",
                    explanation=f"Schema validation failed: {_schema_status}",
                    agent_id=agent_id,
                    validation_issues=[issue] if issue else [],
                ))

        for v in violations:
            v.call_id_new = call_id
            save_violation(v)
            await afire_webhook(v)

        if active_span is not None:
            max_div = max((v.response_divergence for v in violations), default=0.0)
            active_span.set_attribute("cg.violations_count", len(violations))
            active_span.set_attribute("cg.max_divergence", max_div)

    return response_text, violations

import os
from consistencyguard.embedder import embed, cosine_similarity
from consistencyguard.models import (
    LLMCall, SimilarCall, ConsistencyViolation, ViolationSeverity
)
from consistencyguard.store import get_all_calls
from consistencyguard.tracing import span as _span
from datetime import datetime, timedelta


def find_similar_calls(
    new_call: LLMCall,
    threshold: float = None,
    window_days: int = None,
) -> list[SimilarCall]:
    """
    Find past calls whose prompt embedding is above threshold.

    window_days: only compare against calls from the last N days
    (default: COMPARISON_WINDOW_DAYS env var, or unlimited if unset).
    This prevents stale historical baselines from flagging correct new answers.
    """
    if threshold is None:
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))
    if window_days is None:
        raw = os.getenv("COMPARISON_WINDOW_DAYS")
        window_days = int(raw) if raw else None

    cutoff = (
        datetime.utcnow() - timedelta(days=window_days)
        if window_days else None
    )

    with _span("cg.find_similar") as active_span:
        past_calls = get_all_calls()
        matches = []

        for past in past_calls:
            if past.id == new_call.id:
                continue
            if past.prompt_embedding is None:
                continue
            if cutoff and past.timestamp < cutoff:
                continue
            sim = cosine_similarity(
                new_call.prompt_embedding,
                past.prompt_embedding
            )
            if sim >= threshold:
                matches.append(SimilarCall(
                    call_id=past.id,
                    prompt=past.prompt,
                    response=past.response,
                    similarity_score=round(sim, 4),
                    timestamp=past.timestamp,
                ))

        result = sorted(matches, key=lambda x: x.similarity_score, reverse=True)
        if active_span is not None:
            active_span.set_attribute("cg.candidates_count", len(result))
        return result


def response_divergence(response_a: str, response_b: str) -> float:
    """
    Semantic divergence between two responses.
    Returns 1.0 - cosine_similarity of their embeddings.
    Range: 0.0 (identical) to 1.0 (completely different).
    """
    emb_a = embed(response_a)
    emb_b = embed(response_b)
    similarity = cosine_similarity(emb_a, emb_b)
    return round(1.0 - similarity, 4)


def classify_severity(divergence: float) -> ViolationSeverity:
    if divergence >= 0.40:
        return ViolationSeverity.CRITICAL
    elif divergence >= 0.25:
        return ViolationSeverity.WARNING
    else:
        return ViolationSeverity.INFO


def build_explanation(
    prompt: str,
    response_a: str,
    response_b: str,
    divergence: float,
    severity: ViolationSeverity
) -> str:
    return (
        f"[{severity.value.upper()}] Semantic divergence: {divergence:.2f}. "
        f"Two responses to nearly identical prompts differ significantly. "
        f"Prompt (truncated): '{prompt[:80]}...' "
        f"Response A (truncated): '{response_a[:60]}...' "
        f"Response B (truncated): '{response_b[:60]}...'"
    )


def check_consistency(
    new_call: LLMCall,
    divergence_threshold: float = None
) -> list[ConsistencyViolation]:
    """
    Main detection function.
    1. Find similar past prompts
    2. For each similar prompt, compute response divergence
    3. If divergence > threshold, build a ConsistencyViolation
    Returns list of violations (usually 0 or 1).
    """
    if divergence_threshold is None:
        divergence_threshold = float(
            os.getenv("DIVERGENCE_THRESHOLD", "0.25")
        )

    similar = find_similar_calls(new_call)
    violations = []

    for match in similar:
        div = response_divergence(new_call.response, match.response)
        if div >= divergence_threshold:
            severity = classify_severity(div)
            explanation = build_explanation(
                new_call.prompt,
                new_call.response,
                match.response,
                div,
                severity
            )
            violations.append(ConsistencyViolation(
                call_id_new=new_call.id,
                call_id_ref=match.call_id,
                prompt_similarity=match.similarity_score,
                response_divergence=div,
                severity=severity,
                new_prompt=new_call.prompt,
                new_response=new_call.response,
                ref_response=match.response,
                explanation=explanation,
                agent_id=new_call.agent_id or "default",
            ))

    return violations

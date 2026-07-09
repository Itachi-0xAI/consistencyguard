"""
Hallucination Diff — runs the same prompt N times and measures output variance.

Core output:
  - reliability_score: 0.0 (completely unreliable) → 1.0 (perfectly consistent)
  - per-run divergence from the median response
  - pairwise divergence matrix
  - stable vs variable verdict per run
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Union

from consistencyguard.embedder import embed, cosine_similarity
from consistencyguard.detector import response_divergence, classify_severity
from consistencyguard.models import ViolationSeverity
from consistencyguard.providers import get_provider, AnthropicProvider, OpenAIProvider
from consistencyguard.tracing import span as _span

ProviderArg = Optional[Union[str, AnthropicProvider, OpenAIProvider]]


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    run_index: int
    response: str
    divergence_from_median: float        # 0.0 = matches median exactly
    severity: ViolationSeverity
    is_outlier: bool                     # True if divergence > outlier_threshold


@dataclass
class HallucinationDiffReport:
    prompt: str
    model: str
    runs: int
    responses: list[str]
    run_results: list[RunResult]
    reliability_score: float             # 1.0 - mean_pairwise_divergence
    mean_pairwise_divergence: float
    max_pairwise_divergence: float
    min_pairwise_divergence: float
    pairwise_matrix: list[list[float]]   # [i][j] = divergence(i, j)
    median_response: str                 # response closest to all others
    outlier_count: int
    outlier_threshold: float             # stored so reporter can display exact value used
    verdict: str                         # RELIABLE | UNSTABLE | CRITICAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── core logic ────────────────────────────────────────────────────────────────

def _find_median_idx(embeddings: list) -> int:
    """Return index of embedding with lowest average divergence from all others."""
    n = len(embeddings)
    avg_divs = [
        sum(1.0 - cosine_similarity(embeddings[i], embeddings[j]) for j in range(n) if j != i) / (n - 1)
        for i in range(n)
    ]
    return avg_divs.index(min(avg_divs))


def _pairwise_matrix(embeddings: list) -> list[list[float]]:
    """Compute NxN divergence matrix from pre-computed embeddings."""
    n = len(embeddings)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            div = round(max(0.0, 1.0 - cosine_similarity(embeddings[i], embeddings[j])), 4)
            matrix[i][j] = div
            matrix[j][i] = div
    return matrix


def _verdict(reliability_score: float) -> str:
    if reliability_score >= 0.90:
        return "RELIABLE"
    elif reliability_score >= 0.70:
        return "UNSTABLE"
    else:
        return "CRITICAL"


# ── shared helpers ────────────────────────────────────────────────────────────

def _resolve_model(model: str | None) -> str:
    return model or os.getenv("MODEL", "claude-haiku-4-5-20251001")

def _resolve_provider(provider: ProviderArg, api_key: str | None):
    return provider if hasattr(provider, "complete") else get_provider(provider, api_key)


# ── public API ────────────────────────────────────────────────────────────────

def run_reliability_test(
    prompt: str,
    runs: int = 10,
    model: str = None,
    provider: ProviderArg = None,
    api_key: str = None,
    agent_id: str = "reliability-test",
    outlier_threshold: float = 0.25,
) -> HallucinationDiffReport:
    """Run prompt N times (sequentially) and return a HallucinationDiffReport."""
    model = _resolve_model(model)
    llm = _resolve_provider(provider, api_key)
    delay = float(os.getenv("RELIABILITY_RUN_DELAY", "0"))

    with _span("cg.run_reliability_test", **{"cg.runs": runs}) as active_span:
        responses = []
        for i in range(runs):
            responses.append(llm.complete(prompt, model, max_tokens=1024))
            if delay > 0 and i < runs - 1:
                time.sleep(delay)

        report = _build_report(prompt, model, responses, outlier_threshold)

        if active_span is not None:
            active_span.set_attribute("cg.reliability_score", report.reliability_score)
            active_span.set_attribute("cg.verdict", report.verdict)

    return report


async def arun_reliability_test(
    prompt: str,
    runs: int = 10,
    model: str = None,
    provider: ProviderArg = None,
    api_key: str = None,
    outlier_threshold: float = 0.25,
) -> HallucinationDiffReport:
    """Async version — sequential when RELIABILITY_RUN_DELAY > 0 (respects free-tier quotas)."""
    model = _resolve_model(model)
    llm = _resolve_provider(provider, api_key)
    delay = float(os.getenv("RELIABILITY_RUN_DELAY", "0"))

    if delay > 0:
        responses = []
        for i in range(runs):
            responses.append(await llm.acomplete(prompt, model, max_tokens=1024))
            if delay > 0 and i < runs - 1:
                await asyncio.sleep(delay)
    else:
        responses = list(await asyncio.gather(
            *[llm.acomplete(prompt, model, max_tokens=1024) for _ in range(runs)]
        ))
    return _build_report(prompt, model, responses, outlier_threshold)


def _build_report(
    prompt: str,
    model: str,
    responses: list[str],
    outlier_threshold: float,
) -> HallucinationDiffReport:
    """Build the full report. Embeddings computed once and shared across all helpers."""
    n = len(responses)
    embeddings = [embed(r) for r in responses]   # single embedding pass

    matrix = _pairwise_matrix(embeddings)
    median_idx = _find_median_idx(embeddings)
    median_response = responses[median_idx]

    all_divs = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    mean_div = round(sum(all_divs) / len(all_divs), 4) if all_divs else 0.0
    max_div  = round(max(all_divs), 4) if all_divs else 0.0
    min_div  = round(min(all_divs), 4) if all_divs else 0.0
    reliability = round(1.0 - mean_div, 4)

    run_results = [
        RunResult(
            run_index=i + 1,
            response=resp,
            divergence_from_median=round(max(0.0, 1.0 - cosine_similarity(embeddings[i], embeddings[median_idx])), 4),
            severity=classify_severity(round(max(0.0, 1.0 - cosine_similarity(embeddings[i], embeddings[median_idx])), 4)),
            is_outlier=round(max(0.0, 1.0 - cosine_similarity(embeddings[i], embeddings[median_idx])), 4) >= outlier_threshold,
        )
        for i, resp in enumerate(responses)
    ]

    return HallucinationDiffReport(
        prompt=prompt,
        model=model,
        runs=n,
        responses=responses,
        run_results=run_results,
        reliability_score=reliability,
        mean_pairwise_divergence=mean_div,
        max_pairwise_divergence=max_div,
        min_pairwise_divergence=min_div,
        pairwise_matrix=matrix,
        median_response=median_response,
        outlier_count=sum(1 for r in run_results if r.is_outlier),
        outlier_threshold=outlier_threshold,
        verdict=_verdict(reliability),
    )

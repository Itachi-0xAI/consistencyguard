"""
Cross-provider consistency checker.
Runs the same prompt on multiple free providers in parallel and reports
whether they agree or diverge semantically.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from consistencyguard.embedder import embed, cosine_similarity


@dataclass
class CrossProviderReport:
    prompt: str
    provider_responses: dict[str, str]          # provider -> response text
    pairwise_divergence: dict[str, float]        # "A vs B" -> divergence score
    agreement_score: float                       # 0.0 (full drift) – 1.0 (full agreement)
    verdict: str                                 # "AGREEMENT" | "DRIFT"


def _call_provider(provider_name: str, prompt: str, max_tokens: int = 256) -> tuple[str, str]:
    """Call a single provider. Returns (provider_name, response_text)."""
    name = provider_name.lower()

    if name == "groq":
        import httpx
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return provider_name, resp.json()["choices"][0]["message"]["content"]

    if name == "gemini":
        from consistencyguard.providers import GeminiProvider
        p = GeminiProvider()
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return provider_name, p.complete(prompt, model=model, max_tokens=max_tokens)

    if name == "anthropic":
        from consistencyguard.providers import AnthropicProvider
        p = AnthropicProvider()
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return provider_name, p.complete(prompt, model=model, max_tokens=max_tokens)

    if name == "openai":
        from consistencyguard.providers import OpenAIProvider
        p = OpenAIProvider()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return provider_name, p.complete(prompt, model=model, max_tokens=max_tokens)

    raise ValueError(f"Unknown provider '{provider_name}'. Supported: groq, gemini, anthropic, openai")


def cross_provider_check(
    prompt: str,
    providers: Optional[list[str]] = None,
    max_tokens: int = 256,
    drift_threshold: float = 0.25,
) -> CrossProviderReport:
    """
    Run *prompt* on each provider in parallel, compare responses semantically.

    Parameters
    ----------
    prompt:          The prompt to send to all providers.
    providers:       List of provider names. Defaults to ["groq", "gemini"].
    max_tokens:      Max tokens for each provider response.
    drift_threshold: Pairwise divergence above this => DRIFT verdict.

    Returns
    -------
    CrossProviderReport
    """
    if providers is None:
        providers = ["groq", "gemini"]

    if len(providers) < 2:
        raise ValueError("cross_provider_check requires at least 2 providers")

    provider_responses: dict[str, str] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {
            pool.submit(_call_provider, p, prompt, max_tokens): p
            for p in providers
        }
        for future in as_completed(futures):
            pname = futures[future]
            try:
                _, text = future.result()
                provider_responses[pname] = text
            except Exception as exc:
                errors[pname] = str(exc)

    if errors:
        err_summary = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(f"Provider call(s) failed — {err_summary}")

    # Compute embeddings once per provider
    embeddings = {p: embed(t) for p, t in provider_responses.items()}

    # Pairwise divergence (1 - cosine_similarity)
    provider_list = list(provider_responses.keys())
    pairwise_divergence: dict[str, float] = {}
    divergence_values: list[float] = []

    for i in range(len(provider_list)):
        for j in range(i + 1, len(provider_list)):
            a, b = provider_list[i], provider_list[j]
            div = 1.0 - cosine_similarity(embeddings[a], embeddings[b])
            div = max(0.0, min(1.0, div))
            pairwise_divergence[f"{a} vs {b}"] = round(div, 4)
            divergence_values.append(div)

    mean_div = sum(divergence_values) / len(divergence_values) if divergence_values else 0.0
    agreement_score = round(1.0 - mean_div, 4)
    verdict = "DRIFT" if any(d > drift_threshold for d in divergence_values) else "AGREEMENT"

    return CrossProviderReport(
        prompt=prompt,
        provider_responses=provider_responses,
        pairwise_divergence=pairwise_divergence,
        agreement_score=agreement_score,
        verdict=verdict,
    )

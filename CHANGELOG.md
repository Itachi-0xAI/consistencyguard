# Changelog

All notable changes to ConsistencyGuard are documented here.

---

## [1.0.2] — 2026-05-26

### Added

- `consistencyguard/hallucination_diff.py` — Hallucination Diff engine
  - `run_reliability_test(prompt, runs)` — runs same prompt N times, computes pairwise divergence matrix
  - `reliability_score` (0.0–1.0) — 1 − mean pairwise divergence across all run pairs
  - Verdict classification: RELIABLE (≥0.90) | UNSTABLE (≥0.70) | CRITICAL (<0.70)
  - Median response detection — identifies the most representative run
  - Outlier flagging — runs with divergence from median ≥ threshold marked as outliers
  - Async variant `arun_reliability_test()` — runs all N calls concurrently
- `cg reliability` CLI command — `cg reliability "prompt" --runs 10 --provider gemini`
- `GeminiProvider` — Google Gemini via OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`)
- `OpenAIProvider` now accepts `base_url` for any OpenAI-compatible endpoint (Groq, Together, etc.)
- `RELIABILITY_RUN_DELAY` env var — adds delay between runs to respect free-tier rate limits
- `tests/test_hallucination_diff.py` — 14 tests covering pairwise matrix, median detection, verdict, outlier flagging, report structure

### Fixed

- Pairwise divergence matrix could show `-0.00` due to floating-point cosine similarity slightly exceeding 1.0 — clamped to `max(0.0, ...)`

### Total test count: 45 (up from 31)

---

## [1.0.1] — 2026-05-23

### Added

- `tests/test_user_flows.py` — 7 end-to-end user flow tests simulating real developer integration: first call (no history), consistent follow-ups, contradicting response detection, violation metadata validation, stats aggregation, and cross-agent divergence detection
- Documented global consistency scope: ConsistencyGuard compares new calls against all past calls across all agents, not per-agent; cross-agent violations are intentional and now covered by tests

### Total test count: 31 (up from 24)

---

## [1.0.0] — 2026-05-21

### Added

- `guarded_call()` — synchronous proxy wrapping any LLM call with consistency checking
- `aguarded_call()` — async variant for FastAPI and async frameworks
- Local prompt embedding using `all-MiniLM-L6-v2` (no API cost, no data leaves machine)
- Cosine similarity scan over SQLite to find semantically similar past prompts
- Response divergence scoring with three severity levels: INFO, WARNING, CRITICAL
- SQLite persistence for all LLM calls, embeddings, and violations
- `COMPARISON_WINDOW_DAYS` — time-windowed baseline to prevent stale history from flagging correct updated answers
- Webhook alert dispatch (sync + async via httpx) on every violation
- Anthropic provider support (`AnthropicProvider`)
- OpenAI provider support (`OpenAIProvider`)
- Provider abstraction (`get_provider()`) — swap providers via `.env`, no code change
- CLI commands: `cg report`, `cg health`, `cg trend`, `cg agents`, `cg export`, `cg check`
- Rich terminal tables for violations, trend charts, and per-agent stats
- Zero-API-key demo (`demo/run_demo.py`) — simulates 7 calls with 3 injected violations
- 24-test suite with real embeddings and fully isolated SQLite databases per test
- `docs/FAILURE_ANALYSIS.md` — 8 real failure scenarios with root cause analysis
- MIT license

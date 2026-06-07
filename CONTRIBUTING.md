# Contributing to ConsistencyGuard

Thanks for taking the time to contribute.

---

## Local Setup

```bash
git clone https://github.com/sameerpashasyed17-collab/consistencyguard.git
cd consistencyguard
pip install -e ".[dev]"
cp .env.example .env
# Add your ANTHROPIC_API_KEY or OPENAI_API_KEY to .env if testing real calls
```

## Running Tests

```bash
pytest tests/ -v
```

All 45 tests must pass before opening a PR. Tests use real sentence-transformer
embeddings (no mocking) and fully isolated SQLite databases per test via
`conftest.py` — no API key required.

### Test suites

| File | Tests | What it covers |
|---|---|---|
| `test_detector.py` | 8 | Embedder cosine similarity, divergence scoring, severity classification |
| `test_store.py` | 6 | SQLite schema, save/load calls, violation persistence, stats aggregation |
| `test_providers.py` | 10 | Anthropic + OpenAI sync/async — all mocked, no API keys needed |
| `test_user_flows.py` | 7 | End-to-end developer flows: first call, consistency, violations, cross-agent behaviour |
| `test_hallucination_diff.py` | 14 | Pairwise divergence matrix, median detection, verdict classification, outlier flagging, report structure |

### Important: global consistency scope

ConsistencyGuard compares every new call against **all** past calls in the
database, across all agents. A contradicting response from `agent-b` on a
prompt already answered by `agent-a` **will** trigger a violation. This is
by design — cross-agent inconsistency matters in production. See
`test_user_flows.py::test_cross_agent_divergence_is_detected`.

### Testing with a free LLM API (Groq / Gemini)

The `cg reliability` command requires a real LLM provider. Free options:

**Groq** (recommended — fast, generous free tier):
```bash
# .env
PROVIDER=openai
OPENAI_API_KEY=<your-groq-key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL=llama-3.1-8b-instant
```
Get a free key at https://console.groq.com

**Google Gemini** (free tier with per-minute quota):
```bash
# .env
PROVIDER=gemini
GEMINI_API_KEY=<your-gemini-key>
MODEL=gemini-1.5-flash
```
Get a free key at https://aistudio.google.com

If you hit rate limits on free tiers, add a delay between runs:
```bash
$env:RELIABILITY_RUN_DELAY=3   # seconds between each of the N runs
```

---

## Running the Demo

```bash
python demo/run_demo.py
```

The demo must exit cleanly with 3 violations detected (1 critical, 2 warning).
No API key required.

## Project Structure

```
consistencyguard/
├── consistencyguard/
│   ├── models.py              # Pydantic data models — start here
│   ├── embedder.py            # Local sentence-transformer embedding
│   ├── store.py               # SQLite persistence layer
│   ├── detector.py            # Cosine similarity + divergence logic
│   ├── proxy.py               # guarded_call / aguarded_call entry points
│   ├── providers.py           # Anthropic, OpenAI, Gemini provider abstraction
│   ├── hallucination_diff.py  # reliability test, pairwise matrix, verdict
│   ├── webhooks.py            # Webhook alert dispatch
│   ├── reporter.py            # Rich terminal output (incl. hallucination diff report)
│   └── cli.py                 # Click CLI (cg command, incl. cg reliability)
├── tests/
├── demo/
└── docs/
```

## Guidelines

- **Tests first** — add a test for any new behaviour before writing the implementation
- **No external state in tests** — the `isolated_db` fixture in `conftest.py` handles DB isolation automatically; do not add per-test DB setup
- **Provider abstraction** — new LLM providers should implement `complete()` and `acomplete()` and be registered in `get_provider()`
- **No secrets in code** — all credentials via environment variables only
- **Keep the demo clean** — `run_demo.py` must always work with zero API key

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Run `pytest tests/ -v` and `python demo/run_demo.py`
4. Open a PR using the template — fill in every section

## Reporting Bugs

Use the Bug Report issue template. Include your Python version, OS, and the
full error output.

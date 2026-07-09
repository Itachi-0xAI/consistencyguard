# ConsistencyGuard

**Your LLM said something different yesterday. You didn't notice. Your users did.**

[![CI](https://github.com/Itachi-0xAI/consistencyguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Itachi-0xAI/consistencyguard/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.2-informational.svg)](CHANGELOG.md)

---

Real-time consistency monitor for LLM outputs. Wraps any LLM call with one function, detects when the same question gets a different answer across time or agents, and surfaces violations instantly — in your code and in the terminal. Zero infrastructure. No vector database. No API cost for detection.

---

## Complementary Features

- **Cross-provider check** — `cg crosscheck "<prompt>"` runs the same prompt on Groq + Gemini in parallel and reports pairwise semantic divergence, catching when one provider hallucinates while others stay grounded.
- **HTML export** — `cg export --format html` produces a standalone, browser-viewable report with embedded CSS. Share it without any viewer dependency.
- **Discord webhooks** — set `WEBHOOK_URL` to a Discord webhook URL (`discord.com/api/webhooks/...`) and violation alerts arrive as rich embeds automatically; generic webhooks are unchanged.
- **GitHub Actions CI gate** — use `cg health` and `cg reliability` as PR steps; exit 1 fails the build on drift (see `docs/SYSTEM_DESIGN.md`).
- **Markdown export** — `cg export --format markdown` produces a GitHub-pasteable Markdown table (severity emoji, truncated prompt, divergence score) ready to drop into a PR comment.
- **Baseline pinning** — `cg pin --label "v1.0-known-good"` snapshots agent embeddings + thresholds to a secret GitHub Gist (free PAT, `gist` scope); `cg restore <gist_id>` fetches and summarises the snapshot without mutating the local DB.
- **Structured outputs** — pass `expected_schema={"key": type}` to `guarded_call`/`aguarded_call`; responses are JSON-validated, key-order-normalised before embedding, and violations carry `INVALID_JSON` or `SCHEMA_MISMATCH` issues. Stdlib `json` only.
- **OpenTelemetry tracing** — install `opentelemetry-api` and configure a tracer; `guarded_call`, `aguarded_call`, and `find_similar_calls` emit spans with attributes like `cg.violations_count` and `cg.max_divergence`. Zero cost when OTel is absent.

---

## The Problem

Hallucinations are detectable — the model makes something up and it is obviously wrong. **Inconsistency is invisible.**

Your support agent told one customer "full refund within 30 days" and told another "no refunds after purchase." No error was thrown. No alert fired. The pipeline showed green. Both answers were plausible. Just different.

Most LLM observability tools track cost, latency, and errors. None of them answer: **"Is my agent saying the same thing today that it said last week?"** ConsistencyGuard is built to answer exactly that.

---

## Who Is This For

- **Marketing and content teams** using AI to generate content who need brand consistency across personas, campaigns, and channels — without manually reviewing every output
- ML/AI engineers shipping LLM agents to production who need consistency guarantees
- Platform teams running multi-agent systems across departments and need unified visibility
- Support/knowledge bot owners who need to catch contradicting answers to the same question
- RAG pipeline owners monitoring retrieval index drift

---

## Brand Positioning Consistency

Marketing teams use AI to generate content (blog posts, emails, LinkedIn posts, ad copy). The AI doesn't know your positioning playbook — who you're talking to, what you emphasize for a CFO vs. an engineer, what competitors you mention or avoid.

ConsistencyGuard solves this as a **brand governance layer**:

1. Your first approved output for each persona becomes the baseline
2. Every subsequent generation is checked against it semantically
3. Drift from the approved positioning fires a violation before it reaches the approval queue

```python
# First approved output sets the baseline automatically
response, violations = guarded_call(
    prompt="Write a LinkedIn post for a CFO about our risk management product",
    agent_id="marketing-cfo-persona",
)

# Next week, same persona — drift detected if messaging changed materially
response, violations = guarded_call(
    prompt="LinkedIn post for a CFO, risk management angle",
    agent_id="marketing-cfo-persona",
)

for v in violations:
    print(f"[{v.severity.value.upper()}] Brand drift detected: {v.explanation}")

# [WARNING] Semantic divergence 0.31: Prior post emphasized compliance risk
# reduction. New post emphasizes cost savings. Possible persona inconsistency.
```

Works for: marketing content, support bot responses, sales email sequences, product documentation, any AI output that must stay consistent with an approved baseline.

---

## The Demo That Sells Itself

```bash
python demo/run_demo.py
```

Watch what happens when the same question gets different answers:

```
Call 1 — Monday:
  Q: "What is our refund policy?"
  A: "Full refund within 30 days of purchase."
  → Stored as baseline. No violations.

Call 2 — Wednesday (same question):
  Q: "What is our refund policy?"
  A: "Refunds available within 30 days. Contact support."
  → Divergence: 0.08. Severity: INFO. Minor phrasing variation.

Call 3 — Friday (policy quietly changed in the model):
  Q: "What is our refund policy?"
  A: "No refunds after purchase under any circumstances."
  → Divergence: 0.43. Severity: CRITICAL.
  → [CRITICAL] Responses contradict each other. Prior: full refund. Now: no refund.
  → Webhook fired. Slack alert sent.
```

Zero API key required. This runs on local embeddings and injected demo data.

---

## Test Your Model's Reliability

```bash
cg reliability "What is the refund policy?" --runs 5
```

This runs the same prompt 5 times and shows you how consistent the model actually is:

```
╭──────────────── Hallucination Diff Report ────────────────╮
│ Reliability Score:  0.94 / 1.00   ✓ RELIABLE              │
│ Mean pairwise divergence:  0.06                           │
│ Outlier runs (≥0.25 from median): 0 / 5                   │
╰───────────────────────────────────────────────────────────╯

       R01    R02    R03    R04    R05
R01    ——    0.05   0.05   0.06   0.07
R02   0.05    ——    0.03   0.08   0.11
R03   0.05   0.03    ——    0.07   0.09
R04   0.06   0.08   0.07    ——    0.04
R05   0.07   0.11   0.09   0.04    ——
```

Score below 0.85? Your model is inconsistent. Find out before your users do.

---

## Quickstart

```bash
git clone https://github.com/Itachi-0xAI/consistencyguard.git
cd consistencyguard
pip install -e .
cp .env.example .env
# Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env

# No API key? Run the zero-cost demo first
python demo/run_demo.py

# Then send a real prompt
cg check "What is the maximum file upload size?"
cg report
```

---

## Drop-in Integration

Replace your LLM call with `guarded_call`. Same response. Plus a list of violations.

```python
from consistencyguard.proxy import guarded_call

response, violations = guarded_call(
    prompt="What is the refund policy?",
    agent_id="support-bot",
)

for v in violations:
    print(f"[{v.severity.value.upper()}] {v.explanation}")
```

That is the entire integration. `guarded_call` calls the LLM, embeds the prompt locally, scans past calls for semantic matches, computes response divergence, and returns any violations alongside the normal response text.

### Async

```python
from consistencyguard.proxy import aguarded_call

response, violations = await aguarded_call(
    prompt="What is the refund policy?",
    agent_id="support-bot",
)
```

### Provider selection

```python
# Explicit provider — no env var needed
response, violations = guarded_call(
    prompt="What is the upload limit?",
    agent_id="storage-agent",
    provider="openai",   # "anthropic" | "openai" | "gemini"
    model="gpt-4o-mini",
)
```

---

## How It Works

```
Your App  →  guarded_call()  →  LLM API (Anthropic / OpenAI / Gemini)
                 │
                 ├─ 1. embed(prompt)          ← all-MiniLM-L6-v2, local CPU, no API cost
                 ├─ 2. cosine similarity scan ← SQLite, O(n) over stored embeddings
                 ├─ 3. if similarity ≥ 0.92   → compare responses semantically
                 ├─ 4. if divergence ≥ 0.25   → CONSISTENCY VIOLATION
                 └─ 5. log + optional webhook + return violations
```

| Severity | Divergence | What it means |
|----------|------------|---------------|
| `INFO`   | ≥ 0.10     | Minor phrasing variation |
| `WARNING` | ≥ 0.25    | Material difference in content |
| `CRITICAL` | ≥ 0.40  | Responses contradict each other |

All detection runs locally. Embeddings are computed with `sentence-transformers` (`all-MiniLM-L6-v2`) on CPU. Calls and embeddings are stored in a single SQLite file. No data leaves your machine for the detection step.

---

## Hallucination Diff — `cg reliability`

Run the same prompt N times and measure how consistent the model actually is. Produces a reliability score, per-run divergence, a pairwise matrix, and outlier detection.

```bash
cg reliability "What is the refund policy?" --runs 5
```

```
╭─────────────────── Hallucination Diff Report ───────────────────╮
│ Prompt:  What is the refund policy?                             │
│ Model:   claude-haiku-4-5-20251001                              │
│ Runs:    5                                                      │
│                                                                 │
│ Reliability Score:  0.94 / 1.00   RELIABLE                     │
│ Verdict:            RELIABLE                                    │
│                                                                 │
│ Mean pairwise divergence:  0.0601                               │
│ Max  pairwise divergence:  0.1134                               │
│ Outlier runs (≥0.25 from median): 0 / 5                        │
╰─────────────────────────────────────────────────────────────────╯

Per-Run Results
┌─────┬──────────────────┬──────────┬─────────┬─────────────────────────────────────────────────────────────┐
│ Run │ Div. from Median │ Severity │ Outlier │ Response (truncated)                                        │
├─────┼──────────────────┼──────────┼─────────┼─────────────────────────────────────────────────────────────┤
│   1 │           0.0000 │ · INFO   │  median │ We offer a full refund within 30 days of purchase. After …  │
│   2 │           0.0512 │ · INFO   │      no │ Our refund policy allows returns within 30 days. Items mu…  │
│   3 │           0.0489 │ · INFO   │      no │ Refunds are available for 30 days from the date of purchas… │
│   4 │           0.0601 │ · INFO   │      no │ You can request a refund within 30 days. Please contact s…  │
│   5 │           0.0723 │ · INFO   │      no │ We have a 30-day return policy. Refunds are processed with… │
└─────┴──────────────────┴──────────┴─────────┴─────────────────────────────────────────────────────────────┘

────────────────────── Pairwise Divergence Matrix ──────────────────────
      R01   R02   R03   R04   R05
R01   ——  0.05  0.05  0.06  0.07
R02  0.05   ——  0.03  0.08  0.11
R03  0.05  0.03   ——  0.07  0.09
R04  0.06  0.08  0.07   ——  0.06
R05  0.07  0.11  0.09  0.06   ——
```

Use `--runs 10` for a more statistically robust score. Add `--outlier-threshold 0.20` to tighten the outlier boundary.

```bash
cg reliability "What is our cancellation fee?" --runs 10 --provider openai --model gpt-4o-mini
```

---

## CLI Reference

```bash
# System status — DB path and size, env config, embedding model
cg health

# Recent violations
cg report
cg report --severity critical
cg report --agent support-bot
cg report --since 24          # last 24 hours

# Hourly violation bar chart
cg trend
cg trend --hours 48

# Per-agent violation breakdown
cg agents
cg agents --hours 72

# Send a live prompt through the guard (requires API key)
cg check "What is the maximum file upload size?"
cg check "Refund policy?" --agent billing-bot --provider openai

# Reliability test — run N times, score 0.0–1.0
cg reliability "What is the refund policy?" --runs 10
cg reliability "Cancellation fee?" --runs 5 --provider gemini --model gemini-1.5-flash

# Export violations
cg export --format json -o violations.json
cg export --format csv --agent support-bot --since 48
```

---

## Features

| Feature | Detail |
|---------|--------|
| `guarded_call()` / `aguarded_call()` | Drop-in sync and async wrappers — same response, plus violations |
| Local embeddings | `all-MiniLM-L6-v2` on CPU — no API cost, no data leaves the machine |
| Three providers | Anthropic, OpenAI, Google Gemini — swap via `.env`, no code change |
| OpenAI-compatible endpoints | Works with Groq, Together AI, and any OpenAI-compatible API |
| Severity levels | `INFO` / `WARNING` / `CRITICAL` — tunable thresholds |
| `cg reliability` | Run a prompt N times, get reliability score + pairwise matrix + outlier detection |
| Time-windowed baselines | `COMPARISON_WINDOW_DAYS` prevents stale history from flagging correct updated answers |
| Webhook alerts | POST violation JSON to any HTTP endpoint (Slack, PagerDuty, custom) on every detection |
| Zero infrastructure | One SQLite file, zero ops, `pip install` and done |
| 45 tests | Real embeddings, fully isolated DBs per test, no API key required |
| Zero-API-key demo | `python demo/run_demo.py` — 7 calls, 3 injected violations, no credentials needed |

---

## Installation

**Requirements:** Python 3.11+

```bash
pip install -e .
```

On first run, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face and caches it locally. Subsequent runs load from cache instantly.

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `gemini` |
| `ANTHROPIC_API_KEY` | — | Required when `PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `PROVIDER=openai` or `PROVIDER=gemini` |
| `GEMINI_API_KEY` | — | Required when `PROVIDER=gemini` |
| `OPENAI_BASE_URL` | — | Override base URL for any OpenAI-compatible endpoint (Groq, Together, etc.) |
| `MODEL` | `claude-haiku-4-5-20251001` | Model name — must match the provider |
| `SIMILARITY_THRESHOLD` | `0.92` | Prompt cosine similarity required to trigger a comparison |
| `DIVERGENCE_THRESHOLD` | `0.25` | Response divergence required to flag a violation |
| `COMPARISON_WINDOW_DAYS` | unlimited | Only compare against calls from the last N days |
| `DB_PATH` | `consistencyguard.db` | SQLite database file path |
| `WEBHOOK_URL` | — | POST violation JSON here on every detection |
| `RELIABILITY_RUN_DELAY` | `0` | Seconds between runs in `cg reliability` (useful for free-tier rate limits) |

### OpenAI example

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
```

### Gemini example

```env
PROVIDER=gemini
GEMINI_API_KEY=...
MODEL=gemini-1.5-flash
```

### Groq (free tier, fast)

```env
PROVIDER=openai
OPENAI_API_KEY=<your-groq-key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL=llama-3.1-8b-instant
```

Get a free Groq key at [console.groq.com](https://console.groq.com). Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).

---

## Webhook Alerts

Set `WEBHOOK_URL` in `.env` to receive a JSON POST on every violation. Works with any HTTP endpoint that accepts a JSON body — Slack incoming webhooks, PagerDuty, custom receivers.

```json
{
  "event": "consistency_violation",
  "severity": "critical",
  "agent_id": "support-bot",
  "new_prompt": "What is the refund policy?",
  "new_response": "No refunds under any circumstances.",
  "ref_response": "Full refund within 30 days of purchase.",
  "prompt_similarity": 1.0,
  "response_divergence": 0.43,
  "explanation": "[CRITICAL] Semantic divergence: 0.43...",
  "timestamp": "2026-05-26T15:30:00"
}
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

| Suite | Tests | What it covers |
|-------|-------|----------------|
| `test_detector.py` | 8 | Embedder cosine similarity, divergence scoring, severity classification — real embeddings |
| `test_store.py` | 6 | SQLite schema, call persistence, violation storage, stats aggregation |
| `test_providers.py` | 10 | Anthropic and OpenAI sync/async — all mocked, no API key required |
| `test_user_flows.py` | 7 | End-to-end developer flows: first call, consistent follow-up, contradicting response, cross-agent detection |
| `test_hallucination_diff.py` | 14 | Pairwise matrix, median detection, verdict classification, outlier flagging, report structure |
| **Total** | **45** | All pass. No API key required for any test. |

Tests use real `sentence-transformers` embeddings (no mocking of the model) and fully isolated SQLite databases per test via the `conftest.py` fixture.

---

## Architecture

```
consistencyguard/
├── consistencyguard/
│   ├── models.py             # Pydantic data models (LLMCall, ConsistencyViolation, etc.)
│   ├── embedder.py           # all-MiniLM-L6-v2 embedding + cosine similarity
│   ├── store.py              # SQLite — calls, embeddings, violations, trend, agent stats
│   ├── detector.py           # Similarity scan + divergence check + severity classification
│   ├── proxy.py              # guarded_call / aguarded_call — the main entry point
│   ├── providers.py          # AnthropicProvider, OpenAIProvider, GeminiProvider
│   ├── hallucination_diff.py # Reliability test, pairwise matrix, verdict, outlier detection
│   ├── webhooks.py           # Webhook dispatch — sync (httpx) and async
│   ├── reporter.py           # Rich terminal output — tables, trend chart, hallucination diff
│   └── cli.py                # Click CLI (cg command)
├── tests/                    # 45 tests
├── demo/
│   └── run_demo.py           # Zero-API-key demo — 7 calls, 3 injected violations
├── docs/
│   ├── SYSTEM_DESIGN.md      # Architecture, data flow, scaling analysis
│   └── FAILURE_ANALYSIS.md   # 8 real failure scenarios with root cause analysis
└── pyproject.toml
```

**Key design decisions:**

- **Local embeddings** — `all-MiniLM-L6-v2` runs on CPU. No embedding API calls, no data leaves the machine, no cost per prompt.
- **SQLite over a vector database** — O(n) cosine scan over stored embeddings is fast enough to ~50k calls. Zero infrastructure. One file, zero ops.
- **Provider abstraction** — `AnthropicProvider`, `OpenAIProvider`, and `GeminiProvider` all implement the same `complete` / `acomplete` interface. Change providers by editing `.env`, not code.
- **Time-windowed comparison** — `COMPARISON_WINDOW_DAYS` prevents stale historical baselines from flagging correct updated answers when your policy genuinely changes.
- **Prompt normalization** — whitespace collapsed and lowercased before embedding to prevent tokenization artifacts from causing spurious misses.
- **Cross-agent scope** — consistency is checked globally, not per-agent. If `agent-a` and `agent-b` give contradicting answers to the same question, that is a violation. See `test_user_flows.py::test_cross_agent_divergence_is_detected`.

---

## Integration: CoAgent Debate Engine

When [CoAgent](https://github.com/Itachi-0xAI/coagent)'s DEBATE mode runs — two AI agents arguing opposing positions — each agent's claims are wrapped with `guarded_call()`. If an advocate cites something that contradicts a prior session's claim, it's flagged in the DecisionRecord's integrity report before the human moderator sees it.

```python
# Inside CoAgent's DebateOrchestrator, each claim is guarded:
from consistencyguard.proxy import guarded_call

response, violations = guarded_call(
    prompt=claim_text,
    agent_id=f"debate-{agent.name}-{session.id}",
)
if violations:
    claim.flags.append("consistency_violation")
    claim.confidence = "low"
```

Result: A debate where one advocate said *"PostgreSQL handles 10k writes/sec"* in session 1 but says *"PostgreSQL handles 50k writes/sec"* in session 2 is automatically flagged — before it influences the human's final decision.

See [STACK.md](STACK.md) for the full three-tool integration pattern.

---

## Production Setup

Deploy with free APIs in five steps:

1. Get a free Groq key at [console.groq.com](https://console.groq.com) and set `PROVIDER=openai`, `OPENAI_API_KEY=<groq-key>`, `OPENAI_BASE_URL=https://api.groq.com/openai/v1`, `MODEL=llama-3.1-8b-instant` in `.env`.
2. Set `RELIABILITY_RUN_DELAY=3` to respect Groq's free-tier requests-per-minute quota.
3. Create a free Slack incoming webhook at `api.slack.com/messaging/webhooks` and set `WEBHOOK_URL` — violations POST there automatically with 3-retry backoff.
4. Deploy `app.py` to [Streamlit Community Cloud](https://share.streamlit.io) (free) — connect your GitHub repo and paste the env vars into the Secrets panel.
5. Add `HEALTHCHECK CMD cg health` to your Dockerfile — exits 0 when healthy, 1 when the DB, embedding model, or API key is unreachable.

---

## Known Limitations

| Limitation | Impact | Status |
|-----------|--------|--------|
| No PII scrubbing | Prompts stored as plaintext in SQLite | Pre-processing redaction hook planned |
| SQLite single-writer | Throughput bottleneck at high concurrency | `pgvector` / `aiosqlite` migration planned |
| No prompt template support | Variable values shift prompt embeddings | Template registry planned |
| No streaming support | Buffers full response before checking | Async tail-check planned |
| Embedding model drift | Re-embedding required after model upgrade | Migration tooling planned |

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, test instructions, free-tier LLM options, and PR guidelines.

```bash
pytest tests/ -v          # all 45 tests must pass
python demo/run_demo.py   # demo must exit cleanly with 3 violations detected
```

---

## Ecosystem

ConsistencyGuard is one layer of a three-part open-source AI trust stack. Each tool catches a different failure mode:

| Failure mode | Tool | When it fires |
|---|---|---|
| AI contradicts itself over time | **ConsistencyGuard** *(this repo)* | After response |
| AI answers from stale index | **[ARIA](https://github.com/Itachi-0xAI/aria)** | Before response |
| AI decision has no reasoning trail | **[CoAgent](https://github.com/Itachi-0xAI/coagent)** | During decision |

See [STACK.md](STACK.md) for the integration patterns and a worked end-to-end example.

---

## License

MIT — see [LICENSE](LICENSE).

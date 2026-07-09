# ConsistencyGuard Roadmap

## Released: v1.0.0 – v1.0.2

- **Core detection engine**: Multi-call inconsistency detection with configurable thresholds
- **SQLite backend**: Local persistence for call history, responses, and inconsistency logs
- **Webhook notifications**: Real-time alerts when inconsistencies detected
- **CLI tooling**: `cg reliability`, `cg report`, `cg export` commands for analysis and debugging
- **Hallucination diff**: Side-by-side comparison of conflicting responses
- **Multi-provider support**: Anthropic, OpenAI, Gemini, Groq (via unified interface)

## Phase 2: Isolation & Scale (next)

- **Per-agent isolation mode**: Toggle per-LLM-provider to handle provider-specific inconsistencies
- **pgvector support**: Migrate from SQLite for embeddings-based similarity detection
- **Streaming response support**: Handle streaming calls without buffering entire response
- **Prompt template registry**: Store and version prompts used in calls for better context
- **PII scrubbing hook**: Optional pre-logging filter to strip sensitive data
- **Positioning intelligence layer**: A per-persona positioning playbook (CFO, engineer, compliance, etc.) wired to `guarded_call` as the baseline source — drift detection becomes brand-governance enforcement. Pairs with [CoAgent](https://github.com/Itachi-0xAI/coagent) DEBATE for arbitrating playbook updates. Open an issue if this is the use case you'd build on first.

## Phase 3: Enterprise (future)

- **Multi-tenant database**: Isolated namespaces for teams sharing one deployment
- **REST API server mode**: Standalone service (FastAPI) instead of drop-in library
- **Hosted dashboard**: Web UI for reliability metrics, trend analysis, and anomaly detection
- **Slack/PagerDuty integration**: Automated incident creation for critical inconsistencies
- **Custom metric plugins**: Extend detection logic for domain-specific validation rules

## How to Influence the Roadmap

Open an issue on [GitHub Issues](https://github.com/Itachi-0xAI/consistencyguard/issues). Features with concrete use cases get prioritized. Phase 2 PRs are welcome.

---

**Latest release**: v1.0.2 | **Next target**: v1.1.0 (Phase 2 start)

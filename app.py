"""
ConsistencyGuard — Interactive Streamlit Demo
No API key required. Runs entirely on local embeddings + demo data.
"""
import os
import time
from datetime import datetime, timedelta

import streamlit as st

os.environ.setdefault("DB_PATH", "/tmp/cg_demo.db")

st.set_page_config(
    page_title="ConsistencyGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ ConsistencyGuard")
st.sidebar.caption("Real-time LLM consistency monitor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 What Is This", "🔬 Reliability Tester", "⚠️ Violation Feed", "📊 Stats"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Itachi--0xAI-black?logo=github)]"
    "(https://github.com/Itachi-0xAI/consistencyguard)"
)


# ── shared helpers ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _init():
    from consistencyguard.store import init_db
    from consistencyguard.embedder import get_model
    init_db()
    get_model()

_init()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WHAT IS THIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 What Is This":
    st.title("🛡️ ConsistencyGuard")
    st.subheader("Your LLM said something different yesterday. You didn't notice. Your users did.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### The Problem")
        st.markdown(
            "Hallucinations are detectable — the model says something obviously wrong. "
            "**Inconsistency is invisible.** The model gives a plausible answer both times. "
            "Just a different one.\n\n"
            "Your support agent told one customer *'full refund within 30 days'* and told "
            "another *'no refunds after purchase'*. No error was thrown. No alert fired."
        )

    with col2:
        st.markdown("### The Fix")
        st.code(
            'from consistencyguard.proxy import guarded_call\n\n'
            'response, violations = guarded_call(\n'
            '    prompt="What is the refund policy?",\n'
            '    agent_id="support-bot",\n'
            ')\n\n'
            'for v in violations:\n'
            '    print(f"[{v.severity.value}] {v.explanation}")',
            language="python",
        )

    st.markdown("---")
    st.markdown("### How It Works")
    cols = st.columns(4)
    steps = [
        ("1️⃣ Embed", "Every prompt is embedded locally using `all-MiniLM-L6-v2`. No API cost."),
        ("2️⃣ Store", "Call + embedding saved to SQLite. Zero infrastructure."),
        ("3️⃣ Compare", "New calls scanned against history. Cosine similarity detects same-question matches."),
        ("4️⃣ Alert", "Response divergence scored. INFO / WARNING / CRITICAL violation raised."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("---")
    st.markdown("### Try It")
    st.markdown(
        "👈 Use **Reliability Tester** to run any prompt multiple times and see the variance score.\n\n"
        "Or use **Violation Feed** to load the built-in demo and see 3 real inconsistencies detected."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RELIABILITY TESTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Reliability Tester":
    st.title("🔬 Reliability Tester")
    st.caption("Run any prompt N times and measure how consistent the model is.")

    st.info(
        "**Demo mode** — uses pre-written response pairs to show the scoring engine "
        "without needing an API key. Add your Gemini/Groq key in `.env` to run live.",
        icon="ℹ️",
    )

    with st.form("reliability_form"):
        prompt = st.text_area(
            "Prompt",
            value="What is the maximum file upload size?",
            height=80,
        )
        col1, col2 = st.columns(2)
        with col1:
            runs = st.slider("Runs", min_value=2, max_value=10, value=5)
        with col2:
            demo_mode = st.checkbox("Demo mode (no API key)", value=True)
        submitted = st.form_submit_button("▶ Run Reliability Test", type="primary")

    if submitted:
        from consistencyguard.embedder import embed, cosine_similarity
        from consistencyguard.hallucination_diff import _pairwise_matrix, _verdict

        # Demo responses — alternating consistent + one outlier
        DEMO_RESPONSES = [
            "The maximum file upload size is 25MB per file.",
            "Files up to 25MB can be uploaded per file.",
            "The upload limit is 25 megabytes per file.",
            "File size limit is 25MB. Larger files must be split.",
            "You can upload files up to 25MB each.",
            "Contact support to check file limits.",           # outlier
            "The maximum upload size is 25MB per file.",
            "25MB is the per-file limit for uploads.",
            "Upload limit: 25MB per file.",
            "The limit is 25MB for each individual file.",
        ]

        with st.spinner(f"Running {runs} passes…"):
            if demo_mode:
                responses = DEMO_RESPONSES[:runs]
                time.sleep(0.8)
            else:
                st.error("Live mode: add GEMINI_API_KEY to .env and restart.")
                st.stop()

        embeddings = [embed(r) for r in responses]
        matrix = _pairwise_matrix(embeddings)
        n = len(responses)
        all_divs = [matrix[i][j] for i in range(n) for j in range(i+1, n)]
        mean_div = sum(all_divs) / len(all_divs) if all_divs else 0.0
        reliability = round(1.0 - mean_div, 4)
        verdict = _verdict(reliability)

        # ── score banner ──────────────────────────────────────────────────────
        color = {"RELIABLE": "🟢", "UNSTABLE": "🟡", "CRITICAL": "🔴"}[verdict]
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Reliability Score", f"{reliability:.2f} / 1.00")
        m2.metric("Verdict", f"{color} {verdict}")
        m3.metric("Mean Divergence", f"{mean_div:.4f}")

        # ── per-run table ─────────────────────────────────────────────────────
        st.markdown("#### Per-Run Results")
        median_emb = embeddings[0]  # simplification for display
        rows = []
        for i, (resp, emb) in enumerate(zip(responses, embeddings)):
            div = round(max(0.0, 1.0 - cosine_similarity(emb, embeddings[0])), 4)
            outlier = "🔴 YES" if div >= 0.25 else "✅ no"
            rows.append({"Run": i+1, "Divergence from R1": div, "Outlier": outlier, "Response": resp[:100]})

        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── pairwise matrix ───────────────────────────────────────────────────
        if n <= 8:
            st.markdown("#### Pairwise Divergence Matrix")
            import pandas as pd
            labels = [f"R{i+1}" for i in range(n)]
            df = pd.DataFrame(matrix, index=labels, columns=labels)
            st.dataframe(df.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.5), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIOLATION FEED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Violation Feed":
    st.title("⚠️ Violation Feed")
    st.caption("All consistency violations detected across agents.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("▶ Load Demo Data", type="primary"):
            with st.spinner("Running demo…"):
                # Import and run demo inline
                from consistencyguard.store import init_db, save_call, save_violation
                from consistencyguard.embedder import embed
                from consistencyguard.detector import check_consistency
                from consistencyguard.models import LLMCall

                init_db()
                now = datetime.utcnow()

                DEMO_CALLS = [
                    ("What is the maximum file upload size?", "The maximum file upload size is 25MB per file.", "support-agent", 60),
                    ("What payment methods do you accept?", "We accept Visa, Mastercard, and PayPal.", "sales-agent", 50),
                    ("What is the maximum file upload size?", "File uploads are disabled. Please email files to support@company.com.", "support-agent", 10),
                    ("What payment methods do you accept?", "We only accept bank transfers. Credit cards are not supported.", "sales-agent", 5),
                    ("What are your business hours?", "We are open Monday–Friday, 9am–6pm EST.", "support-agent", 1),
                ]

                for prompt, response, agent_id, offset in DEMO_CALLS:
                    emb = embed(prompt)
                    call = LLMCall(
                        prompt=prompt, response=response, model="demo",
                        agent_id=agent_id,
                        timestamp=now - timedelta(minutes=offset),
                        prompt_embedding=emb,
                    )
                    call_id = save_call(call)
                    call.id = call_id
                    for v in check_consistency(call):
                        v.call_id_new = call_id
                        save_violation(v)

                st.success("Demo data loaded — 2 violations injected.")

    from consistencyguard.store import get_all_violations, get_stats
    violations = get_all_violations()

    if not violations:
        st.info("No violations yet. Click **Load Demo Data** or run `guarded_call()` in your app.")
    else:
        SEV_COLOR = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        for v in violations[:20]:
            sev = v.get("severity", "info")
            icon = SEV_COLOR.get(sev, "⚪")
            with st.expander(f"{icon} [{sev.upper()}] {v.get('new_prompt', '')[:70]}…"):
                c1, c2 = st.columns(2)
                c1.metric("Divergence", f"{v.get('response_divergence', 0):.2f}")
                c2.metric("Prompt Similarity", f"{v.get('prompt_similarity', 0):.2f}")
                st.markdown(f"**Agent:** `{v.get('agent_id', '—')}`")
                st.markdown(f"**Previous response:** {v.get('ref_response', '')}")
                st.markdown(f"**New response:** {v.get('new_response', '')}")
                st.caption(v.get("timestamp", "")[:19])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — STATS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stats":
    st.title("📊 Stats")

    from consistencyguard.store import get_stats, get_agent_stats
    stats = get_stats()
    agent_stats = get_agent_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Calls", stats.get("total_calls", 0))
    c2.metric("Total Violations", stats.get("total_violations", 0))
    c3.metric("Critical", stats.get("critical", 0))
    c4.metric("Warning", stats.get("warning", 0))

    rate = (
        stats["total_violations"] / stats["total_calls"] * 100
        if stats.get("total_calls", 0) > 0 else 0
    )
    st.metric("Violation Rate", f"{rate:.1f}%")

    if agent_stats:
        st.markdown("#### Per-Agent Breakdown")
        import pandas as pd
        df = pd.DataFrame(agent_stats)[["agent_id", "total_calls", "total_violations", "critical", "warning", "violation_rate"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Load demo data from the Violation Feed page.")

    st.markdown("---")
    st.markdown("#### Install")
    st.code("pip install consistencyguard", language="bash")
    st.markdown("#### Quick Integration")
    st.code(
        "from consistencyguard.proxy import guarded_call\n\n"
        "response, violations = guarded_call(\n"
        '    prompt="Your prompt here",\n'
        '    agent_id="my-agent",\n'
        ")",
        language="python",
    )

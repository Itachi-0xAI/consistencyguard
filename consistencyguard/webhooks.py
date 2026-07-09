import os
import logging
from consistencyguard.models import ConsistencyViolation

logger = logging.getLogger(__name__)


def _serialize(v: ConsistencyViolation) -> dict:
    d = v.model_dump()
    d["severity"] = d["severity"].value if hasattr(d["severity"], "value") else d["severity"]
    d["timestamp"] = d["timestamp"].isoformat() if hasattr(d["timestamp"], "isoformat") else d["timestamp"]
    d["event"] = "consistency_violation"
    return d


def _failure_log_path() -> str:
    db_dir = os.path.dirname(os.path.abspath(os.getenv("DB_PATH", "consistencyguard.db")))
    return os.path.join(db_dir, "webhook_failures.log")


def _log_failure(target: str, exc: Exception) -> None:
    """Append a failure line to webhook_failures.log — never raises."""
    try:
        import datetime
        ts = datetime.datetime.utcnow().isoformat()
        with open(_failure_log_path(), "a") as f:
            f.write(f"{ts} url={target} error={exc}\n")
    except Exception:
        pass


def _is_discord_url(url: str) -> bool:
    return "discord.com/api/webhooks/" in url or "discordapp.com/api/webhooks/" in url


def _discord_payload(v: ConsistencyViolation) -> dict:
    """Format a ConsistencyViolation as a Discord webhook payload with an embed."""
    SEV_COLOR = {"critical": 0xEF4444, "warning": 0xF59E0B, "info": 0x38BDF8}
    sev = v.severity.value if hasattr(v.severity, "value") else str(v.severity)
    color = SEV_COLOR.get(sev.lower(), 0x94A3B8)
    ts = v.timestamp.isoformat() if hasattr(v.timestamp, "isoformat") else str(v.timestamp)
    return {
        "content": f"**ConsistencyGuard** detected a `{sev.upper()}` violation",
        "embeds": [
            {
                "title": f"Consistency Violation — {sev.upper()}",
                "color": color,
                "fields": [
                    {"name": "Agent", "value": v.agent_id, "inline": True},
                    {"name": "Divergence", "value": f"{v.response_divergence:.4f}", "inline": True},
                    {"name": "Similarity", "value": f"{v.prompt_similarity:.4f}", "inline": True},
                    {"name": "Prompt", "value": v.new_prompt[:256], "inline": False},
                    {"name": "Explanation", "value": v.explanation[:512], "inline": False},
                ],
                "footer": {"text": "ConsistencyGuard"},
                "timestamp": ts,
            }
        ],
    }


def fire_webhook(v: ConsistencyViolation, url: str = None) -> None:
    """POST violation JSON to url. Detects Discord URLs and adjusts payload format.
    Best-effort — 3 retries with exponential backoff."""
    target = url or os.getenv("WEBHOOK_URL")
    if not target:
        return
    try:
        import httpx
        from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

        payload = _discord_payload(v) if _is_discord_url(target) else _serialize(v)

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
        def _post():
            with httpx.Client(timeout=5.0) as client:
                client.post(target, json=payload)

        _post()
    except Exception as exc:
        logger.warning("Webhook delivery failed: %s", exc)
        _log_failure(target, exc)


async def afire_webhook(v: ConsistencyViolation, url: str = None) -> None:
    """Async version of fire_webhook. Detects Discord URLs and adjusts payload format.
    3-retry exponential backoff."""
    target = url or os.getenv("WEBHOOK_URL")
    if not target:
        return
    try:
        import httpx
        from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

        payload = _discord_payload(v) if _is_discord_url(target) else _serialize(v)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(target, json=payload)
    except Exception as exc:
        logger.warning("Webhook delivery failed: %s", exc)
        _log_failure(target, exc)

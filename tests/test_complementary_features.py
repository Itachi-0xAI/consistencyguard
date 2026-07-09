"""
Tests for complementary features:
  - cross_provider_check (mocked — no real API calls)
  - export_violations_html
  - Discord webhook payload formatting
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# cross_provider_check
# ---------------------------------------------------------------------------

def _fake_call(provider_name, prompt, max_tokens=256):
    responses = {
        "groq": "Paris is the capital of France.",
        "gemini": "The capital of France is Paris.",
    }
    return provider_name, responses.get(provider_name, "Some response.")


def _fake_call_drift(provider_name, prompt, max_tokens=256):
    responses = {
        "groq": "Paris is the capital of France.",
        "gemini": "I enjoy hiking in the mountains on weekends with my dog.",
    }
    return provider_name, responses.get(provider_name, "Other.")


class TestCrossProviderCheck:
    def test_returns_report_structure(self):
        from consistencyguard.cross_check import cross_provider_check, CrossProviderReport
        with patch("consistencyguard.cross_check._call_provider", side_effect=_fake_call):
            report = cross_provider_check("What is the capital of France?", providers=["groq", "gemini"])
        assert isinstance(report, CrossProviderReport)
        assert report.prompt == "What is the capital of France?"
        assert set(report.provider_responses.keys()) == {"groq", "gemini"}
        assert "groq vs gemini" in report.pairwise_divergence
        assert 0.0 <= report.agreement_score <= 1.0
        assert report.verdict in ("AGREEMENT", "DRIFT")

    def test_agreement_on_similar_responses(self):
        from consistencyguard.cross_check import cross_provider_check
        with patch("consistencyguard.cross_check._call_provider", side_effect=_fake_call):
            report = cross_provider_check("Capital of France?", providers=["groq", "gemini"])
        assert report.verdict == "AGREEMENT"
        assert report.agreement_score > 0.5

    def test_drift_on_divergent_responses(self):
        from consistencyguard.cross_check import cross_provider_check
        with patch("consistencyguard.cross_check._call_provider", side_effect=_fake_call_drift):
            report = cross_provider_check("Capital of France?", providers=["groq", "gemini"])
        assert report.verdict == "DRIFT"

    def test_requires_at_least_two_providers(self):
        from consistencyguard.cross_check import cross_provider_check
        with pytest.raises(ValueError, match="at least 2"):
            cross_provider_check("hello", providers=["groq"])

    def test_provider_error_raises_runtime(self):
        from consistencyguard.cross_check import cross_provider_check
        def _bad_call(provider_name, prompt, max_tokens=256):
            raise RuntimeError("network failure")
        with patch("consistencyguard.cross_check._call_provider", side_effect=_bad_call):
            with pytest.raises(RuntimeError, match="failed"):
                cross_provider_check("hello", providers=["groq", "gemini"])

    def test_pairwise_divergence_bounded(self):
        from consistencyguard.cross_check import cross_provider_check
        with patch("consistencyguard.cross_check._call_provider", side_effect=_fake_call):
            report = cross_provider_check("test", providers=["groq", "gemini"])
        for val in report.pairwise_divergence.values():
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# export_violations_html
# ---------------------------------------------------------------------------

def _make_violation_dict(sev="critical"):
    return {
        "severity": sev,
        "agent_id": "test-agent",
        "new_prompt": "What is the refund policy?",
        "response_divergence": 0.42,
        "prompt_similarity": 0.95,
        "timestamp": "2026-06-12T10:00:00",
        "explanation": "Responses contradict each other on refund terms.",
    }


class TestExportViolationsHtml:
    def test_returns_string(self):
        from consistencyguard.reporter import export_violations_html
        html = export_violations_html(violations=[_make_violation_dict()])
        assert isinstance(html, str)

    def test_contains_html_structure(self):
        from consistencyguard.reporter import export_violations_html
        html = export_violations_html(violations=[_make_violation_dict()])
        assert "<!DOCTYPE html>" in html
        assert "<table" in html
        assert "ConsistencyGuard" in html

    def test_severity_class_critical(self):
        from consistencyguard.reporter import export_violations_html
        html = export_violations_html(violations=[_make_violation_dict("critical")])
        assert "sev-critical" in html

    def test_severity_class_warning(self):
        from consistencyguard.reporter import export_violations_html
        html = export_violations_html(violations=[_make_violation_dict("warning")])
        assert "sev-warning" in html

    def test_empty_violations_shows_fallback(self):
        from consistencyguard.reporter import export_violations_html
        html = export_violations_html(violations=[])
        assert "No violations found" in html

    def test_prompt_truncated_at_120(self):
        from consistencyguard.reporter import export_violations_html
        long_prompt = "x" * 200
        v = _make_violation_dict()
        v["new_prompt"] = long_prompt
        html = export_violations_html(violations=[v])
        assert "x" * 120 in html
        assert "…" in html

    def test_html_escaping(self):
        from consistencyguard.reporter import export_violations_html
        v = _make_violation_dict()
        v["new_prompt"] = "<script>alert('xss')</script>"
        html = export_violations_html(violations=[v])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_multiple_violations(self):
        from consistencyguard.reporter import export_violations_html
        violations = [_make_violation_dict("critical"), _make_violation_dict("warning")]
        html = export_violations_html(violations=violations)
        assert html.count("<tr>") >= 2

    def test_export_violations_html_format(self):
        """Integration: export_violations(format='html') calls html path."""
        from consistencyguard.reporter import export_violations
        with patch("consistencyguard.reporter.get_violations_filtered", return_value=[_make_violation_dict()]):
            result = export_violations(format="html")
        assert "<!DOCTYPE html>" in result


# ---------------------------------------------------------------------------
# Discord webhook payload
# ---------------------------------------------------------------------------

class TestDiscordWebhook:
    def _make_violation(self):
        from consistencyguard.models import ConsistencyViolation, ViolationSeverity
        return ConsistencyViolation(
            call_id_new=1,
            call_id_ref=2,
            prompt_similarity=0.95,
            response_divergence=0.42,
            severity=ViolationSeverity.CRITICAL,
            new_prompt="What is the refund policy?",
            new_response="No refunds.",
            ref_response="Full refund within 30 days.",
            explanation="Contradictory refund terms.",
            agent_id="test-agent",
            timestamp=datetime(2026, 6, 12, 10, 0, 0, tzinfo=timezone.utc),
        )

    def test_is_discord_url_detection(self):
        from consistencyguard.webhooks import _is_discord_url
        assert _is_discord_url("https://discord.com/api/webhooks/123/abc")
        assert _is_discord_url("https://discordapp.com/api/webhooks/456/xyz")
        assert not _is_discord_url("https://hooks.slack.com/services/xxx")
        assert not _is_discord_url("https://example.com/webhook")

    def test_discord_payload_structure(self):
        from consistencyguard.webhooks import _discord_payload
        v = self._make_violation()
        payload = _discord_payload(v)
        assert "content" in payload
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"]
        assert isinstance(embed["color"], int)
        field_names = [f["name"] for f in embed["fields"]]
        assert "Agent" in field_names
        assert "Divergence" in field_names
        assert "Prompt" in field_names

    def test_discord_payload_critical_color(self):
        from consistencyguard.webhooks import _discord_payload
        from consistencyguard.models import ViolationSeverity
        v = self._make_violation()
        payload = _discord_payload(v)
        # critical = 0xEF4444 = 15681604
        assert payload["embeds"][0]["color"] == 0xEF4444

    def test_fire_webhook_routes_discord(self):
        from consistencyguard.webhooks import fire_webhook
        v = self._make_violation()
        discord_url = "https://discord.com/api/webhooks/123/token"
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            fire_webhook(v, url=discord_url)
            call_kwargs = mock_client.post.call_args
            posted_payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
            assert "embeds" in posted_payload

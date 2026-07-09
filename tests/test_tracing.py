"""
Tests for ConsistencyGuard tracing module.
Verifies that span() works as a no-op context manager when no tracer is configured.
"""

from consistencyguard.tracing import span


def test_span_noop_when_no_tracer():
    """span() yields None when no OTel tracer is configured."""
    with span("test.noop.span", some_attr="value") as s:
        assert s is None


def test_span_noop_body_executes():
    """Code inside a no-op span still runs normally."""
    ran = []
    with span("test.noop.body") as s:
        assert s is None
        ran.append(True)
    assert ran == [True]

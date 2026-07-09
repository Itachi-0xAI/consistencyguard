"""
Tests for ConsistencyGuard tracing module.
Verifies that span() works as a no-op context manager with or without OTel installed.
"""

from consistencyguard.tracing import span, _OTEL_AVAILABLE


def _is_noop_span(s) -> bool:
    """True when the span is either None (OTel absent) or a NonRecordingSpan (OTel present, no exporter)."""
    if s is None:
        return True
    # When OTel is installed but no real TracerProvider is configured, the global
    # tracer returns a NonRecordingSpan with all-zero trace/span IDs.
    type_name = type(s).__name__
    return type_name == "NonRecordingSpan"


def test_span_noop_when_no_tracer():
    """span() yields a no-op span (None or NonRecordingSpan) when no exporter is configured."""
    with span("test.noop.span", some_attr="value") as s:
        assert _is_noop_span(s), f"Expected no-op span, got {s!r}"


def test_span_noop_body_executes():
    """Code inside a no-op span still runs normally."""
    ran = []
    with span("test.noop.body") as s:
        assert _is_noop_span(s)
        ran.append(True)
    assert ran == [True]


def test_otel_available_flag_is_bool():
    """_OTEL_AVAILABLE must be a plain bool regardless of install state."""
    assert isinstance(_OTEL_AVAILABLE, bool)

"""
Tests for hallucination_diff module.
All LLM calls are mocked — no API key required.
"""

import pytest
from unittest.mock import MagicMock
from consistencyguard.hallucination_diff import (
    run_reliability_test,
    _pairwise_matrix,
    _find_median_idx,
    _verdict,
    _build_report,
)
from consistencyguard.embedder import embed
from consistencyguard.models import ViolationSeverity


def _mock_provider(responses: list[str]) -> MagicMock:
    """Provider that returns responses in sequence."""
    p = MagicMock()
    p.complete.side_effect = responses
    return p


# ── pairwise matrix ───────────────────────────────────────────────────────────

def test_pairwise_matrix_diagonal_is_zero():
    embeddings = [embed("Same text.")] * 3
    matrix = _pairwise_matrix(embeddings)
    for i in range(len(embeddings)):
        assert matrix[i][i] == 0.0


def test_pairwise_matrix_is_symmetric():
    embeddings = [embed(r) for r in ["The limit is 25MB.", "Contact support for help.", "The limit is 100MB."]]
    matrix = _pairwise_matrix(embeddings)
    n = len(embeddings)
    for i in range(n):
        for j in range(n):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-6


def test_pairwise_matrix_identical_responses_near_zero():
    embeddings = [embed("The refund window is 30 days.")] * 3
    matrix = _pairwise_matrix(embeddings)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert matrix[i][j] < 0.05


# ── median response ───────────────────────────────────────────────────────────

def test_median_response_returns_one_of_the_inputs():
    responses = ["Answer A.", "Answer B.", "Answer C."]
    embeddings = [embed(r) for r in responses]
    idx = _find_median_idx(embeddings)
    assert responses[idx] in responses
    assert 0 <= idx < len(responses)


def test_median_response_on_identical_inputs():
    text = "The policy is 30 days."
    embeddings = [embed(text)] * 5
    idx = _find_median_idx(embeddings)
    assert 0 <= idx < 5


# ── verdict ───────────────────────────────────────────────────────────────────

def test_verdict_reliable():
    assert _verdict(0.95) == "RELIABLE"

def test_verdict_unstable():
    assert _verdict(0.80) == "UNSTABLE"

def test_verdict_critical():
    assert _verdict(0.50) == "CRITICAL"


# ── full report via mocked provider ──────────────────────────────────────────

def test_reliable_model_high_score():
    """Same response every time → reliability near 1.0."""
    answer = "The upload limit is 25MB per file."
    provider = _mock_provider([answer] * 5)

    report = run_reliability_test(
        "What is the upload limit?",
        runs=5,
        provider=provider,
    )

    assert report.reliability_score >= 0.95
    assert report.verdict == "RELIABLE"
    assert report.outlier_count == 0
    assert report.runs == 5


def test_unreliable_model_low_score():
    """Completely different responses every time → low reliability."""
    responses = [
        "The upload limit is 25MB.",
        "Contact support for file size questions.",
        "Files up to 1GB are supported.",
        "There is no upload limit.",
        "The maximum is 500KB.",
    ]
    provider = _mock_provider(responses)

    report = run_reliability_test(
        "What is the upload limit?",
        runs=5,
        provider=provider,
    )

    assert report.reliability_score < 0.80
    assert report.runs == 5
    assert report.mean_pairwise_divergence > 0.0


def test_report_structure_fields():
    """Report must contain all expected fields."""
    answer = "The policy is 30 days."
    provider = _mock_provider([answer] * 3)

    report = run_reliability_test("Refund policy?", runs=3, provider=provider)

    assert report.prompt == "Refund policy?"
    assert len(report.responses) == 3
    assert len(report.run_results) == 3
    assert len(report.pairwise_matrix) == 3
    assert report.median_response in report.responses
    assert 0.0 <= report.reliability_score <= 1.0
    assert report.verdict in ("RELIABLE", "UNSTABLE", "CRITICAL")
    assert report.outlier_threshold == 0.25


def test_run_results_have_correct_indices():
    provider = _mock_provider(["Answer."] * 4)
    report = run_reliability_test("Q?", runs=4, provider=provider)
    indices = [r.run_index for r in report.run_results]
    assert indices == [1, 2, 3, 4]


def test_outlier_flagged_correctly():
    """One wildly different response should be flagged as outlier."""
    responses = [
        "The limit is 25MB.",
        "The limit is 25MB.",
        "The limit is 25MB.",
        "Call our billing team at 1-800-XXX-XXXX for enterprise pricing.",  # outlier
        "The limit is 25MB.",
    ]
    provider = _mock_provider(responses)
    report = run_reliability_test("Upload limit?", runs=5, provider=provider)
    assert report.outlier_count >= 1


def test_max_divergence_gte_mean():
    responses = [
        "The limit is 25MB.",
        "Contact support.",
        "No file limit exists.",
    ]
    provider = _mock_provider(responses)
    report = run_reliability_test("Upload limit?", runs=3, provider=provider)
    assert report.max_pairwise_divergence >= report.mean_pairwise_divergence
    assert report.min_pairwise_divergence <= report.mean_pairwise_divergence

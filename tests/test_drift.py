"""Unit tests for feature extraction and PSI drift math."""

from __future__ import annotations

from vibe_check.drift import build_histogram, build_reference_stats, compute_psi, evaluate_drift
from vibe_check.features import FEATURE_NAMES, extract_features


def test_extract_features_basic() -> None:
    feats = extract_features("Hello WORLD!! 123")
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["char_count"] == len("Hello WORLD!! 123")
    assert feats["word_count"] >= 2
    assert feats["exclamation_rate"] > 0
    assert feats["digit_ratio"] > 0


def test_psi_identical_distributions_near_zero() -> None:
    expected = [0.2, 0.3, 0.5]
    actual = [0.2, 0.3, 0.5]
    assert compute_psi(expected, actual) < 1e-9


def test_psi_shifted_distributions_positive() -> None:
    expected = [0.7, 0.2, 0.1]
    actual = [0.1, 0.2, 0.7]
    assert compute_psi(expected, actual) > 0.25


def test_build_histogram_handles_constant_values() -> None:
    hist = build_histogram([5.0, 5.0, 5.0], n_bins=5)
    assert len(hist.bins) == 6
    assert abs(sum(hist.proportions) - 1.0) < 1e-6


def test_evaluate_drift_flags_shift() -> None:
    reference_rows = [
        {
            "char_count": 20.0,
            "word_count": 4.0,
            "avg_word_len": 4.0,
            "exclamation_rate": 0.0,
            "question_rate": 0.0,
            "uppercase_ratio": 0.05,
            "digit_ratio": 0.0,
        }
        for _ in range(40)
    ]
    # Mild natural noise around the same regime
    for i in range(40):
        reference_rows[i]["char_count"] = 18 + (i % 5)

    reference = build_reference_stats(reference_rows, n_bins=5)

    live_ok = [dict(row) for row in reference_rows[:35]]
    alert_ok, max_psi_ok, _ = evaluate_drift(reference, live_ok, threshold=0.25)
    assert alert_ok is False
    assert max_psi_ok < 0.25

    live_drift = [
        {
            "char_count": 400.0,
            "word_count": 80.0,
            "avg_word_len": 12.0,
            "exclamation_rate": 0.2,
            "question_rate": 0.1,
            "uppercase_ratio": 0.5,
            "digit_ratio": 0.3,
        }
        for _ in range(35)
    ]
    alert, max_psi, reports = evaluate_drift(reference, live_drift, threshold=0.25)
    assert alert is True
    assert max_psi >= 0.25
    assert any(r["status"] == "drift" for r in reports)
    assert all(isinstance(r["psi"], float) for r in reports)


def test_compute_psi_rejects_mismatched_bins() -> None:
    try:
        compute_psi([0.5, 0.5], [1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass

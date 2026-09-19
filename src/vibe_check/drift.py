"""Population Stability Index (PSI) drift detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from vibe_check.features import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureHistogram:
    """Binned reference distribution for one feature."""

    bins: list[float]
    proportions: list[float]


ReferenceStats = dict[str, FeatureHistogram]


def _safe_proportion(count: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return count / total


def build_histogram(values: Sequence[float], n_bins: int = 10) -> FeatureHistogram:
    """Build equal-width bins with a small epsilon floor for empty bins."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        edges = list(np.linspace(0.0, 1.0, n_bins + 1))
        props = [1.0 / n_bins] * n_bins
        return FeatureHistogram(bins=edges, proportions=props)

    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:
        hi = lo + 1.0
    edges = list(np.linspace(lo, hi, n_bins + 1))
    counts, _ = np.histogram(arr, bins=edges)
    total = float(counts.sum())
    # Floor empty bins so PSI stays defined
    props = [max(_safe_proportion(float(c), total), 1e-4) for c in counts]
    prop_sum = sum(props)
    props = [p / prop_sum for p in props]
    return FeatureHistogram(bins=edges, proportions=props)


def compute_psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    """
    Classic PSI between two proportion vectors.

    Rule of thumb: <0.1 stable, 0.1–0.25 mild shift, >0.25 significant drift.
    """
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of bins")

    psi = 0.0
    for e, a in zip(expected, actual):
        e_safe = max(float(e), 1e-4)
        a_safe = max(float(a), 1e-4)
        psi += (a_safe - e_safe) * np.log(a_safe / e_safe)
    return float(psi)


def histogram_from_live(
    values: Sequence[float],
    reference: FeatureHistogram,
) -> list[float]:
    """Project live values onto the reference bin edges."""
    arr = np.asarray(values, dtype=float)
    counts, _ = np.histogram(arr, bins=reference.bins)
    total = float(counts.sum())
    props = [max(_safe_proportion(float(c), total), 1e-4) for c in counts]
    prop_sum = sum(props)
    return [p / prop_sum for p in props]


def build_reference_stats(
    feature_rows: Iterable[Mapping[str, float]],
    n_bins: int = 10,
) -> ReferenceStats:
    columns: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for row in feature_rows:
        for name in FEATURE_NAMES:
            columns[name].append(float(row[name]))
    return {name: build_histogram(values, n_bins=n_bins) for name, values in columns.items()}


def reference_to_json(stats: ReferenceStats) -> dict[str, dict[str, list[float]]]:
    return {
        name: {"bins": hist.bins, "proportions": hist.proportions}
        for name, hist in stats.items()
    }


def reference_from_json(payload: Mapping[str, Mapping[str, list[float]]]) -> ReferenceStats:
    return {
        name: FeatureHistogram(bins=list(body["bins"]), proportions=list(body["proportions"]))
        for name, body in payload.items()
    }


def evaluate_drift(
    reference: ReferenceStats,
    live_rows: Sequence[Mapping[str, float]],
    threshold: float,
) -> tuple[bool, float, list[dict[str, float | str]]]:
    """Return (alert, max_psi, per-feature reports)."""
    reports: list[dict[str, float | str]] = []
    max_psi = 0.0

    for name in FEATURE_NAMES:
        ref = reference[name]
        values = [float(row[name]) for row in live_rows]
        live_props = histogram_from_live(values, ref)
        psi = compute_psi(ref.proportions, live_props)
        max_psi = max(max_psi, psi)
        status = "ok" if psi < threshold else "drift"
        reports.append({"feature": name, "psi": psi, "status": status})

    alert = max_psi >= threshold
    return alert, max_psi, reports

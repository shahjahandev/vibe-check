"""Sliding-window monitor for live request feature distributions."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Mapping

from vibe_check.config import Settings
from vibe_check.drift import ReferenceStats, evaluate_drift, reference_from_json
from vibe_check.schemas import DriftFeatureReport, DriftStatusResponse

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Keeps a bounded window of live features and compares to reference PSI."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._window: Deque[dict[str, float]] = deque(maxlen=settings.window_size)
        self._samples_seen = 0
        self._reference: ReferenceStats | None = None
        self._last_alert = False
        self._load_reference()

    def _load_reference(self) -> None:
        path = self._settings.reference_stats_path
        if not path.exists():
            logger.warning("Reference stats missing at %s — drift checks disabled", path)
            self._reference = None
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._reference = reference_from_json(payload)

    @property
    def reference_loaded(self) -> bool:
        return self._reference is not None

    def observe(self, features: Mapping[str, float]) -> DriftStatusResponse | None:
        """Record one request. Returns a status payload when an alert fires."""
        row = {k: float(v) for k, v in features.items()}
        with self._lock:
            self._window.append(row)
            self._samples_seen += 1
            status = self._compute_status_unlocked()
            self._persist_row(row)

        if status.alert and not self._last_alert:
            logger.warning("FEATURE DRIFT ALERT: %s", status.message)
        self._last_alert = status.alert
        return status if status.alert else None

    def status(self) -> DriftStatusResponse:
        with self._lock:
            return self._compute_status_unlocked()

    def _compute_status_unlocked(self) -> DriftStatusResponse:
        window_size = len(self._window)
        threshold = self._settings.psi_alert_threshold

        if self._reference is None:
            return DriftStatusResponse(
                window_size=window_size,
                samples_seen=self._samples_seen,
                alert=False,
                alert_threshold=threshold,
                max_psi=0.0,
                features=[],
                message="Reference stats not loaded; drift monitoring inactive.",
            )

        if window_size < self._settings.min_window_for_drift:
            return DriftStatusResponse(
                window_size=window_size,
                samples_seen=self._samples_seen,
                alert=False,
                alert_threshold=threshold,
                max_psi=0.0,
                features=[],
                message=(
                    f"Collecting baseline live traffic "
                    f"({window_size}/{self._settings.min_window_for_drift})."
                ),
            )

        alert, max_psi, reports = evaluate_drift(
            self._reference,
            list(self._window),
            threshold,
        )
        feature_reports = [
            DriftFeatureReport(
                feature=str(r["feature"]),
                psi=float(r["psi"]),
                status=str(r["status"]),
            )
            for r in reports
        ]
        if alert:
            message = (
                f"Feature drift detected (max PSI={max_psi:.3f} ≥ {threshold}). "
                "Check data quality or shifting consumer language."
            )
        else:
            message = f"Distributions stable (max PSI={max_psi:.3f})."

        return DriftStatusResponse(
            window_size=window_size,
            samples_seen=self._samples_seen,
            alert=alert,
            alert_threshold=threshold,
            max_psi=max_psi,
            features=feature_reports,
            message=message,
        )

    def _persist_row(self, row: Mapping[str, float]) -> None:
        path = self._settings.live_window_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

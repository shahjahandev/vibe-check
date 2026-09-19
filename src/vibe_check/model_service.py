"""Model loading and low-latency inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from vibe_check.features import extract_features
from vibe_check.schemas import PredictResponse, SentimentLabel

logger = logging.getLogger(__name__)


class ModelService:
    """Thin wrapper around the trained sklearn pipeline."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._pipeline: Pipeline | None = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self._model_path}. Run `python train.py` first."
            )
        self._pipeline = joblib.load(self._model_path)
        logger.info("Loaded model from %s", self._model_path)

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str, request_id: str | None = None) -> PredictResponse:
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        started = time.perf_counter()
        label_raw: str = self._pipeline.predict([text])[0]
        proba = self._pipeline.predict_proba([text])[0]
        classes: Any = self._pipeline.classes_
        probabilities = {
            SentimentLabel(str(cls)): float(score) for cls, score in zip(classes, proba)
        }
        label = SentimentLabel(str(label_raw))
        features = extract_features(text)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return PredictResponse(
            request_id=request_id,
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            features=features,
            latency_ms=round(latency_ms, 3),
        )

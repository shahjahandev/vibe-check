"""API integration tests with a temporary model + reference stats."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from vibe_check.config import Settings
from vibe_check.drift import build_reference_stats, reference_to_json
from vibe_check.features import extract_features


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    texts = [
        "I love this wonderful day",
        "This is terrible and awful",
        "The meeting is at noon",
        "Fantastic news we won",
        "I hate everything today",
        "Please send the report",
        "So happy and grateful",
        "Angry frustrated and sad",
        "Water boils at one hundred",
        "Joyful celebration tonight",
        "Miserable cold weather",
        "Open the drawer slowly",
    ]
    labels = [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
    ]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(texts, labels)

    model_path = tmp_path / "model.joblib"
    ref_path = tmp_path / "reference.json"
    live_path = tmp_path / "live.jsonl"
    joblib.dump(pipe, model_path)

    rows = [extract_features(t) for t in texts]
    ref = build_reference_stats(rows, n_bins=5)
    ref_path.write_text(json.dumps(reference_to_json(ref)), encoding="utf-8")

    settings = Settings(
        model_path=model_path,
        reference_stats_path=ref_path,
        live_window_path=live_path,
        window_size=50,
        min_window_for_drift=5,
        psi_alert_threshold=0.25,
    )

    monkeypatch.setattr("vibe_check.api.settings", settings)

    # Rebuild services bound to patched settings
    from vibe_check.model_service import ModelService
    from vibe_check.monitor import DriftMonitor
    import vibe_check.api as api

    api.model_service = ModelService(settings.model_path)
    api.drift_monitor = DriftMonitor(settings)
    api.model_service.load()

    with TestClient(api.app) as client:
        yield client


def test_health(api_client: TestClient) -> None:
    res = api_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["reference_stats_loaded"] is True


def test_predict_happy_path(api_client: TestClient) -> None:
    res = api_client.post(
        "/predict",
        json={"text": "I am so happy and grateful today", "request_id": "t-1"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["request_id"] == "t-1"
    assert body["label"] in {"positive", "neutral", "negative"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert "char_count" in body["features"]
    assert body["latency_ms"] >= 0


def test_predict_rejects_short_text(api_client: TestClient) -> None:
    res = api_client.post("/predict", json={"text": "no"})
    assert res.status_code == 422


def test_drift_endpoint(api_client: TestClient) -> None:
    for i in range(6):
        api_client.post("/predict", json={"text": f"Please send the weekly report number {i}"})
    res = api_client.get("/drift")
    assert res.status_code == 200
    body = res.json()
    assert body["samples_seen"] >= 6
    assert "message" in body

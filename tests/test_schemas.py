"""Payload validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibe_check.schemas import PredictRequest, SentimentLabel


def test_predict_request_strips_whitespace() -> None:
    req = PredictRequest(text="  hello there  ")
    assert req.text == "hello there"


def test_predict_request_rejects_too_short() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="hi")


def test_predict_request_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="   ")


def test_sentiment_label_values() -> None:
    assert SentimentLabel.positive.value == "positive"
    assert set(SentimentLabel) == {
        SentimentLabel.negative,
        SentimentLabel.neutral,
        SentimentLabel.positive,
    }

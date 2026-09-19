"""Request/response contracts with strict payload validation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class PredictRequest(BaseModel):
    """Incoming prediction payload."""

    text: str = Field(..., min_length=3, max_length=5_000, description="Raw text to score")
    request_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client correlation id",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("text must contain at least 3 non-whitespace characters")
        return cleaned


class PredictResponse(BaseModel):
    """Prediction result plus monitoring fingerprint."""

    request_id: Optional[str]
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[SentimentLabel, float]
    features: Dict[str, float]
    latency_ms: float
    served_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DriftFeatureReport(BaseModel):
    feature: str
    psi: float
    status: str


class DriftStatusResponse(BaseModel):
    """Live drift report against the training reference distribution."""

    window_size: int
    samples_seen: int
    alert: bool
    alert_threshold: float
    max_psi: float
    features: List[DriftFeatureReport]
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reference_stats_loaded: bool
    version: str

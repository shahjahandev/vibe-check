"""Lightweight feature extraction used for drift monitoring."""

from __future__ import annotations

import re
from typing import Mapping

FEATURE_NAMES: tuple[str, ...] = (
    "char_count",
    "word_count",
    "avg_word_len",
    "exclamation_rate",
    "question_rate",
    "uppercase_ratio",
    "digit_ratio",
)

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def extract_features(text: str) -> dict[str, float]:
    """Derive stable numeric features from raw text for distribution tracking."""
    chars = max(len(text), 1)
    words = _WORD_RE.findall(text)
    word_count = float(len(words))
    avg_word_len = (sum(len(w) for w in words) / word_count) if words else 0.0
    upper = sum(1 for ch in text if ch.isupper())
    digits = sum(1 for ch in text if ch.isdigit())

    return {
        "char_count": float(len(text)),
        "word_count": word_count,
        "avg_word_len": float(avg_word_len),
        "exclamation_rate": text.count("!") / chars,
        "question_rate": text.count("?") / chars,
        "uppercase_ratio": upper / chars,
        "digit_ratio": digits / chars,
    }


def feature_vector(features: Mapping[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]

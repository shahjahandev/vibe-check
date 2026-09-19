"""Train the sentiment model and freeze reference feature distributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vibe_check.config import settings  # noqa: E402
from vibe_check.drift import build_reference_stats, reference_to_json  # noqa: E402
from vibe_check.features import extract_features  # noqa: E402


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(settings.training_data_path)
    pipe = build_pipeline()

    scores = cross_val_score(pipe, df["text"], df["label"], cv=5)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(df["text"], df["label"])

    feature_rows = [extract_features(text) for text in df["text"].astype(str)]
    reference = build_reference_stats(feature_rows, n_bins=10)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, settings.model_path)
    settings.reference_stats_path.write_text(
        json.dumps(reference_to_json(reference), indent=2),
        encoding="utf-8",
    )

    # Clear stale live window so a fresh deploy starts clean.
    if settings.live_window_path.exists():
        settings.live_window_path.unlink()

    print(f"Saved model → {settings.model_path}")
    print(f"Saved reference stats → {settings.reference_stats_path}")


if __name__ == "__main__":
    main()

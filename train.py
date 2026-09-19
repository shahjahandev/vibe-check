"""Train the Vibe Check sentiment model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "vibes.csv"
MODEL_PATH = ROOT / "artifacts" / "vibe_model.joblib"


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
    df = pd.read_csv(DATA_PATH)
    pipe = build_pipeline()
    scores = cross_val_score(pipe, df["text"], df["label"], cv=5)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    pipe.fit(df["text"], df["label"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")


if __name__ == "__main__":
    main()

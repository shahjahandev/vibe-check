"""Vibe Check — paste text, get an ML mood reading."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "vibe_model.joblib"

VIBE_META = {
    "positive": {
        "emoji": "✨",
        "title": "Bright vibe",
        "blurb": "Warm, upbeat energy. Keep riding that wave.",
        "color": "#2f9e6b",
    },
    "neutral": {
        "emoji": "🌫️",
        "title": "Steady vibe",
        "blurb": "Calm and factual. Not charged either way.",
        "color": "#5b6b7c",
    },
    "negative": {
        "emoji": "⛈️",
        "title": "Heavy vibe",
        "blurb": "Tense or low energy. Maybe a pause would help.",
        "color": "#c4493a",
    },
}

EXAMPLES = {
    "Happy win": "We shipped the project and the whole team is celebrating!",
    "Rough day": "Nothing is working and I feel completely drained.",
    "Just facts": "The meeting starts at 3pm in conference room B.",
    "Grateful note": "Thank you for showing up. Your kindness meant everything.",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_vibe(model, text: str) -> tuple[str, dict[str, float]]:
    label = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    scores = {cls: float(prob) for cls, prob in zip(model.classes_, probs)}
    return label, scores


def main() -> None:
    st.set_page_config(
        page_title="Vibe Check",
        page_icon="🎛️",
        layout="centered",
    )

    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(1200px 500px at 10% -10%, #d9f2e6 0%, transparent 55%),
              radial-gradient(900px 420px at 100% 0%, #f7e6d9 0%, transparent 50%),
              linear-gradient(180deg, #f7f4ef 0%, #eef2f6 100%);
          }
          .hero {
            text-align: center;
            padding: 0.5rem 0 1rem;
          }
          .hero h1 {
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            font-size: 2.6rem;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
            color: #1d2a33;
          }
          .hero p {
            color: #52606d;
            font-size: 1.05rem;
            margin-top: 0;
          }
          .result-card {
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            margin-top: 0.75rem;
            border: 1px solid rgba(0,0,0,0.06);
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(8px);
          }
          .result-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.35rem;
          }
          .muted { color: #61707d; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>Vibe Check</h1>
          <p>A tiny ML mood reader for any sentence you throw at it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error("Model missing. Run `python train.py` first, then restart the app.")
        st.stop()

    cols = st.columns(len(EXAMPLES))
    chosen = None
    for col, (name, sample) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            chosen = sample

    default_text = chosen or ""
    text = st.text_area(
        "Paste a message, journal line, review, or tweet",
        value=default_text,
        height=140,
        placeholder="e.g. Today felt productive and strangely joyful…",
    )

    run = st.button("Check the vibe", type="primary", use_container_width=True)

    if run or chosen:
        cleaned = text.strip()
        if len(cleaned) < 3:
            st.warning("Give me at least a few words to work with.")
            return

        label, scores = predict_vibe(model, cleaned)
        meta = VIBE_META[label]
        confidence = scores[label]

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-title" style="color:{meta['color']}">
                {meta['emoji']} {meta['title']}
              </div>
              <div class="muted">{meta['blurb']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(confidence, text=f"Model confidence: {confidence:.0%}")

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        metric_cols = st.columns(3)
        for col, (name, score) in zip(metric_cols, ordered):
            col.metric(name.capitalize(), f"{score:.0%}")

        with st.expander("How it works"):
            st.write(
                "A scikit-learn pipeline turns your text into TF-IDF features, "
                "then a logistic regression classifier predicts **positive**, "
                "**neutral**, or **negative**. Trained on a small curated vibe dataset "
                "in this repo — lightweight, local, and easy to retrain."
            )


if __name__ == "__main__":
    main()

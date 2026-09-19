"""Vibe Check — paste text, get an ML mood reading."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "artifacts" / "vibe_model.joblib"

VIBE_META = {
    "positive": {
        "title": "This leans bright",
        "blurb": "Upbeat wording. Sounds like a good day.",
        "bar": "#1f7a4d",
    },
    "neutral": {
        "title": "Pretty even",
        "blurb": "Mostly factual. Not much emotional charge.",
        "bar": "#3d4f5f",
    },
    "negative": {
        "title": "This leans heavy",
        "blurb": "Tense or low energy. Worth a second look.",
        "bar": "#b33a2b",
    },
}

EXAMPLES = {
    "happy win": "We shipped the project and the whole team is celebrating!",
    "rough day": "Nothing is working and I feel completely drained.",
    "just facts": "The meeting starts at 3pm in conference room B.",
    "thanks": "Thank you for showing up. Your kindness meant everything.",
}

LABEL_ORDER = ("negative", "neutral", "positive")


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


def score_bars_html(scores: dict[str, float], winner: str) -> str:
    rows = []
    for name in LABEL_ORDER:
        pct = int(round(scores.get(name, 0) * 100))
        color = VIBE_META[name]["bar"]
        weight = "700" if name == winner else "500"
        rows.append(
            f"""
            <div class="score-row">
              <div class="score-label" style="font-weight:{weight}">{name}</div>
              <div class="score-track">
                <div class="score-fill" style="width:{pct}%; background:{color}"></div>
              </div>
              <div class="score-pct" style="font-weight:{weight}">{pct}%</div>
            </div>
            """
        )
    return '<div class="score-board">' + "".join(rows) + "</div>"


def inject_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap" rel="stylesheet">
        <style>
          :root {
            --ink: #141c24;
            --muted: #3a4754;
            --line: #c5ced6;
            --panel: #f3f6f8;
            --accent: #0b6e4f;
          }
          html, body, [class*="css"] { color: var(--ink) !important; }
          .stApp {
            background:
              repeating-linear-gradient(
                0deg, transparent, transparent 23px,
                rgba(20, 28, 36, 0.035) 23px, rgba(20, 28, 36, 0.035) 24px
              ),
              linear-gradient(165deg, #d7e1ea 0%, #cfd9e2 45%, #e4ebe8 100%);
            color: var(--ink);
          }
          .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
          .stMarkdown, .stMarkdown p, .stCaption,
          [data-testid="stMarkdownContainer"],
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stWidgetLabel"] p,
          label {
            color: var(--ink) !important;
            opacity: 1 !important;
          }
          [data-testid="stHeader"] { background: transparent; }
          .block-container { max-width: 680px; padding-top: 2.2rem; padding-bottom: 3rem; }
          .masthead {
            margin-bottom: 1.4rem;
            border-bottom: 2px solid var(--ink);
            padding-bottom: 0.85rem;
          }
          .masthead .brand {
            font-family: "Source Serif 4", Georgia, serif;
            font-size: 2.35rem; font-weight: 700; letter-spacing: -0.02em;
            line-height: 1.05; color: var(--ink); margin: 0;
          }
          .masthead .tag {
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
            color: var(--muted); margin-top: 0.45rem;
          }
          .masthead .lede {
            font-family: "Source Sans 3", system-ui, sans-serif;
            font-size: 1.05rem; color: var(--muted); margin: 0.55rem 0 0; max-width: 34rem;
          }
          .panel {
            background: var(--panel); border: 1.5px solid var(--ink);
            padding: 1rem 1.1rem 1.05rem; margin: 1rem 0 0.35rem;
          }
          .panel h2 {
            font-family: "Source Serif 4", Georgia, serif;
            font-size: 1.45rem; margin: 0 0 0.3rem; color: var(--ink);
          }
          .panel .note {
            font-family: "Source Sans 3", system-ui, sans-serif;
            color: var(--muted); margin: 0;
          }
          .score-board { margin-top: 0.85rem; }
          .score-row {
            display: grid; grid-template-columns: 5.2rem 1fr 2.6rem;
            gap: 0.55rem; align-items: center; margin: 0.4rem 0;
            font-family: "IBM Plex Mono", ui-monospace, monospace;
            font-size: 0.86rem; color: var(--ink);
          }
          .score-label { text-transform: lowercase; color: var(--ink); }
          .score-pct { text-align: right; color: var(--ink); }
          .score-track { height: 10px; background: #c9d3dc; border: 1px solid var(--ink); }
          .score-fill { height: 100%; }
          .howto {
            margin-top: 1.1rem; padding-top: 0.85rem; border-top: 1px dashed var(--line);
            font-family: "Source Sans 3", system-ui, sans-serif;
            font-size: 0.95rem; color: var(--muted); line-height: 1.45;
          }
          .howto strong { color: var(--ink); }
          .stTextArea textarea {
            background: #f7fafc !important; color: var(--ink) !important;
            border: 1.5px solid var(--ink) !important; border-radius: 0 !important;
            font-family: "Source Sans 3", system-ui, sans-serif !important;
          }
          .stTextArea textarea::placeholder { color: #667788 !important; opacity: 1 !important; }
          .stButton > button {
            border-radius: 0 !important; border: 1.5px solid var(--ink) !important;
            background: #eef3f7 !important; color: var(--ink) !important;
            font-family: "IBM Plex Mono", ui-monospace, monospace !important;
            font-weight: 600 !important; box-shadow: none !important;
          }
          .stButton > button[kind="primary"],
          .stButton > button[data-testid="baseButton-primary"] {
            background: var(--accent) !important; color: #f4faf7 !important;
            border-color: #084c37 !important;
          }
          footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Vibe Check", page_icon="VC", layout="centered")
    inject_styles()

    if "draft" not in st.session_state:
        st.session_state.draft = ""

    st.markdown(
        """
        <div class="masthead">
          <p class="brand">Vibe Check</p>
          <p class="tag">local ml · sentiment readout</p>
          <p class="lede">
            Type a sentence. A small model reads the mood — a useful gut-check
            for messages, notes, and reviews.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.error("Model missing. Run `python train.py`, then restart.")
        st.stop()

    st.caption("try a sample")
    cols = st.columns(len(EXAMPLES))
    for col, (name, sample) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True, key=f"ex_{name}"):
            st.session_state.draft = sample

    text = st.text_area("your text", key="draft", height=150)

    if st.button("read the vibe", type="primary", use_container_width=True):
        cleaned = (text or "").strip()
        if len(cleaned) < 3:
            st.warning("Need a few more words than that.")
            return

        label, scores = predict_vibe(model, cleaned)
        meta = VIBE_META[label]
        confidence = scores[label]
        st.markdown(
            f"""
            <div class="panel">
              <h2 style="color:{meta['bar']}">{meta['title']}</h2>
              <p class="note">{meta['blurb']} · model is {confidence:.0%} sure</p>
              {score_bars_html(scores, label)}
              <div class="howto">
                <strong>How it works:</strong>
                TF-IDF + logistic regression predicts
                <strong>negative</strong>, <strong>neutral</strong>, or
                <strong>positive</strong>. Retrain with <code>python train.py</code>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

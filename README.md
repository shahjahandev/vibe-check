# Vibe Check

A tiny **Python + ML** mood reader. Paste any sentence and a scikit-learn model scores whether it feels **positive**, **neutral**, or **negative**.

Useful for journaling, feedback, tweets, or messages. Fun enough to click around with the sample buttons.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What's inside

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `train.py` | Trains and saves the model |
| `data/vibes.csv` | Labeled training phrases |
| `model/vibe_model.joblib` | Saved TF-IDF + logistic regression pipeline |

## Model

- **Features:** TF-IDF unigrams + bigrams
- **Classifier:** logistic regression (balanced classes)
- **Labels:** `positive` · `neutral` · `negative`

Retrain anytime after editing `data/vibes.csv`:

```bash
python train.py
```

## Deploy

### GitHub

This repo is meant to live on GitHub. Push it, then anyone can clone and run locally.

### Streamlit Community Cloud (optional live demo)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with:
   - Main file: `app.py`
   - Python version: 3.11+
4. Add an app startup command / note that the model file is committed, or set a build step to run `python train.py`

## License

MIT — do whatever you want with it.

# Vibe Check

A small **Streamlit** app that scores the mood of any sentence with scikit-learn.

This is the fun demo app. For the separate ML **serving + feature-drift monitoring** API, see **[driftwatch](https://github.com/shahjahandev/driftwatch)**.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## What's inside

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `train.py` | Trains TF-IDF + logistic regression |
| `data/vibes.csv` | Labeled training phrases |
| `artifacts/vibe_model.joblib` | Saved model |

## License

MIT

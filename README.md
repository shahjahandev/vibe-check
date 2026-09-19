# Vibe Check

Low-latency **sentiment API** with **real-time feature-drift monitoring**.

This is not just a model notebook. It shows the ML lifecycle past training: validated payloads, fast serving, and post-deployment observability that alerts when live input distributions diverge from the training reference (data quality issues or shifting consumer language).

## Why this project

| Concern | What you get |
| --- | --- |
| Serving | FastAPI `/predict` with millisecond-scale sklearn inference |
| Validation | Pydantic request models (length, blank text, optional `request_id`) |
| Observability | Sliding-window **PSI** (Population Stability Index) on text features |
| Hygiene | Type hints, package layout under `src/`, pytest suite, Docker |

## Architecture

```
client ──► POST /predict ──► ModelService (joblib pipeline)
                │
                └──► DriftMonitor (feature window vs reference histograms)
                           │
                           └──► GET /drift  (alert when max PSI ≥ threshold)
```

Monitored features: `char_count`, `word_count`, `avg_word_len`, `exclamation_rate`, `question_rate`, `uppercase_ratio`, `digit_ratio`.

PSI rule of thumb: `< 0.10` stable · `0.10–0.25` mild shift · `≥ 0.25` alert.

## Project layout

```
src/vibe_check/
  api.py            FastAPI routes
  schemas.py        Request/response contracts
  model_service.py  Inference wrapper
  features.py       Monitoring feature extraction
  drift.py          PSI math + reference histograms
  monitor.py        Live sliding window + alerts
  config.py         Env-configurable settings
tests/              Unit + API tests
train.py            Train model + freeze reference stats
ui.py               Optional Streamlit client
Dockerfile
docker-compose.yml
```

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
PYTHONPATH=src uvicorn vibe_check.api:app --host 0.0.0.0 --port 8000
```

### Try it

```bash
curl -s localhost:8000/health | jq

curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"We shipped it and the whole team is celebrating!","request_id":"demo-1"}' | jq

curl -s localhost:8000/drift | jq
```

Optional UI (API must be running):

```bash
streamlit run ui.py
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Docker

```bash
docker compose up --build
```

API: http://localhost:8000/docs

## Configuration

All settings use the `VIBE_` prefix:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VIBE_WINDOW_SIZE` | `100` | Live feature window length |
| `VIBE_MIN_WINDOW_FOR_DRIFT` | `30` | Samples before PSI is computed |
| `VIBE_PSI_ALERT_THRESHOLD` | `0.25` | Alert when max feature PSI crosses this |
| `VIBE_MODEL_PATH` | `artifacts/vibe_model.joblib` | Model artifact |
| `VIBE_REFERENCE_STATS_PATH` | `artifacts/reference_stats.json` | Training feature histograms |

## API surface

- `GET /health` — liveness + model/reference status
- `POST /predict` — validated inference; updates the drift window
- `GET /drift` — current PSI report and alert flag
- Interactive docs at `/docs`

## License

MIT

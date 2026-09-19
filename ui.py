"""Optional Streamlit demo UI that calls the local prediction API."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("VIBE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Vibe Check", page_icon="VC", layout="centered")
st.title("Vibe Check")
st.caption("UI client → FastAPI `/predict` + live `/drift` monitor")

text = st.text_area("Text", height=140, placeholder="Paste a message…")
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict", type="primary", use_container_width=True):
        if len(text.strip()) < 3:
            st.warning("Need a bit more text.")
        else:
            try:
                res = httpx.post(f"{API_URL}/predict", json={"text": text}, timeout=10.0)
                res.raise_for_status()
                body = res.json()
                st.subheader(body["label"])
                st.write(f"Confidence: **{body['confidence']:.0%}** · {body['latency_ms']:.1f} ms")
                st.json(body["probabilities"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"API error: {exc}")

with col2:
    if st.button("Drift status", use_container_width=True):
        try:
            res = httpx.get(f"{API_URL}/drift", timeout=10.0)
            res.raise_for_status()
            st.json(res.json())
        except Exception as exc:  # noqa: BLE001
            st.error(f"API error: {exc}")

"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the prediction API."""

    model_config = SettingsConfigDict(env_prefix="VIBE_", env_file=".env", extra="ignore")

    app_name: str = "Vibe Check API"
    model_path: Path = ROOT / "artifacts" / "vibe_model.joblib"
    reference_stats_path: Path = ROOT / "artifacts" / "reference_stats.json"
    live_window_path: Path = ROOT / "artifacts" / "live_window.jsonl"
    training_data_path: Path = ROOT / "data" / "vibes.csv"

    # Drift monitoring
    window_size: int = 100
    psi_alert_threshold: float = 0.25
    min_window_for_drift: int = 30

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()

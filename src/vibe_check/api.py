"""FastAPI application: predict, health, and drift endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from vibe_check import __version__
from vibe_check.config import settings
from vibe_check.model_service import ModelService
from vibe_check.monitor import DriftMonitor
from vibe_check.schemas import (
    DriftStatusResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

model_service = ModelService(settings.model_path)
drift_monitor = DriftMonitor(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    model_service.load()
    logger.info("Vibe Check API ready (v%s)", __version__)
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Low-latency sentiment predictions with live feature-drift monitoring. "
        "Validates payloads, serves sklearn inference, and alerts when input "
        "distributions diverge from the training reference."
    ),
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.ready else "degraded",
        model_loaded=model_service.ready,
        reference_stats_loaded=drift_monitor.reference_loaded,
        version=__version__,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    if not model_service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = model_service.predict(payload.text, request_id=payload.request_id)
    alert = drift_monitor.observe(result.features)
    if alert is not None:
        # Surface drift without failing the prediction — observability, not gatekeeping.
        logger.warning("Drift alert on request_id=%s: %s", payload.request_id, alert.message)
    return result


@app.get("/drift", response_model=DriftStatusResponse, tags=["ops"])
def drift_status() -> DriftStatusResponse:
    return drift_monitor.status()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "vibe_check.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()

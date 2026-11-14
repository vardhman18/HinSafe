"""FastAPI application for Hinglish hate-speech detection."""

from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# The API is started from api/, so add the project root for src.predict.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request

from health import router as health_router
from model_loader import load_model_components
from schemas import PredictRequest, PredictResponse
from src.predict import analyze_comment

inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model components at startup and release them at shutdown."""

    print("Starting HinSafe API...")
    prediction_pipeline, explainer = load_model_components()

    app.state.pipeline = prediction_pipeline
    app.state.explainer = explainer
    print("Model ready.")

    yield

    app.state.pipeline = None
    app.state.explainer = None


app = FastAPI(
    title="HinSafe API",
    description=(
        "Hinglish hate-speech and cyberbullying detection "
        "with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(health_router)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Analyze a Hinglish comment",
)
def predict(
    request: PredictRequest,
    http_request: Request,
) -> PredictResponse:
    """Analyze one comment and return its classification and explanation."""

    pipeline = getattr(http_request.app.state, "pipeline", None)
    explainer = getattr(http_request.app.state, "explainer", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded yet.",
        )

    try:
        with inference_lock:
            result = analyze_comment(
                raw_text=request.text,
                prediction_pipeline=pipeline,
                explainer=explainer,
                explain=request.explain,
            )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictResponse(**result)

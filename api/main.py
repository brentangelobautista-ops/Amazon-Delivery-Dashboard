"""Local inference service for the exported LogiTrack regression pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "machine-learning" / "models"
MODEL_PATH = Path(os.getenv("LOGITRACK_MODEL_PATH", MODELS / "delivery_time_pipeline.joblib"))
META_PATH = Path(os.getenv("LOGITRACK_METADATA_PATH", MODELS / "model_metadata.json"))

model: Any | None = None
metadata: dict[str, Any] | None = None
artifact_error: str | None = None


def load_artifacts() -> None:
    """Load the exported pipeline once, while keeping /health available on failure."""

    global artifact_error, metadata, model
    try:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            missing = [str(path.relative_to(ROOT)) for path in (MODEL_PATH, META_PATH) if not path.exists()]
            raise FileNotFoundError(
                f"Missing prediction artifact(s): {', '.join(missing)}. "
                "Run python machine-learning/train_model.py."
            )
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        artifact_error = None
    except Exception as error:  # Keep the API diagnosable instead of crashing at import time.
        model = None
        metadata = None
        artifact_error = f"Prediction artifacts could not be loaded: {error}"


load_artifacts()

origins = [
    item.strip()
    for item in os.getenv(
        "LOGITRACK_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175",
    ).split(",")
    if item.strip()
]

app = FastAPI(title="LogiTrack Prediction API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class DeliveryInput(BaseModel):
    Agent_Age: float | None = None
    Agent_Rating: float | None = None
    Distance_Model_km: float = Field(ge=0, le=50000)
    Order_Hour: int = Field(ge=-1, le=23)
    Day_of_Week: int = Field(ge=0, le=6)
    Weather: str
    Traffic: str
    Vehicle: str
    Area: str
    Category: str


def ready() -> None:
    if model is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail=artifact_error or "Prediction model is unavailable. Run machine-learning/train_model.py.",
        )


def validate_payload(row: dict[str, Any]) -> None:
    assert metadata is not None
    for feature, choices in metadata["allowed_categories"].items():
        if row[feature] not in choices:
            raise HTTPException(status_code=422, detail=f"Unsupported {feature}: {row[feature]}.")
    for feature, bounds in metadata["numeric_bounds"].items():
        if feature == "Distance_Model_km":
            continue
        value = row.get(feature)
        if value is not None and not bounds["min"] <= value <= bounds["max"]:
            raise HTTPException(
                status_code=422,
                detail=f"{feature} must be between {bounds['min']} and {bounds['max']} based on training data.",
            )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": model is not None and metadata is not None,
        "model": metadata.get("model") if metadata else None,
        "detail": artifact_error,
    }


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    ready()
    assert metadata is not None
    return metadata


@app.get("/evaluation")
def evaluation() -> dict[str, Any]:
    ready()
    assert metadata is not None
    return {
        "held_out_metrics": metadata["held_out_metrics"],
        "candidate_models": metadata["candidate_models"],
        "diagnostics": metadata["evaluation_diagnostics"],
        "importance_method": metadata["importance_method"],
    }


@app.post("/predict")
def predict(payload: DeliveryInput) -> dict[str, Any]:
    ready()
    assert metadata is not None and model is not None
    row = payload.model_dump()
    validate_payload(row)
    original_distance = row["Distance_Model_km"]
    row["Distance_Model_km"] = min(original_distance, metadata["distance_model_cap_p99_km"])
    try:
        value = float(model.predict(pd.DataFrame([row]))[0])
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction could not be generated.") from error
    return {
        "predicted_delivery_minutes": round(max(0, value), 1),
        "model": metadata["model"],
        "distance_input_capped": original_distance != row["Distance_Model_km"],
        "disclaimer": "This is a model-based estimate from historical data, not an actual outcome or causal claim.",
    }

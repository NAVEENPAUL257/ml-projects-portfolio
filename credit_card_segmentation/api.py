"""
api.py
------
FastAPI service exposing the segmentation model.

    GET  /health   -> liveness + model metadata
    POST /predict  -> segment + cluster probabilities for 1..N customers
    POST /drift    -> KS drift report for a batch vs. training reference

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clustering import MODEL_PATH, train_segmentation
from data_generator import generate_customers
from drift_monitor import feature_drift_report
from inference import load_bundle, score


class Customer(BaseModel):
    balance: float = Field(..., ge=0)
    purchases: float = Field(..., ge=0)
    oneoff_purchases: float = Field(..., ge=0)
    installments_purchases: float = Field(..., ge=0)
    cash_advance: float = Field(..., ge=0)
    purchases_frequency: float = Field(..., ge=0, le=1)
    credit_limit: float = Field(..., ge=0)
    payments: float = Field(..., ge=0)
    minimum_payments: float = Field(..., ge=0)
    prc_full_payment: float = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0)


class PredictRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1)


app = FastAPI(title="Credit Card Segmentation API", version="1.0.0")


@app.on_event("startup")
def _ensure_model() -> None:
    """Train a model on synthetic data if no artifact exists (self-contained demo)."""
    if not MODEL_PATH.exists():
        train_segmentation(generate_customers(5000))
    load_bundle()


@app.get("/health")
def health():
    b = load_bundle()
    return {"status": "ok", "model": "KMeans", "n_clusters": int(b["model"].n_clusters),
            "segments": list(b["label_map"].values()), "silhouette": round(b["silhouette"], 4)}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return {"predictions": score([c.model_dump() for c in req.customers])}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/drift")
def drift(req: PredictRequest):
    if len(req.customers) < 30:
        raise HTTPException(status_code=422, detail="Need at least 30 records for a drift test")
    current = pd.DataFrame([c.model_dump() for c in req.customers])
    return feature_drift_report(load_bundle()["reference_data"], current)

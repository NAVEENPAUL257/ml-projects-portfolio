"""
inference.py
------------
Scoring layer. Loads the trained bundle once and returns, per customer:
  - segment label + cluster id
  - soft cluster probabilities (softmax over negative squared centroid distances)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from clustering import MODEL_PATH
from preprocessing import validate_input

TEMPERATURE = 1.0  # lower = sharper probabilities


@lru_cache(maxsize=1)
def load_bundle(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def _soft_probabilities(distances: np.ndarray) -> np.ndarray:
    logits = -(distances ** 2) / TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def score(records: List[Dict] | pd.DataFrame, bundle: dict | None = None) -> List[Dict]:
    """Score one or more customers and return segment + cluster probabilities."""
    bundle = bundle or load_bundle()
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
    X = bundle["preprocessor"].transform(validate_input(df))
    clusters = bundle["model"].predict(X)
    probs = _soft_probabilities(bundle["model"].transform(X))
    label_map = bundle["label_map"]
    return [{
        "cluster_id": int(c),
        "segment": label_map[int(c)],
        "probabilities": {label_map[j]: round(float(p[j]), 4) for j in range(len(p))},
    } for c, p in zip(clusters, probs)]

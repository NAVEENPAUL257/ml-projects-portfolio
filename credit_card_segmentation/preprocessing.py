"""
preprocessing.py
----------------
Reusable preprocessing: median imputation -> log1p for heavy-tailed monetary
fields -> standard scaling. Packaged as a fitted sklearn Pipeline so the exact
same transform is applied at training and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data_generator import FEATURES

MONETARY = ["balance", "purchases", "oneoff_purchases", "installments_purchases",
            "cash_advance", "credit_limit", "payments", "minimum_payments"]
RATIO = [f for f in FEATURES if f not in MONETARY]


def build_preprocessor() -> ColumnTransformer:
    monetary = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    ratio = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("monetary", monetary, MONETARY), ("ratio", ratio, RATIO)])


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all features are present, numeric and non-negative."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    out = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if (out < 0).any().any():
        raise ValueError("Features must be non-negative")
    return out

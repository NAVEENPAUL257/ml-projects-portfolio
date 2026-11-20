"""
feature_engineering.py
----------------------
Transforms the raw climate time series into a supervised-learning feature
matrix: lag features, rolling statistics, first differences and polynomial
interaction terms.

All features are computed strictly from *past* observations (shift >= 1 for
target-derived features) to avoid look-ahead leakage.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

TARGET = "sea_level_mm"
DRIVERS = ["co2_ppm", "temp_anomaly_c"]
LAGS = (1, 2, 3, 5)
ROLLING_WINDOWS = (3, 5, 10)


def add_lag_features(df: pd.DataFrame, cols: List[str], lags=LAGS) -> pd.DataFrame:
    """Add lagged copies of the given columns."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: List[str], windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """Add rolling mean / std computed on the *previous* window (shifted by 1)."""
    for col in cols:
        shifted = df[col].shift(1)
        for w in windows:
            df[f"{col}_roll_mean{w}"] = shifted.rolling(w).mean()
            df[f"{col}_roll_std{w}"] = shifted.rolling(w).std()
    return df


def add_polynomial_features(df: pd.DataFrame, cols: List[str], degree: int = 2) -> pd.DataFrame:
    """Add polynomial + interaction terms for the physical drivers."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    values = poly.fit_transform(df[cols])
    names = poly.get_feature_names_out(cols)
    for name, column in zip(names, values.T):
        if name in cols:  # skip degree-1 duplicates
            continue
        df[f"poly_{name.replace(' ', '_x_').replace('^', 'pow')}"] = column
    return df


def build_features(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Full feature pipeline.

    Returns:
        X: feature matrix (indexed by year)
        y: target series aligned with X
    """
    df = raw.copy().sort_values("year").reset_index(drop=True)

    # Target-derived features use only history (lag >= 1).
    df = add_lag_features(df, [TARGET] + DRIVERS)
    df = add_rolling_features(df, [TARGET, "temp_anomaly_c"])

    # Rate-of-change features.
    df["co2_diff1"] = df["co2_ppm"].diff()
    df["temp_diff1"] = df["temp_anomaly_c"].diff()
    df["sea_level_diff1_lag1"] = df[TARGET].diff().shift(1)

    # Cumulative warming is the main physical driver of sea level rise.
    df["cum_temp_anomaly"] = df["temp_anomaly_c"].cumsum()
    df["years_since_start"] = df["year"] - df["year"].min()

    df = add_polynomial_features(df, DRIVERS)

    df = df.dropna().reset_index(drop=True)
    y = df.pop(TARGET)
    X = df.set_index("year", drop=False)
    y.index = X.index
    logger.info("Built feature matrix: %d rows x %d features", *X.shape)
    return X, y


if __name__ == "__main__":
    from data_ingestion import generate_sea_level_data

    logging.basicConfig(level=logging.INFO)
    X, y = build_features(generate_sea_level_data())
    print(X.columns.tolist())

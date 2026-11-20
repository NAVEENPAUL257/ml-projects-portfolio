"""
evaluate.py
-----------
Evaluates a trained model on the chronologically held-out test set, computes
regression metrics and produces diagnostic plots (predicted vs. actual over
time, parity plot and residuals).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI / containers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_mape_pct": float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100),
    }


def plot_predictions(years, y_true, y_pred, out_path: Path, train_years=None, y_train=None) -> Path:
    """Three-panel diagnostic figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    if train_years is not None:
        ax.plot(train_years, y_train, color="grey", alpha=0.6, label="Train (actual)")
    ax.plot(years, y_true, "o-", color="steelblue", ms=3, label="Test (actual)")
    ax.plot(years, y_pred, "s--", color="darkorange", ms=3, label="Test (predicted)")
    ax.set(title="Sea Level: Predicted vs Actual", xlabel="Year", ylabel="GMSL (mm)")
    ax.legend()

    ax = axes[1]
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.scatter(y_true, y_pred, alpha=0.7, color="teal")
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.set(title="Parity Plot", xlabel="Actual (mm)", ylabel="Predicted (mm)")
    ax.legend()

    ax = axes[2]
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    ax.bar(years, residuals, color=np.where(residuals >= 0, "seagreen", "indianred"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Residuals (Actual - Predicted)", xlabel="Year", ylabel="mm")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved evaluation plot to %s", out_path)
    return out_path


def evaluate_model(model, features: List[str], X_test: pd.DataFrame, y_test: pd.Series,
                   out_dir: Path, X_train: pd.DataFrame | None = None,
                   y_train: pd.Series | None = None) -> Dict[str, object]:
    """Score the model on the test set and save plots + prediction CSV."""
    preds = model.predict(X_test[features])
    metrics = regression_metrics(y_test, preds)
    logger.info("Test metrics: %s", {k: round(v, 3) for k, v in metrics.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_predictions(
        X_test["year"].values, y_test.values, preds, out_dir / "predicted_vs_actual.png",
        train_years=None if X_train is None else X_train["year"].values,
        y_train=None if y_train is None else y_train.values,
    )
    pred_path = out_dir / "test_predictions.csv"
    pd.DataFrame({"year": X_test["year"].values, "actual": y_test.values,
                  "predicted": preds}).to_csv(pred_path, index=False)
    return {"metrics": metrics, "plot_path": plot_path, "predictions_path": pred_path, "predictions": preds}

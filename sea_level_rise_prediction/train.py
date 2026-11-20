"""
train.py
--------
Trains a RandomForest + LinearRegression voting ensemble for sea level
prediction with GridSearchCV hyperparameter tuning and 5-fold time-series
cross-validation. Parameters, CV metrics and the fitted model are logged to
`mlruns/` via the ExperimentTracker.

Why TimeSeriesSplit? Sea level is a trending time series; shuffled K-fold would
leak future information into training folds. TimeSeriesSplit(n_splits=5) keeps
the 5-fold protocol while respecting temporal order.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5

PARAM_GRID: Dict[str, list] = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None, 8],
    "rf__min_samples_leaf": [1, 3],
    "weights": [[1, 1], [1, 3]],  # relative weight of RF vs. linear model
}


def build_model() -> VotingRegressor:
    """RandomForest (non-linear interactions) + scaled LinearRegression (trend extrapolation)."""
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    return VotingRegressor(estimators=[("rf", rf), ("lr", lr)])


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Hold out the most recent `test_size` fraction of years for final evaluation."""
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tracker: ExperimentTracker | None = None,
    param_grid: Dict[str, list] | None = None,
) -> Tuple[VotingRegressor, Dict[str, Any]]:
    """Run grid search + CV, refit on the full training set and log everything.

    Returns:
        best_model: fitted VotingRegressor
        summary: dict with best params and CV metrics
    """
    features = [c for c in X_train.columns if c != "year"]
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    search = GridSearchCV(
        estimator=build_model(),
        param_grid=param_grid or PARAM_GRID,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    logger.info("Starting GridSearchCV over %d candidates x %d folds",
                len(ParameterGrid(search.param_grid)), N_SPLITS)
    search.fit(X_train[features], y_train)
    best = search.best_estimator_

    # Re-run CV with the best params to report a full metric panel.
    cv_res = cross_validate(
        best, X_train[features], y_train, cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error",
                 "mae": "neg_mean_absolute_error", "r2": "r2"},
    )
    cv_metrics = {
        "cv_rmse_mean": float(-cv_res["test_rmse"].mean()),
        "cv_rmse_std": float(cv_res["test_rmse"].std()),
        "cv_mae_mean": float(-cv_res["test_mae"].mean()),
        "cv_r2_mean": float(np.mean(cv_res["test_r2"])),
    }
    summary = {"best_params": search.best_params_, "features": features, **cv_metrics}
    logger.info("Best params: %s | CV RMSE: %.3f", search.best_params_, cv_metrics["cv_rmse_mean"])

    if tracker is not None:
        tracker.log_params({k: str(v) for k, v in search.best_params_.items()})
        tracker.log_params({"cv_strategy": f"TimeSeriesSplit({N_SPLITS})",
                            "n_train_rows": len(X_train), "n_features": len(features)})
        tracker.log_metrics(cv_metrics)
        model_path = tracker.artifact_dir / "model.joblib"
        joblib.dump({"model": best, "features": features}, model_path)
        tracker.log_dict(
            pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "rank_test_score"]]
            .assign(params=lambda d: d["params"].astype(str)).to_dict(orient="records"),
            "grid_search_results.json",
        )
        logger.info("Model artifact saved to %s", model_path)

    return best, summary


if __name__ == "__main__":
    from data_ingestion import generate_sea_level_data
    from feature_engineering import build_features

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    X, y = build_features(generate_sea_level_data())
    X_tr, _, y_tr, _ = chronological_split(X, y)
    with ExperimentTracker("sea_level_rise", run_name="standalone-train") as run:
        train_model(X_tr, y_tr, tracker=run)

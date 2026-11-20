"""
pipeline.py
-----------
End-to-end retraining pipeline orchestrator:

    ingestion -> feature engineering -> chronological split
              -> GridSearchCV training (5-fold TS-CV) -> held-out evaluation
              -> prediction / residual drift checks -> experiment logging

Usage:
    python pipeline.py                 # full grid search
    python pipeline.py --quick         # reduced grid (fast smoke test / CI)
    python pipeline.py --test-size 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from data_ingestion import ingest
from drift_detector import detect_prediction_drift, detect_residual_drift
from evaluate import evaluate_model
from experiment_tracker import ExperimentTracker, list_runs
from feature_engineering import build_features
from train import chronological_split, train_model

EXPERIMENT = "sea_level_rise"
QUICK_GRID = {"rf__n_estimators": [100], "rf__max_depth": [None, 8], "weights": [[1, 1]]}

logger = logging.getLogger("pipeline")


def run_pipeline(test_size: float = 0.2, quick: bool = False) -> dict:
    with ExperimentTracker(EXPERIMENT) as run:
        logger.info("=== Run %s started ===", run.run_id)

        # 1. Ingestion
        raw = ingest()
        run.log_params({"n_raw_rows": len(raw), "year_min": int(raw.year.min()),
                        "year_max": int(raw.year.max()), "test_size": test_size})

        # 2. Feature engineering
        X, y = build_features(raw)

        # 3. Split (chronological to mimic forecasting the future)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, test_size)

        # 4. Training + tuning
        model, summary = train_model(X_tr, y_tr, tracker=run,
                                     param_grid=QUICK_GRID if quick else None)
        features = summary["features"]

        # 5. Evaluation
        eval_res = evaluate_model(model, features, X_te, y_te, run.artifact_dir, X_tr, y_tr)
        run.log_metrics(eval_res["metrics"])

        # 6. Drift checks on prediction distributions across splits.
        #    Train is split into early/late halves to give an in-sample baseline.
        train_preds = model.predict(X_tr[features])
        half = len(train_preds) // 2
        drift = detect_prediction_drift({
            "train": train_preds,
            "train_late": train_preds[half:],
            "test": eval_res["predictions"],
        })
        drift["residuals"] = detect_residual_drift(
            y_tr.values - train_preds, y_te.values - eval_res["predictions"])
        run.log_dict(drift, "drift_report.json")
        run.log_metrics({
            "drift_test_psi": drift["test"]["psi"],
            "drift_test_ks_stat": drift["test"]["ks_statistic"],
            "retrain_recommended": float(drift["summary"]["retrain_recommended"]),
        })

        result = {
            "run_id": run.run_id,
            "artifact_dir": str(run.artifact_dir),
            "best_params": summary["best_params"],
            "cv_rmse": summary["cv_rmse_mean"],
            **eval_res["metrics"],
            "drift_summary": drift["summary"],
        }
        logger.info("=== Run %s finished ===", run.run_id)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sea level rise retraining pipeline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true", help="Use a reduced hyperparameter grid")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    np.random.seed(42)
    result = run_pipeline(args.test_size, args.quick)
    print("\n" + json.dumps(result, indent=2, default=str))

    runs = list_runs(EXPERIMENT)
    print(f"\n{len(runs)} run(s) recorded under mlruns/{EXPERIMENT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

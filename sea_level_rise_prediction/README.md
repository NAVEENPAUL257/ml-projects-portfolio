# 🌊 Sea Level Rise Prediction — End-to-End ML Retraining Pipeline

A production-style, reproducible retraining pipeline that forecasts global mean sea level (GMSL) from climate drivers (CO₂ concentration and temperature anomaly). It covers ingestion, feature engineering, hyperparameter tuning with cross-validation, held-out evaluation, MLflow-style experiment tracking and model-drift detection.

## Architecture

```
data_ingestion ──► feature_engineering ──► chronological split
                                               │
                     ┌─────────────────────────┴───────────────┐
                     ▼                                         ▼
     train (GridSearchCV + 5-fold TimeSeriesSplit)      held-out test set
                     │                                         │
                     └──────────► evaluate ◄───────────────────┘
                                     │
                              drift_detector (KS + PSI)
                                     │
                         mlruns/<experiment>/<run_id>/
```

## Modules

| File | Responsibility |
|------|----------------|
| `data_ingestion.py` | Generates a synthetic, physically plausible 1880–2023 annual series (`year`, `co2_ppm`, `temp_anomaly_c`, `sea_level_mm`), runs schema/quality checks and writes it to `data/` |
| `feature_engineering.py` | Lag features (1, 2, 3, 5 yrs), rolling mean/std (3, 5, 10 yrs), first differences, cumulative warming, degree-2 polynomial terms. All features use past values only, so nothing leaks from the future |
| `train.py` | `VotingRegressor` that combines **RandomForest** and **scaled LinearRegression**, tuned with `GridSearchCV` over 5-fold `TimeSeriesSplit`. Logs parameters, CV metrics, grid results and the model artifact |
| `evaluate.py` | RMSE / MAE / R² / MAPE on the chronological hold-out set, plus a 3-panel plot (predicted vs. actual over time, parity plot, residuals) |
| `drift_detector.py` | Compares prediction and residual distributions across splits with the **Kolmogorov–Smirnov test** and the **Population Stability Index**, then flags whether a retrain is recommended |
| `experiment_tracker.py` | Lightweight tracker that writes runs in an MLflow-like layout (`meta.json`, `params.json`, `metrics.json`, `artifacts/`). If `mlflow` is installed, it can also mirror runs to MLflow |
| `pipeline.py` | Runs every step above in order through a CLI |

## Quick start

```bash
cd sea_level_rise_prediction
pip install -r requirements.txt
python pipeline.py            # full hyperparameter grid (16 candidates x 5 folds)
python pipeline.py --quick    # reduced grid for CI / smoke tests
```

Each run creates:

```
mlruns/sea_level_rise/<run_id>/
├── meta.json  params.json  metrics.json
└── artifacts/
    ├── model.joblib
    ├── grid_search_results.json
    ├── predicted_vs_actual.png
    ├── test_predictions.csv
    └── drift_report.json
```

## Design notes

- **Why TimeSeriesSplit?** Sea level trends upward over time. Shuffled K-fold would let the model train on future years and test on past ones, which inflates the scores. TimeSeriesSplit keeps the 5-fold protocol and respects time order.
- **Why an RF + linear ensemble?** RandomForest captures non-linear interactions but cannot extrapolate beyond the target range it saw in training. The linear model carries the trend, and the ensemble weight between the two is tuned in the grid search.
- **Interpreting drift:** a trending target is *expected* to show prediction-distribution drift between the past (train) and the future (test). In production, the same checks run between consecutive scoring windows, and a PSI above 0.25 triggers retraining.

## Tech stack
Python · Pandas · NumPy · scikit-learn · SciPy · Matplotlib · MLflow-style tracking

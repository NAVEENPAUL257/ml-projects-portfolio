# 💳 Credit Card Segmentation: Clustering Inference and Drift Monitoring

This project segments credit card customers by spending behaviour using KMeans. It includes scoring code that is ready for a REST API (FastAPI) and monitors input data drift.

## Modules
| File | Responsibility |
|------|----------------|
| `data_generator.py` | Generates synthetic customers (balance, purchases, cash advance, credit limit, payments, tenure, …) from 4 latent personas, with injected missing values. A `shift` parameter creates drifted cohorts |
| `preprocessing.py` | Fills missing values with the median, applies log1p to heavy-tailed monetary fields, standardizes the data, and validates the input schema |
| `clustering.py` | Chooses k with the elbow method and silhouette score (k = 2..10). Fits KMeans with k=4 and ranks clusters by spend into **Low Spender / Moderate Spender / High Spender / Premium**. Saves the model bundle and segment profiles |
| `inference.py` | `score()` returns the segment, the cluster ID and soft cluster probabilities (a softmax over centroid distances) |
| `drift_monitor.py` | Runs a two-sample KS test on each feature, with an effect-size guard, and reports any shift in the segment mix |
| `api.py` | FastAPI app with `GET /health`, `POST /predict` and `POST /drift` |
| `main.py` | End-to-end demo: train, score, then compare a stable cohort with a drifted one |

## Run
```bash
pip install -r requirements.txt
python main.py                              # train + demo (artifacts in models/, outputs/)
uvicorn api:app --host 0.0.0.0 --port 8000  # REST API (auto-trains if no model exists)
```

Example request:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"customers":[{
 "balance":3200,"purchases":4100,"oneoff_purchases":2000,"installments_purchases":2100,
 "cash_advance":450,"purchases_frequency":0.8,"credit_limit":9000,"payments":4300,
 "minimum_payments":320,"prc_full_payment":0.35,"tenure":12}]}'
```

## Tech stack
Python · Pandas · NumPy · scikit-learn · SciPy · FastAPI · Matplotlib

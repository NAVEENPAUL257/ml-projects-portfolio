"""
clustering.py
-------------
Trains the KMeans segmentation model:
  1. Elbow method (inertia) + silhouette score across k = 2..10
  2. Final KMeans at k = 4 (business-required tiers; validated by the metrics above)
  3. Clusters ranked by mean purchases -> Low / Moderate / High Spender / Premium
Saves the fitted bundle (preprocessor + model + label map + reference data) to models/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_generator import FEATURES, generate_customers
from preprocessing import build_preprocessor, validate_input

logger = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "segmentation_bundle.joblib"
SEGMENT_NAMES = ["Low Spender", "Moderate Spender", "High Spender", "Premium"]


def k_selection(X, k_range=range(2, 11), out_dir: Path = BASE / "outputs") -> pd.DataFrame:
    """Compute inertia and silhouette for each k and save the elbow/silhouette chart."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=42)})
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(res.k, res.inertia, "o-", color="steelblue"); ax1.set(xlabel="k", ylabel="Inertia (elbow)")
    ax2 = ax1.twinx(); ax2.plot(res.k, res.silhouette, "s--", color="darkorange"); ax2.set_ylabel("Silhouette")
    ax1.set_title("Elbow Method & Silhouette Score"); fig.tight_layout()
    fig.savefig(out_dir / "elbow_silhouette.png", dpi=120); plt.close(fig)
    return res


def train_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    X_raw = validate_input(df)
    pre = build_preprocessor().fit(X_raw)
    X = pre.transform(X_raw)
    selection = k_selection(X)
    logger.info("k-selection:\n%s", selection.round(3).to_string(index=False))

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(X)
    # Rank clusters by average purchases so labels are deterministic & business-meaningful
    order = X_raw.assign(c=km.labels_).groupby("c")["purchases"].mean().sort_values().index
    label_map = {int(c): SEGMENT_NAMES[i] for i, c in enumerate(order)}
    profile = X_raw.assign(segment=[label_map[c] for c in km.labels_]).groupby("segment").mean().round(2)
    sil = float(silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=42))

    bundle = {"preprocessor": pre, "model": km, "label_map": label_map,
              "features": FEATURES, "reference_data": X_raw, "silhouette": sil}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    (BASE / "outputs").mkdir(exist_ok=True)
    profile.to_csv(BASE / "outputs" / "segment_profiles.csv")
    (BASE / "outputs" / "training_report.json").write_text(json.dumps(
        {"silhouette": sil, "label_map": label_map,
         "k_selection": selection.to_dict(orient="records")}, indent=2))
    logger.info("Silhouette (k=%d): %.3f | model saved to %s", n_clusters, sil, MODEL_PATH)
    logger.info("Segment profiles:\n%s", profile[["balance", "purchases", "credit_limit"]])
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train_segmentation(generate_customers(5000))

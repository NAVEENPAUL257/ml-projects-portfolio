"""
drift_monitor.py
----------------
Monitors input feature distributions of a new cohort against the training
reference using a per-feature two-sample Kolmogorov-Smirnov test. Also reports
segment-mix shift so business owners see the downstream impact.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
from scipy.stats import ks_2samp

from data_generator import FEATURES

logger = logging.getLogger(__name__)


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                         alpha: float = 0.01, min_effect: float = 0.1) -> Dict:
    """Flag a feature as drifted when p < alpha AND KS statistic >= min_effect
    (the effect-size guard avoids false alarms on large samples)."""
    rows = []
    for f in FEATURES:
        ref, cur = reference[f].dropna(), current[f].dropna()
        ks = ks_2samp(ref, cur)
        rows.append({"feature": f, "ks_stat": round(float(ks.statistic), 4),
                     "p_value": float(ks.pvalue),
                     "ref_mean": round(float(ref.mean()), 2), "cur_mean": round(float(cur.mean()), 2),
                     "drifted": bool(ks.pvalue < alpha and ks.statistic >= min_effect)})
    table = pd.DataFrame(rows)
    drifted = table.loc[table.drifted, "feature"].tolist()
    logger.info("Drift check: %d/%d features drifted %s", len(drifted), len(FEATURES), drifted)
    return {"features": table.to_dict(orient="records"), "drifted_features": drifted,
            "drift_share": len(drifted) / len(FEATURES), "alert": len(drifted) / len(FEATURES) >= 0.3}


def segment_mix(segments: pd.Series) -> Dict[str, float]:
    return segments.value_counts(normalize=True).round(4).to_dict()

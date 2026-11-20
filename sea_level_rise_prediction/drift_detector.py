"""
drift_detector.py
-----------------
Model drift checks that compare *prediction* distributions (and optionally
residual distributions) across data splits using:

* Kolmogorov-Smirnov two-sample test  - non-parametric test of distribution equality
* Population Stability Index (PSI)      - industry-standard drift magnitude score

PSI rule of thumb:
    < 0.10  -> no significant shift
    0.10-0.25 -> moderate shift, monitor
    > 0.25  -> significant shift, retrain recommended
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
KS_ALPHA = 0.05


@dataclass
class DriftResult:
    reference: str
    current: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    psi_level: str
    drift_detected: bool


def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               n_bins: int = 10, eps: float = 1e-6) -> float:
    """Compute PSI using quantile bins derived from the reference distribution."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf  # capture out-of-range values
    exp_pct = np.histogram(expected, edges)[0] / len(expected) + eps
    act_pct = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_level(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def compare_distributions(reference: np.ndarray, current: np.ndarray,
                          ref_name: str = "reference", cur_name: str = "current") -> DriftResult:
    ks = ks_2samp(reference, current)
    psi = population_stability_index(reference, current)
    level = _psi_level(psi)
    result = DriftResult(
        reference=ref_name, current=cur_name,
        ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        psi=psi, psi_level=level,
        drift_detected=bool(ks.pvalue < KS_ALPHA or psi >= PSI_SIGNIFICANT),
    )
    logger.info("Drift %s vs %s | KS=%.3f (p=%.4f) | PSI=%.3f [%s] | drift=%s",
                ref_name, cur_name, result.ks_statistic, result.ks_pvalue,
                psi, level, result.drift_detected)
    return result


def detect_prediction_drift(split_predictions: Dict[str, np.ndarray],
                            reference_split: str = "train") -> Dict[str, dict]:
    """Compare each split's predictions against the reference split.

    Args:
        split_predictions: mapping of split name -> model predictions.
        reference_split: split used as the baseline distribution.

    Returns:
        Mapping of split name -> DriftResult as dict, plus an overall verdict.
    """
    if reference_split not in split_predictions:
        raise KeyError(f"Reference split '{reference_split}' not provided")
    ref = split_predictions[reference_split]
    report = {
        name: asdict(compare_distributions(ref, preds, reference_split, name))
        for name, preds in split_predictions.items() if name != reference_split
    }
    report["summary"] = {
        "reference": reference_split,
        "any_drift": any(r["drift_detected"] for r in report.values()),
        "retrain_recommended": any(r["psi_level"] == "significant" for r in report.values()),
    }
    return report


def detect_residual_drift(train_residuals: np.ndarray, test_residuals: np.ndarray) -> dict:
    """Concept-drift proxy: has the error distribution changed between splits?"""
    return asdict(compare_distributions(train_residuals, test_residuals,
                                        "train_residuals", "test_residuals"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    print(detect_prediction_drift({"train": rng.normal(0, 1, 500),
                                   "same": rng.normal(0, 1, 500),
                                   "shifted": rng.normal(1, 1, 500)}))

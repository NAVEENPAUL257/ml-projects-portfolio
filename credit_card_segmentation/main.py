"""
main.py - end-to-end demo: generate -> train -> score -> drift-monitor two cohorts.
"""

import json
import logging

import pandas as pd

from clustering import train_segmentation
from data_generator import generate_customers
from drift_monitor import feature_drift_report, segment_mix
from inference import score


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    bundle = train_segmentation(generate_customers(5000, seed=42))

    print("\nSample scoring:")
    sample = generate_customers(4, seed=7)
    print(json.dumps(score(sample, bundle), indent=2))

    for name, cohort in {"stable_cohort": generate_customers(2000, seed=1),
                         "drifted_cohort": generate_customers(2000, seed=2, shift=0.6)}.items():
        rep = feature_drift_report(bundle["reference_data"], cohort)
        mix = segment_mix(pd.Series([r["segment"] for r in score(cohort, bundle)]))
        print(f"\n[{name}] drifted={rep['drifted_features']} alert={rep['alert']} segment_mix={mix}")


if __name__ == "__main__":
    main()

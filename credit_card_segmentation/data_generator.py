"""
data_generator.py
-----------------
Generates synthetic credit card customer behaviour data (CC GENERAL-style schema)
drawn from four latent spender personas, with ~2% missing values injected to
exercise the preprocessing layer. A `shift` factor simulates a drifted cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["balance", "purchases", "oneoff_purchases", "installments_purchases",
            "cash_advance", "purchases_frequency", "credit_limit", "payments",
            "minimum_payments", "prc_full_payment", "tenure"]

# persona: (share, balance, purchases, cash_advance, credit_limit, purchase_freq, prc_full)
PERSONAS = {
    "low":      (0.35, 800,   300,   400, 2500,  0.15, 0.05),
    "moderate": (0.30, 1800,  1200,  700, 5000,  0.45, 0.15),
    "high":     (0.25, 3000,  4000,  500, 9000,  0.80, 0.35),
    "premium":  (0.10, 5000, 12000,  300, 20000, 0.95, 0.70),
}


def generate_customers(n: int = 5000, seed: int = 42, shift: float = 0.0,
                       missing_rate: float = 0.02) -> pd.DataFrame:
    """Create `n` synthetic customers. `shift` (e.g. 0.3) inflates spend to simulate drift."""
    rng = np.random.default_rng(seed)
    frames = []
    for name, (share, bal, pur, cash, lim, freq, full) in PERSONAS.items():
        k = int(round(n * share))
        purchases = rng.lognormal(np.log(pur * (1 + shift)), 0.4, k)
        oneoff_share = rng.beta(2, 2, k)
        frames.append(pd.DataFrame({
            "balance": rng.lognormal(np.log(bal), 0.5, k),
            "purchases": purchases,
            "oneoff_purchases": purchases * oneoff_share,
            "installments_purchases": purchases * (1 - oneoff_share),
            "cash_advance": rng.lognormal(np.log(cash), 0.8, k),
            "purchases_frequency": np.clip(rng.normal(freq, 0.1, k), 0, 1),
            "credit_limit": rng.lognormal(np.log(lim), 0.3, k).round(-2),
            "payments": purchases * rng.uniform(0.6, 1.2, k) + rng.lognormal(5, 1, k),
            "minimum_payments": rng.lognormal(np.log(bal * 0.1), 0.5, k),
            "prc_full_payment": np.clip(rng.normal(full, 0.1, k), 0, 1),
            "tenure": rng.choice([6, 8, 10, 12], k, p=[0.05, 0.1, 0.15, 0.7]),
        }))
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "cust_id", [f"C{10000 + i}" for i in range(len(df))])
    # Inject missing values in fields that are commonly sparse in real data
    for col in ("minimum_payments", "credit_limit"):
        df.loc[rng.random(len(df)) < missing_rate, col] = np.nan
    return df.round(4)


if __name__ == "__main__":
    print(generate_customers(10).head())

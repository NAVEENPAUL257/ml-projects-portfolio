"""
data_ingestion.py
-----------------
Generates a synthetic, physically-plausible annual climate time series used to
model global mean sea level (GMSL) rise.

Columns
    year              : calendar year
    co2_ppm           : atmospheric CO2 concentration (ppm)
    temp_anomaly_c    : global surface temperature anomaly vs. 1951-1980 baseline (deg C)
    sea_level_mm      : GMSL relative to the 1900 baseline (mm)

The generator mimics the accelerating Keeling curve, a temperature response
roughly logarithmic in CO2, and sea level driven by cumulative warming
(thermal expansion + ice melt) plus noise and a mild multi-decadal oscillation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


def generate_sea_level_data(
    start_year: int = 1880,
    end_year: int = 2023,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic annual sea level dataset.

    Args:
        start_year: First year in the series (inclusive).
        end_year: Last year in the series (inclusive).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [year, co2_ppm, temp_anomaly_c, sea_level_mm].
    """
    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year")

    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    t = years - start_year

    # CO2: pre-industrial ~285 ppm with exponential growth toward ~420 ppm today.
    co2 = 285 + 0.9 * np.exp(0.0333 * t) + rng.normal(0, 0.4, len(t))

    # Temperature: climate sensitivity ~3C per doubling + ENSO-like noise.
    sensitivity = 3.0 / np.log(2)
    temp = (
        sensitivity * np.log(co2 / 285) * 0.55
        - 0.3
        + 0.08 * np.sin(2 * np.pi * t / 60)  # multi-decadal oscillation
        + rng.normal(0, 0.09, len(t))
    )

    # Sea level: integral of warming (committed rise) + linear background trend.
    warming = np.clip(temp + 0.3, 0, None)
    cumulative = np.cumsum(warming)
    sea_level = 0.6 * t + 2.2 * cumulative + rng.normal(0, 4.0, len(t))
    sea_level = sea_level - sea_level[min(20, len(t) - 1)]  # ~1900 baseline

    df = pd.DataFrame(
        {
            "year": years,
            "co2_ppm": co2.round(2),
            "temp_anomaly_c": temp.round(3),
            "sea_level_mm": sea_level.round(2),
        }
    )
    logger.info("Generated %d rows of synthetic sea level data", len(df))
    return df


def ingest(output_path: Path | None = None, **kwargs) -> pd.DataFrame:
    """Generate data, validate the schema and persist it to CSV."""
    df = generate_sea_level_data(**kwargs)

    # Basic data-quality gates (fail fast in a CI/retraining context).
    expected = {"year", "co2_ppm", "temp_anomaly_c", "sea_level_mm"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.isnull().any().any():
        raise ValueError("Ingested data contains nulls")
    if not df["year"].is_monotonic_increasing:
        raise ValueError("Years must be sorted ascending")

    output_path = output_path or DATA_DIR / "sea_level_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Raw data written to %s", output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print(ingest().tail())

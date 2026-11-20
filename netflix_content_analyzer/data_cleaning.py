"""
data_cleaning.py
----------------
Deterministic cleaning / transformation stage:
  * normalise categorical text (strip, title-case `type`)
  * remove exact + show_id duplicates
  * impute missing country/rating, drop rows without date_added
  * parse dates, split duration into minutes / seasons
  * explode multi-genre strings into a primary genre + list
Returns the clean frame and a data-quality report (useful as a CI gate).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    report = {"rows_in": len(df), "nulls_in": df.isnull().sum().to_dict()}
    df = df.copy()

    for col in ("type", "title", "genre", "country", "rating", "duration", "description"):
        df[col] = df[col].astype("string").str.strip()
    df["type"] = df["type"].str.title().replace({"Tv Show": "TV Show"})

    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="show_id", keep="first")
    report["duplicates_removed"] = before - len(df)

    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("NR")
    df["description"] = df["description"].fillna("")
    df["date_added"] = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
    report["rows_dropped_no_date"] = int(df["date_added"].isna().sum())
    df = df.dropna(subset=["date_added"])

    df["year_added"] = df["date_added"].dt.year.astype(int)
    df["release_year"] = df["release_year"].astype(int)
    num = df["duration"].str.extract(r"(\d+)")[0].astype(float)
    df["duration_min"] = num.where(df["type"] == "Movie")
    df["seasons"] = num.where(df["type"] == "TV Show")
    df["genres"] = df["genre"].str.split(r",\s*")
    df["primary_genre"] = df["genres"].str[0]

    assert set(df["type"].unique()) <= {"Movie", "TV Show"}, "Unexpected content type"
    report.update(rows_out=len(df), nulls_out=int(df[["country", "rating", "date_added"]].isnull().sum().sum()))
    logger.info("Cleaning: %d -> %d rows (%d duplicates removed)", report["rows_in"], len(df),
                report["duplicates_removed"])
    return df.reset_index(drop=True), report

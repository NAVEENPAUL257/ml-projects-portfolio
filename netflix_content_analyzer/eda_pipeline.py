"""
eda_pipeline.py
---------------
Automated EDA: summary statistics, distributions and genre / country / year
trend tables. Returns a dict of DataFrames consumed by visualizations.py and
persisted to outputs/ as CSV + a JSON summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame, out_dir: Path) -> Dict[str, pd.DataFrame]:
    genres = df.explode("genres")
    tables = {
        "numeric_summary": df[["release_year", "year_added", "duration_min", "seasons"]].describe().T,
        "type_counts": df["type"].value_counts().rename_axis("type").to_frame("count"),
        "rating_counts": df["rating"].value_counts().rename_axis("rating").to_frame("count"),
        "top_countries": df["country"].value_counts().head(10).rename_axis("country").to_frame("count"),
        "top_genres": genres["genres"].value_counts().head(10).rename_axis("genre").to_frame("count"),
        "content_by_year": df.groupby(["release_year", "type"]).size().unstack(fill_value=0),
        "genre_by_year": (genres[genres["genres"].isin(genres["genres"].value_counts().head(6).index)]
                          .groupby(["year_added", "genres"]).size().unstack(fill_value=0)),
        "country_type": pd.crosstab(df["country"], df["type"]).sort_values("Movie", ascending=False).head(10),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")

    summary = {
        "total_titles": int(len(df)),
        "movies": int((df["type"] == "Movie").sum()),
        "tv_shows": int((df["type"] == "TV Show").sum()),
        "countries": int(df["country"].nunique()),
        "distinct_genres": int(genres["genres"].nunique()),
        "release_year_range": [int(df.release_year.min()), int(df.release_year.max())],
        "median_movie_minutes": float(df["duration_min"].median()),
        "most_common_rating": df["rating"].mode()[0],
        "top_genre": tables["top_genres"].index[0],
        "top_country": tables["top_countries"].index[0],
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA summary: %s", summary)
    tables["summary"] = summary
    return tables

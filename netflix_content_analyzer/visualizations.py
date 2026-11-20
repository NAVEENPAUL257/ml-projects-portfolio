"""
visualizations.py
-----------------
Generates the dashboard figures from EDA tables and saves them to outputs/:
  1. genre_trends.png         2. country_distribution.png   3. content_by_year.png
  4. rating_breakdown.png     5. movie_vs_tv.png            6. duration_analysis.png
  7. dashboard.png (2x3 overview combining the panels)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="rocket")
NETFLIX_RED = "#E50914"


def genre_trends(ax, t, df):
    t["genre_by_year"].plot(ax=ax, marker="o", ms=3)
    ax.set(title="Top Genre Trends by Year Added", xlabel="Year added", ylabel="Titles")
    ax.legend(fontsize=7)


def country_distribution(ax, t, df):
    d = t["top_countries"].reset_index()
    sns.barplot(data=d, y="country", x="count", ax=ax, color=NETFLIX_RED)
    ax.set(title="Top 10 Producing Countries", xlabel="Titles", ylabel="")


def content_by_year(ax, t, df):
    t["content_by_year"].plot(kind="area", stacked=True, ax=ax, alpha=0.8, color=[NETFLIX_RED, "#221F1F"])
    ax.set(title="Content Released per Year", xlabel="Release year", ylabel="Titles")


def rating_breakdown(ax, t, df):
    d = t["rating_counts"].reset_index()
    sns.barplot(data=d, x="rating", y="count", ax=ax, palette="rocket", hue="rating", legend=False)
    ax.set(title="Maturity Rating Breakdown", xlabel="", ylabel="Titles")
    ax.tick_params(axis="x", rotation=45)


def movie_vs_tv(ax, t, df):
    c = t["type_counts"]["count"]
    ax.pie(c, labels=c.index, autopct="%1.1f%%", colors=[NETFLIX_RED, "#221F1F"],
           startangle=90, wedgeprops={"width": 0.45})
    ax.set_title("Movie vs TV Show Ratio")


def duration_analysis(ax, t, df):
    sns.histplot(df["duration_min"].dropna(), bins=30, kde=True, ax=ax, color=NETFLIX_RED)
    ax.axvline(df["duration_min"].median(), color="black", ls="--", label="Median")
    ax.set(title="Movie Duration Distribution", xlabel="Minutes", ylabel="Movies")
    ax.legend()


PLOTS: Dict[str, Callable] = {
    "genre_trends": genre_trends, "country_distribution": country_distribution,
    "content_by_year": content_by_year, "rating_breakdown": rating_breakdown,
    "movie_vs_tv": movie_vs_tv, "duration_analysis": duration_analysis,
}


def generate_all(tables: Dict, df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in PLOTS.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fn(ax, tables, df)
        fig.tight_layout(); p = out_dir / f"{name}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        paths.append(p)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for ax, fn in zip(axes.flat, PLOTS.values()):
        fn(ax, tables, df)
    fig.suptitle("Netflix Content Analyzer - Dashboard", fontsize=18, color=NETFLIX_RED, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); p = out_dir / "dashboard.png"; fig.savefig(p, dpi=100); plt.close(fig)
    paths.append(p)
    logger.info("Saved %d figures to %s", len(paths), out_dir)
    return paths

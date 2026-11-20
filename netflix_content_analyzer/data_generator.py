"""
data_generator.py
-----------------
Generates a synthetic Netflix-style catalogue with realistic messiness
(missing values, duplicate rows, inconsistent casing / whitespace, mixed
duration formats) so the cleaning stage has real work to do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Dramas", "Comedies", "Documentaries", "Action & Adventure", "International TV Shows",
          "Thrillers", "Kids' TV", "Romantic Movies", "Horror Movies", "Sci-Fi & Fantasy", "Crime TV Shows"]
COUNTRIES = ["United States", "India", "United Kingdom", "Japan", "South Korea", "Canada",
             "Spain", "France", "Mexico", "Brazil", "Germany", "Nigeria"]
COUNTRY_W = [0.32, 0.13, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
RATINGS = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "PG", "TV-Y", "TV-Y7", "G", "NR"]
RATING_W = [0.36, 0.25, 0.10, 0.09, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02]
WORDS = ["Shadow", "Love", "City", "Secret", "Last", "Dark", "Kingdom", "Dream", "Road", "Night",
         "Fire", "Ocean", "Story", "Legend", "Hidden", "Wild", "Broken", "Golden", "Lost", "Echo"]


def generate_netflix_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(1990, 2024)
    year_w = np.exp((years - 1990) / 6.0); year_w /= year_w.sum()  # catalogue skews recent
    release_year = rng.choice(years, n, p=year_w)
    is_tv = rng.random(n) < np.clip(0.15 + (release_year - 1990) * 0.012, 0, 0.45)

    duration = np.where(is_tv, [f"{s} Season{'s' if s > 1 else ''}" for s in rng.geometric(0.55, n)],
                        [f"{m} min" for m in rng.normal(100, 22, n).clip(45, 210).astype(int)])
    genre = [", ".join(rng.choice(GENRES, rng.integers(1, 4), replace=False)) for _ in range(n)]
    added = pd.to_datetime([f"{min(max(y, 2015) + rng.integers(0, 3), 2023)}-{rng.integers(1, 13):02d}-01"
                            for y in release_year])
    df = pd.DataFrame({
        "show_id": [f"s{i + 1}" for i in range(n)],
        "type": np.where(is_tv, "TV Show", "Movie"),
        "title": [" ".join(rng.choice(WORDS, rng.integers(1, 4), replace=False)) + f" {i}" for i in range(n)],
        "genre": genre,
        "country": rng.choice(COUNTRIES, n, p=COUNTRY_W),
        "rating": rng.choice(RATINGS, n, p=RATING_W),
        "release_year": release_year,
        "date_added": added.strftime("%B %d, %Y"),
        "duration": duration,
        "description": [f"A gripping {g.split(',')[0].lower()} story set in {c}." for g, c in
                        zip(genre, rng.choice(COUNTRIES, n))],
    })
    # ---- inject data-quality issues --------------------------------------
    for col, rate in {"country": 0.08, "rating": 0.01, "date_added": 0.01, "description": 0.005}.items():
        df.loc[rng.random(n) < rate, col] = np.nan
    messy = rng.random(n) < 0.05
    df.loc[messy, "type"] = df.loc[messy, "type"].str.upper().str.pad(10, side="both")
    df = pd.concat([df, df.sample(frac=0.02, random_state=seed)], ignore_index=True)  # duplicates
    return df


def ingest(path: Path) -> pd.DataFrame:
    df = generate_netflix_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

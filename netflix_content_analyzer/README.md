# 🎬 Netflix Content Analyzer: Automated EDA Pipeline

This is an automated data pipeline aligned with CI/CD practices. It ingests a Netflix-style content catalogue, cleans and transforms it, runs a full exploratory data analysis (EDA) and produces dashboards of content trends by genre, country and year. The pipeline is containerized with Docker.

## Modules
| File | Responsibility |
|------|----------------|
| `data_generator.py` | Builds a synthetic catalogue (title, type, genre, country, rating, release year, date added, duration, description) with realistic data quality problems: missing values, duplicates, inconsistent casing and mixed duration formats |
| `data_cleaning.py` | Normalizes text, removes duplicates, fills missing values, parses dates, splits duration into minutes or seasons, and explodes genres. Returns a data quality report |
| `eda_pipeline.py` | Summary statistics, type/rating/country/genre distributions, and trend tables by year, genre and country. Saves CSV files and `eda_summary.json` |
| `visualizations.py` | Six plots (genre trends, country distribution, content by year, rating breakdown, movie vs. TV ratio, duration analysis) plus a combined `dashboard.png` |
| `pipeline.py` | Runs ingestion → cleaning → EDA → visualization. Returns a non-zero exit code if a data quality check fails, so it can stop a CI build |
| `Dockerfile` | Slim Python 3.11 image that runs as a non-root user |

## Run
```bash
pip install -r requirements.txt
python pipeline.py               # outputs/ gets plots, CSVs and JSON reports

docker build -t netflix-analyzer .
docker run --rm -v "$(pwd)/outputs:/app/outputs" netflix-analyzer
```

## Tech stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Docker

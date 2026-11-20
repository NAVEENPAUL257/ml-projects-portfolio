"""
pipeline.py
-----------
One-shot, CI/CD-friendly pipeline: ingestion -> cleaning -> EDA -> visualisation.
Exits non-zero if data-quality gates fail, so it can block a CI job.

Usage:  python pipeline.py [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from data_cleaning import clean
from data_generator import ingest
from eda_pipeline import run_eda
from visualizations import generate_all

BASE = Path(__file__).resolve().parent
logger = logging.getLogger("pipeline")


def run(output_dir: Path) -> dict:
    raw = ingest(BASE / "data" / "netflix_raw.csv")
    logger.info("Ingested %d raw rows", len(raw))

    df, dq = clean(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE / "data" / "netflix_clean.csv", index=False)
    (output_dir / "data_quality_report.json").write_text(json.dumps(dq, indent=2, default=str))
    if dq["nulls_out"] or df["show_id"].duplicated().any():
        raise RuntimeError("Data-quality gate failed")

    tables = run_eda(df, output_dir)
    plots = generate_all(tables, df, output_dir)
    return {"summary": tables["summary"], "plots": [p.name for p in plots], "data_quality": {
        k: dq[k] for k in ("rows_in", "rows_out", "duplicates_removed", "rows_dropped_no_date")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Netflix content EDA pipeline")
    ap.add_argument("--output-dir", type=Path, default=BASE / "outputs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    try:
        print(json.dumps(run(args.output_dir), indent=2))
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

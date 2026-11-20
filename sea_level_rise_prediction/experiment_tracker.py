"""
experiment_tracker.py
---------------------
A lightweight, dependency-free, MLflow-compatible-in-spirit experiment tracker.

Directory layout produced (mirrors MLflow's file store):

    mlruns/
      <experiment_name>/
        <run_id>/
          meta.json          # run name, status, start/end time
          params.json        # hyperparameters
          metrics.json       # final metrics
          artifacts/         # models, plots, reports

If the real `mlflow` package is installed, metrics/params are mirrored to it as
well, so the project can be switched to a real tracking server with no code
changes.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional real-MLflow mirroring
    import mlflow  # type: ignore

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False

MLRUNS_DIR = Path(__file__).resolve().parent / "mlruns"


class ExperimentTracker:
    """Context-managed experiment run with param/metric/artifact logging."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None,
                 root: Path = MLRUNS_DIR, use_mlflow: bool = False):
        self.run_id = uuid.uuid4().hex[:12]
        self.run_name = run_name or f"run-{datetime.now():%Y%m%d-%H%M%S}"
        self.run_dir = Path(root) / experiment_name / self.run_id
        self.artifact_dir = self.run_dir / "artifacts"
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self._start = None
        self._use_mlflow = use_mlflow and _HAS_MLFLOW
        self.experiment_name = experiment_name

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> "ExperimentTracker":
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._start = time.time()
        if self._use_mlflow:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)
        self._write_meta("RUNNING")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self._flush()
        self._write_meta(status)
        if self._use_mlflow:
            mlflow.end_run(status=status)

    # ---- logging API -----------------------------------------------------
    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value
        if self._use_mlflow:
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)
        if self._use_mlflow:
            mlflow.log_metric(key, float(value))

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, path: Path | str) -> Path:
        """Copy a file into the run's artifact directory."""
        src = Path(path)
        dst = self.artifact_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        if self._use_mlflow:
            mlflow.log_artifact(str(src))
        return dst

    def log_dict(self, data: Dict[str, Any], filename: str) -> Path:
        dst = self.artifact_dir / filename
        dst.write_text(json.dumps(data, indent=2, default=str))
        return dst

    # ---- internals -------------------------------------------------------
    def _flush(self) -> None:
        (self.run_dir / "params.json").write_text(json.dumps(self.params, indent=2, default=str))
        (self.run_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2))

    def _write_meta(self, status: str) -> None:
        meta = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment": self.experiment_name,
            "status": status,
            "start_time": datetime.fromtimestamp(self._start, tz=timezone.utc).isoformat(),
            "duration_sec": round(time.time() - self._start, 3),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def list_runs(experiment_name: str, root: Path = MLRUNS_DIR):
    """Return all runs for an experiment as a list of dicts (for comparison)."""
    exp_dir = Path(root) / experiment_name
    runs = []
    for run_dir in sorted(exp_dir.glob("*")) if exp_dir.exists() else []:
        try:
            runs.append({
                **json.loads((run_dir / "meta.json").read_text()),
                "metrics": json.loads((run_dir / "metrics.json").read_text()),
            })
        except FileNotFoundError:
            continue
    return runs

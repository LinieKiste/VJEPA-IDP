"""Shared MLflow setup — pins the tracking store to an ABSOLUTE path.

Why this exists: MLflow's default sqlite store is resolved relative to the CWD, so
running `python pour_probe/foo.py` from the repo root and running `python foo.py` from
inside `pour_probe/` silently write to two DIFFERENT databases. That happened: 44 runs
landed in `pouring/pour_probe/mlflow.db` while every earlier experiment lived in the
repo-root `mlflow.db`, so they were invisible in the UI.

Always go through `setup()` instead of calling `mlflow.set_experiment()` directly.
Override with $MLFLOW_TRACKING_URI if you really want a different store.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def setup(experiment: str):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)
    return mlflow.get_tracking_uri()

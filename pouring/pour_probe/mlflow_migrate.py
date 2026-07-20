"""One-off: copy runs from a stray MLflow sqlite store into the canonical repo-root one.

Needed because MLflow resolves its default sqlite store RELATIVE TO THE CWD, so running
these scripts from inside `pour_probe/` created a second `mlflow.db` there, invisible to
a UI launched at the repo root. `mlflow_util.setup()` now pins an absolute path; this
script rescues the runs logged before that fix.

Copies run name, params, tags, and the FULL metric history (so per-epoch curves survive),
and skips runs already present in the destination (matched on name + start time), so it
is safe to re-run after the training chain finishes writing.

    .venv/bin/python pouring/pour_probe/mlflow_migrate.py --src pouring/pour_probe/mlflow.db
    .venv/bin/python pouring/pour_probe/mlflow_migrate.py --src ... --delete_src
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

import mlflow_util

SKIP_EXPERIMENTS = {"Default", "_nan_probe_test"}


def migrate(src_db, dst_uri, dry_run=False):
    src = MlflowClient(tracking_uri=f"sqlite:///{Path(src_db).resolve()}")
    dst = MlflowClient(tracking_uri=dst_uri)
    moved = skipped = 0
    for exp in src.search_experiments():
        if exp.name in SKIP_EXPERIMENTS:
            continue
        d = dst.get_experiment_by_name(exp.name)
        dst_exp_id = d.experiment_id if d else dst.create_experiment(exp.name)
        existing = {(r.data.tags.get("mlflow.runName"), r.info.start_time)
                    for r in dst.search_runs([dst_exp_id], max_results=5000)}
        for r in src.search_runs([exp.experiment_id], max_results=5000):
            name = r.data.tags.get("mlflow.runName")
            if (name, r.info.start_time) in existing:
                skipped += 1
                continue
            if dry_run:
                print(f"  would move {exp.name}/{name}")
                moved += 1
                continue
            new = dst.create_run(dst_exp_id, start_time=r.info.start_time, run_name=name)
            tags = [RunTag(k, str(v)) for k, v in r.data.tags.items()
                    if not k.startswith("mlflow.runName")]
            params = [Param(k, str(v)) for k, v in r.data.params.items()]
            metrics = []
            for k in r.data.metrics:
                metrics += [Metric(k, m.value, m.timestamp, m.step)
                            for m in src.get_metric_history(r.info.run_id, k)]
            # batches of 1000 (MLflow's per-call limit)
            for i in range(0, max(len(metrics), 1), 900):
                dst.log_batch(new.info.run_id, metrics=metrics[i:i + 900],
                              params=params if i == 0 else [],
                              tags=tags if i == 0 else [])
            dst.set_terminated(new.info.run_id, status=r.info.status,
                               end_time=r.info.end_time)
            moved += 1
        print(f"  {exp.name}: moved so far {moved}, skipped {skipped}")
    return moved, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--delete_src", action="store_true")
    args = ap.parse_args()
    dst = mlflow_util.DEFAULT_URI
    print(f"src sqlite:///{Path(args.src).resolve()}\ndst {dst}")
    moved, skipped = migrate(args.src, dst, args.dry_run)
    print(f"moved {moved}, skipped {skipped} (already present)")
    if args.delete_src and not args.dry_run:
        Path(args.src).unlink()
        print(f"deleted {args.src}")


if __name__ == "__main__":
    main()

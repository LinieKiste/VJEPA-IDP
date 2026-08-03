#!/usr/bin/env python3
"""Export the mlflow store to plain CSV under mlflow_export/.

Why: the binary mlflow.db is awkward in git (10 MB re-added on every commit that touches
it, no diffs) and GitHub's secret scanner false-positives on it — a 32-hex mlflow run_uuid
sitting after the bytes "AC" in a sqlite page is byte-identical to a Twilio Account SID.
The CSVs carry the same numbers, diff cleanly, and are readable without mlflow installed.

Writes three files:
  runs.csv    one row per run: experiment, name, status, timestamps
  metrics.csv one row per run x metric, the LAST logged step
  params.csv  one row per run x param

    python pouring/pour_probe/mlflow_export.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "mlflow.db"
OUT = ROOT / "mlflow_export"

QUERIES = {
    "runs.csv": """
        select e.name as experiment, r.name as run, r.run_uuid, r.status,
               datetime(r.start_time/1000, 'unixepoch') as started,
               datetime(r.end_time/1000, 'unixepoch') as ended
        from runs r join experiments e on e.experiment_id = r.experiment_id
        order by e.name, r.start_time
    """,
    # A metric can be logged many times per run; keep the final value per (run, key).
    "metrics.csv": """
        select e.name as experiment, r.name as run, m.key, m.value, m.step
        from metrics m
        join runs r on r.run_uuid = m.run_uuid
        join experiments e on e.experiment_id = r.experiment_id
        where (m.run_uuid, m.key, m.step) in (
            select run_uuid, key, max(step) from metrics group by run_uuid, key
        )
        order by e.name, r.name, m.key
    """,
    "params.csv": """
        select e.name as experiment, r.name as run, p.key, p.value
        from params p
        join runs r on r.run_uuid = p.run_uuid
        join experiments e on e.experiment_id = r.experiment_id
        order by e.name, r.name, p.key
    """,
}


def main() -> int:
    if not DB.exists():
        print(f"no mlflow store at {DB}", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    for fname, sql in QUERIES.items():
        cur = con.execute(sql)
        rows = cur.fetchall()
        with open(OUT / fname, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([d[0] for d in cur.description])
            w.writerows(rows)
        print(f"{fname}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

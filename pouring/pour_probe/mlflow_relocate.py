#!/usr/bin/env python3
"""Repoint the mlflow store's artifact paths at wherever this repo now lives.

Why this exists: mlflow stores artifact locations as ABSOLUTE paths. The committed
`mlflow.db` therefore has 177 runs pointing at `/home/casimir/UNI/SS_26/idp/mlruns/...`.
Metrics and params live in the db itself and always show up, but the UI cannot open a
single logged artifact after the repo is cloned to a different path (a different laptop,
a different username). This rewrites `experiments.artifact_location` and
`runs.artifact_uri` to the current repo root.

Idempotent, and a no-op when the paths already match. Run it once after cloning:

    python pouring/pour_probe/mlflow_relocate.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "mlflow.db"


def main() -> int:
    if not DB.exists():
        print(f"no mlflow store at {DB}", file=sys.stderr)
        return 1

    con = sqlite3.connect(DB)
    # Every path shares the old repo root as a prefix; find it from the data rather than
    # hardcoding it, so this keeps working after the repo moves a second time.
    rows = con.execute(
        "select artifact_location from experiments where artifact_location like '/%'"
    ).fetchall()
    if not rows:
        print("no absolute artifact paths — nothing to do")
        return 0

    # ".../mlruns/4" or ".../pouring/pour_probe/mlruns/7" -> the part before "/mlruns".
    # Two prefixes exist because experiments 7-8 were logged through the CWD-relative
    # store described in mlflow_util.py; the shallowest one is the actual repo root and
    # the deeper one is a subpath of it, so rewriting that prefix fixes both.
    olds = {r[0].split("/mlruns")[0] for r in rows if "/mlruns" in r[0]}
    old = min(olds, key=len)
    if not all(o.startswith(old) for o in olds):
        print(f"artifact roots do not share a prefix: {sorted(olds)}", file=sys.stderr)
        return 1
    new = str(ROOT)
    if old == new:
        print(f"already rooted at {new} — nothing to do")
        return 0

    with con:
        con.execute(
            "update experiments set artifact_location = replace(artifact_location, ?, ?)",
            (old, new),
        )
        con.execute(
            "update runs set artifact_uri = replace(artifact_uri, ?, ?)", (old, new)
        )
    n = con.execute("select count(*) from runs").fetchone()[0]
    print(f"repointed {n} runs and {len(rows)} experiments\n  {old}\n-> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

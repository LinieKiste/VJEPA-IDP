"""eXprt tea-making dataset: trial->video mapping, labels, and frame access.

The eXprt recordings (``datasets/eXprt-Daten/CAM1 Aufnahmen Patrick/``) are PNG frame
sequences (``frame_NNNN.png``, 1886x1056 @ 20 fps). Labels live in
``trial_execution_log_*.json`` as 40 trials = 8 classes x 5 iterations, but the log has
**no video-id column** and there are more recording dirs than trials (aborted re-takes).

The only recording-time signal is the **directory-name timestamp** (``YYYYMMDD_HHMMSS_mmm``
= millisecond recording start) -- the PNGs themselves carry no capture date (empty metadata;
file mtime is just the rclone copy time). So we map each trial to its recording with:

  **bucket each recording dir to the most-recent trial ``Start Time`` that precedes it; if a
  trial's bucket holds multiple recordings (abort + retake) keep the one with the most frames.**

This yields a clean 40<->40, 5-per-class mapping and correctly drops aborts. Running this
module persists the mapping to ``exprt_probe/mapping.json`` and adds a ``video_id`` column to
the CSV (non-destructive; the ~200 GB of frame folders are left untouched).

Usage:
    .venv/bin/python exprt_probe/dataset.py            # build + persist + verify
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "eXprt-Daten" / "CAM1 Aufnahmen Patrick"
LOG_JSON = DATA_DIR / "trial_execution_log_20250827_172954.json"
LOG_CSV = DATA_DIR / "trial_execution_log_20250827_172954.csv"
MAPPING_JSON = Path(__file__).parent / "mapping.json"

# Canonical 8-class order (Normal first so binary == bool(class_id)). Names match the log's
# "Trial Name" (lowercase keys in the data; "Spüli" carries the umlaut).
CLASSES = [
    "Normal", "2tb 2stir", "Spüli", "glass and fork",
    "no tea bag", "not enough water", "perplexity", "sequence",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

_DIR_RE = re.compile(r"(\d{8})_(\d{6})_(\d+)_GX(\d+)_20fps$")


def _dir_timestamp(name: str) -> datetime | None:
    """Recording-start time encoded in a ``..._20fps`` dir name, or None if not a video dir."""
    m = _DIR_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        microsecond=int(m.group(3)) * 1000
    )


def _n_frames(dir_path: Path) -> int:
    return sum(1 for f in dir_path.iterdir() if f.suffix == ".png")


def build_mapping() -> list[dict]:
    """Return one record per trial: trial->video assignment + labels, in execution order.

    Each record: {video_id, trial_name, iteration, execution_order, start_time, class_id,
    binary, n_frames, offset_s}. ``video_id`` is the matched recording dir name.
    """
    log = json.loads(LOG_JSON.read_text())["execution_log"]
    log = sorted(log, key=lambda t: t["execution_order"])
    starts = [(datetime.fromisoformat(t["start_time"]), t) for t in log]

    dirs = []
    for child in sorted(DATA_DIR.iterdir()):
        ts = _dir_timestamp(child.name)
        if ts is not None:
            dirs.append((ts, child.name, _n_frames(child)))
    dirs.sort()

    # bucket each recording to the most-recent trial start that precedes it
    buckets: dict[int, list] = {t["execution_order"]: [] for _, t in starts}
    for ts, name, n in dirs:
        preceding = [t for st, t in starts if st <= ts] or [starts[0][1]]
        owner = max(preceding, key=lambda t: datetime.fromisoformat(t["start_time"]))
        buckets[owner["execution_order"]].append((ts, name, n))

    records = []
    for st, t in starts:
        cands = buckets[t["execution_order"]]
        if not cands:
            raise RuntimeError(f"trial {t['execution_order']} ({t['trial_name']}) has no recording")
        ts, name, n = max(cands, key=lambda x: x[2])  # keep-longest (drop aborts)
        records.append({
            "video_id": name,
            "trial_name": t["trial_name"],
            "iteration": int(t["iteration"]),
            "execution_order": t["execution_order"],
            "start_time": t["start_time"],
            "class_id": CLASS_TO_ID[t["trial_name"]],
            "binary": 0 if t["trial_name"] == "Normal" else 1,
            "n_frames": n,
            "offset_s": round((ts - st).total_seconds(), 1),
        })
    return records


def persist_mapping(records: list[dict]) -> None:
    """Write mapping.json and add a ``video_id`` column to the CSV (keyed by execution_order)."""
    MAPPING_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    by_order = {r["execution_order"]: r["video_id"] for r in records}
    rows = list(csv.DictReader(LOG_CSV.open(newline="")))
    fields = list(rows[0].keys())
    if "video_id" not in fields:
        fields.append("video_id")
    for row in rows:
        row["video_id"] = by_order.get(int(row["Execution Order"]), "")
    with LOG_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@lru_cache(maxsize=1)
def list_videos() -> list[dict]:
    """Mapping records, loaded from mapping.json (build it first via ``python dataset.py``)."""
    if not MAPPING_JSON.exists():
        raise FileNotFoundError(f"{MAPPING_JSON} missing — run `python exprt_probe/dataset.py` first")
    return json.loads(MAPPING_JSON.read_text())


def video_dir(video_id: str) -> Path:
    return DATA_DIR / video_id


def frame_paths(video_id: str) -> list[Path]:
    """Sorted list of frame PNG paths for a recording."""
    return sorted(video_dir(video_id).glob("frame_*.png"))


def _verify(records: list[dict]) -> None:
    from collections import Counter
    vids = [r["video_id"] for r in records]
    assert len(records) == 40, f"expected 40 trials, got {len(records)}"
    assert len(set(vids)) == 40, "video assignments not unique"
    counts = Counter(r["trial_name"] for r in records)
    assert all(counts[c] == 5 for c in CLASSES), f"class counts not 5 each: {dict(counts)}"
    for r in records:
        assert video_dir(r["video_id"]).is_dir(), f"missing dir {r['video_id']}"
    print(f"OK: 40 trials -> 40 distinct dirs, 5 per class, all present.")
    big = [r for r in records if r["offset_s"] > 120]
    if big:
        print(f"note: {len(big)} trial(s) with >120s start->record offset (benign, sole recording):")
        for r in big:
            print(f"  exec {r['execution_order']:>2} {r['trial_name']:<16} {r['video_id']}  +{r['offset_s']:.0f}s")


def main():
    records = build_mapping()
    _verify(records)
    persist_mapping(records)
    print(f"wrote {MAPPING_JSON.relative_to(ROOT)} and added 'video_id' column to {LOG_CSV.name}")


if __name__ == "__main__":
    main()

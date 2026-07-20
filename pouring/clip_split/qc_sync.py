#!/usr/bin/env python3
"""3-camera side-by-side sync preview of one pour (Gate E).

CAM1|CAM2|CAM3 hstacked over the clip window of one event — CAM1 appears HERE
ONLY (never in the final clips) so the scale trace can be eyeballed against the
action: the pour impact must be simultaneous in all three views and the scale
display must ramp exactly while liquid flows. Audio from CAM1 is kept (a desync
between sound and sight is easy to hear).

Usage: qc_sync.py --trial N [--pour K] [--events events.csv] [--height 480]
Output: qc/sync_trial<N>_pour<K>.mp4
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "datasets" / "pouring_processed"
CAMS = ("CAM1", "CAM2", "CAM3")


def annotation_event(stem, pour_idx):
    """Event window from the interactive annotator (completed, non-excluded,
    ordered by start time; pour_idx is 1-based in that order)."""
    ann = json.loads((HERE / "annotations.json").read_text())
    evs = sorted((e for e in ann[stem]["events"]
                  if e["completed"] and not e["excluded"]),
                 key=lambda e: e["clip_start_s"])
    e = evs[int(pour_idx) - 1]
    return {"clip_start_s": e["clip_start_s"], "clip_end_s": e["clip_end_s"],
            "weight_g": e["weight_g"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trial", required=True)
    ap.add_argument("--pour", default="1")
    ap.add_argument("--events", type=Path, default=HERE / "events.csv")
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    trial = next(r for r in csv.DictReader(open(HERE / "trials.csv"))
                 if r["trial_id"] == args.trial)
    stem = Path(trial["cam1_file"]).stem
    if (HERE / "annotations.json").exists():
        ev = annotation_event(stem, args.pour)
    else:
        ev = next(r for r in csv.DictReader(open(args.events))
                  if r["cam1_stem"] == stem and r["pour_idx"] == args.pour)
    a, b = float(ev["clip_start_s"]), float(ev["clip_end_s"])

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    labels = []
    for i, cam in enumerate(CAMS):
        key = cam.lower()
        off = float(trial[f"{key}_offset_ms"]) / 1000.0
        cmd += ["-ss", f"{a - off:.3f}", "-t", f"{b - a:.3f}",
                "-i", str(DATA / cam / trial[f"{key}_file"])]
        labels.append(
            f"[{i}:v]scale=-2:{args.height},"
            f"drawtext=text='{cam}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]"
        )
    fc = ";".join(labels) + f";{''.join(f'[v{i}]' for i in range(3))}hstack=3[out]"

    out = HERE / "qc" / f"sync_trial{args.trial}_pour{args.pour}.mp4"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True)
    print(f"trial {args.trial} pour {args.pour}  window [{a:.2f},{b:.2f}]s "
          f"weight {ev['weight_g']} g -> {out}")


if __name__ == "__main__":
    main()

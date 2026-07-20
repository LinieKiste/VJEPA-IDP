#!/usr/bin/env python3
"""Contact-sheet QC for detected pour boundaries (Gate D).

For every pour in events.csv, grab CAM1 frames at the four boundary times
(clip_start, rise, plateau_start, clip_end) and tile them into one PNG per
trial video (one row per pour, columns annotated with time + weight).
Lets the user audit every clip boundary in minutes without watching video.

Usage: qc_boundaries.py [--events events.csv] [--qc-dir qc/] [--thumb-w 480]
Output: qc/boundaries_<cam1_stem>.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAM1_DIR = ROOT / "datasets" / "pouring_processed" / "CAM1"

COLUMNS = [  # (events.csv column, header label)
    ("clip_start_s", "clip start (-pad)"),
    ("rise_t", "pour start (rise)"),
    ("plateau_t", "plateau start"),
    ("clip_end_s", "clip end"),
]
HEADER_H = 28
LABEL_H = 22


def grab(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def annotate_bar(width, text, bg, fg=(255, 255, 255), h=LABEL_H):
    bar = np.full((h, width, 3), bg, np.uint8)
    cv2.putText(bar, text, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, default=HERE / "events.csv")
    ap.add_argument("--qc-dir", type=Path, default=HERE / "qc")
    ap.add_argument("--thumb-w", type=int, default=480)
    args = ap.parse_args()

    by_stem = defaultdict(list)
    for row in csv.DictReader(open(args.events)):
        by_stem[row["cam1_stem"]].append(row)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    for stem, pours in by_stem.items():
        video = CAM1_DIR / f"{stem}.mp4"
        if not video.exists():
            print(f"WARN: {video.name} missing, skipping")
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)

        rows = []
        # header row: column titles
        header = [annotate_bar(args.thumb_w, label, bg=(60, 60, 60), h=HEADER_H)
                  for _, label in COLUMNS]
        rows.append(np.hstack(header))
        for ev in pours:
            tiles = []
            for col, _ in COLUMNS:
                t = float(ev[col])
                frame = grab(cap, fps, t)
                if frame is None:
                    frame = np.zeros((270, args.thumb_w, 3), np.uint8)
                scale = args.thumb_w / frame.shape[1]
                thumb = cv2.resize(frame, (args.thumb_w, int(frame.shape[0] * scale)))
                label = f"pour {ev['pour_idx']}  t={t:.2f}s  {ev['weight_g']} g"
                if ev.get("flags"):
                    label += f"  [{ev['flags']}]"
                tiles.append(np.vstack([thumb, annotate_bar(args.thumb_w, label, bg=(30, 90, 30))]))
            rows.append(np.hstack(tiles))
        cap.release()

        out = args.qc_dir / f"boundaries_{stem}.png"
        cv2.imwrite(str(out), np.vstack(rows))
        print(f"{stem}: {len(pours)} pours -> {out.name}")


if __name__ == "__main__":
    main()

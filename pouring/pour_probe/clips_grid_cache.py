"""Cache decoded 288-px frames + fps + GT curve per clip, so the attentive-probe
trainer (clips_train_attn.py) can apply pixel-space augmentation (random 256-crop
from 288 + h-flip + temporal jitter) and run the frozen encoder in-loop without
re-decoding the mp4 every epoch.

One npz per clip: frames (Nframes,288,288,3) uint8, fps, gt_t, gt_w, trial_id,
weight_final. ~37 MB/clip; CAM2 only ≈ 4.5 GB.

Usage: .venv/bin/python pour_probe/clips_grid_cache.py --cam CAM2
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import decord
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "datasets" / "pouring_processed" / "clips"
CACHE = Path(os.environ.get("POUR_FRAMES288_DIR", "/home/casimir/.cache/pour_probe/clips_frames288"))


def crop_sq(img, size=288):
    """Short-side resize to ``size`` then centre-crop square. ``size`` must leave a
    margin over the encoder's crop (288/256 by default) for the random-crop augmentation."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    r = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ch, cw = r.shape[:2]
    y0, x0 = (ch - size) // 2, (cw - size) // 2
    return r[y0:y0 + size, x0:x0 + size]


def load_gt(clip_id):
    t, w = [], []
    with open(CLIPS / "csv" / f"{clip_id}.csv") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"])); w.append(float(r["weight"]))
    return np.asarray(t, np.float32), np.asarray(w, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="CAM2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--size", type=int, default=288,
                    help="cached frame size; use 416 for a 384-px encoder run")
    ap.add_argument("--out", default="", help="override cache dir (default POUR_FRAMES288_DIR)")
    args = ap.parse_args()

    global CACHE
    if args.out:
        CACHE = Path(args.out)
    (CACHE / args.cam).mkdir(parents=True, exist_ok=True)
    manifest = list(csv.DictReader(open(CLIPS / "clips_manifest.csv")))
    for row in tqdm(manifest, desc=f"cache {args.cam}"):
        cid = row["clip_id"]
        out = CACHE / args.cam / f"{cid}.npz"
        if out.exists() and not args.overwrite:
            continue
        vr = decord.VideoReader(str(CLIPS / args.cam / f"{cid}.mp4"))
        fps = float(vr.get_avg_fps())
        frames = np.stack([crop_sq(f, args.size) for f in vr[:].asnumpy()]).astype(np.uint8)
        gt_t, gt_w = load_gt(cid)
        np.savez(out, frames=frames, fps=fps, gt_t=gt_t, gt_w=gt_w,
                 trial_id=row["trial_id"], weight_final=float(row["weight_g"]),
                 source_obj=row["source_obj"], target_obj=row["target_obj"])


if __name__ == "__main__":
    main()

"""Extract frozen V-JEPA 2 ViT-L mean-pool features + real weight targets over the
own-lab pour clips (datasets/pouring_processed/clips/).

Each clip (CAM2 and CAM3 third-person views, 29.97 fps) is cut into overlapping
short windows (in SECONDS). Each window is sampled to ``num_frames`` frames
(short-side-256 + center-crop-256, the pour action sits centered), ImageNet-
normalized, and encoded by the frozen V-JEPA 2 ViT-L; the patch-token grid is
**mean-pooled** to a 1024-d vector (the fast "does it work at all" path — token
grids for the attentive-pooler comparison come later if this passes).

Per-window targets from the per-clip GT curve (csv/<id>.csv = t_s,weight, the
human-verified poured mass 0..W at ~120 Hz):
  - ``volume`` = weight at the window's CENTER time (cumulative poured mass so far).
  - ``flow``   = (weight(t_end) - weight(t_start)) / window_s (g/s over the window).

One npz per (clip, camera): feats/<cam>/<clip_id>.npz. Metadata carries trial_id
(the CV group — clips from one trial share scene/container) and modality.

Usage:
    .venv/bin/python pour_probe/clips_extract.py --limit 3      # pilot
    .venv/bin/python pour_probe/clips_extract.py                # all clips, both cams
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
from tqdm import tqdm

from _encoder import load_encoder

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "datasets" / "pouring_processed" / "clips"
FEATURES_DIR = Path(os.environ.get("POUR_CLIPS_FEATURES_DIR",
                                   "/home/casimir/.cache/pour_probe/clips_feats"))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def load_gt(clip_id):
    """(t_s, weight) arrays from the per-clip GT csv."""
    t, w = [], []
    with open(CLIPS / "csv" / f"{clip_id}.csv") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"])); w.append(float(r["weight"]))
    return np.asarray(t), np.asarray(w)


def crop256(img):
    """short-side -> 256, center-crop 256x256 (RGB HWC uint8)."""
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    r = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ch, cw = r.shape[:2]
    y0, x0 = (ch - 256) // 2, (cw - 256) // 2
    return r[y0:y0 + 256, x0:x0 + 256]


def sliding_windows(dur, window_s, stride_s):
    """Yield [t0, t1] windows (seconds) covering [0, dur], last window clamped to end."""
    out, t0 = [], 0.0
    while True:
        t1 = min(dur, t0 + window_s)
        out.append([t0, t1])
        if t1 >= dur:
            break
        t0 += stride_s
    return out


@torch.no_grad()
def embed_batch(enc, batch_u8, device, mean, std):
    x = torch.from_numpy(batch_u8).to(device)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)   # (B,C,T,H,W)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = enc(x)                                    # (B,N,1024)
    return tok.float().mean(dim=1).cpu().numpy()        # mean-pool -> (B,1024)


def extract_clip(enc, clip_id, cam, num_frames, window_s, stride_s, device, batch_clips):
    path = CLIPS / cam / f"{clip_id}.mp4"
    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    total = len(vr)
    dur = total / fps
    tgt, wgt = load_gt(clip_id)

    wins = sliding_windows(dur, window_s, stride_s)
    # frame index per window (num_frames evenly spaced in [t0,t1])
    win_idx = [np.clip((np.linspace(t0, t1, num_frames) * fps).astype(int), 0, total - 1)
               for t0, t1 in wins]
    uniq = sorted({int(i) for idx in win_idx for i in idx})
    pos = {fi: k for k, fi in enumerate(uniq)}
    frames = vr.get_batch(uniq).asnumpy()               # (Nuniq,H,W,3) RGB
    frames = np.stack([crop256(f) for f in frames])     # (Nuniq,256,256,3)

    clip_rows = np.asarray([[pos[int(i)] for i in idx] for idx in win_idx])
    mean, std = MEAN.to(device), STD.to(device)
    feats = []
    for s in range(0, len(clip_rows), batch_clips):
        feats.append(embed_batch(enc, frames[clip_rows[s:s + batch_clips]], device, mean, std))
    feats = np.concatenate(feats).astype(np.float16)

    tmid = np.asarray([(a + b) / 2 for a, b in wins], dtype=np.float32)
    volume = np.interp(tmid, tgt, wgt).astype(np.float32)
    flow = np.asarray([(np.interp(b, tgt, wgt) - np.interp(a, tgt, wgt)) / max(b - a, 1e-3)
                       for a, b in wins], dtype=np.float32)
    return feats, tmid, volume, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", nargs="+", default=["CAM2", "CAM3"])
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--stride_s", type=float, default=0.5)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_clips", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(CLIPS / "clips_manifest.csv")))
    if args.limit:
        manifest = manifest[: args.limit]
    enc = load_encoder(img_size=args.img_size, num_frames=args.num_frames, device=args.device)

    for cam in args.cams:
        (FEATURES_DIR / cam).mkdir(parents=True, exist_ok=True)
    print(f"extracting {len(manifest)} clips x {len(args.cams)} cams -> {FEATURES_DIR} "
          f"(win {args.window_s}s stride {args.stride_s}s, {args.num_frames}f)")
    for row in tqdm(manifest):
        cid = row["clip_id"]
        for cam in args.cams:
            out = FEATURES_DIR / cam / f"{cid}.npz"
            if out.exists() and not args.overwrite:
                continue
            feats, tmid, volume, flow = extract_clip(
                enc, cid, cam, args.num_frames, args.window_s, args.stride_s,
                args.device, args.batch_clips)
            np.savez(out, emb=feats, tmid=tmid, volume=volume, flow=flow,
                     clip_id=cid, trial_id=row["trial_id"], cam=cam,
                     weight_final=float(row["weight_g"]),
                     source_obj=row["source_obj"], target_obj=row["target_obj"])


if __name__ == "__main__":
    main()

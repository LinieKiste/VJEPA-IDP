"""Measure the compression (short-over/long-under bias) of a trained attentive
checkpoint: per-window flow R²/slope + per-clip integrated-total slope & MAE on the
held-out val trials (both cams). slope 1.0 = unbiased; <1 = compression.

Usage: .venv/bin/python pour_probe/clips_attn_slope.py --ckpt <path>
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

import clips_train_attn as ca
from clips_extract import CLIPS
from _encoder import load_encoder
from head import build_head


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lag_s", type=float, default=0.7)
    args = ap.parse_args()

    cams = {c: ca.load_clips(c) for c in ("CAM2", "CAM3")}
    wins = ca.build_windows(cams, 1.0, 0.5, 16, args.lag_s)
    tr = [w for w in wins if w["trial"] not in ca.VAL_TRIALS]
    va = [w for w in wins if w["trial"] in ca.VAL_TRIALS]
    ytr = np.asarray([w["flow"] for w in tr], np.float32)
    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

    man = {r["clip_id"]: float(r["weight_g"])
           for r in csv.DictReader(open(CLIPS / "clips_manifest.csv"))}

    enc = load_encoder(img_size=256, num_frames=16, device="cuda")
    head = build_head(1).to("cuda")
    head.load_state_dict(torch.load(args.ckpt))
    head.eval()
    mean, std = ca.MEAN.to("cuda"), ca.STD.to("cuda")

    r2, mae, preds, ys = ca.run_eval(enc, head, cams, va, "flow", 16, "cuda",
                                     mean, std, ymean, ystd)

    # per (cam, clip) integrate predicted flow -> total; compare to GT total mass
    by = defaultdict(list)
    for w, p in zip(va, preds):
        by[(w["cam"], w["clip"])].append(((w["t0"] + w["t1"]) / 2, float(p)))
    pred_tot, gt_tot = [], []
    for (cam, cid), pts in by.items():
        pts.sort()
        t = np.array([a for a, _ in pts]); pf = np.clip([b for _, b in pts], 0, None)
        pred_tot.append(np.trapezoid(pf, t)); gt_tot.append(man[cid])
    pred_tot, gt_tot = np.array(pred_tot), np.array(gt_tot)

    print(f"\n=== {args.ckpt.split('/')[-1]} ===")
    print(f"  flow  : R²={r2:.3f}  MAE={mae:.2f} g/s  slope={slope(ys, preds):.3f}")
    print(f"  total : R²={r2_score(gt_tot, pred_tot):.3f}  "
          f"MAE={mean_absolute_error(gt_tot, pred_tot):.1f} g  "
          f"slope={slope(gt_tot, pred_tot):.3f}  (1.0=unbiased, <1=compression)")


if __name__ == "__main__":
    main()

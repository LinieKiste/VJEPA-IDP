"""Post-hoc calibration of the deliverable flow probe to remove the residual
short-over/long-under compression WITHOUT retraining (a linear rescale of the
predictions adds no variance, unlike de-hedging the loss).

Honest evaluation: leave-one-val-trial-out. The linear flow calibration (flow_true ~
a + b*flow_pred) is fit on the OTHER val trials and applied to the held-out one, so the
correction is never fit on the clip it scores. Reports integrated-total MAE + slope
before vs after calibration.

Usage: .venv/bin/python pour_probe/clips_calibrate.py --ckpt <deliverable.pt>
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


def integ_totals(rows, man, flow_key):
    """rows: list of (cam,clip,trial,tmid,flow). Integrate per (cam,clip)."""
    by = defaultdict(list)
    for cam, clip, tr, tmid, f in rows:
        by[(cam, clip)].append((tmid, f))
    pred, gt = [], []
    for (cam, clip), pts in by.items():
        pts.sort()
        t = np.array([a for a, _ in pts]); fl = np.clip([b for _, b in pts], 0, None)
        pred.append(np.trapezoid(fl, t)); gt.append(man[clip])
    return np.array(pred), np.array(gt)


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
    _, _, preds, trues = ca.run_eval(enc, head, cams, va, "flow", 16, "cuda",
                                     mean, std, ymean, ystd)

    raw = [(w["cam"], w["clip"], w["trial"], (w["t0"] + w["t1"]) / 2, float(p), float(tf))
           for w, p, tf in zip(va, preds, trues)]

    # uncalibrated
    rows_u = [(c, cl, t, tm, pf) for c, cl, t, tm, pf, tf in raw]
    pu, gu = integ_totals(rows_u, man, None)

    # leave-one-val-trial-out linear flow calibration
    val_trials = sorted(set(r[2] for r in raw))
    rows_c = []
    for held in val_trials:
        fit = [(r[4], r[5]) for r in raw if r[2] != held]        # (pred_flow, true_flow) others
        fp = np.array([a for a, _ in fit]); ft = np.array([b for _, b in fit])
        b, a = np.polyfit(fp, ft, 1)                             # true ~ a + b*pred
        for r in raw:
            if r[2] == held:
                rows_c.append((r[0], r[1], r[2], r[3], a + b * r[4]))
    pc, gc = integ_totals(rows_c, man, None)

    print(f"\n=== calibration [{args.ckpt.split('/')[-1]}] (leave-one-val-trial-out) ===")
    print(f"  {'':<14} {'total MAE':>10} {'total slope':>12} {'total R²':>9}")
    print(f"  {'uncalibrated':<14} {mean_absolute_error(gu, pu):>9.1f}g "
          f"{slope(gu, pu):>12.3f} {r2_score(gu, pu):>9.3f}")
    print(f"  {'calibrated':<14} {mean_absolute_error(gc, pc):>9.1f}g "
          f"{slope(gc, pc):>12.3f} {r2_score(gc, pc):>9.3f}")


if __name__ == "__main__":
    main()

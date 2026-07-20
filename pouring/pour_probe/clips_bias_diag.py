"""Diagnose the short-over / long-under compression seen in the weight reconstruction.

Hypothesis (user): the loss rewards average predictions -> flow is shrunk toward the
mean -> low-flow pours over-integrate, high-flow pours under-integrate. Regularization
is the direct knob for "how much the model hedges toward the mean", so on the cheap
mean-pool RIDGE probe we sweep alpha and measure:
  - per-window flow SLOPE  (pred_flow ~ a + b*true_flow; b<1 = shrinkage)
  - per-clip integrated-total SLOPE (pred_total ~ a + b*GT_total; b<1 = the compression)
If the slope -> 1 as alpha -> 0 (with R2 holding up), the compression is mostly
regularization/loss-hedging and an attentive retrain with a less-hedging loss (MSE
instead of SmoothL1) + lower weight decay should help. If the slope stays <1 even at
alpha~0, a big chunk is irreducible (aleatoric: absolute g/s isn't fully visible from a
wide shot) and the honest fix is a post-hoc calibration.

Usage: .venv/bin/python pour_probe/clips_bias_diag.py --cam CAM2
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from sklearn.metrics import r2_score

import clips_train as ct


def slope(x, y):
    """b in y ~ a + b*x (least squares)."""
    b, a = np.polyfit(x, y, 1)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="CAM2", choices=["CAM2", "CAM3", "both"])
    args = ap.parse_args()

    if args.cam == "both":
        X, y, groups, cids, tmid = ct.load_both("flow")
    else:
        X, y, groups, cids, tmid = ct.load(args.cam, "flow")

    # GT total poured mass per clip (manifest weight_g)
    man = {r["clip_id"]: float(r["weight_g"])
           for r in csv.DictReader(open(ct.CLIPS / "clips_manifest.csv"))}

    uclips = list(dict.fromkeys(cids.tolist()))
    gt_tot = np.array([man[c] for c in uclips])

    print(f"=== compression diagnostic [{args.cam}] "
          f"{len(y)} windows / {len(uclips)} clips (GT total {gt_tot.min():.0f}-{gt_tot.max():.0f} g) ===")
    print(f"  {'alpha':>7} {'flowR2':>7} {'flowSlope':>10} {'totR2':>7} {'totSlope':>9} "
          f"{'totMAE':>8}")
    for alpha in [1000, 100, 30, 10, 3, 1, 0.1]:
        pred, _ = ct.oof(ct.ridge_fit(alpha), X, y, groups, 6)
        flow_r2 = r2_score(y, pred)
        flow_slope = slope(y, pred)
        # integrate predicted (and true) flow per clip on the window-centre grid
        pred_tot, true_tot = [], []
        for c in uclips:
            m = cids == c
            o = np.argsort(tmid[m])
            t = tmid[m][o]
            pf = np.clip(pred[m][o], 0, None)
            pred_tot.append(np.trapezoid(pf, t))
        pred_tot = np.array(pred_tot)
        tot_r2 = r2_score(gt_tot, pred_tot)
        tot_slope = slope(gt_tot, pred_tot)
        tot_mae = np.mean(np.abs(pred_tot - gt_tot))
        print(f"  {alpha:>7} {flow_r2:>7.3f} {flow_slope:>10.3f} "
              f"{tot_r2:>7.3f} {tot_slope:>9.3f} {tot_mae:>7.1f}g")

    print("\n  slope 1.0 = unbiased; <1 = short-over/long-under compression.")
    print("  if totSlope climbs toward ~1 as alpha drops (flowR2 holding) -> loss/reg hedging,")
    print("  fixable by MSE + lower weight_decay on the attentive probe.")


if __name__ == "__main__":
    main()

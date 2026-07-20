"""Measure the water-transit LAG between the visual pour and the scale weight.

Physics: at video time t the camera sees the stream leaving the spout (visual rate
v(t)); the scale only registers that mass after a transit + settling delay tau, so
the measured weight derivative is dW/dt(t) = v(t - tau). The GT flow target
(delta-weight / window) is therefore a DELAYED copy of what V-JEPA sees. To align the
target with the frames, the flow for a window centred at t should be sampled from the
weight curve at t + tau.

The cached V-JEPA mean-pool features do NOT depend on tau (same frames) — only the
TARGETS shift. So we can measure tau almost for free: hold the features fixed, recompute
the flow (and volume) targets for a grid of lags, refit the same trial-grouped OOF ridge
probe, and read off the lag that maximises held-out R^2. A peak at tau>0 confirms the
physical delay and gives the alignment the real (attentive) probe should bake in.

Usage:
    .venv/bin/python pour_probe/clips_lag_sweep.py --target flow --cam CAM2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import clips_train as ct
from clips_extract import load_gt

HERE = Path(__file__).resolve().parent


def load_with_curves(cam):
    """Per-window feats/tmid/trial/clip PLUS each clip's GT curve, so targets can be
    recomputed at an arbitrary lag. Rows align with ct.load(cam,*)."""
    files = sorted((ct.FEATURES_DIR / cam).glob("*.npz"))
    X, grp, cids, tmid = [], [], [], []
    curves = {}
    for f in files:
        a = np.load(f, allow_pickle=True)
        n = len(a["emb"]); cid = str(a["clip_id"])
        X.append(a["emb"].astype(np.float32))
        grp.extend([str(a["trial_id"])] * n); cids.extend([cid] * n)
        tmid.append(a["tmid"])
        if cid not in curves:
            curves[cid] = load_gt(cid)                       # (t_s, weight)
    return (np.concatenate(X), np.asarray(grp), np.asarray(cids),
            np.concatenate(tmid).astype(np.float32), curves)


def targets_at_lag(target, cids, tmid, curves, lag, window_s):
    """Recompute per-window target with the weight curve sampled `lag` seconds LATER."""
    y = np.empty(len(tmid), np.float32)
    half = window_s / 2
    for cid in set(cids.tolist()):
        m = cids == cid
        gt_t, gt_w = curves[cid]
        tc = tmid[m]
        if target == "volume":
            y[m] = np.interp(tc + lag, gt_t, gt_w)
        else:  # flow = (W(t1+lag) - W(t0+lag)) / window
            t0, t1 = tc - half, tc + half
            y[m] = (np.interp(t1 + lag, gt_t, gt_w)
                    - np.interp(t0 + lag, gt_t, gt_w)) / window_s
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["flow", "volume"], default="flow")
    ap.add_argument("--cam", choices=["CAM2", "CAM3", "both"], default="CAM2")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--n_splits", type=int, default=6)
    ap.add_argument("--lag_min", type=float, default=-0.4)
    ap.add_argument("--lag_max", type=float, default=0.8)
    ap.add_argument("--lag_step", type=float, default=0.05)
    ap.add_argument("--window_s", type=float, default=1.0)
    args = ap.parse_args()

    cams = ["CAM2", "CAM3"] if args.cam == "both" else [args.cam]
    Xs, grps, cids, tmids, curves = [], [], [], [], {}
    for cam in cams:
        X, g, c, t, cv = load_with_curves(cam)
        Xs.append(X); grps.append(g); cids.append(c); tmids.append(t); curves.update(cv)
    X = np.concatenate(Xs); groups = np.concatenate(grps)
    cid_arr = np.concatenate(cids); tmid = np.concatenate(tmids)

    lags = np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step)
    fit = ct.ridge_fit(args.alpha)
    r2s, maes = [], []
    for lag in lags:
        y = targets_at_lag(args.target, cid_arr, tmid, curves, lag, args.window_s)
        pred, _ = ct.oof(fit, X, y, groups, args.n_splits)
        r2s.append(r2_score(y, pred)); maes.append(mean_absolute_error(y, pred))
    r2s, maes = np.asarray(r2s), np.asarray(maes)

    k = int(np.argmax(r2s))
    k0 = int(np.argmin(np.abs(lags)))                        # lag = 0 (current pipeline)
    unit = "g/s" if args.target == "flow" else "g"
    print(f"\n=== lag sweep [{args.target}, {args.cam}] "
          f"{len(tmid)} windows, {len(set(groups))} trials, ridge a={args.alpha} ===")
    print(f"  {'lag(s)':>7} {'R2':>8} {'MAE':>9}")
    for lag, r2, mae in zip(lags, r2s, maes):
        mark = "  <- best" if lag == lags[k] else ("  (current, lag=0)" if lag == lags[k0] else "")
        print(f"  {lag:>7.2f} {r2:>8.3f} {mae:>7.2f}{unit}{mark}")
    print(f"\n  best lag = {lags[k]:+.2f}s  R2={r2s[k]:.3f} (vs lag0 R2={r2s[k0]:.3f}, "
          f"delta={r2s[k]-r2s[k0]:+.3f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lags, r2s, "-o", ms=4, color="steelblue")
    ax.axvline(0, color="gray", ls=":", lw=1, label="lag=0 (current)")
    ax.axvline(lags[k], color="crimson", ls="--", lw=1.2, label=f"best {lags[k]:+.2f}s")
    ax.set_xlabel("target lag tau (s)  [weight sampled tau later]")
    ax.set_ylabel("held-out OOF R²")
    ax.set_title(f"water-transit lag — {args.target}, {args.cam}")
    ax.legend(); fig.tight_layout()
    out = HERE / f"qc_lag_sweep_{args.target}_{args.cam}.png"
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

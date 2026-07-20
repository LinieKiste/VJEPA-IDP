"""Probe frozen V-JEPA 2 mean-pool features for pouring volume/flow on the own-lab
clips, and compare against baselines. Grouped by TRIAL (clips from one trial share
scene + container, so trial-level held-out folds prevent appearance leakage).

Targets (real GT, grams):
  volume = cumulative poured mass at the window centre.
  flow   = poured-mass rate over the window (g/s).

Methods compared (same folds, same target, OOF predictions). To keep it honest the
APPEARANCE methods (vjepa, motion) get no clock, and a separate time baseline shows how
much of each target is trivially temporal:
  vjepa        — frozen V-JEPA 2 mean-pool (1024-d) -> ridge          [the probe, appearance]
  motion       — frame-diff energy (1-d) -> ridge                     [naive appearance baseline]
  time         — window-centre time (1-d) -> ridge                    [trivial temporal floor]
  mean         — predict the train-fold mean                          [absolute floor]
  vjepa_shuf   — V-JEPA with train labels permuted                    [null: is it real signal?]

"Works at all" = vjepa R^2 well above 0 and above the motion/time baselines; shuffle ~0 or below.

Usage:
    .venv/bin/python pour_probe/clips_train.py --target volume --cam CAM2
    .venv/bin/python pour_probe/clips_train.py --target flow --cam both
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import decord
import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "datasets" / "pouring_processed" / "clips"
FEATURES_DIR = Path(os.environ.get("POUR_CLIPS_FEATURES_DIR",
                                   "/home/casimir/.cache/pour_probe/clips_feats"))
MOTION_CACHE = FEATURES_DIR / "motion.npz"


def load(cam, target):
    """Stack all windows for one camera: X (n,1024), y, groups(trial), clip ids, tmid, window_s."""
    files = sorted((FEATURES_DIR / cam).glob("*.npz"))
    X, y, grp, cids, tmid = [], [], [], [], []
    for f in files:
        a = np.load(f, allow_pickle=True)
        n = len(a["emb"])
        X.append(a["emb"].astype(np.float32)); y.append(a[target])
        grp.extend([str(a["trial_id"])] * n); cids.extend([str(a["clip_id"])] * n)
        tmid.append(a["tmid"])
    return (np.concatenate(X), np.concatenate(y).astype(np.float32),
            np.asarray(grp), np.asarray(cids), np.concatenate(tmid))


def load_both(target):
    Xs, ys, gs, cs, ts = [], [], [], [], []
    for cam in ("CAM2", "CAM3"):
        X, y, g, c, t = load(cam, target)
        Xs.append(X); ys.append(y); gs.append(g); cs.append(c); ts.append(t)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(cs), np.concatenate(ts))


def motion_features(cam):
    """[frame-diff energy at window centre, window centre time] per window — a naive
    'is liquid moving / how far into the pour' baseline. Rows align with load(cam)
    (both iterate sorted npz files). Cached (decodes video, so slow first time)."""
    if MOTION_CACHE.exists():
        c = np.load(MOTION_CACHE, allow_pickle=True)
        if cam in c.files:
            return c[cam]
    rows = []
    for f in sorted((FEATURES_DIR / cam).glob("*.npz")):
        a = np.load(f, allow_pickle=True)
        cid, tmid = str(a["clip_id"]), a["tmid"]
        vr = decord.VideoReader(str(CLIPS / cam / f"{cid}.mp4"))
        fps = float(vr.get_avg_fps())
        idx = np.linspace(0, len(vr) - 1, min(len(vr), 48)).astype(int)
        fr = vr.get_batch(idx).asnumpy().astype(np.float32).mean(-1)  # grayscale
        fr = np.stack([cv2.resize(g, (64, 64)) for g in fr])
        diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))          # per-step motion energy
        energy = np.interp(tmid, (idx / fps)[1:], diff)               # motion at window centre
        rows.append(energy[:, None])                                  # appearance only (no clock)
    ordered = np.concatenate(rows)
    store = dict(np.load(MOTION_CACHE, allow_pickle=True)) if MOTION_CACHE.exists() else {}
    store[cam] = ordered
    np.savez(MOTION_CACHE, **store)
    return ordered


def oof(fit_fn, X, y, groups, n_splits, rng=None):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y), np.float32)
    base = np.zeros(len(y), np.float32)
    for tr, te in gkf.split(X, y, groups):
        ytr = y[tr]
        if rng is not None:                      # shuffle control: permute train labels
            ytr = ytr[rng.permutation(len(ytr))]
        pred[te] = fit_fn(X[tr], ytr, X[te])
        base[te] = y[tr].mean()
    return pred, base


def ridge_fit(alpha):
    def f(Xtr, ytr, Xte):
        m, s = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        r = Ridge(alpha=alpha).fit((Xtr - m) / s, ytr)
        return r.predict((Xte - m) / s)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["volume", "flow"], default="volume")
    ap.add_argument("--cam", choices=["CAM2", "CAM3", "both"], default="CAM2")
    ap.add_argument("--alpha", type=float, default=100.0, help="ridge L2 (1024-d, ~2k samples)")
    ap.add_argument("--n_splits", type=int, default=6)
    args = ap.parse_args()

    if args.cam == "both":
        X, y, groups, cids, tmid = load_both(args.target)
    else:
        X, y, groups, cids, tmid = load(args.cam, args.target)
    n_trials = len(set(groups.tolist()))
    rng = np.random.default_rng(0)

    results = {}
    # V-JEPA probe
    pred, base = oof(ridge_fit(args.alpha), X, y, groups, args.n_splits)
    results["vjepa"] = pred
    # shuffle null
    pred_s, _ = oof(ridge_fit(args.alpha), X, y, groups, args.n_splits, rng=rng)
    results["vjepa_shuf"] = pred_s
    # appearance baseline: frame-diff motion energy (no clock)
    if args.cam == "both":
        M = np.concatenate([motion_features("CAM2"), motion_features("CAM3")])
    else:
        M = motion_features(args.cam)
    results["motion"], _ = oof(ridge_fit(1.0), M, y, groups, args.n_splits)
    # temporal floors:
    #   time      = raw window-centre time, linear (shows raw time alone is ~nothing)
    #   time_prof = time normalized to [0,1] WITHIN each clip + degree-4 poly = the
    #               average pour PROFILE prior. This is a STRONG baseline but it
    #               cheats: normalizing needs the clip's start/end (the pour
    #               segmentation), which a real stream doesn't have, and it can
    #               only predict the mean profile (can't tell a fast pour from a
    #               slow one at the same phase). V-JEPA needs no boundaries.
    from sklearn.preprocessing import PolynomialFeatures
    results["time"], _ = oof(ridge_fit(1.0), tmid[:, None], y, groups, args.n_splits)
    tnorm = np.zeros_like(tmid)
    for c in set(cids.tolist()):
        m = cids == c
        lo, hi = tmid[m].min(), tmid[m].max()
        tnorm[m] = (tmid[m] - lo) / max(hi - lo, 1e-6)
    tpoly = PolynomialFeatures(4).fit_transform(tnorm[:, None])
    results["time_prof"], _ = oof(ridge_fit(1.0), tpoly, y, groups, args.n_splits)
    results["mean"] = base

    ystd = float(y.std())
    base_mae = mean_absolute_error(y, base)
    rows = {}
    for name, p in results.items():
        rows[name] = {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p),
                      "mae_vs_base": mean_absolute_error(y, p) / base_mae}

    import mlflow_util; mlflow_util.setup("pour_probe_clips")
    with mlflow.start_run(run_name=f"{args.target}_{args.cam}"):
        mlflow.log_params({"target": args.target, "cam": args.cam, "alpha": args.alpha,
                           "n_splits": args.n_splits, "n_trials": n_trials,
                           "n_windows": len(y), "target_std_g": ystd})
        for name, r in rows.items():
            for k, v in r.items():
                mlflow.log_metric(f"{name}_{k}", v)

    unit = "g" if args.target == "volume" else "g/s"
    print(f"\n=== {args.target} [{args.cam}]  {len(y)} windows / {n_trials} trials "
          f"| target std {ystd:.1f} {unit} | predict-mean MAE {base_mae:.1f} {unit} ===")
    print(f"  {'method':<12} {'R2':>7} {'MAE':>9} {'MAE/base':>9}")
    for name in ["vjepa", "motion", "time_prof", "time", "mean", "vjepa_shuf"]:
        r = rows[name]
        print(f"  {name:<12} {r['r2']:>7.3f} {r['mae']:>7.1f}{unit:<2} {r['mae_vs_base']:>9.2f}")


if __name__ == "__main__":
    main()

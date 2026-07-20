"""ORACLE-CONTAINER diagnostic (go/no-go for a container-size model).

Question: on our own-lab clips, does KNOWING the target container (the vessel being
filled -> its physical size) improve held-out ABSOLUTE VOLUME prediction over what the
frozen V-JEPA features (and the clock) already give? If oracle container identity barely
helps, a NOISY predicted size certainly won't, and a whole container-size dataset+model
isn't worth building. If it helps a lot, the direction is validated.

Same protocol as clips_train.py: window-level OOF, GroupKFold by TRIAL (clips of one
trial share scene+container, so trial-held-out folds prevent appearance leakage). Each
container appears across many trials, so with the container feature the model learns a
per-container mapping in train and applies it to held-out trials of the same container
= a fair oracle. We report both window-level R2/MAE and the thing you actually care
about, per-clip FINAL poured mass (the "scale" number).

Feature sets (ridge, standardized):
  vjepa            frozen V-JEPA mean-pool (1024)         [current probe]
  cont             container one-hot (4)  ALONE           [what identity alone explains]
  vjepa+cont       V-JEPA + container offset              [additive oracle on top of V-JEPA]
  time             raw window-centre time (1)             [volume is clock-dominated]
  time+cont        clock + per-container offset           [additive oracle on the clock]
  timeXcont        per-container intercept + time slope   [MULTIPLICATIVE oracle: size rescales the clock]
  vjepa+time+cont  everything                             [ceiling]

Usage: .venv/bin/python pour_probe/clips_oracle_container.py --target volume --cam CAM2
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

FEAT = Path(os.environ.get("POUR_CLIPS_FEATURES_DIR",
                           "/home/casimir/.cache/pour_probe/clips_feats"))


def load(cams, target):
    X, y, grp, cid, tmid, cont = [], [], [], [], [], []
    for cam in cams:
        for f in sorted((FEAT / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            n = len(a["emb"])
            X.append(a["emb"].astype(np.float32))
            y.append(a[target].astype(np.float32))
            grp.extend([str(a["trial_id"])] * n)
            cid.extend([f"{cam}:{a['clip_id']}"] * n)          # cam-qualified so both-cam clips don't collide
            tmid.append(a["tmid"])
            cont.extend([str(a["target_obj"])] * n)
    return (np.concatenate(X), np.concatenate(y), np.asarray(grp),
            np.asarray(cid), np.concatenate(tmid), np.asarray(cont))


def onehot(cont):
    cats = sorted(set(cont.tolist()))
    return np.stack([(cont == c).astype(np.float32) for c in cats], 1), cats


def ridge_fit(alpha):
    def f(Xtr, ytr, Xte):
        m = Xtr.mean(0, keepdims=True)
        s = Xtr.std(0, keepdims=True); s[s < 1e-8] = 1.0     # keep absent-container cols ~0, no blow-up
        r = Ridge(alpha=alpha).fit((Xtr - m) / s, ytr)
        return r.predict((Xte - m) / s)
    return f


def oof(F, alpha, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y), np.float32)
    fit = ridge_fit(alpha)
    for tr, te in gkf.split(F, y, groups):
        pred[te] = fit(F[tr], y[tr], F[te])
    return pred


def per_clip_final(cid, tmid, y, pred):
    """Per clip: value at the LAST window (max tmid) = final cumulative poured mass."""
    gt, pr = [], []
    for c in sorted(set(cid.tolist())):
        m = cid == c
        i = int(np.argmax(tmid[m]))
        gt.append(float(y[m][i])); pr.append(float(pred[m][i]))
    return np.asarray(gt), np.asarray(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["volume", "flow"], default="volume")
    ap.add_argument("--cam", choices=["CAM2", "CAM3", "both"], default="CAM2")
    ap.add_argument("--n_splits", type=int, default=6)
    args = ap.parse_args()
    cams = ("CAM2", "CAM3") if args.cam == "both" else (args.cam,)

    X, y, groups, cid, tmid, cont = load(cams, args.target)
    oh, cats = onehot(cont)
    t = tmid[:, None]
    tX = t * oh                                              # per-container time interaction
    n_trials = len(set(groups.tolist()))
    n_clips = len(set(cid.tolist()))

    sets = {
        "vjepa":           (X, 100.0),
        "cont":            (oh, 1.0),
        "vjepa+cont":      (np.hstack([X, oh]), 100.0),
        "time":            (t, 1.0),
        "time+cont":       (np.hstack([t, oh]), 1.0),
        "timeXcont":       (np.hstack([oh, tX]), 1.0),
        "vjepa+time+cont": (np.hstack([X, t, oh]), 100.0),
    }

    base = np.full(len(y), y.mean(), np.float32)             # note: OOF mean below for honest floor
    base_pred = oof(np.zeros((len(y), 1), np.float32), 1e12, y, groups, args.n_splits)  # ~predict train mean
    base_mae = mean_absolute_error(y, base_pred)
    unit = "g" if args.target == "volume" else "g/s"

    print(f"\n=== ORACLE-CONTAINER | {args.target} [{args.cam}] | {len(y)} windows / "
          f"{n_clips} clips / {n_trials} trials | containers {cats} ===")
    print(f"    target std {y.std():.1f}{unit}  predict-mean MAE {base_mae:.1f}{unit}\n")
    print(f"  {'featureset':<16} {'winR2':>7} {'winMAE':>8}   {'clipFinalR2':>11} "
          f"{'clipFinalMAE':>12} {'slope':>6}")
    for name, (F, alpha) in sets.items():
        pred = oof(F, alpha, y, groups, args.n_splits)
        gt_f, pr_f = per_clip_final(cid, tmid, y, pred)
        slope = float(np.polyfit(gt_f, pr_f, 1)[0])
        print(f"  {name:<16} {r2_score(y, pred):>7.3f} {mean_absolute_error(y, pred):>6.1f}{unit:<2}"
              f"   {r2_score(gt_f, pr_f):>11.3f} {mean_absolute_error(gt_f, pr_f):>10.1f}{unit:<2} "
              f"{slope:>6.2f}")
    print(f"\n  clipFinalMAE = per-clip final poured-mass error (THE scale number); "
          f"slope=1.0 means no compression.")


if __name__ == "__main__":
    main()

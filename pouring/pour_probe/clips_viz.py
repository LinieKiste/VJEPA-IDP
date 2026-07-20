"""Visualize the V-JEPA flow/volume probe: OOF predicted-vs-true scatter + per-clip
predicted-vs-GT curves. Reuses clips_train's loading + GroupKFold OOF."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

import clips_train as ct

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["volume", "flow"], default="flow")
    ap.add_argument("--cam", default="CAM2")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--n_splits", type=int, default=6)
    args = ap.parse_args()

    X, y, groups, cids, tmid = ct.load(args.cam, args.target)
    pred, _ = ct.oof(ct.ridge_fit(args.alpha), X, y, groups, args.n_splits)
    r2 = r2_score(y, pred)
    unit = "g" if args.target == "volume" else "g/s"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # scatter, colored by trial
    trials = sorted(set(groups.tolist()), key=int)
    cmap = plt.cm.turbo(np.linspace(0, 1, len(trials)))
    for t, col in zip(trials, cmap):
        m = groups == t
        ax1.scatter(y[m], pred[m], s=9, color=col, alpha=0.6)
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlabel(f"true {args.target} ({unit})")
    ax1.set_ylabel(f"predicted {args.target} ({unit})")
    ax1.set_title(f"OOF pred vs true — {args.target} [{args.cam}]  R²={r2:.3f}  (color = trial)")

    # per-clip curves: pick 6 clips spanning the weight range
    uniq_clips = list(dict.fromkeys(cids.tolist()))
    finals = {c: y[cids == c].max() if args.target == "volume" else
              ct.load(args.cam, "volume")[1][cids == c].max() for c in uniq_clips}
    pick = [uniq_clips[i] for i in np.linspace(0, len(uniq_clips) - 1, 6).astype(int)]
    for c in pick:
        m = cids == c
        order = np.argsort(tmid[m])
        tt = tmid[m][order]
        ax2.plot(tt, y[m][order], "-", lw=2, alpha=0.5, label=f"clip {c} GT")
        ax2.plot(tt, pred[m][order], "--", lw=1.5,
                 color=ax2.lines[-1].get_color())
    ax2.set_xlabel("window centre time (s)")
    ax2.set_ylabel(f"{args.target} ({unit})")
    ax2.set_title(f"per-clip: GT (solid) vs predicted (dashed)")
    ax2.legend(fontsize=8, ncol=2)

    out = HERE / f"qc_probe_{args.target}_{args.cam}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"wrote {out}  (R²={r2:.3f})")


if __name__ == "__main__":
    main()

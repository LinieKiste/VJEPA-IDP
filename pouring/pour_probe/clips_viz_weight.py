"""Sample results for the lag-corrected flow probe: GT weight vs weight RECONSTRUCTED
by integrating the probe's predicted flow rate.

The attentive probe predicts instantaneous flow (g/s). Integrating that over time
gives cumulative poured mass, which we overlay on the human-verified GT weight curve
for a handful of held-out (val-trial) clips. Because the probe was trained with the
water-transit lag (flow(t) estimates dW/dt at t+lag), the reconstructed weight is
placed at (window-centre + lag) so it aligns with GT on the same clock. The endpoint
of the reconstruction = predicted total poured mass, compared to the GT total (a
lag-independent scalar).

Usage: .venv/bin/python pour_probe/clips_viz_weight.py --cam CAM2 --n 6
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import clips_train_attn as ca
from _encoder import load_encoder
from head import build_head

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="CAM2", choices=["CAM2", "CAM3"])
    ap.add_argument("--lag_s", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    ckpt = args.ckpt or (ca.FRAMES_DIR.parent / f"attn_flow_both_lag{args.lag_s:g}_best.pt")

    clips = ca.load_clips(args.cam)
    cams = {args.cam: clips}
    wins = ca.build_windows(cams, 1.0, 0.5, 16, args.lag_s)
    tr = [w for w in wins if w["trial"] not in ca.VAL_TRIALS]
    va = [w for w in wins if w["trial"] in ca.VAL_TRIALS]
    ytr = np.asarray([w["flow"] for w in tr], np.float32)
    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

    enc = load_encoder(img_size=256, num_frames=16, device="cuda")
    head = build_head(1).to("cuda")
    head.load_state_dict(torch.load(ckpt))
    head.eval()
    mean, std = ca.MEAN.to("cuda"), ca.STD.to("cuda")
    print(f"loaded {ckpt}")

    r2, mae, preds, _ = ca.run_eval(enc, head, cams, va, "flow", 16, "cuda",
                                    mean, std, ymean, ystd)
    print(f"{args.cam} val: flow R²={r2:.3f} MAE={mae:.2f} g/s")

    by = defaultdict(list)
    for w, p in zip(va, preds):
        by[w["clip"]].append((w, float(p)))

    # pick n val clips spanning the range of final poured mass
    finals = {c: float(clips[c]["gt_w"][-1]) for c in by}
    order = sorted(by, key=lambda c: finals[c])
    idx = np.unique(np.linspace(0, len(order) - 1, args.n).round().astype(int))
    picks = [order[i] for i in idx]

    ncol = 3
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, c in zip(axes.flat, picks):
        ax.axis("on")
        items = sorted(by[c], key=lambda wp: (wp[0]["t0"] + wp[0]["t1"]) / 2)
        tc = np.array([(w["t0"] + w["t1"]) / 2 for w, _ in items])
        pf = np.clip(np.array([p for _, p in items]), 0, None)   # predicted flow g/s (>=0)
        ts = tc + args.lag_s                                     # undo training lag
        wpred = np.concatenate([[0.0], np.cumsum((pf[1:] + pf[:-1]) / 2 * np.diff(ts))])
        tpred = np.concatenate([[ts[0]], ts[1:]])
        gt_t, gt_w = clips[c]["gt_t"], clips[c]["gt_w"]

        ax.plot(gt_t, gt_w, "-", color="steelblue", lw=2.4, label="GT weight")
        ax.plot(tpred, wpred, "--", color="crimson", lw=1.8, label="pred (∫ flow)")
        ax.axhline(finals[c], color="steelblue", ls=":", lw=0.8, alpha=0.6)
        tr_id = clips[c]["trial"]
        ax.set_title(f"clip {c} (trial {tr_id})  GT {finals[c]:.0f} g  |  pred {wpred[-1]:.0f} g",
                     fontsize=9)
        ax.set_xlabel("time (s)"); ax.set_ylabel("poured mass (g)")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Lag-corrected V-JEPA 2 flow probe — GT weight vs integrated predicted flow "
                 f"({args.cam} held-out clips, R²={r2:.3f})", fontsize=12)
    fig.tight_layout()
    out = HERE / f"qc_weight_recon_{args.cam}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # lag-independent scalar: predicted total mass vs GT total, all val clips
    errs = []
    for c in by:
        items = sorted(by[c], key=lambda wp: (wp[0]["t0"] + wp[0]["t1"]) / 2)
        tc = np.array([(w["t0"] + w["t1"]) / 2 for w, _ in items])
        pf = np.clip(np.array([p for _, p in items]), 0, None)
        tot = float(np.trapz(pf, tc + args.lag_s))
        errs.append(abs(tot - finals[c]))
    print(f"predicted TOTAL poured mass vs GT (all {len(by)} val clips): "
          f"MAE {np.mean(errs):.1f} g, median {np.median(errs):.1f} g "
          f"(GT range {min(finals.values()):.0f}-{max(finals.values()):.0f} g)")


if __name__ == "__main__":
    main()

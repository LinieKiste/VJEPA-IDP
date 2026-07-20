"""Visualize the trained attentive probe: training curve (mlflow) + val predicted-vs-true
scatter + per-clip predicted-vs-GT flow curves (from the saved best checkpoint)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import r2_score

import clips_train_attn as ca
from _encoder import load_encoder
from head import build_head

HERE = Path(__file__).resolve().parent


def training_curve(target, cam):
    """(steps, train_loss), (val_steps, val_r2) from the latest matching mlflow run."""
    c = mlflow.tracking.MlflowClient()
    exp = c.get_experiment_by_name("pour_probe_clips_attn")
    runs = c.search_runs([exp.experiment_id], f"attributes.run_name = 'attn_{target}_{cam}'",
                         order_by=["attributes.start_time DESC"], max_results=1)
    rid = runs[0].info.run_id
    tl = c.get_metric_history(rid, "train_loss")
    vr = c.get_metric_history(rid, "val_r2")
    return ([m.step for m in tl], [m.value for m in tl],
            [m.step for m in vr], [m.value for m in vr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="flow")
    ap.add_argument("--cam", default="CAM2")
    args = ap.parse_args()

    cams = {args.cam: ca.load_clips(args.cam)}
    wins = ca.build_windows(cams, 1.0, 0.5, 16)
    va = [w for w in wins if w["trial"] in ca.VAL_TRIALS]
    ytr = np.asarray([w[args.target] for w in wins if w["trial"] not in ca.VAL_TRIALS])
    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

    enc = load_encoder(img_size=256, num_frames=16, device="cuda")
    head = build_head(1).to("cuda")
    head.load_state_dict(torch.load(ca.FRAMES_DIR.parent / f"attn_{args.target}_{args.cam}_best.pt"))
    mean, std = ca.MEAN.to("cuda"), ca.STD.to("cuda")
    r2, mae, preds, ys = ca.run_eval(enc, head, cams, va, args.target, 16, "cuda",
                                     mean, std, ymean, ystd)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(19, 5.3))

    # training curve
    ls, lv, vs, vv = training_curve(args.target, args.cam)
    ax0.plot(ls, lv, color="steelblue", lw=0.8, label="train loss (SmoothL1, z)")
    ax0.set_xlabel("step"); ax0.set_ylabel("train loss", color="steelblue")
    axb = ax0.twinx()
    axb.plot(vs, vv, color="crimson", lw=1.6, marker="o", ms=3, label="val R²")
    axb.set_ylabel("val R²", color="crimson"); axb.set_ylim(-0.2, 1.0)
    ax0.set_title(f"training — attentive probe [{args.target}]")

    # scatter (val), color by trial
    unit = "g/s" if args.target == "flow" else "g"
    vt = np.asarray([w["trial"] for w in va])
    for t in sorted(set(vt.tolist())):
        m = vt == t
        ax1.scatter(ys[m], preds[m], s=12, alpha=0.6, label=f"trial {t}")
    lim = [min(ys.min(), preds.min()), max(ys.max(), preds.max())]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlabel(f"true {args.target} ({unit})"); ax1.set_ylabel(f"predicted ({unit})")
    ax1.set_title(f"held-out val  R²={r2:.3f}  MAE={mae:.2f}{unit}"); ax1.legend(fontsize=8)

    # per-clip curves (val clips)
    va_clips = list(dict.fromkeys(w["clip"] for w in va))
    tmid = np.asarray([(w["t0"] + w["t1"]) / 2 for w in va])
    cids = np.asarray([w["clip"] for w in va])
    for c in va_clips[:8]:
        m = cids == c; o = np.argsort(tmid[m])
        ax2.plot(tmid[m][o], ys[m][o], "-", lw=2, alpha=0.5)
        ax2.plot(tmid[m][o], preds[m][o], "--", lw=1.4, color=ax2.lines[-1].get_color())
    ax2.set_xlabel("window centre time (s)"); ax2.set_ylabel(f"{args.target} ({unit})")
    ax2.set_title("val clips: GT (solid) vs predicted (dashed)")

    out = HERE / f"qc_attn_{args.target}_{args.cam}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"wrote {out}  (val R²={r2:.3f}, MAE={mae:.2f}{unit})")


if __name__ == "__main__":
    main()

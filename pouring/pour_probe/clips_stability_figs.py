"""Figures for the 1-frame-stride stability check (clips_stability.py).

Three panels, because "the curve looks smooth" is not evidence:
  A  dense stride-1 predicted curves on held-out clips, against the 0.5 s stride actually
     used for training/eval -- the visual claim.
  B  mean |pred(t+k) - pred(t)| vs shift k, with the ground truth and a SHUFFLED control --
     the quantitative claim. Stability = starts near zero and rises smoothly; no stability
     = flat at the shuffled level from k=1.
  C  distribution of the 1-frame change against the model's own output spread -- the scale
     claim, i.e. how big the jitter is relative to what the model is trying to resolve.

    .venv/bin/python pouring/pour_probe/clips_stability_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clips_stability import load, shift_curve, stats

OUT = Path(__file__).resolve().parents[2] / "datasets/eval/figs"

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_curves(res, names=None):
    """Dense stride-1 prediction vs the coarse stride the probe is normally run at."""
    names = names or sorted(res, key=lambda k: -len(res[k]["pred"]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), facecolor=SURFACE)
    for ax, n in zip(axes.ravel(), names):
        style(ax)
        v = res[n]
        fps = float(v["fps"])
        step = max(1, int(round(0.5 * fps)))          # the 0.5 s stride used in training
        ax.plot(v["tmid"], v["gt"], color=MUTED, lw=2.4, zorder=2, label="ground truth")
        ax.plot(v["tmid"], v["pred"], color=BLUE, lw=1.6, zorder=4,
                label="prediction, 1-frame stride")
        ax.plot(v["tmid"][::step], v["pred"][::step], "o", ms=5, color=ORANGE,
                mec=SURFACE, mew=1.4, zorder=5, label="0.5 s stride (normal eval)")
        ax.set_title(f"{n}   ({len(v['pred'])} windows)", fontsize=9.5, color=INK,
                     loc="left", fontweight="bold", pad=4)
        ax.set_xlim(v["tmid"][0], v["tmid"][-1])
    for ax in axes[-1]:
        ax.set_xlabel("window centre (s)", fontsize=8.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("flow (g/s)", fontsize=8.5, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Sliding-window inference at 1-frame stride, held-out clips",
                 fontsize=13, color=INK, fontweight="bold", y=1.135)
    fig.text(0.5, 1.088, "consecutive windows share ~29/30 of their frames; the dense curve "
             "traces the same shape as the coarse one, without jitter between samples",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = OUT / "fig_stability_curves.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


def fig_shift_and_hist(res, s, kmax=30):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.0), facecolor=SURFACE)
    style(a1); style(a2)

    k = np.arange(1, kmax + 1)
    fps = float(next(iter(res.values()))["fps"])
    cp = shift_curve(res, kmax, "pred")
    cg = shift_curve(res, kmax, "gt")
    csh = shift_curve(res, kmax, "pred", shuffled=True)

    a1.axhline(csh[0], color=ORANGE, lw=2, ls=(0, (5, 3)), zorder=3,
               label=f"shuffled control ({csh[0]:.1f} g/s) — no stability")
    a1.plot(k, cp, color=BLUE, lw=2.4, zorder=5, label="model prediction")
    a1.plot(k, cg, color=MUTED, lw=2, zorder=4, label="ground truth")
    a1.plot([1], [cp[0]], "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=6)
    ytop = max(csh[0], cp.max()) * 1.18
    a1.annotate(f"1 frame → {cp[0]:.2f} g/s\n({100*cp[0]/s['pred_range']:.1f}% of output range,\n"
                f"{cp[0]/csh[0]:.0%} of the shuffled control)",
                xy=(1, cp[0]), xytext=(4.5, ytop * 0.60), textcoords="data",
                fontsize=9, color=BLUE, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2,
                                shrinkA=0, shrinkB=6))
    a1.set_xlabel(f"input shift (frames;  1 frame ≈ {1000/fps:.0f} ms)", fontsize=9.5, color=INK2)
    a1.set_ylabel("mean |change in prediction| (g/s)", fontsize=9.5, color=INK2)
    a1.set_title("Prediction change grows smoothly with input shift",
                 fontsize=11, color=INK, fontweight="bold", loc="left")
    a1.set_xlim(0, kmax); a1.set_ylim(0, max(csh[0], cp.max()) * 1.18)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    dp = np.concatenate([np.abs(np.diff(v["pred"])) for v in res.values()])
    allp = np.concatenate([v["pred"] for v in res.values()])
    bins = np.linspace(0, np.percentile(dp, 99), 45)
    a2.hist(dp, bins=bins, color=BLUE, zorder=3)
    for q, lab, col in ((np.median(dp), "median", INK2),
                        (np.percentile(dp, 95), "p95", ORANGE)):
        a2.axvline(q, color=col, lw=1.8, ls=(0, (4, 3)), zorder=4)
        a2.annotate(f"{lab} {q:.2f}", (q, a2.get_ylim()[1] * 0.92), xytext=(5, 0),
                    textcoords="offset points", fontsize=9, color=col, fontweight="bold")
    a2.axvline(allp.std(), color=AQUA, lw=2, zorder=5)
    a2.annotate(f"model's own output sd\n{allp.std():.1f} g/s", (allp.std(), a2.get_ylim()[1] * 0.55),
                xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=9,
                color=AQUA, fontweight="bold")
    a2.set_xlabel("|change in prediction| for a 1-frame shift (g/s)", fontsize=9.5, color=INK2)
    a2.set_ylabel("windows", fontsize=9.5, color=INK2)
    a2.set_title("One frame moves the prediction far less than the signal spans",
                 fontsize=11, color=INK, fontweight="bold", loc="left")
    a2.set_xlim(0, bins[-1])

    fig.suptitle(f"Representation stability: {s['n_windows']} overlapping windows, "
                 f"{s['n_clips']} held-out clips",
                 fontsize=13, color=INK, fontweight="bold", y=1.04)
    fig.tight_layout()
    p = OUT / "fig_stability_summary.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    return cp, cg, csh


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = load("")
    s = stats(res)
    fig_curves(res)
    cp, cg, csh = fig_shift_and_hist(res, s)
    print(f"\nshift curve (g/s): k=1 {cp[0]:.2f}  k=5 {cp[4]:.2f}  k=15 {cp[14]:.2f}  "
          f"k=30 {cp[29]:.2f}   shuffled {csh[0]:.2f}")
    print(f"model/shuffled at k=1: {cp[0]/csh[0]:.3f}  (lower = more stable)")

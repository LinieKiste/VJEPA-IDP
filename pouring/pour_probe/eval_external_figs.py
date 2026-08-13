"""Figures for the external (eval) pour set: predicted flow curves, integrated volume
curves, and a summary of predicted vs ground-truth totals.

There is no per-frame ground truth for this set -- only ONE final volume per video -- so
the curves are shown as predictions with a fold-spread band and the single GT number is
drawn as a target. The point is qualitative: does the probe produce a plausible pour
shape, and does its integral land anywhere near the one number we can check?

    .venv/bin/python pouring/pour_probe/eval_external_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_external import load_cached, load_gt

OUT = Path(__file__).resolve().parents[2] / "datasets/eval/figs"

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
# first three categorical slots -- the only three that validate on all-pairs
SCENE_COLOR = {"kettle -> stockpot": "#2a78d6",
               "jug -> cup on tray": "#eb6834",
               "outdoor table (wide)": "#1baf7a"}
SCENE = {"IMG_0866": "kettle -> stockpot", "IMG_0868": "kettle -> stockpot",
         "IMG_0872": "kettle -> stockpot",
         "IMG_0875": "jug -> cup on tray", "IMG_0877": "jug -> cup on tray",
         "IMG_0879": "outdoor table (wide)", "IMG_0880": "outdoor table (wide)",
         "IMG_0881": "outdoor table (wide)", "IMG_0882": "outdoor table (wide)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def cumulative(t, f):
    """Trapezoid running integral of a flow curve -> grams, starting at 0."""
    d = np.diff(t)
    return np.concatenate([[0.0], np.cumsum(d * (f[1:] + f[:-1]) / 2)])


def fig_curves(res, gt, kind):
    """3x3 small multiples. kind='flow' (g/s) or 'volume' (g, integrated)."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.2), facecolor=SURFACE)
    for ax, name in zip(axes.ravel(), sorted(res)):
        style(ax)
        col = SCENE_COLOR[SCENE[name]]
        t, P = res[name]["tmid"], res[name]["flow"]
        if kind == "flow":
            band_lo, band_hi = P.min(0), P.max(0)
            mid = P.mean(0)
        else:
            C = np.stack([cumulative(t, p) for p in P])
            band_lo, band_hi, mid = C.min(0), C.max(0), C.mean(0)
        ax.fill_between(t, band_lo, band_hi, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(t, mid, color=col, lw=2, zorder=3)
        if kind == "volume":
            g = gt[name]
            ax.axhline(g, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=4)
            ax.annotate(f"GT {g:.0f} g", (t[-1], g), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color=INK2, fontweight="bold")
            ax.plot([t[-1]], [mid[-1]], "o", ms=7, color=col, mec=SURFACE, mew=2, zorder=5)
            ax.annotate(f"{mid[-1]:.0f} g", (t[-1], mid[-1]), xytext=(-4, -11),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=8, color=col, fontweight="bold")
        else:
            ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left", fontweight="bold", pad=4)
        ax.set_xlim(0, max(t[-1], 1))
    for ax in axes[-1]:
        ax.set_xlabel("time in clip (s)", fontsize=8.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted flow (g/s)" if kind == "flow"
                      else "integrated volume (g)", fontsize=8.5, color=INK2)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=k) for k, c in SCENE_COLOR.items()]
    if kind == "volume":
        handles.append(plt.Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 3)),
                                  label="ground-truth final volume"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    ttl = ("Predicted instantaneous flow, frozen V-JEPA 2 probe on unseen kitchens"
           if kind == "flow" else
           "Integrated volume vs the one ground-truth number per video")
    sub = ("band = spread across the 4 trial-fold checkpoints; none saw this scene"
           if kind == "flow" else
           "curve = mean of 4 folds, band = their spread; dashed line = GT total")
    fig.suptitle(ttl, fontsize=13, color=INK, fontweight="bold", x=0.5, y=1.075)
    fig.text(0.5, 1.038, sub, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = OUT / f"fig_external_{kind}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


def fig_summary(res, gt):
    names = sorted(res)
    pred = np.array([cumulative(res[n]["tmid"], res[n]["flow"].mean(0))[-1] for n in names])
    true = np.array([gt[n] for n in names])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.2), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    style(a1); style(a2)

    hi = max(pred.max(), true.max()) * 1.12
    a1.plot([0, hi], [0, hi], color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a1.annotate("perfect", (hi * 0.93, hi * 0.93), fontsize=8, color=MUTED,
                rotation=45, ha="center", va="bottom")
    placed = []                       # crude label repel: flip side when points collide
    for n, p, t in zip(names, pred, true):
        c = SCENE_COLOR[SCENE[n]]
        a1.plot([t], [p], "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=4)
        dx, dy, ha = 7, -3, "left"
        for (pt, pp) in placed:
            if abs(t - pt) < hi * 0.07 and abs(p - pp) < hi * 0.05:
                dx, dy, ha = -8, 8, "right"
        a1.annotate(n.replace("IMG_", ""), (t, p), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=8, color=INK2)
        placed.append((t, p))
    a1.set_xlim(0, hi); a1.set_ylim(0, hi)
    a1.set_xlabel("ground-truth final volume (g)", fontsize=9.5, color=INK2)
    a1.set_ylabel("predicted total, integrated flow (g)", fontsize=9.5, color=INK2)
    a1.set_title("Predicted vs actual total", fontsize=11, color=INK,
                 fontweight="bold", loc="left")

    err = pred - true
    order = np.argsort(err)
    y = np.arange(len(names))
    a2.axvline(0, color=AXIS, lw=1.2, zorder=2)
    for i, k in enumerate(order):
        c = SCENE_COLOR[SCENE[names[k]]]
        a2.barh(i, err[k], height=0.62, color=c, zorder=3)
        off = 6 if err[k] >= 0 else -6
        a2.annotate(f"{err[k]:+.0f} g", (err[k], i), xytext=(off, 0),
                    textcoords="offset points", va="center",
                    ha="left" if err[k] >= 0 else "right",
                    fontsize=8.5, color=INK2, fontweight="bold")
    a2.set_yticks(y, [names[k].replace("IMG_", "") for k in order], fontsize=8.5)
    a2.set_xlabel("prediction error (g)   —   negative = under-predicted",
                  fontsize=9.5, color=INK2)
    a2.set_title("Per-video error", fontsize=11, color=INK, fontweight="bold", loc="left")
    a2.set_xlim(err.min() * 1.45 - 40, err.max() * 1.3 + 60)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, label=k)
               for k, c in SCENE_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    mae = np.abs(err).mean()
    fig.suptitle(f"Out-of-domain: MAE {mae:.0f} g on 9 unseen pours "
                 f"(in-domain held-out MAE is 23 g)",
                 fontsize=13, color=INK, fontweight="bold", y=1.085)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = OUT / "fig_external_summary.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)
    return names, pred, true


def fig_diagnosis(res, gt):
    """Why the indoor scenes over-predict: the probe never saw a NON-pouring frame in
    training (every clip is tightly cut to a pour), so it has no zero. Its per-video
    floor is a phantom flow that the integral accumulates over the whole video."""
    names = sorted(res)
    true = np.array([gt[n] for n in names])
    floor = np.array([np.percentile(res[n]["flow"].mean(0), 10) for n in names])
    raw = np.array([cumulative(res[n]["tmid"], res[n]["flow"].mean(0))[-1] for n in names])
    corr = np.array([cumulative(res[n]["tmid"],
                                np.clip(res[n]["flow"].mean(0) - f, 0, None))[-1]
                     for n, f in zip(names, floor)])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE)
    style(a1); style(a2)

    order = np.argsort(-floor)
    for i, k in enumerate(order):
        a1.barh(i, floor[k], height=0.62, color=SCENE_COLOR[SCENE[names[k]]], zorder=3)
        a1.annotate(f"{floor[k]:.0f}", (floor[k], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2, fontweight="bold")
    a1.set_yticks(range(len(names)), [names[k].replace("IMG_", "") for k in order],
                  fontsize=8.5)
    a1.set_xlabel("resting predicted flow, 10th percentile (g/s)", fontsize=9.5, color=INK2)
    a1.set_title("Phantom flow when nothing is pouring", fontsize=11, color=INK,
                 fontweight="bold", loc="left")
    a1.set_xlim(0, floor.max() * 1.18)
    a1.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=8, color=c, label=k)
                       for k, c in SCENE_COLOR.items()],
              frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    hi = max(raw.max(), true.max()) * 1.1
    a2.plot([0, hi], [0, hi], color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a2.plot(true, raw, "o", ms=9, color="#eb6834", mec=SURFACE, mew=2, zorder=4,
            label=f"raw integral  (r = {np.corrcoef(raw, true)[0,1]:.2f})")
    a2.plot(true, corr, "D", ms=8, color="#2a78d6", mec=SURFACE, mew=2, zorder=5,
            label=f"minus resting floor  (r = {np.corrcoef(corr, true)[0,1]:.2f})")
    for t, r, c in zip(true, raw, corr):
        a2.plot([t, t], [r, c], color=MUTED, lw=1, zorder=3)
    a2.set_xlabel("ground-truth final volume (g)", fontsize=9.5, color=INK2)
    a2.set_ylabel("predicted total (g)", fontsize=9.5, color=INK2)
    a2.set_title("Removing the floor fixes the ranking, not the scale",
                 fontsize=11, color=INK, fontweight="bold", loc="left")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    a2.set_xlim(0, hi); a2.set_ylim(0, hi)

    fig.suptitle("Diagnosis: the probe has no concept of \"not pouring\"",
                 fontsize=13, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = OUT / "fig_external_diagnosis.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, gt = load_cached(), load_gt()
    fig_curves(res, gt, "flow")
    fig_curves(res, gt, "volume")
    names, pred, true = fig_summary(res, gt)
    fig_diagnosis(res, gt)
    err = pred - true
    print(f"\nMAE {np.abs(err).mean():.1f} g   medAE {np.median(np.abs(err)):.1f} g   "
          f"bias {err.mean():+.1f} g   corr(pred,true) {np.corrcoef(pred, true)[0,1]:+.3f}")
    m = np.array([SCENE[n] != "outdoor table (wide)" for n in names])
    print(f"  indoor only (n={m.sum()}): MAE {np.abs(err[m]).mean():.1f} g, "
          f"corr {np.corrcoef(pred[m], true[m])[0,1]:+.3f}")
    print(f"  outdoor only (n={(~m).sum()}): MAE {np.abs(err[~m]).mean():.1f} g, "
          f"bias {err[~m].mean():+.1f} g")

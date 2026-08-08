"""Bland-Altman plot for the deliverable: per-pour total mass from the attentive
V-JEPA 2 flow probe, against the scale.

    ../.venv/bin/python figs_src/bland_altman.py   ->  public/bland_altman.png

Inputs are the tracked bundle under ../presentation/data/ (the same
`headline_preds.npz` the interim deck's figures read), so this regenerates from a
bare clone.

Conventions, both settled in CLAUDE.md:
  * predicted total = trapezoid integral of the predicted flow curve extended to
    the CLIP BOUNDS. Integrating between window centres drops the first and last
    half-window and biases every total ~12 g low.
  * truth = the scale's final reading (`weight_g`), not the centre-window integral
    of the GT curve.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "presentation" / "data"
OUT = HERE.parent / "public" / "bland_altman.png"

BLUE = "#0065bd"        # accent / TUM blue
DARK = "#003359"        # dk2
ORANGE = "#e37222"      # accent5
INK = "#000000"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def de(x, nd=0):
    """German decimal comma, matching the rest of the deck."""
    return f"{x:+.{nd}f}".replace(".", ",") if nd else f"{x:+.0f}"


def totals():
    """Per-clip predicted vs measured poured mass, in grams."""
    d = np.load(DATA / "headline_preds.npz")
    man = pd.read_csv(DATA / "clips_manifest.csv").set_index("clip_id")
    pred, clip, tmid = d["flow_pred"].astype(float), d["flow_clip"], d["flow_tmid"]

    P, T = [], []
    for cid in np.unique(clip):
        m = clip == cid
        o = np.argsort(tmid[m])
        t, f = tmid[m][o], pred[m][o]
        dur = float(man.loc[int(cid), "duration_s"])
        P.append(np.trapezoid(np.concatenate([[f[0]], f, [f[-1]]]),
                              np.concatenate([[0.0], t, [dur]])))
        T.append(float(man.loc[int(cid), "weight_g"]))
    return np.asarray(P), np.asarray(T)


def main():
    pred, true = totals()
    mean, diff = (pred + true) / 2, pred - true
    bias, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(9.3, 3.9), dpi=300)

    xlim = (0, 1.14 * mean.max())   # right margin keeps the line labels off the points
    ax.axhspan(lo, hi, color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(bias, color=ORANGE, lw=2)
    for y in (lo, hi):
        ax.axhline(y, color=DARK, lw=1.4, ls=(0, (5, 4)))

    ax.scatter(mean, diff, s=26, facecolor=BLUE, edgecolor="white",
               linewidth=0.7, alpha=0.9, zorder=3)

    # bias label right, limits left -- the top-right corner is occupied by points
    ax.text(xlim[1] * 0.995, bias, f"Bias {de(bias, 1)} g", color=ORANGE,
            ha="right", va="bottom", fontsize=10, fontweight="bold")
    xl = xlim[1] * 0.008
    ax.text(xl, hi, f"+1,96 SD   {de(hi)} g", color=DARK, ha="left",
            va="bottom", fontsize=9.5)
    ax.text(xl, lo, f"−1,96 SD   {de(lo)} g", color=DARK, ha="left",
            va="top", fontsize=9.5)

    ax.set_xlabel("Mittelwert aus Waage und Vorhersage (g)")
    ax.set_ylabel("Vorhersage − Waage (g)")
    ax.set_xlim(*xlim)
    ax.set_ylim(min(lo, diff.min()) - 28, max(hi, diff.max()) + 28)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"n={len(diff)}  bias {bias:+.1f} g  LoA {lo:+.1f} .. {hi:+.1f} g  "
          f"MAE {np.abs(diff).mean():.1f} g  -> {OUT}")


if __name__ == "__main__":
    main()

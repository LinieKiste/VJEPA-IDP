#!/usr/bin/env python3
"""Summary figures for the IDP slide deck.

Numbers are transcribed from mlflow (sqlite:///../mlflow.db) and from the
analyses that were not logged as runs (lag sweep, calibration, oracle-container).
Palette = dataviz reference categorical slots (light mode).
"""
from __future__ import annotations

import csv
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[1]
HERE = Path(__file__).parent
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# QC figures produced by the experiment scripts, copied in so the deck is self-contained.
QC_SOURCES = {
    "qc_attn_flow_CAM2.png": "pouring/pour_probe/qc_attn_flow_CAM2.png",
    "qc_attn_map_transfer_flow.png": "pouring/pour_probe/qc_attn_map_transfer_flow.png",
    "qc_lag_sweep_flow_both.png": "pouring/pour_probe/qc_lag_sweep_flow_both.png",
    "qc_lag_sweep_volume_CAM2.png": "pouring/pour_probe/qc_lag_sweep_volume_CAM2.png",
    "qc_roi_crop_detector_v2.png": "pouring/pour_probe/qc_roi_crop_detector_v2.png",
    "qc_weight_recon_CAM2.png": "pouring/pour_probe/qc_weight_recon_CAM2.png",
    "trace_example.png": "pouring/clip_split/qc/trace_20260713_160041_383_GX011276.png",
}


def sync_assets():
    import shutil

    from PIL import Image

    dst = HERE / "assets"
    dst.mkdir(exist_ok=True)
    for name, rel in QC_SOURCES.items():
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, dst / name)

    # The full ROI QC sheet is 8 rows tall and unreadable on a slide: keep 3 rows.
    roi = dst / "qc_roi_crop_detector_v2.png"
    if roi.exists():
        im = Image.open(roi)
        im.crop((0, 0, im.width, int(im.height * 3 / 8))).save(
            dst / "roi_examples.png"
        )

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, VIOLET, RED = "#eda100", "#e87ba4", "#4a3aa7", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GREY = "#b9b8b3"
GRID = "#e6e5e1"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.size": 10,
        "grid.color": GRID,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def hbars(ax, labels, values, colors, fmt="{:.2f}", xlim=None, xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    for yi, v in zip(y, values):
        off = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(
            v + (off if v >= 0 else -off),
            yi,
            fmt.format(v),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9.5,
            color=INK,
        )
    return y


# ---------------------------------------------------------------- 1. dataset
def fig_dataset():
    rows = list(
        csv.DictReader(
            open(ROOT / "datasets/pouring_processed/clips/clips_manifest.csv")
        )
    )
    mass = np.array([float(r["weight_g"]) for r in rows])
    dur = np.array([float(r["duration_s"]) for r in rows])
    combo = collections.Counter(
        (r["source_obj"], r["target_obj"]) for r in rows
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))

    ax = axes[0]
    ax.hist(mass, bins=18, color=BLUE, zorder=3)
    ax.axvline(np.median(mass), color=ORANGE, lw=2, zorder=4)
    ax.text(
        0.97,
        0.94,
        f"median {np.median(mass):.0f} g",
        transform=ax.transAxes,
        ha="right",
        color=ORANGE,
        fontsize=9.5,
    )
    ax.set_xlabel("poured mass (g)")
    ax.set_ylabel("clips")
    ax.set_title(f"Poured mass  ({mass.min():.0f}–{mass.max():.0f} g)")

    ax = axes[1]
    ax.hist(dur, bins=18, color=BLUE, zorder=3)
    ax.axvline(np.median(dur), color=ORANGE, lw=2, zorder=4)
    ax.text(
        0.97,
        0.94,
        f"median {np.median(dur):.1f} s",
        transform=ax.transAxes,
        ha="right",
        color=ORANGE,
        fontsize=9.5,
    )
    ax.set_xlabel("clip duration (s)")
    ax.set_title(f"Duration  ({dur.min():.1f}–{dur.max():.1f} s)")

    ax = axes[2]
    items = combo.most_common()
    labels = [f"{s} → {t}" for (s, t), _ in items]
    vals = [n for _, n in items]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.68, color=BLUE, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("clips")
    ax.set_title("12 source → target combinations")
    ax.tick_params(axis="y", length=0)

    for ax in axes:
        ax.xaxis.grid(True, lw=0.8, zorder=0) if ax is axes[2] else ax.yaxis.grid(
            True, lw=0.8, zorder=0
        )
        ax.set_axisbelow(True)
        despine(ax, keep=("bottom",) if ax is axes[2] else ("bottom", "left"))

    fig.suptitle(
        "Own-lab pouring dataset: 121 clips, 18 trials, 2 synchronised views",
        y=1.04,
        fontsize=12.5,
        color=INK,
    )
    fig.savefig(OUT / "dataset.png")
    plt.close(fig)


# --------------------------------------------------- 2. flow baseline ladder
def fig_baselines_flow():
    rows = [
        ("V-JEPA 2  attentive probe", 0.81, BLUE),
        ("V-JEPA 2  ridge on mean-pool", 0.69, BLUE),
        ("Sound-of-Water  audio model", 0.65, AQUA),
        ("temporal profile  (norm-time poly4)", 0.62, GREY),
        ("s3d  (Kinetics video CNN)", 0.54, GREY),
        ("r2plus1d-18  (Kinetics video CNN)", 0.53, GREY),
        ("raw time  (linear clock)", 0.237, GREY),
        ("motion energy  (frame diff)", 0.18, GREY),
        ("predict mean", -0.01, GREY),
        ("shuffled-label null", -0.32, RED),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        xlim=(-0.42, 0.95),
        xlabel="held-out R²  (flow rate, g/s)",
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title("Flow rate: every baseline we ran, same held-out trials")
    fig.text(
        0.995,
        -0.02,
        "blue = V-JEPA 2   aqua = other pretrained backbone   grey = non-learned / non-V-JEPA   red = null",
        ha="right",
        fontsize=8.5,
        color=MUTED,
    )
    fig.savefig(OUT / "baselines_flow.png")
    plt.close(fig)


# ------------------------------------------------------ 3. flow vs volume
def fig_flow_vs_volume():
    labels = [
        "V-JEPA 2\nattentive",
        "V-JEPA 2\nridge",
        "Sound-of-Water\naudio",
        "video CNN\n(r2plus1d)",
        "temporal\nprofile",
        "raw time\n(clock)",
    ]
    # volume column is the 2026-07-28 4-fold both-cam run (all four bars from the same
    # folds); flow column keeps its established per-script sources.
    flow = [0.81, 0.69, 0.65, 0.53, 0.62, 0.237]
    vol = [0.576, 0.370, 0.80, 0.24, 0.552, 0.778]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.bar(x - w / 2 - 0.01, flow, w, color=BLUE, label="flow rate (g/s)", zorder=3)
    ax.bar(x + w / 2 + 0.01, vol, w, color=ORANGE, label="volume (cumulative g)", zorder=3)
    for xi, (f, v) in enumerate(zip(flow, vol)):
        ax.text(xi - w / 2 - 0.01, f + 0.02, f"{f:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2 + 0.01, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out R²")
    ax.set_ylim(0, 0.95)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Flow is a vision problem; volume is mostly a clock problem", pad=26)
    fig.savefig(OUT / "flow_vs_volume.png")
    plt.close(fig)


# ------------------------------------------------------------- 4. CV folds
def fig_cv_folds():
    folds = ["A\n{8,13,21,24}", "B\n{7,9,11,12}", "C\n{5,15,16,25,26}", "D\n{17,18,20,22,27}"]
    comb = [0.849, 0.800, 0.834, 0.763]
    cam2 = [0.898, 0.828, 0.883, 0.818]
    cam3 = [0.801, 0.772, 0.785, 0.707]

    x = np.arange(len(folds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.bar(x - w - 0.012, comb, w, color=BLUE, label="both views", zorder=3)
    ax.bar(x, cam2, w, color=ORANGE, label="CAM2 (side)", zorder=3)
    ax.bar(x + w + 0.012, cam3, w, color=AQUA, label="CAM3 (front, distant)", zorder=3)
    m = float(np.mean(comb))
    ax.axhline(m, color=INK2, lw=1.2, ls="--", zorder=4)
    ax.text(3.52, m + 0.03, f"mean {m:.2f} ± 0.04", fontsize=9.5, color=INK2, ha="right")
    for xi, v in zip(x - w - 0.012, comb):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, folds, fontsize=9)
    ax.set_xlabel("held-out fold (trial ids)")
    ax.set_ylabel("held-out R²  (flow)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.03))
    ax.set_title(
        "4-fold group CV by trial: all 18 trials held out exactly once", pad=26
    )
    fig.savefig(OUT / "cv_folds.png")
    plt.close(fig)


# --------------------------------------------------------- 5. head init
def fig_head_init():
    rows = [
        ("EK100 action pretrain", 0.854, BLUE),
        ("random init", 0.843, GREY),
        ("Sound-of-Water pouring pretrain", 0.831, GREY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.3), width_ratios=[1.15, 1])

    ax = axes[0]
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        fmt="{:.3f}",
        xlim=(0, 1.0),
        xlabel="held-out R² (flow, split A)",
    )
    ax.set_title("Final performance: a wash")

    ax = axes[1]
    ep = ["epoch 0", "converged"]
    ws = [0.669, 0.849]
    sc = [0.003, 0.843]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2 - 0.01, ws, w, color=BLUE, label="EK100 warm start", zorder=3)
    ax.bar(x + w / 2 + 0.01, sc, w, color=GREY, label="random init", zorder=3)
    for xi, (a, b) in enumerate(zip(ws, sc)):
        ax.text(xi - w / 2 - 0.01, a + 0.02, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2 + 0.01, b + 0.02, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ep)
    ax.set_ylabel("val R²")
    ax.set_ylim(0, 1.2)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left")
    ax.set_title("Warm start only buys convergence speed")
    fig.savefig(OUT / "head_init.png")
    plt.close(fig)


# ----------------------------------------------------------- 6. view study
def fig_views():
    rows = [
        ("both-cam probe → CAM2", 0.874, BLUE),
        ("CAM2-native probe → CAM2", 0.899, BLUE),
        ("both-cam probe → CAM3", 0.834, ORANGE),
        ("CAM3-native probe → CAM3", 0.804, ORANGE),
        ("CAM2 probe → CAM3  (zero-shot)", 0.524, RED),
        ("CAM3-native ridge baseline", 0.593, GREY),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        fmt="{:.3f}",
        xlim=(0, 1.05),
        xlabel="held-out R² (flow)",
    )
    ax.axvline(0.593, color=GREY, lw=1.2, ls="--", zorder=2)
    ax.set_title("One probe holds both views; zero-shot to an unseen view does not")
    fig.savefig(OUT / "views.png")
    plt.close(fig)


# ------------------------------------------------------------- 7. ROI crop
def fig_roi():
    """Three trial folds (A, B, C), both arms at lag 0.7, plus the native-view ridge
    recomputed at the same lag. Fold A is the 2026-07-27 run; B and C were added
    overnight 2026-07-28 to put error bars on a claim that rested on one split."""
    labels = ["CAM2 \u2192 CAM2\n(within view)", "CAM2 \u2192 CAM3\n(unseen view)"]
    # rows = folds A, B, C
    center = np.array([[0.885, 0.507], [0.863, 0.043], [0.906, 0.422]])
    roi = np.array([[0.781, 0.735], [0.863, 0.622], [0.875, 0.635]])
    native = np.array([[0.788, 0.764], [0.705, 0.714], [0.742, 0.638]])
    x = np.arange(2)
    w = 0.32
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for arr, off, col, lab in ((center, -w / 2 - 0.012, GREY, "center crop (full scene)"),
                               (roi, w / 2 + 0.012, BLUE, "detector ROI (crop to vessels)")):
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.bar(x + off, m, w, color=col, label=lab, zorder=3)
        ax.errorbar(x + off, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.3,
                    capsize=4, zorder=6)
        for xi in x:
            ax.scatter([xi + off] * 3, arr[:, xi], s=13, color=INK, zorder=7,
                       alpha=0.55, linewidths=0)
            ax.text(xi + off, m[xi] + sd[xi] + 0.03, f"{m[xi]:.2f}", ha="center",
                    fontsize=9.5, color=INK)
    nm = native.mean(0)
    for xi, v in zip(x, nm):
        ax.plot([xi - 0.42, xi + 0.42], [v, v], color=ORANGE, lw=1.8, ls="--", zorder=5)
    ax.plot([], [], color=ORANGE, lw=1.8, ls="--", label="linear ridge trained on that view")
    ax.text(0.44, nm[0] + 0.02, f"{nm[0]:.2f}", fontsize=9, color=ORANGE, ha="right")
    ax.text(1.44, nm[1] + 0.02, f"{nm[1]:.2f}", fontsize=9, color=ORANGE, ha="right")
    ax.text(0, 1.0, "\u22120.05", ha="center", fontsize=11.5, color=RED, weight="bold")
    ax.text(1, 1.0, "+0.34", ha="center", fontsize=11.5, color=AQUA, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out R\u00b2 (flow), lag 0.7 s")
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Cropping mainly buys STABILITY on an unseen view (3 folds, dots = folds)")
    fig.savefig(OUT / "roi.png")
    plt.close(fig)


# ------------------------------------------- 8. total-mass compression fixes
def fig_totals():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))

    ax = axes[0]
    lab = ["SmoothL1\n(deliverable)", "MSE loss", "post-hoc\ncalibration"]
    vals = [23.0, 31.0, 28.0]
    ax.bar(range(3), vals, 0.55, color=[BLUE, GREY, GREY], zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.6, f"{v:.0f} g", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(3), lab)
    ax.set_ylabel("total poured-mass MAE (g)")
    ax.set_ylim(0, 38)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Nothing beats the plain recipe (lower is better)")

    ax = axes[1]
    lab = ["SmoothL1\n(deliverable)", "MSE loss", "ridge, α → 0\n(held out)", "ridge, α → 0\n(in sample)"]
    vals = [0.927, 0.960, 0.83, 0.995]
    ax.bar(range(4), vals, 0.55, color=[BLUE, GREY, GREY, MAGENTA], zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)
    ax.axhline(1.0, color=ORANGE, lw=1.4, ls="--", zorder=4)
    ax.text(-0.45, 1.06, "no compression", color=ORANGE, fontsize=9, ha="left")
    ax.set_xticks(range(4), lab, fontsize=8.5)
    ax.set_ylabel("pred vs GT total-mass slope")
    ax.set_ylim(0, 1.25)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Held-out slope stalls below 1 whatever we do")
    fig.savefig(OUT / "totals.png")
    plt.close(fig)


# ------------------------------------------------------ 9. oracle container
def fig_oracle():
    rows = [
        ("raw time (clock)", 0.784, GREY),
        ("time + oracle container", 0.776, GREY),
        ("time × oracle container", 0.756, GREY),
        ("V-JEPA 2 ridge", 0.364, BLUE),
        ("V-JEPA 2 ridge + oracle container", 0.365, BLUE),
        ("oracle container alone", 0.012, RED),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        fmt="{:.3f}",
        xlim=(0, 0.95),
        xlabel="held-out R² (volume)",
    )
    ax.set_title("Perfect container identity adds nothing to absolute volume")
    fig.savefig(OUT / "oracle.png")
    plt.close(fig)


# ------------------------------------------------------- 10. sound of water
def fig_sow_crossmodal():
    labels = ["flow rate", "volume"]
    vjepa = [0.81, 0.576]
    audio = [0.648, 0.802]
    clock = [0.237, 0.778]
    x = np.arange(2)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - w - 0.012, vjepa, w, color=BLUE, label="V-JEPA 2 video", zorder=3)
    ax.bar(x, audio, w, color=AQUA, label="Sound-of-Water audio", zorder=3)
    ax.bar(x + w + 0.012, clock, w, color=GREY, label="raw time (clock)", zorder=3)
    for xi, trio in zip(x, zip(vjepa, audio, clock)):
        for dx, v in zip((-w - 0.012, 0, w + 0.012), trio):
            ax.text(xi + dx, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out R² (our clips)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.03))
    ax.set_title("Audio wins on volume only because the clock already does", pad=24)
    fig.savefig(OUT / "sow_crossmodal.png")
    plt.close(fig)


def fig_sow_ondata():
    labels = [
        "S1\ntransparent\n(11 cont.)",
        "S2\ncylindrical\n(21 cont.)",
        "S2o\nopaque\n(8 cont.)",
        "S3\nall shapes\n(45 cont.)",
    ]
    probe = [0.163, 0.287, 0.749, 0.803]
    clock = [0.058, 0.253, 0.976, 0.799]
    x = np.arange(4)
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(x - w / 2 - 0.012, probe, w, color=BLUE, label="V-JEPA 2 probe", zorder=3)
    ax.bar(x + w / 2 + 0.012, clock, w, color=GREY, label="normalised-time poly4", zorder=3)
    for xi, (a, b) in enumerate(zip(probe, clock)):
        ax.text(xi - w / 2 - 0.012, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2 + 0.012, b + 0.015, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("within-video R² (volume)")
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Sound-of-Water videos: constant-flow pours, so the clock explains them")
    fig.savefig(OUT / "sow_ondata.png")
    plt.close(fig)


# --------------------------------------------------- 11. arc / result recap
def fig_inputs():
    """What the probe actually receives, and the shortcut it CANNOT take.

    The ground truth comes from OCR of a scale display, so the first question any
    reviewer asks is whether the probe can just read that display. It cannot: the LCD
    faces the dedicated overhead OCR camera, CAM2 sees only the weighing platform, and
    after short-side-256 + center crop the whole scale is ~30x15 px.
    """
    import decord

    vr = decord.VideoReader(
        str(ROOT / "datasets/pouring_processed/clips/CAM2/0001.mp4"))
    fr = vr[int(2 * float(vr.get_avg_fps()))].asnumpy()
    H, W = fr.shape[:2]

    box = (800, 600, 1120, 720)                      # the scale, native coords
    s = 256 / min(W, H)
    small = np.array(Image.fromarray(fr).resize((int(W * s), int(H * s))))
    l, t = (small.shape[1] - 256) // 2, (small.shape[0] - 256) // 2
    probe = small[t:t + 256, l:l + 256]

    # width_ratios = each panel's aspect, so all three render at equal height undistorted
    ar = [W / H, (box[2] - box[0]) / (box[3] - box[1]), 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 2.9),
                             gridspec_kw={"width_ratios": ar})

    axes[0].imshow(fr)
    axes[0].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                    fill=False, color=ORANGE, lw=2.2))
    axes[0].set_title("CAM2 as recorded  (1920×1080)", fontsize=10.5)

    axes[1].imshow(fr[box[1]:box[3], box[0]:box[2]])
    axes[1].set_title("the scale, zoomed: NO display visible", fontsize=10.5, color=AQUA)

    axes[2].imshow(probe)
    axes[2].set_title("what the probe gets  (256×256)", fontsize=10.5)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.suptitle("The ground truth is OCR'd from a scale the probe cannot read",
                 y=1.03, fontsize=12.5, color=INK)
    fig.savefig(OUT / "inputs.png")
    plt.close(fig)


def fig_volume_curves():
    """Predicted cumulative mass vs ground truth, six clips spanning the dataset, for the
    three best volume methods. All curves are OUT-OF-FOLD.

    The point of the figure is visual: the clock draws the SAME line on every clip, so it
    tracks an average pour and cannot follow a fast or slow one. Selection is by total
    mass quantile, not by eye, so this is not a cherry-pick.
    """
    f = np.load("/home/casimir/.cache/pour_probe/headline_preds.npz", allow_pickle=True)
    bc, bt, by = f["base_clip"], f["base_tmid"], f["base_y"]
    clock, vjc = f["base_clock"], f["base_vjepa_clock"]
    ac, at, ap = f["volume_clip"], f["volume_tmid"], f["volume_pred"]

    def curve(cl, tt, vv, cid):                     # average the two cameras
        m = cl == cid
        o = np.argsort(tt[m])
        t, v = tt[m][o], np.asarray(vv)[m][o]
        ut = np.unique(t)
        return ut, np.array([v[t == u].mean() for u in ut])

    clips = np.unique(bc)
    totals = {c: by[bc == c].max() for c in clips}
    order = sorted(clips, key=lambda c: totals[c])
    pick = [order[int(q * (len(order) - 1))] for q in (0.05, 0.25, 0.45, 0.62, 0.8, 0.96)]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 5.6), sharex=False)
    for ax, cid in zip(axes.ravel(), pick):
        t, g = curve(bc, bt, by, cid)
        ax.plot(t, g, color=INK, lw=2.6, label="ground truth (scale)", zorder=5)
        for vals, col, lab in ((vjc, BLUE, "V-JEPA + clock"),
                               (clock, GREY, "clock only"),
                               (None, ORANGE, "V-JEPA attentive")):
            if vals is None:
                tt, vv = curve(ac, at, ap, cid)
            else:
                tt, vv = curve(bc, bt, vals, cid)
            ax.plot(tt, vv, color=col, lw=1.9, ls="--", label=lab, zorder=4)
        ax.set_title(f"clip {cid} · {totals[cid]:.0f} g poured", fontsize=9.5, color=INK2)
        ax.yaxis.grid(True, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        ax.tick_params(labelsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("time in clip (s)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative mass (g)", fontsize=9)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=4, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Out-of-fold predicted volume. The clock draws one slope for every pour",
                 y=1.035, fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "volume_curves.png")
    plt.close(fig)


def fig_protocol():
    """Stricter protocol borrowed from the Sound-of-Water runs: score the trivial
    controls under the SAME within-pour metric as the probe. clips_eval_protocol.py,
    both cams, 4-fold OOF by trial."""
    methods = ["raw time\n(causal clock)", "time profile\n(ORACLE duration)",
               "V-JEPA 2", "V-JEPA 2\n+ clock"]
    data = {
        "flow": {"global": [0.237, 0.588, 0.718, 0.717],
                 "within": [0.439, 0.698, 0.771, 0.770]},
        "volume": {"global": [0.777, 0.559, 0.371, 0.818],
                   "within": [0.864, 0.783, 0.622, 0.886]},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True)
    x = np.arange(len(methods))
    w = 0.36
    for ax, (tgt, d) in zip(axes, data.items()):
        ax.bar(x - w / 2 - 0.012, d["global"], w, color=GREY,
               label="R² vs global mean", zorder=3)
        ax.bar(x + w / 2 + 0.012, d["within"], w, color=BLUE,
               label="R² within pour (mean removed)", zorder=3)
        for xi, (a, b) in enumerate(zip(d["global"], d["within"])):
            ax.text(xi - w / 2 - 0.012, a + 0.015, f"{a:.2f}", ha="center",
                    fontsize=8.5, color=INK)
            ax.text(xi + w / 2 + 0.012, b + 0.015, f"{b:.2f}", ha="center",
                    fontsize=8.5, color=INK)
        ax.set_xticks(x, methods, fontsize=8.5)
        ax.set_title(tgt, color=INK)
        ax.yaxis.grid(True, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[0].set_ylabel("held-out R²")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Removing the between-pour offset does not change either verdict",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "protocol.png")
    plt.close(fig)


def fig_arc():
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    ax.axis("off")
    stages = [
        ("EgoPER\nprocedural errors", "window AUC 0.75", MUTED),
        ("eXprt tea\nanomaly + actions", "6-way acc 0.71", MUTED),
        ("Pivot 05.07\npouring", "locked centerpiece", ORANGE),
        ("Own-lab clips\nflow rate", "R² 0.81 ± 0.04", BLUE),
    ]
    n = len(stages)
    for i, (title, sub, col) in enumerate(stages):
        x = i / (n - 1) * 0.76 + 0.12
        ax.add_patch(
            plt.Rectangle(
                (x - 0.105, 0.28), 0.21, 0.44, color=col, alpha=0.16, zorder=2
            )
        )
        ax.text(x, 0.60, title, ha="center", va="center", fontsize=11, color=INK)
        ax.text(x, 0.38, sub, ha="center", va="center", fontsize=10, color=col)
        if i < n - 1:
            ax.annotate(
                "",
                xy=((i + 1) / (n - 1) * 0.76 + 0.12 - 0.115, 0.50),
                xytext=(x + 0.115, 0.50),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4),
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    fig.savefig(OUT / "arc.png")
    plt.close(fig)


# ------------------------------------------- 13. why the clock wins on volume
def fig_clock():
    """All numbers measured 2026-07-26 on the mean-pool cache, both cams,
    the same 4-fold-by-trial protocol as every other ridge row."""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.5), width_ratios=[1, 1.05, 1.15])

    # (a) where the variance lives
    ax = axes[0]
    tgt = ["volume\n(cumulative g)", "flow\n(g/s)"]
    within = [0.72, 0.83]
    between = [0.28, 0.17]
    y = np.arange(2)[::-1]
    ax.barh(y, within, 0.5, color=BLUE, label="within a pour", zorder=3)
    ax.barh(y, between, 0.5, left=within, color=GREY, label="between pours", zorder=3)
    for yi, w in zip(y, within):
        ax.text(w / 2, yi, f"{w:.0%}", va="center", ha="center", color="white", fontsize=11)
        ax.text(w + (1 - w) / 2, yi, f"{1-w:.0%}", va="center", ha="center", color=INK, fontsize=11)
    ax.set_yticks(y, tgt)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    despine(ax, keep=())
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.14), fontsize=9)
    ax.set_title("Where the variance lives", pad=30)

    # (b) the clock is one number
    ax = axes[1]
    t = np.linspace(0, 6, 100)
    ax.plot(t, -37 + 50 * t, color=ORANGE, lw=2.4, label="what the clock fits", zorder=4)
    rng = np.random.default_rng(3)
    for tot, dur in [(31, 3.3), (71, 4.0), (135, 4.3), (176, 4.7), (232, 5.0), (335, 5.7)]:
        tt = np.linspace(0, dur, 40)
        s = tot / (1 + np.exp(-(tt - dur * 0.5) * 8 / dur))
        s = (s - s[0]) / (s[-1] - s[0]) * tot
        ax.plot(tt, s, color=BLUE, lw=1.2, alpha=0.55, zorder=3)
    ax.set_xlabel("time since pour start (s)")
    ax.set_ylabel("poured mass (g)")
    ax.set_ylim(-10, 360)
    ax.set_xlim(0, 6)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.text(0.05, 0.93, "volume ≈ −37 + 50·t", transform=ax.transAxes,
            color=ORANGE, fontsize=10.5)
    ax.text(0.05, 0.84, "one global slope, no vision", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    ax.set_title("A single average flow rate of 50 g/s")

    # (c) are the two signals additive?
    ax = axes[2]
    x = np.arange(2)
    w = 0.26
    clock = [0.778, -0.012]
    vjepa = [0.398, 0.577]
    both = [0.818, 0.586]
    ax.bar(x - w - 0.012, clock, w, color=GREY, label="clock only", zorder=3)
    ax.bar(x, vjepa, w, color=BLUE, label="V-JEPA only", zorder=3)
    ax.bar(x + w + 0.012, both, w, color=AQUA, label="both", zorder=3)
    for xi, trio in zip(x, zip(clock, vjepa, both)):
        for dx, v in zip((-w - 0.012, 0, w + 0.012), trio):
            ax.text(xi + dx, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ["volume", "flow"])
    ax.set_ylabel("held-out R² (ridge)")
    ax.set_ylim(-0.1, 1.12)
    ax.axhline(0, color=MUTED, lw=1)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.14), fontsize=8.5)
    ax.set_title("The two signals barely overlap", pad=30)

    fig.savefig(OUT / "clock.png")
    plt.close(fig)


# ---------------------------------------- 14. flow variability, ours vs theirs
def fig_variability():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))

    ax = axes[0]
    labels = ["our clips", "Sound-of-Water"]
    vals = [0.943, 0.989]
    ax.bar(range(2), vals, 0.5, color=[BLUE, GREY], zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("median per-sequence R²")
    ax.set_ylim(0.9, 1.02)
    ax.axhline(1.0, color=ORANGE, lw=1.3, ls="--", zorder=4)
    ax.text(1.45, 1.006, "perfectly linear", color=ORANGE, fontsize=9, ha="right")
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Cumulative volume vs a straight line in time")

    ax = axes[1]
    frac = [0.41, 0.83]
    ax.bar(range(2), frac, 0.5, color=[BLUE, GREY], zorder=3)
    for i, v in enumerate(frac):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("fraction of sequences")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Sequences a straight line fits above R² 0.95")

    fig.savefig(OUT / "variability.png")
    plt.close(fig)


# ------------------------------- 15. did SoW really randomise their flow rates?
def fig_sow_ttf():
    """Their time-to-fill task (paper Table 3) against a no-listening prior.

    Their metric is tau = T - t, the REMAINING time, given audio cut at f*T.
    A baseline that never listens predicts tau_hat = Tbar - t using the free
    audio length t, so its MAE is |Tbar - T|, the same at every cut level.
    Container-mean prior on Test I (containers seen in training), global mean on
    Test II (containers unseen). Computed from their own splits/*.csv.
    """
    theirs = {
        "Test I\n(seen containers)": ([4.16, 1.49, 1.07], 2.11),
        "Test II\n(unseen containers)": ([4.10, 2.99, 2.21], 2.68),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), sharey=True)
    for ax, (name, (vals, base)) in zip(axes, theirs.items()):
        x = np.arange(3)
        ax.bar(x, vals, 0.5, color=BLUE, label="their model (co-supervised)", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)
        ax.axhline(base, color=ORANGE, lw=1.8, ls="--", zorder=4,
                   label="never listen, predict a prior")
        ax.text(2.45, base + 0.1, f"{base:.2f} s", color=ORANGE, fontsize=9.5, ha="right")
        ax.set_xticks(x, ["25%", "50%", "75%"])
        ax.set_xlabel("fraction of the pour heard")
        ax.set_ylim(0, 5.0)
        ax.yaxis.grid(True, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("MAE on remaining time (s)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Sound-of-Water time-to-fill vs a baseline that never listens",
                 y=1.04, fontsize=12, color=INK)
    fig.savefig(OUT / "sow_ttf.png")
    plt.close(fig)


# -------------------------------------------------- 12. example camera views
def fig_views_example():
    import cv2

    clip, cams = "0001", ["CAM2", "CAM3"]
    titles = ["CAM2  side view", "CAM3  front view, distant"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.3))
    for ax, cam, lab in zip(axes, cams, titles):
        cap = cv2.VideoCapture(
            str(ROOT / f"datasets/pouring_processed/clips/{cam}/{clip}.mp4")
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            return
        ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(lab, fontsize=11)
    fig.suptitle(
        "Clip 0001  (kettle → blue_mug, 218 g, 5.0 s)", y=1.02, fontsize=12, color=INK2
    )
    fig.savefig(OUT / "views_example.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sync_assets()
    fig_views_example()
    fig_dataset()
    fig_baselines_flow()
    fig_flow_vs_volume()
    fig_cv_folds()
    fig_head_init()
    fig_views()
    fig_roi()
    fig_totals()
    fig_oracle()
    fig_sow_crossmodal()
    fig_sow_ondata()
    fig_clock()
    fig_variability()
    fig_sow_ttf()
    fig_inputs()
    fig_volume_curves()
    fig_protocol()
    fig_arc()
    print("wrote", len(list(OUT.glob("*.png"))), "figures to", OUT)

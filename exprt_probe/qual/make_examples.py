"""Curated 4-example figure: where off-the-shelf V-JEPA 2 + EK100 gets eXprt actions right vs wrong.

Reuses the predictions already cached in ``exprt_probe/qual/annotations.csv`` (verb/noun/action
columns written by ``ek100_label.py``) — no model/GPU re-run needed. Hand-picks two clearly
correct and two clearly wrong segments (judged on the EK100 *verb*, the fair signal since EPIC's
vocabulary has no tea-specific nouns) and renders a 2x2 montage with green/red borders.

Usage:
    .venv/bin/python exprt_probe/qual/make_examples.py
-> exprt_probe/qual/figures/ek100_examples_6.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
QUAL = ROOT / "exprt_probe" / "qual"
DATA = ROOT / "datasets" / "eXprt-Daten" / "CAM1 Aufnahmen Patrick"
FPS = 20.0

# friendly video name (in annotations.csv) -> eXprt recording dir
WATCH_MAP = {
    "Normal": "20250827_131941_875_GX011018_20fps",
    "Spueli": "20250827_131032_417_GX011014_20fps",
    "glass_and_fork": "20250827_134826_667_GX011027_20fps",
    "2tb_2stir": "20250827_135039_083_GX011028_20fps",
}

# Curated picks: (video, start_s, stop_s, is_correct, why). Matched to annotations.csv rows.
VERB_OK = "#1a8a2e"   # green border  = verb correct
VERB_BAD = "#c0271e"  # red border    = verb wrong
NOUN_BAD = "#e8820c"  # orange text   = noun misclassified

# (video, start_s, stop_s, verb_ok, noun_ok, why). Border colour = verb_ok;
# the EK100 noun line is drawn orange when noun_ok is False.
PICKS = [
    ("Spueli", 62.0, 68.0, True, True,
     "verb + object both right (pour:water)"),
    ("Spueli", 50.0, 53.0, True, True,
     "verb 'put' + noun 'bag' (= teabag) both right"),
    ("Normal", 49.0, 55.0, True, False,
     "verb 'take' right, but the object is misread"),
    ("glass_and_fork", 74.0, 80.0, False, True,
     "verb missed, yet noun 'kettle' is correct"),
    ("glass_and_fork", 84.0, 88.0, False, False,
     "misses the distinctive 'stir'; object wrong too"),
    ("Spueli", 47.0, 49.0, False, False,
     "predicts 'close', the opposite of opening"),
]


def mid_frame(video: str, t: float) -> np.ndarray:
    paths = sorted((DATA / WATCH_MAP[video]).glob("frame_*.png"))
    i = int(np.clip(round(t * FPS), 0, len(paths) - 1))
    return np.asarray(Image.open(paths[i]).convert("RGB"))


def row_for(df: pd.DataFrame, video: str, s: float, e: float) -> pd.Series:
    m = df[(df["video"] == video) & (np.isclose(df["start_s"], s)) & (np.isclose(df["stop_s"], e))]
    if m.empty:
        raise SystemExit(f"no annotations.csv row for {video} {s}-{e}")
    return m.iloc[0]


def caption(ax, lines):
    """Draw caption lines below the image, each with its own colour (for orange nouns)."""
    y0, dy = -0.05, 0.062
    for i, (txt, col, italic) in enumerate(lines):
        ax.text(0.02, y0 - i * dy, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, family="monospace", color=col,
                style="italic" if italic else "normal")


def main():
    df = pd.read_csv(QUAL / "annotations.csv")
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.8))
    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.03, right=0.97, hspace=0.82, wspace=0.12)

    for ax, (video, s, e, verb_ok, noun_ok, why) in zip(axes.ravel(), PICKS):
        r = row_for(df, video, s, e)
        label = str(r["action"]).strip() if pd.notna(r["action"]) and str(r["action"]).strip() else str(r["notes"]).strip()
        ax.imshow(mid_frame(video, (s + e) / 2))
        ax.set_xticks([]); ax.set_yticks([])
        border = VERB_OK if verb_ok else VERB_BAD
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(6)
        mark = "✓ verb correct" if verb_ok else "✗ verb wrong"
        ax.set_title(f"{mark}    {video}  [{s:.0f}-{e:.0f}s]", color=border, fontsize=12.5, fontweight="bold")
        caption(ax, [
            (f"ground truth: {label}", "black", False),
            (f"EK100 verb:   {r['ek100_verbs']}", "black", False),
            (f"EK100 noun:   {r['ek100_nouns']}", "black" if noun_ok else NOUN_BAD, False),
            (f"EK100 action: {r['ek100_action']}", "black", False),
            (f"-> {why}", "#555", True),
        ])

    fig.suptitle("Off-the-shelf V-JEPA 2 + EK100 (zero-shot) on eXprt tea actions: hits vs misses",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             "Green/red border = verb correct/wrong (the fair signal).  Orange = noun misclassified "
             "(EPIC vocabulary has no tea-specific nouns).  Predictions are anticipatory ~1 s.",
             ha="center", fontsize=9.5, style="italic", color="#555")
    (QUAL / "figures").mkdir(exist_ok=True)
    out = QUAL / "figures" / "ek100_examples_6.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

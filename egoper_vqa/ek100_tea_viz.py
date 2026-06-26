"""Qualitative VISUAL results: off-the-shelf V-JEPA 2 + EK100 on EgoPER tea frames.

For a curated set of (video, timestamp) moments, renders the actual tea frame annotated with the
EgoPER ground-truth action and EK100's top verb/noun/action predictions, and assembles a
slide-ready montage. Companion to ek100_tea.py (reuses its model + decoders); this one is for
showing examples in a presentation.

Usage:
    .venv/bin/python egoper_vqa/ek100_tea_viz.py
-> writes egoper_vqa/figures/ek100_tea_montage.png  (+ per-example PNGs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from decord import VideoReader, cpu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "egoper_vqa"))
import egoper  # noqa: E402
from ek100_tea import (ANTICIPATION_S, CLIPS, PRETRAIN_KWARGS, WRAPPER_KWARGS,  # noqa: E402
                       build_decoders, init_module, load_classifier, load_clip)

FIG_DIR = ROOT / "egoper_vqa" / "figures"

# Curated moments: (video_id, T_seconds, short tag). T = the action instant the model anticipates;
# we observe the clip up to T - ANTICIPATION_S and display/score at T. Mix of clean actions, the two
# anomalies, and one honest vocabulary-mismatch (honey).
EXAMPLES = [
    ("tea_u2_av_normal_010", 12, "measure water"),
    ("tea_u2_av_normal_010", 28, "transfer water"),
    ("tea_u2_av_normal_010", 72, "pour into mug"),
    ("tea_u2_aq_normal_009", 98, "stir w/ spoon"),
    ("tea_u2_aq_normal_009", 10, "place tea bag"),
    ("tea_u2_av_normal_010", 122, "honey (vocab gap)"),
    ("tea_u1_a2_error_004", 70, "ANOMALY: microwave"),
    ("tea_u1_a2_error_004", 115, "ANOMALY: stir w/ knife"),
]


def display_frame(path, t):
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    idx = int(np.clip(round(t * fps), 0, len(vr) - 1))
    return vr[idx].asnumpy()


def gt_at(timeline, t):
    for r in timeline:
        if r["start"] <= t <= r["end"]:
            return r["type"], r["desc"]
    return "(none)", ""


@torch.no_grad()
def main():
    FIG_DIR.mkdir(exist_ok=True)
    device = "cuda"
    dec = build_decoders()
    model = init_module(module_name="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar",
                        frames_per_clip=32, frames_per_second=8, resolution=256, checkpoint=str(ROOT / "checkpoints" / "vitl.pt"),
                        model_kwargs=PRETRAIN_KWARGS, wrapper_kwargs=WRAPPER_KWARGS, device=device)
    clf = load_classifier(dec["dims"], device)

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    at = torch.tensor([ANTICIPATION_S], device=device)
    for ax, (vid, T, tag) in zip(axes.ravel(), EXAMPLES):
        path = CLIPS / f"{vid}.mp4"
        tl = egoper.ground_truth("tea", vid)["timeline"]
        gtype, gdesc = gt_at(tl, T)                          # GT at the anticipated instant
        x = load_clip(path, float(T - ANTICIPATION_S), device)  # observe up to T - 1s
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = clf(model(x, at))
        tv = [dec["verb"](i) for i in out["verb"][0].float().topk(3).indices.tolist()]
        tn = [dec["noun"](i) for i in out["noun"][0].float().topk(3).indices.tolist()]
        ta = [dec["action"](i) for i in out["action"][0].float().topk(1).indices.tolist()]

        print(f"  [{tag:18}] T={T:>3}s  GT={gtype}:{gdesc[:34]:34} | verb={tv} noun={tn} action={ta[0]}")
        ax.imshow(display_frame(path, T))
        ax.axis("off")
        anomaly = gtype != "Normal"
        ax.set_title(f"[{tag}]  t={T}s", fontsize=12, fontweight="bold",
                     color="crimson" if anomaly else "black")
        caption = (f"GT: {gtype} — {gdesc}\n"
                   f"verb:   {', '.join(tv)}\n"
                   f"noun:   {', '.join(tn)}\n"
                   f"action: {ta[0]}")
        ax.text(0.5, -0.02, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round", fc="#fff4f4" if anomaly else "#f2f7ff", ec="grey"))

    fig.suptitle("Off-the-shelf V-JEPA 2 + EK100 (zero-shot) on EgoPER tea — anticipated action vs ground truth",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "ek100_tea_montage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

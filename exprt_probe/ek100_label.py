"""Label hand-annotated eXprt action segments with off-the-shelf V-JEPA 2 + EK100 (qualitative).

Reads `exprt_probe/qual/annotations.csv` (you fill in video/start_s/stop_s/action), samples a clip
over each segment from the eXprt PNG frames, runs the same faithful EK100 anticipation pipeline as
`egoper_vqa/ek100_tea.py`, and prints its top verb/noun/action prediction next to your label, plus a
montage PNG. Diagnostic only (EPIC vocab has no tea-specific nouns -> verbs are the fair signal).

Usage:
    .venv/bin/python exprt_probe/ek100_label.py
-> exprt_probe/qual/figures/ek100_exprt_labels.png  (+ printed table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "egoper_vqa"))
from ek100_tea import (ANTICIPATION_S, FRAMES, MEAN, RES, STD, PRETRAIN_KWARGS,  # noqa: E402
                       WRAPPER_KWARGS, build_decoders, init_module, load_classifier)

DATA = ROOT / "datasets" / "eXprt-Daten" / "CAM1 Aufnahmen Patrick"
QUAL = ROOT / "exprt_probe" / "qual"
FPS = 20.0

# friendly video name (in annotations.csv) -> eXprt recording dir
WATCH_MAP = {
    "Normal": "20250827_131941_875_GX011018_20fps",
    "Spueli": "20250827_131032_417_GX011014_20fps",
    "glass_and_fork": "20250827_134826_667_GX011027_20fps",
    "2tb_2stir": "20250827_135039_083_GX011028_20fps",
}


def _frames(video):
    return sorted((DATA / WATCH_MAP[video]).glob("frame_*.png"))


def _crop(im):
    w, h = im.size
    nw, nh = (RES, round(h * RES / w)) if w <= h else (round(w * RES / h), RES)
    im = im.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - RES) // 2, (nh - RES) // 2
    return np.asarray(im)[t:t + RES, l:l + RES, :]


def load_segment(video, start_s, stop_s, device):
    """32 frames evenly spanning [start_s, stop_s] -> (1,3,32,256,256) normalized."""
    paths = _frames(video)
    n = len(paths)
    idx = np.clip(np.linspace(start_s * FPS, stop_s * FPS, FRAMES).round().astype(int), 0, n - 1)
    arr = np.stack([_crop(Image.open(paths[i]).convert("RGB")) for i in idx])  # (32,H,W,3)
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255).to(device)
    return (x - MEAN.to(device)) / STD.to(device)


def display_frame(video, t):
    paths = _frames(video)
    return np.asarray(Image.open(paths[int(np.clip(round(t * FPS), 0, len(paths) - 1))]).convert("RGB"))


@torch.no_grad()
def main():
    df = pd.read_csv(QUAL / "annotations.csv")
    # your label = `action` if filled, else `notes` (you annotated in notes)
    df["label"] = df.apply(
        lambda r: str(r["action"]).strip() if pd.notna(r["action"]) and str(r["action"]).strip()
        else str(r["notes"]).strip(), axis=1)
    ann = df[df["start_s"].notna() & df["stop_s"].notna()
             & ~df["notes"].astype(str).str.startswith("EXAMPLE", na=False)]
    if ann.empty:
        print("No annotated segments yet — fill in exprt_probe/qual/annotations.csv first."); return
    for c in ["ek100_verbs", "ek100_nouns", "ek100_action"]:
        if c not in df.columns:
            df[c] = ""

    device = "cuda"
    dec = build_decoders()
    model = init_module(module_name="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar",
                        frames_per_clip=32, frames_per_second=8, resolution=256,
                        checkpoint=str(ROOT / "checkpoints" / "vitl.pt"), model_kwargs=PRETRAIN_KWARGS,
                        wrapper_kwargs=WRAPPER_KWARGS, device=device)
    clf = load_classifier(dec["dims"], device)
    at = torch.tensor([ANTICIPATION_S], device=device)

    results = []
    print(f"\n{'video':16} {'span(s)':>12}  {'your action':24} | EK100 verb / noun / action")
    for idx, r in ann.iterrows():
        v, s, e = r["video"], float(r["start_s"]), float(r["stop_s"])
        if v not in WATCH_MAP:
            print(f"  ?? unknown video '{v}' — skip"); continue
        x = load_segment(v, s, e, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = clf(model(x, at))
        tv = [dec["verb"](i) for i in out["verb"][0].float().topk(3).indices.tolist()]
        tn = [dec["noun"](i) for i in out["noun"][0].float().topk(3).indices.tolist()]
        ta = dec["action"](int(out["action"][0].float().argmax()))
        df.at[idx, "ek100_verbs"] = ",".join(tv)
        df.at[idx, "ek100_nouns"] = ",".join(tn)
        df.at[idx, "ek100_action"] = ta
        print(f"  {v:16} {s:5.1f}-{e:<5.1f}  {r['label'][:24]:24} | {','.join(tv)}  |  {','.join(tn)}  |  {ta}")
        results.append((v, s, e, r["label"], tv, tn, ta))

    df.drop(columns=["label"]).to_csv(QUAL / "annotations.csv", index=False)
    print(f"wrote predictions back to {QUAL/'annotations.csv'} (ek100_verbs/nouns/action columns)")

    # montage: one mid-frame per segment + caption
    n = len(results)
    cols = min(4, n); rowsn = (n + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(5.2 * cols, 5.4 * rowsn), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (v, s, e, act, tv, tn, ta) in zip(axes.ravel(), results):
        ax.imshow(display_frame(v, (s + e) / 2)); ax.axis("off")
        ax.set_title(f"{v}  [{s:.0f}-{e:.0f}s]", fontsize=11, fontweight="bold")
        ax.text(0.5, -0.02, f"you: {act}\nverb: {', '.join(tv)}\nnoun: {', '.join(tn)}\naction: {ta}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="#f2f7ff", ec="grey"))
    fig.suptitle("Off-the-shelf V-JEPA 2 + EK100 (zero-shot) on hand-annotated eXprt actions",
                 fontsize=13, fontweight="bold")
    (QUAL / "figures").mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = QUAL / "figures" / "ek100_exprt_labels.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")
    print("CAVEAT: EPIC vocab has no tea nouns (verbs are the fair signal); prediction is anticipatory ~1s.")


if __name__ == "__main__":
    main()

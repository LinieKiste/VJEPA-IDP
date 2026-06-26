"""Quantitative back-up for the qualitative montage: EK100 verb-accuracy on EgoPER tea.

Zero-shot V-JEPA 2 + EK100 (the same faithful pipeline as ek100_tea.py) predicts an Epic-Kitchens
verb for each ground-truth tea action segment; we score whether an *acceptable* verb appears in the
top-1 / top-5. "Acceptable" verbs per tea action are a small **hand-built** map (TEA_VERB_MAP), each
key validated against EPIC's 97-verb vocabulary. Verbs (not nouns/actions) are the fair metric
because EPIC's vocabulary has no tea-specific nouns (tea/mug/teabag).

Protocol: for each non-BG normal tea action segment, observe the clip ending at (segment_mid - 1 s)
so the anticipated instant ~ segment mid; compare EK100's verb prediction to the action's acceptable
set. Reports overall + per-action top-1/top-5, vs a random-verb baseline, and a bar chart.

Usage:
    .venv/bin/python egoper_vqa/ek100_tea_eval.py
-> egoper_vqa/figures/ek100_tea_verb_accuracy.{csv,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "egoper_vqa"))
import egoper  # noqa: E402
from ek100_tea import (ANTICIPATION_S, CLIPS, META, PRETRAIN_KWARGS, WRAPPER_KWARGS,  # noqa: E402
                       build_decoders, init_module, load_classifier, load_clip)

FIG = ROOT / "egoper_vqa" / "figures"

# Hand-built map: exact EgoPER tea GT description -> acceptable EPIC verb keys (validated below).
# Subjective by nature; shown in output for transparency.
TEA_VERB_MAP = {
    "Measure 12 ounces of cold water":      {"pour", "fill", "take", "scoop"},
    "Transfer water to kettle":             {"pour", "fill", "empty", "put"},
    "Place tea bag in mug":                 {"put", "insert", "take"},
    "Check water temperature":              {"check", "look", "take", "open"},
    "Pour water from kettle into mug":      {"pour", "fill"},
    "Steep tea bag in mug for a few seconds": {"insert", "dip", "put", "mix", "hold", "soak", "shake"},
    "Put tea bag into trash can":           {"throw", "put", "insert", "remove", "take"},
    "Add honey to mug":                     {"pour", "squeeze", "put", "add"},
    "Stir using spoon":                     {"mix", "stir"},
    "Hold cup in front of you":             {"hold", "take", "move", "lift"},
}


def validate_map():
    """Drop verb keys not in EPIC's 97-verb vocabulary; warn about them."""
    valid = set(pd.read_csv(META / "EPIC_100_verb_classes.csv")["key"])
    cleaned, dropped = {}, set()
    for desc, verbs in TEA_VERB_MAP.items():
        keep = verbs & valid
        dropped |= (verbs - valid)
        cleaned[desc] = keep
    if dropped:
        print(f"note: verb keys not in EPIC vocab, dropped from map: {sorted(dropped)}")
    return cleaned, len(valid)


@torch.no_grad()
def main():
    FIG.mkdir(exist_ok=True)
    device = "cuda"
    dec = build_decoders()
    vmap, n_verbs = validate_map()
    print("\nTEA_VERB_MAP (validated against EPIC 97 verbs):")
    for d, vs in vmap.items():
        print(f"  {d:42s} -> {sorted(vs)}")

    model = init_module(module_name="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar",
                        frames_per_clip=32, frames_per_second=8, resolution=256,
                        checkpoint=str(ROOT / "checkpoints" / "vitl.pt"), model_kwargs=PRETRAIN_KWARGS,
                        wrapper_kwargs=WRAPPER_KWARGS, device=device)
    clf = load_classifier(dec["dims"], device)
    at = torch.tensor([ANTICIPATION_S], device=device)

    vids = sorted(p.stem for p in CLIPS.glob("*.mp4"))
    rows = []  # (desc, hit1, hit5, n_acceptable)
    for vid in vids:
        try:
            tl = egoper.ground_truth("tea", vid)["timeline"]
        except KeyError:
            continue
        for r in tl:
            if r["desc"] not in vmap:
                continue  # only the 10 normal tea actions (skip BG / error-specific actions)
            mid = (r["start"] + r["end"]) / 2
            x = load_clip(CLIPS / f"{vid}.mp4", float(mid - ANTICIPATION_S), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = clf(model(x, at))
            top5 = [dec["verb"](i) for i in out["verb"][0].float().topk(5).indices.tolist()]
            acc = vmap[r["desc"]]
            rows.append((r["desc"], int(top5[0] in acc), int(bool(set(top5) & acc)), len(acc)))

    df = pd.DataFrame(rows, columns=["action", "hit1", "hit5", "n_acc"])
    df["rand_top5"] = 1 - (1 - df["n_acc"] / n_verbs) ** 5   # per-segment chance a random top-5 hits
    # per-action + overall
    g = df.groupby("action").agg(n=("hit1", "size"), top1=("hit1", "mean"), top5=("hit5", "mean"),
                                 n_acc=("n_acc", "first"), rand_top5=("rand_top5", "mean")).reset_index()
    overall = pd.DataFrame([{"action": "OVERALL", "n": len(df), "top1": df.hit1.mean(),
                             "top5": df.hit5.mean(), "n_acc": np.nan, "rand_top5": df.rand_top5.mean()}])
    out = pd.concat([g.sort_values("top5", ascending=False), overall], ignore_index=True)
    out.to_csv(FIG / "ek100_tea_verb_accuracy.csv", index=False)

    print(f"\n=== EK100 zero-shot verb accuracy on EgoPER tea ({len(vids)} videos, {len(df)} segments) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # bar chart: per-action top-5 (and top-1) verb accuracy
    gp = g.sort_values("top5", ascending=False)
    y = np.arange(len(gp))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y + 0.2, gp.top5, height=0.38, color="#3a7bd5", label="top-5 verb")
    ax.barh(y - 0.2, gp.top1, height=0.38, color="#9ec5fe", label="top-1 verb")
    ax.scatter(gp.rand_top5, y + 0.2, color="grey", marker="|", s=200, label="top-5 random chance", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{a}  (n={n})" for a, n in zip(gp.action, gp.n)], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("verb accuracy")
    ax.axvline(df.hit5.mean(), color="#3a7bd5", ls="--", lw=1, label=f"overall top-5 = {df.hit5.mean():.2f}")
    ax.set_title("Off-the-shelf V-JEPA 2 + EK100 (zero-shot): tea action verb accuracy", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ek100_tea_verb_accuracy.png", dpi=130)
    print(f"\nwrote {FIG/'ek100_tea_verb_accuracy.png'} and .csv")
    print("\nCAVEATS: anticipation +1s alignment; EPIC vocab lacks tea nouns (verbs are the fair "
          "metric); TEA_VERB_MAP is hand-built/subjective; subset of tea videos.")


if __name__ == "__main__":
    main()

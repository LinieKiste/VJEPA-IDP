"""Does the FROZEN V-JEPA embedding carry eXprt action signal that zero-shot EK100 misses?

Off-the-shelf EK100 transfers poorly to eXprt (third-person/wide view; see ek100_label.py). This
tests the open question: do the frozen V-JEPA 2 *embeddings* still separate the actions even though
the egocentric-trained EK100 *head* doesn't? We map each hand-annotated segment to an action class
(an EK100 verb), then compare:
  - zero-shot EK100 verb (top-1/top-5 hit of the class verb), vs
  - a probe (logreg) trained on the segment's frozen V-JEPA encoder mean-pool embedding (in-sample +
    leave-one-out), vs majority/chance baselines.

PILOT: only ~18 segments / 4 videos, classes dominated by pour/put (4 singleton classes) — so this is
a feasibility read ("is it promising enough to label more?"), not a final number. Logs to mlflow
experiment `exprt_action_probe` with a comparison bar + LOO confusion matrix.

Usage:
    .venv/bin/python exprt_probe/action_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exprt_probe"))
sys.path.insert(0, str(ROOT / "egoper_vqa"))
from ek100_tea import (ANTICIPATION_S, PRETRAIN_KWARGS, WRAPPER_KWARGS,  # noqa: E402
                       build_decoders, init_module, load_classifier)
from ek100_label import QUAL, load_segment  # noqa: E402

FIG = QUAL / "figures"


def label_to_class(text: str) -> str | None:
    """Map a free-text action note to an EK100 verb class (keyword rules)."""
    t = str(text).lower()
    if "stir" in t:
        return "mix"
    if "pour" in t or "fill" in t:
        return "pour"
    if "open" in t:
        return "open"
    if "close" in t:
        return "close"
    if "take" in t:                       # "takes out teabag"
        return "take"
    if "place" in t or "drop" in t or "put" in t:
        return "put"
    return None                            # e.g. "nothing/confused"


@torch.no_grad()
def main():
    df = pd.read_csv(QUAL / "annotations.csv")
    df["label"] = df.apply(lambda r: str(r["action"]).strip()
                           if pd.notna(r["action"]) and str(r["action"]).strip() else str(r["notes"]).strip(), axis=1)
    df = df[df["start_s"].notna() & df["stop_s"].notna()
            & ~df["notes"].astype(str).str.startswith("EXAMPLE", na=False)].copy()
    df["cls"] = df["label"].map(label_to_class)
    df = df[df["cls"].notna()].reset_index(drop=True)
    classes = sorted(df["cls"].unique())
    counts = df["cls"].value_counts().to_dict()
    print(f"{len(df)} segments, {len(classes)} action classes: {counts}")

    device = "cuda"
    dec = build_decoders()
    model = init_module(module_name="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar",
                        frames_per_clip=32, frames_per_second=8, resolution=256,
                        checkpoint=str(ROOT / "checkpoints" / "vitl.pt"), model_kwargs=PRETRAIN_KWARGS,
                        wrapper_kwargs=WRAPPER_KWARGS, device=device)
    clf = load_classifier(dec["dims"], device)
    at = torch.tensor([ANTICIPATION_S], device=device)

    # per-segment: frozen V-JEPA encoder mean-pool embedding + zero-shot EK100 top-5 verbs
    feats, zs_top1, zs_top5 = [], [], []
    for _, r in df.iterrows():
        x = load_segment(r["video"], float(r["start_s"]), float(r["stop_s"]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.encoder(x)                       # (1, N, 1024) frozen V-JEPA
            out = clf(model(x, at))
        feats.append(tokens.float().mean(1)[0].cpu().numpy())
        top5 = [dec["verb"](i) for i in out["verb"][0].float().topk(5).indices.tolist()]
        zs_top1.append(int(top5[0] == r["cls"]))
        zs_top5.append(int(r["cls"] in top5))
    X = np.stack(feats)
    y = df["cls"].to_numpy()

    # --- probe on frozen embeddings, with overfitting controls ---
    # NB: in-sample fit is VACUOUS (19 pts in 1024-d are always linearly separable) -> not used as
    # evidence. Segment-LOO leaks within-video appearance/timing (same video stays in train), so the
    # honest control is leave-one-VIDEO-out, plus a label-permutation null.
    def mk():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))

    def cv_pred(splitter, yy, grp=None):
        p = np.empty(len(yy), dtype=object)
        for tr, te in splitter.split(X, yy, grp):
            p[te] = mk().fit(X[tr], yy[tr]).predict(X[te])
        return p

    groups = df["video"].to_numpy()
    seg_pred = cv_pred(LeaveOneOut(), y)
    vid_pred = cv_pred(LeaveOneGroupOut(), y, groups)          # overfitting-controlled
    seg_loo = float((seg_pred == y).mean())
    vid_loo = float((vid_pred == y).mean())
    majority = max(counts.values()) / len(y)

    rng = np.random.default_rng(0)
    perm_seg, perm_vid = [], []
    for _ in range(300):                                      # permutation null (shuffle labels)
        ys = rng.permutation(y)
        perm_seg.append((cv_pred(LeaveOneOut(), ys) == ys).mean())
        perm_vid.append((cv_pred(LeaveOneGroupOut(), ys, groups) == ys).mean())
    perm_seg, perm_vid = np.array(perm_seg), np.array(perm_vid)
    p_seg = float((perm_seg >= seg_loo).mean()); p_vid = float((perm_vid >= vid_loo).mean())

    zs1, zs5 = float(np.mean(zs_top1)), float(np.mean(zs_top5))
    print(f"\nzero-shot EK100 verb : top-1={zs1:.2f}  top-5={zs5:.2f}")
    print(f"segment-LOO          = {seg_loo:.2f}  (perm null {perm_seg.mean():.2f}, p={p_seg:.3f})  [leaks within-video]")
    print(f"VIDEO-LOO (honest)   = {vid_loo:.2f}  (perm null {perm_vid.mean():.2f}, p={p_vid:.3f})")
    print(f"majority baseline    = {majority:.2f}")

    # --- figures ---
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["EK100 zero-shot\n(top-1)", "EK100 zero-shot\n(top-5)",
             "probe segment-LOO\n(leaks within-video)", "probe VIDEO-LOO\n(honest)"]
    vals = [zs1, zs5, seg_loo, vid_loo]
    ax.bar(names, vals, color=["#9ec5fe", "#3a7bd5", "#cdbff0", "#2f9e44"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.axhline(majority, color="grey", ls="--", lw=1, label=f"majority class = {majority:.2f}")
    ax.axhline(perm_vid.mean(), color="red", ls=":", lw=1, label=f"permutation null (video-LOO) = {perm_vid.mean():.2f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("action accuracy")
    ax.set_title(f"eXprt action labelling — overfitting-controlled\n"
                 f"{len(df)} segments / {len(classes)} classes; video-LOO p={p_vid:.3f} (pilot)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "exprt_action_probe_compare.png", dpi=130)

    cm = confusion_matrix(y, vid_pred, labels=classes)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(len(classes))); ax2.set_xticklabels(classes, rotation=45, ha="right")
    ax2.set_yticks(range(len(classes))); ax2.set_yticklabels(classes)
    ax2.set_xlabel("predicted (video-LOO)"); ax2.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.set_title("Frozen-embedding probe — leave-one-VIDEO-out confusion")
    fig2.colorbar(im, fraction=0.046); fig2.tight_layout()
    fig2.savefig(FIG / "exprt_action_probe_confusion.png", dpi=130)

    # --- mlflow ---
    mlflow.set_experiment("exprt_action_probe")
    with mlflow.start_run(run_name="vjepa_meanpool_logreg_pilot"):
        mlflow.log_params({"n_segments": len(df), "n_classes": len(classes), "classes": classes,
                           "class_counts": counts, "feature": "vjepa_encoder_meanpool_1024",
                           "probe": "logreg_balanced", "n_videos": len(set(groups))})
        mlflow.log_metrics({"zeroshot_verb_top1": zs1, "zeroshot_verb_top5": zs5,
                            "probe_segment_loo": seg_loo, "probe_video_loo": vid_loo,
                            "perm_null_segment": float(perm_seg.mean()), "perm_null_video": float(perm_vid.mean()),
                            "p_value_segment": p_seg, "p_value_video": p_vid, "majority_baseline": majority})
        mlflow.log_artifact(str(FIG / "exprt_action_probe_compare.png"))
        mlflow.log_artifact(str(FIG / "exprt_action_probe_confusion.png"))
    print(f"\nwrote figures to {FIG} and logged to mlflow experiment 'exprt_action_probe'")
    print("PILOT caveat: 19 segments, 4 singleton classes; in-sample is vacuous (1024-d). The honest "
          "number is VIDEO-LOO with its permutation p-value; segment-LOO over-states due to leakage.")


if __name__ == "__main__":
    main()

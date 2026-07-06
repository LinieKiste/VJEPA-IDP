"""Fine-tune the eXprt action probe WITH data augmentation, evaluated without overfitting.

Same task as action_probe.py (classify hand-annotated eXprt action segments from frozen V-JEPA 2
embeddings), but each training segment is expanded with input-level augmentation and re-encoded:
  - horizontal MIRROR (flip)
  - temporal clip-length JITTER (the 32 frames cover a 0.8-1.2x window, slightly shifted)
  - random CROP (256 crop from a 288 working frame)
The frozen encoder is re-run on every augmented clip, so augmentation acts in pixel space.

HONEST EVAL (no overfitting): leave-one-VIDEO-out (augment only the training videos; test on the
clean center-crop view of the held-out video) + a label-permutation null. Compared head-to-head with
the no-augmentation probe. Logs to mlflow `exprt_action_probe`.

Usage:
    .venv/bin/python exprt_probe/action_probe_aug.py            # ~10-20 min on GPU
    .venv/bin/python exprt_probe/action_probe_aug.py --k 128 --perm 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exprt_probe"))
from _encoder import load_encoder  # noqa: E402
from action_probe import label_to_class  # noqa: E402
from ek100_label import DATA, FPS, WATCH_MAP, QUAL  # noqa: E402

WORK, CROP, NF = 288, 256, 32
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
FIG = QUAL / "figures"


def load_pool(video, s, e):
    """Load frames over [s,e] padded, resized to shorter side WORK -> (P,H,W,3) uint8 + times."""
    paths = sorted((DATA / WATCH_MAP[video]).glob("frame_*.png"))
    n = len(paths)
    m = 0.3 * (e - s)
    i0, i1 = max(0, int((s - m) * FPS)), min(n - 1, int((e + m) * FPS))
    idxs = np.linspace(i0, i1, min(i1 - i0 + 1, 96)).round().astype(int)
    frames = []
    for i in idxs:
        im = Image.open(paths[i]).convert("RGB")
        w, h = im.size
        nw, nh = (WORK, round(h * WORK / w)) if w <= h else (round(w * WORK / h), WORK)
        frames.append(np.asarray(im.resize((nw, nh), Image.BILINEAR)))
    return np.stack(frames), idxs / FPS


def make_clip(arr, times, s, e, augment, rng):
    """Build a (NF,CROP,CROP,3) uint8 clip; augment = mirror + temporal jitter + random crop."""
    if augment:
        span = (e - s) * rng.uniform(0.8, 1.2)
        c = (s + e) / 2 + rng.uniform(-0.15, 0.15) * (e - s)
        a, b = c - span / 2, c + span / 2
    else:
        a, b = s, e
    ti = np.clip(np.searchsorted(times, np.linspace(a, b, NF)), 0, len(times) - 1)
    H, W = arr.shape[1:3]
    if augment:
        x0, y0 = int(rng.integers(0, W - CROP + 1)), int(rng.integers(0, H - CROP + 1))
        flip = rng.random() < 0.5
    else:
        x0, y0, flip = (W - CROP) // 2, (H - CROP) // 2, False
    clip = arr[ti][:, y0:y0 + CROP, x0:x0 + CROP, :]
    return clip[:, :, ::-1, :] if flip else clip


@torch.no_grad()
def encode(enc, clips, device, batch):
    """list of (NF,CROP,CROP,3) uint8 -> (len, 1024) mean-pooled frozen embeddings."""
    out = []
    for i in range(0, len(clips), batch):
        arr = np.ascontiguousarray(np.stack(clips[i:i + batch]))           # (b,NF,H,W,3)
        x = torch.from_numpy(arr).to(device).permute(0, 4, 1, 2, 3).float().div_(255)
        x = (x - MEAN.to(device)) / STD.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = enc(x)
        out.append(tok.float().mean(1).cpu().numpy())
    return np.concatenate(out)


def video_loo(clean, aug, yv, groups, use_aug):
    """Leave-one-VIDEO-out: train on (clean[+aug]) of train videos, test clean of held-out video."""
    pred = np.empty(len(yv), dtype=object)
    for tr, te in LeaveOneGroupOut().split(np.arange(len(yv)), yv, groups):
        Xtr, ytr = [], []
        for i in tr:
            Xtr.append(clean[i][None]); ytr.append(yv[i])
            if use_aug:
                Xtr.append(aug[i]); ytr += [yv[i]] * len(aug[i])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
        clf.fit(np.concatenate(Xtr), np.array(ytr))
        pred[te] = clf.predict(np.stack([clean[i] for i in te]))
    return float((pred == yv).mean()), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128, help="augmented views per segment")
    ap.add_argument("--perm", type=int, default=300, help="permutation reps")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(QUAL / "annotations.csv")
    if "label" not in df.columns:                                   # legacy schema (action/notes cols)
        df["label"] = df.apply(lambda r: str(r["action"]).strip()
                               if pd.notna(r.get("action")) and str(r["action"]).strip()
                               else str(r.get("notes", "")).strip(), axis=1)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["start_s"].notna() & df["stop_s"].notna()
            & ~df["label"].str.startswith("EXAMPLE", na=False)].copy()
    df["cls"] = df["label"].map(label_to_class)
    df = df[df["cls"].notna()].reset_index(drop=True)
    y = df["cls"].to_numpy(); groups = df["video"].to_numpy()
    counts = df["cls"].value_counts().to_dict()
    majority = max(counts.values()) / len(y)
    print(f"{len(df)} segments, classes {counts}; augment k={args.k}")

    enc = load_encoder(img_size=CROP, num_frames=NF, device="cuda")
    rng = np.random.default_rng(args.seed)
    clean, aug = [], []
    for _, r in df.iterrows():
        arr, times = load_pool(r["video"], float(r["start_s"]), float(r["stop_s"]))
        clean.append(encode(enc, [make_clip(arr, times, float(r["start_s"]), float(r["stop_s"]), False, rng)],
                            "cuda", args.batch)[0])
        clips = [make_clip(arr, times, float(r["start_s"]), float(r["stop_s"]), True, rng) for _ in range(args.k)]
        aug.append(encode(enc, clips, "cuda", args.batch))
    clean = np.stack(clean)
    print(f"encoded {len(clean)} clean + {len(clean)*args.k} augmented clips in {time.time()-t0:.0f}s")

    noaug_acc, _ = video_loo(clean, aug, y, groups, use_aug=False)
    aug_acc, aug_pred = video_loo(clean, aug, y, groups, use_aug=True)

    # permutation null for the augmented video-LOO
    perm = np.array([video_loo(clean, aug, rng.permutation(y), groups, use_aug=True)[0] for _ in range(args.perm)])
    p_aug = float((perm >= aug_acc).mean())
    print(f"\nvideo-LOO  no-aug = {noaug_acc:.2f}   AUGMENTED = {aug_acc:.2f}   "
          f"(perm null {perm.mean():.2f}, p={p_aug:.3f}, majority {majority:.2f})")
    print(f"total runtime {time.time()-t0:.0f}s")

    # --- figures ---
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ["probe video-LOO\n(no augmentation)", "probe video-LOO\n(augmented)"]
    ax.bar(bars, [noaug_acc, aug_acc], color=["#9ed5a0", "#2f9e44"])
    for i, v in enumerate([noaug_acc, aug_acc]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.axhline(majority, color="grey", ls="--", lw=1, label=f"majority = {majority:.2f}")
    ax.axhline(perm.mean(), color="red", ls=":", lw=1, label=f"permutation null = {perm.mean():.2f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("action accuracy (leave-one-video-out)")
    ax.set_title(f"eXprt action probe with data augmentation (honest eval)\n"
                 f"{len(df)} segments, k={args.k} aug/seg, augmented p={p_aug:.3f}", fontsize=11)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "exprt_action_probe_aug.png", dpi=130)

    classes = sorted(set(y))
    cm = confusion_matrix(y, aug_pred, labels=classes)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(cm, cmap="Greens")
    ax2.set_xticks(range(len(classes))); ax2.set_xticklabels(classes, rotation=45, ha="right")
    ax2.set_yticks(range(len(classes))); ax2.set_yticklabels(classes)
    ax2.set_xlabel("predicted (video-LOO, augmented)"); ax2.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.set_title("Augmented probe — leave-one-VIDEO-out confusion")
    fig2.colorbar(im, fraction=0.046); fig2.tight_layout()
    fig2.savefig(FIG / "exprt_action_probe_aug_confusion.png", dpi=130)

    mlflow.set_experiment("exprt_action_probe")
    with mlflow.start_run(run_name=f"augmented_k{args.k}"):
        mlflow.log_params({"n_segments": len(df), "classes": classes, "class_counts": counts,
                           "feature": "vjepa_encoder_meanpool_1024", "augment": "mirror+temporal+crop",
                           "k_aug": args.k, "eval": "video-LOO", "perm_reps": args.perm})
        mlflow.log_metrics({"video_loo_noaug": noaug_acc, "video_loo_aug": aug_acc,
                            "perm_null_aug": float(perm.mean()), "p_value_aug": p_aug,
                            "majority_baseline": majority, "runtime_s": time.time() - t0})
        mlflow.log_artifact(str(FIG / "exprt_action_probe_aug.png"))
        mlflow.log_artifact(str(FIG / "exprt_action_probe_aug_confusion.png"))
    print(f"wrote figures to {FIG} and logged to mlflow 'exprt_action_probe'")


if __name__ == "__main__":
    main()

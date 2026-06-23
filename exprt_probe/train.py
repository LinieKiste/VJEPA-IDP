"""Train a linear probe on EK100-pooled V-JEPA 2 features for eXprt anomaly classification.

Loads precomputed pooled embeddings (``pool.py`` -> ``pooled/<video_id>.npz``: one 1024-d vector
per clip from the frozen EK100-warm-started attentive pooler) and trains a small linear head with
a torch loop (per-epoch train **loss** logged to mlflow). Two evaluation levels:

  ``--level video`` (PRIMARY):  mean-pool a video's clip embeddings into ONE clean-labelled vector
    (40 videos), then leave-one-video-out (binary) / stratified k-fold (8-way). This is the
    label-faithful framing and the one that works.

  ``--level clip``  (DIAGNOSTIC): per-clip prediction with the whole-video label broadcast to every
    clip, evaluated out-of-fold (StratifiedGroupKFold by video). Near chance here -- with only 5
    videos/class the probe overfits recording-specific appearance (in-sample AUC ~1.0, OOF ~0.5)
    and most "anomaly" clips are normal-looking tea-making (no within-video localization). Kept to
    document the negative result.

Targets: ``binary`` (anomaly vs normal) and ``multiclass`` (8-way error type).

Usage:
    .venv/bin/python exprt_probe/train.py --level video --target binary
    .venv/bin/python exprt_probe/train.py --level video --target multiclass
    .venv/bin/python exprt_probe/train.py --level clip  --target binary          # negative result
    .venv/bin/python exprt_probe/train.py --level video --target binary --features_dir pooled_rand
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import dataset as ds

HERE = Path(__file__).parent


def load_clips(features_dir: str):
    """Per-clip: X (n,1024), y_bin, y_cls, groups (video-id), records dict."""
    files = sorted((HERE / features_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no pooled features in {HERE/features_dir} — run pool.py first")
    X, yb, yc, groups = [], [], [], []
    for f in files:
        a = np.load(f, allow_pickle=True)
        X.append(a["emb"]); yb.append(a["label_bin"]); yc.append(a["label_cls"])
        groups.extend([f.stem] * len(a["emb"]))
    records = {r["video_id"]: r for r in ds.list_videos()}
    return np.concatenate(X), np.concatenate(yb), np.concatenate(yc), np.asarray(groups), records


def to_video_level(X, yb, yc, groups):
    """Mean-pool each video's clip embeddings -> one clean-labelled vector per video."""
    vids = list(dict.fromkeys(groups))
    Xv = np.stack([X[groups == v].mean(0) for v in vids])
    yb_v = np.array([yb[groups == v][0] for v in vids])
    yc_v = np.array([yc[groups == v][0] for v in vids])
    return Xv, yb_v, yc_v, vids


def class_weights(y, n_classes, device):
    c = np.bincount(y, minlength=n_classes).astype(np.float64)
    return torch.tensor(c.sum() / (n_classes * np.maximum(c, 1)), dtype=torch.float32, device=device)


def fit_predict(Xtr, ytr, Xte, n_classes, args, device, log=None):
    """Standardize on train, fit a full-batch torch linear head, return test softmax probs."""
    mean, std = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    model = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes, device))
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(xt), yt); loss.backward(); opt.step()
        if log:
            log(ep, float(loss))
    with torch.no_grad():
        xe = torch.tensor((Xte - mean) / std, dtype=torch.float32, device=device)
        return torch.softmax(model(xe), dim=1).cpu().numpy()


def binary_metrics(y, prob1):
    return {"roc_auc": roc_auc_score(y, prob1), "ap": average_precision_score(y, prob1)}


def multiclass_metrics(y, pred):
    return {"acc": float((pred == y).mean()), "macro_f1": f1_score(y, pred, average="macro")}


# ---- VIDEO level (primary) ------------------------------------------------- #
def run_video(Xv, yv, n_classes, target, args):
    """LOO (binary) / repeated stratified k-fold (multiclass) over the 40 video vectors."""
    if target == "binary":
        oof = np.zeros(len(Xv))
        # mean per-epoch train-loss curve across LOO folds, for an mlflow loss trace
        curve = np.zeros(args.epochs)
        for tr, te in LeaveOneOut().split(Xv):
            oof[te] = fit_predict(Xv[tr], yv[tr], Xv[te], n_classes, args, args.device,
                                  log=lambda e, l: curve.__setitem__(e, curve[e] + l))[:, 1]
        for e in range(args.epochs):
            mlflow.log_metric("train_loss", curve[e] / len(Xv), step=e)
        return binary_metrics(yv, oof), 1
    # multiclass: repeated stratified 5-fold
    accs, f1s = [], []
    for seed in range(args.seeds):
        skf = StratifiedKFold(args.n_splits, shuffle=True, random_state=seed)
        pred = np.zeros(len(Xv), dtype=int)
        for tr, te in skf.split(Xv, yv):
            pred[te] = fit_predict(Xv[tr], yv[tr], Xv[te], n_classes, args, args.device).argmax(1)
        m = multiclass_metrics(yv, pred)
        accs.append(m["acc"]); f1s.append(m["macro_f1"])
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))}, args.seeds


# ---- CLIP level (diagnostic / negative result) ----------------------------- #
def run_clip(X, y, groups, n_classes, target, args):
    seeds_metrics = []
    for seed in range(args.seeds):
        oof = np.zeros((len(X), n_classes))
        sgkf = StratifiedGroupKFold(args.n_splits, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X, y, groups):
            oof[te] = fit_predict(X[tr], y[tr], X[te], n_classes, args, args.device)
        if target == "binary":
            m = binary_metrics(y, oof[:, 1])
        else:
            m = multiclass_metrics(y, oof.argmax(1))
        seeds_metrics.append(m)
    keys = seeds_metrics[0].keys()
    return {k: float(np.mean([s[k] for s in seeds_metrics])) for k in keys}, \
           {f"{k}_std": float(np.std([s[k] for s in seeds_metrics])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["video", "clip"], default="video")
    ap.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--features_dir", default="pooled", help="pooled/ (EK100) or pooled_rand/ (ablation)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    n_classes = 2 if args.target == "binary" else len(ds.CLASSES)
    X, yb, yc, groups, _ = load_clips(args.features_dir)

    mlflow.set_experiment("exprt_anomaly")
    with mlflow.start_run(run_name=f"{args.level}_{args.target}_{args.features_dir}"):
        mlflow.log_params({"level": args.level, "target": args.target,
                           "features_dir": args.features_dir, "warmstart": args.features_dir == "pooled",
                           "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
                           "n_splits": args.n_splits, "seeds": args.seeds,
                           "n_videos": len(set(groups)), "n_clips": len(X)})
        if args.level == "video":
            Xv, yb_v, yc_v, _ = to_video_level(X, yb, yc, groups)
            yv = yb_v if args.target == "binary" else yc_v
            print(f"VIDEO {args.target} [{args.features_dir}]: {len(Xv)} videos | dist {np.bincount(yv).tolist()}")
            metrics, n = run_video(Xv, yv, n_classes, args.target, args)
            mlflow.log_params({"eval": "LOO" if args.target == "binary" else f"{args.n_splits}fold_x{n}"})
        else:
            y = yb if args.target == "binary" else yc
            print(f"CLIP {args.target} [{args.features_dir}]: {len(X)} clips / {len(set(groups))} videos")
            mean, std = run_clip(X, y, groups, n_classes, args.target, args)
            metrics = {**mean, **std}
        mlflow.log_metrics(metrics)

    print(f"\n=== {args.level} {args.target} [{args.features_dir}] ===")
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.3f}")


if __name__ == "__main__":
    main()

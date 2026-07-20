"""Train a linear **regression** probe on pooled V-JEPA 2 features for UWLPD pouring targets.

Loads precomputed pooled embeddings (``pool.py`` -> ``pooled/<seq>.npz``: one 1024-d vector per
clip from the frozen attentive pooler) and regresses a per-window pouring target with a small torch
linear head (SmoothL1/Huber loss). Evaluation is **clip-level, grouped by sequence** (GroupKFold):
every window is a sample, but whole sequences are held out so appearance can't leak across the split.

Targets (both are mask-derived **proxies** — placeholders for the real mL trace of the Simulated
dataset's ``bowl_volume.csv``):
  ``flow``   = mean liquid-pixel area in the window.
  ``volume`` = running-max liquid area up to the window (monotone accumulation proxy).

The headline number is **test MAE vs a predict-the-train-mean baseline** (the Stage-1 go/no-go),
plus R². Reported in raw pixel units and normalized by the target std.

Usage:
    .venv/bin/python pour_probe/train.py --target flow
    .venv/bin/python pour_probe/train.py --target volume --features_dir pooled_mean
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

POOLED_ROOT = Path(os.environ.get("POUR_POOLED_ROOT", "/home/casimir/.cache/pour_probe"))


def load_clips(features_dir: str):
    """Per-clip: X (n,1024), flow, volume, groups (sequence id), and per-clip condition arrays."""
    root = POOLED_ROOT / features_dir
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no pooled features in {root} — run pool.py first")
    X, flow, volume, groups, fill, profile, combo = [], [], [], [], [], [], []
    for f in files:
        a = np.load(f, allow_pickle=True)
        n = len(a["emb"])
        X.append(a["emb"]); flow.append(a["flow"]); volume.append(a["volume"])
        groups.extend([f.stem] * n)
        fill.extend([str(a["fill"])] * n)
        profile.extend([str(a["profile"])] * n)
        combo.extend([str(a["combo"])] * n)
    return (np.concatenate(X), np.concatenate(flow).astype(np.float32),
            np.concatenate(volume).astype(np.float32), np.asarray(groups),
            np.asarray(fill), np.asarray(profile), np.asarray(combo))


def fit_predict(Xtr, ytr, Xte, args, device, log=None):
    """Standardize X and y on train, fit a full-batch torch linear regressor, return test preds (raw)."""
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    xt = torch.tensor((Xtr - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor((ytr - ym) / ysd, dtype=torch.float32, device=device).unsqueeze(1)
    model = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.SmoothL1Loss()
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(xt), yt); loss.backward(); opt.step()
        if log:
            log(ep, loss.item())
    with torch.no_grad():
        xe = torch.tensor((Xte - xm) / xs, dtype=torch.float32, device=device)
        pred = model(xe).squeeze(1).cpu().numpy() * ysd + ym
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["flow", "volume"], default="flow")
    ap.add_argument("--features_dir", default="pooled", help="pooled/ (EK100) | pooled_mean/ | pooled_rand/")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    X, flow, volume, groups, fill, profile, combo = load_clips(args.features_dir)
    y = flow if args.target == "flow" else volume

    # Group (sequence) cross-validation: OOF prediction + per-fold predict-mean baseline.
    gkf = GroupKFold(n_splits=args.n_splits)
    oof = np.zeros(len(y), dtype=np.float32)
    base = np.zeros(len(y), dtype=np.float32)
    curve = np.zeros(args.epochs)
    for tr, te in gkf.split(X, y, groups):
        oof[te] = fit_predict(X[tr], y[tr], X[te], args, args.device,
                              log=lambda e, l: curve.__setitem__(e, curve[e] + l))
        base[te] = y[tr].mean()

    mae, base_mae = mean_absolute_error(y, oof), mean_absolute_error(y, base)
    r2 = r2_score(y, oof)
    ystd = float(y.std())
    metrics = {
        "mae_px": float(mae), "baseline_mae_px": float(base_mae),
        "mae_norm": float(mae / ystd), "baseline_mae_norm": float(base_mae / ystd),
        "mae_vs_baseline": float(mae / base_mae), "r2": float(r2), "target_std_px": ystd,
    }

    import mlflow_util; mlflow_util.setup("pour_probe")
    with mlflow.start_run(run_name=f"{args.target}_{args.features_dir}"):
        mlflow.log_params({
            "target": args.target, "features_dir": args.features_dir,
            "warmstart": args.features_dir == "pooled",
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "n_splits": args.n_splits, "n_sequences": len(set(groups)), "n_clips": len(X),
        })
        for e in range(args.epochs):
            mlflow.log_metric("train_loss", curve[e] / args.n_splits, step=e)
        mlflow.log_metrics(metrics)
        # per-fill-level MAE (does volume track the coarse 30/60/90% ladder?)
        for fl in sorted(set(fill.tolist())):
            m = fill == fl
            if m.any():
                mlflow.log_metric(f"mae_px_fill_{fl.strip('%')}", float(mean_absolute_error(y[m], oof[m])))

    print(f"\n=== {args.target} [{args.features_dir}]  {len(X)} clips / {len(set(groups))} seqs ===")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.3f}")
    print(f"  {'(target range px)':<20} {y.min():.0f}..{y.max():.0f}")


if __name__ == "__main__":
    main()

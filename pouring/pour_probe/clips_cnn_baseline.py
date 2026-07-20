"""Pretrained-CNN baseline for pouring flow/volume, under the SAME frozen-backbone +
linear-probe + trial-grouped-CV protocol as the V-JEPA probe — so the numbers are
directly comparable ("does the video foundation model beat a plain ImageNet CNN?").

Backbone: frozen ImageNet ResNet-50 (torchvision IMAGENET1K_V2), 2048-d global-avg-pool
features PER FRAME. For each sliding window (1.0 s / stride 0.5 s, same as V-JEPA) we
sample 8 frames and pool their features two ways:
  - mean            (2048-d) : pure appearance (static image baseline).
  - mean ++ std     (4096-d) : appearance + per-feature temporal variation across the
                               window — a cheap motion cue. Schenck & Fox (IJRR 2018,
                               "Perceiving and Reasoning about Liquids using FCNs") found
                               temporal information is important for liquid perception,
                               so a purely static CNN would be an unfair strawman.

Probe: ridge, 4-fold CV grouped by TRIAL (the exact folds used for the V-JEPA attentive
CV), flow target at lag 0.7 s (the measured water-transit lag) and volume at lag 0.
Reports per-fold + mean±std R², for a small alpha sweep.

Related work for the writeup: SimLiquid (sim mL labels); "The Sound of Water" (Bagad et
al. 2024, arXiv 2411.11222) infers pouring rate from AUDIO — same target, other modality.

Usage:
    .venv/bin/python pour_probe/clips_cnn_baseline.py --extract   # once, GPU, ~few min
    .venv/bin/python pour_probe/clips_cnn_baseline.py             # probe (CPU)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

import clips_train_attn as ca

CACHE = Path(os.environ.get("POUR_CNN_FEATS_DIR",
                            "/home/casimir/.cache/pour_probe/clips_cnn_feats/resnet50"))
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# same disjoint trial folds as the V-JEPA attentive CV (cover all 18 trials)
FOLDS = {"A": {"8", "13", "21", "24"}, "B": {"7", "9", "11", "12"},
         "C": {"5", "15", "16", "25", "26"}, "D": {"17", "18", "20", "22", "27"}}
LAG_FLOW = 0.7


def build_backbone(device):
    from torchvision.models import resnet50, ResNet50_Weights
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()                       # -> 2048-d global-avg-pool features
    return m.eval().to(device)


@torch.no_grad()
def cnn_feats(net, frames_u8, device, bs=128):
    """(K,288,288,3) uint8 -> (K,2048) resnet features (resize 224 + ImageNet norm)."""
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
    x = (x - IMEAN.to(device)) / ISTD.to(device)
    out = []
    for s in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(x[s:s + bs]).float().cpu())
    return torch.cat(out).numpy()


def extract(cam, device, num_frames=8):
    clips = ca.load_clips(cam)
    wins = ca.build_windows({cam: clips}, 1.0, 0.5, 16, LAG_FLOW)        # flow @ lag
    vol = {(w["clip"], w["t0"]): v["volume"]
           for w, v in zip(wins, ca.build_windows({cam: clips}, 1.0, 0.5, 16, 0.0))}
    net = build_backbone(device)
    (CACHE / cam).mkdir(parents=True, exist_ok=True)
    by = {}
    for w in wins:
        by.setdefault(w["clip"], []).append(w)
    from tqdm import tqdm
    for cid, ws in tqdm(by.items(), desc=f"cnn {cam}"):
        out = CACHE / cam / f"{cid}.npz"
        if out.exists():
            continue
        c = clips[cid]; fr, fps, N = c["frames"], c["fps"], len(c["frames"])
        allf, spans = [], []
        for w in ws:
            idx = np.clip((np.linspace(w["t0"], w["t1"], num_frames) * fps).astype(int), 0, N - 1)
            spans.append((len(allf), len(allf) + num_frames)); allf.append(fr[idx])
        feats = cnn_feats(net, np.concatenate(allf), device)             # (nwin*8, 2048)
        fmean = np.stack([feats[a:b].mean(0) for a, b in spans]).astype(np.float16)
        fstd = np.stack([feats[a:b].std(0) for a, b in spans]).astype(np.float16)
        np.savez(out, fmean=fmean, fstd=fstd,
                 flow=np.array([w["flow"] for w in ws], np.float32),
                 volume=np.array([vol[(w["clip"], w["t0"])] for w in ws], np.float32),
                 trial=str(c["trial"]), clip=cid)


def load(pool):
    X, flow, volume, trial, clip = [], [], [], [], []
    for cam in ("CAM2", "CAM3"):
        for f in sorted((CACHE / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            fm = a["fmean"].astype(np.float32)
            feat = fm if pool == "mean" else np.concatenate([fm, a["fstd"].astype(np.float32)], 1)
            X.append(feat); flow.append(a["flow"]); volume.append(a["volume"])
            n = len(fm); trial += [str(a["trial"])] * n; clip += [str(a["clip"])] * n
    return (np.concatenate(X), np.concatenate(flow), np.concatenate(volume),
            np.asarray(trial), np.asarray(clip))


def cv_r2(X, y, trial, alpha, normalize=True):
    """4-fold CV R². L2-normalize each feature vector (standard for CNN features) +
    StandardScaler (robust to ResNet's near-dead ReLU units, unlike a hand-rolled
    std+eps floor which explodes them) + ridge.

    ``normalize=False`` drops the L2 row-normalization, which is REQUIRED for
    low-dimensional interpretable features (a decoded wavelength, a timestamp, a poly
    basis). Row-normalizing a 1-D feature maps every sample to the constant 1 and
    silently destroys the signal — it only makes sense as a magnitude-invariance prior
    over a high-dimensional backbone embedding."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, Normalizer
    steps = ([Normalizer()] if normalize else []) + [StandardScaler(), Ridge(alpha=alpha)]
    r2s = []
    for f, tr in FOLDS.items():
        te = np.array([t in tr for t in trial])
        pipe = make_pipeline(*steps)
        pipe.fit(X[~te], y[~te])
        r2s.append(r2_score(y[te], pipe.predict(X[te])))
    return np.array(r2s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_mlflow", action="store_true")
    args = ap.parse_args()

    if args.extract:
        for cam in ("CAM2", "CAM3"):
            extract(cam, args.device)
        print("extraction done")
        return

    if not args.no_mlflow:
        import mlflow
        import mlflow_util; mlflow_util.setup("pour_probe_baselines")
    print("=== ResNet-50 (ImageNet) frozen + ridge, 4-fold CV by trial ===")
    print("  (compare to V-JEPA attentive flow 0.81±0.04, ridge mean-pool 0.71, volume attn 0.67)\n")
    for target in ("flow", "volume"):
        for pool in ("mean", "meanstd"):
            X, flow, volume, trial, clip = load(pool)
            y = flow if target == "flow" else volume
            best = max([(cv_r2(X, y, trial, a).mean(), a)
                        for a in (1, 10, 100, 1e3, 1e4, 1e5)])
            r2s = cv_r2(X, y, trial, best[1])
            folds = "  ".join(f"{k}={v:.2f}" for k, v in zip(FOLDS, r2s))
            print(f"  {target:<7} {pool:<8} (a={best[1]:>4}): "
                  f"R²={r2s.mean():.3f}±{r2s.std():.3f}   [{folds}]")
            if not args.no_mlflow:
                import mlflow
                with mlflow.start_run(run_name=f"resnet50_{pool}_{target}"):
                    mlflow.log_params({"backbone": "resnet50_imagenet", "input_size": 224,
                                       "pool": pool, "target": target, "alpha": best[1],
                                       "n_windows": len(X), "feat_dim": X.shape[1],
                                       "folds": "trial4", "per_frame_2d": True,
                                       "lag_s": LAG_FLOW if target == "flow" else 0.0,
                                       "protocol": "frozen+ridge+4fold_by_trial"})
                    mlflow.log_metrics({"r2_mean": float(r2s.mean()),
                                        "r2_std": float(r2s.std()),
                                        **{f"r2_fold_{k}": float(v)
                                           for k, v in zip(FOLDS, r2s)}})


if __name__ == "__main__":
    main()

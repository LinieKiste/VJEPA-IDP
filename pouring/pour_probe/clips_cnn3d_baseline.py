"""FAIR temporal-CNN baseline for pouring flow/volume.

`clips_cnn_baseline.py` probes a frozen per-frame ImageNet ResNet-50 and scores ~0.00
on held-out flow. That is a structural result, not a fair fight: flow is a *temporal*
quantity and a 2D CNN applied frame-wise cannot represent motion at all (the mean/std
pooling we bolt on is a crude proxy). This module supplies the baseline the comparison
actually needs — a **video** CNN pretrained on Kinetics-400, which has genuine 3D
spatiotemporal filters — under the exact same protocol as the V-JEPA probe:

    frozen backbone -> per-window feature -> ridge -> 4-fold CV grouped by TRIAL

Backbones (torchvision, Kinetics-400):
  r2plus1d_18  512-d, native 112px  — the direct temporal analogue of ResNet-50.
  s3d         1024-d, native 224px  — closer to V-JEPA's input resolution.

r2plus1d is reported at its native 112 AND at 128, so the resolution gap to V-JEPA's
256px input is explicit rather than a hidden confound in the comparison.

Pooling: a video CNN already integrates over time, so `mean` (over the window's clip
chunks) is the honest primary. `meanstd` is kept for symmetry with the 2D baseline.

Usage:
    .venv/bin/python pouring/pour_probe/clips_cnn3d_baseline.py --extract
    .venv/bin/python pouring/pour_probe/clips_cnn3d_baseline.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn

import clips_cnn_baseline as cb          # FOLDS, LAG_FLOW, cv_r2 — identical protocol
import clips_train_attn as ca            # load_clips, build_windows — identical windows

CACHE = Path(os.environ.get("POUR_CNN3D_FEATS_DIR",
                            "/home/casimir/.cache/pour_probe/clips_cnn3d_feats"))
# Kinetics-400 video-model normalization (hand-rolled: weights.transforms() would also
# re-resize, silently overriding the crop geometry we share with the V-JEPA probe).
KMEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KSTD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

BACKBONES = {                            # name -> (builder, feat_dim, native_size)
    "r2plus1d_18": ("r2plus1d_18", 512, 112),
    "s3d": ("s3d", 1024, 224),
}


def build_backbone(name, device):
    import torchvision.models.video as tv
    if name == "r2plus1d_18":
        from torchvision.models.video import R2Plus1D_18_Weights
        m = tv.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        m.fc = nn.Identity()             # -> (B,512)
    elif name == "s3d":
        from torchvision.models.video import S3D_Weights
        m = tv.s3d(weights=S3D_Weights.KINETICS400_V1)
        m.classifier = nn.Identity()     # -> (B,1024) (s3d means over the spatial dims)
    else:
        raise ValueError(name)
    return m.eval().to(device)


@torch.no_grad()
def clip_feats(net, clips_u8, device, size, bs=8):
    """(K,T,256,256,3) uint8 -> (K,D). One forward per window-clip."""
    out = []
    for s in range(0, len(clips_u8), bs):
        x = torch.from_numpy(clips_u8[s:s + bs]).to(device)
        x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)          # (B,C,T,H,W)
        if size != 256:
            B, C, T, H, W = x.shape
            x = nn.functional.interpolate(x.reshape(B, C * T, H, W), size=size,
                                          mode="bilinear", align_corners=False)
            x = x.reshape(B, C, T, size, size)
        x = (x - KMEAN.to(device)) / KSTD.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(x).float().cpu())
    return torch.cat(out).numpy()


def extract(backbone, size, cam, device, num_frames=16):
    """Cache one feature vector per sliding window, from the SAME 288 frame cache and
    the SAME center-crop-256 geometry the V-JEPA eval path uses."""
    clips = ca.load_clips(cam)
    wins = ca.build_windows({cam: clips}, 1.0, 0.5, 16, cb.LAG_FLOW)     # flow @ lag
    vol = {(w["clip"], w["t0"]): v["volume"]
           for w, v in zip(wins, ca.build_windows({cam: clips}, 1.0, 0.5, 16, 0.0))}
    net = build_backbone(backbone, device)
    outdir = CACHE / f"{backbone}_{size}" / cam
    outdir.mkdir(parents=True, exist_ok=True)
    by = {}
    for w in wins:
        by.setdefault(w["clip"], []).append(w)
    from tqdm import tqdm
    rng = np.random.default_rng(0)
    for cid, ws in tqdm(by.items(), desc=f"{backbone}@{size} {cam}"):
        out = outdir / f"{cid}.npz"
        if out.exists():
            continue
        c = clips[cid]
        # eval-mode sampling == center crop, no jitter/flip: identical pixels to V-JEPA
        stack = np.stack([ca.sample_window(c, w, num_frames, False, rng) for w in ws])
        feats = clip_feats(net, stack, device, size)
        np.savez(out, fmean=feats.astype(np.float16),
                 flow=np.array([w["flow"] for w in ws], np.float32),
                 volume=np.array([vol[(w["clip"], w["t0"])] for w in ws], np.float32),
                 trial=str(c["trial"]), clip=cid)


def load(backbone, size):
    X, flow, volume, trial = [], [], [], []
    for cam in ("CAM2", "CAM3"):
        for f in sorted((CACHE / f"{backbone}_{size}" / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            fm = a["fmean"].astype(np.float32)
            X.append(fm); flow.append(a["flow"]); volume.append(a["volume"])
            trial += [str(a["trial"])] * len(fm)
    return (np.concatenate(X), np.concatenate(flow), np.concatenate(volume),
            np.asarray(trial))


def report(backbone, size, log_mlflow=True):
    X, flow, volume, trial = load(backbone, size)
    for target in ("flow", "volume"):
        y = flow if target == "flow" else volume
        alphas = (1, 10, 100, 1e3, 1e4, 1e5)
        best_a = max(alphas, key=lambda a: cb.cv_r2(X, y, trial, a).mean())
        r2s = cb.cv_r2(X, y, trial, best_a)
        folds = "  ".join(f"{k}={v:.2f}" for k, v in zip(cb.FOLDS, r2s))
        print(f"  {backbone:<12}@{size:<4} {target:<7} (a={best_a:>7}): "
              f"R²={r2s.mean():.3f}±{r2s.std():.3f}   [{folds}]")
        if log_mlflow:
            with mlflow.start_run(run_name=f"{backbone}_{size}_{target}"):
                mlflow.log_params({"backbone": backbone, "input_size": size, "pool": "mean",
                                   "target": target, "alpha": best_a, "n_windows": len(X),
                                   "feat_dim": X.shape[1], "folds": "trial4",
                                   "lag_s": cb.LAG_FLOW if target == "flow" else 0.0,
                                   "protocol": "frozen+ridge+4fold_by_trial"})
                mlflow.log_metrics({"r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
                                    **{f"r2_fold_{k}": float(v)
                                       for k, v in zip(cb.FOLDS, r2s)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_mlflow", action="store_true")
    args = ap.parse_args()

    configs = [("r2plus1d_18", 112), ("r2plus1d_18", 128), ("s3d", 224)]
    if args.extract:
        for bb, size in configs:
            for cam in ("CAM2", "CAM3"):
                extract(bb, size, cam, args.device)
        print("extraction done")
        return

    if not args.no_mlflow:
        import mlflow_util; mlflow_util.setup("pour_probe_baselines")
    print("=== Kinetics-400 video CNNs, frozen + ridge, 4-fold CV by trial ===")
    print("  (V-JEPA 2: attentive flow 0.81±0.04 | ridge mean-pool 0.69 | volume attn 0.67)")
    print("  (ResNet-50 per-frame strawman: flow ~0.00)\n")
    for bb, size in configs:
        report(bb, size, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()

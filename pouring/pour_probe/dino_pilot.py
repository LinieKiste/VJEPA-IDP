"""Cheap pilot: which TEMPORAL representation of frozen DINOv3 carries the flow signal?

Motivation. The first DINOv3 run concatenated 8 per-frame patch grids and fed them to the
attentive head. But DINOv3 encodes each frame independently, so every frame emits tokens
with identical positional content -- the concatenated sequence is invariant to frame
ORDER, and the pooler (3 self-attn blocks + 1-query cross-attn, both permutation-
equivariant/invariant) cannot recover motion direction even in principle. V-JEPA does not
have this problem: its RoPE runs over space AND time inside the backbone.

Before spending 80-minute training runs on guesses, cache per-frame DINOv3 features once
and ridge-probe several temporal constructions on the SAME windows/folds the real probe
uses. Minutes, not hours, and it says which representation is worth training.

    .venv/bin/python pouring/pour_probe/dino_pilot.py --extract        # ~10 min GPU
    .venv/bin/python pouring/pour_probe/dino_pilot.py --extract --roi
    .venv/bin/python pouring/pour_probe/dino_pilot.py                  # CPU probe
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

CACHE = Path("/home/casimir/.cache/pour_probe/dino_frame_feats")


def extract(roi: bool, device: str = "cuda", batch: int = 48):
    """One DINOv3 forward per FRAME of every clip -> (N,1024) mean-pooled patch features.

    Mean-pooling the patch grid throws away spatial layout, which is fine here: the
    question is what the TIME axis can support, and a 1024-d per-frame vector is enough to
    compare temporal constructions cheaply.
    """
    if roi:
        os.environ["POUR_FRAMES288_DIR"] = "/home/casimir/.cache/pour_probe/clips_frames288_roi"
    import clips_train_attn as ca            # reads FRAMES_DIR at import time
    import _dino_encoder

    enc = _dino_encoder.DinoV3VideoEncoder().eval().to(device)
    mean = ca.MEAN.to(device).squeeze(2)      # (1,3,1,1)
    std = ca.STD.to(device).squeeze(2)
    out = CACHE / ("roi" if roi else "center")
    out.mkdir(parents=True, exist_ok=True)

    for cam in ("CAM2", "CAM3"):
        (out / cam).mkdir(exist_ok=True)
        for f in sorted((ca.FRAMES_DIR / cam).glob("*.npz")):
            dst = out / cam / f.name
            if dst.exists():
                continue
            a = np.load(f, allow_pickle=True)
            fr = a["frames"][:, 16:16 + 256, 16:16 + 256]      # centre 256 of the 288
            feats = []
            with torch.no_grad():
                for s in range(0, len(fr), batch):
                    x = torch.from_numpy(fr[s:s + batch]).to(device)
                    x = x.permute(0, 3, 1, 2).float().div_(255.0)
                    x = (x - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tok = enc.model(pixel_values=x).last_hidden_state[:, enc.n_prefix:]
                    feats.append(tok.float().mean(1).cpu().numpy())
            np.savez(dst, feats=np.concatenate(feats).astype(np.float16),
                     fps=float(a["fps"]), gt_t=a["gt_t"], gt_w=a["gt_w"],
                     trial_id=str(a["trial_id"]))
        print(f"  {cam}: done", flush=True)
    print("wrote", out)


def build(crop, window_s=1.0, stride_s=0.5, lag_s=0.7, n_sample=8):
    """Per-window per-frame feature stacks + the same flow target the real probe uses."""
    src = CACHE / crop
    X, y, trial, cid, tmid = [], [], [], [], []
    for cam in ("CAM2", "CAM3"):
        for f in sorted((src / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            F, fps = a["feats"].astype(np.float32), float(a["fps"])
            gt, gw = a["gt_t"], a["gt_w"]
            dur, t0 = len(F) / fps, 0.0
            while True:
                t1 = min(dur, t0 + window_s)
                idx = np.clip((np.linspace(t0, t1, n_sample) * fps).astype(int), 0, len(F) - 1)
                X.append(F[idx])                                    # (n_sample, 1024)
                y.append((np.interp(t1 + lag_s, gt, gw) - np.interp(t0 + lag_s, gt, gw))
                         / max(t1 - t0, 1e-3))
                trial.append(str(a["trial_id"])); cid.append(f"{cam}/{f.stem}")
                tmid.append((t0 + t1) / 2)
                if t1 >= dur:
                    break
                t0 += stride_s
    return (np.stack(X), np.asarray(y, np.float32), np.asarray(trial),
            np.asarray(cid), np.asarray(tmid))


def reps(X):
    """Temporal constructions over a (W, T, 1024) per-frame stack."""
    T = X.shape[1]
    d = np.diff(X, axis=1)                                  # consecutive differences
    return {
        "mean over frames (ORDER-BLIND)": X.mean(1),
        "first+last concat": np.concatenate([X[:, 0], X[:, -1]], 1),
        "last - first": X[:, -1] - X[:, 0],
        "mean + mean|diff| (motion energy)": np.concatenate(
            [X.mean(1), np.abs(d).mean(1)], 1),
        "mean + mean diff (signed)": np.concatenate([X.mean(1), d.mean(1)], 1),
        "all frames concat (ordered)": X.reshape(len(X), -1),
        "all diffs concat (ordered)": d.reshape(len(X), -1),
        "frames + diffs concat": np.concatenate(
            [X.reshape(len(X), -1), d.reshape(len(X), -1)], 1),
    }


def fusion(n_sample=8):
    """Does DINOv3 carry anything V-JEPA does not? Concatenate the two frozen feature
    sets on identically-keyed windows and ridge them together."""
    import mlflow
    import mlflow_util
    import clips_train as ct
    import clips_train_attn as ca
    from clips_eval_protocol import oof, ridge_fp, r2

    mlflow_util.setup("pour_probe_dino_pilot")
    X, y, trial, cid, tmid = build("center", n_sample=n_sample)
    dkey = np.array([f"{c}/{t:.2f}" for c, t in zip(cid, tmid)])
    D = np.concatenate([X.mean(1), np.diff(X, axis=1).mean(1)], 1)

    cams = []
    for c in ("CAM2", "CAM3"):
        _, _, _, ci, _ = ct.load(c, "flow")
        cams += [c] * len(ci)
    Xv, yv, gv, cv, tv = ct.load_both("flow")
    yv = ca._retarget(cv, tv, "flow", 0.7, 1.0)
    vkey = np.array([f"{a}/{b}/{t:.2f}" for a, b, t in zip(np.array(cams), cv, tv)])
    pos = {k: i for i, k in enumerate(vkey)}
    keep = np.array([i for i, k in enumerate(dkey) if k in pos])
    vi = np.array([pos[dkey[i]] for i in keep])
    assert np.abs(y[keep] - yv[vi]).max() < 1e-3, "targets disagree after keying"
    yy, tt = y[keep], trial[keep]

    cfg = [("vjepa_only", Xv[vi]),
           ("dinov3_only", D[keep]),
           ("vjepa_plus_dinov3_mean", np.hstack([Xv[vi], X[keep].mean(1)])),
           ("vjepa_plus_dinov3_mean_and_diff", np.hstack([Xv[vi], D[keep]])),
           ("vjepa_plus_dinov3_signed_diff",
            np.hstack([Xv[vi], np.diff(X[keep], axis=1).mean(1)]))]
    print(f"aligned {len(keep)}/{len(dkey)} windows\n{'features':<34}{'dim':>7}{'R2':>8}")
    for name, F in cfg:
        best, best_a = -9, None
        for alpha in (10.0, 100.0, 1e3, 1e4):
            s = r2(oof(F, yy, tt, ridge_fp(alpha)), yy)
            if s > best:
                best, best_a = s, alpha
        print(f"{name:<34}{F.shape[1]:>7}{best:>8.3f}")
        with mlflow.start_run(run_name=f"dinofusion_{name}"):
            mlflow.log_params({"study": "vjepa_dinov3_fusion", "features": name,
                               "feat_dim": F.shape[1], "alpha": best_a, "crop": "center",
                               "target": "flow", "lag_s": 0.7, "window_s": 1.0,
                               "n_sample": n_sample, "n_windows": len(yy),
                               "folds": "trial4",
                               "protocol": "frozen+ridge+4fold_by_trial"})
            mlflow.log_metric("r2_oof", float(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--fusion", action="store_true",
                    help="V-JEPA + DINOv3 concatenated features (does DINOv3 add anything?)")
    ap.add_argument("--n_sample", type=int, default=8)
    args = ap.parse_args()
    if args.extract:
        extract(args.roi)
        return
    if args.fusion:
        fusion(args.n_sample)
        return

    import mlflow
    import mlflow_util
    from clips_eval_protocol import oof, ridge_fp, r2

    mlflow_util.setup("pour_probe_dino_pilot")

    for crop in ("center", "roi"):
        if not (CACHE / crop).exists():
            print(f"[skip {crop}: not extracted]")
            continue
        X, y, trial, cid, tmid = build(crop, n_sample=args.n_sample)
        print(f"\n=== DINOv3 {crop} crop | {len(y)} windows, {args.n_sample} frames/window, "
              f"flow @ lag 0.7, 4-fold OOF by trial ===")
        print(f"  {'temporal representation':<36}{'dim':>7}{'R2':>8}")
        for name, F in reps(X).items():
            best, best_a = -9, None
            for alpha in (10.0, 100.0, 1e3, 1e4):
                s = r2(oof(F, y, trial, ridge_fp(alpha)), y)
                if s > best:
                    best, best_a = s, alpha
            print(f"  {name:<36}{F.shape[1]:>7}{best:>8.3f}")
            slug = name.split("(")[0].strip().replace(" ", "_").replace("+", "and")
            with mlflow.start_run(run_name=f"dinopilot_{crop}_{slug}"):
                mlflow.log_params({"backbone": "dinov3_vitl16", "modality": "video",
                                   "crop": crop, "temporal_rep": name,
                                   "n_sample": args.n_sample, "feat_dim": F.shape[1],
                                   "alpha": best_a, "target": "flow", "lag_s": 0.7,
                                   "window_s": 1.0, "n_windows": len(y),
                                   "folds": "trial4",
                                   "protocol": "frozen+ridge+4fold_by_trial"})
                mlflow.log_metric("r2_oof", float(best))
        # the V-JEPA reference on the identical windows/folds
        import clips_train as ct
        Xv, yv, gv, cv, tv = ct.load_both("flow")
        import clips_train_attn as ca
        yv = ca._retarget(cv, tv, "flow", 0.7, 1.0)
        rv = r2(oof(Xv, yv, gv, ridge_fp(100.0)), yv)
        print(f"  {'V-JEPA 2 mean-pool ridge (reference)':<36}{Xv.shape[1]:>7}{rv:>8.3f}")
        with mlflow.start_run(run_name="dinopilot_reference_vjepa_meanpool"):
            mlflow.log_params({"backbone": "vjepa2_vitl", "modality": "video",
                               "crop": "center", "temporal_rep": "mean-pool (reference)",
                               "feat_dim": Xv.shape[1], "alpha": 100.0, "target": "flow",
                               "lag_s": 0.7, "window_s": 1.0, "n_windows": len(yv),
                               "folds": "trial4",
                               "protocol": "frozen+ridge+4fold_by_trial"})
            mlflow.log_metric("r2_oof", float(rv))


if __name__ == "__main__":
    main()

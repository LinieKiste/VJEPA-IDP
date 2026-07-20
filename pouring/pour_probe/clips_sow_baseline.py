"""Cross-modal baseline: the *Sound of Water* AUDIO foundation model, frozen, probed on
OUR clips under the identical protocol used for every video baseline.

    frozen backbone -> per-window feature -> ridge -> 4-fold CV grouped by TRIAL

This is the modality axis of the comparison. Our clips carry AAC audio, so the pouring
sound is available for free, and SoW is the strongest published audio model for exactly
this quantity (they report SOTA *for audio*; they run no vision comparison at all).

Three feature variants:
  wav2vec  768-d transformer features (mean / mean++std over the window)
  axial     64-d softmax over wavelength bins — "their task head as a feature"
  lambda     1-d decoded wavelength (+ its temporal delta) — the physics readout itself,
             the most interpretable and the most constrained

CAM2 and CAM3 are separate recordings with different microphones, so they are reported
separately as well as pooled — a gap between them is a microphone/placement effect, not
a property of the audio model.

NOTE on lag: the flow target is sampled at +0.7 s (the measured water-transit + scale
delay) exactly as for the video probes, so the modalities are compared on the same
target. Sound is generated at the receiving vessel, so audio may want a different lag
than vision — `--lag_sweep` checks that rather than assuming it.

Usage:
    .venv/bin/python pouring/pour_probe/clips_sow_baseline.py --extract
    .venv/bin/python pouring/pour_probe/clips_sow_baseline.py
    .venv/bin/python pouring/pour_probe/clips_sow_baseline.py --lag_sweep
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import torch

import clips_cnn_baseline as cb          # FOLDS, LAG_FLOW, cv_r2
import clips_train_attn as ca            # load_clips, build_windows
import sow_model as sm
import sow_physics as sp

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "datasets/pouring_processed/clips"
CACHE = Path(os.environ.get("POUR_SOW_FEATS_DIR",
                            "/home/casimir/.cache/pour_probe/clips_sow_feats"))


def extract(cam, device="cuda"):
    """One backbone forward per clip -> per-frame (49 fps) features cached to disk."""
    model = sm.load_sow_model(device=device)
    outdir = CACHE / cam
    outdir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    vids = sorted((CLIPS_DIR / cam).glob("*.mp4"))
    for v in tqdm(vids, desc=f"sow {cam}"):
        out = outdir / f"{v.stem}.npz"
        if out.exists():
            continue
        wav = sm.load_audio(v)
        with torch.no_grad():
            NS = wav.shape[-1]
            dur = NS / sm.SR
            x = wav.reshape(1, 1, 1, -1).to(device)
            t = torch.tensor([[[0.0, dur]]], dtype=torch.float32)
            o = model(x, t)
            axial = o["axial"][0, 0].float().cpu().numpy()      # (F,64)
            feats = o["feats"][0, 0].float().cpu().numpy()      # (F,768)
        lam = axial @ np.linspace(0, sm.W_MAX, sm.N_BINS)
        np.savez(out, feats=feats.astype(np.float16), axial=axial.astype(np.float16),
                 lam=lam.astype(np.float32), dur=dur)


def window_feats(arr, t0, t1, dur, mode):
    """Slice a (F,D) per-frame series to a window and pool it."""
    F = len(arr)
    a = int(np.clip(t0 / dur * F, 0, F - 1))
    b = int(np.clip(np.ceil(t1 / dur * F), a + 1, F))
    seg = arr[a:b].astype(np.float32)
    if mode == "mean":
        return seg.mean(0)
    return np.concatenate([seg.mean(0), seg.std(0)])


def load(cam_list, variant, pool, lag_s):
    X, y_flow, y_vol, trial = [], [], [], []
    for cam in cam_list:
        clips = ca.load_clips(cam)
        wins = ca.build_windows({cam: clips}, 1.0, 0.5, 16, lag_s)
        vol = {(w["clip"], w["t0"]): v["volume"]
               for w, v in zip(wins, ca.build_windows({cam: clips}, 1.0, 0.5, 16, 0.0))}
        cache = {}
        for w in wins:
            cid = w["clip"]
            if cid not in cache:
                f = CACHE / cam / f"{cid}.npz"
                if not f.exists():
                    continue
                cache[cid] = np.load(f)
            a = cache.get(cid)
            if a is None:
                continue
            dur = float(a["dur"])
            if variant == "wav2vec":
                v = window_feats(a["feats"], w["t0"], w["t1"], dur, pool)
            elif variant == "axial":
                v = window_feats(a["axial"], w["t0"], w["t1"], dur, pool)
            else:                                   # lambda: level + its change
                lam = a["lam"]
                F = len(lam)
                i0 = int(np.clip(w["t0"] / dur * F, 0, F - 1))
                i1 = int(np.clip(w["t1"] / dur * F, 0, F - 1))
                v = np.array([lam[i0], lam[i1], lam[i1] - lam[i0]], np.float32)
            X.append(v); y_flow.append(w["flow"]); y_vol.append(vol[(cid, w["t0"])])
            trial.append(w["trial"])
    return (np.stack(X), np.asarray(y_flow), np.asarray(y_vol), np.asarray(trial))


def report(cam_list, label, log_mlflow=True):
    for variant, pools in (("wav2vec", ("mean", "meanstd")),
                           ("axial", ("mean", "meanstd")),
                           ("lambda", ("raw",))):
        for pool in pools:
            for target in ("flow", "volume"):
                lag = cb.LAG_FLOW if target == "flow" else 0.0
                X, yf, yv, trial = load(cam_list, variant, pool, lag)
                y = yf if target == "flow" else yv
                # the decoded-wavelength variant is 3-d: row-normalizing it would erase it
                norm = variant != "lambda"
                alphas = ((1, 10, 100, 1e3, 1e4, 1e5) if norm
                          else (1e-2, 1e-1, 1, 10, 100))
                best_a = max(alphas,
                             key=lambda a: cb.cv_r2(X, y, trial, a, norm).mean())
                r2s = cb.cv_r2(X, y, trial, best_a, norm)
                folds = "  ".join(f"{k}={v:+.2f}" for k, v in zip(cb.FOLDS, r2s))
                print(f"  {label:<5} {variant:<8} {pool:<8} {target:<7}: "
                      f"R²={r2s.mean():+.3f}±{r2s.std():.3f}   [{folds}]")
                if log_mlflow:
                    with mlflow.start_run(
                            run_name=f"sow_{label}_{variant}_{pool}_{target}"):
                        mlflow.log_params({"backbone": "sound_of_water_wav2vec2",
                                           "modality": "audio", "cams": label,
                                           "variant": variant, "pool": pool,
                                           "target": target, "alpha": best_a,
                                           "n_windows": len(X), "feat_dim": X.shape[1],
                                           "lag_s": lag, "folds": "trial4",
                                           "protocol": "frozen+ridge+4fold_by_trial"})
                        mlflow.log_metrics({"r2_mean": float(r2s.mean()),
                                            "r2_std": float(r2s.std()),
                                            **{f"r2_fold_{k}": float(v)
                                               for k, v in zip(cb.FOLDS, r2s)}})


def lag_sweep():
    """Does the AUDIO want a different target lag than vision (0.7 s)?

    Sound is produced at the receiving vessel while our GT lag is dominated by the
    scale's own filtering, so the two need not coincide. Features are lag-independent,
    so only the target is recomputed — the same trick clips_lag_sweep.py uses.
    """
    print("=== audio lag sweep (wav2vec/mean, flow, both cams) ===")
    for lag in (0.0, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1):
        X, yf, _, trial = load(("CAM2", "CAM3"), "wav2vec", "mean", lag)
        r2 = max(cb.cv_r2(X, yf, trial, a).mean() for a in (100, 1e3, 1e4))
        print(f"  lag {lag:+.2f}s : R²={r2:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--lag_sweep", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_mlflow", action="store_true")
    args = ap.parse_args()

    if args.extract:
        for cam in ("CAM2", "CAM3"):
            extract(cam, args.device)
        print("extraction done")
        return
    if args.lag_sweep:
        lag_sweep()
        return

    if not args.no_mlflow:
        import mlflow_util; mlflow_util.setup("pour_probe_baselines")
    print("=== Sound of Water (audio FM) frozen + ridge, 4-fold CV by trial ===")
    print("  (V-JEPA 2 video: attentive flow 0.81±0.04 | r2plus1d 0.53 | ResNet-50 ~0.00)\n")
    for cam_list, label in ((("CAM2",), "CAM2"), (("CAM3",), "CAM3"),
                            (("CAM2", "CAM3"), "both")):
        report(cam_list, label, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()

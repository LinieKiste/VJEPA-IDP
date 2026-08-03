"""Headline metrics for the ATTENTIVE probe, in physical units, plus saved per-window
predictions so the deck's curve plots and the metric table come from the same numbers.

Why this exists. `clips_eval_protocol.py` scores the cheap ridge on the mean-pool cache,
which runs on CPU in a minute but understates the deliverable (ridge totals 32.6 g vs the
attentive probe's ~22 g). This runs the real probe out-of-fold over all four trial folds
and reports the metrics a reader can actually interpret:

    MAE / medAE / P90 / bias      in g/s (flow) or g (volume)
    tolerance bands               share of pours within +-10 / 25 / 50 g
    Bland-Altman limits           how far off ONE reading can be, 95%

Each fold is scored with the checkpoint trained on the other three, and the target
normalisation stats come from that fold's TRAINING windows, exactly as at training time.

GPU inference only, no training. ~8 min.

Usage:
    .venv/bin/python pouring/pour_probe/clips_headline_metrics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import clips_train_attn as ca
from _encoder import load_encoder
from head import build_head
from clips_cnn_baseline import FOLDS, LAG_FLOW
from clips_eval_protocol import (bland_altman, tolerance_table, totals_from_flow,
                                 unit_metrics)

OUT = Path("/home/casimir/.cache/pour_probe/headline_preds.npz")
CKPT = {"flow": "attn_flow_both_lag0.7_fold{}_best.pt",
        "volume": "attn_volume_both_vol{}_best.pt"}


def oof_predictions(target, device="cuda"):
    """Out-of-fold attentive predictions over both cameras, all 4 trial folds."""
    lag = LAG_FLOW if target == "flow" else 0.0
    cams = {c: ca.load_clips(c) for c in ("CAM2", "CAM3")}
    wins = ca.build_windows(cams, 1.0, 0.5, 16, lag)

    enc = load_encoder(img_size=256, num_frames=16, device=device)
    mean, std = ca.MEAN.to(device), ca.STD.to(device)

    P, Y, C, T = [], [], [], []
    for fold, trials in FOLDS.items():
        tr = [w for w in wins if w["trial"] not in trials]
        va = [w for w in wins if w["trial"] in trials]
        ytr = np.asarray([w[target] for w in tr], np.float32)
        ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

        head = build_head(1).to(device)
        head.load_state_dict(torch.load(ca.FRAMES_DIR.parent / CKPT[target].format(fold)))

        r2, mae, preds, ys = ca.run_eval(enc, head, cams, va, target, 16,
                                         device, mean, std, ymean, ystd)
        print(f"  fold {fold}: {len(va):4d} windows  R2 {r2:.3f}  MAE {mae:.2f}", flush=True)
        P.append(preds); Y.append(ys)
        C.append(np.array([w["clip"] for w in va]))
        T.append(np.array([(w["t0"] + w["t1"]) / 2 for w in va]))
    return (np.concatenate(P), np.concatenate(Y),
            np.concatenate(C), np.concatenate(T))


def report(target, pred, y, cids, tmid):
    unit = "g/s" if target == "flow" else "g"
    m = unit_metrics(pred, y, unit)
    print(f"\n=== {target}: attentive probe, 4-fold OOF, both cams ===")
    print(f"  per-window   MAE {m['MAE']:.2f} {unit}   medAE {m['medAE']:.2f}   "
          f"P90 {m['P90AE']:.2f}   bias {m['bias']:+.2f}   nMAE {m['nMAE%']:.0f}%")
    if target == "flow":
        Pt, Tt = totals_from_flow(pred, y, cids, tmid)
        mt = unit_metrics(Pt, Tt, "g")
        tol = tolerance_table(Pt, Tt)
        b, lo, hi = bland_altman(Pt, Tt)
        print(f"  per-clip TOTAL MASS (integrated flow), n={len(Pt)}")
        print(f"    MAE {mt['MAE']:.1f} g   medAE {mt['medAE']:.1f} g   "
              f"bias {mt['bias']:+.1f} g")
        print(f"    " + "   ".join(f"{k} {v:.0f}%" for k, v in tol.items()))
        print(f"    95% limits of agreement {lo:+.0f} to {hi:+.0f} g")


def ridge_volume_baselines():
    """OOF volume predictions for the two non-attentive methods the deck plots against:
    the causal clock (one global line) and V-JEPA mean-pool ridge + clock. CPU, seconds.

    Rows are keyed by (clip, window-centre) so they align with the attentive predictions
    even though the caches are built by different scripts.
    """
    import clips_train as ct
    from clips_eval_protocol import oof, ridge_fp, linear_fp

    X, y, groups, cids, tmid = ct.load_both("volume")
    clock = tmid[:, None].astype(float)
    return {
        "base_clip": cids, "base_tmid": tmid, "base_y": y,
        "base_clock": oof(clock, y, groups, linear_fp),
        "base_vjepa_clock": oof(np.hstack([X, clock]), y, groups, ridge_fp(100.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    store = {}
    for target in ("flow", "volume"):
        print(f"\n--- {target} ---", flush=True)
        p, y, c, t = oof_predictions(target, args.device)
        report(target, p, y, c, t)
        store.update({f"{target}_{k}": v for k, v in
                      zip(("pred", "y", "clip", "tmid"), (p, y, c, t))})

    print("\n--- ridge/clock volume baselines (CPU) ---", flush=True)
    store.update(ridge_volume_baselines())
    for k in ("base_clock", "base_vjepa_clock"):
        m = unit_metrics(store[k], store["base_y"], "g")
        print(f"  {k:<18} MAE {m['MAE']:.1f} g   medAE {m['medAE']:.1f} g   "
              f"nMAE {m['nMAE%']:.0f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"\nsaved per-window predictions -> {OUT}")


if __name__ == "__main__":
    main()

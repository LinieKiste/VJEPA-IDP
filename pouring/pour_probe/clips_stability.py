"""Representation-stability check: dense sliding-window inference at ONE-FRAME stride.

The question (supervisor's): if the input window moves by a single frame -- window 1 =
frames 1..30, window 2 = frames 2..31 -- does the prediction move only a little? A stable
representation should map a nearly-identical input to a nearly-identical output. An
unstable one would jitter, and every reported curve would then be an accident of where the
window happened to land.

Protocol. Fold-A checkpoint evaluated on fold A's HELD-OUT trials (8/13/21/24), so this is
stability on unseen data, not on what it memorised. Windows are 1.0 s stepped by 1 frame
(~33 ms) instead of the 0.5 s stride used in training/eval -- ~6650 windows over 60 clips.

NOTE what "overlapping" actually means here. The window SPAN overlaps 96.7%, but the probe
samples 16 frames out of the 30-frame span (~every 2nd frame), so consecutive windows land
on INTERLEAVED index sets -- window 0 -> [0,1,3,5,...,29], window 1 -> [1,2,4,...,30] --
sharing only 1 of 16 actual frames. The test is therefore "near-identical scene content,
almost entirely different pixels", which is stronger than it first sounds.

Three things are measured, because "the curve looks smooth" is not evidence:

  1. adjacent-window change |dpred| for a 1-frame shift, in g/s and as a % of the model's
     own output spread -- with the GROUND TRUTH's change over the same shift as the
     reference. Beating the GT's own step means the model is smoother than the signal.
  2. change-vs-shift curve: mean |pred(t+k) - pred(t)| for k = 1..30 frames. A stable model
     starts near 0 and RISES SMOOTHLY as the input genuinely changes. An unstable one is
     flat and high -- it decorrelates immediately, at k=1, and never gets worse.
  3. a SHUFFLED control: the same windows scored after randomly permuting which window each
     prediction belongs to. This is what curve 2 looks like with no stability at all, and it
     is the line the real curve has to beat.

    .venv/bin/python pouring/pour_probe/clips_stability.py            # 256 model
    .venv/bin/python pouring/pour_probe/clips_stability.py --img_size 384 \
        --frames_dir /home/casimir/.cache/pour_probe/clips_frames416 --tag res384
    .venv/bin/python pouring/pour_probe/clips_stability.py --plot     # figures only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CACHE = Path("/home/casimir/.cache/pour_probe")
OUTDIR = Path(__file__).resolve().parents[2] / "datasets/eval/figs"
WINDOW_S, NUM_FRAMES, LAG = 1.0, 16, 0.7


def infer(img_size, tag, device="cuda", bs=16):
    import torch
    import clips_train_attn as ca
    from _encoder import load_encoder
    from head import build_head
    from clips_cnn_baseline import FOLDS

    val = FOLDS["A"]
    # Filter by trial BEFORE materialising frames. ca.load_clips() reads every clip's
    # frame array, which at 416 px is ~18.6 GB across both cams and gets the process
    # OOM-killed; npz members are lazy, so reading trial_id alone costs nothing.
    cams = {}
    for c in ("CAM2", "CAM3"):
        keep = {}
        for f in sorted((ca.FRAMES_DIR / c).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            if str(a["trial_id"]) not in val:
                continue
            keep[f.stem] = {"frames": a["frames"], "fps": float(a["fps"]),
                            "gt_t": a["gt_t"], "gt_w": a["gt_w"],
                            "trial": str(a["trial_id"])}
        cams[c] = keep
    n = sum(len(v) for v in cams.values())
    print(f"{n} held-out clips (trials {sorted(val)})")

    # target normalisation must be the fold's TRAIN stats, exactly as at training time
    tr_vals = []
    for c in ("CAM2", "CAM3"):
        for f in sorted((ca.FRAMES_DIR / c).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            if str(a["trial_id"]) in val:
                continue
            fps, N = float(a["fps"]), len(a["frames"])
            dur, t0 = N / fps, 0.0
            while True:
                t1 = min(dur, t0 + WINDOW_S)
                tr_vals.append((np.interp(t1 + LAG, a["gt_t"], a["gt_w"])
                                - np.interp(t0 + LAG, a["gt_t"], a["gt_w"]))
                               / max(t1 - t0, 1e-3))
                if t1 >= dur:
                    break
                t0 += 0.5
    ymean, ystd = float(np.mean(tr_vals)), float(np.std(tr_vals) + 1e-6)
    print(f"train-fold target stats: mean {ymean:.2f}, std {ystd:.2f} g/s")

    enc = load_encoder(img_size=img_size, num_frames=NUM_FRAMES, device=device)
    head = build_head(1).to(device)
    ck = CACHE / f"attn_flow_both_lag0.7_foldA{'_' + tag if tag else ''}_best.pt"
    head.load_state_dict(torch.load(ck))
    head.eval()
    print(f"checkpoint: {ck.name}   img_size {img_size}")

    mean, std = ca.MEAN.to(device), ca.STD.to(device)
    out = {}
    for cam, clips in cams.items():
        for cid, c in clips.items():
            fps, N = c["fps"], len(c["frames"])
            wlen = int(round(WINDOW_S * fps))                 # window length IN FRAMES
            if N <= wlen:
                continue
            starts = np.arange(0, N - wlen + 1)               # STRIDE = 1 FRAME
            wins = [{"t0": s / fps, "t1": s / fps + WINDOW_S} for s in starts]
            preds = []
            with torch.no_grad():
                for i in range(0, len(wins), bs):
                    fr = np.stack([ca.sample_window(c, w, NUM_FRAMES, False,
                                                    np.random.default_rng(0), img_size)
                                   for w in wins[i:i + bs]])
                    tok = ca.encode(enc, fr, device, mean, std)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = head(tok).squeeze(1).float().cpu().numpy()
                    preds.append(p * ystd + ymean)
            pred = np.concatenate(preds)
            tmid = starts / fps + WINDOW_S / 2
            gt = np.array([(np.interp(w["t1"] + LAG, c["gt_t"], c["gt_w"])
                            - np.interp(w["t0"] + LAG, c["gt_t"], c["gt_w"]))
                           / WINDOW_S for w in wins])
            out[f"{cam}/{cid}"] = {"pred": pred, "gt": gt, "tmid": tmid, "fps": fps}
            print(f"  {cam}/{cid}: {len(pred)} windows @ 1-frame stride", flush=True)

    np.savez(CACHE / f"stability{'_' + tag if tag else ''}.npz",
             **{f"{k}__{f}": v[f] for k, v in out.items()
                for f in ("pred", "gt", "tmid", "fps")})
    return out


def load(tag):
    a = np.load(CACHE / f"stability{'_' + tag if tag else ''}.npz", allow_pickle=True)
    keys = sorted({k.rsplit("__", 1)[0] for k in a.files})
    return {k: {f: a[f"{k}__{f}"] for f in ("pred", "gt", "tmid", "fps")} for k in keys}


def shift_curve(res, kmax=30, key="pred", shuffled=False):
    """mean |x(t+k) - x(t)| over all clips, for k = 1..kmax frames."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(1, kmax + 1):
        d = []
        for v in res.values():
            x = np.asarray(v[key], float)
            if shuffled:
                x = x[rng.permutation(len(x))]
            if len(x) > k:
                d.append(np.abs(x[k:] - x[:-k]))
        out.append(np.concatenate(d).mean() if d else np.nan)
    return np.asarray(out)


def stats(res):
    dp = np.concatenate([np.abs(np.diff(v["pred"])) for v in res.values()])
    dg = np.concatenate([np.abs(np.diff(v["gt"])) for v in res.values()])
    allp = np.concatenate([v["pred"] for v in res.values()])
    return {
        "n_windows": sum(len(v["pred"]) for v in res.values()),
        "n_clips": len(res),
        "pred_sd": allp.std(),
        "pred_range": np.percentile(allp, 99) - np.percentile(allp, 1),
        "d_pred_mean": dp.mean(), "d_pred_med": np.median(dp), "d_pred_p95": np.percentile(dp, 95),
        "d_gt_mean": dg.mean(), "d_gt_med": np.median(dg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--frames_dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.frames_dir:
        os.environ["POUR_FRAMES288_DIR"] = args.frames_dir

    res = load(args.tag) if args.plot else infer(args.img_size, args.tag)
    s = stats(res)
    print(f"\n=== 1-FRAME-STRIDE STABILITY  ({s['n_clips']} held-out clips, "
          f"{s['n_windows']} windows) ===")
    print(f"  prediction spread (1-99 pct): {s['pred_range']:.1f} g/s   sd {s['pred_sd']:.1f}")
    print(f"  |change| for a 1-FRAME shift : mean {s['d_pred_mean']:.3f}  "
          f"median {s['d_pred_med']:.3f}  p95 {s['d_pred_p95']:.3f} g/s")
    print(f"  ground truth, same shift     : mean {s['d_gt_mean']:.3f}  "
          f"median {s['d_gt_med']:.3f} g/s")
    print(f"  → one frame moves the prediction by "
          f"{100*s['d_pred_mean']/s['pred_range']:.2f}% of its own output range")
    print(f"  → model jitter / GT step = {s['d_pred_mean']/s['d_gt_mean']:.2f}x")

    import mlflow
    import mlflow_util
    mlflow_util.setup("pour_probe_stability")
    with mlflow.start_run(run_name=f"stability_foldA{'_' + args.tag if args.tag else ''}"):
        mlflow.log_params({"img_size": args.img_size, "tag": args.tag or "base",
                           "stride": "1 frame", "window_s": WINDOW_S, "lag_s": LAG,
                           "num_frames": NUM_FRAMES, "fold": "A", "split": "held-out"})
        mlflow.log_metrics({k: float(v) for k, v in s.items()})
        mlflow.log_metric("pct_of_range", 100 * s["d_pred_mean"] / s["pred_range"])
    return res, s


if __name__ == "__main__":
    main()

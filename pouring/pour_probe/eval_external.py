"""Run the frozen V-JEPA 2 flow probe on an EXTERNAL, unlabelled-in-time pour set.

`datasets/eval/videos/` holds 9 iPhone 4K/30fps pours with a single ground-truth FINAL
volume each (`datasets/eval/gt.csv`) -- no weight trace, so there is no per-window target
and nothing to fit. This is a pure out-of-domain qualitative check: predict the flow curve,
integrate it to a volume curve, and see where the curve lands against the one number we do
have.

Protocol matches training exactly: short-side 288 -> centre 256, 1.0 s windows, 16 frames,
lag-0.7 flow checkpoints. All four trial-fold checkpoints are run and reported as an
ENSEMBLE (mean) plus their spread, which doubles as a free uncertainty band -- every fold
is equally "out of domain" here, since none of them saw this kitchen.

    .venv/bin/python pouring/pour_probe/eval_external.py            # infer + cache
    .venv/bin/python pouring/pour_probe/eval_external.py --plot     # figures only
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

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "datasets/eval"
VID_DIR = EVAL_DIR / "videos"
OUT = Path("/home/casimir/.cache/pour_probe/external_preds.npz")

WINDOW_S, STRIDE_S, NUM_FRAMES = 1.0, 0.25, 16
CKPT = "attn_flow_both_lag0.7_fold{}_best.pt"


def load_gt():
    """gt.csv is index -> final volume, with NO filename column. The only defensible
    mapping is row order == filename sort order; flagged loudly by the caller."""
    rows = np.atleast_1d(np.loadtxt(EVAL_DIR / "gt.csv", delimiter=","))
    if rows.ndim == 2:                      # tolerate an index column if one appears
        rows = rows[:, -1]
    vids = sorted(VID_DIR.glob("*.MOV"))
    if len(rows) != len(vids):
        raise SystemExit(f"gt.csv has {len(rows)} rows but {len(vids)} videos")
    return {v.stem: float(g) for v, g in zip(vids, rows)}


def decode(path, short_side=288, crop=288):
    """(N,288,288,3) uint8 at native fps, short-side resized then centre cropped."""
    import decord
    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    h, w = vr[0].shape[:2]
    scale = short_side / min(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    vr = decord.VideoReader(str(path), width=nw, height=nh)
    fr = vr.get_batch(np.arange(len(vr))).asnumpy()
    y0, x0 = (nh - crop) // 2, (nw - crop) // 2
    return fr[:, y0:y0 + crop, x0:x0 + crop], fps


def fold_target_stats(target, lag):
    """(ymean, ystd) over each fold's TRAINING windows -- the exact normalisation the
    checkpoint was trained with. Reads only the GT curves from the frames cache, never
    the frame arrays, so it costs nothing."""
    meta = {}
    for cam in ("CAM2", "CAM3"):
        for f in sorted((ca.FRAMES_DIR / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            meta[(cam, f.stem)] = (len(a["frames"]), float(a["fps"]),
                                   a["gt_t"], a["gt_w"], str(a["trial_id"]))
    stats = {}
    for fold, trials in FOLDS.items():
        vals = []
        for (cam, cid), (n, fps, gt, gw, trial) in meta.items():
            if trial in trials:
                continue
            dur, t0 = n / fps, 0.0
            while True:
                t1 = min(dur, t0 + WINDOW_S)
                vals.append((np.interp(t1 + lag, gt, gw) - np.interp(t0 + lag, gt, gw))
                            / max(t1 - t0, 1e-3))
                if t1 >= dur:
                    break
                t0 += 0.5
        v = np.asarray(vals, np.float32)
        stats[fold] = (float(v.mean()), float(v.std() + 1e-6))
    return stats


@torch.no_grad()
def infer(device="cuda"):
    stats = fold_target_stats("flow", LAG_FLOW)
    print("fold target stats (mean, std) g/s:",
          {k: (round(m, 2), round(s, 2)) for k, (m, s) in stats.items()})

    enc = load_encoder(img_size=256, num_frames=NUM_FRAMES, device=device)
    mean, std = ca.MEAN.to(device), ca.STD.to(device)
    heads = {}
    for fold in FOLDS:
        h = build_head(1).to(device)
        h.load_state_dict(torch.load(ca.FRAMES_DIR.parent / CKPT.format(fold)))
        h.eval()
        heads[fold] = h

    out = {}
    for path in sorted(VID_DIR.glob("*.MOV")):
        frames, fps = decode(path)
        dur = len(frames) / fps
        tmid, idxs = [], []
        t0 = 0.0
        while True:
            t1 = min(dur, t0 + WINDOW_S)
            idxs.append(np.clip((np.linspace(t0, t1, NUM_FRAMES) * fps).astype(int),
                                0, len(frames) - 1))
            tmid.append((t0 + t1) / 2)
            if t1 >= dur:
                break
            t0 += STRIDE_S
        preds = {f: [] for f in heads}
        for s in range(0, len(idxs), 8):
            chunk = np.stack([frames[i][:, 16:16 + 256, 16:16 + 256] for i in idxs[s:s + 8]])
            tok = enc_forward(enc, chunk, device, mean, std)
            for f, h in heads.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = h(tok).squeeze(1).float().cpu().numpy()
                m, sd = stats[f]
                preds[f].append(p * sd + m)
        P = np.stack([np.concatenate(preds[f]) for f in FOLDS])   # (4, W)
        out[path.stem] = {"tmid": np.asarray(tmid), "flow": P, "dur": dur, "fps": fps}
        tot = np.trapezoid(P.mean(0), np.asarray(tmid))
        print(f"  {path.stem}  dur {dur:5.1f}s  {P.shape[1]:3d} win  "
              f"peak {P.mean(0).max():6.1f} g/s  integral {tot:7.1f} g")

    np.savez(OUT, **{f"{k}__{f}": v[f] for k, v in out.items()
                     for f in ("tmid", "flow", "dur", "fps")})
    print(f"\nwrote {OUT}")
    return out


@torch.no_grad()
def enc_forward(enc, batch_u8, device, mean, std):
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(device)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(x).float()


def load_cached():
    a = np.load(OUT, allow_pickle=True)
    names = sorted({k.split("__")[0] for k in a.files})
    return {n: {f: a[f"{n}__{f}"] for f in ("tmid", "flow", "dur", "fps")} for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="skip inference, use the cache")
    args = ap.parse_args()
    res = load_cached() if args.plot else infer()
    gt = load_gt()
    print(f"\n{'video':<12}{'GT g':>8}{'pred g':>8}{'err':>8}{'dur s':>8}")
    for n in sorted(res):
        tot = float(np.trapezoid(res[n]["flow"].mean(0), res[n]["tmid"]))
        print(f"{n:<12}{gt[n]:8.0f}{tot:8.0f}{tot - gt[n]:+8.0f}{float(res[n]['dur']):8.1f}")

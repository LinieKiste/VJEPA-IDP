"""Extract & cache frozen V-JEPA 2 ViT-L token grids + proxy targets over UWLPD sequences.

Each pouring sequence is cut into overlapping short clips (windows in FRAME units — the
UWLPD real-robot recordings ship no reliable fps, so windows are defined over frame indices,
not seconds). Each clip is encoded by the frozen V-JEPA 2 ViT-L into its **full token grid**
``(N, 1024)`` (fp16). Alongside, a per-window **proxy** pouring target is derived from the
binary liquid masks (``ground_truth*.png``):

  - ``flow``   = mean liquid-pixel area over the window's frames (visible-liquid signal).
  - ``volume`` = running-max liquid area up to the window's last frame (monotone accumulation
                 proxy).

These are placeholders for the real mL trace (the Simulated dataset's ``bowl_volume.csv``);
they exist to smoke-test the full pipeline. The token grid — not a mean-pool — is cached
because the downstream head is an attentive pooler (``head.py``).

Adapted from ``exprt_probe/extract.py``: same ImageNet normalization, bf16-autocast encoder
forward, decode-once/batched-window logic; frames are read from the zips via
``dataset.SequenceReader`` and each npz also stores the per-window proxy targets + metadata.

Usage:
    .venv/bin/python pour_probe/extract.py                 # all 180 sequences
    .venv/bin/python pour_probe/extract.py --limit 2       # smoke test
    .venv/bin/python pour_probe/extract.py --window_frames 32 --num_frames 16 --stride_frames 16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dataset as ds  # pour_probe/dataset.py
from _encoder import load_encoder  # shared encoder loader

# Cache on the SSD: the /mnt/storage NTFS (ntfs3) driver hangs on sustained writes
# (uninterruptible D-state in do_truncate). Override with $POUR_FEATURES_DIR.
FEATURES_DIR = Path(os.environ.get("POUR_FEATURES_DIR", "/home/casimir/.cache/pour_probe/features"))

# ImageNet mean/std — V-JEPA 2 default normalization. (1,C,1,1,1) to broadcast over (B,C,T,H,W).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


@torch.no_grad()
def embed_batch(enc, batch_u8: np.ndarray, device: str, mean, std) -> np.ndarray:
    """Encode a batch of clips (B,T,H,W,3) uint8 -> token grids (B,N,1024) as fp16 numpy."""
    x = torch.from_numpy(batch_u8).to(device)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)  # (B,C,T,H,W)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = enc(x)                                  # (B, N, 1024)
    return tok.half().cpu().numpy()


def sliding_windows(total: int, num_frames: int, window_frames: int, stride_frames: int):
    """Yield (frame_positions, [pos_start, pos_end]) for overlapping clips over ``total`` frames.

    A clip spans ``window_frames`` (>= num_frames) frames, sampled to ``num_frames`` positions,
    stepping by ``stride_frames``. Positions index into the sequence's sorted frame list."""
    span = max(num_frames, window_frames)
    step = max(1, stride_frames)
    out, start = [], 0
    while start < total:
        end = min(total - 1, start + span - 1)
        idx = np.clip(np.linspace(start, end, num_frames).astype(int), 0, total - 1).tolist()
        out.append((idx, [start, end]))
        if end >= total - 1:
            break
        start += step
    return out


def extract_sequence(enc, video_id, size=256, num_frames=16, window_frames=32, stride_frames=16,
                     device="cuda", batch_clips=16):
    """Encode a sequence's windows and derive per-window proxy targets from the liquid masks."""
    with ds.SequenceReader(video_id) as sr:
        total = len(sr.frames)
        wins = sliding_windows(total, num_frames, window_frames, stride_frames)
        uniq = sorted({i for idx, _ in wins for i in idx})
        pos = {fi: k for k, fi in enumerate(uniq)}
        frames = sr.read_rgb(uniq, size)                       # (Nuniq, size, size, 3) uint8
        areas_u = sr.mask_area(uniq)                           # (Nuniq,) liquid px per unique frame

    cummax_u = np.maximum.accumulate(areas_u)                  # running max by frame position
    clip_rows = np.asarray([[pos[i] for i in idx] for idx, _ in wins])  # (n_clips, num_frames)
    flow = clip_rows.astype(int)                               # rows of unique-frame positions
    flow = areas_u[flow].mean(axis=1).astype(np.float32)       # mean area over window frames
    volume = np.asarray([cummax_u[pos[max(idx)]] for idx, _ in wins], dtype=np.float32)

    mean, std = MEAN.to(device), STD.to(device)
    feats = []
    for s in range(0, len(clip_rows), batch_clips):
        batch_u8 = frames[clip_rows[s:s + batch_clips]]        # (b,num_frames,size,size,3)
        feats.append(embed_batch(enc, batch_u8, device, mean, std))
    feats = np.concatenate(feats, axis=0)                      # (n_clips, N, 1024) fp16
    times = np.asarray([t for _, t in wins], dtype=np.int32)   # [start_frame, end_frame]
    return feats, times, flow, volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--window_frames", type=int, default=32)
    ap.add_argument("--stride_frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_clips", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N sequences (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(img_size=args.img_size, num_frames=args.num_frames, device=args.device)

    vids = ds.list_videos()
    if args.limit:
        vids = vids[: args.limit]
    print(f"extracting {len(vids)} sequences -> {FEATURES_DIR} (num_frames={args.num_frames}, "
          f"window_frames={args.window_frames}, stride_frames={args.stride_frames})")
    for rec in tqdm(vids):
        vid = rec["video_id"]
        out = FEATURES_DIR / f"{vid}.npz"
        if out.exists() and not args.overwrite:
            continue
        feats, times, flow, volume = extract_sequence(
            enc, vid, args.img_size, args.num_frames, args.window_frames, args.stride_frames,
            args.device, args.batch_clips,
        )
        np.savez(
            out,
            feats=feats,                                       # (n_clips, N, 1024) fp16
            times=times,                                       # (n_clips, 2) [start_frame, end_frame]
            flow=flow,                                         # (n_clips,) mean liquid px in window
            volume=volume,                                     # (n_clips,) running-max liquid px
            video_id=vid, combo=rec["combo"], fill=rec["fill"],
            profile=rec["profile"], motion=rec["motion"],
        )


if __name__ == "__main__":
    main()

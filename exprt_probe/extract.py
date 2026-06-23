"""Extract & cache frozen V-JEPA 2 ViT-L token grids over sliding clips of eXprt videos.

Each tea-making recording (a PNG frame sequence) is cut into overlapping short clips; each
clip is encoded by the frozen V-JEPA 2 ViT-L into its **full token grid** ``(N, 1024)`` and
cached (fp16) to ``exprt_probe/features/<video_id>.npz`` together with per-clip times and the
video's (whole-trial) labels. The token grid -- not a mean-pool -- is kept because the
downstream head is an attentive pooler (``head.py``) that attends over tokens.

Adapted from ``egoper_probe/extract.py``: same ImageNet normalization, bf16-autocast encoder
forward, and sliding-window logic, but the decord/mp4 decode is replaced by a PIL PNG loader
(eXprt ships frames, not videos) and the cache stores the token grid instead of pooled views.

Usage:
    .venv/bin/python exprt_probe/extract.py                 # all 40 videos
    .venv/bin/python exprt_probe/extract.py --limit 1       # smoke test
    .venv/bin/python exprt_probe/extract.py --num_frames 32 --window_s 4 --stride_s 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import dataset as ds  # exprt_probe/dataset.py
from _encoder import load_encoder  # shared encoder loader

FPS = 20.0  # eXprt recordings are 20 fps (dir names + camera_metadata)
FEATURES_DIR = Path(__file__).parent / "features"

# ImageNet mean/std — V-JEPA 2 default normalization. (1,C,1,1,1) to broadcast over (B,C,T,H,W).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def _load_frames(paths: list[Path], idx: list[int], size: int) -> np.ndarray:
    """Read the unique frames at ``idx``, resize shorter side -> size, center-crop -> size^2.

    Returns (len(idx), size, size, 3) uint8. PIL resize replaces decord's decode-time resize."""
    out = np.empty((len(idx), size, size, 3), dtype=np.uint8)
    for k, fi in enumerate(idx):
        im = Image.open(paths[fi]).convert("RGB")
        w, h = im.size
        if w <= h:
            nw, nh = size, round(h * size / w)
        else:
            nh, nw = size, round(w * size / h)
        im = im.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - size) // 2, (nh - size) // 2
        out[k] = np.asarray(im)[top:top + size, left:left + size, :]
    return out


@torch.no_grad()
def embed_batch(enc, batch_u8: np.ndarray, device: str, mean, std) -> np.ndarray:
    """Encode a batch of clips (B,T,H,W,3) uint8 -> token grids (B,N,1024) as fp16 numpy.

    Float cast + normalize on GPU; ViT-L forward under bf16 autocast (frozen encoder)."""
    x = torch.from_numpy(batch_u8).to(device)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)  # (B,C,T,H,W)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = enc(x)                                  # (B, N, 1024)
    return tok.half().cpu().numpy()


def sliding_windows(total: int, fps: float, num_frames: int, window_s: float, stride_s: float):
    """Yield (frame_indices, [t_start, t_end]) for overlapping clips over a frame sequence.

    Identical logic to egoper_probe: a clip spans ``window_s`` (>= num_frames), sampled to
    ``num_frames`` indices, stepping by ``stride_s``."""
    span = max(num_frames, int(round(window_s * fps)))
    step = max(1, int(round(stride_s * fps)))
    out, start = [], 0
    while start < total:
        end = min(total - 1, start + span - 1)
        idx = np.clip(np.linspace(start, end, num_frames).astype(int), 0, total - 1).tolist()
        out.append((idx, [round(start / fps, 2), round(end / fps, 2)]))
        if end >= total - 1:
            break
        start += step
    return out


def extract_video(enc, video_id, size=256, num_frames=16, window_s=4.0, stride_s=2.0,
                  device="cuda", batch_clips=16):
    """Load only the frames the clips need (one PIL read each), then encode in GPU batches."""
    paths = ds.frame_paths(video_id)
    total = len(paths)
    wins = sliding_windows(total, FPS, num_frames, window_s, stride_s)
    uniq = sorted({i for idx, _ in wins for i in idx})
    pos = {fi: k for k, fi in enumerate(uniq)}
    frames = _load_frames(paths, uniq, size)               # (Nuniq, size, size, 3) uint8
    clip_rows = np.asarray([[pos[i] for i in idx] for idx, _ in wins])  # (n_clips, num_frames)

    mean, std = MEAN.to(device), STD.to(device)
    feats = []
    for s in range(0, len(clip_rows), batch_clips):
        batch_u8 = frames[clip_rows[s:s + batch_clips]]    # (b,num_frames,size,size,3)
        feats.append(embed_batch(enc, batch_u8, device, mean, std))
    feats = np.concatenate(feats, axis=0)                  # (n_clips, N, 1024) fp16
    times = np.asarray([t for _, t in wins], dtype=np.float32)
    return feats, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--window_s", type=float, default=4.0)
    ap.add_argument("--stride_s", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_clips", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(img_size=args.img_size, num_frames=args.num_frames, device=args.device)

    vids = ds.list_videos()
    if args.limit:
        vids = vids[: args.limit]
    print(f"extracting {len(vids)} videos -> {FEATURES_DIR} "
          f"(num_frames={args.num_frames}, window_s={args.window_s}, stride_s={args.stride_s})")
    for rec in tqdm(vids):
        vid = rec["video_id"]
        out = FEATURES_DIR / f"{vid}.npz"
        if out.exists() and not args.overwrite:
            continue
        feats, times = extract_video(
            enc, vid, args.img_size, args.num_frames, args.window_s, args.stride_s,
            args.device, args.batch_clips,
        )
        n = len(feats)
        np.savez(
            out,
            feats=feats,                                     # (n_clips, N, 1024) fp16
            times=times,                                     # (n_clips, 2)
            label_bin=np.full(n, rec["binary"], dtype=np.int64),
            label_cls=np.full(n, rec["class_id"], dtype=np.int64),
            video_id=vid,
            trial_name=rec["trial_name"],
        )


if __name__ == "__main__":
    main()

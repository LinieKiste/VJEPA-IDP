"""Extract & cache frozen V-JEPA 2 ViT-L features over sliding windows of EgoPER videos.

This is the V-JEPA-centric core of the project: each video is cut into overlapping
temporal windows, each window is encoded by the frozen V-JEPA 2 ViT-L and mean-pooled
to one 1024-d vector, and labelled 1 if it overlaps any GT error segment else 0.
Cached per video to ``egoper_probe/features/<task>/<video_id>.npz`` (arrays: feats, times,
labels). Encoder runs once here; the probe (``probe.py``) then trains on the cache.

Reuses the proven loader in ``video_qa/model.py::build_encoder`` (local vjepa2 pkg +
``checkpoints/vitl.pt``). Frame preprocessing matches ``video_qa/dataset.py`` (ImageNet
normalization, resize-shorter-side + center-crop).

Usage:
    .venv/bin/python egoper_probe/extract.py --task coffee
    .venv/bin/python egoper_probe/extract.py --task coffee --window_s 4 --stride_s 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "video_qa"))    # build_encoder (pulls in the vjepa2 pkg)
sys.path.insert(0, str(ROOT / "egoper_vqa"))  # EgoPER data + GT helpers

from model import build_encoder  # noqa: E402
import egoper  # noqa: E402

# ImageNet mean/std — V-JEPA 2 default normalization (see video_qa/dataset.py).
# Shaped (1,C,1,1,1) to broadcast over a (B,C,T,H,W) batch on-device.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

FEATURES_DIR = Path(__file__).parent / "features"


def _center_crop_uint8(frames: np.ndarray, size: int) -> np.ndarray:
    """(N,H,W,3) uint8 whose shorter side already == size -> center-cropped (N,size,size,3).

    decord decoded at shorter-side==size, so cropping is a pure array slice (no resize)."""
    _, h, w, _ = frames.shape
    top, left = (h - size) // 2, (w - size) // 2
    return frames[:, top:top + size, left:left + size, :]


def load_encoder(checkpoint: Path | None = None, img_size: int = 256, num_frames: int = 16,
                 device: str = "cuda"):
    ckpt = checkpoint or (ROOT / "checkpoints" / "vitl.pt")
    enc = build_encoder("vit_large", str(ckpt), img_size, num_frames)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


# V-JEPA 2 ViT-L token grid for a 16-frame @256px window: tubelet=2 -> T'=8 temporal,
# patch=16 -> 16x16 spatial = 2048 tokens. SlowFast spatial-detail pool target.
GRID_T, GRID_H, GRID_W = 8, 16, 16
SF_SPATIAL = 4  # slow-pathway spatial grid (4x4) — preserves coarse spatial layout


@torch.no_grad()
def embed_batch(enc, batch_u8: np.ndarray, device: str, mean, std):
    """Encode a batch of windows (B,T,H,W,3) uint8 -> two pooled views of the token grid.

    Uint8 stays on CPU until here; the float cast + normalize run on the GPU, and the
    ViT-L forward runs under bf16 autocast (~2.5x faster than fp32, frozen encoder so no
    precision concern). From the single forward we pool two views:
      - ``mean``: global mean over all 2048 tokens -> (B, 1024). The simplest baseline.
      - ``sf``: SlowFast-style detail-preserving feature. Slow = mean over time then 4x4
        spatial adaptive-pool (16 tokens, keeps *where* something happened); Fast = mean
        over space keeping all 8 temporal slices (8 tokens, keeps *when* / within-window
        motion). Concat+flatten -> (B, 24*1024 = 24576).
    """
    x = torch.from_numpy(batch_u8).to(device)            # (B,T,H,W,3) uint8
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)     # (B,C,T,H,W)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = enc(x)                                     # (B, 2048, 1024)
    tok = tok.float()
    B, N, D = tok.shape
    mean_feat = tok.mean(dim=1)                          # (B, D)

    g = tok.view(B, GRID_T, GRID_H, GRID_W, D)
    # slow: temporal-mean -> (B,H,W,D) -> 4x4 spatial pool -> (B, 16, D)
    slow = g.mean(dim=1).permute(0, 3, 1, 2)            # (B, D, H, W)
    slow = torch.nn.functional.adaptive_avg_pool2d(slow, SF_SPATIAL)  # (B, D, 4, 4)
    slow = slow.permute(0, 2, 3, 1).reshape(B, -1)      # (B, 16*D)
    # fast: spatial-mean keeping all temporal slices -> (B, 8, D)
    fast = g.mean(dim=(2, 3)).reshape(B, -1)            # (B, 8*D)
    sf = torch.cat([slow, fast], dim=1)                 # (B, 24*D)
    return mean_feat.cpu().numpy(), sf.cpu().numpy()


def sliding_windows(total: int, fps: float, num_frames: int, window_s: float, stride_s: float):
    """Yield (frame_indices, [t_start, t_end]) for overlapping windows over a video."""
    span = max(num_frames, int(round(window_s * fps)))
    step = max(1, int(round(stride_s * fps)))
    out, start = [], 0
    while start < total:
        end = min(total - 1, start + span - 1)
        idx = np.clip(np.linspace(start, end, num_frames).astype(int), 0, total - 1).tolist()
        out.append((idx, [round(start / fps, 2), round(end / fps, 2)]))
        if end >= total - 1:  # reached the last frame
            break
        start += step
    return out


def label_window(task: str, video_id: str, t0: float, t1: float) -> int:
    """1 if window [t0,t1] overlaps any GT error segment (action_type != Normal)."""
    for e in egoper.ground_truth(task, video_id)["errors"]:
        if t0 < e["end"] and t1 > e["start"]:
            return 1
    return 0


def extract_video(enc, task, video_id, size=256, num_frames=16, window_s=4.0, stride_s=2.0,
                  device="cuda", batch_windows=32):
    """Decode the video ONCE at reduced resolution, read all needed frames in one batched
    pass, then encode windows in GPU batches (avoids per-window random-access I/O)."""
    path = str(egoper.video_path(task, video_id))
    probe = VideoReader(path, ctx=cpu(0))
    h, w = probe[0].asnumpy().shape[:2]
    total = len(probe)
    fps = float(probe.get_avg_fps()) or 15.0
    del probe

    # decode resized so the shorter side == size (cheap frames; center-crop happens later)
    if w <= h:
        nw, nh = size, round(h * size / w)
    else:
        nh, nw = size, round(w * size / h)
    vr = VideoReader(path, ctx=cpu(0), width=nw, height=nh)

    wins = sliding_windows(total, fps, num_frames, window_s, stride_s)
    uniq = sorted({i for idx, _ in wins for i in idx})
    pos = {fi: k for k, fi in enumerate(uniq)}
    frames = np.empty((len(uniq), nh, nw, 3), dtype=np.uint8)
    for s in range(0, len(uniq), 1500):  # chunk the read to bound peak memory
        frames[s:s + 1500] = vr.get_batch(uniq[s:s + 1500]).asnumpy()
    frames = _center_crop_uint8(frames, size)  # crop each unique frame ONCE -> (Nuniq,size,size,3)

    # map each window to row indices into the cropped unique-frame array
    win_rows = np.asarray([[pos[i] for i in idx] for idx, _ in wins])  # (W, num_frames)
    mean, std = MEAN.to(device), STD.to(device)
    feats_mean, feats_sf = [], []
    for s in range(0, len(win_rows), batch_windows):
        batch_u8 = frames[win_rows[s:s + batch_windows]]  # (b,num_frames,size,size,3) uint8
        m, sf = embed_batch(enc, batch_u8, device, mean, std)
        feats_mean.append(m)
        feats_sf.append(sf)

    feats_mean = np.concatenate(feats_mean, axis=0)
    feats_sf = np.concatenate(feats_sf, axis=0)
    times = np.asarray([t for _, t in wins], dtype=np.float32)
    labels = np.asarray([label_window(task, video_id, t0, t1) for _, (t0, t1) in wins],
                        dtype=np.int64)
    return feats_mean, feats_sf, times, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="coffee", choices=egoper.tasks())
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--window_s", type=float, default=4.0)
    ap.add_argument("--stride_s", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_windows", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (debug)")
    args = ap.parse_args()

    out_dir = FEATURES_DIR / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(img_size=args.img_size, num_frames=args.num_frames, device=args.device)

    vids = egoper.list_videos(args.task)
    if args.limit:
        vids = vids[: args.limit]
    print(f"{args.task}: extracting {len(vids)} videos -> {out_dir}")
    for vid in tqdm(vids):
        f, sf, t, l = extract_video(
            enc, args.task, vid, args.img_size, args.num_frames, args.window_s, args.stride_s,
            args.device, args.batch_windows,
        )
        # feats = mean-pool (1024-d, validated baseline); feats_sf = SlowFast detail (24576-d)
        np.savez(out_dir / f"{vid}.npz", feats=f, feats_sf=sf, times=t, labels=l)


if __name__ == "__main__":
    main()

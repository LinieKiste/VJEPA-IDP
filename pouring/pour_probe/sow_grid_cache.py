"""Frame cache for the Sound-of-Water videos, exposing the SAME interface as
`clips_train_attn.load_clips` so the trainer, window builder and augmentation run
unchanged on their data.

Two decisions worth recording:

**JPEG-in-npz.** Raw 288x288 uint8 frames for 780 videos would be ~66 GB on disk and
far more than this machine's ~16 GB of free RAM once loaded. Frames are stored
JPEG-encoded (q92) and decoded lazily per access through `JpegFrames`, which implements
the `frames[idx_array] -> (K,288,288,3) uint8` contract the sampler relies on. Cost is
~3 GB on disk and a few ms per window; the trainer is GPU-bound, so this is free.

**Container-anchored crop, not a center crop.** These videos are portrait (270x480) and
the container sits near the BOTTOM — the median annotated box centre is at 80% of frame
height. A short-side-resize + center crop, which is what our own clips use, would cut the
container out of frame entirely and the probe would be regressing on the ceiling. So we
crop a full-width square positioned so the container sits just above the bottom edge
(using `annotations/container_bboxes/<video_id>_box.npy`), which keeps both the vessel
and the space above it where the falling stream is visible. Videos without a box fall
back to the dataset-median box.

Usage:
    .venv/bin/python pouring/pour_probe/sow_grid_cache.py --build --subset S3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import sow_targets as st

ROOT = Path(__file__).resolve().parents[2]
SOW = ROOT / "datasets/sound-of-water"
CACHE = Path(os.environ.get("POUR_SOW_FRAMES_DIR",
                            "/home/casimir/.cache/pour_probe/sow_frames288"))
SIZE = 288
# dataset-median container box (x0,y0,x1,y1) for videos without an annotation
MEDIAN_BOX = np.array([76.9, 312.7, 167.8, 465.1], np.float32)


_POOL = None


def _pool():
    """Shared decode thread pool. cv2.imdecode releases the GIL, so threads give real
    parallelism here — and decoding is the training loop's bottleneck (a 16-frame window
    is 16 JPEG decodes, which serially outran the GPU forward by ~4x)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("SOW_DECODE_WORKERS", 8)))
    return _POOL


class JpegFrames:
    """Lazy (N,288,288,3) uint8 view over a list of JPEG buffers."""

    def __init__(self, buffers):
        self._b = buffers

    def __len__(self):
        return len(self._b)

    def _one(self, i):
        import cv2
        return cv2.imdecode(self._b[i], cv2.IMREAD_COLOR)[:, :, ::-1]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._one(int(idx))
        ids = [int(i) for i in np.asarray(idx).reshape(-1)]
        return np.stack(list(_pool().map(self._one, ids)))


def crop_box(video_id):
    """Full-width square crop window (y0, y1) anchored on the container."""
    f = SOW / "annotations/container_bboxes" / f"{video_id}_box.npy"
    box = np.load(f) if f.exists() else MEDIAN_BOX
    H, W = 480, 270
    side = W                                   # full width -> square of 270
    bottom = min(H, float(box[3]) + 0.10 * side)   # a little air below the container
    y0 = int(np.clip(bottom - side, 0, H - side))
    return y0, y0 + side


def build(subset="S3", quality=92):
    import cv2
    from decord import VideoReader
    from tqdm import tqdm
    d = st.metadata()
    sel = st.subsets(d)[subset]
    CACHE.mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(sel.iterrows(), total=len(sel), desc=f"frames {subset}"):
        out = CACHE / f"{r['item_id']}.npz"
        if out.exists():
            continue
        tgt = st.CACHE / f"{r['item_id']}.npz"
        if not tgt.exists():
            continue
        vr = VideoReader(str(SOW / r["file_name"]), width=270, height=480)
        y0, y1 = crop_box(r["video_id"])
        n = len(vr)
        fps = float(vr.get_avg_fps())
        arr = vr.get_batch(np.arange(n)).asnumpy()[:, y0:y1, :, :]     # (n,270,270,3)
        bufs = []
        for fr in arr:
            fr = cv2.resize(fr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", fr[:, :, ::-1],
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            assert ok
            bufs.append(buf.squeeze())
        a = np.load(tgt)
        np.savez(out, jpegs=np.array(bufs, dtype=object), fps=fps, n=n,
                 gt_t=a["t"], gt_w=a["v"], container=str(a["container"]),
                 item_id=str(r["item_id"]), allow_pickle=True)


def load_clips(subset="S2", max_items=None, drop_implausible=True, filter_subset=True):
    """Mirror of clips_train_attn.load_clips: item_id -> dict(frames, fps, gt_t, gt_w, trial).

    `gt_w` is the decoded poured volume V(t) in mL, so `build_windows` derives BOTH the
    `volume` target (V at window centre) and the `flow` target (dV/dt over the window)
    from it, exactly as it does from the scale curve on our own clips.

    `trial` carries the CONTAINER id — the grouping variable for CV, since clips of one
    container share appearance and geometry.
    """
    d = st.metadata().set_index("item_id")
    keep = subset_items(subset) if filter_subset else None
    clips = {}
    for f in sorted(CACHE.glob("*.npz")):
        if f.stem not in d.index or (keep is not None and f.stem not in keep):
            continue
        a = np.load(f, allow_pickle=True)
        v = a["gt_w"]
        if drop_implausible:
            r = d.loc[f.stem]
            cap = np.pi * r["r_cm"] ** 2 * r["meas"].get("net_height", np.nan)
            if v[-1] < 5 or (cap == cap and v[-1] > 1.25 * cap):
                continue
        clips[f.stem] = {"frames": JpegFrames(list(a["jpegs"])), "fps": float(a["fps"]),
                         "gt_t": a["gt_t"], "gt_w": v, "trial": str(a["container"])}
        if max_items and len(clips) >= max_items:
            break
    return clips


def subset_items(subset):
    return set(st.subsets(st.metadata())[subset]["item_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--subset", default="S3")
    args = ap.parse_args()
    if args.build:
        build(args.subset)
    n = len(list(CACHE.glob("*.npz")))
    sz = sum(f.stat().st_size for f in CACHE.glob("*.npz")) / 1e9
    print(f"cached {n} videos, {sz:.2f} GB")


if __name__ == "__main__":
    main()

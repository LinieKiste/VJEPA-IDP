"""CROP-TO-CONTAINER prototype (view/distance robustness of the flow probe).

Hypothesis: the attentive flow probe fails to transfer CAM2->CAM3 (0.524) because the
pooler latches onto a FIXED-FRAME cue ("look upper-left at the torso") that the
center-crop makes available. Cropping each clip to the POUR REGION instead should remove
the absolute-position cue and normalize the pour's scale, so ONE probe can work across
angles.

Builds a drop-in replacement for the clips_frames288 cache: same npz schema as
clips_grid_cache.py, but each frame is cropped to a per-clip motion ROI (the source
vessel / stream / filling container all move; the standing person's body mostly does
not) instead of center-cropped. Point $POUR_FRAMES288_DIR at this dir and the existing
trainer + cross-view eval run UNCHANGED.

ROI = weighted 5-95 percentile bounding box of temporal motion energy
(mean |frame_t - frame_{t-1}|), squared + expanded for context, floored so tiny motion
doesn't over-zoom. No detection model, no new deps.

Usage:
  # QC first (default): sample a few clips per cam, write qc_roi_crop.png, cache NOTHING
  .venv/bin/python pour_probe/clips_roi_cache.py --qc
  # then, after sign-off, build the full ROI cache for a camera:
  .venv/bin/python pour_probe/clips_roi_cache.py --full --cam CAM2
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import decord
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "datasets" / "pouring_processed" / "clips"
CACHE = Path(os.environ.get("POUR_ROI_FRAMES_DIR",
                            "/home/casimir/.cache/pour_probe/clips_frames288_roi"))
MSMALL = 160  # motion computed at this short-side resolution (speed)


def load_gt(clip_id):
    t, w = [], []
    with open(CLIPS / "csv" / f"{clip_id}.csv") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"])); w.append(float(r["weight"]))
    return np.asarray(t, np.float32), np.asarray(w, np.float32)


def motion_map(frames):
    """(T,H,W,3) uint8 -> (h,w) float motion energy at MSMALL short side, + scale to orig."""
    H, W = frames.shape[1:3]
    s = MSMALL / min(H, W)
    h, w = round(H * s), round(W * s)
    idx = np.linspace(0, len(frames) - 1, min(len(frames), 60)).astype(int)  # subsample time
    small = np.stack([cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
                      for i in idx]).astype(np.float32).mean(-1)             # grayscale (T,h,w)
    diff = np.abs(np.diff(small, axis=0)).mean(0)                            # (h,w) mean motion
    diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=2.5)
    return diff, H / h, W / w


def roi_box(frames, pct=80, lo_hi=(0.05, 0.95), expand=1.4, min_frac=0.45, down_bias=0.18):
    """Square ROI (x0,y0,x1,y1) in ORIGINAL pixel coords around the motion energy.

    Robustness fixes over the naive percentile box: (1) restrict to the LARGEST
    connected motion blob so a stray edge distractor can't drag the box (clip 0073),
    (2) bias the centre DOWNWARD toward the vessel/container — the arm above dominates
    the raw motion centroid, but the pour lands lower."""
    diff, sy, sx = motion_map(frames)
    h, w = diff.shape
    thr = np.percentile(diff, pct)
    e = np.maximum(diff - thr, 0.0)                                          # keep top motion
    if e.sum() < 1e-6:                                                        # no motion -> center box
        e = np.ones_like(diff)
    else:                                                                    # keep only the largest blob
        n, lbl, stats, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            e = np.where(lbl == k, e, 0.0)

    def bounds(marg):
        c = np.cumsum(marg); c = c / c[-1]
        return np.searchsorted(c, lo_hi[0]), np.searchsorted(c, lo_hi[1])
    x0, x1 = bounds(e.sum(0)); y0, y1 = bounds(e.sum(1))
    cx = (x0 + x1) / 2 * sx
    cy = ((y0 + y1) / 2 + down_bias * (y1 - y0)) * sy                        # shift toward the vessel
    side = max((x1 - x0) * sx, (y1 - y0) * sy) * expand
    H, W = frames.shape[1:3]
    side = max(side, min_frac * min(H, W))                                   # floor: don't over-zoom
    side = min(side, min(H, W))                                             # can't exceed frame
    half = side / 2
    cx = min(max(cx, half), W - half); cy = min(max(cy, half), H - half)     # clamp inside frame
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def crop_resize(img, box, out=288):
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA)


def square_clamp(x0, y0, x1, y1, H, W, expand, min_frac, up_bias=0.0):
    """Box -> centred square, expanded for context, floored, clamped to frame. ``up_bias``
    shifts the centre up by that fraction of the side (to include the stream above)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * expand
    side = max(side, min_frac * min(H, W))
    side = min(side, min(H, W))
    half = side / 2
    cy -= up_bias * side
    cx = min(max(cx, half), W - half); cy = min(max(cy, half), H - half)
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


# --- detector backend: off-the-shelf zero-shot GroundingDINO (no training) ---------
_DET = {}
# prompt per KNOWN vessel class (from the manifest) so we localize the RIGHT source and
# target vessel and don't grab the background electric kettle / spare bottles.
SRC_PROMPT = {"kettle": "a kettle. a teapot.", "teapot": "a teapot. a kettle.",
              "bottle": "a bottle."}
TGT_PROMPT = {"blue_mug": "a mug. a cup.", "white_mug": "a mug. a cup.",
              "glass": "a drinking glass. a cup.", "ikea_glass": "a drinking glass. a cup."}


def _detector():
    if not _DET:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        mid = "IDEA-Research/grounding-dino-tiny"
        _DET["proc"] = AutoProcessor.from_pretrained(mid)
        _DET["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
    return _DET["proc"], _DET["model"]


def _detect(frame, prompt, thr=0.25):
    import torch
    from PIL import Image
    proc, model = _detector()
    img = Image.fromarray(frame)
    inp = proc(images=img, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=thr, text_threshold=0.2,
        target_sizes=[img.size[::-1]])[0]
    return [(float(s), [float(v) for v in b]) for s, b in zip(res["scores"], res["boxes"])]


def _best_box(raw, prompt, idx):
    """Highest-confidence box for a prompt over the sampled frames, or None."""
    if not prompt:
        return None
    best = None
    for i in idx:
        for s, b in _detect(raw[i], prompt):
            if best is None or s > best[0]:
                best = (s, b)
    return best[1] if best else None


def detector_box(row, raw, expand=1.22, min_frac=0.3, gap_frac=0.6):
    """ROI from the known SOURCE + TARGET vessel classes (manifest), sampled over a few
    mid-pour frames. If both vessels are found and CLOSE, union them (source+stream+
    target). If they are far apart (a spurious source detection), anchor on the TARGET
    container alone — the reliable scale anchor — expanded upward to keep the stream.
    Falls back to the motion box if the detector finds nothing. Returns (box, used_boxes)."""
    H, W = raw.shape[1:3]
    idx = np.clip((np.array([0.4, 0.55, 0.7]) * len(raw)).astype(int), 0, len(raw) - 1)
    bs = _best_box(raw, SRC_PROMPT.get(row["source_obj"]), idx)
    bt = _best_box(raw, TGT_PROMPT.get(row["target_obj"]), idx)

    def union(*bs_):
        return (min(b[0] for b in bs_), min(b[1] for b in bs_),
                max(b[2] for b in bs_), max(b[3] for b in bs_))

    if bs and bt:
        u = union(bs, bt)
        if max(u[2] - u[0], u[3] - u[1]) <= gap_frac * min(H, W):            # close -> union
            return square_clamp(*u, H, W, expand, min_frac), [bs, bt]
        return square_clamp(*bt, H, W, 1.9, min_frac, up_bias=0.12), [bt]     # far -> target only
    if bt:
        return square_clamp(*bt, H, W, 1.9, min_frac, up_bias=0.12), [bt]
    if bs:
        return square_clamp(*bs, H, W, 1.7, min_frac), [bs]
    return roi_box(raw), []                                                  # motion fallback


def process(cam, row, backend, want_frames=True):
    cid = row["clip_id"]
    vr = decord.VideoReader(str(CLIPS / cam / f"{cid}.mp4"))
    fps = float(vr.get_avg_fps())
    raw = vr[:].asnumpy()                                                    # (T,H,W,3) uint8
    if backend == "detector":
        box, comps = detector_box(row, raw)
    else:
        box, comps = roi_box(raw), []
    frames = (np.stack([crop_resize(f, box) for f in raw]).astype(np.uint8)
              if want_frames else None)
    return fps, box, frames, raw, comps


def cache_clip(cam, row, backend):
    out = CACHE / cam / f"{row['clip_id']}.npz"
    if out.exists():
        return
    fps, box, frames, _, _ = process(cam, row, backend)
    gt_t, gt_w = load_gt(row["clip_id"])
    np.savez(out, frames=frames, fps=fps, gt_t=gt_t, gt_w=gt_w,
             trial_id=row["trial_id"], weight_final=float(row["weight_g"]),
             source_obj=row["source_obj"], target_obj=row["target_obj"], roi_box=np.asarray(box))


def qc(manifest, cams, n, backend):
    """Sample n clips spread across the manifest; draw ROI box (+ detector components)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [manifest[i] for i in np.linspace(0, len(manifest) - 1, n).astype(int)]
    fig, ax = plt.subplots(len(rows), len(cams), figsize=(4.2 * len(cams), 2.6 * len(rows)))
    ax = np.atleast_2d(ax)
    for ri, row in enumerate(rows):
        for ci, cam in enumerate(cams):
            _, box, _, raw, comps = process(cam, row, backend, want_frames=False)
            mid = raw[len(raw) // 2].copy()
            for b in comps:                                                 # detector components (thin green)
                cv2.rectangle(mid, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (40, 220, 40), max(1, mid.shape[1] // 400))
            x0, y0, x1, y1 = box                                            # final ROI (red)
            cv2.rectangle(mid, (x0, y0), (x1, y1), (255, 40, 40), max(2, mid.shape[1] // 200))
            a = ax[ri, ci]
            a.imshow(mid)
            a.set_title(f"{cam} {row['clip_id']} ({row['source_obj']}->{row['target_obj']})", fontsize=8)
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    p = Path(__file__).resolve().parent / f"qc_roi_crop_{backend}.png"
    fig.savefig(p, dpi=110); print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="detector", choices=["motion", "detector"])
    ap.add_argument("--qc", action="store_true", help="QC only: sample clips, draw ROIs, cache nothing")
    ap.add_argument("--full", action="store_true", help="build the full ROI cache for --cam")
    ap.add_argument("--cam", default="CAM2", choices=["CAM2", "CAM3"])
    ap.add_argument("--n_qc", type=int, default=8)
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(CLIPS / "clips_manifest.csv")))
    if args.full:
        (CACHE / args.cam).mkdir(parents=True, exist_ok=True)
        for row in tqdm(manifest, desc=f"roi-cache[{args.backend}] {args.cam}"):
            cache_clip(args.cam, row, args.backend)
    else:  # default = QC
        qc(manifest, ["CAM2", "CAM3"], args.n_qc, args.backend)


if __name__ == "__main__":
    main()

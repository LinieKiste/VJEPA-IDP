"""Attention-map interpretability for the attentive probe (Stage-1 decision gate).

The attentive pooler answers "which patches does the probe read?" for free: its single query
cross-attends over the 2048 V-JEPA tokens (8 temporal x 16x16 spatial). We recompute that
cross-attention softmax (the frozen ``CrossAttention`` uses SDPA, which returns no weights, so
we replicate its q/kv projections), average over heads + the 8 temporal slices -> a 16x16 spatial
map, upsample, and overlay on the window's middle RGB frame.

This visualizes the **frozen EK100-warm-started pooler** (the cheap default). Whether it lands on
the liquid stream / fill line vs a shadow/hand is the Stage-1 gate; a pooler *trained* on the real
mL target (``--train_pooler``, deferred with the Simulated-dataset mL) is the faithful version.

Usage:
    .venv/bin/python pour_probe/attn_map.py                       # a few high-flow sequences
    .venv/bin/python pour_probe/attn_map.py --video scene_left_bowl_mug_60%_hold_moderate_large_bowls
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from head import build_head, set_pooler_trainable, warm_start_from_ek100

FEATURES_DIR = Path(os.environ.get("POUR_FEATURES_DIR", "/home/casimir/.cache/pour_probe/features"))
FIG_DIR = Path(__file__).parent / "qual" / "figures"
GRID_T, GRID_HW = 8, 16


@torch.no_grad()
def cross_attn_map(head, tokens: np.ndarray, device: str) -> np.ndarray:
    """(N,1024) token grid -> (16,16) spatial attention of the pooler query, [0,1]-normalized."""
    x = torch.tensor(tokens[None].astype(np.float32), device=device)  # (1,N,D)
    pooler = head.pooler
    if pooler.blocks is not None:
        for blk in pooler.blocks:
            x = blk(x)
    cab = pooler.cross_attention_block
    xa = cab.xattn
    q = xa.q(pooler.query_tokens)                                    # (1,1,D)
    H, D = xa.num_heads, x.shape[-1]
    q = q.reshape(1, 1, H, D // H).permute(0, 2, 1, 3)               # (1,H,1,d)
    k = xa.kv(cab.norm1(x)).reshape(1, -1, 2, H, D // H).permute(2, 0, 3, 1, 4)[0]  # (1,H,N,d)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * xa.scale, dim=-1)[0, :, 0, :]  # (H,N)
    attn = attn.mean(0).reshape(GRID_T, GRID_HW, GRID_HW).mean(0).cpu().numpy()      # (16,16)
    return (attn - attn.min()) / (np.ptp(attn) + 1e-9)


def render(video_id: str, head, device: str, size: int = 256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(FEATURES_DIR / f"{video_id}.npz", allow_pickle=True)
    feats, flow, times = d["feats"], d["flow"], d["times"]
    w = int(np.argmax(flow))                                         # busiest (mid-pour) window
    amap = cross_attn_map(head, feats[w], device)

    start, end = int(times[w][0]), int(times[w][1])
    with ds.SequenceReader(video_id) as sr:
        mid = (start + end) // 2
        frame = sr.read_rgb([mid], size)[0]

    up = np.kron(amap, np.ones((size // GRID_HW, size // GRID_HW)))  # 16x16 -> size x size
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].imshow(frame); ax[0].set_title(f"frame {mid}"); ax[0].axis("off")
    ax[1].imshow(frame); ax[1].imshow(up, cmap="jet", alpha=0.5)
    ax[1].set_title(f"pooler attention (flow={flow[w]:.0f}px)"); ax[1].axis("off")
    fig.suptitle(video_id, fontsize=9)
    out = FIG_DIR / f"attn_{video_id}.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="one sequence; default = a spread of high-flow ones")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    head = build_head(1).to(args.device)
    warm_start_from_ek100(head)
    set_pooler_trainable(head, False)
    head.eval()

    if args.video:
        vids = [args.video]
    else:
        # spread across combos, pick sequences with real liquid (non-empty fill)
        recs = [r for r in ds.list_videos() if r["fill"] != "empty"
                and (FEATURES_DIR / f"{r['video_id']}.npz").exists()]
        by_combo: dict[str, dict] = {}
        for r in recs:
            by_combo.setdefault(r["combo"], r)
        vids = [r["video_id"] for r in list(by_combo.values())[: args.n]]
    for v in vids:
        render(v, head, args.device)


if __name__ == "__main__":
    main()

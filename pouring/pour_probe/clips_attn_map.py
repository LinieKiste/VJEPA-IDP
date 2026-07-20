"""Test the cross-view-transfer hypothesis visually: does the CAM2-TRAINED attentive
pooler attend to the pour action in CAM2 but to the WRONG patches when shown CAM3?

For a few held-out (val) clips, take the highest-flow window and render the trained
pooler's query cross-attention (16x16 spatial, averaged over heads + the 8 temporal
slices) overlaid on the middle frame — once for CAM2 (the view it trained on) and once
for CAM3 (the transfer view) of the SAME synchronized pour. If the hypothesis holds,
CAM2's heatmap sits on the kettle/stream/mug and CAM3's sits somewhere unrelated.

Usage: .venv/bin/python pour_probe/clips_attn_map.py --target flow
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import clips_train_attn as ca
from _encoder import load_encoder
from head import build_head

HERE = Path(__file__).resolve().parent
GRID_T, GRID_HW = 8, 16


@torch.no_grad()
def cross_attn_map(head, tokens_bnd):
    """(1,N,1024) token grid -> (16,16) pooler-query spatial attention, [0,1]-normed.
    SDPA returns no weights, so replicate the CrossAttention q/kv projections + softmax."""
    x = tokens_bnd
    pooler = head.pooler
    if pooler.blocks is not None:
        for blk in pooler.blocks:
            x = blk(x)
    cab = pooler.cross_attention_block
    xa = cab.xattn
    q = xa.q(pooler.query_tokens)
    H, D = xa.num_heads, x.shape[-1]
    q = q.reshape(1, 1, H, D // H).permute(0, 2, 1, 3)
    k = xa.kv(cab.norm1(x)).reshape(1, -1, 2, H, D // H).permute(2, 0, 3, 1, 4)[0]
    attn = torch.softmax((q @ k.transpose(-2, -1)) * xa.scale, dim=-1)[0, :, 0, :]  # (H,N)
    amap = attn.mean(0).reshape(GRID_T, GRID_HW, GRID_HW).mean(0).float().cpu().numpy()
    return (amap - amap.min()) / (np.ptp(amap) + 1e-9)


def window_frames_and_mid(clips, cam_clips, cid, w, num_frames=16):
    """center-cropped 256 frames of window w (for encoding) + its middle RGB frame."""
    c = cam_clips[cid]
    frames, fps, N = c["frames"], c["fps"], len(c["frames"])
    idx = np.clip((np.linspace(w["t0"], w["t1"], num_frames) * fps).astype(int), 0, N - 1)
    fr = frames[idx][:, 16:16 + 256, 16:16 + 256]          # center crop 256
    mid = fr[num_frames // 2]
    return np.ascontiguousarray(fr), mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="flow")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cam2 = ca.load_clips("CAM2")
    cam3 = ca.load_clips("CAM3")
    wins = ca.build_windows({"CAM2": cam2}, 1.0, 0.5, 16)
    va = [w for w in wins if w["trial"] in ca.VAL_TRIALS]

    enc = load_encoder(img_size=256, num_frames=16, device="cuda")
    head = build_head(1).to("cuda")
    head.load_state_dict(torch.load(ca.FRAMES_DIR.parent / f"attn_{args.target}_CAM2_best.pt"))
    head.eval()
    mean, std = ca.MEAN.to("cuda"), ca.STD.to("cuda")

    # pick the highest-flow window per val clip, take the top-n distinct clips
    best_per_clip = {}
    for w in va:
        if w["flow"] > best_per_clip.get(w["clip"], {"flow": -1})["flow"]:
            best_per_clip[w["clip"]] = w
    picks = sorted(best_per_clip.values(), key=lambda w: -w["flow"])[: args.n]

    fig, axes = plt.subplots(len(picks), 2, figsize=(8, 4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None]
    for row, w in zip(axes, picks):
        for ax, cam, cam_clips in [(row[0], "CAM2", cam2), (row[1], "CAM3", cam3)]:
            fr, mid = window_frames_and_mid(None, cam_clips, w["clip"], w)
            tok = ca.encode(enc, fr[None], "cuda", mean, std)
            amap = cross_attn_map(head, tok)
            up = np.kron(amap, np.ones((256 // GRID_HW, 256 // GRID_HW)))
            ax.imshow(mid); ax.imshow(up, cmap="jet", alpha=0.5); ax.axis("off")
            tag = "trained view" if cam == "CAM2" else "transfer view"
            ax.set_title(f"{cam} ({tag}) — clip {w['clip']}, flow {w['flow']:.0f} g/s", fontsize=9)
    fig.suptitle("CAM2-trained pooler attention: where it looks on CAM2 vs CAM3", fontsize=11)
    out = HERE / f"qc_attn_map_transfer_{args.target}.png"
    fig.tight_layout(); fig.savefig(out, dpi=115)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

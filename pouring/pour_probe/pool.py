"""Precompute frozen attentive-pooler embeddings from cached token grids (one vector per clip).

Running the frozen pooler once here (instead of inside the training loop, which would reload
tens of GB of token grids every epoch) writes one 1024-d vector per clip to ``pooled/<seq>.npz``,
carrying the per-window proxy targets (flow/volume) and condition metadata through unchanged.
Training (``train.py``) then loads tiny features and runs in seconds.

Token grids in ``features/`` are kept for the ``--train_pooler`` stretch (pooler in-loop).

Usage:
    .venv/bin/python pour_probe/pool.py                 # EK100 warm-started pooler (default)
    .venv/bin/python pour_probe/pool.py --pool mean     # plain mean over tokens (no pooler)
    .venv/bin/python pour_probe/pool.py --pool rand     # random pooler (ablation)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from head import build_head, set_pooler_trainable, warm_start_from_ek100

FEATURES_DIR = Path(os.environ.get("POUR_FEATURES_DIR", "/home/casimir/.cache/pour_probe/features"))
POOLED_ROOT = Path(os.environ.get("POUR_POOLED_ROOT", "/home/casimir/.cache/pour_probe"))

# keys copied straight through from the extract.py npz into the pooled npz
CARRY = ("times", "flow", "volume", "video_id", "combo", "fill", "profile", "motion")


@torch.no_grad()
def pool_grids(pooler_head, grids: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """(n_clips, N, 1024) fp16 grids -> (n_clips, 1024) fp32 pooled embeddings."""
    out = []
    for s in range(0, len(grids), batch):
        x = torch.from_numpy(np.asarray(grids[s:s + batch], dtype=np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = pooler_head.pooler(x)            # (b, 1, 1024)
        out.append(q.squeeze(1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["ek100", "mean", "rand"], default="ek100",
                    help="ek100=EK100-warm-started attentive pooler, rand=random pooler, "
                         "mean=plain mean over tokens (no pooler)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = POOLED_ROOT / {"ek100": "pooled", "rand": "pooled_rand", "mean": "pooled_mean"}[args.pool]
    out_dir.mkdir(parents=True, exist_ok=True)

    head = None
    if args.pool != "mean":
        head = build_head(1).to(args.device)
        if args.pool == "ek100":
            n, tot = warm_start_from_ek100(head)
            print(f"pooler warm-started {n}/{tot} from EK100")
        set_pooler_trainable(head, False)
        head.eval()

    files = sorted(FEATURES_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no token grids in {FEATURES_DIR} — run extract.py first")
    for f in tqdm(files, desc=f"pooling[{args.pool}]"):
        d = np.load(f, allow_pickle=True)
        if args.pool == "mean":
            emb = np.asarray(d["feats"], dtype=np.float32).mean(axis=1)  # mean over tokens
        else:
            emb = pool_grids(head, d["feats"], args.device)              # (n_clips, 1024)
        np.savez(out_dir / f.name, emb=emb.astype(np.float32), **{k: d[k] for k in CARRY})
    print(f"wrote pooled embeddings -> {out_dir}")


if __name__ == "__main__":
    main()

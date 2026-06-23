"""Precompute frozen attentive-pooler embeddings from cached token grids.

The default experiment keeps the EK100-warm-started pooler **frozen** and trains only a linear
probe on top. Running the pooler inside the training loop means reloading 9.8 GB of token grids
from disk every epoch (CPU-bound, GPU idle) -- so instead we run the frozen pooler **once** here
and cache one 1024-d vector per clip to ``pooled/<video_id>.npz``. Training (``train.py``) then
loads tiny features and runs in seconds, enabling proper multi-seed cross-validation.

(Token grids in ``features/`` are kept for the optional ``--train_pooler`` stretch experiment,
which must run the pooler in-loop.)

Usage:
    .venv/bin/python exprt_probe/pool.py                 # EK100 warm-started pooler (default)
    .venv/bin/python exprt_probe/pool.py --no_warmstart  # random-pooler ablation -> pooled_rand/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from head import build_head, set_pooler_trainable, warm_start_from_ek100

FEATURES_DIR = Path(__file__).parent / "features"


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
                    help="ek100=EK100-warm-started attentive pooler, rand=random pooler (ablation), "
                         "mean=plain mean over tokens (no pooler)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / {"ek100": "pooled", "rand": "pooled_rand", "mean": "pooled_mean"}[args.pool]
    out_dir.mkdir(exist_ok=True)

    head = None
    if args.pool != "mean":
        head = build_head(2).to(args.device)
        if args.pool == "ek100":
            n, tot = warm_start_from_ek100(head)
            print(f"pooler warm-started {n}/{tot} from EK100")
        set_pooler_trainable(head, False)
        head.eval()

    for f in tqdm(sorted(FEATURES_DIR.glob("*.npz")), desc=f"pooling[{args.pool}]"):
        d = np.load(f, allow_pickle=True)
        if args.pool == "mean":
            emb = np.asarray(d["feats"], dtype=np.float32).mean(axis=1)  # mean over tokens
        else:
            emb = pool_grids(head, d["feats"], args.device)              # (n_clips, 1024)
        np.savez(out_dir / f.name, emb=emb.astype(np.float32), times=d["times"],
                 label_bin=d["label_bin"], label_cls=d["label_cls"], video_id=str(d["video_id"]))
    print(f"wrote pooled embeddings -> {out_dir}")


if __name__ == "__main__":
    main()

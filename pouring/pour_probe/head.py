"""Attentive **regression** head for UWLPD pouring flow/volume, warm-started from the EK100 probe.

Wraps V-JEPA 2's ``AttentiveClassifier`` (attentive pooler + linear) with ``num_classes=1`` so
the linear is a single regression output. The pooler is optionally warm-started from the
Epic-Kitchens-100 attentive probe (``checkpoints/ek100-vitl-256.pt``, its "action" query) — the
same trick as ``exprt_probe/head.py``; a strong init for pooling V-JEPA kitchen/manipulation
features. This is the V-JEPA 2 Appendix-C probe spec (1 learnable query, depth-4 pooler).

The EK100 pooler is: num_queries=3, depth=4, num_heads=16, embed_dim=1024.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # repo root (pouring/pour_probe/head.py)
sys.path.insert(0, str(ROOT / "vjepa2"))

from src.models.attentive_pooler import AttentiveClassifier  # noqa: E402

EK100_CKPT = ROOT / "checkpoints" / "ek100-vitl-256.pt"
POOLER_DEPTH, POOLER_HEADS, EMBED_DIM = 4, 16, 1024


def build_head(num_outputs: int = 1, depth: int = POOLER_DEPTH, num_heads: int = POOLER_HEADS):
    """AttentiveClassifier (num_queries=1) matching the EK100 pooler shape; ``num_outputs`` linear out."""
    return AttentiveClassifier(
        embed_dim=EMBED_DIM, num_heads=num_heads, depth=depth, num_classes=num_outputs,
        complete_block=True,
    )


def warm_start_from_ek100(head: AttentiveClassifier, ckpt: Path = EK100_CKPT,
                          query_slice: int = 2) -> tuple[int, int]:
    """Load EK100 pooler weights into ``head.pooler``; return (#loaded, #pooler params).

    EK100's pooler has 3 queries (verb/noun/action); copy one (``query_slice``, default 2 = action)
    into our single-query pooler. Cross-attention + self-attn blocks transfer by name. The fresh
    ``head.linear`` regression layer keeps its random init.
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["classifiers"][0]
    src = {}
    for k, v in sd.items():
        if not k.startswith("module.pooler."):
            continue
        key = k[len("module.pooler."):]
        if key == "query_tokens":
            v = v[:, query_slice:query_slice + 1, :].clone()  # (1,3,D) -> (1,1,D)
        src[key] = v
    tgt = head.pooler.state_dict()
    loaded = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    missing = [k for k in tgt if k not in loaded]
    if missing:
        raise RuntimeError(f"EK100 warm-start: {len(missing)} pooler params unmatched: {missing[:4]}...")
    head.pooler.load_state_dict(loaded, strict=True)
    return len(loaded), len(tgt)


def set_pooler_trainable(head: AttentiveClassifier, trainable: bool) -> None:
    for p in head.pooler.parameters():
        p.requires_grad_(trainable)

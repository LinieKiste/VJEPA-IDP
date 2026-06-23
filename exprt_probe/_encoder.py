"""Shared frozen V-JEPA 2 ViT-L encoder loader (reuses video_qa/model.py::build_encoder)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "video_qa"))  # build_encoder (pulls in the local vjepa2 pkg)

from model import build_encoder  # noqa: E402


def load_encoder(checkpoint: Path | None = None, img_size: int = 256, num_frames: int = 16,
                 device: str = "cuda"):
    """Frozen V-JEPA 2 ViT-L. Forward: (B,C,T,H,W) -> (B,N,1024) patch tokens."""
    ckpt = checkpoint or (ROOT / "checkpoints" / "vitl.pt")
    enc = build_encoder("vit_large", str(ckpt), img_size, num_frames)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc

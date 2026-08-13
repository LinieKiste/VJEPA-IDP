"""Frozen DINOv3 ViT-L/16 as a drop-in replacement for the V-JEPA 2 video encoder.

DINOv3 is an IMAGE model, so "video" features are per-frame patch grids concatenated along
time. The wrapper exposes the exact interface the pouring probe already expects --
``(B,C,T,H,W) -> (B,N,1024)`` -- so ``clips_train_attn`` runs unchanged and the SAME
AttentiveClassifier head does the temporal pooling for both backbones.

Matching the comparison. At 256px/patch-16 DINOv3 emits 256 patch tokens per frame (plus a
CLS and 4 register tokens, both dropped). Taking every SECOND frame of the 16-frame window
mirrors V-JEPA's tubelet_size=2 temporal downsampling and yields 8 x 256 = 2048 tokens at
1024-d -- byte-for-byte the same token budget, sequence length and width V-JEPA 2 ViT-L
produces for the same window. So the head's capacity, the input length and the optimisation
are identical, and the only thing that differs is the representation itself.

What the comparison actually tests: DINOv3 sees each frame independently and never attends
across time, so any motion sensitivity must be constructed by the pooler from a stack of
static grids. V-JEPA 2 attends across space AND time inside the backbone. That is the
hypothesis -- the pouring flow signal is a motion signal, and a per-frame image backbone
should lose exactly that (cf. the ResNet-50 strawman at R2 ~ 0.00, which had no temporal
pooling at all; this is the strong version of that control).
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
FRAME_STRIDE = 2          # 16 -> 8 frames, mirroring V-JEPA's tubelet_size=2


def sincos_temporal(n_frames: int, dim: int, device, dtype) -> torch.Tensor:
    """Unit-RMS sinusoidal embedding over the TIME index, shape (n_frames, dim).

    Why this is load-bearing. DINOv3 encodes each frame independently, so every frame
    emits tokens carrying the SAME positional content. The attentive head is 3
    permutation-equivariant self-attention blocks followed by a permutation-INVARIANT
    single-query cross-attention pool, so without a time stamp on each token the whole
    representation is invariant to frame order and motion direction is unrecoverable in
    principle. V-JEPA has no such problem: its RoPE covers space AND time inside the
    backbone. A fixed (non-learned) embedding keeps the backbone honestly frozen.
    """
    pos = torch.arange(n_frames, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)[None, :]
    ang = pos / (10000 ** (2 * i / dim))
    e = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    return (e / e.pow(2).mean().sqrt()).to(dtype)


class DinoV3VideoEncoder(nn.Module):
    """Per-frame DINOv3 patch tokens, concatenated over time."""

    def __init__(self, model_id: str = MODEL_ID, frame_stride: int = FRAME_STRIDE,
                 temporal_embed: bool = True, temporal_scale: float = 0.5):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        # DINOv3 prepends 1 CLS + num_register_tokens registers to the patch grid; the
        # attentive pooler wants the spatial grid only, matching V-JEPA's patch tokens.
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self.frame_stride = frame_stride
        self.embed_dim = int(self.model.config.hidden_size)
        self.temporal_embed = temporal_embed
        self.temporal_scale = temporal_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W) -> (B, T/stride * P, 1024)."""
        b, c, t, h, w = x.shape
        x = x[:, :, ::self.frame_stride]                       # (B,C,T',H,W)
        tt = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * tt, c, h, w)  # frames as a batch
        tok = self.model(pixel_values=x).last_hidden_state     # (B*T', 1+R+P, D)
        tok = tok[:, self.n_prefix:]                           # drop CLS + registers
        p, d = tok.shape[1], tok.shape[2]
        tok = tok.reshape(b, tt, p, d)
        if self.temporal_embed:
            # scale to the batch's own token RMS so the stamp is readable but does not
            # swamp the features (DINOv3 tokens are small-magnitude, ~0.14 RMS pooled)
            rms = tok.pow(2).mean().sqrt().detach()
            e = sincos_temporal(tt, d, tok.device, tok.dtype)  # (T',D)
            tok = tok + self.temporal_scale * rms * e[None, :, None, :]
        return tok.reshape(b, tt * p, d)


def load_encoder(img_size: int = 256, num_frames: int = 16, device: str = "cuda",
                 model_id: str = MODEL_ID, temporal_embed: bool = True):
    """Frozen DINOv3 ViT-L/16. Signature mirrors ``_encoder.load_encoder``."""
    enc = DinoV3VideoEncoder(model_id, temporal_embed=temporal_embed).eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc

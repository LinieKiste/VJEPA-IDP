"""VideoQAModel: V-JEPA 2 encoder + MLP projector + QLoRA LLM.

Replicates the LLaVA-style alignment described in V-JEPA 2 (arXiv 2506.09985,
Sec. 14 / Appendix E): a frozen video encoder produces patch tokens, an MLP
projector maps them into the LLM embedding space, and an autoregressive LLM is
adapted with LoRA. Visual tokens are spliced into the text embedding sequence
in place of a single placeholder token (LLaVA convention).
"""

import os
import sys

import torch
import torch.nn as nn

# Make the sibling vjepa2/ package importable.
_VJEPA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vjepa2")
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from src.models import vision_transformer as vit  # noqa: E402
from src.utils.checkpoint_loader import robust_checkpoint_loader  # noqa: E402

# Sentinel token id placed in input_ids to mark where visual tokens go.
IMAGE_TOKEN_INDEX = -200

# embed_dim of each V-JEPA 2 backbone (see src/models/vision_transformer.py).
ENCODER_EMBED_DIM = {
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant_xformers": 1408,
}


def _clean_backbone_key(state_dict):
    """Strip the 'module.' / 'backbone.' prefixes from checkpoint keys."""
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def build_encoder(arch, checkpoint_path, img_size, num_frames, checkpoint_key="target_encoder"):
    """Construct a V-JEPA 2 ViT encoder and load pretrained weights from disk.

    Mirrors the kwargs used in vjepa2/src/hub/backbones.py::_make_vjepa2_model
    but loads from a local checkpoint instead of a URL. RoPE is enabled, so the
    (unused) absolute pos_embed mismatch is expected and loaded non-strict.
    """
    encoder = vit.__dict__[arch](
        patch_size=16,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    sd = robust_checkpoint_loader(checkpoint_path)
    encoder_sd = _clean_backbone_key(sd[checkpoint_key])
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    # pos_embed is the only expected miss (RoPE replaces it).
    unexpected = [k for k in unexpected if "pos_embed" not in k]
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading encoder: {unexpected}")
    return encoder


class Projector(nn.Module):
    """2-layer MLP mapping encoder tokens -> LLM embedding space (LLaVA-1.5)."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VideoQAModel(nn.Module):
    def __init__(
        self,
        encoder,
        projector,
        llm,
        tubelet_size=2,
        patch_size=16,
        spatial_pool_stride=2,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.spatial_pool_stride = spatial_pool_stride

        # Encoder is always frozen (Appendix E: feature-quality probe setup).
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    # --- visual feature extraction -------------------------------------- #
    @torch.no_grad()
    def encode_visual(self, pixel_values):
        """pixel_values: (B, C, T, H, W) -> (B, N_pooled, encoder_dim).

        Encoder returns flattened (T', H', W') tokens. We reshape and average
        pool spatially by `spatial_pool_stride` to cut visual token count
        (e.g. 1024 -> 256 for 8 frames at 256px).
        """
        B, C, T, H, W = pixel_values.shape
        feats = self.encoder(pixel_values)  # (B, N, D)
        Tp = T // self.tubelet_size
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        D = feats.shape[-1]
        feats = feats.view(B, Tp, Hp, Wp, D)

        s = self.spatial_pool_stride
        if s > 1:
            # (B, Tp, Hp, Wp, D) -> avg pool over Hp, Wp
            feats = feats.permute(0, 1, 4, 2, 3).reshape(B * Tp, D, Hp, Wp)
            feats = nn.functional.avg_pool2d(feats, kernel_size=s, stride=s)
            Hp, Wp = Hp // s, Wp // s
            feats = feats.reshape(B, Tp, D, Hp, Wp).permute(0, 1, 3, 4, 2)

        return feats.reshape(B, Tp * Hp * Wp, D)

    # --- splice visual tokens into text embeddings ---------------------- #
    def _merge(self, input_ids, labels, visual_embeds, pad_mask=None):
        """Build inputs_embeds/labels by replacing each IMAGE_TOKEN_INDEX with
        that sample's visual token sequence. Returns right-padded tensors.

        `pad_mask` (1 = real token, 0 = collator padding) trims each sample to
        its true length before splicing, so collator padding never enters the
        LLM. Re-padding to the merged max length is done here.
        """
        embed_tokens = self.llm.get_input_embeddings()
        device = visual_embeds.device

        merged_embeds, merged_labels = [], []
        for b in range(input_ids.shape[0]):
            if pad_mask is not None:
                n_real = int(pad_mask[b].sum().item())
                ids = input_ids[b, :n_real]
                lbl = labels[b, :n_real]
            else:
                ids = input_ids[b]
                lbl = labels[b]
            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged_embeds.append(embed_tokens(ids.clamp(min=0)))
                merged_labels.append(lbl)
                continue
            p = img_pos[0].item()
            pre_ids, post_ids = ids[:p], ids[p + 1 :]
            pre_emb = embed_tokens(pre_ids.clamp(min=0))
            post_emb = embed_tokens(post_ids.clamp(min=0))
            vis = visual_embeds[b].to(pre_emb.dtype)
            emb = torch.cat([pre_emb, vis, post_emb], dim=0)
            n_vis = vis.shape[0]
            vis_lbl = torch.full((n_vis,), -100, dtype=lbl.dtype, device=device)
            new_lbl = torch.cat([lbl[:p], vis_lbl, lbl[p + 1 :]], dim=0)
            merged_embeds.append(emb)
            merged_labels.append(new_lbl)

        # Right-pad to the longest merged sequence.
        max_len = max(e.shape[0] for e in merged_embeds)
        pad_id = self.llm.config.pad_token_id or 0
        pad_emb = embed_tokens(torch.tensor([pad_id], device=device)).squeeze(0)

        inputs_embeds, out_labels, attn = [], [], []
        for emb, lbl in zip(merged_embeds, merged_labels):
            n = emb.shape[0]
            pad = max_len - n
            if pad > 0:
                emb = torch.cat([emb, pad_emb.unsqueeze(0).expand(pad, -1)], dim=0)
                lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=lbl.dtype, device=device)])
            mask = torch.cat([torch.ones(n, device=device), torch.zeros(max_len - n, device=device)])
            inputs_embeds.append(emb)
            out_labels.append(lbl)
            attn.append(mask)

        return (
            torch.stack(inputs_embeds),
            torch.stack(out_labels),
            torch.stack(attn).long(),
        )

    def forward(self, input_ids, labels, pixel_values, pad_mask=None):
        visual = self.encode_visual(pixel_values)
        visual = self.projector(visual)
        inputs_embeds, labels, attention_mask = self._merge(input_ids, labels, visual, pad_mask)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, pad_mask=None, **gen_kwargs):
        visual = self.projector(self.encode_visual(pixel_values))
        dummy_labels = torch.full_like(input_ids, -100)
        inputs_embeds, _, attention_mask = self._merge(input_ids, dummy_labels, visual, pad_mask)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

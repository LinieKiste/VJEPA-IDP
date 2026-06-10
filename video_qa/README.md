# Video QA training — V-JEPA 2 Appendix E replication

A self-contained LLaVA-style training loop that aligns a frozen **V-JEPA 2**
video encoder with an LLM for video question answering, following the recipe in
V-JEPA 2 (arXiv:2506.09985, Sec. 14 / Appendix E).

## Architecture

```
V-JEPA 2 ViT-L (frozen)  ──►  MLP projector  ──►  Qwen2.5-7B-Instruct (4-bit + LoRA)
   patch tokens (d=1024)        d=1024→2048→3584      autoregressive cross-entropy
```

Visual tokens are spliced into the text embedding sequence in place of an
`<image>` / `<video>` placeholder (LLaVA `IMAGE_TOKEN_INDEX = -200` convention).
The loss is cross-entropy on the assistant response tokens only.

For 8 frames at 256px (patch 16, tubelet 2) the encoder emits
`4 × 16 × 16 = 1024` tokens; `spatial_pool_stride: 2` average-pools them to 256.

## Three stages (paper Appendix E)

| Stage | Trainable | Data | Config |
|-------|-----------|------|--------|
| 1 | projector only | image captions (e.g. LLaVA-CC3M-595K) | `configs/stage1.yaml` |
| 2 | projector + LoRA | image QA (e.g. LLaVA-Instruct-150K) | `configs/stage2.yaml` |
| 3 | projector + LoRA | video QA (e.g. ActivityNet-QA, NExT-QA) | `configs/stage3.yaml` |

Each stage loads the previous stage's `projector.pt` (and `lora_adapters.pt`)
through the `projector.init_from` / `llm.lora_init_from` config keys.

## Setup

Dependencies are in the project `pyproject.toml` (`uv sync`): `bitsandbytes`,
`python-box`, `decord`, `timm`, etc.

The ViT-L checkpoint is expected at `checkpoints/vitl.pt` (already present).
Qwen2.5-7B-Instruct downloads from the HF Hub on first run.

**CUDA note:** torch here is built for CUDA 13.2 (`cu132`), but bitsandbytes only
ships binaries up to `cuda130`. CUDA 13.x minor versions are compatible, so
`train.py` sets `BNB_CUDA_VERSION=130` automatically (overridable). Remove that
line once a cu132 bitsandbytes wheel exists.

## Data format

A JSON list of LLaVA-style conversations:

```json
[
  {"id": "0", "image": "cat.jpg",
   "conversations": [
     {"from": "human", "value": "<image>\nDescribe this image."},
     {"from": "gpt",   "value": "A cat on a mat."}]},
  {"id": "1", "video": "cook.mp4",
   "conversations": [
     {"from": "human", "value": "<video>\nWhat is the person doing?"},
     {"from": "gpt",   "value": "Pouring water into a glass."}]}
]
```

Paths are resolved relative to `data.media_root`.

## Run

```bash
# Sanity check: one forward pass on a synthetic batch (no dataset needed).
# Exercises encoder -> projector -> LLM and prints shapes + loss.
.venv/bin/python video_qa/train.py --config video_qa/configs/stage1.yaml --dry_run

# Full stages (run in order).
.venv/bin/python video_qa/train.py --config video_qa/configs/stage1.yaml
.venv/bin/python video_qa/train.py --config video_qa/configs/stage2.yaml
.venv/bin/python video_qa/train.py --config video_qa/configs/stage3.yaml
```

Always launch from the project root so `checkpoints/` and `data/` paths resolve.

## Files

- `model.py`  — encoder loader, projector, `VideoQAModel` (splice + forward + generate)
- `dataset.py` — frame sampling (decord), Qwen2.5 chat tokenization, label masking
- `collate.py` — padding collator
- `train.py`  — config-driven 3-stage training loop (AdamW, warmup+cosine, bf16)

## Differences from the paper

| Paper | Here | Why |
|-------|------|-----|
| Qwen2-7B / LLaMA 3.1 8B, full FT | Qwen2.5-7B, 4-bit + LoRA | fits 16 GB VRAM |
| 18M–88M pairs | small public sets | single-GPU iteration |
| encoder frozen *or* end-to-end | encoder always frozen | feature-quality probe |
| up to 384px, 16 frames | 256px, 8 frames (configurable) | memory |

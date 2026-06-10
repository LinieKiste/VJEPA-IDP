# egoper_vqa — video-QA inference baseline on EgoPER

Sanity-check a SOTA-class video LLM on EgoPER's ~10-12 min egocentric procedural
videos with **user-defined questions**, and compare answers against EgoPER's
ground-truth error labels. This is the *baseline / workflow validation* step; the
project's actual contribution (SlowFast-style pooling over V-JEPA 2 features) is
separate — see the design notes in the repo `CLAUDE.md`.

## What's here
- `egoper.py` — load `datasets/egoper/annotation.json`, list error/normal videos per
  task, get structured ground truth (`has_error`, error type/description/timestamp,
  full action timeline), and uniformly sample frames from a clip.
- `vqa.py` — load Qwen2.5-VL 4-bit and `ask()` a free-text question about a video given
  sampled frames; `ask_slowfast()` runs the training-free SF-LLaVA v1 two-stream scheme.
  No `qwen_vl_utils` dependency. **Defaults to the 3B** — the 7B OOMs on 16 GB shared with
  a display (4-bit LLM ~4.5 GB + bf16 vision tower ~1.3 GB + video forward + display).
- `inference.ipynb` — end-to-end: pick a clip → show GT → sample frames → load model →
  ask questions → control run on a normal clip.

## Run
Uses the project `.venv` (transformers 5.10.2, torch 2.12/cu132, bitsandbytes — all
present). No new installs needed. First model load downloads ~16 GB; resident ~5-6 GB
in 4-bit.

```bash
.venv/bin/jupyter lab egoper_vqa/inference.ipynb   # or open in VS Code
```

## Why Qwen2.5-VL and not SF-LLaVA-1.5 / LLaVA-Video-7B-Qwen2
- **SF-LLaVA-1.5** (the model we actually want to emulate, arXiv 2503.18943) has **no
  publicly discoverable trained weights** — Apple's `ml-slowfast-llava` GitHub is the
  older *training-free* v1.
- **LLaVA-Video-7B-Qwen2** (LongVideoBench leaderboard pick) loads only via the
  **LLaVA-NeXT** package, which pins `torch==2.1.2` + a frozen old transformers commit.
  That (a) can't share this project's torch 2.12 / transformers 5.x env, and (b) has no
  Blackwell (sm_120) kernels, so it **can't use the RTX 5060 Ti** at all.
- **Qwen2.5-VL-7B** is native in transformers 5.10.2, SOTA-class on video, and runs
  4-bit on 16 GB today. Best obtainable baseline for this card.

### LLaVA-Video off-ramp (only if you need the exact leaderboard number)
Use an **isolated** environment so it can't break `vjepa2/` / `video_qa/`:
```bash
python -m venv .venv-llavavideo && . .venv-llavavideo/bin/activate
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# then OVERRIDE the torch pin with a Blackwell-capable build, or it won't run on this GPU:
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Expect friction (the frozen transformers commit may fight the newer torch).

## Benchmarks (7B, for reference)
| Model | LongVideoBench | MLVU |
|---|---|---|
| SF-LLaVA-1.5-7B (unavailable) | 62.5 | 71.5 |
| LLaVA-Video-7B-Qwen2 | 58.2 | 70.8 |

## Known limits
- Uniform sampling of ~32 frames over ~12 min ≈ 1 frame / 23 s → brief/subtle errors
  can be missed. Raise `NUM_FRAMES` or sample around a window. This temporal-coverage
  gap is exactly what the SlowFast-over-V-JEPA direction is meant to address.

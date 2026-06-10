"""Thin wrapper around Qwen2.5-VL-7B-Instruct for video QA on a 16 GB card.

Loaded 4-bit (nf4) via bitsandbytes so the 7B fits alongside activations on the
RTX 5060 Ti. Frames are passed straight to the processor as a list of PIL images,
so this has no dependency on ``qwen_vl_utils``.

Why Qwen2.5-VL and not LLaVA-Video-7B-Qwen2 (the leaderboard pick): LLaVA-NeXT
pins torch==2.1.2 + a frozen old transformers commit, which (a) can't share this
project's torch 2.12/cu132 + transformers 5.x env and (b) has no Blackwell (sm_120)
kernels, so it can't even use this GPU. Qwen2.5-VL is native in transformers 5.10.2
and is SOTA-class on video. See README for the LLaVA-Video isolated-env off-ramp.
"""
from __future__ import annotations

# 3B is the safe default on a 16 GB card shared with a display; the 7B (4-bit LLM
# ~4.5 GB + bf16 vision tower ~1.3 GB + video forward) is tight and can OOM.
# DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, load_in_4bit: bool = True):
    """Returns (model, processor). First call downloads ~16 GB of weights."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        print(f"loaded {model_id} | GPU used {used:.1f} / {total / 1024**3:.1f} GiB "
              f"| {free / 1024**3:.1f} GiB free for the forward pass")
    return model, processor


def ask(
    model,
    processor,
    frames,
    question: str,
    max_new_tokens: int = 256,
    max_pixels: int = 256 * 450,
    context: str | None = None,
) -> str:
    """Ask one free-text question about a video given as sampled frames.

    ``frames`` is a uint8 array (T, H, W, 3) or a list of PIL images (e.g. from
    ``egoper.sample_frames``). ``context`` (e.g. ``egoper.procedure_text(task)``) is
    prepended to ground the model in the intended procedure. Returns the answer.

    ``max_pixels`` caps the per-frame resolution the processor resizes to, which
    bounds tokens/frame (≈ ``max_pixels / 784 / 2`` after spatial+temporal merge).
    This is the manual SlowFast lever AND the memory lever: on 16 GB, keep
    ``frames × max_pixels`` roughly constant (~7.4 M) or the vision encoder OOMs.
    E.g. 32@230400, 64@115200, 128@57600 all fit the same envelope.
    """
    import torch
    from PIL import Image

    if not isinstance(frames, list):
        frames = [Image.fromarray(f) for f in frames]

    if context:
        question = f"{context.strip()}\n\n{question}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], videos=[frames], max_pixels=max_pixels, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _resize_to_budget(frames, max_pixels: int):
    """Return frames as a list of PIL images, each downscaled so H*W <= max_pixels
    (aspect ratio preserved). Frames already under budget are left as-is."""
    from PIL import Image

    out = []
    for f in frames:
        img = f if isinstance(f, Image.Image) else Image.fromarray(f)
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((max(28, round(w * scale)), max(28, round(h * scale))))
        out.append(img)
    return out


# preamble so the model knows the two streams are the SAME video at two granularities
_SLOWFAST_PREFIX = (
    "You are shown the same video twice: first a few high-detail frames (for fine "
    "spatial detail), then many lower-detail frames (for full temporal coverage). "
    "Use both to answer.\n\n"
)


def ask_slowfast(
    model,
    processor,
    slow_frames,
    fast_frames,
    question: str,
    max_new_tokens: int = 256,
    slow_max_pixels: int = 360 * 640,
    fast_max_pixels: int = 200 * 200,
    context: str | None = None,
) -> str:
    """Training-free SlowFast (SF-LLaVA v1 scheme) on Qwen2.5-VL.

    ``slow_frames`` = few frames kept at high res (spatial detail); ``fast_frames`` =
    many frames at low res (temporal coverage). Both are passed as separate video
    streams and concatenated in the prompt. This is the genuine SF-LLaVA v1 idea on a
    Blackwell-compatible backbone — no LLaVA-NeXT, no training.

    Memory (16 GB): keep ``len(slow)*slow_max_pixels + len(fast)*fast_max_pixels``
    under ~7.4 M total pixels (the envelope where 32 uniform frames fit). Defaults
    (16 slow @ 230400 + 96 fast @ 40000 ≈ 7.5 M) sit right at that line.
    """
    import torch

    slow = _resize_to_budget(slow_frames, slow_max_pixels)
    fast = _resize_to_budget(fast_frames, fast_max_pixels)

    user_text = _SLOWFAST_PREFIX + (f"{context.strip()}\n\n" if context else "") + question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},  # slow stream
                {"type": "video"},  # fast stream
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # high max_pixels so the processor doesn't re-shrink our pre-sized frames
    inputs = processor(
        text=[text], videos=[slow, fast], max_pixels=slow_max_pixels, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

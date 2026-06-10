"""LLaVA-style dataset for image / video question answering.

Reads a JSON list of conversations (LLaVA format), loads the associated image
or video, samples frames, and tokenizes the dialogue into input_ids/labels with
the visual placeholder replaced by IMAGE_TOKEN_INDEX. Labels are masked (-100)
on everything except assistant ("gpt") response tokens.
"""

import json
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset

from model import IMAGE_TOKEN_INDEX

# ImageNet mean/std (V-JEPA 2 default normalization).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

MEDIA_PLACEHOLDERS = ("<image>", "<video>")


def _resize_center_crop(img, size):
    """Resize shorter side to `size` then center-crop to size x size."""
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _to_tensor(frames, size):
    """List of PIL frames -> normalized (C, T, H, W) float tensor."""
    arr = np.stack([np.asarray(_resize_center_crop(f, size)) for f in frames])  # (T, H, W, C)
    t = torch.from_numpy(arr).float().permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
    return (t - MEAN) / STD


def load_video(path, num_frames, frame_step, size):
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    span = num_frames * frame_step
    start = max(0, (total - span) // 2)
    idx = [min(total - 1, start + i * frame_step) for i in range(num_frames)]
    frames = [Image.fromarray(vr[i].asnumpy()) for i in idx]
    return _to_tensor(frames, size)


def load_image(path, num_frames, size):
    """Load a still image and repeat it to num_frames (so the 3D encoder runs)."""
    img = Image.open(path).convert("RGB")
    return _to_tensor([img] * num_frames, size)


class VideoQADataset(Dataset):
    def __init__(self, json_path, media_root, tokenizer, num_frames, frame_step, img_size,
                 system_prompt="You are a helpful assistant."):
        with open(json_path) as f:
            self.data = json.load(f)
        self.media_root = media_root
        self.tok = tokenizer
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.img_size = img_size
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def _encode_text(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def _tokenize(self, conversations):
        """Build input_ids + labels with Qwen2.5 chat formatting.

        Human turns and all role headers are masked; only assistant content and
        its <|im_end|> are supervised. The media placeholder becomes a single
        IMAGE_TOKEN_INDEX entry that the model expands into visual tokens.
        """
        ids, labels = [], []

        def add(token_ids, supervise):
            ids.extend(token_ids)
            labels.extend(token_ids if supervise else [-100] * len(token_ids))

        # System turn.
        add(self._encode_text(f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"), False)

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = turn["value"]
            add(self._encode_text(f"<|im_start|>{role}\n"), False)

            if role == "user":
                # Split out the media placeholder -> IMAGE_TOKEN_INDEX.
                for ph in MEDIA_PLACEHOLDERS:
                    content = content.replace(ph, "<image>")
                segments = content.split("<image>")
                for i, seg in enumerate(segments):
                    if seg:
                        add(self._encode_text(seg), False)
                    if i < len(segments) - 1:
                        ids.append(IMAGE_TOKEN_INDEX)
                        labels.append(-100)
                add(self._encode_text("<|im_end|>\n"), False)
            else:
                add(self._encode_text(f"{content}<|im_end|>\n"), True)

        return torch.tensor(ids), torch.tensor(labels)

    def __getitem__(self, i):
        sample = self.data[i]
        if "video" in sample:
            pixel_values = load_video(
                os.path.join(self.media_root, sample["video"]),
                self.num_frames, self.frame_step, self.img_size,
            )
        else:
            pixel_values = load_image(
                os.path.join(self.media_root, sample["image"]),
                self.num_frames, self.img_size,
            )
        input_ids, labels = self._tokenize(sample["conversations"])
        return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values}

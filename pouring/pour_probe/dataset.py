"""UW Liquid Pouring Dataset (UWLPD) — sequence index, metadata, and frame/label access.

The download at ``datasets/UWLPD/`` is the UWLPD **Real Robot Dataset**, ``large_bowls``
subset: 5 zips = source->target combos (``bowl<-{bottle,cup,mug}``, ``fruitBowl<-{bottle,cup}``),
36 conditions each = **180 sequences**, kept zipped (~61 GB). Per sequence
(``scene_left_<bowl>_<cup>_<fill>_<profile>_<motion>_large_bowls/render_v3/``):
  - ``data<NNNN>.jpg``          RGB camera frame (640x480). ``data*.png`` is a byte-identical copy.
  - ``ground_truth<NNNN>.png``  binary liquid mask (RGBA, 0/255) — the per-pixel liquid label.
  - ``sim_args.txt``            a python-dict literal with {fill, cup, bowl, pouringProfile, motion}.

There is **no mL ground truth here** (that lives in the separate *Simulated* dataset's
``bowl_volume.csv``). So this module derives a per-frame **proxy** target from the liquid mask:
the liquid-pixel *area*. ``dataset.py`` is written so the real mL trace drops in later as an
alternate target keyed by frame index.

Frames are read straight from the zips (random-access, no 61 GB extraction).

Usage:
    .venv/bin/python pour_probe/dataset.py            # build + verify + persist mapping.json
"""
from __future__ import annotations

import ast
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # repo root (pouring/pour_probe/dataset.py)
DATA_DIR = ROOT / "datasets" / "UWLPD"
MAPPING_JSON = Path(__file__).parent / "mapping.json"

_SEQ_RE = re.compile(r"^(scene_left_[^/]+)/render_v3/$")


def _parse_sim_args(text: str) -> dict:
    """``sim_args.txt`` is a python-dict literal; pull the pouring condition out of its 'args' list."""
    d = ast.literal_eval(text)
    args = d.get("args", [])
    out = {}
    for i in range(0, len(args) - 1, 2):
        if isinstance(args[i], str) and args[i].startswith("--"):
            out[args[i][2:]] = args[i + 1]
    return out  # keys: motion, cup, dataset, pouringProfile, bowl, arm, fill


def _seq_dirs(zf: zipfile.ZipFile) -> list[str]:
    """Sorted list of ``scene_left_...`` sequence names inside a zip (one per render_v3 dir)."""
    seqs = []
    for n in zf.namelist():
        m = _SEQ_RE.match(n)
        if m:
            seqs.append(m.group(1))
    return sorted(seqs)


def build_mapping() -> list[dict]:
    """One record per sequence across all 5 zips: metadata + frame count + zip location."""
    records = []
    for zpath in sorted(DATA_DIR.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for seq in _seq_dirs(zf):
                base = f"{seq}/render_v3"
                sim = _parse_sim_args(zf.read(f"{base}/sim_args.txt").decode())
                frames = sorted(
                    n for n in zf.namelist()
                    if n.startswith(f"{base}/data") and n.endswith(".jpg")
                )
                records.append({
                    "video_id": seq,
                    "zip": zpath.name,
                    "bowl": sim.get("bowl", ""),
                    "cup": sim.get("cup", ""),
                    "combo": f"{sim.get('bowl','')}_{sim.get('cup','')}",
                    "fill": sim.get("fill", ""),
                    "profile": sim.get("pouringProfile", ""),
                    "motion": sim.get("motion", ""),
                    "n_frames": len(frames),
                })
    return records


def persist_mapping(records: list[dict]) -> None:
    MAPPING_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2))


@lru_cache(maxsize=1)
def list_videos() -> list[dict]:
    """Mapping records, loaded from mapping.json (build it first via ``python dataset.py``)."""
    if not MAPPING_JSON.exists():
        raise FileNotFoundError(f"{MAPPING_JSON} missing — run `python pour_probe/dataset.py` first")
    return json.loads(MAPPING_JSON.read_text())


@lru_cache(maxsize=1)
def _zip_of() -> dict[str, str]:
    return {r["video_id"]: r["zip"] for r in list_videos()}


class SequenceReader:
    """Open a sequence's zip once; read RGB frames (resized/cropped) and liquid-mask areas.

    ``frames`` is the sorted list of frame indices present. RGB comes from ``data<i>.jpg``;
    the proxy target from ``ground_truth<i>.png`` (binary liquid mask -> pixel count).
    """

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.zf = zipfile.ZipFile(DATA_DIR / _zip_of()[video_id])
        self.base = f"{video_id}/render_v3"
        self.frames = sorted(
            int(re.search(r"data(\d+)\.jpg$", n).group(1))
            for n in self.zf.namelist()
            if n.startswith(f"{self.base}/data") and n.endswith(".jpg")
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zf.close()

    def read_rgb(self, idx: list[int], size: int) -> np.ndarray:
        """Read frames at ``idx`` (positions into ``self.frames``), resize shorter side->size,
        center-crop -> size^2. Returns (len(idx), size, size, 3) uint8."""
        out = np.empty((len(idx), size, size, 3), dtype=np.uint8)
        for k, fi in enumerate(idx):
            name = f"{self.base}/data{self.frames[fi]:04d}.jpg"
            im = Image.open(self.zf.open(name)).convert("RGB")
            w, h = im.size
            if w <= h:
                nw, nh = size, round(h * size / w)
            else:
                nh, nw = size, round(w * size / h)
            im = im.resize((nw, nh), Image.BILINEAR)
            left, top = (nw - size) // 2, (nh - size) // 2
            out[k] = np.asarray(im)[top:top + size, left:left + size, :]
        return out

    def mask_area(self, idx: list[int]) -> np.ndarray:
        """Liquid-pixel count of ``ground_truth<i>.png`` at positions ``idx``. (len(idx),) float32.

        Liquid is drawn blue ``(0,0,255,255)`` on a transparent-black ``(0,0,0,0)`` background, so
        count pixels whose max RGB channel is high (``convert('L')`` would collapse blue to ~29)."""
        out = np.empty(len(idx), dtype=np.float32)
        for k, fi in enumerate(idx):
            name = f"{self.base}/ground_truth{self.frames[fi]:04d}.png"
            gt = np.asarray(Image.open(self.zf.open(name)).convert("RGBA"))
            out[k] = float((gt[..., :3].max(axis=-1) > 127).sum())
        return out


def _verify(records: list[dict]) -> None:
    from collections import Counter
    assert len(records) == 180, f"expected 180 sequences, got {len(records)}"
    combos = Counter(r["combo"] for r in records)
    assert all(v == 36 for v in combos.values()), f"expected 36/combo: {dict(combos)}"
    fills = Counter(r["fill"] for r in records)
    profiles = Counter(r["profile"] for r in records)
    motions = Counter(r["motion"] for r in records)
    print(f"OK: {len(records)} sequences, {len(combos)} combos x 36.")
    print(f"  combos:   {dict(combos)}")
    print(f"  fill:     {dict(fills)}")
    print(f"  profile:  {dict(profiles)}")
    print(f"  motion:   {dict(motions)}")
    print(f"  n_frames: min {min(r['n_frames'] for r in records)} "
          f"max {max(r['n_frames'] for r in records)} "
          f"mean {np.mean([r['n_frames'] for r in records]):.0f}")


def main():
    records = build_mapping()
    _verify(records)
    persist_mapping(records)
    print(f"wrote {MAPPING_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

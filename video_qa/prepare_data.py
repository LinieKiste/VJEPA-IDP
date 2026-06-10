"""Download & convert public VQA datasets into the LLaVA-style JSON the loader
expects (see dataset.py). Paths match the defaults in configs/stage{1,2,3}.yaml.

    python video_qa/prepare_data.py stage1 --subset 2000
    python video_qa/prepare_data.py stage2
    python video_qa/prepare_data.py stage3

The LLaVA annotation files are already in our schema (an item has "image" or
"video" + a "conversations" list with from/value, the human turn containing
"<image>"/"<video>"), so "conversion" is mostly downloading, optionally
subsetting, and pointing media_root at the right folder.

Run from the project root so data/ lands next to checkpoints/ and the configs
resolve. Large media (COCO, video sets) is fetched per stage; --subset keeps
disk use small while validating the pipeline.
"""

import argparse
import json
import os
import zipfile

from huggingface_hub import hf_hub_download

DATA_DIR = "data"


def _write_json(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f)
    print(f"Wrote {len(items)} items -> {path}")


# --------------------------------------------------------------------------- #
# Stage 1: LLaVA-CC3M-Pretrain-595K (image captions, projector warm-up).
# Fully automated: annotations + images both come from the HF dataset repo.
# --------------------------------------------------------------------------- #
def stage1(subset):
    repo = "liuhaotian/LLaVA-CC3M-Pretrain-595K"
    print(f"Downloading {repo} (chat.json + images.zip)...")
    chat = hf_hub_download(repo, "chat.json", repo_type="dataset")
    img_zip = hf_hub_download(repo, "images.zip", repo_type="dataset")

    with open(chat) as f:
        items = json.load(f)
    if subset:
        items = items[:subset]
    needed = {it["image"] for it in items}

    img_root = os.path.join(DATA_DIR, "cc3m")
    os.makedirs(img_root, exist_ok=True)
    print(f"Extracting {len(needed)} images to {img_root}...")
    with zipfile.ZipFile(img_zip) as z:
        members = [m for m in z.namelist() if os.path.basename(m) in needed or m in needed]
        for m in members:
            z.extract(m, img_root)
    # Flatten if the zip nested everything under a top folder.
    _flatten_to_root(img_root, needed)

    _write_json(items, os.path.join(DATA_DIR, "llava_cc3m_595k.json"))
    print("Stage 1 ready. Run: train.py --config video_qa/configs/stage1.yaml")


def _flatten_to_root(root, needed):
    """Ensure each needed image sits directly at root/<basename>."""
    for dirpath, _, files in os.walk(root):
        if dirpath == root:
            continue
        for fn in files:
            if fn in needed:
                src = os.path.join(dirpath, fn)
                dst = os.path.join(root, fn)
                if not os.path.exists(dst):
                    os.replace(src, dst)


# --------------------------------------------------------------------------- #
# Stage 2: LLaVA-Instruct-150K (image QA). Annotations auto; images = COCO.
# --------------------------------------------------------------------------- #
def stage2(subset):
    repo = "liuhaotian/LLaVA-Instruct-150K"
    print(f"Downloading {repo} (llava_instruct_150k.json)...")
    ann = hf_hub_download(repo, "llava_instruct_150k.json", repo_type="dataset")
    with open(ann) as f:
        items = json.load(f)
    if subset:
        items = items[:subset]
    # LLaVA-Instruct image paths are bare COCO ids like "000000123456.jpg".
    _write_json(items, os.path.join(DATA_DIR, "llava_instruct_150k.json"))
    print(
        "Stage 2 annotations ready. Images are COCO train2017 -> place them at\n"
        f"  {os.path.join(DATA_DIR, 'coco')}/<image_id>.jpg\n"
        "Download: http://images.cocodataset.org/zips/train2017.zip (~18 GB),\n"
        "or fetch only the subset's ids. Then run stage2.yaml."
    )


# --------------------------------------------------------------------------- #
# Stage 3: video QA. NExT-QA is the most manageable public option.
# --------------------------------------------------------------------------- #
def stage3(subset):
    print(
        "Stage 3 (video QA) needs a video QA set. Recommended: NExT-QA or\n"
        "ActivityNet-QA. Produce data/activitynet_qa.json (or nextqa.json) where\n"
        "each item is:\n"
        '  {"video": "<file>.mp4", "conversations": [\n'
        '     {"from": "human", "value": "<video>\\n<question>"},\n'
        '     {"from": "gpt",   "value": "<answer>"}]}\n'
        "and set data.media_root in stage3.yaml to the video folder.\n\n"
        "For the IDP project, the natural choice is to build this from EgoPER\n"
        "videos + error annotations (anomaly QA) — ask Claude to wire that up."
    )


STAGES = {"stage1": stage1, "stage2": stage2, "stage3": stage3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N items (keeps disk/time small).")
    args = parser.parse_args()
    STAGES[args.stage](args.subset)

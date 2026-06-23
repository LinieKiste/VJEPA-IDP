"""Qualitative probe: off-the-shelf V-JEPA 2 + EK100 anticipation head on EgoPER tea videos.

Runs the *faithful* EK100 action-anticipation pipeline (frozen V-JEPA 2 ViT-L encoder +
predictor `concat_ar` wrapper + the trained EK100 AttentiveClassifier) on EgoPER tea clips and
prints its top verb / noun / action predictions next to the EgoPER ground-truth action at that
moment. Purely diagnostic: are EK100's Epic-Kitchens predictions sensible for tea-making?

The pipeline anticipates the action ~`anticipation_time` s in the future (EK100's task), matching
configs/inference/vitl/ek100.yaml (32 frames @ 8 fps, 256 px, anticipate 1.0 s, predictor depth 12).
Class indices are decoded by rebuilding EK100's verb/noun/action dicts from EPIC_100_train.csv
exactly as the dataloader does (`enumerate(set(...))`).

Usage:
    .venv/bin/python egoper_vqa/ek100_tea.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vjepa2"))   # `import src...`, `import evals...`
sys.path.insert(0, str(ROOT / "egoper_vqa"))
import egoper  # noqa: E402
from evals.action_anticipation_frozen.models import AttentiveClassifier, init_module  # noqa: E402

META = ROOT / "egoper_vqa" / "epic_meta"
CLIPS = ROOT / "egoper_vqa" / "tea_clips"
CKPT = ROOT / "checkpoints" / "vitl.pt"
EK100_CKPT = ROOT / "checkpoints" / "ek100-vitl-256.pt"
FRAMES, FPS_SAMPLE, RES, ANTICIPATION_S = 32, 8, 256, 1.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

PRETRAIN_KWARGS = {
    "encoder": {"model_name": "vit_large", "checkpoint_key": "target_encoder",
                "tubelet_size": 2, "patch_size": 16, "uniform_power": True, "use_rope": True},
    "predictor": {"model_name": "vit_predictor", "checkpoint_key": "predictor", "num_frames": 64,
                  "depth": 12, "num_heads": 12, "predictor_embed_dim": 384, "num_mask_tokens": 10,
                  "uniform_power": True, "use_mask_tokens": True, "use_sdpa": True,
                  "use_silu": False, "wide_silu": False, "use_rope": True},
}
WRAPPER_KWARGS = {"no_predictor": False, "num_output_frames": 2, "num_steps": 1}


def build_decoders():
    """Rebuild EK100 verb/noun/action index->name maps from EPIC_100_train.csv (as the dataloader)."""
    tdf = pd.read_csv(META / "EPIC_100_train.csv")
    tactions = set((int(v), int(n)) for v, n in zip(tdf["verb_class"], tdf["noun_class"]))
    tverbs = set(v for v, _ in tactions)
    tnouns = set(n for _, n in tactions)
    inv_verb = {i: k for i, k in enumerate(tverbs)}          # model idx -> EPIC verb id
    inv_noun = {i: k for i, k in enumerate(tnouns)}
    inv_action = {i: k for i, k in enumerate(tactions)}      # model idx -> (verb id, noun id)
    vname = pd.read_csv(META / "EPIC_100_verb_classes.csv").set_index("id")["key"].to_dict()
    nname = pd.read_csv(META / "EPIC_100_noun_classes.csv").set_index("id")["key"].to_dict()
    print(f"vocab: {len(tverbs)} verbs, {len(tnouns)} nouns, {len(tactions)} actions")
    return {
        "verb": lambda i: vname.get(inv_verb[i], f"v{i}"),
        "noun": lambda i: nname.get(inv_noun[i], f"n{i}"),
        "action": lambda i: f"{vname.get(inv_action[i][0],'?')}:{nname.get(inv_action[i][1],'?')}",
        "dims": (len(tverbs), len(tnouns), len(tactions)),
    }


def load_classifier(dims, device):
    nv, nn_, na = dims
    clf = AttentiveClassifier(verb_classes={i: i for i in range(nv)}, noun_classes={i: i for i in range(nn_)},
                              action_classes={i: i for i in range(na)}, embed_dim=1024, num_heads=16,
                              depth=4, use_activation_checkpointing=False)
    sd = torch.load(EK100_CKPT, map_location="cpu", weights_only=False)["classifiers"][0]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = clf.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys: {unexp[:4]}"
    return clf.to(device).eval()


def load_clip(path, t_end, device):
    """32 frames @ 8 fps ending at t_end s, shorter-side->256, center-crop -> (1,3,32,256,256)."""
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    total = len(vr)
    times = t_end - (FRAMES - 1 - np.arange(FRAMES)) / FPS_SAMPLE   # t_end-4s .. t_end
    idx = np.clip((times * fps).round().astype(int), 0, total - 1)
    fr = vr.get_batch(idx.tolist()).asnumpy()                       # (32,H,W,3)
    h, w = fr.shape[1:3]
    nh, nw = (RES, round(w * RES / h)) if h <= w else (round(h * RES / w), RES)
    # resize via torch interpolate, then center crop
    x = torch.from_numpy(fr).permute(0, 3, 1, 2).float()            # (32,3,H,W)
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = (nh - RES) // 2, (nw - RES) // 2
    x = x[:, :, top:top + RES, left:left + RES] / 255.0
    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)              # (1,3,32,256,256)
    return (x - MEAN.to(device)) / STD.to(device)


def gt_at(timeline, t):
    for r in timeline:
        if r["start"] <= t <= r["end"]:
            return f"{r['type']}: {r['desc']}"
    return "(none)"


@torch.no_grad()
def main():
    device = "cuda"
    dec = build_decoders()
    model = init_module(module_name="evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar",
                        frames_per_clip=FRAMES, frames_per_second=FPS_SAMPLE, resolution=RES,
                        checkpoint=str(CKPT), model_kwargs=PRETRAIN_KWARGS,
                        wrapper_kwargs=WRAPPER_KWARGS, device=device)
    clf = load_classifier(dec["dims"], device)

    vids = ["tea_u2_av_normal_010", "tea_u2_aq_normal_009", "tea_u1_a2_error_004", "tea_u1_a5_error_010"]
    for vid in vids:
        path = CLIPS / f"{vid}.mp4"
        if not path.exists():
            continue
        tl = egoper.ground_truth("tea", vid)["timeline"]
        dur = tl[-1]["end"]
        print(f"\n{'='*100}\n{vid}   (duration {dur:.0f}s)\n{'='*100}")
        print(f"{'t(s)':>5} | {'GT now':<42} | top-3 verb / noun / action (anticipate +{ANTICIPATION_S:.0f}s)")
        for t in np.arange(8, dur - 1, max(6, (dur - 10) / 8)):
            x = load_clip(path, float(t), device)
            at = torch.tensor([ANTICIPATION_S], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model(x, at)
                out = clf(feats)
            tv = [dec["verb"](i) for i in out["verb"][0].float().topk(3).indices.tolist()]
            tn = [dec["noun"](i) for i in out["noun"][0].float().topk(3).indices.tolist()]
            ta = [dec["action"](i) for i in out["action"][0].float().topk(2).indices.tolist()]
            print(f"{t:5.0f} | {gt_at(tl, t)[:42]:<42} | {','.join(tv)}  |  {','.join(tn)}  |  {','.join(ta)}")


if __name__ == "__main__":
    main()

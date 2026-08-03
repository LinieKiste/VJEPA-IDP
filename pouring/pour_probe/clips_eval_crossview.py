"""Cross-view transfer check: does the attentive probe TRAINED ON CAM2 generalize
to CAM3's different camera angle, with zero adaptation?

Loads the CAM2-trained checkpoint (attn_<target>_CAM2_best.pt) and evaluates it
directly on CAM3 frames for the SAME held-out trials used at training time (so
this is a clean same-split comparison, not a new random split). Also reports the
CAM3-native ridge/temporal-profile baselines (from clips_train.py's mean-pool
cache) for context, and — if a CAM3-trained checkpoint exists — that number too.

Usage: .venv/bin/python pour_probe/clips_eval_crossview.py --target flow
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import clips_train_attn as ca
from _encoder import load_encoder
from head import build_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="flow")
    ap.add_argument("--train_cam", default="CAM2", help="camera the checkpoint was trained on")
    ap.add_argument("--eval_cam", default="CAM3", help="camera to test transfer on")
    ap.add_argument("--tag", default="", help="checkpoint tag suffix, e.g. '_roi' for the "
                    "detector-ROI crop run (attn_<target>_<cam><tag>_best.pt). Set "
                    "$POUR_FRAMES288_DIR to the matching frame cache.")
    ap.add_argument("--lag_s", type=float, default=0.0,
                    help="target lag the checkpoint was TRAINED with (e.g. 0.7). Must match, "
                         "or both the train-split normalization stats and the eval targets are "
                         "sampled at the wrong time and the transfer number is meaningless.")
    ap.add_argument("--val_trials", default=",".join(sorted(ca.VAL_TRIALS)),
                    help="held-out trials the checkpoint was TRAINED against. Must match the "
                         "training split, or train-set windows leak into the eval set and the "
                         "transfer number is optimistic. Defaults to clips_train_attn.VAL_TRIALS.")
    args = ap.parse_args()

    val_trials = set(args.val_trials.split(","))

    # targets are shared (same GT curve); only the video frames differ across cams.
    # train-split target stats come from the checkpoint's TRAINING camera.
    tr_cams = {args.train_cam: ca.load_clips(args.train_cam)}
    tr_only = [w for w in ca.build_windows(tr_cams, 1.0, 0.5, 16, args.lag_s)
               if w["trial"] not in val_trials]
    ytr = np.asarray([w[args.target] for w in tr_only], np.float32)
    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

    eval_cams = {args.eval_cam: ca.load_clips(args.eval_cam)}
    va_eval = [w for w in ca.build_windows(eval_cams, 1.0, 0.5, 16, args.lag_s)
               if w["trial"] in val_trials]
    print(f"eval on {args.eval_cam}: {len(va_eval)} val windows, "
          f"trials {sorted(val_trials)}")

    enc = load_encoder(img_size=256, num_frames=16, device="cuda")
    head = build_head(1).to("cuda")
    ckpt = ca.FRAMES_DIR.parent / f"attn_{args.target}_{args.train_cam}{args.tag}_best.pt"
    head.load_state_dict(torch.load(ckpt))
    mean, std = ca.MEAN.to("cuda"), ca.STD.to("cuda")

    r2, mae, preds, ys = ca.run_eval(enc, head, eval_cams, va_eval, args.target, 16,
                                     "cuda", mean, std, ymean, ystd)

    # native baselines on the eval camera, same held-out trials
    base = ca.baselines_on_split(args.eval_cam, args.target, val_trials, args.lag_s)

    unit = "g/s" if args.target == "flow" else "g"
    print(f"\n=== cross-view: probe trained on {args.train_cam}, evaluated on "
          f"{args.eval_cam} [{args.target}] ===")
    print(f"  {'method':<28} {'R2':>7} {'MAE':>9}")
    print(f"  {'attn (' + args.train_cam + '->' + args.eval_cam + ')':<28} {r2:>7.3f} {mae:>7.2f}{unit}")
    for name, (r2b, maeb) in base.items():
        print(f"  {name + ' (' + args.eval_cam + '-native)':<28} {r2b:>7.3f} {maeb:>7.2f}{unit}")

    # also print the original CAM2-on-CAM2 number for reference if same target
    print(f"\n  (for reference: attn trained+evaluated on {args.train_cam} itself: "
          f"see earlier run, val R2~0.899 MAE~7.0g/s for flow)")


if __name__ == "__main__":
    main()
